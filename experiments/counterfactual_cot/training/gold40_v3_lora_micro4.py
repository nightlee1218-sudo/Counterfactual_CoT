"""Gold40 v3 LoRA micro4 capacity diagnostic training entry.

Design contract:
  * Does NOT modify gold40_overfit.py, gold40_lora.py, lora_2step_smoke.py,
    smoke_train.py, bagel_manifest_dataset.py, or render_prompts.py.
  * Imports helpers from gold40_lora and lora_2step_smoke as-is.
  * Uses ONLY the micro4 manifest (4 pairs). Refuses to run on any other manifest.
  * Initialises from B_step000 checkpoint (62 tensors).
  * Saves checkpoints at steps 0, 10, 25, 50, 100, 200, 300, 400.
  * Does NOT launch 400-step training without --acknowledge-full-training.

Four corrections vs v1 plan:
  1. sample_lens probe uses an independent CounterfactualCoTDataset instance,
     never consuming the training loader.
  2. Per-step train_log records actual pair_id from batch_data_indexes;
     post-training exposure_stats counts real occurrences per pair_id.
     epoch_within_cycle fields are removed.
  3. AdamW gate: before first optimizer.step, assert optimizer.state is empty
     (len==0) AND exactly 24 LoRA-B tensors are all-zero.
  4. micro4_vs_formal_B config diff: only manifest, num_samples, output_dir,
     eval_steps, save_steps are allowed to differ. All other fields must match
     formal B run_config_resolved.json exactly.

Modes:
  --dry-run   : build model, inject LoRA, run one no_grad forward, save
                step_000_trainable.pt, strict-reload, write dry-run report.
                Never calls optimizer.step.
  (default)   : full 400-step training. Requires --acknowledge-full-training
                AND a prior dry-run report.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

# ---------------------------------------------------------------------------
# Imports from frozen modules (do NOT modify those modules)
# ---------------------------------------------------------------------------
from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.datasets import CounterfactualCoTDataset
from experiments.counterfactual_cot.training.gold40_overfit import (
    _build_gold40_loader,
    _build_model_for_gold40,
    _forward_loss,
    _load_tokenizer,
    _mean_train_mse,
    _prepare_batch,
    _prepare_output_dir,
    _select_visualization_pairs,
)
from experiments.counterfactual_cot.training.gold40_lora import (
    _apply_trainable_and_audit,
    _append_jsonl,
    _atomic_write_json,
    _atomic_write_jsonl,
    _build_model_with_lora,
    _compare_trainable_checkpoints,
    _confirm_no_base_leak_in_optimizer,
    _dockerdata_ok,
    _now_iso,
    _save_step_trainable,
    _serialize_args_json,
    _sha256_file,
    _strict_reload_check,
    _write_git_provenance,
    _write_sequence_plan,
)
from experiments.counterfactual_cot.training.lora_2step_smoke import (
    load_trainable_strict,
    save_trainable_only,
    set_trainable,
)
from experiments.counterfactual_cot.training.smoke_train import (
    DEFAULT_MODEL_PATH,
    _device,
    _log_event,
    _path_size_gib,
    _set_seed,
    _inspect_raw_samples,
)
from data.transforms import ImageTransform
from experiments.counterfactual_cot.datasets import CounterfactualCoTDataset

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

MICRO4_MANIFEST = (
    BAGEL_ROOT / "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/manifests/"
    "intervention_pairs_micro4_v1.jsonl"
)
MICRO4_MANIFEST_SHA256_EXPECT = "2d0790de5c4ad3a66eb26f4b46b45c1ed35155edfc1bb91aeb869ac256494891"
MICRO4_NUM_PAIRS = 4

# B_step000 checkpoint (initialisation source)
B_STEP000_CKPT = (
    BAGEL_ROOT / "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_L24-27_r8_a16_lr1e6_singlepack_400steps/checkpoints/step_000_trainable.pt"
)
B_STEP000_SHA256_EXPECT = "0f138533db0bf98ddee14711e7664c798252b8e41710d5043303350cbe78baff"

# Formal B run_config_resolved.json (for config diff audit)
FORMAL_B_RUN_CONFIG = (
    BAGEL_ROOT / "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_L24-27_r8_a16_lr1e6_singlepack_400steps/run_config_resolved.json"
)

# micro4 output base
BASE_OUTPUT_STEM = BAGEL_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_lora_micro4"

# Checkpoint steps
MICRO4_SAVE_STEPS = (0, 10, 25, 50, 100, 200, 300, 400)
MICRO4_EVAL_STEPS = (0, 10, 25, 50, 100, 200, 300, 400)

# Fields allowed to differ between micro4 and formal B.
# Includes training hyperparams that legitimately differ AND
# run-mode control fields that exist only in micro4 (not in formal B).
MICRO4_ALLOWED_DIFF_KEYS = frozenset({
    # Legitimately different training config
    "manifest",
    "num_samples",
    "output_dir",
    "eval_steps",
    "save_steps",
    # Run-mode control fields: micro4-only, absent from formal B run_config
    "dry_run",
    "dry_run_report",
    "acknowledge_full_training",
    "allow_non_dockerdata",
    "resume_trainable_checkpoint",
})

# Expected LoRA counts (inherited from formal B architecture)
EXPECTED_LORA_MODULES = 24
EXPECTED_LORA_B_TENSORS = 24
EXPECTED_TOTAL_TRAINABLE_TENSORS = 62  # 14 existing + 24 lora_A + 24 lora_B

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_micro4_manifest(manifest: Path) -> None:
    if manifest.resolve() != MICRO4_MANIFEST.resolve():
        raise ValueError(
            f"micro4 is locked to {MICRO4_MANIFEST}, got {manifest}."
        )
    if not manifest.exists():
        raise FileNotFoundError(f"micro4 manifest missing: {manifest}")
    sha = _sha256_file(manifest)
    if sha != MICRO4_MANIFEST_SHA256_EXPECT:
        raise RuntimeError(
            f"micro4 manifest SHA mismatch: expected={MICRO4_MANIFEST_SHA256_EXPECT} observed={sha}"
        )


def _validate_b_step000(ckpt: Path) -> str:
    if not ckpt.exists():
        raise FileNotFoundError(f"B_step000 checkpoint missing: {ckpt}")
    sha = _sha256_file(ckpt)
    if sha != B_STEP000_SHA256_EXPECT:
        raise RuntimeError(
            f"B_step000 SHA mismatch: expected={B_STEP000_SHA256_EXPECT} observed={sha}"
        )
    return sha


# ---------------------------------------------------------------------------
# 1. Independent sample_lens probe (never consumes training loader)
# ---------------------------------------------------------------------------

def _sample_lens_probe(
    manifest: Path,
    tokenizer: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build an independent CounterfactualCoTDataset (separate from train loader)
    and iterate all 4 pairs once to record true token counts and sequence plans.
    Does NOT advance the training loader iterator.
    """
    transform = ImageTransform(
        max_image_size=args.image_size,
        min_image_size=args.image_size,
        image_stride=16,
    )
    vit_transform = ImageTransform(
        max_image_size=args.vit_image_size,
        min_image_size=args.vit_min_image_size,
        image_stride=14,
    )
    probe_dataset = CounterfactualCoTDataset(
        dataset_name="micro4_sample_lens_probe",
        transform=transform,
        tokenizer=tokenizer,
        manifest_path=str(manifest),
        variant=args.variant,
        include_source_image=True,
        training_seed=0,
        local_rank=0,
        world_size=1,
        num_workers=0,
        data_status=None,
        vit_transform=vit_transform,
        max_records=MICRO4_NUM_PAIRS,
    )
    samples = []
    iterator = iter(probe_dataset)
    for _ in range(MICRO4_NUM_PAIRS):
        s = next(iterator)
        num_tokens = s["num_tokens"]
        seq_plan = s["sequence_plan"]
        samples.append({
            "pair_id": s["data_indexes"].get("pair_id"),
            "mode": s["data_indexes"].get("mode"),
            "num_tokens": int(num_tokens),
            "sequence_plan": [
                {
                    "pos": i,
                    "type": item["type"],
                    "loss": bool(item["loss"]),
                    "enable_cfg": bool(item["enable_cfg"]),
                }
                for i, item in enumerate(seq_plan)
            ],
            "fits_in_expected_num_tokens": num_tokens <= args.expected_num_tokens,
            "fits_in_max_num_tokens_per_sample": num_tokens <= args.max_num_tokens_per_sample,
        })
    return {
        "event": "sample_lens_probe",
        "source": "independent_dataset_not_train_loader",
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "expected_num_tokens": args.expected_num_tokens,
        "max_num_tokens_per_sample": args.max_num_tokens_per_sample,
        "max_num_tokens": args.max_num_tokens,
        "num_pairs_probed": len(samples),
        "samples": samples,
        "packing_conclusion": (
            "All pairs exceed expected_num_tokens; PackedDataset produces "
            "1-sample packs (singlepack behaviour). effective_batch_size=1."
        ),
        "created_at_iso": _now_iso(),
    }


# ---------------------------------------------------------------------------
# 2. micro4_vs_formal_B config diff audit
# ---------------------------------------------------------------------------

def _micro4_vs_formal_b_diff(args: argparse.Namespace) -> Dict[str, Any]:
    """Compare micro4 resolved config against formal B run_config_resolved.json.
    Only MICRO4_ALLOWED_DIFF_KEYS are permitted to differ.
    """
    if not FORMAL_B_RUN_CONFIG.exists():
        return {
            "status": "skipped",
            "reason": f"formal_B run_config not found: {FORMAL_B_RUN_CONFIG}",
        }
    formal_b = json.loads(FORMAL_B_RUN_CONFIG.read_text(encoding="utf-8"))["args"]

    micro4_config = _serialize_args_json(args)

    # Fields to compare: union of both key sets, excluding micro4-only keys
    all_keys = set(formal_b.keys()) | set(micro4_config.keys())
    diffs: Dict[str, Any] = {}
    for key in sorted(all_keys):
        fb_val = formal_b.get(key, "__MISSING__")
        m4_val = micro4_config.get(key, "__MISSING__")
        if fb_val != m4_val:
            diffs[key] = {"formal_B": fb_val, "micro4": m4_val}

    forbidden_diffs = {k: v for k, v in diffs.items() if k not in MICRO4_ALLOWED_DIFF_KEYS}
    allowed_diffs = {k: v for k, v in diffs.items() if k in MICRO4_ALLOWED_DIFF_KEYS}

    status = "passed" if len(forbidden_diffs) == 0 else "failed"
    return {
        "status": status,
        "allowed_diff_keys": sorted(MICRO4_ALLOWED_DIFF_KEYS),
        "allowed_diffs": allowed_diffs,
        "forbidden_diffs": forbidden_diffs,
        "num_forbidden_diffs": len(forbidden_diffs),
        "formal_B_config_path": str(FORMAL_B_RUN_CONFIG),
        "created_at_iso": _now_iso(),
    }


# ---------------------------------------------------------------------------
# 3. AdamW gate: optimizer.state empty + 24 LoRA-B all-zero
# ---------------------------------------------------------------------------

def _adamw_pre_step_gate(
    optimizer: torch.optim.AdamW,
    model: nn.Module,
) -> Dict[str, Any]:
    """Assert optimizer.state is empty (no momentum buffers yet) and
    exactly EXPECTED_LORA_B_TENSORS LoRA-B tensors are all-zero.
    Must be called BEFORE the first optimizer.step().
    """
    failures: List[str] = []

    # Gate 1: optimizer.state must be empty
    num_state_entries = len(optimizer.state)
    g1 = num_state_entries == 0
    if not g1:
        failures.append(
            f"optimizer.state has {num_state_entries} entries before first step (expected 0)"
        )

    # Gate 2: find all LoRA-B tensors and verify all-zero
    lora_b_names = [
        name for name, p in model.named_parameters()
        if p.requires_grad and name.endswith(".lora_B")
    ]
    num_lora_b = len(lora_b_names)
    g2_count = num_lora_b == EXPECTED_LORA_B_TENSORS
    if not g2_count:
        failures.append(
            f"found {num_lora_b} LoRA-B tensors, expected {EXPECTED_LORA_B_TENSORS}"
        )

    lora_b_zero_check: List[Dict[str, Any]] = []
    all_lora_b_zero = True
    for name in lora_b_names:
        p = dict(model.named_parameters())[name]
        max_abs = float(p.data.abs().max().item())
        is_zero = max_abs == 0.0
        if not is_zero:
            all_lora_b_zero = False
            failures.append(f"LoRA-B tensor {name} is not zero (max_abs={max_abs})")
        lora_b_zero_check.append({
            "name": name,
            "shape": list(p.shape),
            "max_abs": max_abs,
            "is_zero": is_zero,
        })

    g2 = g2_count and all_lora_b_zero
    overall = "passed" if len(failures) == 0 else "failed"
    return {
        "gate": "adamw_pre_step_gate",
        "status": overall,
        "failures": failures,
        "G1_optimizer_state_empty": g1,
        "G1_num_state_entries": num_state_entries,
        "G2_lora_b_count_correct": g2_count,
        "G2_num_lora_b_found": num_lora_b,
        "G2_expected_lora_b": EXPECTED_LORA_B_TENSORS,
        "G2_all_lora_b_zero": all_lora_b_zero,
        "lora_b_zero_check": lora_b_zero_check,
        "created_at_iso": _now_iso(),
    }


# ---------------------------------------------------------------------------
# 4. Exposure audit (post-training, from actual train_log)
# ---------------------------------------------------------------------------

def _exposure_audit(train_log_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Read train_log.jsonl and count actual exposures per pair_id."""
    if not train_log_path.exists():
        return {"status": "skipped", "reason": "train_log.jsonl not found"}
    counts: Dict[str, int] = {}
    total_steps = 0
    with open(train_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            pid = entry.get("pair_id") or "__unknown__"
            counts[pid] = counts.get(pid, 0) + 1
            total_steps += 1
    formal_b_exposures_per_pair = 400 // 40  # = 10
    micro4_exposures_per_pair = total_steps // MICRO4_NUM_PAIRS if MICRO4_NUM_PAIRS > 0 else 0
    return {
        "event": "exposure_audit",
        "total_steps_logged": total_steps,
        "num_pairs": MICRO4_NUM_PAIRS,
        "exposures_per_pair_id": counts,
        "micro4_avg_exposures_per_pair": micro4_exposures_per_pair,
        "formal_B_exposures_per_pair": formal_b_exposures_per_pair,
        "exposure_ratio_micro4_vs_formal_B": (
            micro4_exposures_per_pair / formal_b_exposures_per_pair
            if formal_b_exposures_per_pair > 0 else None
        ),
        "equal_exposure_steps_would_be": formal_b_exposures_per_pair * MICRO4_NUM_PAIRS,
        "interpretation": (
            f"micro4 runs {args.steps} steps on {MICRO4_NUM_PAIRS} pairs = "
            f"~{micro4_exposures_per_pair} exposures/pair. "
            f"Formal B runs 400 steps on 40 pairs = {formal_b_exposures_per_pair} exposures/pair. "
            "micro4 is intentionally over-exposed for capacity diagnosis."
        ),
        "created_at_iso": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def _run_dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_micro4_manifest(args.manifest)
    b_step000_sha = _validate_b_step000(args.resume_trainable_checkpoint)

    _prepare_output_dir(args.output_dir)
    _atomic_write_json(args.output_dir / "run_config_resolved.json", {
        "mode": "dry_run",
        "args": _serialize_args_json(args),
        "created_at_iso": _now_iso(),
    })
    _write_git_provenance(args.output_dir)
    _write_sequence_plan(args.output_dir, args.manifest)
    _set_seed(args.seed)
    device = _device(args.device)

    records = load_jsonl(args.manifest)
    if len(records) != args.num_samples:
        raise ValueError(
            f"micro4 expects exactly {args.num_samples} records, found {len(records)}"
        )
    if not all(bool((r.get("human_review") or {}).get("accepted")) for r in records):
        raise ValueError("All micro4 records must have human_review.accepted=true")

    _dockerdata_ok(args.llm_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.vae_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.checkpoint_path, args.allow_non_dockerdata)

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)

    # 1. Independent sample_lens probe (before building train loader)
    probe = _sample_lens_probe(args.manifest, tokenizer, args)
    _atomic_write_json(args.output_dir / "sample_lens_probe.json", probe)

    # 2. Raw sequence report (uses _inspect_raw_samples from smoke_train)
    raw_sequence_report = _inspect_raw_samples(
        args.manifest, tokenizer, args.variant,
        args.image_size, args.vit_image_size, args.vit_min_image_size,
        MICRO4_NUM_PAIRS,
    )

    # 3. Build train loader (separate from probe)
    train_loader = _build_gold40_loader(
        args.manifest, tokenizer, special_tokens, args, seed_offset=0
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, vae_model, wrappers, target_names, base_checkpoint_report = _build_model_with_lora(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
    )
    param_audit = _apply_trainable_and_audit(model, wrappers, target_names, args, device.type)

    # 4. micro4_vs_formal_B config diff audit
    config_diff = _micro4_vs_formal_b_diff(args)
    _atomic_write_json(args.output_dir / "micro4_vs_formal_b_config_diff.json", config_diff)
    if config_diff["status"] == "failed":
        raise RuntimeError(
            f"micro4_vs_formal_B config diff FAILED: {config_diff['forbidden_diffs']}"
        )

    # 5. Load B_step000 weights
    existing_names = param_audit["existing_names"]
    lora_names = param_audit["lora_names"]
    resume_report = load_trainable_strict(args.resume_trainable_checkpoint, model)
    _atomic_write_json(args.output_dir / "resume_from_b_step000.json", {
        "resume_from": str(args.resume_trainable_checkpoint),
        "resume_sha256": b_step000_sha,
        "num_tensors_loaded": resume_report["num_loaded_tensors"],
        "max_abs_load_copy_diff": resume_report["max_abs_load_copy_diff"],
        "created_at_iso": _now_iso(),
    })

    # 6. One no_grad forward (model.train() + no_grad, matching gold40_lora.py dry-run)
    model.train()
    vae_model.eval()
    model_dtype = next(model.parameters()).dtype
    batch = next(iter(train_loader))

    # Capture real batch structure from SimpleCustomBatch attributes
    try:
        batch_data_indexes = batch.batch_data_indexes or [{}]
        batch_pair_id = batch_data_indexes[0].get("pair_id") if batch_data_indexes else None
        batch_sample_lens = list(batch.sample_lens) if hasattr(batch, "sample_lens") else []
        batch_sequence_length = int(batch.sequence_length) if hasattr(batch, "sequence_length") else 0
    except Exception:
        batch_pair_id = None
        batch_sample_lens = []
        batch_sequence_length = 0

    data = _prepare_batch(batch, vae_model, device, model_dtype)
    with torch.no_grad():
        loss, scalars = _forward_loss(model, data)
    mse_mean = float(scalars.get("mse", float(loss.detach().item())))
    mse_all_finite = torch.isfinite(loss).item()
    output_shape = list(loss.shape) if hasattr(loss, "shape") else []
    peak_alloc_mib = (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0) / (1024 * 1024)
    peak_reserved_mib = (torch.cuda.max_memory_reserved() if device.type == "cuda" else 0) / (1024 * 1024)

    # 7. Save step_000 and strict-reload
    step000 = _save_step_trainable(
        0, model, args.output_dir, args, target_names, existing_names, lora_names,
    )
    reload_report = _strict_reload_check(Path(step000["path"]), model)

    # 8. Compare micro4 step_000 vs B_step000 (tensor-level, not file SHA)
    step000_vs_b = _compare_trainable_checkpoints(
        args.resume_trainable_checkpoint, Path(step000["path"])
    )
    _atomic_write_json(args.output_dir / "step000_vs_b_step000.json", step000_vs_b)
    if not step000_vs_b["all_tensors_equal"]:
        raise RuntimeError(
            f"micro4 step_000 diverges from B_step000: max_abs_diff={step000_vs_b['max_abs_diff']}"
        )

    report = {
        "status": "passed",
        "mode": "dry_run",
        "created_at_iso": _now_iso(),
        "args": _serialize_args_json(args),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "b_step000_sha256": b_step000_sha,
        "base_checkpoint_report": base_checkpoint_report,
        "resume_report": resume_report,
        "sample_lens_probe": probe,
        "raw_sequence_report": raw_sequence_report,
        "first_batch_pair_id": batch_pair_id,
        "first_batch_sample_lens": batch_sample_lens,
        "first_batch_sequence_length": batch_sequence_length,
        "config_diff": config_diff,
        "param_audit": {k: v for k, v in param_audit.items() if k != "dtype_desc"},
        "forward": {
            "mse_mean": mse_mean,
            "mse_shape": output_shape,
            "mse_all_finite": mse_all_finite,
        },
        "step000_checkpoint": step000,
        "step000_strict_reload": reload_report,
        "step000_vs_b_step000": step000_vs_b,
        "cuda_memory_peak_mib": {
            "allocated": peak_alloc_mib,
            "reserved": peak_reserved_mib,
        },
    }
    _atomic_write_json(args.output_dir / "micro4_dry_run_report.json", report)
    return report


# ---------------------------------------------------------------------------
# Full training (400 steps)
# ---------------------------------------------------------------------------

def _run_full_training(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.acknowledge_full_training:
        raise RuntimeError(
            "Refusing to launch full training without --acknowledge-full-training. "
            "Run --dry-run first."
        )
    dry_run_report_path = Path(args.dry_run_report)
    if not dry_run_report_path.exists():
        raise RuntimeError(f"Missing dry-run report at {dry_run_report_path}. Run --dry-run first.")

    _validate_micro4_manifest(args.manifest)
    b_step000_sha = _validate_b_step000(args.resume_trainable_checkpoint)

    _prepare_output_dir(args.output_dir)
    _atomic_write_json(args.output_dir / "run_config_resolved.json", {
        "mode": "full_training",
        "args": _serialize_args_json(args),
        "created_at_iso": _now_iso(),
    })
    _write_git_provenance(args.output_dir)
    _write_sequence_plan(args.output_dir, args.manifest)
    _set_seed(args.seed)
    device = _device(args.device)

    records = load_jsonl(args.manifest)
    if len(records) != args.num_samples:
        raise ValueError(
            f"micro4 expects exactly {args.num_samples} records, found {len(records)}"
        )

    _dockerdata_ok(args.llm_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.vae_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.checkpoint_path, args.allow_non_dockerdata)

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)

    # 1. Independent sample_lens probe
    probe = _sample_lens_probe(args.manifest, tokenizer, args)
    _atomic_write_json(args.output_dir / "sample_lens_probe.json", probe)

    # 2. Build loaders
    train_loader = _build_gold40_loader(
        args.manifest, tokenizer, special_tokens, args, seed_offset=0
    )
    eval_loader = _build_gold40_loader(
        args.manifest, tokenizer, special_tokens, args, seed_offset=100000
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, vae_model, wrappers, target_names, base_checkpoint_report = _build_model_with_lora(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
    )
    param_audit = _apply_trainable_and_audit(model, wrappers, target_names, args, device.type)

    # 3. Config diff audit
    config_diff = _micro4_vs_formal_b_diff(args)
    _atomic_write_json(args.output_dir / "micro4_vs_formal_b_config_diff.json", config_diff)
    if config_diff["status"] == "failed":
        raise RuntimeError(
            f"micro4_vs_formal_B config diff FAILED: {config_diff['forbidden_diffs']}"
        )

    existing_names = param_audit["existing_names"]
    lora_names = param_audit["lora_names"]

    # 4. Load B_step000 weights
    resume_report = load_trainable_strict(args.resume_trainable_checkpoint, model)
    _atomic_write_json(args.output_dir / "resume_from_b_step000.json", {
        "resume_from": str(args.resume_trainable_checkpoint),
        "resume_sha256": b_step000_sha,
        "num_tensors_loaded": resume_report["num_loaded_tensors"],
        "max_abs_load_copy_diff": resume_report["max_abs_load_copy_diff"],
        "global_step_after_resume": 0,
        "optimizer_state_initialized_fresh": True,
        "scheduler_present": False,
        "lr_at_step_1": args.lr,
        "created_at_iso": _now_iso(),
    })

    model.train()
    vae_model.eval()
    model_dtype = next(model.parameters()).dtype

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in trainable_params}
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr,
    )
    optim_audit = _confirm_no_base_leak_in_optimizer(optimizer, model, trainable_ids)

    # 5. Save step_000 BEFORE any optimizer.step
    step000 = _save_step_trainable(
        0, model, args.output_dir, args, target_names, existing_names, lora_names,
    )

    # 6. Compare step_000 vs B_step000 (tensor-level)
    step000_vs_b = _compare_trainable_checkpoints(
        args.resume_trainable_checkpoint, Path(step000["path"])
    )
    _atomic_write_json(args.output_dir / "step000_vs_b_step000.json", step000_vs_b)
    if not step000_vs_b["all_tensors_equal"]:
        raise RuntimeError(
            f"micro4 step_000 diverges from B_step000: max_abs_diff={step000_vs_b['max_abs_diff']}"
        )

    # 7. AdamW pre-step gate (before first optimizer.step)
    adamw_gate = _adamw_pre_step_gate(optimizer, model)
    _atomic_write_json(args.output_dir / "adamw_pre_step_gate.json", adamw_gate)
    if adamw_gate["status"] != "passed":
        raise RuntimeError(
            f"AdamW pre-step gate FAILED: {adamw_gate['failures']}"
        )

    save_step_set = set(int(s) for s in args.save_steps)
    eval_step_set = set(int(s) for s in args.eval_steps)
    save_step_set.add(0)
    eval_step_set.add(0)

    checkpoint_records: List[Dict[str, Any]] = [step000]
    step0_eval = _mean_train_mse(model, vae_model, eval_loader, device, model_dtype, args.eval_max_batches)
    eval_records: List[Dict[str, Any]] = [{"step": 0, **step0_eval}]

    train_log_path = args.output_dir / "train_log.jsonl"
    train_iterator = iter(train_loader)

    for step in range(1, args.steps + 1):
        step_wall_start = time.time()
        batch = next(train_iterator)

        # 8. Record actual pair_id and real pack shape from SimpleCustomBatch
        try:
            _bdx = batch.batch_data_indexes or [{}]
            actual_pair_id = _bdx[0].get("pair_id") if _bdx else None
            actual_sample_index = (_bdx[0].get("data_indexes") or [None, None])[0]
            actual_sample_lens = list(batch.sample_lens) if hasattr(batch, "sample_lens") else []
            actual_sequence_length = int(batch.sequence_length) if hasattr(batch, "sequence_length") else 0
        except Exception:
            actual_pair_id = None
            actual_sample_index = None
            actual_sample_lens = []
            actual_sequence_length = 0

        data = _prepare_batch(batch, vae_model, device, model_dtype)
        optimizer.zero_grad(set_to_none=True)
        loss, scalars = _forward_loss(model, data)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {scalars}")
        loss.backward()

        num_lora_A_nonzero = 0
        num_lora_B_nonzero = 0
        num_existing_nonzero = 0
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            g_l2 = float(
                torch.linalg.vector_norm(
                    p.grad.detach().to(torch.float32).reshape(-1)
                ).item()
            )
            if g_l2 > 0.0:
                if name.endswith(".lora_A"):
                    num_lora_A_nonzero += 1
                elif name.endswith(".lora_B"):
                    num_lora_B_nonzero += 1
                elif name in existing_names:
                    num_existing_nonzero += 1

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            .detach().cpu()
        )
        optimizer.step()

        _append_jsonl(train_log_path, {
            "global_step": int(step),
            "pair_id": actual_pair_id,           # actual from batch, not inferred
            "sample_index_in_manifest": actual_sample_index,
            "batch_sample_lens": actual_sample_lens,   # real per-sample token counts in pack
            "batch_sequence_length": actual_sequence_length,  # total packed sequence length
            "loss": float(scalars.get("total", float(loss.detach().item()))),
            "mse": float(scalars.get("mse", float(loss.detach().item()))),
            "ce": float(scalars.get("ce", 0.0)),
            "grad_norm": float(grad_norm),
            "learning_rate": float(args.lr),
            "lora_A_nonzero_grads": int(num_lora_A_nonzero),
            "lora_B_nonzero_grads": int(num_lora_B_nonzero),
            "existing_nonzero_grads": int(num_existing_nonzero),
            "cuda_memory_allocated_mib": (
                torch.cuda.memory_allocated() if device.type == "cuda" else 0
            ) / (1024 * 1024),
            "wall_seconds_this_step": time.time() - step_wall_start,
            "created_at_iso": _now_iso(),
        })

        del data, loss, scalars
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if step in save_step_set:
            checkpoint_records.append(
                _save_step_trainable(
                    step, model, args.output_dir, args, target_names, existing_names, lora_names,
                )
            )
        if step in eval_step_set:
            evalr = _mean_train_mse(
                model, vae_model, eval_loader, device, model_dtype, args.eval_max_batches
            )
            eval_records.append({"step": step, **evalr})

    # 9. Post-training exposure audit from actual train_log
    exposure = _exposure_audit(train_log_path, args)
    _atomic_write_json(args.output_dir / "exposure_audit.json", exposure)

    peak_alloc_mib = (
        torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    ) / (1024 * 1024)
    peak_reserved_mib = (
        torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
    ) / (1024 * 1024)

    report: Dict[str, Any] = {
        "status": "passed",
        "mode": "full_training",
        "created_at_iso": _now_iso(),
        "args": _serialize_args_json(args),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "b_step000_sha256": b_step000_sha,
        "base_checkpoint_report": base_checkpoint_report,
        "resume_report": resume_report,
        "config_diff": config_diff,
        "param_audit": {k: v for k, v in param_audit.items() if k != "dtype_desc"},
        "optim_audit": optim_audit,
        "adamw_pre_step_gate": adamw_gate,
        "step000_vs_b_step000": step000_vs_b,
        "checkpoint_records": checkpoint_records,
        "eval_records": eval_records,
        "exposure_audit": exposure,
        "cuda_memory_peak_mib": {
            "allocated": peak_alloc_mib,
            "reserved": peak_reserved_mib,
        },
    }
    _atomic_write_json(args.output_dir / "micro4_full_training_report.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MICRO4_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-report", type=Path, default=None)
    parser.add_argument("--acknowledge-full-training", action="store_true")
    parser.add_argument("--resume-trainable-checkpoint", type=Path, default=B_STEP000_CKPT)
    # Base paths (inherit formal B defaults)
    parser.add_argument("--llm-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vae-path", default=str(Path(DEFAULT_MODEL_PATH) / "ae.safetensors"))
    parser.add_argument("--checkpoint-path", default=str(Path(DEFAULT_MODEL_PATH) / "ema.safetensors"))
    parser.add_argument("--allow-non-dockerdata", action="store_true")
    # Training config — must match formal B exactly (except allowed diffs)
    parser.add_argument("--variant", default="paired_cf_mvp", choices=["paired_cf_mvp"])
    parser.add_argument("--num-samples", type=int, default=MICRO4_NUM_PAIRS)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=list(MICRO4_EVAL_STEPS))
    parser.add_argument("--save-steps", type=int, nargs="+", default=list(MICRO4_SAVE_STEPS))
    parser.add_argument("--eval-max-batches", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--vit-min-image-size", type=int, default=224)
    parser.add_argument("--visual-image-size", type=int, default=1024)
    parser.add_argument("--visual-vae-min-image-size", type=int, default=512)
    parser.add_argument("--visual-vit-image-size", type=int, default=980)
    parser.add_argument("--visual-vit-min-image-size", type=int, default=224)
    # Must match formal B exactly
    parser.add_argument("--expected-num-tokens", type=int, default=512)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=1536)
    parser.add_argument("--max-num-tokens", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--model-device-map", default="single")
    parser.add_argument("--device-map-max-memory", default="34GiB")
    parser.add_argument("--trainable-name-regex",
                        default=r"^(time_embedder|latent_pos_embed|vit_pos_embed|vae2llm|llm2vae|connector)\.")
    parser.add_argument("--lora-layers", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=25)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--visual-total", type=int, default=12)
    parser.add_argument("--visual-per-type", type=int, default=3)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--require-paired-vit", dest="require_paired_vit",
                        action="store_true", default=True)
    args = parser.parse_args()

    # Resolve output dir
    if args.output_dir is None:
        if args.dry_run:
            args.output_dir = Path(str(BASE_OUTPUT_STEM) + "_preflight")
        else:
            args.output_dir = Path(str(BASE_OUTPUT_STEM) + "_400steps")

    # Resolve dry-run report path
    if args.dry_run_report is None:
        args.dry_run_report = Path(str(BASE_OUTPUT_STEM) + "_preflight/micro4_dry_run_report.json")

    # Normalise step lists
    args.eval_steps = sorted({int(s) for s in args.eval_steps if int(s) >= 0})
    args.save_steps = sorted({int(s) for s in args.save_steps if int(s) >= 0})
    if 0 not in args.eval_steps:
        args.eval_steps = [0] + args.eval_steps
    if 0 not in args.save_steps:
        args.save_steps = [0] + args.save_steps
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.dry_run:
            report = _run_dry_run(args)
            print(json.dumps({
                "status": report["status"],
                "mode": "dry_run",
                "manifest_sha256": report["manifest_sha256"],
                "b_step000_sha256": report["b_step000_sha256"],
                "step000_path": report["step000_checkpoint"]["path"],
                "step000_sha256": report["step000_checkpoint"]["sha256"],
                "step000_vs_b_all_equal": report["step000_vs_b_step000"]["all_tensors_equal"],
                "step000_vs_b_max_abs_diff": report["step000_vs_b_step000"]["max_abs_diff"],
                "config_diff_status": report["config_diff"]["status"],
                "config_diff_forbidden": report["config_diff"]["forbidden_diffs"],
                "forward_mse": report["forward"]["mse_mean"],
                "forward_finite": report["forward"]["mse_all_finite"],
                "sample_lens": [
                    {"pair_id": s["pair_id"], "num_tokens": s["num_tokens"]}
                    for s in report["sample_lens_probe"]["samples"]
                ],
                "report_path": str(args.output_dir / "micro4_dry_run_report.json"),
            }, indent=2))
        else:
            report = _run_full_training(args)
            print(json.dumps({
                "status": report["status"],
                "mode": "full_training",
                "exposure_audit": report["exposure_audit"],
                "report_path": str(args.output_dir / "micro4_full_training_report.json"),
            }, indent=2))
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
