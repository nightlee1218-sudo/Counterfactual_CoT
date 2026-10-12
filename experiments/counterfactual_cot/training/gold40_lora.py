"""P1 Gold40 + L24-L27 LoRA formal training entry.

Design contract (do not violate; the CI gate will re-read this docstring):

  * NO modification to ``experiments/counterfactual_cot/training/gold40_overfit.py``.
    All Gold40 data loading, tokenizer setup, deterministic evaluation and
    visualization helpers are imported from that module as-is.
  * NO modification to ``experiments/counterfactual_cot/training/lora_2step_smoke.py``.
    ``LoRALinear`` / ``TARGET_MODULE_SUFFIXES`` / ``_iter_target_linear_names``
    / ``inject_lora`` / ``set_trainable`` / ``save_trainable_only`` /
    ``load_trainable_strict`` / dtype-audit helpers are imported from there.
  * NO change to the flow-matching loss in ``modeling/bagel/bagel.py`` or to
    the forward paths of ``modeling/bagel/qwen2_navit.py`` /
    ``modeling/bagel/siglip_navit.py``.
  * NO mask-loss, region-weight, source-preservation loss, or auxiliary loss.
  * Single A100 only. No DDP / FSDP / DeepSpeed / verl / LLaMA-Factory /
    manual device_map. ``batch_size == 1`` end-to-end.
  * The Base BAGEL model directory / ae.safetensors / ema.safetensors MUST
    live under ``/dockerdata`` in production.
  * Optimizer must match the old Gold40 baseline: ``torch.optim.AdamW`` with
    default (betas=(0.9,0.999), eps=1e-8, weight_decay=0.01), i.e. only
    ``lr=args.lr`` is passed. Every other hyperparameter must be shared with
    the baseline. The ``baseline_vs_lora_config_diff.json`` gate fails hard
    when non-LoRA differences appear.

Modes:

  * ``--dry-run`` (preflight): build, inject, run ONE no_grad forward, save
    ``step_000_trainable.pt`` under
    ``<base>_preflight/`` and strict-reload it. Never calls ``optimizer.step``.
  * (default) full training: refuses to launch unless
    ``--acknowledge-full-training`` is passed AND a preflight report exists
    at ``--dry-run-report`` (auto-resolved when defaults are used) AND
    ``baseline_vs_lora_config_diff.json`` (auto-generated) reports zero
    non-LoRA differences. Saves its own ``step_000_trainable.pt`` under
    ``<base>_400steps/checkpoints/`` BEFORE any ``optimizer.step``, and
    compares the 62 trainable tensors bit-for-bit with the dry-run
    Step000. Any mismatch aborts before Step 1.

Persistent artefacts written by every run:

  <output_dir>/
    ├── run_config_resolved.json      (resolved argparse namespace)
    ├── baseline_vs_lora_config_diff.json
    ├── manifest_audit_ref.json       (pointer to the CPU-stage audit)
    ├── sequence_plan.json            (dataloader pair order)
    ├── train_log.jsonl               (per-step JSON lines)
    ├── checkpoint_manifest.jsonl     (one line per saved checkpoint)
    ├── git_provenance.json           (commit / branch / status / script hashes)
    ├── error_report.json             (only when an exception is caught)
    └── gold40_lora_{dry_run,full_training}_report.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

# ---- Import (do NOT modify) helpers from the two source modules. ----

from experiments.counterfactual_cot.data.schema import load_jsonl  # noqa: E402
from experiments.counterfactual_cot.training.gold40_overfit import (  # noqa: E402
    DEFAULT_TRAINABLE_REGEX,
    FIXED_GOLD40_MANIFEST,
    _autocast_for_model,
    _build_model_for_gold40,
    _build_gold40_loader,
    _forward_loss,
    _load_tokenizer,
    _mean_train_mse,
    _prepare_batch,
    _prepare_output_dir,
    _select_visualization_pairs,
)
from experiments.counterfactual_cot.training.smoke_train import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    _assert_paired_vit_inputs,
    _batch_report,
    _cuda_memory_report,
    _device,
    _inspect_raw_samples,
    _log_event,
    _path_size_gib,
    _set_seed,
)
from experiments.counterfactual_cot.training.lora_2step_smoke import (  # noqa: E402
    LoRALinear,
    TARGET_MODULE_SUFFIXES,
    _assert_dtype_invariants,
    _describe_dtypes,
    _iter_target_linear_names,
    inject_lora,
    load_trainable_strict,
    save_trainable_only,
    set_trainable,
)
from experiments.counterfactual_cot.training.single_pair_overfit_diag import (  # noqa: E402
    _sha256_file,
)


# ---------------------------------------------------------------------------
# Frozen constants for this P1 run.
# ---------------------------------------------------------------------------

# ``<base>_preflight`` and ``<base>_400steps`` are the two output dirs.
# We keep them siblings so cold_reload_verifier can reference either without
# path munging.
BASE_OUTPUT_STEM = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_L24-27_r8_a16_lr1e6"
)

# Expected counts for the parameter-audit gate.
EXPECTED_EXISTING_TENSORS = 14
EXPECTED_EXISTING_PARAMETERS = 63_454_784
EXPECTED_LORA_MODULES = 24
EXPECTED_LORA_TENSORS = 48
EXPECTED_LORA_PARAMETERS = 2_162_688
EXPECTED_TOTAL_PARAMETERS = EXPECTED_EXISTING_PARAMETERS + EXPECTED_LORA_PARAMETERS  # 65_617_472

DEFAULT_EVAL_STEPS = (0, 25, 50, 100, 200, 300, 400)
DEFAULT_SAVE_STEPS = DEFAULT_EVAL_STEPS

# ---- Baseline reference for the config-diff gate. ----
# Values below reflect the frozen ``gold40_overfit.py`` (git-committed) defaults
# for the old 14-tensor 400-step Gold40 experiment. Anything the new LoRA run
# changes MUST fall into the whitelist (LoRA-only). Any other divergence is a
# hard gate failure.
BASELINE_CONFIG = {
    "manifest": None,  # Filled in below to point at Gold40 v2. v1 is still
                       # available at experiments/counterfactual_cot/reruns/
                       # nanobanana_v4_absolute_region/manifests/
                       # intervention_pairs_gold40_human_verified_final_normalized.jsonl
                       # but is NOT the training-side manifest for the new
                       # v2 formal experiments.
    "variant": "paired_cf_mvp",
    "num_samples": 40,
    "steps": 400,
    "lr": 1e-6,
    "seed": 1234,
    "generation_seed": 1234,
    "precision": "bf16",
    "model_device_map": "single",
    "trainable_name_regex": DEFAULT_TRAINABLE_REGEX,
    "image_size": 256,
    "vit_image_size": 224,
    "vit_min_image_size": 224,
    "expected_num_tokens": 512,
    "max_num_tokens_per_sample": 1536,
    "max_num_tokens": 1536,
    "visual_image_size": 1024,
    "visual_vae_min_image_size": 512,
    "visual_vit_image_size": 980,
    "visual_vit_min_image_size": 224,
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 2.0,
    "cfg_interval_start": 0.0,
    "cfg_interval_end": 1.0,
    "timestep_shift": 3.0,
    "num_timesteps": 25,
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "text_channel",
    "max_grad_norm": 1.0,
    "eval_max_batches": 4,
    "log_every": 1,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1e-8,
    "adam_weight_decay": 0.01,
    "prompt_construction": "paired_prompt(include_source_image=True)",
    "cot_construction": "render_cot_text",
    "native_loss": "flow_matching_mse_mean_full_frame",
    "eval_steps": [0, 25, 50, 100, 200, 300, 400],
    "save_steps": [0, 25, 50, 100, 200, 300, 400],
    # Reference for the actual-runtime-record of lr and every shared knob.
    "baseline_runtime_reference": str(
        Path(
            "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
            "outputs/gold40_overfit_single_400/checkpoints/step_000_trainable.pt"
        )
    ),
    # v1 and v2 manifests kept for provenance only. Training runs on v3.
    "gold40_v1_manifest": str(FIXED_GOLD40_MANIFEST),
    "gold40_v2_manifest": str(
        FIXED_GOLD40_MANIFEST.parent
        / "intervention_pairs_gold40_human_verified_final_normalized_v2_metadata_fixed.jsonl"
    ),
}
BASELINE_CONFIG["manifest"] = str(
    FIXED_GOLD40_MANIFEST.parent
    / "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl"
)

# Whitelist of divergences allowed for the LoRA run.
LORA_ONLY_DELTA_KEYS = {
    "lora_layers",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_modules",
    "lora_tensors",
    "lora_parameters",
    "lora_dtype",
    "total_trainable_parameters",
}

# Allowed Gold40 manifests for the LoRA run. v3 is the current training manifest;
# v1 and v2 are recognized only so provenance tooling can be re-run against
# historical snapshots without touching the shared v1 lock in gold40_overfit.
_ALLOWED_LORA_MANIFESTS = (
    FIXED_GOLD40_MANIFEST.parent
    / "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl",
    FIXED_GOLD40_MANIFEST.parent
    / "intervention_pairs_gold40_human_verified_final_normalized_v2_metadata_fixed.jsonl",
    FIXED_GOLD40_MANIFEST,
)


def _validate_lora_manifest(manifest: Path) -> None:
    """Assert *manifest* is one of the whitelisted Gold40 manifests.

    Replaces the v1-locked ``_validate_fixed_manifest`` from gold40_overfit for
    the LoRA run so training can point at Gold40 v3 without modifying any
    shared code path.
    """
    resolved = manifest.resolve()
    allowed_resolved = {p.resolve() for p in _ALLOWED_LORA_MANIFESTS}
    if resolved not in allowed_resolved:
        raise ValueError(
            "Gold40 LoRA is locked to the v1/v2/v3 manifest whitelist. "
            f"Got: {manifest}. Allowed: {sorted(str(p) for p in allowed_resolved)}"
        )
    if not manifest.exists():
        raise FileNotFoundError(f"Gold40 manifest does not exist: {manifest}")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _serialize_args_json(args: argparse.Namespace) -> Dict[str, Any]:
    serial: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serial[key] = str(value)
        elif isinstance(value, (list, tuple)):
            serial[key] = list(value)
        else:
            serial[key] = value
    return serial


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _atomic_write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _dockerdata_ok(path: Any, allow_non_dockerdata: bool) -> None:
    if allow_non_dockerdata:
        return
    if not str(path).startswith("/dockerdata/"):
        raise RuntimeError(
            f"Base path must live under /dockerdata for the formal run (got: {path}). "
            "Pass --allow-non-dockerdata only for local dry-run smoke; the resulting "
            "checkpoints are NOT the official artefacts."
        )


def _run_git(*cmd: str) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(BAGEL_ROOT), *cmd], stderr=subprocess.STDOUT,
        )
        return out.decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001
        return None


def _write_git_provenance(output_dir: Path) -> Dict[str, Any]:
    scripts = {
        "gold40_lora.py": _sha256_file(Path(__file__)),
        "gold40_full_manifest_audit.py": _sha256_file(
            Path(__file__).parent / "gold40_full_manifest_audit.py"
        ) if (Path(__file__).parent / "gold40_full_manifest_audit.py").exists() else None,
        "gold40_base_provenance.py": _sha256_file(
            Path(__file__).parent / "gold40_base_provenance.py"
        ) if (Path(__file__).parent / "gold40_base_provenance.py").exists() else None,
    }
    payload = {
        "created_at_iso": _now_iso(),
        "commit": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "status": _run_git("status", "--porcelain"),
        "diff_stat": _run_git("diff", "--stat"),
        "has_uncommitted_changes": bool((_run_git("status", "--porcelain") or "").strip()),
        "script_sha256": scripts,
        "bagel_root": str(BAGEL_ROOT),
    }
    _atomic_write_json(output_dir / "git_provenance.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Trainable checkpoint helpers (use lora_smoke format so cold_reload_verifier
# can consume it without any change).
# ---------------------------------------------------------------------------

def _make_lora_smoke_meta(
    step: int,
    args: argparse.Namespace,
    target_names: Sequence[str],
    existing_names: Sequence[str],
    lora_names: Sequence[str],
    output_dir: Path,
) -> Dict[str, Any]:
    return {
        "step": int(step),
        "global_step": int(step),
        "lora_target_names": list(target_names),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "existing_trainable_regex": args.trainable_name_regex,
        "existing_trainable_names": list(existing_names),
        "lora_trainable_names": list(lora_names),
        "created_at_iso": _now_iso(),
        "training_source": "gold40_lora.py",
        "output_dir": str(output_dir),
        "seed": int(args.seed),
    }


def _save_step_trainable(
    step: int,
    model: nn.Module,
    output_dir: Path,
    args: argparse.Namespace,
    target_names: Sequence[str],
    existing_names: Sequence[str],
    lora_names: Sequence[str],
) -> Dict[str, Any]:
    ckpt_path = output_dir / "checkpoints" / f"step_{step:03d}_trainable.pt"
    meta = _make_lora_smoke_meta(step, args, target_names, existing_names, lora_names, output_dir)
    save_trainable_only(ckpt_path, model, meta)
    sha = _sha256_file(ckpt_path)
    size = ckpt_path.stat().st_size
    entry = {
        "step": int(step),
        "path": str(ckpt_path),
        "sha256": sha,
        "size_bytes": int(size),
        "meta": meta,
        "created_at_iso": _now_iso(),
    }
    _append_jsonl(output_dir / "checkpoint_manifest.jsonl", entry)
    return entry


def _strict_reload_check(ckpt_path: Path, model: nn.Module) -> Dict[str, Any]:
    return load_trainable_strict(ckpt_path, model)


def _compare_trainable_checkpoints(a_path: Path, b_path: Path) -> Dict[str, Any]:
    """Bit-for-bit compare the trainable_state_dict tensors of two checkpoints.

    The overall ``.pt`` SHA256 is expected to differ due to non-deterministic
    serialization metadata; what MUST match is the 62 tensors, key-set,
    shape, dtype, and every element.
    """
    payload_a = torch.load(a_path, map_location="cpu", weights_only=False)
    payload_b = torch.load(b_path, map_location="cpu", weights_only=False)
    a_tensors = payload_a["trainable_state_dict"]
    b_tensors = payload_b["trainable_state_dict"]
    a_keys = set(a_tensors.keys())
    b_keys = set(b_tensors.keys())
    missing_in_b = sorted(a_keys - b_keys)
    missing_in_a = sorted(b_keys - a_keys)
    per_tensor: Dict[str, Dict[str, Any]] = {}
    max_abs_diff = 0.0
    dtype_mismatch: List[str] = []
    shape_mismatch: List[str] = []
    for name in sorted(a_keys & b_keys):
        ta = a_tensors[name]
        tb = b_tensors[name]
        if tuple(ta.shape) != tuple(tb.shape):
            shape_mismatch.append(name)
            per_tensor[name] = {
                "shape_a": list(ta.shape), "shape_b": list(tb.shape),
                "dtype_a": str(ta.dtype), "dtype_b": str(tb.dtype),
                "equal": False, "max_abs_diff": None,
            }
            continue
        if ta.dtype != tb.dtype:
            dtype_mismatch.append(name)
        diff = (ta.to(torch.float32) - tb.to(torch.float32)).abs().max().item()
        max_abs_diff = max(max_abs_diff, float(diff))
        per_tensor[name] = {
            "shape": list(ta.shape),
            "dtype_a": str(ta.dtype), "dtype_b": str(tb.dtype),
            "max_abs_diff": float(diff),
            "equal": diff == 0.0 and ta.dtype == tb.dtype,
        }
    return {
        "a_path": str(a_path),
        "b_path": str(b_path),
        "a_num_tensors": len(a_keys),
        "b_num_tensors": len(b_keys),
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
        "max_abs_diff": max_abs_diff,
        "all_tensors_equal": (
            not missing_in_a and not missing_in_b and not shape_mismatch
            and not dtype_mismatch and max_abs_diff == 0.0
        ),
        "per_tensor": per_tensor,
    }


# ---------------------------------------------------------------------------
# LoRA build + parameter audit
# ---------------------------------------------------------------------------

def _build_model_with_lora(
    args: argparse.Namespace,
    device: torch.device,
    tokenizer_size: int,
    num_new_tokens: int,
) -> Tuple[nn.Module, Any, Dict[str, LoRALinear], List[str], Dict[str, Any]]:
    _log_event(
        "bagel_lora_build_start",
        llm_path=args.llm_path,
        vae_path=args.vae_path,
        checkpoint_path=args.checkpoint_path,
        precision=args.precision,
    )
    model, vae_model, base_checkpoint_report = _build_model_for_gold40(
        args,
        device,
        tokenizer_size=tokenizer_size,
        num_new_tokens=num_new_tokens,
    )
    target_names = _iter_target_linear_names(model, args.lora_layers)
    if len(target_names) != len(args.lora_layers) * len(TARGET_MODULE_SUFFIXES):
        raise RuntimeError(
            f"LoRA target-name count mismatch: got {len(target_names)}, expected "
            f"{len(args.lora_layers) * len(TARGET_MODULE_SUFFIXES)} "
            f"({len(args.lora_layers)} layers x {len(TARGET_MODULE_SUFFIXES)} modules)"
        )
    wrappers = inject_lora(model, target_names, args.lora_r, args.lora_alpha, args.lora_dropout)
    return model, vae_model, wrappers, target_names, base_checkpoint_report


def _apply_trainable_and_audit(
    model: nn.Module,
    wrappers: Dict[str, LoRALinear],
    target_names: Sequence[str],
    args: argparse.Namespace,
    device_type: str,
) -> Dict[str, Any]:
    trainable_report = set_trainable(model, args.trainable_name_regex, wrappers)
    existing_names: List[str] = list(trainable_report["existing_trainable_names"])
    lora_names: List[str] = list(trainable_report["lora_trainable_names"])

    named = dict(model.named_parameters())
    existing_params = [(n, named[n]) for n in existing_names]
    lora_params = [(n, named[n]) for n in lora_names]

    existing_numel = int(sum(p.numel() for _, p in existing_params))
    lora_numel = int(sum(p.numel() for _, p in lora_params))
    total_numel = existing_numel + lora_numel

    lora_A_tensors = sum(1 for n in lora_names if n.endswith(".lora_A"))
    lora_B_tensors = sum(1 for n in lora_names if n.endswith(".lora_B"))

    audit = {
        "existing_tensors": len(existing_names),
        "existing_parameters": existing_numel,
        "lora_modules": len(target_names),
        "lora_tensors": len(lora_names),
        "lora_A_tensors": lora_A_tensors,
        "lora_B_tensors": lora_B_tensors,
        "lora_parameters": lora_numel,
        "total_trainable_parameters": total_numel,
        "target_names": list(target_names),
        "existing_names": existing_names,
        "lora_names": lora_names,
        "expected": {
            "existing_tensors": EXPECTED_EXISTING_TENSORS,
            "existing_parameters": EXPECTED_EXISTING_PARAMETERS,
            "lora_modules": EXPECTED_LORA_MODULES,
            "lora_tensors": EXPECTED_LORA_TENSORS,
            "lora_parameters": EXPECTED_LORA_PARAMETERS,
            "total_trainable_parameters": EXPECTED_TOTAL_PARAMETERS,
        },
    }

    problems: List[str] = []
    if len(existing_names) != EXPECTED_EXISTING_TENSORS:
        problems.append(f"existing_tensors={len(existing_names)} != {EXPECTED_EXISTING_TENSORS}")
    if existing_numel != EXPECTED_EXISTING_PARAMETERS:
        problems.append(f"existing_parameters={existing_numel} != {EXPECTED_EXISTING_PARAMETERS}")
    if len(target_names) != EXPECTED_LORA_MODULES:
        problems.append(f"lora_modules={len(target_names)} != {EXPECTED_LORA_MODULES}")
    if len(lora_names) != EXPECTED_LORA_TENSORS:
        problems.append(f"lora_tensors={len(lora_names)} != {EXPECTED_LORA_TENSORS}")
    if lora_A_tensors != EXPECTED_LORA_MODULES:
        problems.append(f"lora_A_tensors={lora_A_tensors} != {EXPECTED_LORA_MODULES}")
    if lora_B_tensors != EXPECTED_LORA_MODULES:
        problems.append(f"lora_B_tensors={lora_B_tensors} != {EXPECTED_LORA_MODULES}")
    if lora_numel != EXPECTED_LORA_PARAMETERS:
        problems.append(f"lora_parameters={lora_numel} != {EXPECTED_LORA_PARAMETERS}")
    if total_numel != EXPECTED_TOTAL_PARAMETERS:
        problems.append(f"total_trainable_parameters={total_numel} != {EXPECTED_TOTAL_PARAMETERS}")
    if problems:
        raise RuntimeError("Parameter audit failed:\n  - " + "\n  - ".join(problems))

    dtype_desc = _describe_dtypes(model, existing_names, lora_names, list(target_names))
    _assert_dtype_invariants(dtype_desc, expected_device_type=device_type)
    audit["dtype_desc"] = dtype_desc
    return audit


def _confirm_no_base_leak_in_optimizer(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    trainable_ids: set,
) -> Dict[str, Any]:
    optim_ids: set = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            optim_ids.add(id(p))
    extra_in_optim = optim_ids - trainable_ids
    missing_from_optim = trainable_ids - optim_ids
    frozen_leak: List[str] = []
    for name, p in model.named_parameters():
        if not p.requires_grad and id(p) in optim_ids:
            frozen_leak.append(name)
    if extra_in_optim or missing_from_optim or frozen_leak:
        raise RuntimeError(
            "Optimizer param audit failed: "
            f"extra_in_optim={len(extra_in_optim)}, missing_from_optim={len(missing_from_optim)}, "
            f"frozen_leaked_into_optim={frozen_leak[:8]}"
        )
    return {
        "num_optimizer_params": len(optim_ids),
        "num_trainable_params": len(trainable_ids),
        "frozen_leaked_into_optim": frozen_leak,
    }


# ---------------------------------------------------------------------------
# baseline_vs_lora_config_diff.json generator
# ---------------------------------------------------------------------------

def _diff_baseline_vs_lora(args: argparse.Namespace, param_audit: Dict[str, Any]) -> Dict[str, Any]:
    """Produce a machine-readable diff between the frozen Gold40 14-tensor
    baseline and the new LoRA run's resolved configuration.

    The gate PASSES iff every divergence key is in ``LORA_ONLY_DELTA_KEYS``.
    Any other divergence -> ``status='failed'``, and the caller (dry-run /
    full-training) must abort.
    """
    # Snapshot LoRA-specific fields.
    lora_side = {
        "lora_layers": list(args.lora_layers),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_dtype": "fp32",
        "lora_modules": int(param_audit["lora_modules"]),
        "lora_tensors": int(param_audit["lora_tensors"]),
        "lora_parameters": int(param_audit["lora_parameters"]),
        "total_trainable_parameters": int(param_audit["total_trainable_parameters"]),
    }

    # Everything the LoRA run resolves for a shared-with-baseline field.
    lora_shared = {
        "manifest": str(args.manifest),
        "variant": args.variant,
        "num_samples": int(args.num_samples),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "generation_seed": int(args.generation_seed),
        "precision": args.precision,
        "model_device_map": args.model_device_map,
        "trainable_name_regex": args.trainable_name_regex,
        "image_size": int(args.image_size),
        "vit_image_size": int(args.vit_image_size),
        "vit_min_image_size": int(args.vit_min_image_size),
        "expected_num_tokens": int(args.expected_num_tokens),
        "max_num_tokens_per_sample": int(args.max_num_tokens_per_sample),
        "max_num_tokens": int(args.max_num_tokens),
        "visual_image_size": int(args.visual_image_size),
        "visual_vae_min_image_size": int(args.visual_vae_min_image_size),
        "visual_vit_image_size": int(args.visual_vit_image_size),
        "visual_vit_min_image_size": int(args.visual_vit_min_image_size),
        "cfg_text_scale": float(args.cfg_text_scale),
        "cfg_img_scale": float(args.cfg_img_scale),
        "cfg_interval_start": float(args.cfg_interval_start),
        "cfg_interval_end": float(args.cfg_interval_end),
        "timestep_shift": float(args.timestep_shift),
        "num_timesteps": int(args.num_timesteps),
        "cfg_renorm_min": float(args.cfg_renorm_min),
        "cfg_renorm_type": args.cfg_renorm_type,
        "max_grad_norm": float(args.max_grad_norm),
        "eval_max_batches": int(args.eval_max_batches),
        "log_every": int(args.log_every),
        # Optimizer surface: we pass ONLY ``lr=args.lr`` to AdamW, matching
        # the baseline. Everything else is torch defaults.
        "adam_betas": [0.9, 0.999],
        "adam_eps": 1e-8,
        "adam_weight_decay": 0.01,
        "prompt_construction": "paired_prompt(include_source_image=True)",
        "cot_construction": "render_cot_text",
        "native_loss": "flow_matching_mse_mean_full_frame",
        "eval_steps": list(args.eval_steps),
        "save_steps": list(args.save_steps),
    }

    baseline_shared = {k: BASELINE_CONFIG[k] for k in lora_shared}
    diffs: Dict[str, Dict[str, Any]] = {}
    for key in sorted(lora_shared):
        if lora_shared[key] != baseline_shared[key]:
            diffs[key] = {"baseline": baseline_shared[key], "lora": lora_shared[key]}

    ok = len(diffs) == 0
    return {
        "status": "passed" if ok else "failed",
        "baseline_reference_runtime": BASELINE_CONFIG["baseline_runtime_reference"],
        "gold40_v1_manifest": BASELINE_CONFIG["gold40_v1_manifest"],
        "gold40_v2_manifest": BASELINE_CONFIG["gold40_v2_manifest"],
        "expected_lora_only_delta_keys": sorted(LORA_ONLY_DELTA_KEYS),
        "baseline_shared": baseline_shared,
        "lora_shared": lora_shared,
        "lora_only": lora_side,
        "non_lora_differences": diffs,
        "created_at_iso": _now_iso(),
    }


# ---------------------------------------------------------------------------
# sequence_plan.json (fixed dataloader pair order)
# ---------------------------------------------------------------------------

def _write_sequence_plan(output_dir: Path, manifest: Path) -> Dict[str, Any]:
    records = load_jsonl(manifest)
    order = [
        {
            "index": i,
            "pair_id": r.get("pair_id"),
            "intervention_type": (r.get("intervention") or {}).get("intervention_type"),
        }
        for i, r in enumerate(records)
    ]
    plan = {
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "num_records": len(records),
        "order": order,
        "notes": (
            "Order corresponds to the on-disk manifest order. The dataloader "
            "shuffles per-epoch using args.seed; the reproducibility contract "
            "is (seed, worker_id, epoch) -> deterministic access. "
            "This file is a static reference of the source order."
        ),
        "created_at_iso": _now_iso(),
    }
    _atomic_write_json(output_dir / "sequence_plan.json", plan)
    return plan


# ---------------------------------------------------------------------------
# error_report emission
# ---------------------------------------------------------------------------

def _emit_error_report(output_dir: Path, exc: BaseException, stage: str, extras: Optional[Dict[str, Any]] = None) -> None:
    payload = {
        "status": "failed",
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "extras": extras or {},
        "created_at_iso": _now_iso(),
    }
    try:
        _atomic_write_json(output_dir / "error_report.json", payload)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def _run_dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_lora_manifest(args.manifest)
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
        raise ValueError(f"P1 expects exactly {args.num_samples} Gold40 records, found {len(records)} in {args.manifest}")
    if not all(bool(row.get("human_review", {}).get("accepted")) for row in records):
        raise ValueError("All Gold40 records must be human_review.accepted=true before P1")

    _dockerdata_ok(args.llm_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.vae_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.checkpoint_path, args.allow_non_dockerdata)

    _log_event(
        "gold40_lora_dry_run_start",
        manifest=str(args.manifest),
        output_dir=str(args.output_dir),
        llm_path=args.llm_path,
        checkpoint_path=args.checkpoint_path,
        checkpoint_size_gib=_path_size_gib(args.checkpoint_path),
        vae_path=args.vae_path,
        vae_size_gib=_path_size_gib(args.vae_path),
        cuda_memory=_cuda_memory_report(device),
    )

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)
    raw_sequence_report = _inspect_raw_samples(
        args.manifest,
        tokenizer,
        args.variant,
        args.image_size,
        args.vit_image_size,
        args.vit_min_image_size,
        min(4, args.num_samples),
    )
    train_loader = _build_gold40_loader(args.manifest, tokenizer, special_tokens, args, seed_offset=0)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model, vae_model, wrappers, target_names, base_checkpoint_report = _build_model_with_lora(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
    )
    param_audit = _apply_trainable_and_audit(model, wrappers, target_names, args, device.type)

    # Baseline vs LoRA diff (hard gate).
    diff = _diff_baseline_vs_lora(args, param_audit)
    _atomic_write_json(args.output_dir / "baseline_vs_lora_config_diff.json", diff)
    if diff["status"] != "passed":
        raise RuntimeError(
            "baseline_vs_lora_config_diff FAILED. Non-LoRA divergences from the frozen "
            "Gold40 14-tensor baseline detected:\n"
            + json.dumps(diff["non_lora_differences"], ensure_ascii=False, indent=2)
        )

    model.train()
    vae_model.eval()
    model_dtype = next(model.parameters()).dtype

    existing_names = param_audit["existing_names"]
    lora_names = param_audit["lora_names"]

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in trainable_params}
    # Match gold40_overfit.py optimizer construction verbatim (only lr).
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr,
    )
    optim_audit = _confirm_no_base_leak_in_optimizer(optimizer, model, trainable_ids)

    batch = next(iter(train_loader))
    data = _prepare_batch(batch, vae_model, device, model_dtype)
    first_batch_report = _batch_report(batch, data, device)
    _assert_paired_vit_inputs(args, raw_sequence_report, first_batch_report, data)

    with torch.no_grad():
        with _autocast_for_model(model):
            out = model(**data)
    mse = out.get("mse") if isinstance(out, dict) else None
    if mse is None:
        raise RuntimeError("model forward returned no mse (visual_gen path missing)")
    mse_mean = float(mse.detach().float().mean().item())
    mse_all_finite = bool(torch.isfinite(mse).all().item())
    output_shape = list(mse.shape)
    if not mse_all_finite:
        raise RuntimeError(f"non-finite mse in dry-run forward: mean={mse_mean}")

    peak_alloc_mib = (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0) / (1024 * 1024)
    peak_reserved_mib = (torch.cuda.max_memory_reserved() if device.type == "cuda" else 0) / (1024 * 1024)

    del out, mse, data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    step000 = _save_step_trainable(
        0, model, args.output_dir, args, target_names, existing_names, lora_names,
    )
    reload_report = _strict_reload_check(Path(step000["path"]), model)

    report: Dict[str, Any] = {
        "status": "passed",
        "mode": "dry_run",
        "evaluation_type": "diagnostic_monitoring_only",
        "created_at_iso": _now_iso(),
        "args": _serialize_args_json(args),
        "records_seen": len(records),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "base": {
            "llm_path": str(args.llm_path),
            "vae_path": str(args.vae_path),
            "checkpoint_path": str(args.checkpoint_path),
            "allow_non_dockerdata": bool(args.allow_non_dockerdata),
        },
        "base_checkpoint_report": base_checkpoint_report,
        "raw_sequence_report": raw_sequence_report,
        "first_batch_report": first_batch_report,
        "baseline_vs_lora_config_diff": diff,
        "param_audit": {k: v for k, v in param_audit.items() if k != "dtype_desc"},
        "dtype_desc": param_audit["dtype_desc"],
        "optimizer_audit": optim_audit,
        "forward": {
            "mse_mean": mse_mean,
            "mse_shape": output_shape,
            "mse_all_finite": mse_all_finite,
        },
        "cuda_memory_peak_mib": {
            "allocated": peak_alloc_mib,
            "reserved": peak_reserved_mib,
        },
        "step000_checkpoint": step000,
        "step000_strict_reload": reload_report,
        "notes": [
            "The MSE and any imagery generated during training are diagnostic "
            "monitoring only. Formal evaluation must be launched per-checkpoint "
            "via cold_reload_verifier.py after training completes.",
        ],
    }

    _atomic_write_json(args.output_dir / "gold40_lora_dry_run_report.json", report)
    _log_event(
        "gold40_lora_dry_run_finished",
        status=report["status"],
        step000_path=step000["path"],
        step000_sha256=step000["sha256"],
        step000_bytes=step000["size_bytes"],
        strict_reload_num_loaded=reload_report.get("num_loaded_tensors"),
        strict_reload_max_abs_diff=reload_report.get("max_abs_load_copy_diff"),
        forward_mse_mean=mse_mean,
        forward_mse_all_finite=mse_all_finite,
        peak_allocated_mib=peak_alloc_mib,
        peak_reserved_mib=peak_reserved_mib,
    )
    return report


# ---------------------------------------------------------------------------
# Full training (400 steps by default).
# ---------------------------------------------------------------------------

def _run_full_training(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.acknowledge_full_training:
        raise RuntimeError(
            "Refusing to launch full training without --acknowledge-full-training. "
            "Run --dry-run first and only pass --acknowledge-full-training after the "
            "operator has approved."
        )
    dry_run_report_path = Path(args.dry_run_report)
    if not dry_run_report_path.exists():
        raise RuntimeError(
            f"Missing dry-run report at {dry_run_report_path}. Run --dry-run first."
        )
    dry_run_step000_path = Path(json.loads(dry_run_report_path.read_text("utf-8"))["step000_checkpoint"]["path"])
    if not dry_run_step000_path.exists():
        raise RuntimeError(f"Dry-run Step000 missing at {dry_run_step000_path}")

    _validate_lora_manifest(args.manifest)
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
        raise ValueError(f"P1 expects exactly {args.num_samples} Gold40 records, found {len(records)} in {args.manifest}")

    _dockerdata_ok(args.llm_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.vae_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.checkpoint_path, args.allow_non_dockerdata)

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)
    train_loader = _build_gold40_loader(args.manifest, tokenizer, special_tokens, args, seed_offset=0)
    eval_loader = _build_gold40_loader(args.manifest, tokenizer, special_tokens, args, seed_offset=100000)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, vae_model, wrappers, target_names, base_checkpoint_report = _build_model_with_lora(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
    )
    param_audit = _apply_trainable_and_audit(model, wrappers, target_names, args, device.type)

    diff = _diff_baseline_vs_lora(args, param_audit)
    _atomic_write_json(args.output_dir / "baseline_vs_lora_config_diff.json", diff)
    if diff["status"] != "passed":
        raise RuntimeError(
            "baseline_vs_lora_config_diff FAILED (full-training gate). See "
            f"{args.output_dir / 'baseline_vs_lora_config_diff.json'}"
        )

    existing_names = param_audit["existing_names"]
    lora_names = param_audit["lora_names"]

    model.train()
    vae_model.eval()
    model_dtype = next(model.parameters()).dtype

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in trainable_params}
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr,
    )
    optim_audit = _confirm_no_base_leak_in_optimizer(optimizer, model, trainable_ids)

    _ = _select_visualization_pairs(records, args.visual_per_type, args.visual_total)

    save_step_set = set(int(s) for s in args.save_steps)
    eval_step_set = set(int(s) for s in args.eval_steps)
    save_step_set.add(0)
    eval_step_set.add(0)

    # ---- Save Step000 BEFORE any optimizer.step and cross-check with dry-run.
    step000 = _save_step_trainable(
        0, model, args.output_dir, args, target_names, existing_names, lora_names,
    )
    cross_check = _compare_trainable_checkpoints(dry_run_step000_path, Path(step000["path"]))
    _atomic_write_json(args.output_dir / "step000_cross_check_with_dry_run.json", cross_check)
    if not cross_check["all_tensors_equal"]:
        raise RuntimeError(
            "Step000 divergence between dry-run and full-training initial state:\n"
            f"  max_abs_diff = {cross_check['max_abs_diff']}\n"
            f"  missing_in_a = {cross_check['missing_in_a'][:8]}\n"
            f"  missing_in_b = {cross_check['missing_in_b'][:8]}\n"
            f"  shape_mismatch = {cross_check['shape_mismatch'][:8]}\n"
            f"  dtype_mismatch = {cross_check['dtype_mismatch'][:8]}\n"
            "Abort before Step 1."
        )
    _log_event("step000_cross_check_passed",
               max_abs_diff=cross_check["max_abs_diff"],
               a_path=cross_check["a_path"], b_path=cross_check["b_path"])

    checkpoint_records: List[Dict[str, Any]] = [step000]
    step0_eval = _mean_train_mse(model, vae_model, eval_loader, device, model_dtype, args.eval_max_batches)
    eval_records: List[Dict[str, Any]] = [{"step": 0, **step0_eval}]

    train_log_path = args.output_dir / "train_log.jsonl"
    train_iterator = iter(train_loader)

    for step in range(1, args.steps + 1):
        step_wall_start = time.time()
        batch = next(train_iterator)
        pair_id = None
        try:
            pair_id = (batch.get("batch_data_indexes") or [{}])[0].get("pair_id")
        except Exception:  # noqa: BLE001
            pair_id = None
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
            g_l2 = float(torch.linalg.vector_norm(p.grad.detach().to(torch.float32).reshape(-1)).item())
            if g_l2 > 0.0:
                if name.endswith(".lora_A"):
                    num_lora_A_nonzero += 1
                elif name.endswith(".lora_B"):
                    num_lora_B_nonzero += 1
                elif name in existing_names:
                    num_existing_nonzero += 1

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm).detach().cpu()
        )
        optimizer.step()

        _append_jsonl(train_log_path, {
            "global_step": int(step),
            "pair_id": pair_id,
            "loss": float(scalars.get("total", float(loss.detach().item()))),
            "mse": float(scalars.get("mse", float(loss.detach().item()))),
            "ce": float(scalars.get("ce", 0.0)),
            "grad_norm": float(grad_norm),
            "learning_rate": float(args.lr),
            "lora_A_nonzero_grads": int(num_lora_A_nonzero),
            "lora_B_nonzero_grads": int(num_lora_B_nonzero),
            "existing_nonzero_grads": int(num_existing_nonzero),
            "cuda_memory_allocated_mib": (torch.cuda.memory_allocated() if device.type == "cuda" else 0) / (1024 * 1024),
            "cuda_memory_reserved_mib": (torch.cuda.memory_reserved() if device.type == "cuda" else 0) / (1024 * 1024),
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
            evalr = _mean_train_mse(model, vae_model, eval_loader, device, model_dtype, args.eval_max_batches)
            eval_records.append({"step": step, **evalr})
            _log_event("train_step_eval", step=step, **evalr)

    peak_alloc_mib = (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0) / (1024 * 1024)
    peak_reserved_mib = (torch.cuda.max_memory_reserved() if device.type == "cuda" else 0) / (1024 * 1024)

    report: Dict[str, Any] = {
        "status": "passed",
        "mode": "full_training",
        "evaluation_type": "diagnostic_monitoring_only",
        "created_at_iso": _now_iso(),
        "args": _serialize_args_json(args),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "base_checkpoint_report": base_checkpoint_report,
        "baseline_vs_lora_config_diff": diff,
        "param_audit": {k: v for k, v in param_audit.items() if k != "dtype_desc"},
        "optimizer_audit": optim_audit,
        "step000_cross_check_with_dry_run": cross_check,
        "checkpoint_records": checkpoint_records,
        "eval_records": eval_records,
        "cuda_memory_peak_mib": {
            "allocated": peak_alloc_mib,
            "reserved": peak_reserved_mib,
        },
        "notes": [
            "In-process MSE and any imagery generated during training are "
            "diagnostic monitoring only. Formal evaluation is per-checkpoint, "
            "in independent subprocesses, via cold_reload_verifier.py.",
        ],
    }
    _atomic_write_json(args.output_dir / "gold40_lora_full_training_report.json", report)
    return report


# ---------------------------------------------------------------------------
# Command-line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Manifest / output
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FIXED_GOLD40_MANIFEST.parent
        / "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl",
        help="Gold40 v3 (schema-consistent) manifest by default. v1 and v2 remain "
        "available for provenance but are NOT used for training after v3.",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Explicit output dir. If unset, defaults to "
                             "<BASE_OUTPUT_STEM>_preflight for --dry-run, "
                             "or <BASE_OUTPUT_STEM>_400steps otherwise.")
    parser.add_argument("--dry-run-report", type=Path, default=None,
                        help="Path to a prior dry-run report. Required for "
                             "full-training; auto-resolved to "
                             "<BASE_OUTPUT_STEM>_preflight/gold40_lora_dry_run_report.json "
                             "when unset.")
    # Base paths
    parser.add_argument("--llm-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vae-path", default=str(Path(DEFAULT_MODEL_PATH) / "ae.safetensors"))
    parser.add_argument("--checkpoint-path", default=str(Path(DEFAULT_MODEL_PATH) / "ema.safetensors"))
    parser.add_argument("--allow-non-dockerdata", action="store_true",
                        help="Allow --llm-path/--vae-path/--checkpoint-path outside /dockerdata. "
                             "Only for local dry-run smoke on the /group mirror.")
    # Dataset config (mirror gold40_overfit.py defaults exactly).
    parser.add_argument("--variant", default="paired_cf_mvp", choices=["paired_cf_mvp"])
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--vit-min-image-size", type=int, default=224)
    parser.add_argument("--expected-num-tokens", type=int, default=512)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=1536)
    parser.add_argument("--max-num-tokens", type=int, default=1536)
    # Visual generation (used only in full training).
    parser.add_argument("--visual-image-size", type=int, default=1024)
    parser.add_argument("--visual-vae-min-image-size", type=int, default=512)
    parser.add_argument("--visual-vit-image-size", type=int, default=980)
    parser.add_argument("--visual-vit-min-image-size", type=int, default=224)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=25)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    parser.add_argument("--visual-total", type=int, default=12)
    parser.add_argument("--visual-per-type", type=int, default=3)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--no-require-paired-vit", dest="require_paired_vit", action="store_false")
    parser.set_defaults(require_paired_vit=True)
    # Optimizer / training. AdamW receives only `lr`, matching gold40_overfit.py.
    # NOTE: the actual-runtime lr for the 14-tensor Gold40 400-step baseline is
    # 1e-6 (confirmed by optimizer_state_dict.param_groups[0].lr / initial_lr /
    # config.lr embedded in step_000/100/400 checkpoints across three baseline
    # directories). We use --lr 1e-6 as the default here, but every command
    # MUST pass --lr explicitly so the resolved-config record does not depend
    # on the argparse default.
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--save-steps", type=int, nargs="+", default=list(DEFAULT_SAVE_STEPS))
    parser.add_argument("--eval-steps", type=int, nargs="+", default=list(DEFAULT_EVAL_STEPS))
    parser.add_argument("--eval-max-batches", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generation-seed", type=int, default=1234)
    # System
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--model-device-map", default="single",
                        choices=("single",),
                        help="Single-card only for P1. Multi-card device_map is intentionally disabled.")
    parser.add_argument("--device-map-max-memory", default="34GiB")
    parser.add_argument("--trainable-name-regex", default=DEFAULT_TRAINABLE_REGEX)
    # LoRA
    parser.add_argument("--lora-layers", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    # Mode
    parser.add_argument("--dry-run", action="store_true",
                        help="Preflight only. Never calls optimizer.step.")
    parser.add_argument("--acknowledge-full-training", action="store_true",
                        help="Required to launch full N-step training. Ignored during --dry-run.")
    args = parser.parse_args()

    # Resolve output-dir defaults per mode.
    if args.output_dir is None:
        suffix = "_preflight" if args.dry_run else "_400steps"
        args.output_dir = Path(str(BASE_OUTPUT_STEM) + suffix)
    if args.dry_run_report is None:
        args.dry_run_report = Path(str(BASE_OUTPUT_STEM) + "_preflight") / "gold40_lora_dry_run_report.json"

    args.save_steps = sorted({int(s) for s in args.save_steps if int(s) >= 0})
    if 0 not in args.save_steps:
        args.save_steps = [0] + args.save_steps
    if args.steps not in args.save_steps:
        args.save_steps = sorted(args.save_steps + [int(args.steps)])

    args.eval_steps = sorted({int(s) for s in args.eval_steps if int(s) >= 0})
    if 0 not in args.eval_steps:
        args.eval_steps = [0] + args.eval_steps
    if args.steps not in args.eval_steps:
        args.eval_steps = sorted(args.eval_steps + [int(args.steps)])
    return args


def main() -> int:
    args = parse_args()
    stage = "dry_run" if args.dry_run else "full_training"
    try:
        if args.dry_run:
            _run_dry_run(args)
        else:
            _run_full_training(args)
        return 0
    except BaseException as exc:  # noqa: BLE001
        _emit_error_report(args.output_dir, exc, stage, extras={"argv": sys.argv})
        raise


if __name__ == "__main__":
    sys.exit(main())
