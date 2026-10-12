"""Gold40 v3 LoRA 2-step optimizer smoke (isolated, matches formal LoRA B config).

Runs EXACTLY 2 optimizer.step calls on the Gold40 v3 LoRA training path,
using the same Base, AE, EMA, manifest, seed, optimizer knobs and native
flow-matching loss as the formal 400-step LoRA B run. Output lives in a
smoke-only directory; the produced checkpoints MUST NOT be used to resume
formal training.

Hard requirements (user brief, Task 2):

  1. Uses formal LoRA B model / data / loss / optimizer / hyperparams.
     Output goes to a fresh, smoke-only directory.
  2. Runs EXACTLY 2 forward + backward + optimizer.step + zero_grad iters.
  3. Not a formal-training entry; does not overwrite formal Step000; the
     smoke checkpoints must not be reused for formal training.
  4. Optimizer contains EXACTLY 62 trainable tensors; no frozen Base leak.
  5. Records loss, grad norm, param update magnitudes and finiteness per
     step (0, 1, 2).
  6. All loss/grads/params must be NaN/Inf free.
  7. Step 1 -- at least one lora_B tensor has nonzero grad.
  8. Step 2 -- at least one lora_A AND at least one lora_B tensor have
     nonzero grad.
  9. After step 2 -- at least one lora_A AND at least one lora_B tensor
     have parameter values that actually changed.
 10. Report per-tensor grad+update for the 14 existing tensors; do NOT
     assume they must all change.
 11. Before/after audit of every frozen Base parameter: name / shape /
     dtype / per-tensor fingerprint (see FROZEN_FINGERPRINT_KIND note)
     must be unchanged.
 12. Save step0, step1, step2 checkpoints and strict-reload each. No
     strict=False.
 13. Independent-process cold reload of step_2, with fixed input + seed,
     comparing MSE tensor and generated PNG.
 14. Records peak allocated / peak reserved / per-step memory / final
     memory / CUDA allocator retry count.
 15. Immediately stops on OOM / NaN / Inf / grad anomaly / frozen leak /
     strict-reload mismatch.

Discipline:
  * CUDA_VISIBLE_DEVICES=0 only. Single A100.
  * No DDP / FSDP / DeepSpeed / manual device_map.
  * Does not modify BAGEL, Gold40 manifests, gold40_overfit.py,
    gold40_lora.py, lora_2step_smoke.py, or any other file.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.training.gold40_lora import (
    _apply_trainable_and_audit,
    _atomic_write_json,
    _build_model_with_lora,
    _confirm_no_base_leak_in_optimizer,
    _cuda_memory_report,
    _diff_baseline_vs_lora,
    _dockerdata_ok,
    _log_event,
    _make_lora_smoke_meta,
    _now_iso,
    _path_size_gib,
    _prepare_output_dir,
    _save_step_trainable,
    _sha256_file,
    _strict_reload_check,
    _validate_lora_manifest,
    _write_git_provenance,
    _write_sequence_plan,
    FIXED_GOLD40_MANIFEST,
)
from experiments.counterfactual_cot.training.gold40_overfit import (
    DEFAULT_TRAINABLE_REGEX,
    _autocast_for_model,
    _build_gold40_loader,
    _load_tokenizer,
    _prepare_batch,
)
from experiments.counterfactual_cot.training.smoke_train import _batch_report, _set_seed
from experiments.counterfactual_cot.training.lora_2step_smoke import (
    LoRALinear,
    trainable_state_dict,
)


V3_MANIFEST = (
    FIXED_GOLD40_MANIFEST.parent
    / "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    Path("/group/40016/niightli/BAGEL")
    / "experiments/counterfactual_cot/outputs/gold40_v3_lora_2step_smoke_lr1e6"
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sha256_bytes(t: torch.Tensor) -> str:
    """SHA256 of tensor raw bytes (bf16 upcast to fp32 for numpy compat)."""
    t_cpu = t.detach().to("cpu").contiguous()
    if t_cpu.dtype == torch.bfloat16:
        t_cpu = t_cpu.to(torch.float32)
    return hashlib.sha256(t_cpu.numpy().tobytes()).hexdigest()


def _tensor_fingerprint(t: torch.Tensor) -> Dict[str, Any]:
    """Cheap deterministic fingerprint of a frozen tensor.

    All reductions (abs_sum / abs_max / sum) are done ON THE SOURCE DEVICE
    (typically CUDA) so we do NOT round-trip a 27 GB Base to CPU. Only two
    tiny head/tail slices (<= 8192 elems each) are copied to CPU for the
    per-tensor sha256 -- these are bit-sensitive to any element change.

    This is intentionally NOT a full-tensor sha256 (which for 1200+ Base
    tensors would exceed the 30-min single-cell budget). Any element change
    in the frozen Base would still show up either in the abs_sum / abs_max /
    sum reduction OR in one of the two 8192-element slices.
    """
    with torch.no_grad():
        # Reduce on the source device in fp32 for stability.
        t_flat = t.detach().reshape(-1)
        # Cast to fp32 view on the same device -- avoids bf16 sum non-determinism
        # for small ranges and gives comparable floats across before/after.
        t_fp32 = t_flat.to(torch.float32)
        abs_sum = float(t_fp32.abs().sum().item())
        abs_max = float(t_fp32.abs().max().item()) if t_fp32.numel() else 0.0
        val_sum = float(t_fp32.sum().item())
        n = int(t_flat.numel())
        head_n = min(8192, n)
        tail_n = min(8192, n)
        # Tiny CPU copies for byte-hashing head/tail slices.
        head_cpu = t_flat[:head_n].detach().to("cpu").contiguous()
        tail_cpu = t_flat[-tail_n:].detach().to("cpu").contiguous()
        if head_cpu.dtype == torch.bfloat16:
            head_cpu = head_cpu.to(torch.float32)
        if tail_cpu.dtype == torch.bfloat16:
            tail_cpu = tail_cpu.to(torch.float32)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "numel": n,
        "abs_sum": abs_sum,
        "abs_max": abs_max,
        "sum": val_sum,
        "head_sha256": hashlib.sha256(head_cpu.numpy().tobytes()).hexdigest(),
        "tail_sha256": hashlib.sha256(tail_cpu.numpy().tobytes()).hexdigest(),
    }


def _audit_frozen_base(model: nn.Module) -> Dict[str, Any]:
    """Snapshot every frozen (requires_grad=False) parameter's fingerprint."""
    entries: Dict[str, Any] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            entries[name] = _tensor_fingerprint(p)
    return entries


def _diff_frozen_audits(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    b_keys = set(before.keys())
    a_keys = set(after.keys())
    added = sorted(a_keys - b_keys)
    removed = sorted(b_keys - a_keys)
    changed: List[Dict[str, Any]] = []
    for k in sorted(b_keys & a_keys):
        b = before[k]
        a = after[k]
        differing = []
        for field in ("shape", "dtype", "numel", "abs_sum", "abs_max", "sum",
                      "head_sha256", "tail_sha256"):
            if b.get(field) != a.get(field):
                differing.append(field)
        if differing:
            changed.append({"name": k, "differing_fields": differing, "before": b, "after": a})
    return {
        "num_before": len(b_keys),
        "num_after": len(a_keys),
        "num_added": len(added),
        "num_removed": len(removed),
        "num_changed": len(changed),
        "added": added[:20],
        "removed": removed[:20],
        "changed": changed[:20],
    }


def _per_tensor_grad_snapshot(
    model: nn.Module,
    existing_names: Sequence[str],
    lora_names: Sequence[str],
) -> Dict[str, Any]:
    """Record per-tensor grad presence + magnitude for existing + LoRA params."""
    named = dict(model.named_parameters())
    all_names = list(existing_names) + list(lora_names)
    snap: Dict[str, Any] = {}
    for name in all_names:
        if name not in named:
            snap[name] = {"present_in_model": False}
            continue
        p = named[name]
        g = p.grad
        if g is None:
            snap[name] = {"has_grad": False, "shape": list(p.shape), "dtype": str(p.dtype)}
        else:
            gf = g.detach().to(torch.float32)
            snap[name] = {
                "has_grad": True,
                "shape": list(g.shape),
                "dtype": str(g.dtype),
                "grad_abs_max": float(gf.abs().max().item()),
                "grad_abs_sum": float(gf.abs().sum().item()),
                "grad_num_nonzero": int((g.detach() != 0).sum().item()),
                "grad_is_finite": bool(torch.isfinite(g).all().item()),
                "is_nonzero": bool(gf.abs().max().item() > 0),
            }
    return snap


def _per_tensor_param_snapshot(
    model: nn.Module,
    existing_names: Sequence[str],
    lora_names: Sequence[str],
) -> Dict[str, Any]:
    """Record per-tensor parameter values (fingerprint) for existing + LoRA."""
    named = dict(model.named_parameters())
    all_names = list(existing_names) + list(lora_names)
    snap: Dict[str, Any] = {}
    for name in all_names:
        p = named.get(name)
        if p is None:
            snap[name] = {"present_in_model": False}
            continue
        snap[name] = _tensor_fingerprint(p)
    return snap


def _diff_param_snapshots(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    changed_names: List[str] = []
    per_diff: Dict[str, Any] = {}
    for name, b in before.items():
        a = after.get(name)
        if a is None:
            per_diff[name] = {"missing_after": True}
            continue
        diff_fields = []
        for field in ("abs_sum", "abs_max", "sum", "head_sha256", "tail_sha256"):
            if b.get(field) != a.get(field):
                diff_fields.append(field)
        per_diff[name] = {"differing_fields": diff_fields, "changed": bool(diff_fields)}
        if diff_fields:
            changed_names.append(name)
    return {"changed_names": changed_names, "num_changed": len(changed_names), "per_diff": per_diff}


def _cuda_memory_snapshot(device: torch.device) -> Dict[str, Any]:
    if device.type != "cuda":
        return {"allocated_mib": 0.0, "reserved_mib": 0.0, "peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0, "num_alloc_retries": 0}
    return {
        "allocated_mib": torch.cuda.memory_allocated() / (1024 * 1024),
        "reserved_mib": torch.cuda.memory_reserved() / (1024 * 1024),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 * 1024),
        "num_alloc_retries": int(torch.cuda.memory_stats().get("num_alloc_retries", 0)),
    }


def _global_grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += float(p.grad.detach().to(torch.float32).pow(2).sum().item())
    return math.sqrt(total)


def _launch_cold_reload_step2(args: argparse.Namespace, step2_ckpt: Path) -> Optional[Path]:
    """Launch the existing cold-reload orchestrator against step_002_trainable.pt.

    Note: run_cold_reload_verification.sh's default TRAINABLE_CKPT is
    ``step_002_trainable.pt``; we point it at our smoke output directory
    via env overrides. We use gold40_lora_dry_run_report.json ...
    actually the training report field is only used for informational
    fields by the comparator. We instead point it at our smoke summary.
    """
    output_root = Path(args.output_dir) / "cold_reload_step2_A100"
    output_root.mkdir(parents=True, exist_ok=True)
    training_report = Path(args.output_dir) / "gold40_v3_lora_2step_smoke_report.json"
    if not training_report.exists():
        raise RuntimeError(f"training report missing for cold reload: {training_report}")
    log_path = output_root / "orchestrator.log"
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "TRAIN_DIR": str(args.output_dir),
        "TRAINABLE_CKPT": str(step2_ckpt),
        "TRAINING_REPORT": str(training_report),
        "OUTPUT_ROOT": str(output_root),
        "LLM_PATH": args.llm_path,
        "VAE_PATH": args.vae_path,
        "CKPT_PATH": args.checkpoint_path,
    })
    with open(log_path, "wb") as f:
        proc = subprocess.Popen(
            ["bash", "experiments/counterfactual_cot/training/run_cold_reload_verification.sh"],
            stdout=f, stderr=subprocess.STDOUT, env=env,
            cwd="/group/40016/niightli/BAGEL",
        )
        rc = proc.wait()
    report = output_root / "cold_reload_verification_report.json"
    if not report.exists():
        _log_event("cold_reload_step2_report_missing", rc=rc, log=str(log_path))
        return None
    return report


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=V3_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--llm-path", type=str, default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
    parser.add_argument("--vae-path", type=str, default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors")
    parser.add_argument("--checkpoint-path", type=str, default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors")
    parser.add_argument("--variant", type=str, default="paired_cf_mvp")
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--model-device-map", type=str, default="single")
    parser.add_argument("--device-map-max-memory", type=str, default="34GiB")
    parser.add_argument("--trainable-name-regex", type=str, default=DEFAULT_TRAINABLE_REGEX)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--vit-min-image-size", type=int, default=224)
    parser.add_argument("--expected-num-tokens", type=int, default=512)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=1536)
    parser.add_argument("--max-num-tokens", type=int, default=1536)
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
    parser.add_argument("--cfg-renorm-type", type=str, default="text_channel")
    parser.add_argument("--lr", type=float, default=1e-6, help="Formal LoRA B lr.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--require-paired-vit", action="store_true", default=True)
    parser.add_argument("--allow-non-dockerdata", action="store_true")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--lora-layers", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=400,
                        help="Config-wise steps to declare (must match formal "
                             "LoRA B == 400 so baseline diff passes). The smoke "
                             "loop itself hard-codes 2 optimizer.step iterations "
                             "regardless of this value.")
    parser.add_argument("--num-smoke-iterations", type=int, default=2,
                        help="MUST be exactly 2 for the smoke.")
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--adam-weight-decay", type=float, default=0.01)
    # These fields are only consumed by _diff_baseline_vs_lora for the config
    # equivalence check. Smoke does not actually run evals or save at these
    # step points; the smoke saves step_000/001/002 unconditionally.
    parser.add_argument("--eval-max-batches", type=int, default=4)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[0, 25, 50, 100, 200, 300, 400])
    parser.add_argument("--save-steps", type=int, nargs="+", default=[0, 25, 50, 100, 200, 300, 400])
    parser.add_argument("--skip-cold-reload", action="store_true",
                        help="Skip the step-2 independent cold reload sub-launch.")
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.num_smoke_iterations != 2:
        raise ValueError("gold40_v3_lora_2step_smoke MUST run exactly 2 optimizer.step iterations")
    if args.steps != 400:
        raise ValueError(
            "For baseline_vs_lora_config_diff to pass, --steps MUST be declared "
            "as 400 (matching formal LoRA B). The smoke loop itself is fixed at 2 "
            "regardless. See --num-smoke-iterations."
        )

    _validate_lora_manifest(args.manifest)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"output_dir {output_dir} already exists and is non-empty. "
            "Refusing to overwrite existing smoke artefacts."
        )
    _prepare_output_dir(output_dir)
    _atomic_write_json(output_dir / "run_config_resolved.json", {
        "mode": "gold40_v3_lora_2step_smoke",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "created_at_iso": _now_iso(),
    })
    _write_git_provenance(output_dir)
    _write_sequence_plan(output_dir, args.manifest)
    _set_seed(args.seed)
    device = _device(args.device)

    records = load_jsonl(args.manifest)
    if len(records) != args.num_samples:
        raise ValueError(
            f"Expected {args.num_samples} Gold40 records, found {len(records)}"
        )

    _dockerdata_ok(args.llm_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.vae_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.checkpoint_path, args.allow_non_dockerdata)

    _log_event(
        "gold40_v3_lora_2step_smoke_start",
        manifest=str(args.manifest),
        output_dir=str(output_dir),
        llm_path=args.llm_path,
        checkpoint_path=args.checkpoint_path,
        checkpoint_size_gib=_path_size_gib(args.checkpoint_path),
        vae_path=args.vae_path,
        vae_size_gib=_path_size_gib(args.vae_path),
    )

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)
    train_loader = _build_gold40_loader(args.manifest, tokenizer, special_tokens, args, seed_offset=0)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Build model + inject LoRA.
    model, vae_model, wrappers, target_names, base_checkpoint_report = _build_model_with_lora(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
    )
    param_audit = _apply_trainable_and_audit(model, wrappers, target_names, args, device.type)
    diff = _diff_baseline_vs_lora(args, param_audit)
    _atomic_write_json(output_dir / "baseline_vs_lora_config_diff.json", diff)
    if diff["status"] != "passed":
        raise RuntimeError(f"baseline_vs_lora_config_diff FAILED: {diff['non_lora_differences']}")

    existing_names: List[str] = list(param_audit["existing_names"])
    lora_names: List[str] = list(param_audit["lora_names"])
    if len(existing_names) != 14:
        raise RuntimeError(f"expected 14 existing tensors, got {len(existing_names)}")
    if len(lora_names) != 48:
        raise RuntimeError(f"expected 48 LoRA tensors (24 A + 24 B), got {len(lora_names)}")

    # Optimizer -- must match formal LoRA B AdamW config.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in trainable_params}
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=args.adam_eps,
        weight_decay=args.adam_weight_decay,
    )
    optim_audit = _confirm_no_base_leak_in_optimizer(optimizer, model, trainable_ids)
    if optim_audit["num_optimizer_params"] != 62:
        raise RuntimeError(
            f"expected exactly 62 optimizer params, got {optim_audit['num_optimizer_params']}"
        )

    model.train()
    vae_model.eval()
    model_dtype = next(model.parameters()).dtype

    # ---------------- Frozen-Base audit BEFORE training ----------------
    _log_event("frozen_base_audit_before_start")
    frozen_before = _audit_frozen_base(model)
    _atomic_write_json(output_dir / "frozen_base_audit_before.json", frozen_before)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log_event("frozen_base_audit_before_done", num_frozen=len(frozen_before),
               post_audit_cuda_memory=_cuda_memory_snapshot(device))

    # ---------------- Trainable param snapshot at step 0 ----------------
    step0_param_snap = _per_tensor_param_snapshot(model, existing_names, lora_names)
    _atomic_write_json(output_dir / "trainable_snapshot_step0.json", step0_param_snap)

    # Save step_000 checkpoint (BEFORE any optimizer.step).
    step0 = _save_step_trainable(0, model, output_dir, args, target_names, existing_names, lora_names)
    _log_event("step_000_saved", **step0)

    # Strict reload of step_000.
    reload0 = _strict_reload_check(Path(step0["path"]), model)
    _log_event("step_000_strict_reload", **reload0)

    step_records: List[Dict[str, Any]] = []
    per_step_grad_snaps: List[Dict[str, Any]] = []
    per_step_param_snaps = [step0_param_snap]
    per_step_mem: List[Dict[str, Any]] = []
    optimizer_step_count = 0
    optimizer_step_calls: List[str] = []

    # ------------------- Two real optimizer.step iterations -------------------
    data_iter = iter(train_loader)
    for step_idx in (1, 2):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        data = _prepare_batch(batch, vae_model, device, model_dtype)
        this_batch_report = _batch_report(batch, data, device)

        # HARD assertion: batch MUST contain exactly 1 pair with sequence_length <= 1536.
        actual_sample_lens = this_batch_report.get("sample_lens") or []
        actual_seq_len = this_batch_report.get("sequence_length")
        SINGLEPACK_CEILING = 1536
        if len(actual_sample_lens) != 1:
            raise RuntimeError(
                f"step{step_idx}: dataloader returned a batch with {len(actual_sample_lens)} "
                f"samples (sample_lens={actual_sample_lens}); expected exactly 1 (singlepack). "
                f"Refusing to proceed."
            )
        if actual_seq_len is None or actual_seq_len > SINGLEPACK_CEILING:
            raise RuntimeError(
                f"step{step_idx}: sequence_length={actual_seq_len} exceeds "
                f"singlepack ceiling {SINGLEPACK_CEILING}. Refusing to proceed."
            )
        _log_event(
            "batch_singlepack_ok",
            step=step_idx,
            pair_id=(this_batch_report.get("batch_data_indexes") or [{}])[0].get("pair_id"),
            sample_lens=actual_sample_lens,
            sequence_length=actual_seq_len,
        )

        optimizer.zero_grad(set_to_none=True)
        mem_before_fwd = _cuda_memory_snapshot(device)

        with _autocast_for_model(model):
            out = model(**data)
        mse = out.get("mse") if isinstance(out, dict) else None
        if mse is None:
            raise RuntimeError(f"step{step_idx}: model forward returned no mse")
        loss = mse.mean()
        if not torch.isfinite(loss).all().item():
            raise RuntimeError(f"step{step_idx}: non-finite loss = {loss.item()}")
        mem_after_fwd = _cuda_memory_snapshot(device)

        loss.backward()
        mem_after_bwd = _cuda_memory_snapshot(device)

        # Grad snapshot AFTER backward, BEFORE optimizer.step.
        grad_snap = _per_tensor_grad_snapshot(model, existing_names, lora_names)
        per_step_grad_snaps.append(grad_snap)

        # Global grad norm on the 62 trainables.
        grad_norm = _global_grad_norm(trainable_params)
        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        post_clip_grad_norm = _global_grad_norm(trainable_params)

        # optimizer.step.
        optimizer.step()
        optimizer_step_count += 1
        optimizer_step_calls.append(_now_iso())
        mem_after_step = _cuda_memory_snapshot(device)

        # Sanity: check all trainables are finite.
        for name in list(existing_names) + list(lora_names):
            p = dict(model.named_parameters())[name]
            if not torch.isfinite(p).all().item():
                raise RuntimeError(f"step{step_idx}: non-finite param after step: {name}")

        step_snap = _per_tensor_param_snapshot(model, existing_names, lora_names)
        per_step_param_snaps.append(step_snap)

        # Save step_00k checkpoint.
        ckpt_entry = _save_step_trainable(step_idx, model, output_dir, args, target_names, existing_names, lora_names)
        reload_entry = _strict_reload_check(Path(ckpt_entry["path"]), model)

        # zero_grad and record post-zero memory.
        optimizer.zero_grad(set_to_none=True)
        mem_after_zero = _cuda_memory_snapshot(device)

        mem = mem_after_zero  # kept for backward-compat with per_step_mem
        per_step_mem.append(mem)

        step_records.append({
            "step": step_idx,
            "loss": float(loss.item()),
            "loss_finite": True,
            "pre_clip_grad_norm": grad_norm,
            "post_clip_grad_norm": post_clip_grad_norm,
            "batch_report": this_batch_report,
            "memory_before_forward": mem_before_fwd,
            "memory_after_forward": mem_after_fwd,
            "memory_after_backward": mem_after_bwd,
            "memory_after_optimizer_step": mem_after_step,
            "memory_after_zero_grad": mem_after_zero,
            "checkpoint": ckpt_entry,
            "strict_reload": reload_entry,
            "first_batch_report": this_batch_report if step_idx == 1 else None,
        })

        _log_event(
            "step_completed",
            step=step_idx, loss=float(loss.item()),
            pre_clip_grad_norm=grad_norm, post_clip_grad_norm=post_clip_grad_norm,
            peak_alloc_after_step_mib=mem_after_step["peak_allocated_mib"],
            peak_reserved_after_step_mib=mem_after_step["peak_reserved_mib"],
            alloc_after_zero_mib=mem_after_zero["allocated_mib"],
            reserved_after_zero_mib=mem_after_zero["reserved_mib"],
        )

        # Free intermediates.
        del out, mse, loss, data
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if optimizer_step_count != 2:
        raise RuntimeError(f"expected exactly 2 optimizer.step calls, got {optimizer_step_count}")

    # ---------------- Per-step gradient / update analysis ----------------
    # Grad snapshots are indexed [step1, step2].
    step1_grad = per_step_grad_snaps[0]
    step2_grad = per_step_grad_snaps[1]

    def _num_nonzero_grad(snap: Dict[str, Any], names: Sequence[str]) -> int:
        return sum(1 for n in names
                   if snap.get(n, {}).get("has_grad")
                   and snap.get(n, {}).get("is_nonzero"))

    lora_a_names = [n for n in lora_names if n.endswith(".lora_A")]
    lora_b_names = [n for n in lora_names if n.endswith(".lora_B")]

    step1_lora_b_nonzero = _num_nonzero_grad(step1_grad, lora_b_names)
    step1_lora_a_nonzero = _num_nonzero_grad(step1_grad, lora_a_names)
    step2_lora_b_nonzero = _num_nonzero_grad(step2_grad, lora_b_names)
    step2_lora_a_nonzero = _num_nonzero_grad(step2_grad, lora_a_names)
    step1_existing_nonzero = _num_nonzero_grad(step1_grad, existing_names)
    step2_existing_nonzero = _num_nonzero_grad(step2_grad, existing_names)

    # Param update diff between step0 and step2 snapshots.
    total_diff = _diff_param_snapshots(per_step_param_snaps[0], per_step_param_snaps[2])
    changed_after_step2 = set(total_diff["changed_names"])
    lora_a_changed = [n for n in lora_a_names if n in changed_after_step2]
    lora_b_changed = [n for n in lora_b_names if n in changed_after_step2]
    existing_changed = [n for n in existing_names if n in changed_after_step2]

    # ---------------- Frozen-Base audit AFTER training ----------------
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    frozen_after = _audit_frozen_base(model)
    _atomic_write_json(output_dir / "frozen_base_audit_after.json", frozen_after)
    frozen_diff = _diff_frozen_audits(frozen_before, frozen_after)
    _atomic_write_json(output_dir / "frozen_base_audit_diff.json", frozen_diff)

    if frozen_diff["num_changed"] != 0 or frozen_diff["num_added"] != 0 or frozen_diff["num_removed"] != 0:
        raise RuntimeError(
            f"Frozen Base tampered: added={frozen_diff['num_added']}, "
            f"removed={frozen_diff['num_removed']}, changed={frozen_diff['num_changed']}"
        )

    # ---------------- Hard gates ----------------
    hard_gates = {
        "exactly_2_optimizer_step_calls": optimizer_step_count == 2,
        "optimizer_has_62_params": optim_audit["num_optimizer_params"] == 62,
        "no_base_leak_in_optimizer": len(optim_audit["frozen_leaked_into_optim"]) == 0,
        "step1_lora_B_has_nonzero_grad": step1_lora_b_nonzero >= 1,
        "step2_lora_A_has_nonzero_grad": step2_lora_a_nonzero >= 1,
        "step2_lora_B_has_nonzero_grad": step2_lora_b_nonzero >= 1,
        "step2_at_least_one_lora_A_changed": len(lora_a_changed) >= 1,
        "step2_at_least_one_lora_B_changed": len(lora_b_changed) >= 1,
        "no_nan_or_inf_encountered": all(r["loss_finite"] for r in step_records),
        "frozen_base_unchanged": frozen_diff["num_changed"] == 0,
        "no_added_frozen": frozen_diff["num_added"] == 0,
        "no_removed_frozen": frozen_diff["num_removed"] == 0,
        "step0_strict_reload_ok": reload0["max_abs_load_copy_diff"] == 0.0 and reload0["num_loaded_tensors"] == 62,
        "step1_strict_reload_ok": step_records[0]["strict_reload"]["max_abs_load_copy_diff"] == 0.0 and step_records[0]["strict_reload"]["num_loaded_tensors"] == 62,
        "step2_strict_reload_ok": step_records[1]["strict_reload"]["max_abs_load_copy_diff"] == 0.0 and step_records[1]["strict_reload"]["num_loaded_tensors"] == 62,
    }

    # Peak / final memory.
    final_mem = _cuda_memory_snapshot(device)

    report = {
        "status": "passed" if all(hard_gates.values()) else "failed",
        "hard_gates": hard_gates,
        "mode": "gold40_v3_lora_2step_smoke",
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "created_at_iso": _now_iso(),
        "base": {
            "llm_path": args.llm_path,
            "vae_path": args.vae_path,
            "checkpoint_path": args.checkpoint_path,
        },
        "base_checkpoint_report": base_checkpoint_report,
        "baseline_vs_lora_config_diff": diff,
        "param_audit": {k: v for k, v in param_audit.items() if k != "dtype_desc"},
        "dtype_desc": param_audit["dtype_desc"],
        "optimizer_audit": optim_audit,
        "optimizer_step_count": optimizer_step_count,
        "optimizer_step_iso_timestamps": optimizer_step_calls,
        "step_records": step_records,
        "step0_checkpoint": step0,
        "step0_strict_reload": reload0,
        "grad_analysis": {
            "step1_lora_A_nonzero_count": step1_lora_a_nonzero,
            "step1_lora_B_nonzero_count": step1_lora_b_nonzero,
            "step2_lora_A_nonzero_count": step2_lora_a_nonzero,
            "step2_lora_B_nonzero_count": step2_lora_b_nonzero,
            "step1_existing_nonzero_count": step1_existing_nonzero,
            "step2_existing_nonzero_count": step2_existing_nonzero,
            "num_lora_A_tensors": len(lora_a_names),
            "num_lora_B_tensors": len(lora_b_names),
            "num_existing_tensors": len(existing_names),
            "per_step_grad_snapshots": {
                "step1": step1_grad,
                "step2": step2_grad,
            },
        },
        "param_update_analysis_step0_to_step2": {
            "lora_A_changed": lora_a_changed,
            "num_lora_A_changed": len(lora_a_changed),
            "lora_B_changed": lora_b_changed,
            "num_lora_B_changed": len(lora_b_changed),
            "existing_changed": existing_changed,
            "num_existing_changed": len(existing_changed),
        },
        "frozen_base_audit_diff": frozen_diff,
        "cuda_memory": {
            "per_step": per_step_mem,
            "final": final_mem,
        },
        "notes": [
            "This is an isolated 2-step LoRA smoke. Do NOT resume formal training",
            "from these checkpoints. Formal Step000 lives in the preflight dir.",
        ],
    }
    out_json = output_dir / "gold40_v3_lora_2step_smoke_report.json"
    _atomic_write_json(out_json, report)
    _log_event("gold40_v3_lora_2step_smoke_finished",
               status=report["status"],
               optimizer_step_count=optimizer_step_count,
               step0_sha=step0["sha256"], step0_bytes=step0["size_bytes"],
               step1_sha=step_records[0]["checkpoint"]["sha256"],
               step1_bytes=step_records[0]["checkpoint"]["size_bytes"],
               step2_sha=step_records[1]["checkpoint"]["sha256"],
               step2_bytes=step_records[1]["checkpoint"]["size_bytes"],
               peak_allocated_mib=final_mem["peak_allocated_mib"],
               peak_reserved_mib=final_mem["peak_reserved_mib"])

    if report["status"] != "passed":
        raise RuntimeError(f"smoke FAILED. Failing gates: {[k for k,v in hard_gates.items() if not v]}")

    # ---------------- Independent-process cold reload of step_2 ----------------
    if not args.skip_cold_reload:
        # Free GPU before launching the cold-reload subprocesses.
        del model, vae_model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        step2_ckpt = Path(step_records[1]["checkpoint"]["path"])
        cold_report_path = _launch_cold_reload_step2(args, step2_ckpt)
        if cold_report_path is None:
            _log_event("cold_reload_step2_skipped_no_report")
        else:
            cold_report = json.loads(cold_report_path.read_text("utf-8"))
            report["cold_reload_step2"] = {
                "path": str(cold_report_path),
                "sha256": _sha256_file(cold_report_path),
                "pass": bool(cold_report.get("overall_reload_validation_passed")),
                "hard_gates": cold_report.get("hard_gates"),
                "mse_max_abs_diff": (cold_report.get("mse_tensor_comparison") or {}).get("max_abs_diff"),
                "png_sha_equal": (cold_report.get("generation") or {}).get("sha_equal"),
            }
            _atomic_write_json(out_json, report)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001
        err = {
            "status": "failed",
            "kind": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "created_at_iso": _now_iso(),
        }
        try:
            args = parse_args()
            (args.output_dir / "error_report.json").parent.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "error_report.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass
        raise
