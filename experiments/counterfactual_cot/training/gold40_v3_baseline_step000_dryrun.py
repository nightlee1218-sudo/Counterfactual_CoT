"""Gold40 v3 baseline (14 existing trainable tensors) Step000 dry-run.

Symmetric counterpart of ``gold40_lora.py --dry-run`` for the A/B Step000
crosscheck. It:

  * Uses **exactly** the same Base BAGEL, AE, EMA, tokenizer setup, Gold40 v3
    manifest, dataloader construction (same ``args.seed``), and same
    ``_prepare_batch`` helper as gold40_lora dry-run.
  * DOES NOT inject any LoRA. Only the 14 existing trainable tensors matched
    by ``args.trainable_name_regex`` are enabled.
  * Runs exactly ONE ``torch.no_grad`` forward on a real Gold40 v3 batch.
  * Saves ``checkpoints/step_000_trainable.pt`` and strict-reloads it.
  * NEVER constructs an optimizer step -- ``optimizer`` is only built for the
    "no frozen-base leak" audit, and ``.step()`` is NEVER called.
  * NEVER modifies gold40_overfit.py, gold40_lora.py, lora_2step_smoke.py or
    any shared BAGEL / schema code.

This file exists ONLY to support the A/B Step000 crosscheck. It is not a
training entry.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn

from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.training.gold40_lora import (
    BASELINE_CONFIG,
    FIXED_GOLD40_MANIFEST,
    _atomic_write_json,
    _confirm_no_base_leak_in_optimizer,
    _cuda_memory_report,
    _dockerdata_ok,
    _log_event,
    _now_iso,
    _path_size_gib,
    _prepare_output_dir,
    _save_step_trainable,
    _sha256_file,
    _strict_reload_check,
    _validate_lora_manifest,
    _write_git_provenance,
    _write_sequence_plan,
)
from experiments.counterfactual_cot.training.gold40_overfit import (
    DEFAULT_TRAINABLE_REGEX,
    _autocast_for_model,
    _build_gold40_loader,
    _build_model_for_gold40,
    _load_tokenizer,
    _prepare_batch,
)
from experiments.counterfactual_cot.training.smoke_train import _batch_report, _set_seed
from experiments.counterfactual_cot.training.lora_2step_smoke import (
    set_trainable,
    save_trainable_only,
    load_trainable_strict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FIXED_GOLD40_MANIFEST.parent
        / "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-path", type=str, default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
    parser.add_argument("--vae-path", type=str, default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors")
    parser.add_argument("--checkpoint-path", type=str, default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors")
    parser.add_argument("--variant", type=str, default=BASELINE_CONFIG["variant"])
    parser.add_argument("--num-samples", type=int, default=BASELINE_CONFIG["num_samples"])
    parser.add_argument("--seed", type=int, default=BASELINE_CONFIG["seed"])
    parser.add_argument("--generation-seed", type=int, default=BASELINE_CONFIG["generation_seed"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", type=str, default=BASELINE_CONFIG["precision"])
    parser.add_argument("--model-device-map", type=str, default=BASELINE_CONFIG["model_device_map"])
    parser.add_argument("--device-map-max-memory", type=str, default="34GiB")
    parser.add_argument("--trainable-name-regex", type=str, default=DEFAULT_TRAINABLE_REGEX)
    parser.add_argument("--image-size", type=int, default=BASELINE_CONFIG["image_size"])
    parser.add_argument("--vit-image-size", type=int, default=BASELINE_CONFIG["vit_image_size"])
    parser.add_argument("--vit-min-image-size", type=int, default=BASELINE_CONFIG["vit_min_image_size"])
    parser.add_argument("--expected-num-tokens", type=int, default=BASELINE_CONFIG["expected_num_tokens"])
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=BASELINE_CONFIG["max_num_tokens_per_sample"])
    parser.add_argument("--max-num-tokens", type=int, default=BASELINE_CONFIG["max_num_tokens"])
    parser.add_argument("--visual-image-size", type=int, default=BASELINE_CONFIG["visual_image_size"])
    parser.add_argument("--visual-vae-min-image-size", type=int, default=BASELINE_CONFIG["visual_vae_min_image_size"])
    parser.add_argument("--visual-vit-image-size", type=int, default=BASELINE_CONFIG["visual_vit_image_size"])
    parser.add_argument("--visual-vit-min-image-size", type=int, default=BASELINE_CONFIG["visual_vit_min_image_size"])
    parser.add_argument("--cfg-text-scale", type=float, default=BASELINE_CONFIG["cfg_text_scale"])
    parser.add_argument("--cfg-img-scale", type=float, default=BASELINE_CONFIG["cfg_img_scale"])
    parser.add_argument("--cfg-interval-start", type=float, default=BASELINE_CONFIG["cfg_interval_start"])
    parser.add_argument("--cfg-interval-end", type=float, default=BASELINE_CONFIG["cfg_interval_end"])
    parser.add_argument("--timestep-shift", type=float, default=BASELINE_CONFIG["timestep_shift"])
    parser.add_argument("--num-timesteps", type=int, default=BASELINE_CONFIG["num_timesteps"])
    parser.add_argument("--cfg-renorm-min", type=float, default=BASELINE_CONFIG["cfg_renorm_min"])
    parser.add_argument("--cfg-renorm-type", type=str, default=BASELINE_CONFIG["cfg_renorm_type"])
    parser.add_argument("--lr", type=float, default=BASELINE_CONFIG["lr"])
    parser.add_argument("--allow-non-dockerdata", action="store_true")
    parser.add_argument("--require-paired-vit", action="store_true", default=True)
    parser.add_argument("--log-every", type=int, default=BASELINE_CONFIG["log_every"])
    # LoRA-hyperparameter placeholders. Baseline path never injects LoRA, but
    # _make_lora_smoke_meta unconditionally reads these fields to produce the
    # same meta schema as the LoRA path. Setting them to zero clearly signals
    # "no LoRA".
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    return parser.parse_args()


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return torch.device(name)


def main() -> int:
    args = parse_args()

    _validate_lora_manifest(args.manifest)
    _prepare_output_dir(args.output_dir)
    _atomic_write_json(args.output_dir / "run_config_resolved.json", {
        "mode": "baseline_a_step000_dry_run",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "created_at_iso": _now_iso(),
    })
    _write_git_provenance(args.output_dir)
    _write_sequence_plan(args.output_dir, args.manifest)
    _set_seed(args.seed)
    device = _device(args.device)

    records = load_jsonl(args.manifest)
    if len(records) != args.num_samples:
        raise ValueError(
            f"Baseline dry-run expects exactly {args.num_samples} Gold40 records, "
            f"found {len(records)} in {args.manifest}"
        )
    if not all(bool(row.get("human_review", {}).get("accepted")) for row in records):
        raise ValueError("All Gold40 records must be human_review.accepted=true")

    _dockerdata_ok(args.llm_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.vae_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.checkpoint_path, args.allow_non_dockerdata)

    _log_event(
        "gold40_v3_baseline_dry_run_start",
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
    train_loader = _build_gold40_loader(args.manifest, tokenizer, special_tokens, args, seed_offset=0)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, vae_model, base_checkpoint_report = _build_model_for_gold40(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
    )

    # Baseline path: no LoRA injection. Enable ONLY the 14 existing tensors
    # matched by trainable_name_regex; freeze everything else.
    trainable_report = set_trainable(model, args.trainable_name_regex, wrappers={})
    existing_names: List[str] = list(trainable_report["existing_trainable_names"])
    lora_names: List[str] = list(trainable_report["lora_trainable_names"])
    if lora_names:
        raise RuntimeError(
            f"Baseline path unexpectedly reports LoRA trainables: {lora_names[:8]}"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    num_trainable = len(trainable_params)
    total_trainable_params = int(sum(p.numel() for p in trainable_params))
    if num_trainable != 14:
        raise RuntimeError(
            f"Expected 14 baseline trainable tensors, got {num_trainable}: "
            f"{[n for n, p in model.named_parameters() if p.requires_grad][:20]}"
        )

    # Optimizer construction: only for the "no frozen-base leak" audit.
    # optimizer.step is NEVER called on baseline path.
    trainable_ids = {id(p) for p in trainable_params}
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr,
    )
    optim_audit = _confirm_no_base_leak_in_optimizer(optimizer, model, trainable_ids)
    del optimizer

    model.train()
    vae_model.eval()
    model_dtype = next(model.parameters()).dtype

    batch = next(iter(train_loader))
    data = _prepare_batch(batch, vae_model, device, model_dtype)
    first_batch_report = _batch_report(batch, data, device)

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
        raise RuntimeError(f"non-finite mse in baseline dry-run: mean={mse_mean}")

    peak_alloc_mib = (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0) / (1024 * 1024)
    peak_reserved_mib = (torch.cuda.max_memory_reserved() if device.type == "cuda" else 0) / (1024 * 1024)

    del out, mse, data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    step000 = _save_step_trainable(
        0, model, args.output_dir, args,
        target_names=[],           # No LoRA target names on baseline
        existing_names=existing_names,
        lora_names=lora_names,     # will be []
    )
    reload_report = _strict_reload_check(Path(step000["path"]), model)

    report: Dict[str, Any] = {
        "status": "passed",
        "mode": "baseline_a_step000_dry_run",
        "evaluation_type": "diagnostic_monitoring_only",
        "created_at_iso": _now_iso(),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "base": {
            "llm_path": str(args.llm_path),
            "vae_path": str(args.vae_path),
            "checkpoint_path": str(args.checkpoint_path),
            "allow_non_dockerdata": bool(args.allow_non_dockerdata),
        },
        "base_checkpoint_report": base_checkpoint_report,
        "num_trainable_tensors": num_trainable,
        "total_trainable_parameters": total_trainable_params,
        "existing_trainable_names": existing_names,
        "lora_trainable_names": lora_names,
        "optimizer_audit": optim_audit,
        "forward": {
            "mse_mean": mse_mean,
            "mse_shape": output_shape,
            "mse_all_finite": mse_all_finite,
        },
        "first_batch_report": first_batch_report,
        "cuda_memory_peak_mib": {
            "allocated": peak_alloc_mib,
            "reserved": peak_reserved_mib,
        },
        "step000_checkpoint": step000,
        "step000_strict_reload": reload_report,
        "notes": [
            "Baseline A path: NO LoRA injected. Only 14 existing trainable tensors.",
            "This entry exists to support the A/B Step000 crosscheck.",
            "optimizer.step is NEVER called on this path.",
        ],
    }

    _atomic_write_json(args.output_dir / "gold40_v3_baseline_dry_run_report.json", report)
    _log_event(
        "gold40_v3_baseline_dry_run_finished",
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
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
