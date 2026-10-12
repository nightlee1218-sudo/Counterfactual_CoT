#!/usr/bin/env python3
"""Export a Gold40 AutoMask-v5 trainable checkpoint as a standard full BAGEL model dir.

The Gold40 checkpoints only store the *trainable* tensors (LoRA A/B + the 14
bridge tensors). Standard BAGEL benchmark scripts (GenEval / GEdit / WISE /
ImgEdit) only accept a full model directory containing ``ema.safetensors``.

This script:

  1. rebuilds the exact v5 training model on CPU from ``ema.safetensors``,
  2. strict-reloads the Gold40 trainable checkpoint,
  3. merges every LoRA branch into its frozen base Linear
     (``W <- W + (alpha / r) * B @ A``, accumulated in fp32 then cast to bf16),
  4. unwraps the LoRALinear modules so the parameter names match the stock
     BAGEL state dict,
  5. writes ``ema.safetensors`` containing exactly the same key set as the
     base checkpoint, plus the tokenizer / config files.

The exported directory is a drop-in ``--model-path`` for the stock eval scripts,
so BASE and Gold40 are compared through one identical inference code path.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

import torch.nn as nn

from experiments.counterfactual_cot.full_backbone_lora_region1024_fsdp_v3 import train as v3_runtime
from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.train import DEFAULT_TRAINABLE_REGEX, _read_num_layers
from experiments.counterfactual_cot.training.gold40_overfit import _load_tokenizer
from experiments.counterfactual_cot.training.lora_2step_smoke import (
    LoRALinear,
    _iter_target_linear_names,
    inject_lora,
    load_trainable_strict,
    set_trainable,
)

DEFAULT_MODEL = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_CHECKPOINT = BAGEL_ROOT / "experiments/counterfactual_cot/outputs/gold40_automask_v5_train960/checkpoints/latest.pt"
EXPECTED_LAYERS = 28
# Keep the stock model dir shape: these are all the non-weight files the eval
# scripts read (llm_config.json / vit_config.json / tokenizer / ae.safetensors).
CONFIG_FILES = (
    "config.json",
    "generation_config.json",
    "llm_config.json",
    "vit_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainable-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--base-checkpoint-path", type=Path, default=DEFAULT_MODEL / "ema.safetensors")
    parser.add_argument("--vae-path", type=Path, default=DEFAULT_MODEL / "ae.safetensors")
    parser.add_argument("--output-model-dir", type=Path, required=True)
    parser.add_argument("--link-ae", action="store_true", help="symlink ae.safetensors instead of copying 335MB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for field in ("trainable_checkpoint", "llm_path", "base_checkpoint_path", "vae_path"):
        if not getattr(args, field).exists():
            raise FileNotFoundError(f"missing {field}: {getattr(args, field)}")
    if args.output_model_dir.exists() and any(args.output_model_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output model dir is non-empty: {args.output_model_dir}; use --overwrite")
    return args


def _ema_key_set(path: Path) -> set[str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return set(handle.keys())


def merge_lora_into_base(model: nn.Module, wrappers: dict[str, LoRALinear]) -> dict[str, Any]:
    """Fold every LoRA branch into its frozen base weight and unwrap the module."""
    per_tensor: list[dict[str, Any]] = []
    total_delta_absmax = 0.0
    for full_name, wrapper in wrappers.items():
        base_lin: nn.Linear = wrapper.base
        with torch.no_grad():
            a = wrapper.lora_A.detach().to(torch.float32)
            b = wrapper.lora_B.detach().to(torch.float32)
            delta = (b @ a) * float(wrapper.scaling)
            merged = base_lin.weight.detach().to(torch.float32) + delta
            base_lin.weight.data = merged.to(base_lin.weight.dtype)
        delta_absmax = float(delta.abs().max().item())
        weight_absmax = float(merged.abs().max().item())
        total_delta_absmax = max(total_delta_absmax, delta_absmax)
        per_tensor.append({
            "name": full_name,
            "r": wrapper.r,
            "scaling": wrapper.scaling,
            "delta_absmax": delta_absmax,
            "merged_weight_absmax": weight_absmax,
            "relative_delta": delta_absmax / weight_absmax if weight_absmax else 0.0,
        })
        # unwrap: replace parent's child module with the merged nn.Linear so the
        # exported parameter names are exactly the stock BAGEL names.
        parent_name, _, child_name = full_name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, base_lin)
    per_tensor.sort(key=lambda row: -row["relative_delta"])
    return {
        "num_merged": len(wrappers),
        "delta_absmax_over_all": total_delta_absmax,
        "max_relative_delta": per_tensor[0]["relative_delta"] if per_tensor else 0.0,
        "top_relative_delta": per_tensor[:10],
    }


def main() -> None:
    args = parse_args()
    args.output_model_dir.mkdir(parents=True, exist_ok=True)

    ema_keys = _ema_key_set(args.base_checkpoint_path)
    print(json.dumps({"event": "base_ema_keys", "count": len(ema_keys)}), flush=True)

    tokenizer, special_tokens, added_tokens = _load_tokenizer(str(args.llm_path))
    build_args = argparse.Namespace(llm_path=args.llm_path, checkpoint_path=args.base_checkpoint_path)
    model, base_report = v3_runtime._build_cpu_model(build_args, len(tokenizer), added_tokens)
    layers = _read_num_layers(args.llm_path)
    if layers != EXPECTED_LAYERS:
        raise RuntimeError(f"expected {EXPECTED_LAYERS} decoder layers, found {layers}")

    target_names = _iter_target_linear_names(model, list(range(layers)))
    wrappers = inject_lora(model, target_names, 8, 16, 0.0)
    trainable_report = set_trainable(model, DEFAULT_TRAINABLE_REGEX, wrappers)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable_names:
        raise RuntimeError("no trainable parameters after v5 model reconstruction")
    for _, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(dtype=torch.float32)

    load_report = load_trainable_strict(args.trainable_checkpoint, model)
    if load_report["num_loaded_tensors"] != len(trainable_names) or load_report["max_abs_load_copy_diff"] != 0.0:
        raise RuntimeError(f"strict trainable reload failed: {load_report}")
    print(json.dumps({"event": "trainable_reloaded", **load_report}), flush=True)

    # Snapshot the trainable tensors that are NOT LoRA (bridge tensors such as
    # time_embedder / vae2llm / llm2vae / connector) so we can prove they were
    # exported unchanged into the merged file.
    direct_trainable = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad and not name.endswith((".lora_A", ".lora_B"))
    }

    merge_report = merge_lora_into_base(model, wrappers)
    print(json.dumps({"event": "lora_merged", **{k: v for k, v in merge_report.items() if k != "top_relative_delta"}}), flush=True)

    model.eval()
    named = dict(model.named_parameters())
    named.update(dict(model.named_buffers()))

    missing_in_model = sorted(k for k in ema_keys if k not in named)
    export: dict[str, torch.Tensor] = {}
    for key in sorted(ema_keys):
        tensor = named.get(key)
        if tensor is None:
            continue
        export[key] = tensor.detach().to(torch.bfloat16).contiguous()

    # Audit: every directly-trained tensor must be byte-identical in the export.
    # The stock checkpoint stores every weight in bf16, so the trained fp32
    # tensors are rounded on export exactly like the base weights are. Compare
    # against the bf16-rounded snapshot; anything above 0 here is a real bug.
    mismatched: list[str] = []
    rounding: list[dict[str, Any]] = []
    for name, snapshot in direct_trainable.items():
        if name not in export:
            mismatched.append(f"{name}: absent from ema key set")
            continue
        diff = float((export[name].to(torch.float32) - snapshot.to(torch.bfloat16).to(torch.float32)).abs().max().item())
        rounding.append({"name": name, "fp32_to_bf16_rounding_absmax": diff})
        if diff != 0.0:
            mismatched.append(f"{name}: max_abs_diff={diff}")
    if mismatched:
        raise RuntimeError(f"direct trainable tensors not exported exactly: {mismatched[:8]}")
    rounding.sort(key=lambda row: -row["fp32_to_bf16_rounding_absmax"])

    out_ema = args.output_model_dir / "ema.safetensors"
    save_file(export, str(out_ema), metadata={"format": "pt", "source": "gold40_automask_v5_merged"})
    print(json.dumps({"event": "ema_written", "path": str(out_ema), "num_tensors": len(export)}), flush=True)

    for name in CONFIG_FILES:
        src = args.llm_path / name
        if src.exists():
            shutil.copy2(src, args.output_model_dir / name)
    out_ae = args.output_model_dir / "ae.safetensors"
    if args.link_ae:
        if out_ae.exists() or out_ae.is_symlink():
            out_ae.unlink()
        out_ae.symlink_to(args.vae_path)
    else:
        shutil.copy2(args.vae_path, out_ae)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "output_model_dir": str(args.output_model_dir.resolve()),
        "base_checkpoint": str(args.base_checkpoint_path),
        "trainable_checkpoint": str(args.trainable_checkpoint.resolve()),
        "checkpoint_step": torch.load(args.trainable_checkpoint, map_location="cpu", weights_only=False).get("step"),
        "num_layers": layers,
        "base_load_report": base_report,
        "trainable_reload": load_report,
        "trainable_report": trainable_report,
        "lora_merge": merge_report,
        "exported_tensors": len(export),
        "base_ema_tensors": len(ema_keys),
        "ema_keys_missing_in_model": missing_in_model[:20],
        "direct_trainable_tensors": len(direct_trainable),
        "direct_trainable_export_diff": 0.0,
        "direct_trainable_rounding_top": rounding[:10],
    }
    (args.output_model_dir / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"event": "done", "exported": len(export), "expected": len(ema_keys)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
