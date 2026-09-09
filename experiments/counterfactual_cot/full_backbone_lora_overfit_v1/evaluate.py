#!/usr/bin/env python3
"""Cold-load one training checkpoint and render the locked pair at a fixed seed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
BAGEL_ROOT = EXPERIMENT_DIR.parents[2]
for path in (BAGEL_ROOT, EXPERIMENT_DIR):
    while str(path) in sys.path:
        sys.path.remove(str(path))
sys.path.insert(0, str(BAGEL_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from train import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    MANIFEST,
    PAIR_ID,
    _atomic_json,
    _fixed_mse,
    _pick_and_validate_pair,
    _read_num_layers,
    _resolve_pair_image,
    _validate_base_load_report,
    _validate_gpu_capacity,
    _validate_model_files,
)
from experiments.counterfactual_cot.data.schema import load_jsonl  # noqa: E402
from experiments.counterfactual_cot.training.fixed_deterministic_eval import (  # noqa: E402
    _build_raw_sample_dataset,
    _build_singleton_packed_dataset,
    _stable_pair_seed,
)
from experiments.counterfactual_cot.training.gold40_overfit import (  # noqa: E402
    DEFAULT_TRAINABLE_REGEX,
    _build_model_for_gold40,
    _generate_one,
    _load_tokenizer,
    _make_inferencer,
    _save_reference_image,
)
from experiments.counterfactual_cot.training.lora_2step_smoke import (  # noqa: E402
    TARGET_MODULE_SUFFIXES,
    _iter_target_linear_names,
    inject_lora,
    set_trainable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full-backbone LoRA overfit checkpoint")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vae-path", type=Path, default=DEFAULT_MODEL / "ae.safetensors")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_MODEL / "ema.safetensors")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--trainable-name-regex", default=DEFAULT_TRAINABLE_REGEX)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=("bf16",))
    parser.add_argument("--model-device-map", default="single", choices=("single",))
    parser.add_argument("--device-map-max-memory", default="38GiB")
    parser.add_argument("--min-gpu-memory-gib", type=float, default=38.0)
    parser.add_argument("--training-activation-checkpointing", action="store_true")
    parser.add_argument("--variant", default="paired_cf_mvp", choices=("paired_cf_mvp",))
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--vit-min-image-size", type=int, default=224)
    parser.add_argument("--expected-num-tokens", type=int, default=1536)
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
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    args = parser.parse_args()
    args.pair_id = PAIR_ID
    if args.resume_checkpoint is None:
        args.resume_checkpoint = args.output_dir / "checkpoints" / "latest.pt"
    if args.eval_dir is None:
        args.eval_dir = args.output_dir / "evaluation"
    return args


def _load_trainables(path: Path, model: torch.nn.Module, target_names: list[str], args: argparse.Namespace) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "full_backbone_lora_overfit_resume_v1":
        raise RuntimeError(f"Unsupported checkpoint format: {payload.get('format')}")
    if payload.get("lora_target_names") != target_names:
        raise RuntimeError("Checkpoint LoRA target list does not match full-backbone targets")
    config = payload.get("config", {})
    expected_config = {
        "pair_id": PAIR_ID,
        "manifest": str(args.manifest),
        "llm_path": str(args.llm_path),
        "vae_path": str(args.vae_path),
        "checkpoint_path": str(args.checkpoint_path),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "trainable_name_regex": args.trainable_name_regex,
        "seed": args.seed,
        "variant": args.variant,
        "precision": args.precision,
        "model_device_map": args.model_device_map,
        "image_size": args.image_size,
        "vit_image_size": args.vit_image_size,
        "vit_min_image_size": args.vit_min_image_size,
        "expected_num_tokens": args.expected_num_tokens,
        "max_num_tokens_per_sample": args.max_num_tokens_per_sample,
        "max_num_tokens": args.max_num_tokens,
        "activation_checkpointing": args.training_activation_checkpointing,
    }
    mismatches = {
        key: {"checkpoint": config.get(key), "evaluation": value}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Checkpoint/evaluation configuration mismatch: {json.dumps(mismatches, default=str)}")
    saved = payload["trainable_state_dict"]
    named = dict(model.named_parameters())
    expected = {name for name, parameter in named.items() if parameter.requires_grad}
    if set(saved) != expected:
        raise RuntimeError(
            f"Checkpoint key mismatch: missing={sorted(expected - set(saved))[:10]}, "
            f"unexpected={sorted(set(saved) - expected)[:10]}"
        )
    for name, tensor in saved.items():
        parameter = named[name]
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise RuntimeError(f"Checkpoint shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(parameter.shape)}")
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return {"step": int(payload.get("global_step", payload.get("step", 0))), "num_loaded_tensors": len(saved)}


def main() -> None:
    args = parse_args()
    _validate_model_files(args)
    if not args.resume_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    pair = _pick_and_validate_pair(load_jsonl(args.manifest))
    if args.device != "cuda":
        raise RuntimeError("This experiment requires --device cuda")
    _validate_gpu_capacity(args.min_gpu_memory_gib)
    device = torch.device(args.device)
    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(str(args.llm_path))
    model, vae_model, base_load_report = _build_model_for_gold40(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens
    )
    _validate_base_load_report(base_load_report)
    num_layers = _read_num_layers(args.llm_path)
    target_names = _iter_target_linear_names(model, list(range(num_layers)))
    expected_count = num_layers * len(TARGET_MODULE_SUFFIXES)
    if len(target_names) != expected_count:
        raise RuntimeError(f"Expected {expected_count} LoRA targets, found {len(target_names)}")
    wrappers = inject_lora(model, target_names, args.lora_r, args.lora_alpha, args.lora_dropout)
    set_trainable(model, args.trainable_name_regex, wrappers)
    checkpoint_report = _load_trainables(args.resume_checkpoint, model, target_names, args)

    raw_dataset = _build_raw_sample_dataset(args, tokenizer)
    packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)
    pair_seed = _stable_pair_seed(PAIR_ID, args.seed)
    fixed_mse, trace = _fixed_mse(model, vae_model, packer, raw_dataset, pair, pair_seed + 1_000_000, device)

    step = checkpoint_report["step"]
    step_dir = args.eval_dir / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    source_path = _save_reference_image(pair, "source", step_dir / "source.png", args.visual_image_size)
    target_path = _save_reference_image(pair, "target", step_dir / "golden_target.png", args.visual_image_size)
    model.eval()
    vae_model.eval()
    inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, args)
    output_path = step_dir / "model_output.png"
    _generate_one(inferencer, pair, output_path, args, seed=args.generation_seed)

    report = {
        "status": "completed",
        "pair_id": PAIR_ID,
        "step": step,
        "checkpoint": str(args.resume_checkpoint),
        "checkpoint_report": checkpoint_report,
        "base_load_report": base_load_report,
        "fixed_mse": fixed_mse,
        "fixed_timesteps_hash": trace.get("packed_timesteps_hash"),
        "source_original": str(_resolve_pair_image(pair, "source")),
        "target_original": str(_resolve_pair_image(pair, "target")),
        "source_visualization": source_path,
        "target_visualization": target_path,
        "model_output": str(output_path),
        "generation_seed": args.generation_seed,
    }
    _atomic_json(step_dir / "evaluation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
