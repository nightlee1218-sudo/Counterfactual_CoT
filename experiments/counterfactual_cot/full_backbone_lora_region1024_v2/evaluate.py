#!/usr/bin/env python3
"""Cold-load round-2 checkpoint and render a multi-prompt, multi-seed matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image, ImageDraw

EXPERIMENT_DIR = Path(__file__).resolve().parent
BAGEL_ROOT = EXPERIMENT_DIR.parents[2]
for path in (BAGEL_ROOT, EXPERIMENT_DIR):
    while str(path) in sys.path:
        sys.path.remove(str(path))
sys.path.insert(0, str(BAGEL_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from train import (  # noqa: E402
    CHECKPOINT_FORMAT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    MANIFEST,
    PAIR_ID,
    PROMPT_PROFILE,
    _atomic_json,
    _pick_and_validate_pair,
    _read_num_layers,
    _resolve_pair_image,
    _validate_base_load_report,
    _validate_gpu_capacity,
    _validate_model_files,
)
from experiments.counterfactual_cot.data.prompt_profiles import render_prompt_profile  # noqa: E402
from experiments.counterfactual_cot.data.schema import load_jsonl  # noqa: E402
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

PROFILES = (
    "cot_core_v1",
    "legacy_paired_v1",
    "full_paired_v2",
    "full_without_target_cot_v1",
    "target_prompt_only_v1",
    "delta_only_v2",
)
SEEDS = (1234, 2234, 3234)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate round-2 1024 CoT-core checkpoint")
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
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES), choices=PROFILES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=("bf16",))
    parser.add_argument("--model-device-map", default="single", choices=("single",))
    parser.add_argument("--device-map-max-memory", default="38GiB")
    parser.add_argument("--min-gpu-memory-gib", type=float, default=38.0)
    parser.add_argument("--variant", default="paired_cf_mvp", choices=("paired_cf_mvp",))
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--vit-image-size", type=int, default=980)
    parser.add_argument("--vit-min-image-size", type=int, default=980)
    parser.add_argument("--expected-num-tokens", type=int, default=16384)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=16384)
    parser.add_argument("--max-num-tokens", type=int, default=16384)
    parser.add_argument("--visual-image-size", type=int, default=1024)
    parser.add_argument("--visual-vae-min-image-size", type=int, default=1024)
    parser.add_argument("--visual-vit-image-size", type=int, default=980)
    parser.add_argument("--visual-vit-min-image-size", type=int, default=980)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=25)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    parser.add_argument("--skip-cfg-ablation", action="store_true")
    args = parser.parse_args()
    args.pair_id = PAIR_ID
    if args.resume_checkpoint is None:
        args.resume_checkpoint = args.output_dir / "checkpoints" / "step_000500.pt"
    if args.eval_dir is None:
        args.eval_dir = args.output_dir / "evaluation_prompt_matrix"
    locked = {
        "image_size": (args.image_size, 1024),
        "vit_image_size": (args.vit_image_size, 980),
        "vit_min_image_size": (args.vit_min_image_size, 980),
        "visual_image_size": (args.visual_image_size, 1024),
        "visual_vae_min_image_size": (args.visual_vae_min_image_size, 1024),
        "visual_vit_image_size": (args.visual_vit_image_size, 980),
        "visual_vit_min_image_size": (args.visual_vit_min_image_size, 980),
    }
    mismatches = {key: {"actual": actual, "required": required} for key, (actual, required) in locked.items() if actual != required}
    if mismatches:
        parser.error(f"Round-2 evaluation resolution mismatch: {json.dumps(mismatches)}")
    return args


def _load_trainables(path: Path, model: torch.nn.Module, target_names: List[str], args: argparse.Namespace) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Unsupported checkpoint format: {payload.get('format')}")
    if payload.get("lora_target_names") != target_names:
        raise RuntimeError("Checkpoint LoRA target list does not match all-layer targets")
    config = payload.get("config", {})
    expected = {
        "pair_id": PAIR_ID,
        "manifest": str(args.manifest),
        "llm_path": str(args.llm_path),
        "vae_path": str(args.vae_path),
        "checkpoint_path": str(args.checkpoint_path),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "trainable_name_regex": args.trainable_name_regex,
        "prompt_profile": PROMPT_PROFILE,
        "region_objective": "0.25_old_0.25_new_0.50_preserve",
        "max_latent_size": 64,
        "image_size": 1024,
        "vit_image_size": 980,
        "vit_min_image_size": 980,
        "activation_checkpointing": True,
    }
    mismatches = {key: {"checkpoint": config.get(key), "evaluation": value} for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Checkpoint/evaluation configuration mismatch: {json.dumps(mismatches, default=str)}")
    saved = payload.get("trainable_state_dict", {})
    named = dict(model.named_parameters())
    required = {name for name, parameter in named.items() if parameter.requires_grad}
    if set(saved) != required:
        raise RuntimeError(
            f"Checkpoint key mismatch: missing={sorted(required - set(saved))[:10]}, "
            f"unexpected={sorted(set(saved) - required)[:10]}"
        )
    for name, tensor in saved.items():
        parameter = named[name]
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise RuntimeError(f"Checkpoint shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(parameter.shape)}")
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return {
        "step": int(payload.get("global_step", payload.get("step", 0))),
        "num_loaded_tensors": len(saved),
        "checkpoint_format": payload.get("format"),
    }


def _contact_sheet(image_paths: List[Path], labels: List[str], output: Path) -> None:
    if len(image_paths) != len(labels) or not image_paths:
        raise RuntimeError("Invalid contact-sheet inputs")
    thumb_size = 384
    label_height = 54
    columns = 3
    rows = (len(image_paths) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(zip(image_paths, labels)):
        with Image.open(path) as image:
            tile = image.convert("RGB")
            tile.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_size
        y = (index // columns) * (thumb_size + label_height)
        canvas.paste(tile, (x + (thumb_size - tile.width) // 2, y))
        draw.text((x + 8, y + thumb_size + 6), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def main() -> None:
    args = parse_args()
    _validate_model_files(args)
    if not args.resume_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")
    if args.device != "cuda":
        raise RuntimeError("This experiment requires --device cuda")
    _validate_gpu_capacity(args.min_gpu_memory_gib)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    pair = _pick_and_validate_pair(load_jsonl(args.manifest))
    device = torch.device("cuda")
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

    step = checkpoint_report["step"]
    step_dir = args.eval_dir / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(_save_reference_image(pair, "source", step_dir / "source.png", 1024))
    target_path = Path(_save_reference_image(pair, "target", step_dir / "golden_target.png", 1024))
    model.eval()
    vae_model.eval()
    inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, args)

    rows: List[Dict[str, Any]] = []
    image_paths: List[Path] = [source_path, target_path]
    labels: List[str] = ["source", "golden target"]
    prompt_manifest: Dict[str, str] = {}
    for profile in args.profiles:
        prompt = render_prompt_profile(pair, profile)
        prompt_manifest[profile] = prompt
        for seed in args.seeds:
            output = step_dir / profile / f"seed_{seed}.png"
            _generate_one(inferencer, pair, output, args, seed=seed, prompt=prompt)
            rows.append({"profile": profile, "seed": seed, "cfg_img_scale": args.cfg_img_scale, "path": str(output), "prompt": prompt})
            image_paths.append(output)
            labels.append(f"{profile} / {seed}")

    if not args.skip_cfg_ablation:
        original_cfg = args.cfg_img_scale
        for profile in ("cot_core_v1", "delta_only_v2"):
            prompt = render_prompt_profile(pair, profile)
            for cfg_img in (1.0, 1.5):
                args.cfg_img_scale = cfg_img
                output = step_dir / "cfg_img_ablation" / profile / f"cfgimg_{cfg_img:.1f}_seed_1234.png"
                _generate_one(inferencer, pair, output, args, seed=1234, prompt=prompt)
                rows.append({"profile": profile, "seed": 1234, "cfg_img_scale": cfg_img, "path": str(output), "prompt": prompt})
                image_paths.append(output)
                labels.append(f"{profile} / cfg_img={cfg_img:.1f}")
        args.cfg_img_scale = original_cfg

    _contact_sheet(image_paths, labels, step_dir / "contact_sheet.jpg")
    report = {
        "status": "completed",
        "pair_id": PAIR_ID,
        "step": step,
        "checkpoint": str(args.resume_checkpoint),
        "checkpoint_report": checkpoint_report,
        "base_load_report": base_load_report,
        "training_prompt_profile": PROMPT_PROFILE,
        "source_original": str(_resolve_pair_image(pair, "source")),
        "target_original": str(_resolve_pair_image(pair, "target")),
        "source_visualization": str(source_path),
        "target_visualization": str(target_path),
        "prompt_manifest": prompt_manifest,
        "generations": rows,
        "contact_sheet": str(step_dir / "contact_sheet.jpg"),
    }
    _atomic_json(step_dir / "evaluation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
