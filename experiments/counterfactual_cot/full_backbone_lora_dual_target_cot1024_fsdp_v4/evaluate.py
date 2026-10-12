#!/usr/bin/env python3
"""Cold-load a v4 checkpoint and test same-seed left/right Target-CoT switching."""

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

from dual_target_data import BRANCHES, PROMPT_PROFILE, prompt_manifest, render_dual_target_prompt  # noqa: E402
from region_objective import OBJECTIVE_ID  # noqa: E402
from train import (  # noqa: E402
    CHECKPOINT_FORMAT,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT,
    MANIFEST,
    PAIR_ID,
    _atomic_json,
    _pick_and_validate_pair,
    _read_num_layers,
    _resolve_pair_image,
)
from experiments.counterfactual_cot.data.schema import load_jsonl  # noqa: E402
from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.train import (  # noqa: E402
    _validate_base_load_report,
    _validate_model_files,
)
from experiments.counterfactual_cot.full_backbone_lora_region1024_fsdp_v3.train import _validate_gpu_capacity  # noqa: E402
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

SEEDS = tuple(1234 + 1000 * index for index in range(16))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()
    args.pair_id = PAIR_ID
    if args.resume_checkpoint is None:
        args.resume_checkpoint = args.output_dir / "checkpoints/step_000500.pt"
    if args.eval_dir is None:
        args.eval_dir = args.output_dir / "evaluation_dual_target_cot_switch"
    locked = {
        "image_size": 1024,
        "vit_image_size": 980,
        "vit_min_image_size": 980,
        "visual_image_size": 1024,
        "visual_vae_min_image_size": 1024,
        "visual_vit_image_size": 980,
        "visual_vit_min_image_size": 980,
        "lora_dropout": 0.0,
    }
    mismatches = {key: {"actual": getattr(args, key), "required": value} for key, value in locked.items() if getattr(args, key) != value}
    if mismatches:
        parser.error(f"v4 evaluation configuration mismatch: {json.dumps(mismatches)}")
    if tuple(args.seeds) != SEEDS:
        parser.error(f"Strict v4 evaluation requires the locked 16 seeds: {SEEDS}")
    return args


def _load_trainables(
    path: Path,
    model: torch.nn.Module,
    target_names: List[str],
    args: argparse.Namespace,
    prompts: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("experiment_name") != "cf_singlepair_fullL0-27_lora_r8_dualTargetCoT1024_fsdpFullShard_overfit_v4":
        raise RuntimeError(
            f"Unsupported checkpoint identity: experiment={payload.get('experiment_name')}, format={payload.get('format')}"
        )
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
        "prompt_sha256": {branch: prompts[branch]["sha256"] for branch in BRANCHES},
        "paired_targets": {"left": "source_image", "right": "golden_target_image"},
        "paired_macro_step": True,
        "matched_noise_and_timestep": True,
        "branch_loss_scale": 0.5,
        "timestep_shift": 3.0,
        "timestep_mixture": {"native": 0.50, "high": 0.40, "pure_noise": 0.10},
        "region_objective": OBJECTIVE_ID,
        "max_latent_size": 64,
        "image_size": 1024,
        "vit_image_size": 980,
        "vit_min_image_size": 980,
        "activation_checkpointing": True,
        "fsdp_sharding": "FULL_SHARD",
        "fsdp_world_size": 2,
        "fsdp_use_orig_params": True,
        "trainable_parameter_dtype": "float32",
        "optimizer_state_dtype": "float32",
        "lora_dtype": "float32",
    }
    mismatches = {
        key: {"checkpoint": config.get(key), "evaluation": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
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
            raise RuntimeError(f"Checkpoint shape mismatch for {name}")
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    return {
        "step": int(payload.get("global_step", payload.get("step", 0))),
        "num_loaded_tensors": len(saved),
        "checkpoint_format": payload.get("format"),
        "experiment_name": payload.get("experiment_name"),
    }


def _contact_sheet(rows: List[Dict[str, Any]], source: Path, target: Path, output: Path) -> None:
    size, label_height, columns = 320, 58, 4
    canvas = Image.new("RGB", (columns * size, len(rows) * (size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        y = index * (size + label_height)
        entries = (
            (source, "SOURCE / LEFT REFERENCE"),
            (target, "GOLD / RIGHT REFERENCE"),
            (Path(row["left_path"]), "TARGET COT = LEFT"),
            (Path(row["right_path"]), "TARGET COT = RIGHT"),
        )
        for column, (path, label) in enumerate(entries):
            with Image.open(path) as image:
                tile = image.convert("RGB")
                tile.thumbnail((size, size), Image.Resampling.LANCZOS)
            x = column * size
            canvas.paste(tile, (x + (size - tile.width) // 2, y))
            draw.text((x + 6, y + size + 5), label, fill="black")
        draw.text((6, y + size + 29), f"same initial-noise seed={row['seed']}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def main() -> None:
    args = _parse_args()
    _validate_model_files(args)
    if not args.resume_checkpoint.is_file():
        raise FileNotFoundError(args.resume_checkpoint)
    if args.device != "cuda":
        raise RuntimeError("This experiment requires --device cuda")
    _validate_gpu_capacity(args.min_gpu_memory_gib)
    pair = _pick_and_validate_pair(load_jsonl(args.manifest))
    prompts = prompt_manifest(pair)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
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
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=torch.float32)
    non_fp32 = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.dtype != torch.float32]
    if non_fp32:
        raise RuntimeError(f"Evaluation requires FP32 trainables to match v4 checkpoint: {non_fp32[:10]}")
    checkpoint_report = _load_trainables(args.resume_checkpoint, model, target_names, args, prompts)
    step = checkpoint_report["step"]
    step_dir = args.eval_dir / f"step_{step:06d}"
    if step_dir.exists() and any(step_dir.iterdir()):
        raise RuntimeError(f"Evaluation directory is not empty: {step_dir}")
    step_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(_save_reference_image(pair, "source", step_dir / "source.png", 1024))
    target_path = Path(_save_reference_image(pair, "target", step_dir / "golden_target.png", 1024))
    model.eval()
    vae_model.eval()
    inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, args)
    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        row: Dict[str, Any] = {"seed": seed}
        for branch in BRANCHES:
            prompt = render_dual_target_prompt(pair, branch)
            output = step_dir / branch / f"seed_{seed}.png"
            _generate_one(inferencer, pair, output, args, seed=seed, prompt=prompt)
            row[f"{branch}_path"] = str(output)
            row[f"{branch}_prompt"] = prompt
        rows.append(row)
    contact_sheet = step_dir / "paired_switch_contact_sheet.jpg"
    _contact_sheet(rows, source_path, target_path, contact_sheet)
    report = {
        "status": "completed",
        "pair_id": PAIR_ID,
        "step": step,
        "checkpoint": str(args.resume_checkpoint),
        "checkpoint_report": checkpoint_report,
        "base_load_report": base_load_report,
        "training_prompt_profile": PROMPT_PROFILE,
        "prompt_manifest": prompts,
        "source_original": str(_resolve_pair_image(pair, "source")),
        "target_original": str(_resolve_pair_image(pair, "target")),
        "source_visualization": str(source_path),
        "target_visualization": str(target_path),
        "seeds": args.seeds,
        "generation_parameters": {
            "visual_image_size": args.visual_image_size,
            "visual_vae_min_image_size": args.visual_vae_min_image_size,
            "visual_vit_image_size": args.visual_vit_image_size,
            "visual_vit_min_image_size": args.visual_vit_min_image_size,
            "cfg_text_scale": args.cfg_text_scale,
            "cfg_img_scale": args.cfg_img_scale,
            "cfg_interval": [args.cfg_interval_start, args.cfg_interval_end],
            "timestep_shift": args.timestep_shift,
            "num_timesteps": args.num_timesteps,
            "cfg_renorm_min": args.cfg_renorm_min,
            "cfg_renorm_type": args.cfg_renorm_type,
        },
        "pairs": rows,
        "contact_sheet": str(contact_sheet),
    }
    _atomic_json(step_dir / "evaluation_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
