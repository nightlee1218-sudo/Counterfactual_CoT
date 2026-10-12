#!/usr/bin/env python3
"""Unified mini-benchmark generator for BASE vs Gold40-V5.

Contract: BASE and Gold40 run through *exactly* the same code path.
The only difference is whether the trainable checkpoint (LoRA A/B + bridge
tensors) is strict-loaded on top of the identical base ``ema.safetensors``.

  BASE   : base ema.safetensors, LoRA left at zero-init (lora_B == 0)
  GOLD40 : same base ema.safetensors + latest.pt trainable_state_dict

Everything else -- model graph, tokenizer, VAE, prompt rendering, seed,
num_timesteps, cfg scales, timestep shift -- is shared.

Manifest item schema (jsonl, one per line):
  {"id": str, "task": "t2i"|"edit", "prompt": str,
   "image_path": str (edit only), "meta": {...}}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.full_backbone_lora_dual_target_cot1024_fsdp_v4 import train as v4_runtime
from experiments.counterfactual_cot.full_backbone_lora_region1024_fsdp_v3 import train as v3_runtime
from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.train import DEFAULT_TRAINABLE_REGEX, _read_num_layers
from experiments.counterfactual_cot.training.gold40_overfit import _load_tokenizer, _make_inferencer
from experiments.counterfactual_cot.training.lora_2step_smoke import (
    _iter_target_linear_names,
    inject_lora,
    load_trainable_strict,
    set_trainable,
)
from modeling.autoencoder import load_ae

DEFAULT_MODEL = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_TRAINABLE = BAGEL_ROOT / "experiments/counterfactual_cot/outputs/gold40_automask_v5_train960/checkpoints/latest.pt"
EXPECTED_LAYERS = 28


def fit_shape(size: tuple[int, int], target_side: int) -> tuple[int, int]:
    """Largest (h, w) multiple of 32 that keeps the aspect ratio and fits target_side^2."""
    width, height = size
    scale = target_side / max(width, height)
    new_w = max(32, int(round(width * scale / 32.0)) * 32)
    new_h = max(32, int(round(height * scale / 32.0)) * 32)
    while new_w * new_h > target_side * target_side:
        if new_w >= new_h:
            new_w -= 32
        else:
            new_h -= 32
    return (new_h, new_w)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("base", "gold40", "gold40_lora0", "final991"),
        required=True,
        help="gold40_lora0 = load latest.pt then zero lora_B (isolates the non-LoRA trainables)",
    )
    parser.add_argument(
        "--keep-aspect",
        action="store_true",
        help="edit only: derive image_shapes from the source image (long side 1024, /32) "
        "instead of forcing a 1024x1024 square",
    )
    parser.add_argument("--trainable-checkpoint", type=Path, default=DEFAULT_TRAINABLE)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vae-path", type=Path, default=DEFAULT_MODEL / "ae.safetensors")
    parser.add_argument("--base-checkpoint-path", type=Path, default=DEFAULT_MODEL / "ema.safetensors")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-timesteps", type=int, default=25)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    parser.add_argument("--target-side", type=int, default=1024, help="long side used by --keep-aspect")
    parser.add_argument("--think", action="store_true", help="WISE-style think-then-generate")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    tokenizer, special_tokens, added_tokens = _load_tokenizer(str(args.llm_path))
    build_args = argparse.Namespace(llm_path=args.llm_path, checkpoint_path=args.base_checkpoint_path)
    model, base_report = v3_runtime._build_cpu_model(build_args, len(tokenizer), added_tokens)
    layers = _read_num_layers(args.llm_path)
    if layers != EXPECTED_LAYERS:
        raise RuntimeError(f"expected {EXPECTED_LAYERS} layers, found {layers}")
    model.timestep_shift = args.timestep_shift
    model.config.timestep_shift = args.timestep_shift

    # Identical graph for both variants: LoRA is always injected.
    target_names = _iter_target_linear_names(model, list(range(layers)))
    wrappers = inject_lora(model, target_names, 8, 16, 0.0)
    trainable_report = set_trainable(model, DEFAULT_TRAINABLE_REGEX, wrappers)
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters")
    for _, p in trainable:
        p.data = p.data.to(dtype=torch.float32)

    model.to(device=torch.device("cuda"))
    hooks, bridge_report = v4_runtime._install_bridge_output_cast_hooks(model)

    if args.variant in ("gold40", "gold40_lora0", "final991"):
        load_report = load_trainable_strict(args.trainable_checkpoint, model)
        if load_report["num_loaded_tensors"] != len(trainable) or load_report["max_abs_load_copy_diff"] != 0.0:
            raise RuntimeError(f"strict trainable reload failed: {load_report}")
        if args.variant == "gold40_lora0":
            for wrapper in wrappers.values():
                with torch.no_grad():
                    wrapper.lora_B.data.zero_()
            load_report["lora_b_zeroed"] = True
        payload = torch.load(args.trainable_checkpoint, map_location="cpu", weights_only=False)
        ckpt_step = payload.get("step")
        ckpt_sha = sha256_file(args.trainable_checkpoint)
        ckpt_path = str(args.trainable_checkpoint.resolve())
    else:
        if not all(bool((w.lora_B == 0).all()) for w in wrappers.values()):
            raise RuntimeError("BASE requires zero-init lora_B")
        load_report = {"mode": "base_skip_trainable_reload", "num_trainable_tensors": len(trainable)}
        ckpt_step, ckpt_sha, ckpt_path = 0, None, None

    model.eval()
    vae, _ = load_ae(str(args.vae_path))
    vae.to(device=torch.device("cuda"), dtype=torch.bfloat16).eval()
    infer_args = argparse.Namespace(
        visual_image_size=1024,
        visual_vae_min_image_size=512,
        visual_vit_image_size=980,
        visual_vit_min_image_size=224,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval_start=args.cfg_interval_start,
        cfg_interval_end=args.cfg_interval_end,
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
    )
    inferencer = _make_inferencer(model, vae, tokenizer, special_tokens, infer_args)
    audit = {
        "variant": args.variant,
        "base_load_report": base_report,
        "trainable_reload": load_report,
        "trainable_report": trainable_report,
        "bridge_output_cast": bridge_report,
        "num_layers": layers,
        "checkpoint": {"path": ckpt_path, "sha256": ckpt_sha, "step": ckpt_step},
        "generation": {
            "seed": args.seed,
            "num_timesteps": args.num_timesteps,
            "cfg_text_scale": args.cfg_text_scale,
            "cfg_img_scale": args.cfg_img_scale,
            "timestep_shift": args.timestep_shift,
            "cfg_interval": [args.cfg_interval_start, args.cfg_interval_end],
            "think": args.think,
            "keep_aspect": args.keep_aspect,
            "target_side": args.target_side,
        },
    }
    return model, vae, inferencer, infer_args, audit


def main() -> None:
    args = parse_args()
    if not args.manifest.exists():
        raise FileNotFoundError(args.manifest)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"non-empty output dir: {args.output_dir}; use --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        items = items[: args.limit]

    model, vae, inferencer, infer_args, audit = build_runtime(args)
    rows: list[dict[str, Any]] = []
    manifest_path = args.output_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()
    try:
        for index, item in enumerate(items):
            item_id = str(item["id"])
            prompt = str(item["prompt"])
            out_path = args.output_dir / "images" / f"{item_id}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)

            if item["task"] == "edit":
                src = Image.open(item["image_path"]).convert("RGB")
                inputs: list[Any] = [src, prompt]
                shape = fit_shape(src.size, args.target_side) if args.keep_aspect else (1024, 1024)
            else:
                inputs = [prompt]
                shape = (1024, 1024)

            with torch.no_grad():
                out = inferencer.interleave_inference(
                    inputs,
                    think=args.think,
                    cfg_text_scale=args.cfg_text_scale,
                    cfg_img_scale=args.cfg_img_scale,
                    cfg_interval=[args.cfg_interval_start, args.cfg_interval_end],
                    timestep_shift=args.timestep_shift,
                    num_timesteps=args.num_timesteps,
                    cfg_renorm_min=args.cfg_renorm_min,
                    cfg_renorm_type=args.cfg_renorm_type,
                    image_shapes=shape,
                )
            image = out[-1]
            if isinstance(image, str):  # think mode returns text then image
                image = out[-1]
            if not isinstance(image, Image.Image):
                raise RuntimeError(f"unexpected output type for {item_id}: {type(image)}")
            image.save(out_path)
            if not out_path.is_file() or out_path.stat().st_size == 0:
                raise RuntimeError(f"generation failed: {item_id}")

            row = {
                "id": item_id,
                "task": item["task"],
                "prompt": prompt,
                "image_path": item.get("image_path"),
                "meta": item.get("meta", {}),
                "variant": args.variant,
                "output_path": str(out_path.resolve()),
                "output_sha256": sha256_file(out_path),
                "seed": args.seed,
                "image_shapes": list(shape),
            }
            rows.append(row)
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            print(json.dumps({"event": "generated", "index": index, "id": item_id, "variant": args.variant}), flush=True)
    finally:
        del inferencer, vae, model
        torch.cuda.empty_cache()

    report = {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "manifest": str(args.manifest.resolve()),
        "expected": len(items),
        "generated": len(rows),
        **audit,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "done", "variant": args.variant, "generated": len(rows), "expected": len(items)}), flush=True)


if __name__ == "__main__":
    main()
