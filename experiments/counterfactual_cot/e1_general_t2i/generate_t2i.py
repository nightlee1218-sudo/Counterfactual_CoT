#!/usr/bin/env python3
"""E1 general T2I generation: base vs final991 (LoRA direct-load) on benchmark prompts.

Reuses the exact model-building path of the 10.8 in-domain evaluation
(gold40_automask_dual_target_cot1024_fsdp_v5.generate_same_seed_states.build_inference_model),
so base = zero-init LoRA (exact identity) and final991 = LoRA loaded strictly from
outputs/final991_automask_v5_train2973/checkpoints/latest.pt.

Text-to-image call: inferencer.interleave_inference(input_lists=[prompt]) — the
standard BAGEL T2I path. Sampling matches training: 25 steps, CFG text 4.0 /
img 2.0, shift 3.0, 1024px, seed 1234.

Usage (one model per GPU):
  CUDA_VISIBLE_DEVICES=0 python3 generate_t2i.py --model base --output-dir outputs/e1_general_t2i/gen/base
  CUDA_VISIBLE_DEVICES=1 python3 generate_t2i.py --model final991 --output-dir outputs/e1_general_t2i/gen/final991
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BAGEL_ROOT = Path("/group/40016/niightli/BAGEL")
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.gold40_automask_dual_target_cot1024_fsdp_v5.generate_same_seed_states import (  # noqa: E402
    build_inference_model,
)

HERE = Path(__file__).resolve().parent
E1_ROOT = BAGEL_ROOT / "experiments/counterfactual_cot/e1_general_t2i"
DEFAULT_MODEL_DIR = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_CKPT = BAGEL_ROOT / "experiments/counterfactual_cot/outputs/final991_automask_v5_train2973/checkpoints/latest.pt"

MODELS = {
    "base": {"trainable_checkpoint": None, "skip_trainable_reload": True},
    "final991": {"trainable_checkpoint": DEFAULT_CKPT, "skip_trainable_reload": False},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--prompts-dir", type=Path, default=HERE / "prompts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-timesteps", type=int, default=25)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--max-items", type=int, default=0, help="0 = all (smoke runs may limit)")
    parser.add_argument("--subsets", nargs="*", default=None, help="default: all subsets")
    return parser.parse_args()


def set_all_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("E1 generation requires CUDA")

    prompt_files = sorted(args.prompts_dir.glob("*.jsonl"))
    if args.subsets:
        prompt_files = [p for p in prompt_files if p.stem in args.subsets]
    if not prompt_files:
        raise FileNotFoundError(f"no prompt jsonl under {args.prompts_dir}")

    items: list[dict] = []
    for pf in prompt_files:
        for line in pf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(json.loads(line))
    if args.max_items:
        items = items[: args.max_items]
    total = len(items)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "generation_manifest.jsonl"
    done_ids: set[str] = set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["item_id"])
    # resume safety: also treat an existing non-empty image as done
    image_root = args.output_dir / "images"

    m = MODELS[args.model]
    ns = argparse.Namespace(
        llm_path=args.llm_path,
        vae_path=args.llm_path / "ae.safetensors",
        base_checkpoint_path=args.llm_path / "ema.safetensors",
        trainable_checkpoint=m["trainable_checkpoint"],
        skip_trainable_reload=m["skip_trainable_reload"],
        num_timesteps=args.num_timesteps,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval_start=0.0,
        cfg_interval_end=1.0,
        timestep_shift=args.timestep_shift,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )
    t0 = time.time()
    print(json.dumps({"event": "model_load_start", "model": args.model, "total": total,
                      "resumed_done": len(done_ids), "time": datetime.now(timezone.utc).isoformat()}), flush=True)
    model, vae, inferencer, infer_args, audit = build_inference_model(ns)
    print(json.dumps({"event": "model_load_done", "model": args.model, "audit": {
        "lora_wrappers": audit["lora_wrappers"], "trainable_reload": audit["trainable_reload"],
        "num_layers": audit["num_layers"]}}), flush=True)

    ckpt_step = None
    if m["trainable_checkpoint"] is not None:
        payload = torch.load(m["trainable_checkpoint"], map_location="cpu", weights_only=False)
        ckpt_step = payload.get("step")
        ckpt_sha = sha256_file(m["trainable_checkpoint"])
        ckpt_path_str = str(m["trainable_checkpoint"].resolve())
    else:
        ckpt_sha, ckpt_path_str = None, None

    completed = 0
    with torch.no_grad():
        for idx, item in enumerate(items):
            item_id = item["item_id"]
            out_png = image_root / item["subset"] / f"{item_id}__seed{args.seed}.png"
            if item_id in done_ids and out_png.exists() and out_png.stat().st_size > 0:
                completed += 1
                continue
            out_png.parent.mkdir(parents=True, exist_ok=True)
            set_all_seeds(args.seed)
            result = inferencer.interleave_inference(
                input_lists=[item["prompt"]],
                think=False,
                understanding_output=False,
                cfg_text_scale=args.cfg_text_scale,
                cfg_img_scale=args.cfg_img_scale,
                cfg_interval=[infer_args.cfg_interval_start, infer_args.cfg_interval_end],
                timestep_shift=args.timestep_shift,
                num_timesteps=args.num_timesteps,
                cfg_renorm_min=infer_args.cfg_renorm_min,
                cfg_renorm_type=infer_args.cfg_renorm_type,
                image_shapes=(args.image_size, args.image_size),
            )
            images = [x for x in result if isinstance(x, Image.Image)]
            if not images:
                raise RuntimeError(f"no image generated for {item_id}")
            images[-1].save(str(out_png))
            if out_png.stat().st_size == 0:
                raise RuntimeError(f"empty image for {item_id}")
            row = {
                "schema_version": "e1_general_t2i_generation_v1",
                "item_id": item_id,
                "subset": item["subset"],
                "source_category": item["source_category"],
                "prompt": item["prompt"],
                "model": args.model,
                "seed": args.seed,
                "output_path": str(out_png.resolve()),
                "output_sha256": sha256_file(out_png),
                "checkpoint_path": ckpt_path_str,
                "checkpoint_sha256": ckpt_sha,
                "checkpoint_step": ckpt_step,
                "generation": {
                    "num_timesteps": args.num_timesteps,
                    "cfg_text_scale": args.cfg_text_scale,
                    "cfg_img_scale": args.cfg_img_scale,
                    "timestep_shift": args.timestep_shift,
                    "visual_image_size": args.image_size,
                },
            }
            with manifest_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            completed += 1
            elapsed = time.time() - t0
            done_now = completed
            print(json.dumps({"event": "generated", "model": args.model, "item_id": item_id,
                              "done": done_now, "total": total,
                              "sec_per_img": round(elapsed / max(done_now, 1), 2),
                              "eta_min": round((total - done_now) * elapsed / max(done_now, 1) / 60, 1)}),
                  flush=True)

    (args.output_dir / "generation_report.json").write_text(
        json.dumps({
            "status": "completed",
            "model": args.model,
            "seed": args.seed,
            "expected": total,
            "generated_or_skipped": completed,
            "manifest": str(manifest_path.resolve()),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
    print(json.dumps({"event": "all_done", "model": args.model, "total": total}), flush=True)


if __name__ == "__main__":
    main()
