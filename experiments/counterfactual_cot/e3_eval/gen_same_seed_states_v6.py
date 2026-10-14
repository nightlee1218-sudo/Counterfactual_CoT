#!/usr/bin/env python3
"""Generate same-seed source/target-state images from a Gold40 AutoMask-v5 checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.full_backbone_lora_dual_target_cot1024_fsdp_v4 import train as v4_runtime
from experiments.counterfactual_cot.full_backbone_lora_region1024_fsdp_v3 import train as v3_runtime
from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.train import DEFAULT_TRAINABLE_REGEX, _read_num_layers
from experiments.counterfactual_cot.gold40_automask_dual_target_cot1024_fsdp_v5.paired_data import (
    STATE_IDS,
    load_automask_manifest,
    prompt_audit,
    render_state_prompt,
)
from experiments.counterfactual_cot.training.gold40_overfit import _generate_one, _load_tokenizer, _make_inferencer
from experiments.counterfactual_cot.training.lora_2step_smoke import _iter_target_linear_names, inject_lora, load_trainable_strict, set_trainable
from modeling.autoencoder import load_ae

DEFAULT_MODEL = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_CHECKPOINT = BAGEL_ROOT / "experiments/counterfactual_cot/outputs/gold40_automask_v5_train960/checkpoints/latest.pt"
DEFAULT_MANIFEST = BAGEL_ROOT / "experiments/counterfactual_cot/gold40_automask_v5_data/artifacts/splits/val.jsonl"
DEFAULT_OUTPUT = BAGEL_ROOT / "experiments/counterfactual_cot/outputs/gold40_automask_v5_val_same_seed_generation"
EXPECTED_LAYERS = 28


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--trainable-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vae-path", type=Path, default=DEFAULT_MODEL / "ae.safetensors")
    parser.add_argument("--base-checkpoint-path", type=Path, default=DEFAULT_MODEL / "ema.safetensors")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1234, 1250)))
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    parser.add_argument(
        "--skip-trainable-reload",
        action="store_true",
        help="BASE/STEP0 control: load only base ema.safetensors and keep LoRA at zero-init "
        "(lora_B == 0, so the LoRA path contributes exactly 0). No trainable checkpoint is required.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must contain at least one unique integer")
    if args.num_timesteps < 1:
        parser.error("--num-timesteps must be positive")
    return args


def build_inference_model(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, int], dict[str, Any]]:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("AutoMask-v5 generation requires CUDA")
    tokenizer, special_tokens, added_tokens = _load_tokenizer(str(args.llm_path))
    build_args = argparse.Namespace(
        llm_path=args.llm_path,
        checkpoint_path=args.base_checkpoint_path,
    )
    model, base_report = v3_runtime._build_cpu_model(build_args, len(tokenizer), added_tokens)
    layers = _read_num_layers(args.llm_path)
    if layers != EXPECTED_LAYERS:
        raise RuntimeError(f"expected {EXPECTED_LAYERS} decoder layers, found {layers}")
    model.timestep_shift = args.timestep_shift
    model.config.timestep_shift = args.timestep_shift
    target_names = _iter_target_linear_names(model, list(range(layers)))
    wrappers = inject_lora(model, target_names, 8, 16, 0.0)
    trainable_report = set_trainable(model, DEFAULT_TRAINABLE_REGEX, wrappers)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable LoRA/bridge parameters after v5 model reconstruction")
    for _, parameter in trainable:
        parameter.data = parameter.data.to(dtype=torch.float32)
    model.to(device=device)
    hooks, bridge_report = v4_runtime._install_bridge_output_cast_hooks(model)
    if args.skip_trainable_reload:
        zero_b = all(bool((wrapper.lora_B == 0).all()) for wrapper in wrappers.values())
        if not zero_b:
            raise RuntimeError("BASE mode requires every lora_B to be zero-initialized")
        load_report = {
            "mode": "base_skip_trainable_reload",
            "num_trainable_tensors": len(trainable),
            "lora_all_zero": zero_b,
        }
    else:
        load_report = load_trainable_strict(args.trainable_checkpoint, model)
        expected = len(trainable)
        if load_report["num_loaded_tensors"] != expected or load_report["max_abs_load_copy_diff"] != 0.0:
            raise RuntimeError(f"strict trainable reload failed: {load_report}")
    model.eval()
    vae, _ = load_ae(str(args.vae_path))
    vae.to(device=device, dtype=torch.bfloat16).eval()
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
        "base_load_report": base_report,
        "trainable_reload": load_report,
        "num_layers": layers,
        "lora_wrappers": len(wrappers),
        "trainable_report": trainable_report,
        "bridge_output_cast": bridge_report,
        "hook_count": len(hooks),
    }
    return model, vae, inferencer, infer_args, audit


def main() -> None:
    args = parse_args()
    required_fields = ["manifest", "llm_path", "vae_path", "base_checkpoint_path"]
    if not args.skip_trainable_reload:
        required_fields.append("trainable_checkpoint")
    for field in required_fields:
        value = getattr(args, field)
        if not value.exists():
            raise FileNotFoundError(f"missing {field}: {value}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is non-empty: {args.output_dir}; use a new directory or --overwrite")
    if args.output_dir.exists() and args.overwrite:
        previous = args.output_dir / "generation_manifest.jsonl"
        if previous.exists():
            previous.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, record_audit = load_automask_manifest(args.manifest, require_files=True)
    model, vae, inferencer, infer_args, model_audit = build_inference_model(args)
    if args.skip_trainable_reload:
        checkpoint_payload: dict[str, Any] = {"format": "base_skip_trainable_reload", "step": 0}
        checkpoint_sha256: str | None = None
        checkpoint_path_str: str | None = None
    else:
        checkpoint_payload = torch.load(args.trainable_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint_payload.get("format") not in ("gold40_automask_v5_paired_fsdp", "gold40_automask_v6_align_fsdp"):
            raise RuntimeError(f"not a Gold40 AutoMask-v5/v6-align checkpoint: {checkpoint_payload.get('format')!r}")
        checkpoint_sha256 = sha256_file(args.trainable_checkpoint)
        checkpoint_path_str = str(args.trainable_checkpoint.resolve())
    manifest_path = args.output_dir / "generation_manifest.jsonl"
    image_root = args.output_dir / "images"
    rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for pair_index, record in enumerate(records):
                pair_id = record["pair_id"]
                for seed in args.seeds:
                    for state in STATE_IDS:
                        output_path = image_root / pair_id / f"seed_{seed:010d}__{state}.png"
                        if output_path.exists() and not args.overwrite:
                            raise FileExistsError(f"refusing to overwrite generation: {output_path}")
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        generated = _generate_one(
                            inferencer,
                            record,
                            output_path,
                            infer_args,
                            seed=seed,
                            prompt=render_state_prompt(record, state),
                        )
                        path = Path(generated).resolve()
                        if not path.is_file() or path.stat().st_size == 0:
                            raise RuntimeError(f"generation failed for {pair_id}/{seed}/{state}")
                        row = {
                            "schema_version": "gold40_automask_v5_same_seed_generation_v1",
                            "pair_id": pair_id,
                            "source_scene_id": record["source_scene_id"],
                            "intervention_type": record["intervention_type"],
                            "seed": seed,
                            "state": state,
                            "output_path": str(path),
                            "output_sha256": sha256_file(path),
                            "checkpoint_path": checkpoint_path_str,
                            "checkpoint_sha256": checkpoint_sha256,
                            "checkpoint_step": checkpoint_payload.get("step"),
                            "prompt": render_state_prompt(record, state),
                            "prompt_sha256": hashlib.sha256(render_state_prompt(record, state).encode()).hexdigest(),
                            "generation": {
                                "num_timesteps": args.num_timesteps,
                                "cfg_text_scale": args.cfg_text_scale,
                                "cfg_img_scale": args.cfg_img_scale,
                                "cfg_interval": [args.cfg_interval_start, args.cfg_interval_end],
                                "timestep_shift": args.timestep_shift,
                                "visual_image_size": 1024,
                            },
                        }
                        rows.append(row)
                        with manifest_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                        print(json.dumps({"event": "generated", "pair_index": pair_index, "pair_id": pair_id, "seed": seed, "state": state, "output": str(path)}, ensure_ascii=False), flush=True)
    finally:
        del inferencer, vae, model
        torch.cuda.empty_cache()
    atomic_json(args.output_dir / "generation_report.json", {
        "status": "completed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "manifest_prompt_sha256": prompt_audit(records),
        "checkpoint": {"path": checkpoint_path_str, "sha256": checkpoint_sha256, "step": checkpoint_payload.get("step"), "mode": "base" if args.skip_trainable_reload else "gold40_trainable"},
        "seeds": args.seeds,
        "expected_records": len(records) * len(args.seeds) * len(STATE_IDS),
        "generated_records": len(rows),
        "generation_manifest": str(manifest_path.resolve()),
        "record_audit": record_audit,
        "model_audit": model_audit,
    })


if __name__ == "__main__":
    main()
