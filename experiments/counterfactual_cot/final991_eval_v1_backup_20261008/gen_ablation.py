#!/usr/bin/env python3
"""Module B: prompt-profile ablation generation (7 profiles x val 51 pairs x seed 1234).

Reuses data/prompt_profiles.py render_prompt_profile for the 6 legacy profiles and adds
wrong_cot_v1 (Target CoT slot filled with the pair's own Source CoT).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

BAGEL = Path("/group/40016/niightli/BAGEL")
sys.path.insert(0, str(BAGEL))
from experiments.counterfactual_cot.data.prompt_profiles import (  # noqa: E402
    PROFILES,
    render_prompt_profile,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.counterfactual_cot.gold40_automask_dual_target_cot1024_fsdp_v5.paired_data import (  # noqa: E402
    load_automask_manifest,
)
from experiments.counterfactual_cot.gold40_automask_dual_target_cot1024_fsdp_v5.generate_same_seed_states import (  # noqa: E402
    build_inference_model,
    _generate_one,
)

PROFILES_TO_RUN = [
    "cot_core_v1",
    "full_without_target_cot_v1",
    "target_prompt_only_v1",
    "delta_only_v2",
    "full_paired_v2",
    "legacy_paired_v1",
    "wrong_cot_v1",
]


def render_wrong_cot(pair: dict) -> str:
    """Target CoT slot filled with Source CoT: does the model blindly follow any CoT?"""
    source_prompt = pair["source"]["prompt"]
    source_cot = pair["source"]["cot_text"]
    return (
        "[MODE=COUNTERFACTUAL_EDIT_COT_CORE]\n"
        f"Source Prompt: {source_prompt}\n"
        f"Target CoT: {source_cot}\n"
        "Generate the target image by strictly following Target CoT."
    )


def render_profile(pair: dict, profile_id: str) -> str:
    if profile_id == "wrong_cot_v1":
        return render_wrong_cot(pair)
    return render_prompt_profile(pair, profile_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "final991"], required=True)
    parser.add_argument("--manifest", default=str(BAGEL / "experiments/counterfactual_cot/final991_automask_v5_data/artifacts/splits/val.jsonl"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-timesteps", type=int, default=25)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records, _ = load_automask_manifest(Path(args.manifest), require_files=True)

    gen = argparse.Namespace(
        llm_path=Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT"),
        vae_path=Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors"),
        base_checkpoint_path=Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors"),
        trainable_checkpoint=Path("/dockerdata/niightli/BAGEL/models/final991_automask_v5_latest.pt"),
        skip_trainable_reload=(args.model == "base"),
        num_timesteps=args.num_timesteps,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval_start=0.0,
        cfg_interval_end=1.0,
        timestep_shift=args.timestep_shift,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )
    model, vae, inferencer, infer_args, audit = build_inference_model(gen)

    manifest_path = out / "generation_manifest.jsonl"
    if manifest_path.exists() and args.overwrite:
        manifest_path.unlink()
    rows = []
    with torch.no_grad():
        for record in records:
            pair_id = record["pair_id"]
            for profile_id in PROFILES_TO_RUN:
                prompt = render_profile(record, profile_id)
                output_path = out / "images" / pair_id / f"{profile_id}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists() and not args.overwrite:
                    continue
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
                _generate_one(inferencer, record, output_path, infer_args, seed=args.seed, prompt=prompt)
                row = {
                    "schema_version": "final991_eval_ablation_v1",
                    "model": args.model,
                    "pair_id": pair_id,
                    "source_scene_id": record["source_scene_id"],
                    "intervention_type": record["intervention_type"],
                    "profile_id": profile_id,
                    "seed": args.seed,
                    "output_path": str(output_path.resolve()),
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
                rows.append(row)
                with manifest_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                print(json.dumps({"event": "generated", "pair_id": pair_id, "profile": profile_id}), flush=True)
    (out / "generation_report.json").write_text(json.dumps({
        "status": "completed",
        "model": args.model,
        "seed": args.seed,
        "profiles": PROFILES_TO_RUN,
        "generated": len(rows),
        "model_audit": audit,
    }, indent=1))
    print(json.dumps({"event": "done", "generated": len(rows)}))


if __name__ == "__main__":
    main()
