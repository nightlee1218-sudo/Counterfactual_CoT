#!/usr/bin/env python3
"""Run the two frozen Sentinel-6 training-matched prompt conditions.

This is a thin, locked adapter around the previously audited Micro4 LoRA
trainer. It changes only the frozen manifest, pair count, LR, duration, and
checkpoint schedule required by the preregistered Sentinel-6 gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.training import gold40_v3_lora_micro4_lr_probe as trainer


DATA_DIR = BAGEL_ROOT / "experiments/counterfactual_cot/data/sentinel6_final_v2"
OUTPUTS_DIR = BAGEL_ROOT / "experiments/counterfactual_cot/outputs"
MODEL_DIR = BAGEL_ROOT / "models/BAGEL-7B-MoT"
PROFILE_CONTRACTS = {
    "delta_only_v2": {
        "manifest": DATA_DIR / "sentinel6_gold40_final__profile=delta_only_v2.jsonl",
        "sha256": "65c9f7b801e2611b1c10bee6edd12c86a17d96c5a24adc1ea5fe9f3ee3bf3f0b",
    },
    "cot_core_v1": {
        "manifest": DATA_DIR / "sentinel6_gold40_final__profile=cot_core_v1.jsonl",
        "sha256": "30df3437e1e8de611e6e4b99a8d58bdbc2c5511aeae231e0335998938072639c",
    },
}
NUM_PAIRS = 6
STEPS = (0, 10, 25, 50)
LR = 3e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(profile: str, manifest: Path, expected_sha: str) -> None:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    actual_sha = sha256_file(manifest)
    if actual_sha != expected_sha:
        raise RuntimeError(f"Manifest SHA mismatch: got={actual_sha} expected={expected_sha}")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != NUM_PAIRS:
        raise ValueError(f"Expected {NUM_PAIRS} records, got {len(records)}")
    for record in records:
        provenance = record.get("sentinel6_provenance", {})
        prompt = record.get("training_prompts", {}).get("paired_edit", "")
        if provenance.get("prompt_profile") != profile:
            raise ValueError(f"Profile mismatch for {record.get('pair_id')}: {provenance}")
        if not prompt.startswith("[MODE=COUNTERFACTUAL_EDIT_"):
            raise ValueError(f"Missing frozen training prompt for {record.get('pair_id')}")
    classroom = next(record for record in records if record["pair_id"] == "cf_source_classroom_object_replace_0000")
    if classroom["intervention"].get("new_value") != "globe":
        raise ValueError("Sentinel-6 classroom target must be globe")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILE_CONTRACTS))
    parser.add_argument("--mode", required=True, choices=("validate", "dry-run", "train"))
    args = parser.parse_args()

    contract = PROFILE_CONTRACTS[args.profile]
    manifest = Path(contract["manifest"])
    validate_manifest(args.profile, manifest, contract["sha256"])
    for required in ("ema.safetensors", "ae.safetensors", "llm_config.json", "vit_config.json"):
        if not (MODEL_DIR / required).is_file():
            raise FileNotFoundError(MODEL_DIR / required)
    if args.mode == "validate":
        print(json.dumps({
            "status": "validated",
            "profile": args.profile,
            "manifest": str(manifest),
            "manifest_sha256": contract["sha256"],
            "num_pairs": NUM_PAIRS,
            "lr": LR,
            "steps": list(STEPS),
            "classroom_target": "globe",
        }, indent=2))
        return

    stem = (
        OUTPUTS_DIR
        / f"sentinel6_gold40_final_v2__profile={args.profile}__method=bridge14_loraL24-27r8a16__lr=3e-5__steps=50"
    )
    dry_run_dir = Path(f"{stem}__dry_run")
    training_dir = Path(f"{stem}__training")
    dry_run_report = dry_run_dir / "micro4_dry_run_report.json"

    trainer.MICRO4_MANIFEST = manifest
    trainer.MICRO4_MANIFEST_SHA256_EXPECT = contract["sha256"]
    trainer.MICRO4_NUM_PAIRS = NUM_PAIRS
    trainer.MICRO4_SAVE_STEPS = STEPS
    trainer.MICRO4_EVAL_STEPS = STEPS
    trainer.BASE_OUTPUT_STEM = stem
    trainer.MICRO4_ALLOWED_DIFF_KEYS = trainer.MICRO4_ALLOWED_DIFF_KEYS | {
        "llm_path",
        "vae_path",
        "checkpoint_path",
        "eval_max_batches",
    }

    forwarded = [
        str(Path(trainer.__file__)),
        "--manifest", str(manifest),
        "--llm-path", str(MODEL_DIR),
        "--vae-path", str(MODEL_DIR / "ae.safetensors"),
        "--checkpoint-path", str(MODEL_DIR / "ema.safetensors"),
        "--allow-non-dockerdata",
        "--num-samples", str(NUM_PAIRS),
        "--steps", "50",
        "--eval-steps", "0", "10", "25", "50",
        "--save-steps", "0", "10", "25", "50",
        "--eval-max-batches", str(NUM_PAIRS),
        "--lr", str(LR),
        "--output-dir", str(dry_run_dir if args.mode == "dry-run" else training_dir),
        "--dry-run-report", str(dry_run_report),
    ]
    if args.mode == "dry-run":
        forwarded.append("--dry-run")
    else:
        forwarded.append("--acknowledge-full-training")
    sys.argv = forwarded
    trainer.main()


if __name__ == "__main__":
    main()
