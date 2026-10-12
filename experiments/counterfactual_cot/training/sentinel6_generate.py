#!/usr/bin/env python3
"""Generate matched Step-0/Step-50 Sentinel-6 images for one prompt profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.training.sentinel6_train import (
    MODEL_DIR,
    NUM_PAIRS,
    OUTPUTS_DIR,
    PROFILE_CONTRACTS,
    sha256_file,
    validate_manifest,
)


CHECKPOINTS = ("step_000", "step_050")
GENERATION_SEEDS = (1234, 2234, 3234)
NUM_TIMESTEPS = 25
CFG_TEXT_SCALE = 4.0
CFG_IMG_SCALE = 2.0
VISUAL_IMAGE_SIZE = 1024
EXPECTED_TRAINABLE_TENSORS = 62


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_records(manifest: Path, profile: str) -> List[Dict[str, Any]]:
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != NUM_PAIRS:
        raise ValueError(f"Expected {NUM_PAIRS} records, got {len(records)}")
    for record in records:
        if record.get("sentinel6_provenance", {}).get("prompt_profile") != profile:
            raise ValueError(f"Prompt profile mismatch for {record.get('pair_id')}")
        prompt = record.get("training_prompts", {}).get("paired_edit", "")
        if not prompt:
            raise ValueError(f"Missing frozen paired_edit prompt for {record.get('pair_id')}")
    return records


def run_dir(profile: str) -> Path:
    return OUTPUTS_DIR / (
        f"sentinel6_gold40_final_v2__profile={profile}__method=bridge14_loraL24-27r8a16"
        "__lr=3e-5__steps=50__training"
    )


def output_dir(profile: str) -> Path:
    return OUTPUTS_DIR / (
        f"sentinel6_gold40_final_v2__profile={profile}__checkpoints=000_050"
        "__seeds=1234_2234_3234__sampling=nt25_cfgT4_cfgI2_1024__inference"
    )


def checkpoint_contracts(profile: str) -> Dict[str, Dict[str, Any]]:
    training = run_dir(profile)
    report_path = training / "micro4_full_training_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError(f"Training report is not passed: {report_path}")
    contracts = {}
    for item in report.get("checkpoint_records", []):
        label = f"step_{int(item['step']):03d}"
        if label in CHECKPOINTS:
            contracts[label] = item
    missing = [label for label in CHECKPOINTS if label not in contracts]
    if missing:
        raise ValueError(f"Training report is missing checkpoints: {missing}")
    for label, item in contracts.items():
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Checkpoint SHA mismatch for {label}: got={actual} expected={item['sha256']}")
    return contracts


def validate(profile: str, resume: bool = False) -> Dict[str, Any]:
    contract = PROFILE_CONTRACTS[profile]
    manifest = Path(contract["manifest"])
    validate_manifest(profile, manifest, contract["sha256"])
    records = load_records(manifest, profile)
    checkpoints = checkpoint_contracts(profile)
    out = output_dir(profile)
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite non-empty inference directory: {out}")
    return {
        "status": "validated",
        "profile": profile,
        "manifest": str(manifest),
        "manifest_sha256": contract["sha256"],
        "pair_count": len(records),
        "checkpoints": {label: {"path": item["path"], "sha256": item["sha256"]} for label, item in checkpoints.items()},
        "generation_seeds": list(GENERATION_SEEDS),
        "expected_image_count": len(records) * len(CHECKPOINTS) * len(GENERATION_SEEDS),
        "output_dir": str(out),
    }


def generate(profile: str, resume: bool = False) -> None:
    validation = validate(profile, resume=resume)
    import torch
    from PIL import Image
    from experiments.counterfactual_cot.data.schema import load_jsonl
    from experiments.counterfactual_cot.training.gold40_lora import (
        _apply_trainable_and_audit,
        _build_model_with_lora,
    )
    from experiments.counterfactual_cot.training.gold40_overfit import (
        _generate_one,
        _make_inferencer,
        _save_reference_image,
    )
    from experiments.counterfactual_cot.training.lora_2step_smoke import load_trainable_strict
    from experiments.counterfactual_cot.training.smoke_train import _device, _load_tokenizer

    manifest = Path(PROFILE_CONTRACTS[profile]["manifest"])
    records = list(load_jsonl(manifest))
    checkpoints = checkpoint_contracts(profile)
    out = output_dir(profile)
    out.mkdir(parents=True, exist_ok=resume)
    images_dir = out / "images"
    references_dir = out / "references"
    images_dir.mkdir(exist_ok=resume)
    references_dir.mkdir(exist_ok=resume)

    def valid_generated_image(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                return image.size == (VISUAL_IMAGE_SIZE, VISUAL_IMAGE_SIZE)
        except (OSError, ValueError):
            return False

    model_args = argparse.Namespace(
        llm_path=str(MODEL_DIR),
        vae_path=str(MODEL_DIR / "ae.safetensors"),
        checkpoint_path=str(MODEL_DIR / "ema.safetensors"),
        precision="bf16",
        device="cuda",
        model_device_map="single",
        device_map_max_memory="34GiB",
        lora_layers=[24, 25, 26, 27],
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        trainable_name_regex=r"^(time_embedder|latent_pos_embed|vit_pos_embed|vae2llm|llm2vae|connector)\.",
        image_size=256,
        vit_image_size=224,
        vit_min_image_size=224,
        visual_image_size=VISUAL_IMAGE_SIZE,
        visual_vae_min_image_size=512,
        visual_vit_image_size=980,
        visual_vit_min_image_size=224,
        expected_num_tokens=512,
        max_num_tokens_per_sample=1536,
        max_num_tokens=1536,
        seed=1234,
        variant="paired_cf_mvp",
        num_samples=NUM_PAIRS,
        require_paired_vit=True,
        cfg_text_scale=CFG_TEXT_SCALE,
        cfg_img_scale=CFG_IMG_SCALE,
        cfg_interval_start=0.0,
        cfg_interval_end=1.0,
        timestep_shift=3.0,
        num_timesteps=NUM_TIMESTEPS,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )

    device = _device(model_args.device)
    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(model_args.llm_path)
    model, vae_model, wrappers, target_names, base_report = _build_model_with_lora(
        model_args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens
    )
    trainable_audit = _apply_trainable_and_audit(model, wrappers, target_names, model_args, device.type)
    trainable_count = len({name for name, parameter in model.named_parameters() if parameter.requires_grad})
    if trainable_count != EXPECTED_TRAINABLE_TENSORS:
        raise RuntimeError(f"Expected {EXPECTED_TRAINABLE_TENSORS} trainable tensors, got {trainable_count}")

    for record in records:
        pair_id = record["pair_id"]
        _save_reference_image(record, "source", references_dir / f"{pair_id}__source.png", VISUAL_IMAGE_SIZE)
        _save_reference_image(record, "target", references_dir / f"{pair_id}__golden.png", VISUAL_IMAGE_SIZE)

    inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, model_args)
    generated = []
    for checkpoint_label in CHECKPOINTS:
        checkpoint = checkpoints[checkpoint_label]
        load_result = load_trainable_strict(Path(checkpoint["path"]), model)
        if load_result["num_loaded_tensors"] != EXPECTED_TRAINABLE_TENSORS:
            raise RuntimeError(f"Strict load count mismatch for {checkpoint_label}: {load_result}")
        if float(load_result["max_abs_load_copy_diff"]) != 0.0:
            raise RuntimeError(f"Strict load copy mismatch for {checkpoint_label}: {load_result}")
        model.eval()
        for record in records:
            pair_id = record["pair_id"]
            prompt = record["training_prompts"]["paired_edit"]
            for seed in GENERATION_SEEDS:
                image_path = images_dir / f"{pair_id}__{checkpoint_label}__seed_{seed}.png"
                reused = resume and valid_generated_image(image_path)
                if not reused:
                    if image_path.exists():
                        image_path.unlink()
                    with torch.no_grad():
                        _generate_one(
                            inferencer,
                            record,
                            image_path,
                            model_args,
                            seed=seed,
                            prompt=prompt,
                        )
                generated.append({
                    "pair_id": pair_id,
                    "intervention_type": record["intervention"]["intervention_type"],
                    "profile": profile,
                    "prompt": prompt,
                    "checkpoint": checkpoint_label,
                    "checkpoint_path": checkpoint["path"],
                    "checkpoint_sha256": checkpoint["sha256"],
                    "seed": seed,
                    "image": str(image_path),
                    "image_sha256": sha256_file(image_path),
                })

    expected = validation["expected_image_count"]
    if len(generated) != expected:
        raise RuntimeError(f"Generated {len(generated)} images, expected {expected}")
    atomic_json(out / "generation_manifest.json", {
        "schema": "sentinel6_matched_inference_v1",
        "status": "completed",
        "profile": profile,
        "manifest": str(manifest),
        "manifest_sha256": PROFILE_CONTRACTS[profile]["sha256"],
        "training_run": str(run_dir(profile)),
        "model_base_report": base_report,
        "trainable_audit": {key: value for key, value in trainable_audit.items() if key != "dtype_desc"},
        "generation_protocol": {
            "checkpoints": list(CHECKPOINTS),
            "seeds": list(GENERATION_SEEDS),
            "num_timesteps": NUM_TIMESTEPS,
            "cfg_text_scale": CFG_TEXT_SCALE,
            "cfg_img_scale": CFG_IMG_SCALE,
            "image_size": VISUAL_IMAGE_SIZE,
            "prompt_source": "training_prompts.paired_edit",
            "training_inference_prompt_match": True,
        },
        "generated_image_count": len(generated),
        "records": generated,
    })
    print(json.dumps({"status": "completed", "profile": profile, "images": len(generated), "output_dir": str(out)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILE_CONTRACTS))
    parser.add_argument("--mode", required=True, choices=("validate", "generate"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a non-empty inference directory and reuse verified completed images.",
    )
    args = parser.parse_args()
    if args.resume and args.mode != "generate":
        parser.error("--resume is only valid with --mode generate")
    if args.mode == "validate":
        print(json.dumps(validate(args.profile), indent=2))
    else:
        generate(args.profile, resume=args.resume)


if __name__ == "__main__":
    main()
