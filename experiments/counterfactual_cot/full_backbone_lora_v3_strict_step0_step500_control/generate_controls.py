#!/usr/bin/env python3
"""Generate zero-update and step-500 outputs in independent cold-start processes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch

BAGEL_ROOT = Path(__file__).resolve().parents[3]
V3_DIR = BAGEL_ROOT / "experiments/counterfactual_cot/full_backbone_lora_region1024_fsdp_v3"
EVALUATE = V3_DIR / "evaluate.py"
TRAIN_OUTPUT = BAGEL_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_overfit_v3"
)
DEFAULT_OUTPUT = BAGEL_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_"
    "strict_step0_step500_control_v1"
)
DEFAULT_MODEL = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
CHECKPOINTS = {0: TRAIN_OUTPUT / "checkpoints/step_000000.pt", 500: TRAIN_OUTPUT / "checkpoints/step_000500.pt"}
PROFILES = (
    "cot_core_v1",
    "legacy_paired_v1",
    "full_paired_v2",
    "full_without_target_cot_v1",
    "target_prompt_only_v1",
    "delta_only_v2",
)
SEEDS = (1234, 2234, 3234)
EXPECTED_FORMAT = "full_backbone_lora_region1024_cotcore_fsdp_full_shard_resume_v3"
EXPECTED_EXPERIMENT = "cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_overfit_v3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _load_checkpoint_identity(path: Path, expected_step: int) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    step = int(payload.get("global_step", payload.get("step", -1)))
    if payload.get("format") != EXPECTED_FORMAT or payload.get("experiment_name") != EXPECTED_EXPERIMENT or step != expected_step:
        raise RuntimeError(
            f"Checkpoint identity mismatch for {path}: experiment={payload.get('experiment_name')}, "
            f"format={payload.get('format')}, step={step}"
        )
    config = payload.get("config", {})
    locked_config = {
        key: config.get(key)
        for key in (
            "pair_id", "manifest", "llm_path", "vae_path", "checkpoint_path", "lora_r", "lora_alpha",
            "lora_dropout", "trainable_name_regex", "prompt_profile", "region_objective", "max_latent_size",
            "image_size", "vit_image_size", "vit_min_image_size", "activation_checkpointing", "fsdp_sharding",
            "fsdp_world_size", "fsdp_use_orig_params", "fsdp_trainables_replicated", "lora_dtype",
        )
    }
    trainable_state = payload.get("trainable_state_dict", {})
    tensor_schema = [(name, list(tensor.shape), str(tensor.dtype)) for name, tensor in sorted(trainable_state.items())]
    lora_b = [tensor for name, tensor in trainable_state.items() if name.endswith("lora_B")]
    history = payload.get("loss_history", [])
    history_steps = [int(row.get("step", -1)) for row in history]
    result = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "experiment_name": payload.get("experiment_name"),
        "created_at": payload.get("created_at"),
        "format": payload.get("format"),
        "step": step,
        "num_trainable_tensors": len(trainable_state),
        "trainable_schema_sha256": _json_sha256(tensor_schema),
        "num_lora_targets": len(payload.get("lora_target_names", [])),
        "lora_target_names_sha256": _json_sha256(payload.get("lora_target_names", [])),
        "num_lora_b_tensors": len(lora_b),
        "num_nonzero_lora_b_tensors": sum(bool(torch.count_nonzero(tensor).item()) for tensor in lora_b),
        "loss_history_length": len(history),
        "loss_history_steps_sha256": _json_sha256(history_steps),
        "loss_history_sha256": _json_sha256(history),
        "locked_config": locked_config,
    }
    del payload
    return result


def _validate_pair(step0: Dict[str, Any], step500: Dict[str, Any]) -> None:
    common_fields = (
        "experiment_name", "format", "num_trainable_tensors", "trainable_schema_sha256", "num_lora_targets",
        "lora_target_names_sha256", "num_lora_b_tensors", "locked_config",
    )
    for key in common_fields:
        if step0[key] != step500[key]:
            raise RuntimeError(f"Checkpoint comparison is not controlled: field {key!r} differs")
    config = step0["locked_config"]
    expected = {
        "prompt_profile": "cot_core_v1", "region_objective": "0.25_old_0.25_new_0.50_preserve",
        "max_latent_size": 64, "image_size": 1024, "vit_image_size": 980, "vit_min_image_size": 980,
        "activation_checkpointing": True, "fsdp_sharding": "FULL_SHARD", "fsdp_world_size": 2,
        "fsdp_use_orig_params": True,
    }
    mismatches = {key: {"actual": config.get(key), "expected": value} for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Locked v3 checkpoint configuration mismatch: {json.dumps(mismatches)}")
    if step0["num_lora_b_tensors"] != 168 or step0["num_nonzero_lora_b_tensors"] != 0:
        raise RuntimeError("Step 0 is not the expected zero-update LoRA initialization")
    if step500["num_nonzero_lora_b_tensors"] != 168:
        raise RuntimeError("Step 500 does not contain updates in every LoRA-B tensor")
    if step0["loss_history_length"] != 0 or step500["loss_history_length"] != 500:
        raise RuntimeError("Checkpoint loss histories do not match zero-update and 500-update semantics")
    expected_history_steps_sha = _json_sha256(list(range(1, 501)))
    if step500["loss_history_steps_sha256"] != expected_history_steps_sha:
        raise RuntimeError("Step-500 loss history is not a contiguous 1..500 trajectory")
    if not step0.get("created_at") or not step500.get("created_at") or step0["created_at"] >= step500["created_at"]:
        raise RuntimeError("Checkpoint creation timestamps do not support step0 -> step500 ordering")


def _file_identity(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _runtime_identities(model_dir: Path) -> Dict[str, Dict[str, Any]]:
    assets = {
        "base_checkpoint": model_dir / "ema.safetensors",
        "vae_checkpoint": model_dir / "ae.safetensors",
        "llm_config": model_dir / "llm_config.json",
        "vit_config": model_dir / "vit_config.json",
        "model_config": model_dir / "config.json",
        "evaluator": EVALUATE,
        "v3_train": V3_DIR / "train.py",
        "prompt_profiles": BAGEL_ROOT / "experiments/counterfactual_cot/data/prompt_profiles.py",
    }
    return {label: _file_identity(path) for label, path in assets.items()}


def _validate_report_and_build_receipt(
    report_path: Path,
    step: int,
    checkpoint_identity: Dict[str, Any],
    runtime_identities: Dict[str, Dict[str, Any]],
    run_id: str,
) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_step_dir = report_path.parent.resolve()
    if report.get("status") != "completed" or int(report.get("step", -1)) != step:
        raise RuntimeError(f"Incomplete or wrong-step evaluation report: {report_path}")
    if Path(report.get("checkpoint", "")).resolve() != Path(checkpoint_identity["path"]):
        raise RuntimeError(f"Evaluation report checkpoint does not match protocol at step {step}")
    checkpoint_report = report.get("checkpoint_report", {})
    if int(checkpoint_report.get("step", -1)) != step or checkpoint_report.get("checkpoint_format") != EXPECTED_FORMAT:
        raise RuntimeError(f"Evaluation checkpoint report mismatch at step {step}")
    if report.get("pair_id") != checkpoint_identity["locked_config"].get("pair_id"):
        raise RuntimeError(f"Evaluation pair mismatch at step {step}")
    if report.get("training_prompt_profile") != "cot_core_v1":
        raise RuntimeError(f"Training prompt-profile mismatch at step {step}")
    generations = report.get("generations", [])
    expected_keys = {(profile, seed) for profile in PROFILES for seed in SEEDS}
    actual_keys = {(str(row.get("profile")), int(row.get("seed", -1))) for row in generations}
    if len(generations) != len(expected_keys) or actual_keys != expected_keys:
        raise RuntimeError(f"Generation matrix mismatch at step {step}: {actual_keys ^ expected_keys}")
    files: List[Dict[str, Any]] = []
    for row in generations:
        profile = str(row["profile"])
        if row.get("prompt") != report.get("prompt_manifest", {}).get(profile):
            raise RuntimeError(f"Prompt text/manifest mismatch at step {step}, profile={profile}")
        if float(row.get("cfg_img_scale", -1)) != 2.0:
            raise RuntimeError(f"Unexpected image CFG at step {step}")
        path = Path(row["path"]).resolve()
        if expected_step_dir not in path.parents:
            raise RuntimeError(f"Generation escaped expected step directory: {path}")
        identity = _file_identity(path)
        identity.update(profile=profile, seed=int(row["seed"]), cfg_img_scale=float(row["cfg_img_scale"]))
        files.append(identity)
    for label in ("source_visualization", "target_visualization", "contact_sheet"):
        path = Path(report[label]).resolve()
        if expected_step_dir not in path.parents and path != expected_step_dir:
            raise RuntimeError(f"Report asset escaped expected step directory: {path}")
        identity = _file_identity(path)
        identity["role"] = label
        files.append(identity)
    if _runtime_identities(Path(checkpoint_identity["locked_config"]["llm_path"])) != runtime_identities:
        raise RuntimeError("Runtime model/code assets changed during generation")
    return {
        "run_id": run_id,
        "step": step,
        "checkpoint_sha256": checkpoint_identity["sha256"],
        "checkpoint_path": checkpoint_identity["path"],
        "report": _file_identity(report_path),
        "files": files,
        "runtime_identities": runtime_identities,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    if tuple(args.profiles) != PROFILES or tuple(args.seeds) != SEEDS:
        parser.error(f"Strict protocol locks profiles={PROFILES} and seeds={SEEDS}; overrides are not allowed")
    return args


def main() -> None:
    args = _parse_args()
    if not EVALUATE.is_file():
        raise FileNotFoundError(EVALUATE)
    if (args.output_dir / "control_protocol.json").exists() or (args.output_dir / "generations").exists():
        raise RuntimeError(
            f"Strict output already contains a prior run: {args.output_dir}. "
            "Use a new --output-dir; stale or mixed generations are never overwritten."
        )
    identities = {str(step): _load_checkpoint_identity(path, step) for step, path in CHECKPOINTS.items()}
    _validate_pair(identities["0"], identities["500"])
    runtime_identities = _runtime_identities(args.model_dir)
    run_id = str(uuid.uuid4())
    protocol = {
        "status": "generating",
        "run_id": run_id,
        "created_at": _utc_now(),
        "design": "paired_checkpoint_comparison_with_independent_cold_start_processes",
        "baseline_label": "zero_update_step0_not_a_separate_pure_base_checkpoint",
        "treatment_label": "same_experiment_step500_with_strong_internal_trajectory_evidence",
        "ancestry_scope": (
            "Legacy checkpoints contain no run UUID, so cryptographic parent-child ancestry cannot be proven retrospectively. "
            "The protocol instead enforces identical experiment/config/schema, zero LoRA-B and empty history at step0, "
            "all LoRA-B updated plus contiguous 1..500 history at step500, and ordered creation timestamps."
        ),
        "checkpoint_identities": identities,
        "runtime_identities": runtime_identities,
        "locked_generation": {
            "profiles": args.profiles,
            "seeds": args.seeds,
            "source_image": "same locked pair for every generation",
            "visual_image_size": 1024,
            "visual_vae_min_image_size": 1024,
            "visual_vit_image_size": 980,
            "visual_vit_min_image_size": 980,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 2.0,
            "cfg_interval": [0.0, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 25,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "text_channel",
            "cfg_ablation": False,
        },
        "execution_order": [0, 500],
        "independent_process_per_checkpoint": True,
        "same_physical_gpu": args.gpu_id,
        "commands": [],
        "generation_receipts": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "control_protocol.json", protocol)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu_id),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128",
    )
    for step in (0, 500):
        command = [
            sys.executable,
            str(EVALUATE),
            "--llm-path", str(args.model_dir),
            "--vae-path", str(args.model_dir / "ae.safetensors"),
            "--checkpoint-path", str(args.model_dir / "ema.safetensors"),
            "--output-dir", str(TRAIN_OUTPUT),
            "--eval-dir", str(args.output_dir / "generations"),
            "--resume-checkpoint", str(CHECKPOINTS[step]),
            "--profiles", *args.profiles,
            "--seeds", *[str(seed) for seed in args.seeds],
            "--visual-image-size", "1024",
            "--visual-vae-min-image-size", "1024",
            "--visual-vit-image-size", "980",
            "--visual-vit-min-image-size", "980",
            "--cfg-text-scale", "4.0",
            "--cfg-img-scale", "2.0",
            "--cfg-interval-start", "0.0",
            "--cfg-interval-end", "1.0",
            "--timestep-shift", "3.0",
            "--num-timesteps", "25",
            "--cfg-renorm-min", "0.0",
            "--cfg-renorm-type", "text_channel",
            "--skip-cfg-ablation",
        ]
        protocol["commands"].append(command)
        _atomic_json(args.output_dir / "control_protocol.json", protocol)
        log_path = args.output_dir / f"generate_step_{step:06d}.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            subprocess.run(
                command,
                cwd=BAGEL_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        report_path = args.output_dir / "generations" / f"step_{step:06d}" / "evaluation_report.json"
        if not report_path.is_file():
            raise RuntimeError(f"Cold-start generation produced no report: {report_path}")
        receipt = _validate_report_and_build_receipt(
            report_path,
            step,
            identities[str(step)],
            runtime_identities,
            run_id,
        )
        receipt_path = args.output_dir / "receipts" / f"step_{step:06d}.json"
        _atomic_json(receipt_path, receipt)
        protocol["generation_receipts"][str(step)] = _file_identity(receipt_path)
        _atomic_json(args.output_dir / "control_protocol.json", protocol)
    protocol["status"] = "completed"
    protocol["completed_at"] = _utc_now()
    _atomic_json(args.output_dir / "control_protocol.json", protocol)
    print(json.dumps({"status": "completed", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
