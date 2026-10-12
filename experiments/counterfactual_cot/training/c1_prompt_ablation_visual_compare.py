#!/usr/bin/env python3
"""Audited prompt-only ablation for the frozen C1 single-ball edit.

This entrypoint intentionally reuses the proven single-pair visual comparison
worker and changes only the prompt profile supplied to inference.  It never
modifies training checkpoints or legacy visual-comparison outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import secrets
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from experiments.counterfactual_cot.data.prompt_profiles import (
    PROFILES,
    profile_ids_for_suite,
    prompt_profile_manifest,
    render_prompt_profile,
)
from experiments.counterfactual_cot.training import c1_single_pair_visual_compare as base

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = Path(__file__).resolve()
PROMPT_MODULE_PATH = REPO_ROOT / "experiments/counterfactual_cot/data/prompt_profiles.py"
FORMAL_OUTPUT_ROOT = REPO_ROOT / "experiments/counterfactual_cot/outputs/c1_prompt_ablation_v1"
PREFLIGHT_OUTPUT_ROOT = REPO_ROOT / "experiments/counterfactual_cot/outputs/c1_prompt_ablation_preflight_v1"
PAIR_ID = base.PAIR_ID
SEEDS: Tuple[int, ...] = base.SEEDS

FULL_IMAGE_CHECKPOINT_DIR = REPO_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_c1_single_ball_p4d_v1/checkpoints"
)
REGION_SPLIT_CHECKPOINT_DIR = REPO_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_c1_single_ball_region_split_v1_500steps/checkpoints"
)

CHECKPOINTS: Dict[str, Dict[str, Any]] = {
    "common_step000": {
        "architecture": "common_initialization",
        "step": 0,
        "path": REGION_SPLIT_CHECKPOINT_DIR / "step_000_trainable.pt",
        "sha256": "9658b383e277fc9ffec65985f29e4ffa18747f8f437c56e584ec400531659dc8",
        "size_bytes": 135587762,
        "purpose": "Common initialization and pretrained prompt-response baseline.",
    },
    "full_image_step500": {
        "architecture": "full_image_loss",
        "step": 500,
        "path": FULL_IMAGE_CHECKPOINT_DIR / "step_500_trainable.pt",
        "sha256": "ca15aef699e9255e305a865fb3ea113531f101f021f60a58c87a92fed4a602ac",
        "size_bytes": 135587762,
        "purpose": "Legacy paired-v1 training with native full-image MSE.",
    },
    "region_split_step400": {
        "architecture": "region_split_loss",
        "step": 400,
        "path": REGION_SPLIT_CHECKPOINT_DIR / "step_400_trainable.pt",
        "sha256": "4535d99ccdc009ff369074897a95fec39313f612d50d822ec7893493aeaff4d8",
        "size_bytes": 135587762,
        "purpose": "Region-split fixed-evaluation best checkpoint.",
    },
    "region_split_step500": {
        "architecture": "region_split_loss",
        "step": 500,
        "path": REGION_SPLIT_CHECKPOINT_DIR / "step_500_trainable.pt",
        "sha256": "98ec12e8330b744394a3c688faaf1302485af504d080d9f8a4c016b80d6dd740",
        "size_bytes": 135587762,
        "purpose": "Region-split final checkpoint.",
    },
}

DEFAULT_CHECKPOINT_IDS: Tuple[str, ...] = tuple(CHECKPOINTS)
DEFAULT_PROFILE_IDS: Tuple[str, ...] = profile_ids_for_suite("core")
EXPECTED_EXISTING_TENSOR_COUNT = base.EXPECTED_EXISTING_TENSORS
EXPECTED_LORA_TENSOR_COUNT = base.EXPECTED_LORA_TENSORS
EXPECTED_TOTAL_TENSOR_COUNT = base.EXPECTED_LORA_TOTAL_TENSORS
PARENT_ENV_FLAG = "C1_PROMPT_ABLATION_PARENT"
PARENT_RUN_ID_ENV = "C1_PROMPT_ABLATION_RUN_ID"
PARENT_REPORT_SHA_ENV = "C1_PROMPT_ABLATION_PREFLIGHT_SHA256"
WORKER_ALLOW_GPU_ENV = "C1_PROMPT_ABLATION_WORKER_ALLOW_GPU"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _load_pair() -> Dict[str, Any]:
    pair = base._load_unique_pair()
    if pair.get("pair_id") != PAIR_ID:
        raise RuntimeError(f"Unexpected pair_id: {pair.get('pair_id')!r}")
    return pair


def _parse_csv(value: str, allowed: Sequence[str], label: str) -> Tuple[str, ...]:
    if value == "all":
        return tuple(allowed)
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    if not selected:
        raise ValueError(f"{label} selection is empty")
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; allowed={list(allowed)}")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Duplicate {label} selection: {selected}")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "generate", "worker"), required=True)
    parser.add_argument("--suite", choices=("core", "conflict", "all"), default="core")
    parser.add_argument("--profiles", default=None, help="Comma-separated profile IDs; overrides --suite.")
    parser.add_argument("--checkpoints", default="all", help="Comma-separated checkpoint IDs or all.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-id", choices=tuple(CHECKPOINTS), default=None)
    parser.add_argument("--profile-id", choices=tuple(PROFILES), default=None)
    parser.add_argument("--worker-output-dir", type=Path, default=None)
    parser.add_argument("--parent-run-id", default=None)
    parser.add_argument("--parent-report-sha256", default=None)
    return parser.parse_args()


def _selection(args: argparse.Namespace) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    allowed_profiles = profile_ids_for_suite(args.suite)
    profile_ids = (
        _parse_csv(args.profiles, tuple(PROFILES), "profile")
        if args.profiles is not None
        else allowed_profiles
    )
    checkpoint_ids = _parse_csv(args.checkpoints, DEFAULT_CHECKPOINT_IDS, "checkpoint")
    return profile_ids, checkpoint_ids


def _validate_checkpoint(checkpoint_id: str) -> Dict[str, Any]:
    spec = CHECKPOINTS[checkpoint_id]
    path = Path(spec["path"])
    result: Dict[str, Any] = {
        "checkpoint_id": checkpoint_id,
        "architecture": spec["architecture"],
        "step": int(spec["step"]),
        "purpose": spec["purpose"],
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "size_bytes_expected": spec["size_bytes"],
        "sha256": _sha256(path) if path.is_file() else None,
        "sha256_expected": spec["sha256"],
        "status": "failed",
    }
    if not path.is_file():
        result["error"] = "checkpoint_missing"
        return result
    try:
        if path.stat().st_size != spec["size_bytes"] or result["sha256"] != spec["sha256"]:
            raise RuntimeError("checkpoint frozen size/SHA256 identity mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("trainable_state_dict"), dict):
            raise RuntimeError("checkpoint is missing trainable_state_dict")
        state = payload["trainable_state_dict"]
        lora_keys = {
            key for key in state
            if key.endswith(".lora_A") or key.endswith(".lora_B")
        }
        existing_keys = set(state) - lora_keys
        result.update(
            {
                "tensor_count": len(state),
                "expected_tensor_count": EXPECTED_TOTAL_TENSOR_COUNT,
                "existing_tensor_count": len(existing_keys),
                "expected_existing_tensor_count": EXPECTED_EXISTING_TENSOR_COUNT,
                "lora_tensor_count": len(lora_keys),
                "expected_lora_tensor_count": EXPECTED_LORA_TENSOR_COUNT,
            }
        )
        if (
            len(state) != EXPECTED_TOTAL_TENSOR_COUNT
            or len(existing_keys) != EXPECTED_EXISTING_TENSOR_COUNT
            or len(lora_keys) != EXPECTED_LORA_TENSOR_COUNT
        ):
            raise RuntimeError("trainable tensor counts violate the frozen 14+48 C1 contract")
        if not all(torch.is_tensor(tensor) and torch.isfinite(tensor).all().item() for tensor in state.values()):
            raise RuntimeError("checkpoint contains a non-tensor or non-finite trainable value")
        result["status"] = "passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _source_target_records(pair: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    image_root = Path(str(pair["final_image_root"]))
    for side in ("source", "target"):
        raw = Path(str(pair[side]["image"]))
        path = raw if raw.is_absolute() else image_root / raw
        records[side] = {
            "path": str(path.resolve()),
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
        }
    return records


def _build_preflight_payload(
    output_root: Path,
    profile_ids: Sequence[str],
    checkpoint_ids: Sequence[str],
) -> Dict[str, Any]:
    frozen_bindings, pair = base._verify_frozen_inputs()
    prompts = prompt_profile_manifest(pair, profile_ids)
    checkpoints = {key: _validate_checkpoint(key) for key in checkpoint_ids}
    images = _source_target_records(pair)
    failures: List[str] = []
    if any(item["status"] != "passed" for item in checkpoints.values()):
        failures.append("checkpoint_validation_failed")
    if not all(item["exists"] for item in images.values()):
        failures.append("source_or_target_missing")
    if len({item["sha256"] for item in prompts.values()}) != len(prompts):
        failures.append("prompt_hash_collision")
    legacy_text = render_prompt_profile(pair, "legacy_paired_v1")
    if "Target Prompt:" in legacy_text:
        failures.append("legacy_profile_semantics_changed")
    return {
        "schema_version": 1,
        "kind": "c1_prompt_ablation_preflight",
        "created_at_utc": _utc_now(),
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "pair_id": PAIR_ID,
        "output_root": str(output_root),
        "selection": {
            "profile_ids": list(profile_ids),
            "checkpoint_ids": list(checkpoint_ids),
            "seeds": list(SEEDS),
        },
        "inference_contract": {
            **base.FROZEN_INFERENCE,
            "prompt_mode": "versioned_prompt_profile",
            "source_image_is_input": True,
            "target_image_is_reference_only": True,
            "region_mask_is_not_inference_input": True,
        },
        "images": images,
        "prompts": prompts,
        "checkpoints": checkpoints,
        "frozen_input_bindings": frozen_bindings,
        "code_identity": {
            "script": {"path": str(SCRIPT_PATH), "sha256": _sha256(SCRIPT_PATH)},
            "prompt_profiles": {"path": str(PROMPT_MODULE_PATH), "sha256": _sha256(PROMPT_MODULE_PATH)},
            "base_driver": {"path": str(base.SCRIPT_PATH), "sha256": _sha256(base.SCRIPT_PATH)},
        },
    }


def _assert_empty_or_create(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"Output path is not a directory: {path}")
        if any(path.iterdir()):
            raise RuntimeError(f"Refusing to reuse non-empty output directory: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)


def _copy_references(pair: Mapping[str, Any], output_root: Path) -> Dict[str, str]:
    refs = output_root / "references"
    refs.mkdir(parents=True, exist_ok=True)
    image_root = Path(str(pair["final_image_root"]))
    copied: Dict[str, str] = {}
    for side in ("source", "target"):
        raw = Path(str(pair[side]["image"]))
        source = raw if raw.is_absolute() else image_root / raw
        target = refs / f"{side}{source.suffix.lower()}"
        shutil.copy2(source, target)
        copied[side] = str(target)
    return copied


def _worker_command(
    checkpoint_id: str,
    profile_id: str,
    worker_output_dir: Path,
    device: str,
    parent_run_id: str,
    parent_report_sha256: str,
) -> List[str]:
    return [
        sys.executable,
        str(SCRIPT_PATH),
        "--mode", "worker",
        "--checkpoint-id", checkpoint_id,
        "--profile-id", profile_id,
        "--worker-output-dir", str(worker_output_dir),
        "--device", device,
        "--parent-run-id", parent_run_id,
        "--parent-report-sha256", parent_report_sha256,
    ]


def _run_worker_subprocess(
    command: Sequence[str], log_path: Path, env: Mapping[str, str]
) -> Dict[str, Any]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command), cwd=str(REPO_ROOT), env=dict(env), stdout=handle,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
    return {
        "command": list(command),
        "returncode": int(completed.returncode),
        "duration_seconds": time.time() - started,
        "log_path": str(log_path),
    }


def _run_generate(args: argparse.Namespace) -> int:
    profile_ids, checkpoint_ids = _selection(args)
    output_root = (args.output_root or FORMAL_OUTPUT_ROOT).resolve()
    _assert_empty_or_create(output_root)
    pair = _load_pair()
    preflight = _build_preflight_payload(output_root, profile_ids, checkpoint_ids)
    preflight_path = output_root / "preflight_report.json"
    _atomic_write_json(preflight_path, preflight)
    if preflight["status"] != "passed":
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 2

    references = _copy_references(pair, output_root)
    prompt_manifest = {
        "schema_version": 1,
        "pair_id": PAIR_ID,
        "prompts": preflight["prompts"],
    }
    _atomic_write_json(output_root / "prompt_manifest.json", prompt_manifest)

    run_id = f"c1-prompt-ablation-{int(time.time())}-{os.getpid()}"
    parent_report_sha256 = _sha256(preflight_path)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "c1_prompt_ablation_report",
        "run_id": run_id,
        "created_at_utc": _utc_now(),
        "status": "running",
        "failures": [],
        "pair_id": PAIR_ID,
        "output_root": str(output_root),
        "references": references,
        "selection": preflight["selection"],
        "inference_contract": preflight["inference_contract"],
        "code_identity": preflight["code_identity"],
        "preflight_report_sha256": parent_report_sha256,
        "jobs": [],
        "review": {
            "status": "pending_blind_manual_review",
            "template": "blind_manual_review.json",
            "condition_key": "blind_review_key.json",
            "primary_success_rule": "right_ball_present AND NOT left_ball_present AND ball_count == 1",
        },
    }
    report_path = output_root / "prompt_ablation_report.json"
    _atomic_write_json(report_path, report)

    env = dict(os.environ)
    env[PARENT_ENV_FLAG] = "1"
    env[PARENT_RUN_ID_ENV] = run_id
    env[PARENT_REPORT_SHA_ENV] = parent_report_sha256
    env[WORKER_ALLOW_GPU_ENV] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    total_jobs = len(checkpoint_ids) * len(profile_ids)
    job_index = 0
    for checkpoint_id in checkpoint_ids:
        for profile_id in profile_ids:
            job_index += 1
            worker_dir = output_root / "generations" / checkpoint_id / profile_id
            worker_dir.mkdir(parents=True, exist_ok=False)
            log_path = output_root / "logs" / checkpoint_id / f"{profile_id}.log"
            command = _worker_command(
                checkpoint_id, profile_id, worker_dir, args.device,
                run_id, parent_report_sha256,
            )
            print(f"[{job_index}/{total_jobs}] {checkpoint_id} / {profile_id}", flush=True)
            execution = _run_worker_subprocess(command, log_path, env)
            job: Dict[str, Any] = {
                "checkpoint_id": checkpoint_id,
                "profile_id": profile_id,
                "output_dir": str(worker_dir),
                "execution": execution,
                "status": "failed",
            }
            result_path = worker_dir / "worker_result.json"
            if result_path.is_file():
                try:
                    worker_result = json.loads(result_path.read_text(encoding="utf-8"))
                    job["worker_result"] = worker_result
                    expected_outputs = {
                        str((worker_dir / f"seed_{seed}.png").resolve()): seed for seed in SEEDS
                    }
                    observed_outputs = {
                        str(Path(item["path"]).resolve()): item.get("seed")
                        for item in worker_result.get("outputs", [])
                        if isinstance(item, dict) and isinstance(item.get("path"), str)
                    }
                    worker_identity_ok = (
                        worker_result.get("checkpoint_id") == checkpoint_id
                        and worker_result.get("profile_id") == profile_id
                        and worker_result.get("checkpoint_sha256") == preflight["checkpoints"][checkpoint_id]["sha256"]
                        and worker_result.get("prompt_sha256") == preflight["prompts"][profile_id]["sha256"]
                        and observed_outputs == expected_outputs
                        and all(
                            Path(path).is_file()
                            and _sha256(Path(path)) == next(
                                item["png_sha256"] for item in worker_result["outputs"]
                                if str(Path(item["path"]).resolve()) == path
                            )
                            for path in expected_outputs
                        )
                    )
                    if (
                        execution["returncode"] == 0
                        and worker_result.get("status") == "passed"
                        and worker_identity_ok
                    ):
                        job["status"] = "passed"
                    elif not worker_identity_ok:
                        job["result_error"] = "worker identity/output verification failed"
                except Exception as exc:
                    job["result_error"] = f"{type(exc).__name__}: {exc}"
            else:
                job["result_error"] = "worker_result_missing"
            if job["status"] != "passed":
                report["failures"].append(f"{checkpoint_id}/{profile_id}")
            report["jobs"].append(job)
            _atomic_write_json(report_path, report)
            if job["status"] != "passed":
                report["status"] = "failed"
                report["finished_at_utc"] = _utc_now()
                _atomic_write_json(report_path, report)
                print(f"Worker failed; see {log_path}", file=sys.stderr)
                return 3

    report["status"] = "passed"
    report["finished_at_utc"] = _utc_now()
    _atomic_write_json(report_path, report)
    review_template, review_key = _build_manual_review_template(report, preflight["prompts"])
    _atomic_write_json(output_root / "blind_manual_review.json", review_template)
    _atomic_write_json(output_root / "blind_review_key.json", review_key)
    print(json.dumps({"status": "passed", "report": str(report_path)}, indent=2))
    return 0


def _worker_authorized(args: argparse.Namespace) -> None:
    checks = {
        "parent_flag": os.environ.get(PARENT_ENV_FLAG) == "1",
        "gpu_allow": os.environ.get(WORKER_ALLOW_GPU_ENV) == "1",
        "run_id": bool(args.parent_run_id) and args.parent_run_id == os.environ.get(PARENT_RUN_ID_ENV),
        "report_sha": bool(args.parent_report_sha256)
        and args.parent_report_sha256 == os.environ.get(PARENT_REPORT_SHA_ENV),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Unauthorized worker invocation: {checks}")


def _run_worker(args: argparse.Namespace) -> int:
    if args.checkpoint_id is None or args.profile_id is None or args.worker_output_dir is None:
        raise ValueError("worker requires --checkpoint-id, --profile-id, and --worker-output-dir")
    _worker_authorized(args)
    checkpoint_id = args.checkpoint_id
    profile_id = args.profile_id
    output_dir = args.worker_output_dir.resolve()
    pair = _load_pair()
    prompt = render_prompt_profile(pair, profile_id)
    spec = CHECKPOINTS[checkpoint_id]
    result: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(spec["path"]),
        "checkpoint_sha256": _sha256(Path(spec["path"])),
        "profile_id": profile_id,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "seeds": list(SEEDS),
        "outputs": [],
        "started_at_utc": _utc_now(),
    }
    result_path = output_dir / "worker_result.json"
    _atomic_write_json(result_path, result)
    try:
        from experiments.counterfactual_cot.training.gold40_overfit import (
            _build_model_for_gold40,
            _generate_one,
            _load_tokenizer,
            _make_inferencer,
        )
        from experiments.counterfactual_cot.training.lora_2step_smoke import (
            TARGET_MODULE_SUFFIXES,
            _iter_target_linear_names,
            inject_lora,
            set_trainable,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in generation worker")
        device = torch.device(args.device)
        inference_args = base._inference_namespace()
        random.seed(SEEDS[0])
        np.random.seed(SEEDS[0])
        torch.manual_seed(SEEDS[0])
        torch.cuda.manual_seed_all(SEEDS[0])
        tokenizer, special_tokens, num_new_tokens = _load_tokenizer(str(base.LLM_PATH))
        model, vae_model, base_checkpoint_report = _build_model_for_gold40(
            inference_args,
            device,
            tokenizer_size=len(tokenizer),
            num_new_tokens=num_new_tokens,
        )
        expected_missing = {"latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"}
        if (
            not isinstance(base_checkpoint_report, dict)
            or base_checkpoint_report.get("checkpoint_path") != str(base.EMA_PATH)
            or base_checkpoint_report.get("num_loaded_tensors") != base.EXPECTED_EMA_LOADED_TENSORS
            or base_checkpoint_report.get("num_missing_keys") != len(expected_missing)
            or set(base_checkpoint_report.get("missing_keys_sample", [])) != expected_missing
            or base_checkpoint_report.get("num_unexpected_keys") != 0
            or base_checkpoint_report.get("unexpected_keys_sample", []) != []
        ):
            raise RuntimeError(f"EMA loader compatibility contract mismatch: {base_checkpoint_report}")
        target_names = _iter_target_linear_names(model, inference_args.lora_layers)
        if (
            len(target_names) != base.EXPECTED_LORA_MODULES
            or len(target_names) != len(inference_args.lora_layers) * len(TARGET_MODULE_SUFFIXES)
        ):
            raise RuntimeError("LoRA target matrix is not exactly frozen L24-L27 x target suffixes")
        wrappers = inject_lora(
            model,
            target_names,
            inference_args.lora_r,
            inference_args.lora_alpha,
            inference_args.lora_dropout,
        )
        trainable_report = set_trainable(model, base.TRAINABLE_REGEX, wrappers)
        payload = torch.load(Path(spec["path"]), map_location="cpu", weights_only=False)
        load_result = base._strict_load_trainable(payload, model, "c1_lora", torch)
        model.eval()
        vae_model.eval()
        inferencer = _make_inferencer(
            model, vae_model, tokenizer, special_tokens, inference_args
        )
        for seed in SEEDS:
            output_path = output_dir / f"seed_{seed}.png"
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            _generate_one(
                inferencer,
                pair,
                output_path,
                inference_args,
                seed=seed,
                prompt=prompt,
            )
            record = base._image_record(output_path)
            record["seed"] = seed
            result["outputs"].append(record)
            _atomic_write_json(result_path, result)
        result["load_result"] = load_result
        result["base_checkpoint_report"] = base_checkpoint_report
        result["trainable_report"] = {
            "existing_count": len(trainable_report["existing_trainable_names"]),
            "lora_count": len(trainable_report["lora_trainable_names"]),
            "lora_target_count": len(target_names),
        }
        result["status"] = "passed"
        result["finished_at_utc"] = _utc_now()
        _atomic_write_json(result_path, result)
        return 0
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        result["finished_at_utc"] = _utc_now()
        _atomic_write_json(result_path, result)
        return 1


def _build_manual_review_template(
    report: Mapping[str, Any], prompts: Mapping[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    blind_items: List[Dict[str, Any]] = []
    key_items: List[Dict[str, Any]] = []
    for job in report["jobs"]:
        worker_result = job.get("worker_result", {})
        for output in worker_result.get("outputs", []):
            item_id = f"review_{secrets.token_hex(8)}"
            blind_items.append(
                {
                    "item_id": item_id,
                    "image_path": output["path"],
                    "image_png_sha256": output["png_sha256"],
                    "right_ball_present": None,
                    "left_ball_present": None,
                    "ball_count": None,
                    "ball_color_preserved": None,
                    "sofa_position_preserved": None,
                    "background_preserved": None,
                    "viewpoint_preserved": None,
                    "lighting_preserved": None,
                    "image_quality_valid": None,
                    "notes": "",
                }
            )
            key_items.append(
                {
                    "item_id": item_id,
                    "checkpoint_id": job["checkpoint_id"],
                    "profile_id": job["profile_id"],
                    "profile_purpose": prompts[job["profile_id"]]["purpose"],
                    "seed": output["seed"],
                    "image_path": output["path"],
                    "image_png_sha256": output["png_sha256"],
                }
            )
    random.Random(20260902).shuffle(blind_items)
    shared = {
        "schema_version": 1,
        "pair_id": PAIR_ID,
        "run_id": report["run_id"],
        "generation_report_sha256": _sha256(Path(report["output_root"]) / "prompt_ablation_report.json"),
        "expected_item_count": len(blind_items),
    }
    review = {
        **shared,
        "kind": "c1_prompt_ablation_blind_manual_review",
        "reviewer": None,
        "reviewed_at_utc": None,
        "instructions": {
            "primary_success_rule": "right_ball_present == true AND left_ball_present == false AND ball_count == 1",
            "do_not_count_as_success": "A new right ball while the original left ball remains.",
            "review_policy": "Fill every null score plus reviewer/reviewed_at_utc; do not change identity/path/hash fields.",
        },
        "items": blind_items,
    }
    key = {**shared, "kind": "c1_prompt_ablation_blind_key", "items": key_items}
    return review, key


def _run_preflight(args: argparse.Namespace) -> int:
    profile_ids, checkpoint_ids = _selection(args)
    output_root = (args.output_root or PREFLIGHT_OUTPUT_ROOT).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report = _build_preflight_payload(output_root, profile_ids, checkpoint_ids)
    report_path = output_root / f"preflight_report_{int(time.time())}.json"
    if report_path.exists():
        raise RuntimeError(f"Refusing to overwrite preflight report: {report_path}")
    _atomic_write_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path), "failures": report["failures"]}, indent=2))
    return 0 if report["status"] == "passed" else 2


def main() -> int:
    args = _parse_args()
    if args.mode == "preflight":
        return _run_preflight(args)
    if args.mode == "generate":
        return _run_generate(args)
    return _run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
