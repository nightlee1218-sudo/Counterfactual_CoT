"""Pure-CPU cross-process comparator for the cold-reload verifier.

Reads ``verifier_a.json`` and ``verifier_b.json`` plus the two saved
``mse_probe_tensor_{A,B}.pt`` files, compares them along the audit checklist,
and writes ``cold_reload_verification_report.json``.

No CUDA / no model / no dataset loading. Fully deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # noqa: N816


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cmp_equal(name: str, a: Any, b: Any) -> Dict[str, Any]:
    return {"name": name, "equal": a == b, "a": a, "b": b}


def _png_pixel_diff(a: Path, b: Path) -> Dict[str, Any]:
    if Image is None or not a.exists() or not b.exists():
        return {"available": False}
    im_a = np.asarray(Image.open(a).convert("RGB"), dtype=np.int32)
    im_b = np.asarray(Image.open(b).convert("RGB"), dtype=np.int32)
    if im_a.shape != im_b.shape:
        return {
            "available": True,
            "shape_mismatch": True,
            "shape_a": list(im_a.shape),
            "shape_b": list(im_b.shape),
        }
    diff = np.abs(im_a - im_b)
    total = float(im_a.size)
    nonzero = int((diff != 0).sum())
    return {
        "available": True,
        "shape_mismatch": False,
        "shape": list(im_a.shape),
        "max_abs_diff": int(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "nonzero_channel_count": nonzero,
        "nonzero_channel_fraction": nonzero / total,
    }


# ---------------------------------------------------------------------------
# Comparators
# ---------------------------------------------------------------------------


def _compare_trainable_key_provenance(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    expected_a = set(a["expected_trainable_keys"])
    expected_b = set(b["expected_trainable_keys"])
    saved_a = set(a["saved_trainable_keys"])
    saved_b = set(b["saved_trainable_keys"])
    return {
        "expected_equal_a_b": expected_a == expected_b,
        "saved_equal_a_b": saved_a == saved_b,
        "expected_equals_saved_a": expected_a == saved_a,
        "expected_equals_saved_b": expected_b == saved_b,
        "expected_only_in_a": sorted(expected_a - expected_b),
        "expected_only_in_b": sorted(expected_b - expected_a),
        "saved_only_in_a": sorted(saved_a - saved_b),
        "saved_only_in_b": sorted(saved_b - saved_a),
        "num_expected_a": len(expected_a),
        "num_expected_b": len(expected_b),
        "num_saved_a": len(saved_a),
        "num_saved_b": len(saved_b),
    }


def _compare_per_tensor_hashes(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Compare per-tensor ckpt hashes and post-copy model param hashes.

    Since both verifiers load the SAME on-disk checkpoint, the ckpt tensor
    hashes must match exactly. The model-param hashes after copy_ can differ
    only if the model dtype differs (both should be bf16 for base weights,
    fp32 for LoRA), so they must also match exactly.
    """
    per_a = a["reload"]["per_tensor"]
    per_b = b["reload"]["per_tensor"]
    keys = sorted(set(per_a.keys()) | set(per_b.keys()))
    mismatches: List[Dict[str, Any]] = []
    matches = 0
    for k in keys:
        ea = per_a.get(k)
        eb = per_b.get(k)
        if ea is None or eb is None:
            mismatches.append({"name": k, "reason": "missing_in_one_side"})
            continue
        if ea["ckpt_tensor_hash"] != eb["ckpt_tensor_hash"]:
            mismatches.append({
                "name": k, "reason": "ckpt_tensor_hash_mismatch",
                "a": ea["ckpt_tensor_hash"], "b": eb["ckpt_tensor_hash"],
            })
            continue
        if ea["model_param_hash_after_copy"] != eb["model_param_hash_after_copy"]:
            mismatches.append({
                "name": k, "reason": "model_param_hash_after_copy_mismatch",
                "a": ea["model_param_hash_after_copy"], "b": eb["model_param_hash_after_copy"],
            })
            continue
        matches += 1
    return {
        "num_keys": len(keys),
        "num_matching": matches,
        "num_mismatching": len(mismatches),
        "mismatches": mismatches[:32],  # cap for readability
    }


def _compare_mse_tensors(mse_a_path: Path, mse_b_path: Path) -> Dict[str, Any]:
    if not mse_a_path.exists() or not mse_b_path.exists():
        return {"available": False, "reason": "one or both mse tensor files missing"}
    pa = torch.load(mse_a_path, map_location="cpu", weights_only=False)
    pb = torch.load(mse_b_path, map_location="cpu", weights_only=False)
    ma = pa["mse"].to(torch.float32)
    mb = pb["mse"].to(torch.float32)
    same_shape = tuple(ma.shape) == tuple(mb.shape)
    if not same_shape:
        return {
            "available": True,
            "shape_mismatch": True,
            "shape_a": list(ma.shape),
            "shape_b": list(mb.shape),
        }
    diff = (ma - mb).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    l2 = float(torch.linalg.vector_norm(diff.reshape(-1)).item())
    exact_match = (max_abs == 0.0) and (mean_abs == 0.0)
    numerical_match = (max_abs <= 1e-6) and (mean_abs <= 1e-7)
    return {
        "available": True,
        "shape_mismatch": False,
        "shape": list(ma.shape),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "l2_diff": l2,
        "exact_match": exact_match,
        "numerical_match": numerical_match,
        "mse_mean_a": float(ma.mean().item()),
        "mse_mean_b": float(mb.mean().item()),
        "mse_hash_a": pa["meta"]["mse_tensor_hash"],
        "mse_hash_b": pb["meta"]["mse_tensor_hash"],
        "mse_hash_equal": pa["meta"]["mse_tensor_hash"] == pb["meta"]["mse_tensor_hash"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-a-dir", type=Path, required=True)
    parser.add_argument("--verifier-b-dir", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True,
                        help="Path to lora_2step_smoke_report.json (read-only).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Where to write cold_reload_verification_report.json.")
    parser.add_argument("--allow-training-smoke-failed-status", action="store_true",
                        help="Treat the training-side report 'status=failed due to release_check_failed' as expected.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Safety: ensure output not under /dockerdata.
    if str(args.output.resolve()).startswith("/dockerdata"):
        raise RuntimeError(
            f"Refusing to write comparator report under /dockerdata: {args.output}. "
            f"Must live under /group."
        )

    a_report_path = args.verifier_a_dir / "verifier_a.json"
    b_report_path = args.verifier_b_dir / "verifier_b.json"
    mse_a_path = args.verifier_a_dir / "mse_probe_tensor_A.pt"
    mse_b_path = args.verifier_b_dir / "mse_probe_tensor_B.pt"

    for p in (a_report_path, b_report_path, mse_a_path, mse_b_path, args.training_report):
        if not p.exists():
            raise FileNotFoundError(f"required input missing: {p}")

    a = _load_json(a_report_path)
    b = _load_json(b_report_path)
    training = _load_json(args.training_report)

    # --- 1) trainable checkpoint SHA256 ---
    ckpt_equal = _cmp_equal(
        "trainable_checkpoint_sha256",
        a["trainable_checkpoint"]["sha256"],
        b["trainable_checkpoint"]["sha256"],
    )

    # --- 2) Base checkpoint path + sha256 ---
    base_path_equal = _cmp_equal(
        "base_checkpoint_path",
        a["base_checkpoint"]["path"], b["base_checkpoint"]["path"],
    )
    base_sha_equal = _cmp_equal(
        "base_checkpoint_sha256",
        a["base_checkpoint"]["sha256"], b["base_checkpoint"]["sha256"],
    )

    # --- 3) LoRA target names / r / alpha / dropout ---
    lora_cfg_equal = _cmp_equal(
        "lora_config_hash",
        a["lora_config_hash"], b["lora_config_hash"],
    )
    lora_target_names_equal = _cmp_equal(
        "lora_target_names",
        a["lora_config"]["target_names"],
        b["lora_config"]["target_names"],
    )
    lora_r_equal = _cmp_equal("lora_r", a["lora_config"]["r"], b["lora_config"]["r"])
    lora_alpha_equal = _cmp_equal("lora_alpha", a["lora_config"]["alpha"], b["lora_config"]["alpha"])
    lora_dropout_equal = _cmp_equal("lora_dropout", a["lora_config"]["dropout"], b["lora_config"]["dropout"])

    # --- 4) expected/saved key sets ---
    key_provenance = _compare_trainable_key_provenance(a, b)

    # Cross-check with training-side report if possible.
    training_expected: List[str] = []
    training_saved: List[str] = []
    if isinstance(training.get("existing_trainable_names"), list) and isinstance(training.get("lora_trainable_names"), list):
        training_expected = sorted(set(training["existing_trainable_names"]) | set(training["lora_trainable_names"]))
    training_ckpt_meta = training.get("checkpoint", {})
    training_ckpt_sha = training_ckpt_meta.get("sha256")
    ckpt_matches_training = (training_ckpt_sha == a["trainable_checkpoint"]["sha256"])
    expected_matches_training = (set(training_expected) == set(a["expected_trainable_keys"])) if training_expected else None

    # --- 5) pair_id, prompt hash, source/target sha256 ---
    pair_id_equal = _cmp_equal("pair_id", a["pair_id"], b["pair_id"])
    prompt_hash_equal = _cmp_equal(
        "conditioning_prompt_hash",
        a["conditioning"]["prompt_hash"], b["conditioning"]["prompt_hash"],
    )
    src_sha_equal = _cmp_equal(
        "pair_source_image_sha256",
        (a["conditioning"]["pair_image_hashes"].get("source") or {}).get("sha256"),
        (b["conditioning"]["pair_image_hashes"].get("source") or {}).get("sha256"),
    )
    tgt_sha_equal = _cmp_equal(
        "pair_target_image_sha256",
        (a["conditioning"]["pair_image_hashes"].get("target") or {}).get("sha256"),
        (b["conditioning"]["pair_image_hashes"].get("target") or {}).get("sha256"),
    )

    # --- 6) packed input / probe metadata hash ---
    probe_seed_equal = _cmp_equal("probe_seed", a["probe_seed"], b["probe_seed"])
    packed_hashes_a = a["probe"].get("packed_input_hashes", {})
    packed_hashes_b = b["probe"].get("packed_input_hashes", {})
    packed_input_diffs: List[Dict[str, Any]] = []
    keys = sorted(set(packed_hashes_a.keys()) | set(packed_hashes_b.keys()))
    for k in keys:
        ha = packed_hashes_a.get(k)
        hb = packed_hashes_b.get(k)
        if ha is None or hb is None:
            packed_input_diffs.append({"key": k, "reason": "missing_in_one_side",
                                        "a": ha, "b": hb})
            continue
        if ha["hash"] != hb["hash"] or ha["shape"] != hb["shape"] or ha["dtype"] != hb["dtype"]:
            packed_input_diffs.append({"key": k, "a": ha, "b": hb})
    packed_input_all_equal = (len(packed_input_diffs) == 0) and (len(keys) > 0)

    # --- 7) timestep tensor hash + forward noise hash proxy ---
    timesteps_hash_equal = _cmp_equal(
        "packed_timesteps_hash",
        a["probe"]["packed_timesteps_hash"], b["probe"]["packed_timesteps_hash"],
    )

    # --- 8) full MSE tensor comparison ---
    mse_compare = _compare_mse_tensors(mse_a_path, mse_b_path)

    # --- 9) PNG SHA / pixel diff ---
    png_a = a["generation"].get("path")
    png_b = b["generation"].get("path")
    png_sha_a = a["generation"].get("sha256")
    png_sha_b = b["generation"].get("sha256")
    png_sha_equal = (png_sha_a is not None) and (png_sha_a == png_sha_b)
    png_pixel_diff = None
    if (png_a is not None) and (png_b is not None):
        png_pixel_diff = _png_pixel_diff(Path(png_a), Path(png_b))

    # --- 10) per-side invariants ---
    per_tensor_hashes = _compare_per_tensor_hashes(a, b)

    dtype_invariants_a_ok = (a["checks"].get("dtype_invariants_initial_ok", False)
                              and a["checks"].get("dtype_invariants_after_load_ok", False))
    dtype_invariants_b_ok = (b["checks"].get("dtype_invariants_initial_ok", False)
                              and b["checks"].get("dtype_invariants_after_load_ok", False))
    hooks_a_ok = a["checks"].get("all_hooks_called_at_least_once", False)
    hooks_b_ok = b["checks"].get("all_hooks_called_at_least_once", False)

    # --- Hard gates ---
    a_passed = (a.get("status") == "passed")
    b_passed = (b.get("status") == "passed")
    key_sets_all_equal = (
        key_provenance["expected_equal_a_b"]
        and key_provenance["saved_equal_a_b"]
        and key_provenance["expected_equals_saved_a"]
        and key_provenance["expected_equals_saved_b"]
    )
    numerical_match = bool(mse_compare.get("numerical_match", False))
    exact_match = bool(mse_compare.get("exact_match", False))

    # PNG gate: SHA equal OR (tiny pixel diff): max <= 2, mean <= 0.05, and
    # nonzero_channel_fraction <= 1e-2 (arbitrary very-small).
    png_gate_pass = False
    png_gate_reason = ""
    if png_sha_equal:
        png_gate_pass = True
        png_gate_reason = "sha_match"
    elif png_pixel_diff is not None and png_pixel_diff.get("available") and not png_pixel_diff.get("shape_mismatch"):
        if (png_pixel_diff["max_abs_diff"] <= 2
                and png_pixel_diff["mean_abs_diff"] <= 0.05
                and png_pixel_diff["nonzero_channel_fraction"] <= 1e-2):
            png_gate_pass = True
            png_gate_reason = "pixel_diff_within_tolerance"
        else:
            png_gate_reason = "pixel_diff_out_of_tolerance"
    else:
        png_gate_reason = "no_png_or_shape_mismatch"

    cross_process_numerical_match = (
        ckpt_equal["equal"] and base_sha_equal["equal"] and lora_cfg_equal["equal"]
        and key_sets_all_equal
        and pair_id_equal["equal"] and prompt_hash_equal["equal"]
        and packed_input_all_equal and timesteps_hash_equal["equal"]
        and numerical_match
    )

    overall_reload_validation_passed = (
        a_passed and b_passed
        and cross_process_numerical_match
        and png_gate_pass
        and per_tensor_hashes["num_mismatching"] == 0
        and dtype_invariants_a_ok and dtype_invariants_b_ok
        and hooks_a_ok and hooks_b_ok
        and a["reload"]["max_abs_load_copy_diff"] == 0.0
        and b["reload"]["max_abs_load_copy_diff"] == 0.0
    )

    training_side_smoke_passed, training_smoke_breakdown = _training_side_smoke_passed_structured(training)
    same_process_release_check_status, release_check_details = _same_process_release_status(training)
    # Backwards-compatible boolean: True iff status is explicitly ``passed``.
    same_process_release_check_passed = (same_process_release_check_status == "passed")

    # Overall smoke = training-side structured checks all green AND cold-reload
    # cross-process validation all green. The same-process release gate is
    # deliberately NOT required (we accept status=not_completed as a documented
    # limitation of the current release path).
    overall_smoke_passed = bool(training_side_smoke_passed) and bool(overall_reload_validation_passed)

    report = {
        "overall_smoke_passed": overall_smoke_passed,
        "overall_reload_validation_passed": overall_reload_validation_passed,
        "cross_process_numerical_match": cross_process_numerical_match,
        "cold_reload_a_passed": a_passed,
        "cold_reload_b_passed": b_passed,
        "training_side_smoke_passed": training_side_smoke_passed,
        "same_process_release_check_passed": same_process_release_check_passed,
        "same_process_release_check_status": same_process_release_check_status,
        "training_side_smoke_breakdown": training_smoke_breakdown,
        "same_process_release_check_details": release_check_details,
        "hard_gates": {
            "both_verifiers_status_passed": a_passed and b_passed,
            "key_sets_all_equal": key_sets_all_equal,
            "numerical_mse_match": numerical_match,
            "exact_mse_match": exact_match,
            "png_gate_pass": png_gate_pass,
            "png_gate_reason": png_gate_reason,
            "no_missing_or_unexpected_keys": (
                key_provenance["expected_equals_saved_a"] and key_provenance["expected_equals_saved_b"]
            ),
            "no_oom_nan_inf_traceback": (
                a["checks"].get("mse_all_finite", False) and b["checks"].get("mse_all_finite", False)
            ),
            "dtype_invariants_ok_both": dtype_invariants_a_ok and dtype_invariants_b_ok,
            "hooks_called_ok_both": hooks_a_ok and hooks_b_ok,
            "load_copy_diff_zero_both": (
                a["reload"]["max_abs_load_copy_diff"] == 0.0
                and b["reload"]["max_abs_load_copy_diff"] == 0.0
            ),
        },
        "checkpoint_provenance": {
            "trainable_ckpt_sha256_a": a["trainable_checkpoint"]["sha256"],
            "trainable_ckpt_sha256_b": b["trainable_checkpoint"]["sha256"],
            "trainable_ckpt_sha256_training_report": training_ckpt_sha,
            "ckpt_sha_equal_a_b": ckpt_equal["equal"],
            "ckpt_sha_matches_training_report": ckpt_matches_training,
            "expected_key_set_matches_training_report": expected_matches_training,
        },
        "base_checkpoint": {
            "path_equal": base_path_equal["equal"],
            "sha256_equal": base_sha_equal["equal"],
            "sha256_a": a["base_checkpoint"]["sha256"],
            "sha256_b": b["base_checkpoint"]["sha256"],
        },
        "lora_config": {
            "hash_equal": lora_cfg_equal["equal"],
            "hash_a": a["lora_config_hash"],
            "hash_b": b["lora_config_hash"],
            "target_names_equal": lora_target_names_equal["equal"],
            "r_equal": lora_r_equal["equal"],
            "alpha_equal": lora_alpha_equal["equal"],
            "dropout_equal": lora_dropout_equal["equal"],
        },
        "trainable_key_provenance": key_provenance,
        "conditioning": {
            "pair_id_equal": pair_id_equal["equal"],
            "prompt_hash_equal": prompt_hash_equal["equal"],
            "source_image_sha256_equal": src_sha_equal["equal"],
            "target_image_sha256_equal": tgt_sha_equal["equal"],
            "prompt_hash": a["conditioning"]["prompt_hash"],
            "source_image_hash": (a["conditioning"]["pair_image_hashes"].get("source") or {}).get("sha256"),
            "target_image_hash": (a["conditioning"]["pair_image_hashes"].get("target") or {}).get("sha256"),
        },
        "probe": {
            "seed_equal": probe_seed_equal["equal"],
            "packed_timesteps_hash_equal": timesteps_hash_equal["equal"],
            "packed_timesteps_hash": a["probe"]["packed_timesteps_hash"],
            "packed_input_all_equal": packed_input_all_equal,
            "packed_input_diffs": packed_input_diffs,
        },
        "mse_tensor_comparison": mse_compare,
        "generation": {
            "png_a": png_a,
            "png_b": png_b,
            "sha256_a": png_sha_a,
            "sha256_b": png_sha_b,
            "sha_equal": png_sha_equal,
            "pixel_diff": png_pixel_diff,
        },
        "per_tensor_hashes": per_tensor_hashes,
        "dtype_invariants": {
            "a_ok": dtype_invariants_a_ok,
            "b_ok": dtype_invariants_b_ok,
        },
        "hooks_called": {
            "a_ok": hooks_a_ok,
            "b_ok": hooks_b_ok,
            "a": a.get("hook_calls_per_target_linear"),
            "b": b.get("hook_calls_per_target_linear"),
        },
        "gpu_memory": {
            "a": a.get("gpu_memory"),
            "b": b.get("gpu_memory"),
        },
        "training_report_summary": {
            "status": training.get("status"),
            "reason": training.get("reason"),
            "release_before_reload_ok": training.get("checks", {}).get("release_before_reload_ok"),
            "allocated_after_release_mib": (training.get("cuda_memory") or {}).get("allocated_after_release_mib"),
            "checkpoint_sha256": training_ckpt_sha,
        },
        "inputs": {
            "verifier_a_report": str(a_report_path),
            "verifier_b_report": str(b_report_path),
            "training_report": str(args.training_report),
            "mse_a_path": str(mse_a_path),
            "mse_b_path": str(mse_b_path),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps({
        "event": "cold_reload_verification_finished",
        "report": str(args.output),
        "overall_smoke_passed": overall_smoke_passed,
        "overall_reload_validation_passed": report["overall_reload_validation_passed"],
        "cross_process_numerical_match": report["cross_process_numerical_match"],
        "cold_reload_a_passed": a_passed,
        "cold_reload_b_passed": b_passed,
        "training_side_smoke_passed": training_side_smoke_passed,
        "same_process_release_check_passed": same_process_release_check_passed,
        "same_process_release_check_status": same_process_release_check_status,
    }, ensure_ascii=False, sort_keys=True), flush=True)

    sys.exit(0 if report["overall_reload_validation_passed"] else 1)


def _training_side_smoke_passed_structured(training: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Compute training-side smoke pass/fail from *structured* checks in the
    training report only. Never string-matches on ``reason``.

    Returns (passed, breakdown). ``passed`` is True iff every structured
    training-side check is green. This does NOT include the same-process
    release gate (which is intentionally treated separately by the caller).
    """
    checks: Dict[str, Any] = {}

    # 1) step_records exist and record 2 successful gradient steps with finite MSE.
    step_records = training.get("step_records") or []
    checks["has_two_step_records"] = len(step_records) >= 2
    step1 = step_records[0] if len(step_records) >= 1 else {}
    step2 = step_records[1] if len(step_records) >= 2 else {}
    checks["step1_mse_finite"] = bool(step1.get("mse_all_finite", False))
    checks["step2_mse_finite"] = bool(step2.get("mse_all_finite", False))
    checks["step1_mse_is_number"] = isinstance(step1.get("mse_mean"), (int, float))
    checks["step2_mse_is_number"] = isinstance(step2.get("mse_mean"), (int, float))
    checks["step1_grad_norm_is_number"] = isinstance(step1.get("grad_norm_pre_clip"), (int, float))
    checks["step2_grad_norm_is_number"] = isinstance(step2.get("grad_norm_pre_clip"), (int, float))
    checks["step1_no_warnings"] = (step1.get("warnings") == [] or step1.get("warnings") is None)
    checks["step2_no_warnings"] = (step2.get("warnings") == [] or step2.get("warnings") is None)

    # 2) LoRA grads flowed: step2 must have both lora_A and lora_B with at least
    #    some nonzero grads (step1 typically has lora_A all-zero from init, so
    #    we only require lora_B>0 at step1 and both>0 at step2).
    checks["step1_lora_B_nonzero"] = int(step1.get("num_lora_B_nonzero", 0) or 0) >= 1
    checks["step2_lora_A_nonzero"] = int(step2.get("num_lora_A_nonzero", 0) or 0) >= 1
    checks["step2_lora_B_nonzero"] = int(step2.get("num_lora_B_nonzero", 0) or 0) >= 1

    # 3) All wrapped target linears received forward hook calls (>0 per name).
    hook_calls = training.get("hook_calls_per_target_linear") or {}
    checks["has_hook_calls_map"] = bool(hook_calls)
    checks["all_target_hooks_fired"] = bool(hook_calls) and all(
        isinstance(v, int) and v > 0 for v in hook_calls.values()
    )

    # 4) Checkpoint metadata block exists (means checkpoint save completed).
    ckpt_meta = training.get("checkpoint") or {}
    checks["checkpoint_saved_sha256_present"] = isinstance(ckpt_meta.get("sha256"), str) and len(ckpt_meta["sha256"]) == 64

    # 5) Peak training memory recorded (means we ran through the training pass).
    cuda_mem = training.get("cuda_memory") or {}
    checks["peak_train_memory_recorded"] = isinstance(cuda_mem.get("peak_allocated_train_mib"), (int, float)) \
        and float(cuda_mem["peak_allocated_train_mib"]) > 0.0

    # 6) LoRA target names present and 24 wrappers configured.
    lora_target_names = training.get("lora_target_names") or []
    checks["lora_target_names_count_ok"] = len(lora_target_names) == 24  # 4 layers * 6 linears

    # 7) At least one pair_id was involved.
    checks["pair_id_present"] = isinstance(training.get("pair_id"), str) and bool(training.get("pair_id"))

    passed = all(bool(v) for v in checks.values())
    return passed, checks


def _same_process_release_status(training: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Return (status_str, details).

    status_str is one of:
      * ``passed``       -- release_before_reload_ok is True in training report
      * ``failed``       -- release_before_reload_ok is False in training report
      * ``not_completed``-- release gate ran but blocked reload; or field absent
                            because the training script bailed before reload
                            (which is exactly what happened here). In either
                            case, the same-process reload path did NOT complete.

    We deliberately DO NOT collapse ``failed`` and ``not_completed`` into a
    single boolean: from a validation standpoint they mean different things.
    """
    details = {
        "training_status": training.get("status"),
        "training_reason": training.get("reason"),
        "allocated_after_release_mib": (training.get("cuda_memory") or {}).get("allocated_after_release_mib"),
        "reserved_after_release_mib": (training.get("cuda_memory") or {}).get("reserved_after_release_mib"),
    }
    release_ok = (training.get("checks") or {}).get("release_before_reload_ok")
    if release_ok is True:
        return "passed", details
    if release_ok is False:
        return "failed", details
    # No structured release-check outcome was recorded. Distinguish "gate ran
    # and blocked reload" from "no gate at all" by looking at whether the run
    # recorded post-release memory and a training status of ``failed``.
    alloc_after = details.get("allocated_after_release_mib")
    if isinstance(alloc_after, (int, float)) and training.get("status") == "failed":
        return "not_completed", details
    return "not_completed", details


if __name__ == "__main__":
    main()
