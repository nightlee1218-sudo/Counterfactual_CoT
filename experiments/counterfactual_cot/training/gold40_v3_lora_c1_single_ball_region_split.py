"""C1 single-ball LoRA training with split edit/preserve region loss.

This is an isolated experiment wrapper around the frozen C1 training driver. It
keeps the model, packing, RNG trajectory, optimizer, checkpoint, and dry-run
contracts unchanged while replacing only the reduction of the target-image MSE.

The manifest mask is an automatic candidate difference mask, not semantic ground
truth. For this fixed C1 pair it is intersected with two audited ROIs covering the
old and new ball locations. Full training requires an explicit acknowledgement.
"""

from __future__ import annotations

import gc
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_PATH = Path(__file__).resolve()
BAGEL_ROOT = SCRIPT_PATH.parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from experiments.counterfactual_cot.training import fixed_deterministic_eval as fixed_eval_module
from experiments.counterfactual_cot.training import gold40_lora as gold40_lora_module
from experiments.counterfactual_cot.training import gold40_overfit as gold40_overfit_module
from experiments.counterfactual_cot.training import gold40_v3_lora_c1_single_ball as legacy
from experiments.counterfactual_cot.training import lora_2step_smoke as lora_smoke_module
from experiments.counterfactual_cot.training.gold40_overfit import (
    _autocast_for_model,
    _prepare_batch,
)


BASE_OUTPUT_STEM = (
    BAGEL_ROOT
    / "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_c1_single_ball_region_split_v1"
)
ACK_FLAG = "--acknowledge-audit-derived-mask"
EDIT_LOSS_WEIGHT = 0.5
PRESERVE_LOSS_WEIGHT = 0.5
TOKEN_GRID = (16, 16)
EXPECTED_TARGET_TOKENS = TOKEN_GRID[0] * TOKEN_GRID[1]
FIXED_EVAL_SEED = legacy.C1_PAIR_SEED
EXPECTED_SOURCE_SHA256 = "71d756241d84721b7885c68bd4da650bf4873277d48be363fb6990d74ace1539"
EXPECTED_TARGET_SHA256 = "8500c4401b5dd17ca7aa06d78bf9f3e1dbab3ceb67904c0a2903e034477f9dd1"
CANDIDATE_THRESHOLD = 24
CANDIDATE_DILATION_PX = 9
MIN_FOCUSED_EDIT_RATIO = 0.005
MAX_FOCUSED_EDIT_RATIO = 0.15

# PIL boxes are (left, upper, right-exclusive, lower-exclusive). These fixed ROIs
# contain the old ball plus shadow and the new ball plus shadow, respectively.
EDIT_ROIS: Tuple[Tuple[int, int, int, int], ...] = (
    (40, 590, 270, 850),
    (810, 610, 1024, 860),
)
REGION_STRATEGY = "candidate_difference_mask_intersect_old_and_new_ball_rois"
REGION_CONFIG_KEYS = frozenset(
    {
        "region_mask_path",
        "region_mask_sha256",
        "region_mask_strategy",
        "region_edit_rois",
        "region_token_grid",
        "edit_loss_weight",
        "preserve_loss_weight",
        "acknowledge_audit_derived_mask",
    }
)

_REGION_SPEC: Dict[str, Any] = {}
_EDIT_TOKEN_WEIGHTS_CPU: torch.Tensor | None = None
_LAST_REGION_SCALARS: Dict[str, float] = {}
_LAST_MODEL_MSE_SHAPE: List[int] = []
_FIXED_EVAL_CONTEXT: Dict[str, Any] = {}
_CURRENT_ARGS: Any = None
_DRY_RUN_REPORT_BINDING: Dict[str, Any] = {}
_ORIGINAL_APPEND_JSONL = legacy._append_jsonl
_ORIGINAL_ATOMIC_WRITE_JSON = legacy._atomic_write_json
_ORIGINAL_BUILD_GOLD40_LOADER = legacy._build_gold40_loader
_ORIGINAL_CONFIG_DIFF = legacy._micro4_vs_formal_b_diff


def _sha256_file(path: Path) -> str:
    return legacy._sha256_file(path)


def _resolve_c1_paths() -> Tuple[Path, Path, Path]:
    records = legacy.load_jsonl(legacy.MICRO4_MANIFEST)
    if len(records) != 1 or records[0].get("pair_id") != legacy.C1_PAIR_ID:
        raise RuntimeError("C1 manifest no longer contains exactly the frozen C1 pair")
    pair = records[0]
    root = Path(pair["final_image_root"])
    source_path = root / pair["source"]["image"]
    target_path = root / pair["target"]["image"]
    mask_value = pair.get("target", {}).get("mask")
    if not isinstance(mask_value, str) or not mask_value:
        raise RuntimeError("C1 target.mask is missing")
    mask_path = root / mask_value
    return source_path.resolve(), target_path.resolve(), mask_path.resolve()


def _load_region_spec() -> Tuple[Dict[str, Any], torch.Tensor]:
    source_path, target_path, mask_path = _resolve_c1_paths()
    for label, path in (
        ("source", source_path),
        ("target", target_path),
        ("candidate mask", mask_path),
    ):
        if not path.is_file():
            raise RuntimeError(f"C1 {label} file is missing: {path}")

    source_sha256 = _sha256_file(source_path)
    target_sha256 = _sha256_file(target_path)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"C1 source SHA256 mismatch: {source_sha256} != {EXPECTED_SOURCE_SHA256}"
        )
    if target_sha256 != EXPECTED_TARGET_SHA256:
        raise RuntimeError(
            f"C1 target SHA256 mismatch: {target_sha256} != {EXPECTED_TARGET_SHA256}"
        )

    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.int16)
        source_size = image.size
    with Image.open(target_path) as image:
        target = np.asarray(image.convert("RGB"), dtype=np.int16)
        target_size = image.size
    with Image.open(mask_path) as image:
        if image.mode != "L":
            raise RuntimeError(f"C1 candidate mask must have native mode L, got {image.mode!r}")
        mask_size = image.size
        candidate = np.asarray(image, dtype=np.uint8)

    if source_size != target_size or target_size != mask_size:
        raise RuntimeError(
            "C1 source/target/mask size mismatch: "
            f"source={source_size}, target={target_size}, mask={mask_size}"
        )
    expected_binary = (
        np.max(np.abs(source - target), axis=2) > CANDIDATE_THRESHOLD
    ).astype(np.uint8) * 255
    expected_mask = Image.fromarray(expected_binary, mode="L")
    expected_mask = expected_mask.filter(ImageFilter.MaxFilter(CANDIDATE_DILATION_PX))
    expected_mask = expected_mask.filter(ImageFilter.MedianFilter(3))
    expected_candidate = np.asarray(expected_mask, dtype=np.uint8)
    if not np.array_equal(candidate, expected_candidate):
        mismatch_count = int(np.count_nonzero(candidate != expected_candidate))
        raise RuntimeError(
            "C1 candidate mask is not the deterministic threshold=24/dilation=9 mask "
            f"derived from the fixed source/target assets; mismatched_pixels={mismatch_count}"
        )

    width, height = mask_size
    roi_union = np.zeros((height, width), dtype=bool)
    for index, (left, upper, right, lower) in enumerate(EDIT_ROIS):
        if not (0 <= left < right <= width and 0 <= upper < lower <= height):
            raise RuntimeError(f"Invalid edit ROI {index}: {(left, upper, right, lower)}")
        roi_union[upper:lower, left:right] = True

    candidate_binary = candidate > 0
    focused_binary = candidate_binary & roi_union
    if not focused_binary.any() or focused_binary.all():
        raise RuntimeError("Focused C1 edit mask must be non-empty and non-full")
    focused_ratio = float(focused_binary.mean())
    if not MIN_FOCUSED_EDIT_RATIO <= focused_ratio <= MAX_FOCUSED_EDIT_RATIO:
        raise RuntimeError(
            "Focused C1 edit mask ratio is outside the audited safety range: "
            f"ratio={focused_ratio}, range=[{MIN_FOCUSED_EDIT_RATIO}, {MAX_FOCUSED_EDIT_RATIO}]"
        )
    roi_edit_pixels: List[int] = []
    for index, (left, upper, right, lower) in enumerate(EDIT_ROIS):
        count = int(focused_binary[upper:lower, left:right].sum())
        if count <= 0:
            raise RuntimeError(f"Focused C1 edit mask has no pixels in ROI {index}")
        roi_edit_pixels.append(count)

    native = torch.from_numpy(focused_binary.astype(np.float32))[None, None]
    token_weights = F.adaptive_avg_pool2d(native, TOKEN_GRID).reshape(-1).contiguous()
    if token_weights.numel() != EXPECTED_TARGET_TOKENS:
        raise RuntimeError(
            f"Region token count mismatch: {token_weights.numel()} != {EXPECTED_TARGET_TOKENS}"
        )
    if not torch.isfinite(token_weights).all():
        raise RuntimeError("Region token weights contain non-finite values")
    if float(token_weights.min()) < 0.0 or float(token_weights.max()) > 1.0:
        raise RuntimeError("Region token weights are outside [0, 1]")

    edit_mass = float(token_weights.sum().item())
    preserve_mass = float((1.0 - token_weights).sum().item())
    if edit_mass <= 0.0 or preserve_mass <= 0.0:
        raise RuntimeError(
            f"Both regions must be non-empty: edit_mass={edit_mass}, preserve_mass={preserve_mass}"
        )

    candidate_ratio = float(candidate_binary.mean())
    spec: Dict[str, Any] = {
        "objective": "target_flow_mse_region_split",
        "candidate_mask_warning": (
            "The manifest mask is automatic and candidate-only. This experiment uses only its "
            "intersection with the fixed old/new-ball ROIs and requires explicit acknowledgement."
        ),
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "target_path": str(target_path),
        "target_sha256": target_sha256,
        "expected_target_sha256": EXPECTED_TARGET_SHA256,
        "mask_path": str(mask_path),
        "mask_sha256": _sha256_file(mask_path),
        "candidate_mask_reconstruction": {
            "threshold": CANDIDATE_THRESHOLD,
            "dilation_px": CANDIDATE_DILATION_PX,
            "median_filter_px": 3,
            "pixel_exact_match": True,
        },
        "native_size_wh": [width, height],
        "strategy": REGION_STRATEGY,
        "edit_rois_ltrb": [list(roi) for roi in EDIT_ROIS],
        "edit_pixels_per_roi": roi_edit_pixels,
        "candidate_pixel_ratio": candidate_ratio,
        "focused_edit_pixel_ratio": focused_ratio,
        "focused_edit_ratio_bounds": [MIN_FOCUSED_EDIT_RATIO, MAX_FOCUSED_EDIT_RATIO],
        "token_grid_hw": list(TOKEN_GRID),
        "token_flatten_order": "row_major_h_then_w",
        "token_weight_projection": "adaptive_average_pool_native_binary_mask_to_16x16",
        "edit_token_mass": edit_mass,
        "preserve_token_mass": preserve_mass,
        "edit_token_mass_ratio": edit_mass / EXPECTED_TARGET_TOKENS,
        "preserve_token_mass_ratio": preserve_mass / EXPECTED_TARGET_TOKENS,
        "edit_loss_weight": EDIT_LOSS_WEIGHT,
        "preserve_loss_weight": PRESERVE_LOSS_WEIGHT,
        "edit_per_pixel_emphasis_vs_native": EDIT_LOSS_WEIGHT
        / (edit_mass / EXPECTED_TARGET_TOKENS),
        "preserve_per_pixel_emphasis_vs_native": PRESERVE_LOSS_WEIGHT
        / (preserve_mass / EXPECTED_TARGET_TOKENS),
        "preserve_target": "gold_target_flow_velocity_not_source_latent",
    }
    return spec, token_weights


def _region_forward_loss(
    model: torch.nn.Module,
    data: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    global _LAST_REGION_SCALARS, _LAST_MODEL_MSE_SHAPE
    if _EDIT_TOKEN_WEIGHTS_CPU is None:
        raise RuntimeError("Region weights were not initialized")

    with _autocast_for_model(model):
        loss_dict = model(**data)
    mse = loss_dict.get("mse")
    if mse is None:
        raise RuntimeError("Model returned no diffusion MSE")
    if mse.ndim != 2 or mse.shape[0] != EXPECTED_TARGET_TOKENS:
        raise RuntimeError(
            "Expected per-target-token MSE with shape "
            f"[{EXPECTED_TARGET_TOKENS}, D], got {list(mse.shape)}"
        )
    _LAST_MODEL_MSE_SHAPE = [int(value) for value in mse.shape]

    per_token_mse = mse.to(torch.float32).mean(dim=-1)
    edit_weights = _EDIT_TOKEN_WEIGHTS_CPU.to(
        device=per_token_mse.device, dtype=per_token_mse.dtype
    )
    preserve_weights = 1.0 - edit_weights
    edit_mass = edit_weights.sum()
    preserve_mass = preserve_weights.sum()
    if edit_mass.item() <= 0.0 or preserve_mass.item() <= 0.0:
        raise RuntimeError("Region mass became empty on model device")

    native_mse = per_token_mse.mean()
    edit_mse = (per_token_mse * edit_weights).sum() / edit_mass
    preserve_mse = (per_token_mse * preserve_weights).sum() / preserve_mass
    weighted_edit = EDIT_LOSS_WEIGHT * edit_mse
    weighted_preserve = PRESERVE_LOSS_WEIGHT * preserve_mse
    region_mse = weighted_edit + weighted_preserve

    native_recomposition = (
        (edit_mass / EXPECTED_TARGET_TOKENS) * edit_mse
        + (preserve_mass / EXPECTED_TARGET_TOKENS) * preserve_mse
    )
    recomposition_error = torch.abs(native_recomposition - native_mse)
    tolerance = 2e-6 * torch.maximum(native_mse.detach().abs(), torch.ones_like(native_mse))
    if recomposition_error.detach() > tolerance:
        raise RuntimeError(
            "Region partition does not recompose native MSE: "
            f"native={native_mse.item()}, recomposed={native_recomposition.item()}, "
            f"error={recomposition_error.item()}, tolerance={tolerance.item()}"
        )

    ce = loss_dict.get("ce")
    ce_loss = ce.mean().to(torch.float32) if ce is not None else torch.zeros_like(region_mse)
    total = region_mse + ce_loss
    scalars = {
        "total": float(total.detach().cpu()),
        "mse": float(region_mse.detach().cpu()),
        "native_mse": float(native_mse.detach().cpu()),
        "edit_mse": float(edit_mse.detach().cpu()),
        "preserve_mse": float(preserve_mse.detach().cpu()),
        "weighted_edit_mse": float(weighted_edit.detach().cpu()),
        "weighted_preserve_mse": float(weighted_preserve.detach().cpu()),
        "native_recomposition_error": float(recomposition_error.detach().cpu()),
        "ce": float(ce_loss.detach().cpu()),
    }
    _LAST_REGION_SCALARS = dict(scalars)
    return total, scalars


def _build_gold40_loader_with_fixed_eval_context(
    manifest_path: Path,
    tokenizer: Any,
    special_tokens: Dict[str, int],
    args: Any,
    *,
    seed_offset: int = 0,
) -> Any:
    loader = _ORIGINAL_BUILD_GOLD40_LOADER(
        manifest_path,
        tokenizer,
        special_tokens,
        args,
        seed_offset=seed_offset,
    )
    records = legacy.load_jsonl(manifest_path)
    if len(records) != 1 or records[0].get("pair_id") != legacy.C1_PAIR_ID:
        raise RuntimeError("Fixed region eval requires exactly the frozen C1 pair")
    _FIXED_EVAL_CONTEXT.clear()
    _FIXED_EVAL_CONTEXT.update(
        {
            "packer": legacy._build_singleton_packed_dataset(args, tokenizer, special_tokens),
            "raw_dataset": legacy._build_raw_sample_dataset(args, tokenizer),
            "pair": records[0],
        }
    )
    return loader


def _mean_region_mse(
    model: torch.nn.Module,
    vae_model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    model_dtype: torch.dtype,
    max_eval_batches: int,
) -> Dict[str, Any]:
    del loader, max_eval_batches
    if set(_FIXED_EVAL_CONTEXT) != {"packer", "raw_dataset", "pair"}:
        raise RuntimeError("Fixed region eval context was not initialized")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        batch, trace = legacy._pack_single_pair(
            _FIXED_EVAL_CONTEXT["packer"],
            _FIXED_EVAL_CONTEXT["raw_dataset"],
            _FIXED_EVAL_CONTEXT["pair"],
            FIXED_EVAL_SEED,
        )
        data = _prepare_batch(batch, vae_model, device, model_dtype)
        legacy._set_deterministic_seed(FIXED_EVAL_SEED)
        model.train()
        with torch.no_grad():
            _, scalars = _region_forward_loss(model, data)
        result: Dict[str, Any] = {
            "mean_mse": float(scalars["mse"]),
            "mean_native_mse": float(scalars["native_mse"]),
            "mean_edit_mse": float(scalars["edit_mse"]),
            "mean_preserve_mse": float(scalars["preserve_mse"]),
            "mean_weighted_edit_mse": float(scalars["weighted_edit_mse"]),
            "mean_weighted_preserve_mse": float(scalars["weighted_preserve_mse"]),
            "mean_native_recomposition_error": float(
                scalars["native_recomposition_error"]
            ),
            "num_batches": 1,
            "num_seen_pairs": 1,
            "evaluation_protocol": "fixed_c1_pack_and_forward_seed",
            "eval_seed": FIXED_EVAL_SEED,
            "packed_timesteps_hash": trace["packed_timesteps_hash"],
        }
        del data, batch, scalars
        return result
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _append_jsonl_with_regions(path: Path, payload: Dict[str, Any]) -> None:
    enriched = dict(payload)
    if path.name == "train_log.jsonl":
        for key in (
            "native_mse",
            "edit_mse",
            "preserve_mse",
            "weighted_edit_mse",
            "weighted_preserve_mse",
            "native_recomposition_error",
        ):
            if key not in _LAST_REGION_SCALARS:
                raise RuntimeError(f"Missing region scalar before train log write: {key}")
            enriched[key] = _LAST_REGION_SCALARS[key]
    _ORIGINAL_APPEND_JSONL(path, enriched)


def _module_code_identity() -> Dict[str, Dict[str, str]]:
    modules = {
        "region_wrapper": SCRIPT_PATH,
        "legacy_c1_driver": Path(legacy.__file__).resolve(),
        "gold40_overfit": Path(gold40_overfit_module.__file__).resolve(),
        "gold40_lora": Path(gold40_lora_module.__file__).resolve(),
        "lora_2step_smoke": Path(lora_smoke_module.__file__).resolve(),
        "fixed_deterministic_eval": Path(fixed_eval_module.__file__).resolve(),
    }
    identity: Dict[str, Dict[str, str]] = {}
    for name, path in modules.items():
        if not path.is_file():
            raise RuntimeError(f"Bound code module is missing: {name}={path}")
        identity[name] = {"path": str(path), "sha256": _sha256_file(path)}
    return identity


def _region_binding_payload(args: Any) -> Dict[str, Any]:
    return {
        **_REGION_SPEC,
        "script_path": str(SCRIPT_PATH),
        "script_sha256": _sha256_file(SCRIPT_PATH),
        "code_identity": _module_code_identity(),
        "acknowledgement_required_for_full_training": ACK_FLAG,
        "acknowledged_for_this_run": bool(args.acknowledge_audit_derived_mask),
    }


def _assert_dry_run_region_binding(args: Any) -> Dict[str, Any]:
    report_path = Path(args.dry_run_report).resolve()
    report_sha256 = _sha256_file(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected = _region_binding_payload(args)
    actual = report.get("region_objective")
    if not isinstance(actual, dict):
        raise RuntimeError("Dry-run report has no region_objective binding")
    failures = []
    if report.get("status") != "passed":
        failures.append(f"status: expected='passed', actual={report.get('status')!r}")
    if report.get("mode") != "dry_run":
        failures.append(f"mode: expected='dry_run', actual={report.get('mode')!r}")
    if (report.get("config_diff") or {}).get("status") != "passed":
        failures.append("config_diff.status is not 'passed'")
    forward = report.get("forward") or {}
    if forward.get("mse_all_finite") is not True:
        failures.append("forward.mse_all_finite is not true")
    model_mse_shape = forward.get("model_mse_shape")
    if not (
        isinstance(model_mse_shape, list)
        and len(model_mse_shape) == 2
        and model_mse_shape[0] == EXPECTED_TARGET_TOKENS
        and isinstance(model_mse_shape[1], int)
        and model_mse_shape[1] > 0
    ):
        failures.append(
            f"forward.model_mse_shape is invalid: {model_mse_shape!r}"
        )
    if forward.get("target_token_count") != EXPECTED_TARGET_TOKENS:
        failures.append(
            "forward.target_token_count mismatch: "
            f"{forward.get('target_token_count')!r} != {EXPECTED_TARGET_TOKENS}"
        )
    for key in (
        "objective",
        "source_sha256",
        "target_sha256",
        "mask_sha256",
        "strategy",
        "edit_rois_ltrb",
        "token_grid_hw",
        "edit_token_mass",
        "preserve_token_mass",
        "edit_loss_weight",
        "preserve_loss_weight",
        "script_sha256",
        "code_identity",
    ):
        if actual.get(key) != expected.get(key):
            failures.append(f"{key}: dry_run={actual.get(key)!r}, current={expected.get(key)!r}")
    if failures:
        raise RuntimeError("Region dry-run binding verification failed:\n  " + "\n  ".join(failures))
    return {
        "path": str(report_path),
        "sha256": report_sha256,
        "status": report["status"],
        "mode": report["mode"],
        "verified_before_full_training": True,
    }


def _strict_config_diff(args: Any) -> Dict[str, Any]:
    report = _ORIGINAL_CONFIG_DIFF(args)
    if report.get("status") == "passed":
        return report
    report = dict(report)
    original_status = report.get("status")
    report["status"] = "failed"
    report["forbidden_diffs"] = report.get("forbidden_diffs") or {
        "formal_b_config": {
            "reason": report.get(
                "reason", f"unexpected config audit status: {original_status!r}"
            )
        }
    }
    report["num_forbidden_diffs"] = len(report["forbidden_diffs"])
    return report


def _atomic_write_json_with_region(path: Path, payload: Dict[str, Any]) -> None:
    if _CURRENT_ARGS is not None and path.name == "micro4_dry_run_report.json":
        payload["region_objective"] = _region_binding_payload(_CURRENT_ARGS)
        forward = payload.setdefault("forward", {})
        forward["region_scalars"] = dict(_LAST_REGION_SCALARS)
        forward["model_mse_shape"] = list(_LAST_MODEL_MSE_SHAPE)
        forward["target_token_count"] = EXPECTED_TARGET_TOKENS
        forward["reduced_total_shape"] = []
        forward["mse_shape_note"] = (
            "legacy mse_shape is the reduced scalar loss shape; "
            "model_mse_shape is authoritative"
        )
    elif _CURRENT_ARGS is not None and path.name == "micro4_full_training_report.json":
        if not _DRY_RUN_REPORT_BINDING:
            raise RuntimeError("Full report cannot be written without dry-run report binding")
        payload["region_objective"] = _region_binding_payload(_CURRENT_ARGS)
        payload["dry_run_report_binding"] = dict(_DRY_RUN_REPORT_BINDING)
    _ORIGINAL_ATOMIC_WRITE_JSON(path, payload)


def _configure_legacy_module() -> None:
    legacy.BASE_OUTPUT_STEM = BASE_OUTPUT_STEM
    legacy.MICRO4_ALLOWED_DIFF_KEYS = legacy.MICRO4_ALLOWED_DIFF_KEYS | REGION_CONFIG_KEYS
    legacy._build_gold40_loader = _build_gold40_loader_with_fixed_eval_context
    legacy._micro4_vs_formal_b_diff = _strict_config_diff
    legacy._forward_loss = _region_forward_loss
    legacy._mean_train_mse = _mean_region_mse
    legacy._append_jsonl = _append_jsonl_with_regions
    legacy._atomic_write_json = _atomic_write_json_with_region


def _parse_args() -> Any:
    original_argv = list(sys.argv)
    acknowledged = ACK_FLAG in sys.argv[1:]
    filtered = [arg for arg in sys.argv[1:] if arg != ACK_FLAG]
    explicit_output = any(
        arg == "--output-dir" or arg.startswith("--output-dir=") for arg in filtered
    )
    explicit_dry_report = any(
        arg == "--dry-run-report" or arg.startswith("--dry-run-report=")
        for arg in filtered
    )
    try:
        sys.argv = [sys.argv[0], *filtered]
        args = legacy.parse_args()
    finally:
        sys.argv = original_argv

    if not explicit_output:
        suffix = "_preflight" if args.dry_run else "_500steps"
        args.output_dir = Path(str(BASE_OUTPUT_STEM) + suffix)
    if not explicit_dry_report:
        args.dry_run_report = Path(
            str(BASE_OUTPUT_STEM) + "_preflight/micro4_dry_run_report.json"
        )

    args.region_mask_path = Path(_REGION_SPEC["mask_path"])
    args.region_mask_sha256 = _REGION_SPEC["mask_sha256"]
    args.region_mask_strategy = REGION_STRATEGY
    args.region_edit_rois = [list(roi) for roi in EDIT_ROIS]
    args.region_token_grid = list(TOKEN_GRID)
    args.edit_loss_weight = EDIT_LOSS_WEIGHT
    args.preserve_loss_weight = PRESERVE_LOSS_WEIGHT
    args.acknowledge_audit_derived_mask = acknowledged
    return args


def main() -> None:
    global _REGION_SPEC, _EDIT_TOKEN_WEIGHTS_CPU, _CURRENT_ARGS
    global _DRY_RUN_REPORT_BINDING
    _REGION_SPEC, _EDIT_TOKEN_WEIGHTS_CPU = _load_region_spec()
    _configure_legacy_module()
    args = _parse_args()
    _CURRENT_ARGS = args

    try:
        if args.dry_run:
            report = legacy._run_dry_run(args)
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "mode": "dry_run",
                        "region_objective": report["region_objective"],
                        "forward": report["forward"],
                        "report_path": str(
                            args.output_dir / "micro4_dry_run_report.json"
                        ),
                    },
                    indent=2,
                )
            )
            return

        if not args.acknowledge_audit_derived_mask:
            raise RuntimeError(
                f"Full training requires {ACK_FLAG}. Review the preflight region_objective first."
            )
        _DRY_RUN_REPORT_BINDING = _assert_dry_run_region_binding(args)
        report = legacy._run_full_training(args)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "mode": "full_training",
                    "region_objective": report["region_objective"],
                    "exposure_audit": report["exposure_audit"],
                    "report_path": str(
                        args.output_dir / "micro4_full_training_report.json"
                    ),
                },
                indent=2,
            )
        )
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        legacy.traceback.print_exc(file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
