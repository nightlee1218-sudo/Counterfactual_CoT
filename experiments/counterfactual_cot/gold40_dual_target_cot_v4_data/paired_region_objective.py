"""Dynamic per-pair region objective for reviewed Gold40 paired records.

This module intentionally does not load a manifest or pack model inputs.  Its caller must
construct each branch's model input from the `target_image` recorded in the reviewed
manifest, then pass the matching branch id to :func:`paired_region_forward_loss`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_SIZE = (1024, 1024)
TOKEN_GRID_HW = (64, 64)
EXPECTED_TARGET_TOKENS = TOKEN_GRID_HW[0] * TOKEN_GRID_HW[1]
FLOW_CHANNELS = 64
LOSS_WEIGHTS = {
    "source_change": 0.25,
    "target_change": 0.25,
    "preserve": 0.50,
}
BRANCH_TARGET_STATES = {"source_state": "source", "target_state": "target"}
OBJECTIVE_ID = "gold40_dynamic_source_target_preserve_target_flow_v1"


class RegionObjectiveValidationError(RuntimeError):
    """A reviewed pair cannot safely be used by the dynamic objective."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RegionObjectiveValidationError(f"{label} must be an object")
    return value


def _path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RegionObjectiveValidationError(f"{label} must be a non-empty path")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return path


def _validate_asset(path_value: Any, expected_sha256: Any, label: str) -> Path:
    path = _path(path_value, label)
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RegionObjectiveValidationError(f"{label} has no valid expected SHA256")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise RegionObjectiveValidationError(
            f"{label} SHA256 mismatch: observed={observed}, expected={expected_sha256}"
        )
    return path


def _read_binary(path: Path, label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            values = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    except (OSError, ValueError) as exc:
        raise RegionObjectiveValidationError(f"Unable to decode {label}: {path}: {exc}") from exc
    if values.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0]):
        raise RegionObjectiveValidationError(
            f"{label} must be {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}, got {values.shape[::-1]}"
        )
    return values


def _validate_image_size(path: Path, label: str) -> None:
    try:
        with Image.open(path) as image:
            if image.size != IMAGE_SIZE:
                raise RegionObjectiveValidationError(
                    f"{label} must be {IMAGE_SIZE}, got {image.size}: {path}"
                )
    except (OSError, ValueError) as exc:
        raise RegionObjectiveValidationError(f"Unable to decode {label}: {path}: {exc}") from exc


def _parse_ltrb(value: Any, label: str) -> Tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise RegionObjectiveValidationError(f"{label} must be an [left, top, right, bottom] array")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise RegionObjectiveValidationError(f"{label} coordinates must be integers")
    left, top, right, bottom = value
    width, height = IMAGE_SIZE
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise RegionObjectiveValidationError(f"{label} is outside image bounds: {list(value)}")
    return left, top, right, bottom


def _rectangle_pixels(ltrb: Tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = ltrb
    result = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]), dtype=bool)
    result[top:bottom, left:right] = True
    return result


def _find_value(record: Mapping[str, Any], field: str) -> Any:
    for container in (
        record,
        record.get("dynamic_regions"),
        record.get("review"),
        record.get("human_review"),
        record.get("dynamic_candidate_mask"),
    ):
        if isinstance(container, Mapping) and field in container:
            return container[field]
    return None


def _roi_pixels(
    record: Mapping[str, Any],
    region: str,
    candidate: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resolve an approved binary ROI mask; bbox metadata is never optimization input."""
    value = _find_value(record, region)
    if value is None:
        return np.zeros_like(candidate, dtype=bool), {"kind": "absent"}
    if isinstance(value, Mapping):
        path_value = value.get("path", value.get("mask_path"))
        expected_hash = value.get("sha256")
    else:
        path_value = value
        expected_hash = None
    roi_path = _path(path_value, region)
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise RegionObjectiveValidationError(f"{region} ROI requires a valid SHA256")
    observed = _sha256(roi_path)
    if observed != expected_hash:
        raise RegionObjectiveValidationError(
            f"{region} ROI SHA256 mismatch: observed={observed}, expected={expected_hash}"
        )
    roi = _read_binary(roi_path, region)
    return candidate & roi, {
        "kind": "mask",
        "path": str(roi_path.resolve()),
        "sha256": observed,
    }


def _approved_review(record: Mapping[str, Any]) -> Mapping[str, Any]:
    review = _mapping(record.get("review", record.get("human_review")), "review")
    readiness = record.get("training_readiness", review)
    readiness = _mapping(readiness, "training_readiness")
    candidate = _mapping(record.get("dynamic_candidate_mask"), "dynamic_candidate_mask")
    if review.get("approved") is not True:
        raise RegionObjectiveValidationError("review.approved must be true")
    if readiness.get("ready_for_training", review.get("ready_for_training")) is not True:
        raise RegionObjectiveValidationError("training readiness must be explicitly true")
    if candidate.get("review_status") != "approved":
        raise RegionObjectiveValidationError("dynamic_candidate_mask.review_status must be 'approved'")
    return review


def _validated_branches(record: Mapping[str, Any], source: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_branches = record.get("paired_branches")
    if not isinstance(raw_branches, list) or len(raw_branches) != 2:
        raise RegionObjectiveValidationError("paired_branches must contain exactly two branches")
    branches: Dict[str, Dict[str, Any]] = {}
    expected_assets = {"source_state": source, "target_state": target}
    for raw_branch in raw_branches:
        branch = dict(_mapping(raw_branch, "paired_branches item"))
        branch_id = branch.get("branch_id")
        if branch_id not in BRANCH_TARGET_STATES or branch_id in branches:
            raise RegionObjectiveValidationError("paired_branches must contain source_state and target_state exactly once")
        expected = expected_assets[branch_id]
        if branch.get("target_state") != BRANCH_TARGET_STATES[branch_id]:
            raise RegionObjectiveValidationError(f"{branch_id} has an invalid target_state")
        if branch.get("target_image") != expected.get("image"):
            raise RegionObjectiveValidationError(
                f"{branch_id} must use its corresponding real target image, not a substituted flow"
            )
        if branch.get("target_image_sha256") != expected.get("image_sha256"):
            raise RegionObjectiveValidationError(f"{branch_id} target-image SHA256 differs from its paired target")
        branches[branch_id] = branch
    if set(branches) != set(BRANCH_TARGET_STATES):
        raise RegionObjectiveValidationError("missing source_state or target_state branch")
    return branches


def _validate_weights(weights: Mapping[str, torch.Tensor]) -> float:
    if set(weights) != set(LOSS_WEIGHTS):
        raise RegionObjectiveValidationError(f"Unexpected region labels: {sorted(weights)}")
    for label, tensor in weights.items():
        if tensor.ndim != 1 or tensor.numel() != EXPECTED_TARGET_TOKENS:
            raise RegionObjectiveValidationError(f"{label} weights must contain {EXPECTED_TARGET_TOKENS} tokens")
        if not torch.isfinite(tensor).all() or float(tensor.min()) < -1e-6 or float(tensor.max()) > 1.0 + 1e-6:
            raise RegionObjectiveValidationError(f"{label} weights are non-finite or outside [0, 1]")
        if float(tensor.sum()) <= 0.0:
            raise RegionObjectiveValidationError(f"{label} has zero token mass")
    partition_error = float(torch.max(torch.abs(sum(weights.values()) - 1.0)))
    if partition_error > 1e-6:
        raise RegionObjectiveValidationError(f"Region token weights do not partition unity: {partition_error}")
    return partition_error


def build_paired_region_objective(
    record: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, torch.Tensor]]]:
    """Build validated 64x64 weights for both paths of one reviewed macro-pair.

    The source and target changed regions are always the candidate mask intersected
    with their reviewed ROI.  The residual is the preserve region.  Each returned
    branch gets its own tensor mapping so training cannot accidentally reuse a
    different record's weights; both use the same spatial partition but different
    true target flows selected by `paired_branches`.
    """
    nested_candidate = record.get("candidate")
    if nested_candidate is not None:
        candidate_record = _mapping(nested_candidate, "candidate")
        record = {**candidate_record, **record}
    pair_id = record.get("pair_id")
    if not isinstance(pair_id, str) or not pair_id:
        raise RegionObjectiveValidationError("record.pair_id must be non-empty")
    if record.get("macro_pair_id") != f"dual_cot::{pair_id}":
        raise RegionObjectiveValidationError("macro_pair_id does not match pair_id")
    review = _approved_review(record)
    source = _mapping(record.get("source"), "source")
    target = _mapping(record.get("target"), "target")
    source_path = _validate_asset(source.get("image"), source.get("image_sha256"), "source image")
    target_path = _validate_asset(target.get("image"), target.get("image_sha256"), "target image")
    _validate_image_size(source_path, "source image")
    _validate_image_size(target_path, "target image")
    branches = _validated_branches(record, source, target)

    candidate_meta = _mapping(record.get("dynamic_candidate_mask"), "dynamic_candidate_mask")
    candidate_path = _validate_asset(candidate_meta.get("path"), candidate_meta.get("sha256"), "candidate mask")
    candidate = _read_binary(candidate_path, "candidate mask")
    if not candidate.any():
        raise RegionObjectiveValidationError("candidate mask is empty")
    source_only, source_roi_spec = _roi_pixels(record, "source_changed_roi", candidate)
    target_only, target_roi_spec = _roi_pixels(record, "target_changed_roi", candidate)
    shared_changed, shared_roi_spec = _roi_pixels(record, "shared_changed_roi", candidate)
    regions = (source_only, target_only, shared_changed)
    overlap_pixels = sum(
        int(np.logical_and(regions[i], regions[j]).sum())
        for i in range(len(regions))
        for j in range(i + 1, len(regions))
    )
    if overlap_pixels:
        raise RegionObjectiveValidationError(f"Reviewed source-only/target-only/shared masks overlap by {overlap_pixels} pixels")
    changed_union = np.logical_or.reduce(regions)
    if not changed_union.any():
        raise RegionObjectiveValidationError("At least one reviewed changed region must cover candidate-mask pixels")
    uncovered = np.logical_and(candidate, ~changed_union)
    if uncovered.any():
        raise RegionObjectiveValidationError(
            f"Reviewed changed regions fail to cover {int(uncovered.sum())} candidate-mask pixels"
        )

    source_only_tokens = F.adaptive_avg_pool2d(
        torch.from_numpy(source_only.astype(np.float32))[None, None], TOKEN_GRID_HW
    ).reshape(-1).contiguous()
    target_only_tokens = F.adaptive_avg_pool2d(
        torch.from_numpy(target_only.astype(np.float32))[None, None], TOKEN_GRID_HW
    ).reshape(-1).contiguous()
    shared_tokens = F.adaptive_avg_pool2d(
        torch.from_numpy(shared_changed.astype(np.float32))[None, None], TOKEN_GRID_HW
    ).reshape(-1).contiguous()
    source_tokens = (source_only_tokens + 0.5 * shared_tokens).contiguous()
    target_tokens = (target_only_tokens + 0.5 * shared_tokens).contiguous()
    preserve_tokens = (1.0 - source_only_tokens - target_only_tokens - shared_tokens).contiguous()
    shared_weights = {
        "source_change": source_tokens,
        "target_change": target_tokens,
        "preserve": preserve_tokens,
    }
    partition_error = _validate_weights(shared_weights)
    branch_weights = {
        branch_id: {label: tensor.clone() for label, tensor in shared_weights.items()}
        for branch_id in branches
    }
    for weights in branch_weights.values():
        _validate_weights(weights)

    masses = {label: float(tensor.sum()) for label, tensor in shared_weights.items()}
    spec = {
        "objective_id": OBJECTIVE_ID,
        "pair_id": pair_id,
        "formula": "0.25*source_change_mse + 0.25*target_change_mse + 0.50*preserve_mse",
        "loss_weights": dict(LOSS_WEIGHTS),
        "candidate_mask": {"path": str(candidate_path.resolve()), "sha256": candidate_meta["sha256"]},
        "source_changed_roi": source_roi_spec,
        "target_changed_roi": target_roi_spec,
        "shared_changed_roi": shared_roi_spec,
        "asset_size_wh": list(IMAGE_SIZE),
        "token_grid_hw": list(TOKEN_GRID_HW),
        "expected_target_tokens": EXPECTED_TARGET_TOKENS,
        "flow_channels": FLOW_CHANNELS,
        "token_masses": masses,
        "token_mass_ratios": {label: mass / EXPECTED_TARGET_TOKENS for label, mass in masses.items()},
        "partition_max_error": partition_error,
        "candidate_pixels": int(candidate.sum()),
        "source_only_pixels": int(source_only.sum()),
        "target_only_pixels": int(target_only.sum()),
        "shared_changed_pixels": int(shared_changed.sum()),
        "review": {
            "reviewer": review.get("reviewer"),
            "reviewed_at": review.get("reviewed_at"),
        },
        "branches": {
            branch_id: {
                "target_state": branch["target_state"],
                "target_image": branch["target_image"],
                "target_image_sha256": branch["target_image_sha256"],
                "target_flow": "model MSE for this branch's recorded target image",
            }
            for branch_id, branch in branches.items()
        },
    }
    return spec, branch_weights


def paired_region_forward_loss(
    model: torch.nn.Module,
    data: Dict[str, Any],
    branch_id: str,
    branch_weights_cpu: Mapping[str, Mapping[str, torch.Tensor]],
    autocast_context: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run one branch and aggregate its own real target-flow MSE by region.

    `data` must have been packed using `spec["branches"][branch_id]["target_image"]`.
    The shape check rejects non-target flow tensors and accidental packed-token drift.
    """
    if branch_id not in BRANCH_TARGET_STATES:
        raise RegionObjectiveValidationError(f"Unknown paired branch: {branch_id}")
    weights_cpu = branch_weights_cpu.get(branch_id)
    if weights_cpu is None:
        raise RegionObjectiveValidationError(f"No weights supplied for {branch_id}")
    partition_error = _validate_weights(weights_cpu)
    with autocast_context(model):
        output = model(**data)
    mse = output.get("mse")
    if mse is None:
        raise RuntimeError("Model returned no diffusion MSE for the branch target flow")
    if mse.ndim != 2 or tuple(mse.shape) != (EXPECTED_TARGET_TOKENS, FLOW_CHANNELS):
        raise RuntimeError(
            f"Expected branch target-flow MSE [{EXPECTED_TARGET_TOKENS}, {FLOW_CHANNELS}], got {list(mse.shape)}"
        )
    per_token = mse.float().mean(dim=-1)
    regions: Dict[str, torch.Tensor] = {}
    for label, weights in weights_cpu.items():
        current = weights.to(device=per_token.device, dtype=per_token.dtype)
        regions[label] = (per_token * current).sum() / current.sum()
    native = per_token.mean()
    recomposed = sum(
        (weights_cpu[label].to(device=per_token.device, dtype=per_token.dtype).sum() / EXPECTED_TARGET_TOKENS)
        * regions[label]
        for label in LOSS_WEIGHTS
    )
    recomposition_error = torch.abs(recomposed - native)
    tolerance = 2e-6 * torch.maximum(native.detach().abs(), torch.ones_like(native))
    if recomposition_error.detach() > tolerance:
        raise RuntimeError(f"Region partition failed to recompose native target-flow MSE: {float(recomposition_error)}")
    region_loss = sum(LOSS_WEIGHTS[label] * regions[label] for label in LOSS_WEIGHTS)
    ce = output.get("ce")
    ce_loss = ce.float().mean() if ce is not None else torch.zeros_like(region_loss)
    total = region_loss + ce_loss
    scalars = {
        "total": float(total.detach().cpu()),
        "region_mse": float(region_loss.detach().cpu()),
        "native_mse": float(native.detach().cpu()),
        "source_change_mse": float(regions["source_change"].detach().cpu()),
        "target_change_mse": float(regions["target_change"].detach().cpu()),
        "preserve_mse": float(regions["preserve"].detach().cpu()),
        "weighted_source_change_mse": float((LOSS_WEIGHTS["source_change"] * regions["source_change"]).detach().cpu()),
        "weighted_target_change_mse": float((LOSS_WEIGHTS["target_change"] * regions["target_change"]).detach().cpu()),
        "weighted_preserve_mse": float((LOSS_WEIGHTS["preserve"] * regions["preserve"]).detach().cpu()),
        "native_recomposition_error": float(recomposition_error.detach().cpu()),
        "token_partition_max_error": partition_error,
        "ce": float(ce_loss.detach().cpu()),
    }
    return total, scalars
