"""Auditable automatic spatial-weight objective for Gold40 AutoMask-v5.

The public paired API consumes one AutoMask-v5 macro-pair record and creates
identical edit/outside spatial weights for its source and target branches.  The
automatic difference mask is only a spatial weighting signal: it is not a
human-reviewed semantic ROI and must not be interpreted as one.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_SIZE_WH = (1024, 1024)
TOKEN_GRID_HW = (64, 64)
EXPECTED_TARGET_TOKENS = TOKEN_GRID_HW[0] * TOKEN_GRID_HW[1]
FLOW_CHANNELS = 64
LOSS_WEIGHTS = {"edit": 0.50, "outside": 0.50}
BRANCH_TARGET_STATES = {"source_state": "source", "target_state": "target"}
OBJECTIVE_ID = "gold40_v5_automask_edit_outside_target_flow_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AutomaskObjectiveValidationError(RuntimeError):
    """The v5 record cannot safely define an automatic-mask objective."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaskObjectiveValidationError(f"{label} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_file(path_value: Any, expected_sha256: Any, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise AutomaskObjectiveValidationError(f"{label}.path must be a non-empty path")
    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise AutomaskObjectiveValidationError(f"{label}.sha256 must be a lowercase 64-character SHA256")
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise AutomaskObjectiveValidationError(
            f"{label} SHA256 mismatch: observed={observed}, expected={expected_sha256}"
        )
    return path.resolve()


def _automatic_difference_mask(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the sole official AutoMask-v5 mask field."""
    if "dynamic_candidate_mask" in record or "auto_candidate_mask" in record:
        raise AutomaskObjectiveValidationError(
            "AutoMask-v5 accepts only automatic_difference_mask; legacy candidate-mask fields are forbidden"
        )
    return _mapping(record.get("automatic_difference_mask"), "automatic_difference_mask")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AutomaskObjectiveValidationError(f"{label} must be non-empty text")
    return value.strip()


def _read_strict_l_mask(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            if image.mode != "L":
                raise AutomaskObjectiveValidationError(
                    f"auto candidate mask must have native PIL mode 'L', got {image.mode}: {path}"
                )
            if image.size != IMAGE_SIZE_WH:
                raise AutomaskObjectiveValidationError(
                    f"auto candidate mask must be {IMAGE_SIZE_WH}, got {image.size}: {path}"
                )
            pixels = np.asarray(image, dtype=np.uint8) > 0
    except (OSError, ValueError) as exc:
        raise AutomaskObjectiveValidationError(f"Unable to decode auto candidate mask {path}: {exc}") from exc
    if not pixels.any():
        raise AutomaskObjectiveValidationError("auto candidate mask is empty")
    return pixels


def _validate_target_image(path: Path, label: str) -> None:
    try:
        with Image.open(path) as image:
            if image.size != IMAGE_SIZE_WH:
                raise AutomaskObjectiveValidationError(
                    f"{label} must be {IMAGE_SIZE_WH}, got {image.size}: {path}"
                )
    except (OSError, ValueError) as exc:
        raise AutomaskObjectiveValidationError(f"Unable to decode {label} {path}: {exc}") from exc


def _branch_target(record: Mapping[str, Any]) -> Dict[str, Any]:
    branch_id = record.get("branch_id")
    if branch_id not in BRANCH_TARGET_STATES:
        raise AutomaskObjectiveValidationError(
            "record.branch_id must be exactly 'source_state' or 'target_state'"
        )
    target = _mapping(record.get("target"), "target")
    expected_state = BRANCH_TARGET_STATES[branch_id]
    if target.get("state") != expected_state:
        raise AutomaskObjectiveValidationError(
            f"{branch_id} must use target.state={expected_state!r}, not {target.get('state')!r}"
        )
    target_path = _validated_file(target.get("image"), target.get("image_sha256"), "branch target image")
    _validate_target_image(target_path, "branch target image")
    return {
        "branch_id": branch_id,
        "target_state": expected_state,
        "target_image": str(target_path),
        "target_image_sha256": target["image_sha256"],
        "target_flow": "model MSE packed from this branch record's target_image",
    }


def _validate_partition(weights: Mapping[str, torch.Tensor]) -> float:
    if set(weights) != set(LOSS_WEIGHTS):
        raise AutomaskObjectiveValidationError(
            f"token partition must contain only {sorted(LOSS_WEIGHTS)}, got {sorted(weights)}"
        )
    for label, tensor in weights.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1 or tensor.numel() != EXPECTED_TARGET_TOKENS:
            raise AutomaskObjectiveValidationError(
                f"{label} weights must be a one-dimensional {EXPECTED_TARGET_TOKENS}-token tensor"
            )
        if not torch.isfinite(tensor).all() or float(tensor.min()) < -1e-6 or float(tensor.max()) > 1.0 + 1e-6:
            raise AutomaskObjectiveValidationError(f"{label} weights are non-finite or outside [0, 1]")
        if float(tensor.sum()) <= 0.0:
            raise AutomaskObjectiveValidationError(f"{label} token partition is empty")
    partition_error = float(torch.max(torch.abs(weights["edit"] + weights["outside"] - 1.0)))
    if partition_error > 1e-6:
        raise AutomaskObjectiveValidationError(
            f"edit/outside token weights do not form a unique partition: {partition_error}"
        )
    return partition_error


def _validated_macro_branches(record: Mapping[str, Any], pair_id: str) -> Dict[str, Dict[str, Any]]:
    """Validate that each paired branch names its own real endpoint target."""
    source = _mapping(record.get("source"), "source")
    target = _mapping(record.get("target"), "target")
    raw_branches = record.get("paired_branches")
    if not isinstance(raw_branches, list) or len(raw_branches) != len(BRANCH_TARGET_STATES):
        raise AutomaskObjectiveValidationError("paired_branches must contain exactly two branches")
    endpoints = {"source_state": source, "target_state": target}
    branches: Dict[str, Dict[str, Any]] = {}
    for raw_branch in raw_branches:
        branch = _mapping(raw_branch, "paired_branches item")
        branch_id = branch.get("branch_id")
        if branch_id not in BRANCH_TARGET_STATES or branch_id in branches:
            raise AutomaskObjectiveValidationError(
                "paired_branches must contain source_state and target_state exactly once"
            )
        expected_state = BRANCH_TARGET_STATES[branch_id]
        endpoint = endpoints[branch_id]
        if branch.get("target_state") != expected_state:
            raise AutomaskObjectiveValidationError(f"{branch_id} must use its {expected_state!r} target state")
        if endpoint.get("state") is not None and endpoint.get("state") != expected_state:
            raise AutomaskObjectiveValidationError(
                f"{branch_id} endpoint state conflicts with its {expected_state!r} branch target"
            )
        target_path = _validated_file(
            endpoint.get("image"), endpoint.get("image_sha256"), f"{branch_id} endpoint target image"
        )
        _validate_target_image(target_path, f"{branch_id} endpoint target image")
        branch_path = _validated_file(
            branch.get("target_image"), branch.get("target_image_sha256"), f"{branch_id} branch target image"
        )
        _validate_target_image(branch_path, f"{branch_id} branch target image")
        if branch_path != target_path or branch.get("target_image_sha256") != endpoint.get("image_sha256"):
            raise AutomaskObjectiveValidationError(
                f"{branch_id} target image must exactly match its corresponding real endpoint target"
            )
        branches[branch_id] = {
            "branch_id": branch_id,
            "target_state": expected_state,
            "target_image": str(target_path),
            "target_image_sha256": endpoint["image_sha256"],
            "target_flow": "model MSE packed from this branch's recorded target_image",
        }
    if set(branches) != set(BRANCH_TARGET_STATES):
        raise AutomaskObjectiveValidationError("missing source_state or target_state branch")
    return branches


def _validate_v5_macro_record(record: Mapping[str, Any]) -> Tuple[str, Mapping[str, Any], Dict[str, Dict[str, Any]]]:
    record = _mapping(record, "record")
    if record.get("schema_version") != "gold40_automask_v5":
        raise AutomaskObjectiveValidationError("schema_version must be 'gold40_automask_v5'")
    forbidden = [
        field for field in ("review", "human_review", "reviewed_mask", "dynamic_regions", "dynamic_candidate_mask")
        if field in record
    ]
    if forbidden:
        raise AutomaskObjectiveValidationError(f"AutoMask-v5 forbids human-review/candidate fields: {forbidden}")
    pair_id = _text(record.get("pair_id"), "pair_id")
    if record.get("macro_pair_id") != f"dual_cot::{pair_id}":
        raise AutomaskObjectiveValidationError("macro_pair_id does not match pair_id")
    _text(record.get("source_scene_id"), "source_scene_id")
    intervention_type = _text(record.get("intervention_type"), "intervention_type")
    intervention = _mapping(record.get("intervention"), "intervention")
    if intervention.get("intervention_type") != intervention_type:
        raise AutomaskObjectiveValidationError("top-level intervention_type must match intervention.intervention_type")
    readiness = _mapping(record.get("training_readiness"), "training_readiness")
    if readiness.get("ready_for_training") is not True or readiness.get("mode") != "automatic_difference_mask":
        raise AutomaskObjectiveValidationError(
            "training_readiness requires ready_for_training=true and mode='automatic_difference_mask'"
        )
    provenance = _mapping(record.get("mask_provenance"), "mask_provenance")
    expected_provenance = {
        "kind": "automatic_rgb_difference_mask",
        "semantic_annotation": False,
        "state_specific": False,
    }
    if {key: provenance.get(key) for key in expected_provenance} != expected_provenance:
        raise AutomaskObjectiveValidationError(f"mask_provenance must contain {expected_provenance}")
    mask = _automatic_difference_mask(record)
    branches = _validated_macro_branches(record, pair_id)
    return pair_id, mask, branches


def _partition_from_mask(mask: Mapping[str, Any]) -> Tuple[Path, np.ndarray, Dict[str, torch.Tensor], float]:
    mask_path = _validated_file(mask.get("path"), mask.get("sha256"), "automatic difference mask")
    candidate_pixels = _read_strict_l_mask(mask_path)
    edit = F.adaptive_avg_pool2d(
        torch.from_numpy(candidate_pixels.astype(np.float32))[None, None], TOKEN_GRID_HW
    ).reshape(-1).contiguous()
    weights = {"edit": edit, "outside": (1.0 - edit).contiguous()}
    return mask_path, candidate_pixels, weights, _validate_partition(weights)


def build_automask_objective(record: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Build one validated 64×64 partition from an AutoMask-v5 macro record.

    This lower-level API returns the single spatial partition shared by both
    branches.  Use :func:`build_paired_automask_objective` with the paired trainer.
    """
    pair_id, mask, branches = _validate_v5_macro_record(record)
    mask_path, candidate_pixels, weights, partition_error = _partition_from_mask(mask)
    masses = {label: float(tensor.sum()) for label, tensor in weights.items()}
    spec = {
        "objective_id": OBJECTIVE_ID,
        "pair_id": pair_id,
        "formula": "0.50*edit_mse + 0.50*outside_mse + ce",
        "loss_weights": dict(LOSS_WEIGHTS),
        "automatic_mask_only": True,
        "interpretation_boundary": "The automatic difference mask is a spatial loss weight, not a human semantic ROI.",
        "automatic_difference_mask": {"path": str(mask_path), "sha256": mask["sha256"]},
        "asset_size_wh": list(IMAGE_SIZE_WH),
        "mask_mode": "L",
        "candidate_pixels": int(candidate_pixels.sum()),
        "token_grid_hw": list(TOKEN_GRID_HW),
        "expected_target_tokens": EXPECTED_TARGET_TOKENS,
        "flow_channels": FLOW_CHANNELS,
        "token_masses": masses,
        "token_mass_ratios": {label: mass / EXPECTED_TARGET_TOKENS for label, mass in masses.items()},
        "partition_max_error": partition_error,
        "branches": branches,
    }
    return spec, weights


def build_paired_automask_objective(
    record: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, torch.Tensor]]]:
    """Build the shared spatial partition for source and target target flows.

    Both branch mappings deliberately reference the same immutable-by-contract CPU
    tensors; the model data, not the loss mask, selects each branch's true flow.
    """
    spec, weights = build_automask_objective(record)
    branch_weights = {branch_id: weights for branch_id in BRANCH_TARGET_STATES}
    for branch_id, current in branch_weights.items():
        if current is not weights or _validate_partition(current) != spec["partition_max_error"]:
            raise AutomaskObjectiveValidationError(f"{branch_id} does not reuse the audited shared partition")
    return spec, branch_weights


def automask_forward_loss(
    model: torch.nn.Module,
    data: Dict[str, Any],
    weights_cpu: Mapping[str, torch.Tensor],
    autocast_context: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the v5 automatic-mask loss for the already-packed branch target flow.

    The caller must invoke this only with model inputs packed from the target image
    audited by :func:`build_automask_objective` for the same branch record.
    """
    partition_error = _validate_partition(weights_cpu)
    with autocast_context(model):
        output = model(**data)
    mse = output.get("mse")
    if mse is None or mse.ndim != 2 or tuple(mse.shape) != (EXPECTED_TARGET_TOKENS, FLOW_CHANNELS):
        raise RuntimeError(
            f"Expected branch target-flow MSE [{EXPECTED_TARGET_TOKENS}, {FLOW_CHANNELS}], "
            f"got {None if mse is None else list(mse.shape)}"
        )
    per_token = mse.float().mean(dim=-1)
    regions: Dict[str, torch.Tensor] = {}
    for label in LOSS_WEIGHTS:
        current = weights_cpu[label].to(device=per_token.device, dtype=per_token.dtype)
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
        raise RuntimeError(
            "Automatic-mask partition failed to recompose native target-flow MSE: "
            f"{float(recomposition_error)}"
        )
    region_loss = sum(LOSS_WEIGHTS[label] * regions[label] for label in LOSS_WEIGHTS)
    ce = output.get("ce")
    ce_loss = ce.float().mean() if ce is not None else torch.zeros_like(region_loss)
    total = region_loss + ce_loss
    return total, {
        "total": float(total.detach().cpu()),
        "region_mse": float(region_loss.detach().cpu()),
        "native_mse": float(native.detach().cpu()),
        "edit_mse": float(regions["edit"].detach().cpu()),
        "outside_mse": float(regions["outside"].detach().cpu()),
        "weighted_edit_mse": float((LOSS_WEIGHTS["edit"] * regions["edit"]).detach().cpu()),
        "weighted_outside_mse": float((LOSS_WEIGHTS["outside"] * regions["outside"]).detach().cpu()),
        "native_recomposition_error": float(recomposition_error.detach().cpu()),
        "token_partition_max_error": partition_error,
        "ce": float(ce_loss.detach().cpu()),
    }


def paired_automask_forward_loss(
    model: torch.nn.Module,
    data: Dict[str, Any],
    branch_id: str,
    branch_weights_cpu: Mapping[str, Mapping[str, torch.Tensor]],
    autocast_context: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Run one paired branch using its own packed real target flow.

    `branch_id` selects only the audited shared spatial partition.  The actual
    source/target flow is selected by the caller's separately packed ``data``;
    `build_paired_automask_objective` verifies those targets in the macro record.
    """
    if branch_id not in BRANCH_TARGET_STATES:
        raise AutomaskObjectiveValidationError(f"Unknown paired branch: {branch_id}")
    weights = branch_weights_cpu.get(branch_id)
    if weights is None:
        raise AutomaskObjectiveValidationError(f"No automatic-mask weights supplied for {branch_id}")
    if set(branch_weights_cpu) != set(BRANCH_TARGET_STATES):
        raise AutomaskObjectiveValidationError("paired weights must contain source_state and target_state exactly once")
    other_branch = "target_state" if branch_id == "source_state" else "source_state"
    if branch_weights_cpu.get(other_branch) is not weights:
        raise AutomaskObjectiveValidationError("paired branches must reuse the same audited edit/outside weights")
    return automask_forward_loss(model, data, weights, autocast_context)
