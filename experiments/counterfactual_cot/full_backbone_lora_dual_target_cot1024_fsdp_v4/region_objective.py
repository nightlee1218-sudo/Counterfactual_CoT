"""Position-symmetric region objective for paired left/right Target-CoT training."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.region_objective import (
    EXPECTED_TARGET_TOKENS,
    LOSS_WEIGHTS,
    build_region_objective,
)

OBJECTIVE_ID = "0.25_left_roi_0.25_right_roi_0.50_outside_v1"


def build_paired_region_objective(source_path, target_path, mask_path):
    base_spec, base_weights = build_region_objective(source_path, target_path, mask_path)
    weights = {
        "left_roi": base_weights["old"],
        "right_roi": base_weights["new"],
        "outside": base_weights["preserve"],
    }
    spec = dict(base_spec)
    spec.update(
        objective="paired_left_right_outside_target_flow_mse",
        objective_id=OBJECTIVE_ID,
        formula="0.25*left_roi_mse + 0.25*right_roi_mse + 0.50*outside_mse",
        loss_weights={
            "left_roi": LOSS_WEIGHTS["old"],
            "right_roi": LOSS_WEIGHTS["new"],
            "outside": LOSS_WEIGHTS["preserve"],
        },
        branch_semantics={
            "left": "left ROI contains the ball; right ROI remains empty",
            "right": "left ROI removes the ball; right ROI contains the ball",
        },
    )
    spec["token_masses"] = {label: float(tensor.sum()) for label, tensor in weights.items()}
    spec["token_mass_ratios"] = {
        label: float(tensor.sum()) / EXPECTED_TARGET_TOKENS for label, tensor in weights.items()
    }
    return spec, weights


def paired_region_forward_loss(
    model: torch.nn.Module,
    data: Dict[str, Any],
    weights_cpu: Dict[str, torch.Tensor],
    autocast_context: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    with autocast_context(model):
        output = model(**data)
    mse = output.get("mse")
    if mse is None:
        raise RuntimeError("Model returned no diffusion MSE")
    if mse.ndim != 2 or tuple(mse.shape) != (EXPECTED_TARGET_TOKENS, 64):
        raise RuntimeError(f"Expected target-flow MSE [{EXPECTED_TARGET_TOKENS}, 64], got {list(mse.shape)}")
    per_token = mse.float().mean(dim=-1)
    region_mses: Dict[str, torch.Tensor] = {}
    device_weights: Dict[str, torch.Tensor] = {}
    for label, weights in weights_cpu.items():
        current = weights.to(device=per_token.device, dtype=per_token.dtype)
        device_weights[label] = current
        region_mses[label] = (per_token * current).sum() / current.sum()
    native = per_token.mean()
    recomposed = sum(
        (device_weights[label].sum() / EXPECTED_TARGET_TOKENS) * region_mses[label]
        for label in ("left_roi", "right_roi", "outside")
    )
    recomposition_error = torch.abs(recomposed - native)
    tolerance = 2e-6 * torch.maximum(native.detach().abs(), torch.ones_like(native))
    if recomposition_error.detach() > tolerance:
        raise RuntimeError(f"Region partition failed to recompose native MSE: {float(recomposition_error)}")
    loss_weights = {
        "left_roi": LOSS_WEIGHTS["old"],
        "right_roi": LOSS_WEIGHTS["new"],
        "outside": LOSS_WEIGHTS["preserve"],
    }
    region_loss = sum(loss_weights[label] * region_mses[label] for label in loss_weights)
    ce = output.get("ce")
    ce_loss = ce.float().mean() if ce is not None else torch.zeros_like(region_loss)
    total = region_loss + ce_loss
    scalars = {
        "total": float(total.detach().cpu()),
        "region_mse": float(region_loss.detach().cpu()),
        "native_mse": float(native.detach().cpu()),
        "left_roi_mse": float(region_mses["left_roi"].detach().cpu()),
        "right_roi_mse": float(region_mses["right_roi"].detach().cpu()),
        "outside_mse": float(region_mses["outside"].detach().cpu()),
        "weighted_left_roi_mse": float((loss_weights["left_roi"] * region_mses["left_roi"]).detach().cpu()),
        "weighted_right_roi_mse": float((loss_weights["right_roi"] * region_mses["right_roi"]).detach().cpu()),
        "weighted_outside_mse": float((loss_weights["outside"] * region_mses["outside"]).detach().cpu()),
        "native_recomposition_error": float(recomposition_error.detach().cpu()),
        "ce": float(ce_loss.detach().cpu()),
    }
    return total, scalars
