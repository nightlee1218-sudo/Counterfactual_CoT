"""Dynamic Gold40 ROI objective. Public API: ``build_weights_from_record(record)``."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch

from .paired_data import review_and_regions

OBJECTIVE_ID = "gold40_dynamic_source_target_roi_outside_v1"
TARGET_TOKENS = 4096
GRID = 64


def _roi_grid(ltrb: Any) -> torch.Tensor:
    left, top, right, bottom = [int(value) for value in ltrb]
    # 1024px -> 64 target-flow tokens (each token spans 16px).
    x0, y0 = left * GRID // 1024, top * GRID // 1024
    x1, y1 = (right + 15) * GRID // 1024, (bottom + 15) * GRID // 1024
    x1, y1 = min(GRID, max(x0 + 1, x1)), min(GRID, max(y0 + 1, y1))
    mask = torch.zeros((GRID, GRID), dtype=torch.float32)
    mask[y0:y1, x0:x1] = 1.0
    return mask.flatten()


def build_weights_from_record(record: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Return CPU token partitions for source/target changed ROIs and preservation area.

    This API intentionally consumes the reviewed manifest record directly, making the
    review provenance and dynamic mask geometry inseparable from optimization.
    """
    _, regions = review_and_regions(record)
    source = _roi_grid(regions["source_changed_roi_ltrb"])
    target = _roi_grid(regions["target_changed_roi_ltrb"])
    changed = torch.clamp(source + target, max=1.0)
    outside = 1.0 - changed
    if not all(float(value.sum()) > 0 for value in (source, target, outside)):
        raise ValueError(f"{record.get('pair_id')}: dynamic ROI leaves an empty objective partition")
    weights = {"source_changed": source, "target_changed": target, "outside": outside}
    spec = {
        "objective_id": OBJECTIVE_ID,
        "formula": "0.25*source_changed + 0.25*target_changed + 0.50*outside",
        "source_changed_roi_ltrb": list(regions["source_changed_roi_ltrb"]),
        "target_changed_roi_ltrb": list(regions["target_changed_roi_ltrb"]),
        "token_masses": {key: float(value.sum()) for key, value in weights.items()},
    }
    return spec, weights


def forward_loss(model: torch.nn.Module, data: Dict[str, Any], weights_cpu: Mapping[str, torch.Tensor], autocast_context: Any):
    with autocast_context(model):
        output = model(**data)
    mse = output.get("mse")
    if mse is None or tuple(mse.shape) != (TARGET_TOKENS, 64):
        raise RuntimeError(f"Expected target flow MSE [{TARGET_TOKENS}, 64], got {None if mse is None else list(mse.shape)}")
    per_token = mse.float().mean(dim=-1)
    region = {}
    for label, weight in weights_cpu.items():
        current = weight.to(per_token.device, dtype=per_token.dtype)
        region[label] = (per_token * current).sum() / current.sum()
    region_loss = .25 * region["source_changed"] + .25 * region["target_changed"] + .50 * region["outside"]
    ce = output.get("ce")
    ce_loss = ce.float().mean() if ce is not None else torch.zeros_like(region_loss)
    total = region_loss + ce_loss
    return total, {"total": float(total.detach().cpu()), "region_mse": float(region_loss.detach().cpu()), "ce": float(ce_loss.detach().cpu()), **{f"{key}_mse": float(value.detach().cpu()) for key, value in region.items()}}
