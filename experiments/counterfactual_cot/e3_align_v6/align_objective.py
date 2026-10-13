"""E3 v6 spatial objective: v5 automask partition + configurable anti-overshoot.

Public API matches what the v5 trainer's ``_load_objective`` expects for the
paired interface (``OBJECTIVE_ID``, ``build_paired_automask_objective``,
``paired_automask_forward_loss``), so v6 training is drop-in compatible with
the v5 objective loader.  Differences from the v5 objective module:

* ``configure_overshoot`` sets two anti-over-edit switches (both default off
  until configured; the v6 trainer configures them from CLI args):

  - ``edit_weight``: rebalances the region loss to ``ew * edit + (1-ew) *
    outside``.  v5 hardcodes 0.5/0.5; over-editing is countered by shifting
    weight toward outside preservation (v6 default 0.4/0.6 when the master
    switch is on).
  - ``debias``: computes each region loss as the unweighted per-token mean over
    the hard region membership (``w_edit >= 0.5``) instead of the
    continuous-mass mean.  The v5 continuous mean lets the 9px-dilated mask
    halo blend weakly-masked tokens into both regions, diluting the edit-region
    signal and under-penalizing halo spillover; the hard partition gives halo
    tokens full outside pressure.  Empty hard regions fall back to the
    continuous form for that call and flag it in metrics.

* Region-loss identity is re-audited per call: continuous mode keeps the v5
  mass-weighted recomposition check; hard mode uses the exact count identity
  ``native == (n_in/N)*mean_in + (n_out/N)*mean_out``.

The shared spatial partition itself (64x64 row-major edit/outside weights with
strict mask provenance + SHA256 audit) is reused verbatim from the v5 module.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch

from experiments.counterfactual_cot.gold40_automask_v5_data.automask_objective import (
    AutomaskObjectiveValidationError,
    EXPECTED_TARGET_TOKENS,
    FLOW_CHANNELS,
    _validate_partition,
    build_paired_automask_objective as _v5_build_paired,
    automask_forward_loss as _v5_forward,
)

__all__ = [
    "OBJECTIVE_ID",
    "configure_overshoot",
    "overshoot_state",
    "build_paired_automask_objective",
    "paired_automask_forward_loss",
]

OBJECTIVE_ID = "gold40_v6_automask_edit_outside_debias_flow_v1"
HARD_IN_THRESHOLD = 0.5

_STATE: Dict[str, Any] = {"edit_weight": 0.5, "debias": False}


def configure_overshoot(edit_weight: float = 0.5, debias: bool = False) -> Dict[str, Any]:
    """Set the anti-overshoot switches (called once by the v6 trainer per rank)."""
    edit_weight = float(edit_weight)
    if not (0.0 < edit_weight < 1.0):
        raise AutomaskObjectiveValidationError(f"edit_weight must be in (0, 1), got {edit_weight}")
    _STATE["edit_weight"] = edit_weight
    _STATE["debias"] = bool(debias)
    return overshoot_state()


def overshoot_state() -> Dict[str, Any]:
    return {"edit_weight": _STATE["edit_weight"], "debias": _STATE["debias"]}


def build_paired_automask_objective(record: Mapping[str, Any]):
    """Identical audited partition to v5 (shared by both branches)."""
    return _v5_build_paired(record)


def _hard_region_losses(per_token: torch.Tensor, weights: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, int, int, str]:
    hard_in = weights["edit"] >= HARD_IN_THRESHOLD
    n_in = int(hard_in.sum())
    n_out = int(per_token.numel() - n_in)
    if n_in == 0 or n_out == 0:
        edit_region = (per_token * weights["edit"]).sum() / weights["edit"].sum()
        outside_region = (per_token * weights["outside"]).sum() / weights["outside"].sum()
        return edit_region, outside_region, n_in, n_out, "debias_fallback_continuous"
    edit_region = per_token[hard_in].mean()
    outside_region = per_token[~hard_in].mean()
    return edit_region, outside_region, n_in, n_out, "hard_partition"


def _continuous_region_losses(per_token: torch.Tensor, weights: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    edit_region = (per_token * weights["edit"]).sum() / weights["edit"].sum()
    outside_region = (per_token * weights["outside"]).sum() / weights["outside"].sum()
    return edit_region, outside_region


def paired_automask_forward_loss(
    model: torch.nn.Module,
    data: Dict[str, Any],
    branch_id: str,
    branch_weights_cpu: Mapping[str, Mapping[str, torch.Tensor]],
    autocast_context: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """One paired branch with the v6 region-loss configuration."""
    if branch_id not in branch_weights_cpu:
        raise AutomaskObjectiveValidationError(f"Unknown paired branch: {branch_id}")
    weights = branch_weights_cpu.get(branch_id)
    if weights is None:
        raise AutomaskObjectiveValidationError(f"No automatic-mask weights supplied for {branch_id}")
    partition_error = _validate_partition(weights)
    edit_weight = float(_STATE["edit_weight"])
    debias = bool(_STATE["debias"])
    with autocast_context(model):
        output = model(**data)
    mse = output.get("mse")
    if mse is None or mse.ndim != 2 or tuple(mse.shape) != (EXPECTED_TARGET_TOKENS, FLOW_CHANNELS):
        raise RuntimeError(
            f"Expected branch target-flow MSE [{EXPECTED_TARGET_TOKENS}, {FLOW_CHANNELS}], "
            f"got {None if mse is None else list(mse.shape)}"
        )
    per_token = mse.float().mean(dim=-1)
    device, dtype = per_token.device, per_token.dtype
    on_device = {
        label: weights[label].to(device=device, dtype=dtype)
        for label in ("edit", "outside")
    }

    region_mode = "continuous"
    if debias:
        edit_region, outside_region, n_in, n_out, region_mode = _hard_region_losses(per_token, on_device)
        if region_mode == "hard_partition":
            recomposed = (n_in * edit_region + n_out * outside_region) / EXPECTED_TARGET_TOKENS
        else:
            recomposed = sum(
                (on_device[label].sum() / EXPECTED_TARGET_TOKENS) * region
                for label, region in (("edit", edit_region), ("outside", outside_region))
            )
    else:
        edit_region, outside_region = _continuous_region_losses(per_token, on_device)
        recomposed = sum(
            (on_device[label].sum() / EXPECTED_TARGET_TOKENS) * region
            for label, region in (("edit", edit_region), ("outside", outside_region))
        )

    native = per_token.mean()
    recomposition_error = torch.abs(recomposed - native)
    tolerance = 2e-6 * torch.maximum(native.detach().abs(), torch.ones_like(native))
    if recomposition_error.detach() > tolerance:
        raise RuntimeError(
            f"v6 region partition failed to recompose native target-flow MSE: {float(recomposition_error)}"
        )

    region_loss = edit_weight * edit_region + (1.0 - edit_weight) * outside_region
    ce = output.get("ce")
    ce_loss = ce.float().mean() if ce is not None else torch.zeros_like(region_loss)
    total = region_loss + ce_loss
    return total, {
        "total": float(total.detach().cpu()),
        "region_mse": float(region_loss.detach().cpu()),
        "native_mse": float(native.detach().cpu()),
        "edit_mse": float(edit_region.detach().cpu()),
        "outside_mse": float(outside_region.detach().cpu()),
        "weighted_edit_mse": float((edit_weight * edit_region).detach().cpu()),
        "weighted_outside_mse": float(((1.0 - edit_weight) * outside_region).detach().cpu()),
        "native_recomposition_error": float(recomposition_error.detach().cpu()),
        "token_partition_max_error": partition_error,
        "ce": float(ce_loss.detach().cpu()),
        "region_mode": region_mode,
        "edit_weight": edit_weight,
        "debias": float(debias),
    }


def _v5_reference_forward_loss(model, data, weights_cpu, autocast_context):
    """Direct v5 reference for the CPU selftest (identical numbers when ew=0.5)."""
    return _v5_forward(model, data, weights_cpu, autocast_context)
