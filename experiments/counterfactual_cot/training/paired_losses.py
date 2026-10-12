from __future__ import annotations


class PairedLossUnavailable(RuntimeError):
    """Raised when a requested full paired loss is not supported by current BAGEL hooks."""


def require_mask_aware_visual_tokens() -> None:
    raise PairedLossUnavailable(
        "Current experiment code only implements Paired CF MVP with serialized pair conditioning and "
        "BAGEL native target-image MSE loss. Explicit mask-aware preservation or delta-alignment loss "
        "requires spatial visual-token hooks that are not exposed by the audited training path."
    )


def compute_preservation_loss(*args, **kwargs):
    require_mask_aware_visual_tokens()


def compute_delta_alignment_loss(*args, **kwargs):
    require_mask_aware_visual_tokens()
