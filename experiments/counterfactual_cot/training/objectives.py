from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    target_generation: bool = True
    cot_ce: bool = False
    preserve_loss: bool = False
    delta_alignment_loss: bool = False
    note: str = ""


STANDARD_COT = ObjectiveSpec(
    name="standard_cot",
    target_generation=True,
    note="Prompt + structured CoT -> target image using BAGEL native MSE image loss.",
)

INDEPENDENT_CF = ObjectiveSpec(
    name="independent_cf",
    target_generation=True,
    note="Source and target records are exposed as independent CoT-to-image samples.",
)

PAIRED_CF_MVP = ObjectiveSpec(
    name="paired_cf_mvp",
    target_generation=True,
    note=(
        "Source image, source CoT, intervention, target CoT are serialized as conditioning; "
        "only BAGEL native target-image generation loss is optimized."
    ),
)

PAIRED_CF_FULL_PLACEHOLDER = ObjectiveSpec(
    name="paired_cf_full_placeholder",
    target_generation=True,
    preserve_loss=False,
    delta_alignment_loss=False,
    note=(
        "Explicit preservation and delta-alignment losses are intentionally disabled until "
        "BAGEL exposes spatial visual-token features or mask-weighted MSE loss indexes."
    ),
)


def get_objective_spec(name: str) -> ObjectiveSpec:
    specs = {
        spec.name: spec
        for spec in [STANDARD_COT, INDEPENDENT_CF, PAIRED_CF_MVP, PAIRED_CF_FULL_PLACEHOLDER]
    }
    if name not in specs:
        raise KeyError(f"Unknown objective spec: {name}")
    return specs[name]
