from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Tuple

from .render_prompts import paired_prompt, render_intervention_text, render_preservation_text


@dataclass(frozen=True)
class PromptProfile:
    profile_id: str
    suite: str
    purpose: str
    source_prompt: bool
    source_cot: bool
    intervention: bool
    target_prompt: bool
    target_cot: bool
    preservation: bool


PROFILES: Dict[str, PromptProfile] = {
    "legacy_paired_v1": PromptProfile(
        "legacy_paired_v1", "core", "Historical paired CF-CoT MVP baseline.",
        True, True, True, False, True, True,
    ),
    "cot_core_v1": PromptProfile(
        "cot_core_v1", "core", "Primary Target-CoT-following test: Source Prompt versus Target CoT.",
        True, False, False, False, True, False,
    ),
    "full_paired_v2": PromptProfile(
        "full_paired_v2", "core", "Capability upper bound with every consistent textual signal.",
        True, True, True, True, True, True,
    ),
    "target_prompt_only_v1": PromptProfile(
        "target_prompt_only_v1", "core", "Natural-language target control without structured CoT.",
        False, False, False, True, False, False,
    ),
    "delta_only_v2": PromptProfile(
        "delta_only_v2", "core", "Direct edit command with explicit removal and count constraints.",
        False, False, True, False, False, True,
    ),
    "full_without_target_cot_v1": PromptProfile(
        "full_without_target_cot_v1", "core", "Target-CoT ablation paired with full_paired_v2.",
        True, True, True, True, False, True,
    ),
    "conflict_prompt_right_cot_left_v1": PromptProfile(
        "conflict_prompt_right_cot_left_v1", "conflict", "Target Prompt requests right while Target CoT requests left.",
        True, False, False, True, True, False,
    ),
    "conflict_prompt_left_cot_right_v1": PromptProfile(
        "conflict_prompt_left_cot_right_v1", "conflict", "Strong CoT test: image and prompts request left; Target CoT requests right.",
        True, False, False, True, True, False,
    ),
}

CORE_PROFILE_IDS: Tuple[str, ...] = tuple(
    profile_id for profile_id, profile in PROFILES.items() if profile.suite == "core"
)
CONFLICT_PROFILE_IDS: Tuple[str, ...] = tuple(
    profile_id for profile_id, profile in PROFILES.items() if profile.suite == "conflict"
)


def profile_ids_for_suite(suite: str) -> Tuple[str, ...]:
    if suite == "core":
        return CORE_PROFILE_IDS
    if suite == "conflict":
        return CONFLICT_PROFILE_IDS
    if suite == "all":
        return CORE_PROFILE_IDS + CONFLICT_PROFILE_IDS
    raise ValueError(f"Unknown prompt suite: {suite!r}")


def _required_text(pair: Dict[str, Any], side: str, field: str) -> str:
    value = pair.get(side, {}).get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty pair[{side!r}][{field!r}]")
    return value


def _readable(value: Any) -> str:
    return str(value).replace("_", " ")


def _cot_assignments(pair: Dict[str, Any], side: str) -> Dict[str, str]:
    text = str(pair.get(side, {}).get("cot_text", ""))
    assignments: Dict[str, str] = {}
    for fragment in text.split(";"):
        key, separator, value = fragment.strip().partition("=")
        if separator and key:
            assignments[key.strip()] = value.strip()
    return assignments


def _entity_label(pair: Dict[str, Any], entity_id: str) -> str:
    for side in ("source", "target"):
        entities = pair.get(side, {}).get("scene_graph", {}).get("entities", [])
        for entity in entities:
            if str(entity.get("id")) == entity_id:
                return str(entity.get("category") or entity_id)
        label = _cot_assignments(pair, side).get(f"{entity_id}.object")
        if label:
            return label
    return _readable(entity_id)


def _relation_target_label(pair: Dict[str, Any], entity_id: str, field: str) -> str:
    try:
        relation_index = int(field.split(".")[1])
    except (IndexError, ValueError):
        return "referenced object"
    for side in ("source", "target"):
        entities = pair.get(side, {}).get("scene_graph", {}).get("entities", [])
        by_id = {str(entity.get("id")): entity for entity in entities}
        entity = by_id.get(entity_id)
        relations = entity.get("relations", []) if entity else []
        if relation_index < len(relations):
            target_id = str(relations[relation_index].get("target_id", ""))
            target = by_id.get(target_id, {})
            return str(target.get("category") or target_id or "referenced object")

        assignments = _cot_assignments(pair, side)
        relation = assignments.get(f"{entity_id}.relation.{relation_index}", "")
        target_id = relation.partition(":")[2]
        if target_id:
            return assignments.get(f"{target_id}.object", _readable(target_id))
    return "referenced object"


def _render_delta_instruction(pair: Dict[str, Any]) -> str:
    intervention = pair.get("intervention", {})
    intervention_type = str(intervention.get("intervention_type", ""))
    entity_id = str(intervention.get("entity_id", "the target object"))
    entity = _entity_label(pair, entity_id)
    old_value = _readable(intervention.get("old_value", "the original state"))
    new_value = _readable(intervention.get("new_value", "the target state"))
    field = str(intervention.get("field", "state"))

    if intervention_type == "attribute_replace":
        attribute = _readable(field.removeprefix("attributes."))
        action = f"Change the {attribute} of the {entity} from {old_value} to {new_value}."
        completion = f"The final {entity} must have {new_value} {attribute} and must not retain the old {old_value} {attribute}."
    elif intervention_type == "count_change":
        action = f"Change the number of {entity} instances from {old_value} to {new_value}."
        completion = f"The final image must contain exactly {new_value} {entity} instances, not {old_value}."
    elif intervention_type == "object_replace":
        action = f"Replace the {old_value} with {new_value}."
        completion = f"Preserve the original count and placement, and do not retain the old {old_value}."
    elif intervention_type == "spatial_position_change":
        action = f"Move the {entity} from {old_value} to {new_value}."
        completion = f"Remove the {entity} from {old_value} and preserve its original count at {new_value}."
    elif intervention_type == "spatial_relation_change":
        relation_target = _relation_target_label(pair, entity_id, field)
        action = f"Change the relation of the {entity} to the {relation_target} from {old_value} to {new_value}."
        completion = f"The final relation must be {new_value}, not {old_value}; preserve both objects and their counts."
    else:
        raise ValueError(f"delta_only_v2 does not support intervention type {intervention_type!r}")

    preserve = ", ".join(pair.get("preservation", {}).get("must_preserve", []))
    return "\n".join((
        "[MODE=COUNTERFACTUAL_EDIT_DELTA_ONLY_V2]",
        action,
        completion,
        f"Preserve: {preserve or 'all non-target content, viewpoint, and lighting'}.",
        "Use the provided source image as the visual anchor.",
    ))


def render_prompt_profile(pair: Dict[str, Any], profile_id: str) -> str:
    if profile_id not in PROFILES:
        raise KeyError(f"Unknown prompt profile: {profile_id!r}")

    profile = PROFILES[profile_id]
    source_prompt = _required_text(pair, "source", "prompt") if profile.source_prompt else ""
    source_cot = _required_text(pair, "source", "cot_text") if profile.source_cot or profile_id == "conflict_prompt_right_cot_left_v1" else ""
    target_prompt = _required_text(pair, "target", "prompt") if profile.target_prompt else ""
    target_cot = _required_text(pair, "target", "cot_text") if profile.target_cot else ""

    if profile_id == "legacy_paired_v1":
        return paired_prompt(pair, include_source_image=True)
    if profile_id == "cot_core_v1":
        return (
            "[MODE=COUNTERFACTUAL_EDIT_COT_CORE]\n"
            f"Source Prompt: {source_prompt}\n"
            f"Target CoT: {target_cot}\n"
            "Generate the target image by strictly following Target CoT."
        )
    if profile_id == "full_paired_v2":
        return (
            "[MODE=COUNTERFACTUAL_EDIT]\n"
            f"Source Prompt: {source_prompt}\n"
            f"Source CoT: {source_cot}\n"
            f"Intervention: {render_intervention_text(pair['intervention'])}\n"
            f"Target Prompt: {target_prompt}\n"
            f"Target CoT: {target_cot}\n"
            f"{render_preservation_text(pair.get('preservation', {}))}\n"
            "Generate the target image."
        )
    if profile_id == "target_prompt_only_v1":
        return (
            "[MODE=COUNTERFACTUAL_EDIT_TARGET_PROMPT_ONLY]\n"
            f"Target Prompt: {target_prompt}\n"
            "Edit the provided source image to satisfy the Target Prompt."
        )
    if profile_id == "delta_only_v2":
        return _render_delta_instruction(pair)
    if profile_id == "full_without_target_cot_v1":
        return (
            "[MODE=COUNTERFACTUAL_EDIT]\n"
            f"Source Prompt: {source_prompt}\n"
            f"Source CoT: {source_cot}\n"
            f"Intervention: {render_intervention_text(pair['intervention'])}\n"
            f"Target Prompt: {target_prompt}\n"
            f"{render_preservation_text(pair.get('preservation', {}))}\n"
            "Generate the target image."
        )
    if profile_id == "conflict_prompt_right_cot_left_v1":
        return (
            "[MODE=COUNTERFACTUAL_EDIT_COT_CONFLICT]\n"
            f"Source Prompt: {source_prompt}\n"
            f"Target Prompt: {target_prompt}\n"
            f"Target CoT: {source_cot}\n"
            "Generate the target image by strictly following Target CoT."
        )
    if profile_id == "conflict_prompt_left_cot_right_v1":
        return (
            "[MODE=COUNTERFACTUAL_EDIT_COT_CONFLICT]\n"
            f"Source Prompt: {source_prompt}\n"
            f"Target Prompt: {source_prompt}\n"
            f"Target CoT: {target_cot}\n"
            "Generate the target image by strictly following Target CoT."
        )
    raise AssertionError(f"Unhandled prompt profile: {profile_id}")


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompt_profile_manifest(
    pair: Dict[str, Any], profile_ids: Iterable[str]
) -> Dict[str, Dict[str, Any]]:
    manifest: Dict[str, Dict[str, Any]] = {}
    for profile_id in profile_ids:
        profile = PROFILES[profile_id]
        text = render_prompt_profile(pair, profile_id)
        manifest[profile_id] = {
            **asdict(profile),
            "text": text,
            "sha256": prompt_sha256(text),
        }
    return manifest
