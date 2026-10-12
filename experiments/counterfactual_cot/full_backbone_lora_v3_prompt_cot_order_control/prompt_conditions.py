from __future__ import annotations

import hashlib
from typing import Any, Dict, Tuple

CONDITION_IDS: Tuple[str, ...] = (
    "target_prompt_then_source_cot_v1",
    "source_cot_then_target_prompt_v1",
    "target_prompt_then_target_cot_v1",
)
MODE = "[MODE=COUNTERFACTUAL_EDIT_PROMPT_COT_ORDER_CONTROL]"
FINAL_INSTRUCTION = "Generate the target image."
ORDERED_FIELDS = {
    "target_prompt_then_source_cot_v1": (("Target Prompt", "target", "prompt"), ("Source CoT", "source", "cot_text")),
    "source_cot_then_target_prompt_v1": (("Source CoT", "source", "cot_text"), ("Target Prompt", "target", "prompt")),
    "target_prompt_then_target_cot_v1": (("Target Prompt", "target", "prompt"), ("Target CoT", "target", "cot_text")),
}


def _required(pair: Dict[str, Any], side: str, field: str) -> str:
    value = pair.get(side, {}).get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty pair[{side!r}][{field!r}]")
    return value


def render_prompt_condition(pair: Dict[str, Any], condition_id: str) -> str:
    if condition_id not in ORDERED_FIELDS:
        raise KeyError(f"Unknown condition: {condition_id!r}")
    lines = [MODE]
    for label, side, field in ORDERED_FIELDS[condition_id]:
        lines.append(f"{label}: {_required(pair, side, field)}")
    lines.append(FINAL_INSTRUCTION)
    return "\n".join(lines)


def condition_manifest(pair: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for condition_id in CONDITION_IDS:
        text = render_prompt_condition(pair, condition_id)
        result[condition_id] = {
            "condition_id": condition_id,
            "ordered_fields": [
                {"label": label, "json_path": f"{side}.{field}"}
                for label, side, field in ORDERED_FIELDS[condition_id]
            ],
            "mode": MODE,
            "final_instruction": FINAL_INSTRUCTION,
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    first = result["target_prompt_then_source_cot_v1"]["text"].splitlines()
    second = result["source_cot_then_target_prompt_v1"]["text"].splitlines()
    if first[0] != second[0] or first[-1] != second[-1] or sorted(first[1:-1]) != sorted(second[1:-1]):
        raise RuntimeError("A/B order conditions differ by more than field order")
    return result
