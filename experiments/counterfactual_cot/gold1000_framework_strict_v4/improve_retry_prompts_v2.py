#!/usr/bin/env python3
"""Improved retry prompts (v2) for the human-flagged failed pairs.

The original planning manifests are never modified.  For every chunk that
contains at least one flagged pair, a full patched copy is written next to it
as ``chunk_XX.retry_v2.jsonl``; regeneration must use the patched copy as
``--manifest`` while keeping the original ``chunk_XX.generated.jsonl`` as
``--output-manifest`` so ``--resume`` still skips the healthy pairs.

Patch goals (from the 2026-09-21 human review):
- count edits: physical imperative first, exact arithmetic, survivor identity
  (melted wax / wear) preservation, explicit failure conditions;
- occlusion edits: "pick up and move" phrasing with a checkable overlap
  criterion, keeping the validator-required "both objects remain visible";
- absolute moves: quantified displacement (>= 1/3 of the frame) and an
  explicit "large objects must still be moved" clause;
- category replaces: inject a concrete visual definition plus a silhouette
  requirement.
Training semantics are untouched: only ``target.prompt`` (the API edit
instruction) changes; Source/Target CoT and prompts used for training are
read-only here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
CHUNK_DIR = SCRIPT_DIR / "full_1000_20260918" / "chunks"

PATCH_PAIR_IDS = {
    "g1000_picnic_blanket__bottle_cups_forks__3_to_2_001",
    "g1000_picnic_blanket__bottle_cups_forks__2_to_1_001",
    "g1000_dining_table__vase_flowers_candles__2_to_1_001",
    "g1000_bedroom_nightstand__glasses_charger_glass__occlusion_state_003",
    "g1000_bedroom_nightstand__glasses_charger_glass__occlusion_state_008",
    "g1000_bookshelf_study__mugcollection_folder__absolute_vertical_001",
    "g1000_classroom__n2_globe__2_to_3_001",
    "g1000_classroom__notebook_fill__3_to_2_001",
    "g1000_kitchen_counter__toaster_dish_jar__occlusion_state_002",
    "g1000_warehouse_shelf__wrench_crate_boxes__2_to_3_001",
    "g1000_warehouse_shelf__pallet_rack__absolute_vertical_002",
    "g1000_warehouse_shelf__supply_shelf__absolute_vertical_002",
    "g1000_kitchen_counter__bowl_block_mill__occlusion_state_004",
    "g1000_office_desk__stand_keyboard_pad__occlusion_state_008",
    "g1000_classroom__n3_chair__3_to_2_002",
    "g1000_playground__slide_swing_bench__absolute_horizontal_003",
    "g1000_playground__slide_swing_bench__absolute_horizontal_004",
    "g1000_playground__slide_swing_bench__shape_divergent_replace_002",
    "g1000_classroom__shelf_textbooks_globe__same_affordance_replace_003",
}

COUNT_ADD_SUBTYPES = {"1_to_2", "1_to_3", "2_to_3"}
COUNT_REMOVE_SUBTYPES = {"2_to_1", "3_to_1", "3_to_2"}

CATEGORY_VISUAL_DEFS = {
    "playground climbing dome": (
        "a jungle gym shaped like half a ball, built from straight bars connected in a triangular grid with open gaps between the bars"
    ),
    "top-spiral notepad": (
        "a paper notepad with a clearly visible metal wire spiral binding along its top edge "
        "and a flat stack of pages"
    ),
}

TAIL = (
    "The edit must be obvious when the two images are compared side by side. "
    "Keep a native square, full-bleed scene filling the canvas edge to edge. "
    "Do not introduce an inset image, photo frame, border, blank margin, white wedge, collage, card, poster, page, or mockup. "
    "Do not alter anything beyond this required atomic visual change. "
    "No readable text, labels, letters, numbers, logos, watermarks, signs, or UI."
)


def natural_value(value: Any) -> str:
    return str(value).replace("_", " ")


def _stem(word: str) -> str:
    return word[:-2] if word.endswith("en") and len(word) > 4 else word


def medium_phrase(entity: Dict[str, Any]) -> str:
    """color + material + category, dropping any value whose words (stem-normalized)
    already appear in the category or an earlier value (e.g. 'paper' + 'paper cup',
    'natural wood'/'wood' + 'wooden cutting board')."""
    attrs = entity.get("attributes", {}) or {}
    used = {_stem(word) for word in natural_value(str(entity["category"])).lower().split()}
    parts: list[str] = []
    for key in ("color", "material"):
        value = natural_value(str(attrs.get(key, ""))).strip()
        tokens = value.lower().split() if value else []
        if tokens and all(_stem(token) not in used for token in tokens):
            parts.append(value)
            used.update(_stem(token) for token in tokens)
    parts.append(natural_value(str(entity["category"])))
    return " ".join(parts)


def patch_kind(record: Dict[str, Any]) -> str:
    subtype = str(record["subtype"])
    if subtype in COUNT_ADD_SUBTYPES:
        return "count_add"
    if subtype in COUNT_REMOVE_SUBTYPES:
        return "count_remove"
    if subtype == "occlusion_state":
        return "occlusion"
    if subtype in ("absolute_horizontal", "absolute_vertical"):
        return "spatial"
    if subtype.endswith("_replace"):
        return "replace"
    raise SystemExit(f"{record['pair_id']}: unsupported subtype {subtype!r} for patching")


def new_prompt(record: Dict[str, Any]) -> str:
    kind = patch_kind(record)
    iv = record["intervention"]
    entity = next(item for item in record["source"]["cot"]["entities"] if item["id"] == iv["entity_id"])
    desc = medium_phrase(entity)
    old_value, new_value = iv["old_value"], iv["new_value"]

    if kind in ("count_add", "count_remove"):
        delta = int(new_value) - int(old_value)
        if kind == "count_add":
            core = (
                f"Change only the number of the {desc} from {old_value} to {new_value}. "
                f"Add exactly {delta} more of it right beside the existing one(s) on a free spot of the same surface, with a clear visible gap. "
                f"The new instance must match the existing ones in category, color, material, and size: the SAME SIZE, not smaller, not blurry, not cropped, not half outside the frame. "
                f"After the edit exactly {new_value} instance(s) are clearly visible and countable at one glance, not {old_value}, not {int(new_value) + 1} or more. "
                f"Every other object in the scene is still present in its exact original position. Do not delete or add any other object category. {TAIL}"
            )
        else:
            core = (
                f"Change only the number of the {desc} from {old_value} to {new_value}. "
                f"Remove exactly {int(-delta)} of them, for example the rightmost one. "
                f"Erase it completely and let the bare surface show through where it stood: no outline, no shadow, no leftover. "
                f"CRITICAL: remove exactly {int(-delta)}; after the edit exactly {new_value} instance(s) must remain, not one more and not one less. "
                f"The surviving instance(s) must stay exactly as they appear in the source image: same position, same size, and every detail such as melted wax, drips, stains, or wear preserved. "
                f"Do not swap a surviving instance for a new clean-looking copy. "
                f"Every other object in the scene is still present in its exact original position. Do not delete or add any other object category. {TAIL}"
            )
        return core

    if kind == "occlusion":
        reference = next(
            relation for relation in entity.get("relations", []) if relation["predicate"] == "occlusion_reference"
        )
        occluder = next(
            item for item in record["source"]["cot"]["entities"] if item["id"] == reference["target"]
        )
        occ = medium_phrase(occluder)
        if str(new_value) == "unoccluded":
            geometry = (
                f"TASK: pick up the {occ} and move it away so nothing overlaps anymore. "
                f"Step 1: erase the {occ} from its current spot and fill that area with the uninterrupted background. "
                f"Step 2: redraw the same {occ} in a clearly different spot on the same surface, at least one object width away from the {desc}, with no overlap. "
                f"After the edit both objects remain visible and fully separated, and the complete outline of the {desc} is visible: no part of it is hidden anymore. "
                "The relative arrangement of the two objects must differ from the source image. Do not return the source image unchanged."
            )
        else:
            geometry = (
                f"TASK: pick up the {occ} and place it in front of the {desc} so it clearly hides about one third of the {desc}'s outline. "
                f"Step 1: erase the {occ} from its current spot and fill that area with the uninterrupted background. "
                f"Step 2: redraw the same {occ} overlapping the {desc}, with the hidden part obviously blocked from view. "
                f"After the edit both objects remain visible, and the overlap must be obvious at a glance: a clear part of the {desc} sits behind the {occ}. "
                "The relative arrangement of the two objects must differ from the source image. Do not return the source image unchanged."
            )
        return (
            f"Change only the visibility of the {desc}. {geometry} "
            "Preserve every object identity, count, non-occlusion attribute, background, lighting, camera, and unrelated geometry exactly. "
            "CRITICAL: both objects must still be present in the output; moving one object never means deleting it. "
            f"{TAIL}"
        )

    if kind == "spatial":
        prompt = str(record["target"]["prompt"]).rstrip()
        extra = (
            " QUANTIFIED DISPLACEMENT REQUIREMENT: the object's center must end up at least one third of the frame away from its original position; "
            "a smaller shift counts as no change and is a failure. "
            "Even though the object may be large, move the entire object: do not shrink it, do not tilt it, and leave nothing at the old position."
        )
        return prompt if "QUANTIFIED DISPLACEMENT REQUIREMENT" in prompt else prompt + extra

    # kind == "replace"
    prompt = str(record["target"]["prompt"])
    if "silhouette must clearly differ" not in prompt:
        definition = CATEGORY_VISUAL_DEFS.get(str(new_value))
        if definition:
            prompt = prompt.replace(f"with a {natural_value(new_value)}.", f"with a {natural_value(new_value)}: {definition}.", 1)
        prompt = prompt.rstrip() + (
            " The replacement must be recognizable as the new category at a glance: "
            "its silhouette must clearly differ from the source object's silhouette; "
            "do not draw an abstract blob or an unidentifiable shape."
        )
    return prompt


def check_prompt(pair_id: str, prompt: str) -> None:
    if re.search(r"\b[a-z]+_\d+\b", prompt) or "_region" in prompt:
        raise SystemExit(f"{pair_id}: internal token leaked into patched prompt")
    if re.search(r"\b(\w+)\s+\1\b", prompt.lower()):
        raise SystemExit(f"{pair_id}: repeated-word bigram in patched prompt")
    if "occlusion" in pair_id and "both objects remain visible" not in prompt:
        raise SystemExit(f"{pair_id}: occlusion prompt lost validator phrase")
    if (
        ("absolute_" in pair_id)
        and "do not return the source image unchanged" not in prompt.lower()
    ):
        raise SystemExit(f"{pair_id}: spatial prompt lost displacement phrase")


def main() -> None:
    import re as _re

    patched: Dict[str, int] = {}
    chunk_paths = sorted(
        path for path in CHUNK_DIR.glob("chunk_*.jsonl")
        if _re.fullmatch(r"chunk_\d{2}\.jsonl", path.name)
    )
    for chunk_path in chunk_paths:
        records = []
        hit_ids = []
        with chunk_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("pair_id") in PATCH_PAIR_IDS:
                    old_prompt = str(record["target"]["prompt"])
                    record["target"]["prompt"] = new_prompt(record)
                    record["target"]["prompt_retry_v2"] = True
                    record["target"]["prompt_v1_backup"] = old_prompt
                    check_prompt(record["pair_id"], record["target"]["prompt"])
                    hit_ids.append(record["pair_id"])
                records.append(record)
        if not hit_ids:
            continue
        output_path = chunk_path.with_name(chunk_path.stem + ".retry_v2.jsonl")
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        patched[chunk_path.name] = len(hit_ids)
        for pair_id in hit_ids:
            print(f"patched {pair_id} -> {output_path.name}")

    total = sum(patched.values())
    patched_ids = set()
    for chunk_path in chunk_paths:
        retry_path = chunk_path.with_name(chunk_path.stem + ".retry_v2.jsonl")
        if not retry_path.is_file():
            continue
        with retry_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("target", {}).get("prompt_retry_v2") is True:
                    patched_ids.add(record["pair_id"])
    missing = sorted(PATCH_PAIR_IDS - patched_ids)
    print(json.dumps({"patched_total": total, "unique_patched": len(patched_ids), "chunks": patched, "expected": len(PATCH_PAIR_IDS), "missing": missing}, indent=2))
    if missing:
        raise SystemExit("not every flagged pair was patched")


if __name__ == "__main__":
    main()
