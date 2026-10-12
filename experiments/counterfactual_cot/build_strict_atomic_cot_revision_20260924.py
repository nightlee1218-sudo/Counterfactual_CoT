#!/usr/bin/env python3
"""Build a strict-v4 atomic-COT revision manifest for round-two repairs.

This script only writes isolated planning artifacts. It never changes the
original 1000-pair manifest or any original image.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

BAGEL_ROOT = Path("/group/40016/niightli/BAGEL")
ROOT = BAGEL_ROOT / "experiments/counterfactual_cot/gold1000_framework_strict_v4"
DATA = ROOT / "full_1000_20260918"
ROUND2_PLAN = ROOT / "detailed_repair_plan_20260924_round2.jsonl"
ORIGINAL_MANIFEST = DATA / "gold1000.generated.jsonl"
ORIGINAL_REGISTRY = DATA / "source_registry.generated.jsonl"
OUTPUT_DIR = Path(os.environ.get(
    "STRICT_ATOMIC_COT_OUTPUT_DIR",
    str(ROOT / "repair_round2_strict_atomic_20260924"),
))
REVISED_MANIFEST = OUTPUT_DIR / "revised_cot_manifest.jsonl"
REVISED_REGISTRY = OUTPUT_DIR / "revised_source_registry.jsonl"
COT_PLAN = OUTPUT_DIR / "cot_revision_plan.jsonl"
TRACKING = OUTPUT_DIR / "round2_cot_tracking.jsonl"
BLOCKED = OUTPUT_DIR / "blocked_semantic_review.jsonl"
SUMMARY = OUTPUT_DIR / "summary.json"

if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.build_second_round_repair_plan_20260924 import SPECS

LOCATIONS = {
    "left_region",
    "center_region",
    "right_region",
    "top_region",
    "bottom_region",
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
}
LOCATION_WORDS = {
    "left_region": "the left region",
    "center_region": "the center region",
    "right_region": "the right region",
    "top_region": "the top region",
    "bottom_region": "the bottom region",
    "upper_left": "the upper-left region",
    "upper_right": "the upper-right region",
    "lower_left": "the lower-left region",
    "lower_right": "the lower-right region",
}

# These are baseline revisions shared by source and target. They do not count
# as the source->target intervention. The final source and target must still
# differ by exactly one canonical COT atom.
LAYOUT_REQUIREMENTS = {
    "g1000_bedroom_nightstand__glasses_charger_glass__occlusion_state_008": "Place the bedside lamp and water glass on the same support surface with a clear visible gap in the source; both full silhouettes must remain visible.",
    "g1000_bedroom_nightstand__lamp_glass_tray__absolute_vertical_001": "Use a compact mini table lamp that is fully visible, with its base, stem, and shade inside the frame; keep a full lamp-height clearance in the lower-left destination.",
    "g1000_bedroom_nightstand__lamp_glass_tray__absolute_vertical_004": "Use a compact mini table lamp that is fully visible and leave a complete lamp-height clearance in the center destination.",
    "g1000_bookshelf_study__mugcollection_folder__absolute_vertical_001": "Show two clearly separated shelf levels, keep the upright folder on the upper level, and leave the lower destination completely open for a full vertical move.",
    "g1000_bookshelf_study__penjar_folder_trophy__absolute_vertical_001": "Place the trophy in the middle or lower part of the right side, clearly away from the upper-right destination, with both positions physically measurable.",
    "g1000_classroom__chair_backpack_notebook__1_to_2_002": "Keep exactly one jacket in one vertical stack, leave one jacket-height clearance above it, and keep the notebook beside the stack fully visible.",
    "g1000_classroom__n3_chair__3_to_2_002": "Show exactly three individual chairs with non-overlapping silhouettes in one clearly defined group; do not create background chairs that can be counted as part of the group.",
    "g1000_closet_wardrobe__beanie_hatbox_scarf__1_to_2_002": "Show exactly one knitted beanie in one stack and leave one beanie-height clearance directly above it; do not create a second stack.",
    "g1000_closet_wardrobe__n2_sweater__2_to_3_002": "Show exactly two folded sweaters as two clearly countable layers in one stack, with one sweater-height clearance above the stack and no second stack.",
    "g1000_closet_wardrobe__socks_basket_pajamas_hat__occlusion_state_001": "Make the specified partial overlap clear while keeping the hat laterally offset from the sweater rather than directly above it; both objects remain recognizable.",
    "g1000_dining_table__napkin_fill__3_to_2_001": "Show exactly three individual napkins with clearly separated edges and no full occlusion by the plate or bowl.",
    "g1000_dining_table__plate_quota__2_to_3_001": "Keep two plates clearly grouped with one free plate-sized position, and keep all three water glasses away from the plate expansion path.",
    "g1000_grocery_display__biscuit_tea_honey__3_to_2_001": "Show exactly three individual honey jars in one clear group with no hidden or duplicate row; keep the biscuit tin and tea caddy separate.",
    "g1000_grocery_display__flowers_vase_plants__absolute_horizontal_001": "Show exactly one intervention plant with a complete unobstructed path between its old and new horizontal positions; keep the crate and basket out of that path.",
    "g1000_grocery_display__gd04b_flowers_vase_separate__containment_001": "Show exactly one complete flower beside one open vase, with enough visible space to make the containment change auditable.",
    "g1000_grocery_display__gd05b_milk_crate_separate__containment_001": "Show exactly one complete milk carton beside one open crate, with the crate opening and the full movement path visible.",
    "g1000_living_room_play__n1_rattle__2_to_1_001": "Show exactly two baby rattles with fully separated handles and bodies in one group; keep unrelated toys away from that group.",
    "g1000_living_room_play__n1_rattle__2_to_3_001": "Show exactly two baby rattles with fully separated handles and bodies, and leave one same-size position in the same group for the third rattle.",
    "g1000_living_room_play__n1_rattle__3_to_2_001": "Show exactly three baby rattles with fully separated handles and bodies in one group; the cube group remains a separate preserved object group.",
    "g1000_living_room_play__puzzle_fill__3_to_2_001": "Show exactly three separate puzzle pieces with individually visible outlines; do not depict one assembled puzzle as three pieces.",
    "g1000_office_desk__stand_keyboard_pad__occlusion_state_002": "Make the specified partial overlap clear at roughly one-quarter to one-third of the occluded outline while keeping both objects recognizable.",
    "g1000_pantry_shelf_closeup__cans_open_crate_out__containment_001": "Show every can outside one open crate with all can bodies individually countable, and leave the crate opening large enough for all of them to remain partly visible after insertion.",
    "g1000_pantry_shelf_closeup__jars_open_box_out__containment_001": "Show every jar outside one open box with all jar bodies individually countable, and leave enough box opening for every jar to remain visibly identifiable after insertion.",
    "g1000_toolshed_garage__hammer_oilcan_basket__functional_state_002": "Use one compact hinged metal toolbox with a clearly visible lid and enough surrounding space to distinguish open from closed.",
    "g1000_warehouse_shelf__bag_quota__3_to_2_001": "Show exactly three individual paper bags in one clear group, with non-overlapping silhouettes and no second row; keep the paint cans as a separate preserved group.",
    "g1000_warehouse_shelf__box_sack_pallet__2_to_1_001": "Show exactly two pallets in two clearly separated support positions, never perfectly stacked or hidden behind the box or sack.",
    "g1000_warehouse_shelf__box_tote_drum__2_to_1_001": "Show exactly two target boxes together in one compact continuous group, not as two repeated rows, with a clear gap after removing one.",
    "g1000_warehouse_shelf__drum_pallet_tote__2_to_3_001": "Show exactly two same-size pallets in one group with one same-size position directly above the original pallet area; do not solve capacity by resizing.",
    "g1000_warehouse_shelf__helmets_gearbin_handtruck__2_to_3_001": "Use a larger open gear bin while keeping exactly two helmets fully countable and leaving one same-size helmet position available; the handtruck does not occlude them.",
    "g1000_warehouse_shelf__helmets_gearbin_handtruck__containment_001": "Show exactly two helmets inside one open gear bin with both helmet silhouettes sufficiently exposed to count, and leave the exterior destination clear.",
    "g1000_warehouse_shelf__sack_bundle_bin__2_to_1_001": "Keep the two intervention sacks in one fixed group so that removing one does not require moving the surviving sack or any unrelated object.",
    "g1000_warehouse_shelf__sack_bundle_bin__2_to_3_001": "Show exactly two same-size sacks together in one continuous group with one same-size position beside or above them; do not create a second row or second pile.",
}

# Explicit semantic baseline revisions requested by the user. They are applied
# to both source and target unless marked source_only. The intervention remains
# one field change after these revisions.
COMMON_ENTITY_PATCHES: dict[str, list[dict[str, Any]]] = {
    "g1000_bedroom_nightstand__lamp_glass_tray__absolute_vertical_001": [
        {"entity": "lamp_1", "field": "category", "value": "compact ceramic table lamp"},
        {"entity": "lamp_1", "field": "attributes.size_class", "value": "mini"},
    ],
    "g1000_bedroom_nightstand__lamp_glass_tray__absolute_vertical_004": [
        {"entity": "lamp_1", "field": "category", "value": "compact ceramic table lamp"},
        {"entity": "lamp_1", "field": "attributes.size_class", "value": "mini"},
    ],
    "g1000_bookshelf_study__mugcollection_folder__absolute_vertical_001": [
        {"entity": "folder_2", "field": "attributes.orientation", "value": "upright"},
    ],
    "g1000_closet_wardrobe__beanie_hatbox_scarf__1_to_2_002": [
        {"entity": "scarf_1", "field": "id", "value": "beanie_1"},
        {"entity": "beanie_1", "field": "category", "value": "knitted beanie"},
    ],
    "g1000_grocery_display__flowers_vase_plants__absolute_horizontal_001": [
        {"entity": "plants_1", "field": "count", "value": 1},
    ],
    "g1000_grocery_display__gd04b_flowers_vase_separate__containment_001": [
        {"entity": "flowers_1", "field": "count", "value": 1},
    ],
    "g1000_grocery_display__gd05b_milk_crate_separate__containment_001": [
        {"entity": "milk_1", "field": "count", "value": 1},
    ],
    "g1000_toolshed_garage__hammer_oilcan_basket__functional_state_002": [
        {"entity": "oilcan_1", "field": "id", "value": "toolbox_1"},
        {"entity": "toolbox_1", "field": "category", "value": "compact hinged metal toolbox"},
    ],
}
SOURCE_ONLY_PATCHES: dict[str, list[dict[str, Any]]] = {
    "g1000_bookshelf_study__penjar_folder_trophy__absolute_vertical_001": [
        {"entity": "trophy_1", "field": "location", "value": "lower_right"},
    ],
}

# These two revisions deliberately split an old shared source scene because the
# requested old count is different from the old shared source state.
INTERVENTION_OVERRIDES: dict[str, dict[str, Any]] = {
    "g1000_warehouse_shelf__bag_quota__3_to_2_001": {
        "entity_id": "bag_1",
        "field": "count",
        "old_value": 3,
        "new_value": 2,
    },
    "g1000_living_room_play__n1_rattle__3_to_2_001": {
        "entity_id": "rattle_1",
        "field": "count",
        "old_value": 3,
        "new_value": 2,
    },
}

# These source variants are intentionally not shared with the old count=2 or
# old object-state variants in the same original scene.
SPLIT_VARIANT_PAIR_IDS = set(INTERVENTION_OVERRIDES)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def sha_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def strict_render(cot: dict[str, Any]) -> str:
    parts: list[str] = []
    for entity in cot["entities"]:
        eid = entity["id"]
        parts.extend([f"{eid}.object={entity['category']}", f"{eid}.count={entity['count']}", f"{eid}.location={entity['location']}"])
        parts.extend(f"{eid}.attribute.{key}={value}" for key, value in entity.get("attributes", {}).items())
        parts.extend(
            f"{eid}.relation.{index}={relation['predicate']}:{relation['target']}"
            for index, relation in enumerate(entity.get("relations", []))
        )
    parts.extend(f"background.{key}={value}" for key, value in cot["background"].items())
    return "; ".join(parts)


def entity(cot: dict[str, Any], entity_id: str) -> dict[str, Any]:
    for item in cot["entities"]:
        if item["id"] == entity_id:
            return item
    raise KeyError(f"missing entity {entity_id}")


def get_field(cot: dict[str, Any], entity_id: str, field: str) -> Any:
    item = entity(cot, entity_id)
    if field == "category":
        return item["category"]
    if field == "count":
        return item["count"]
    if field == "location":
        return item["location"]
    if field.startswith("attributes."):
        return item.setdefault("attributes", {}).get(field.split(".", 1)[1])
    if field == "relations":
        relations = item.get("relations", [])
        if len(relations) != 1:
            return relations
        relation = relations[0]
        return f"{relation['predicate']}:{relation['target']}"
    raise ValueError(field)


def set_field(cot: dict[str, Any], entity_id: str, field: str, value: Any) -> None:
    item = entity(cot, entity_id)
    if field == "category":
        item["category"] = str(value)
    elif field == "count":
        item["count"] = int(value)
    elif field == "location":
        if value not in LOCATIONS:
            raise ValueError(f"invalid location {value}")
        item["location"] = value
    elif field.startswith("attributes."):
        item.setdefault("attributes", {})[field.split(".", 1)[1]] = value
    elif field == "relations":
        if isinstance(value, str):
            predicate, target = value.split(":", 1)
            item["relations"] = [{"predicate": predicate, "target": target}]
        else:
            item["relations"] = copy.deepcopy(value)
    else:
        raise ValueError(field)


def rename_entity(cot: dict[str, Any], old_id: str, new_id: str) -> None:
    if old_id == new_id:
        return
    if any(item["id"] == new_id for item in cot["entities"]):
        raise ValueError(f"cannot rename {old_id} to existing {new_id}")
    for item in cot["entities"]:
        if item["id"] == old_id:
            item["id"] = new_id
            break
    else:
        raise KeyError(old_id)
    for item in cot["entities"]:
        for relation in item.get("relations", []):
            if relation.get("target") == old_id:
                relation["target"] = new_id


def apply_patch(cot: dict[str, Any], patch: dict[str, Any]) -> None:
    if patch["field"] == "id":
        rename_entity(cot, patch["entity"], str(patch["value"]))
    else:
        set_field(cot, patch["entity"], patch["field"], patch["value"])


def apply_intervention(cot: dict[str, Any], intervention: dict[str, Any]) -> dict[str, Any]:
    target = copy.deepcopy(cot)
    entity_id = intervention["entity_id"]
    field = intervention["field"]
    old_value = intervention["old_value"]
    new_value = intervention["new_value"]
    actual = get_field(target, entity_id, field)
    if field == "relations" and isinstance(old_value, str):
        if actual != old_value:
            raise ValueError(f"old relation mismatch: {entity_id}.{field}: {actual!r} != {old_value!r}")
    elif actual != old_value:
        raise ValueError(f"old value mismatch: {entity_id}.{field}: {actual!r} != {old_value!r}")
    set_field(target, entity_id, field, new_value)
    return target


def atom_key(entity_id: str, field: str) -> str:
    if field == "category":
        return f"{entity_id}.object"
    if field in {"count", "location"}:
        return f"{entity_id}.{field}"
    if field.startswith("attributes."):
        return f"{entity_id}.attribute.{field.split('.', 1)[1]}"
    if field == "relations":
        return f"{entity_id}.relation.0"
    raise ValueError(field)


def preservation_for(cot: dict[str, Any], intervention: dict[str, Any]) -> dict[str, list[str]]:
    changed = atom_key(intervention["entity_id"], intervention["field"])
    must_preserve: list[str] = []
    for atom in strict_render(cot).split("; "):
        key = atom.split("=", 1)[0]
        if key == changed:
            continue
        if key.endswith(".object"):
            must_preserve.append(key[:-6] + ".category")
        elif ".attribute." in key:
            left, right = key.split(".attribute.", 1)
            must_preserve.append(f"{left}.attributes.{right}")
        elif ".relation." in key:
            left = key.rsplit(".relation.", 1)[0]
            must_preserve.append(f"{left}.relations")
        else:
            must_preserve.append(key)
    allowed = f"{intervention['entity_id']}.{intervention['field']}"
    return {
        "allowed_to_change": [allowed],
        "must_preserve": must_preserve,
        "must_revalidate": [allowed],
    }


def location_text(value: str) -> str:
    return LOCATION_WORDS.get(value, str(value).replace("_", " "))


def describe_entity(item: dict[str, Any]) -> str:
    attrs = item.get("attributes", {})
    details = []
    for key in ("color", "material", "pattern", "size_class", "orientation", "functional_state", "occlusion_state"):
        if key in attrs:
            value = str(attrs[key]).replace("_", " ")
            details.append(f"{key.replace('_', ' ')} {value}")
    detail_text = f" ({', '.join(details)})" if details else ""
    return f"{item['count']} {item['category']}{detail_text} in {location_text(item['location'])}"


def relation_text(cot: dict[str, Any]) -> str:
    relations = []
    for item in cot["entities"]:
        for relation in item.get("relations", []):
            if relation["predicate"] == "occlusion_reference":
                continue
            target = entity(cot, relation["target"])
            relations.append(f"the {item['category']} is {relation['predicate'].replace('_', ' ')} the {target['category']}")
    return "; ".join(relations)


def base_prompt(cot: dict[str, Any]) -> str:
    background = cot["background"]
    entity_text = "; ".join(describe_entity(item) for item in cot["entities"])
    relation = relation_text(cot)
    relation_text_part = f" Relationship requirement: {relation}." if relation else ""
    return (
        f"Create a natural 1024x1024 square full-bleed image of {background.get('scene', 'the specified scene')}. "
        f"Show exactly {entity_text}. "
        f"Physical layout requirement: {background.get('layout_anchor', '')}. "
        f"The surface is {background.get('shelf_material', background.get('countertop_material', background.get('floor_material', 'a stable support surface')))} "
        f"with {background.get('wall', background.get('lighting', 'natural ambient light'))}."
        f"{relation_text_part} Every listed object is distinct, countable, complete, and inside the frame. "
        "Use stable eye-level composition, no crop, no frame, no border, no blank margin, no collage, no text, no logo, no watermark, no UI, and no readable text."
    )


def humanize(text: str, cot: dict[str, Any]) -> str:
    result = str(text)
    labels = {item["id"]: item["category"] for item in cot["entities"]}
    for key in sorted(labels, key=len, reverse=True):
        result = result.replace(key, f"the {labels[key]}")
    result = re.sub(r"\b[a-z]+_\d+\b", "the specified object", result)
    result = result.replace("entity_id=", "the specified ")
    result = result.replace("source COT", "fixed scene plan").replace("target COT", "target scene plan")
    for key, value in LOCATION_WORDS.items():
        result = result.replace(key, value)
    result = result.replace("COT", "fixed scene plan")
    return result


def source_geometry_clause(intervention: dict[str, Any]) -> str:
    if intervention.get("field") != "attributes.occlusion_state":
        return ""
    if str(intervention.get("old_value")) == "unoccluded":
        return "The source must have a clear gap and no overlap; both objects are visible."
    return "The source must show both objects are visible with approximately one third of the target outline occluded."


def target_action(source: dict[str, Any], intervention: dict[str, Any]) -> str:
    item = entity(source, intervention["entity_id"])
    field = intervention["field"]
    old = intervention["old_value"]
    new = intervention["new_value"]
    descriptor = f"the {item['category']}"
    if field == "count":
        return f"Change only the number of {descriptor} from {old} to {new}."
    if field == "location":
        return f"Move only {descriptor} from {location_text(str(old))} to {location_text(str(new))}."
    if field == "relations":
        return f"Change only the relationship of {descriptor} from {str(old).replace('_', ' ')} to {str(new).replace('_', ' ')}."
    if field.startswith("attributes."):
        name = field.split(".", 1)[1].replace("_", " ")
        return f"Change only the {name} of {descriptor} from {old} to {new}."
    return f"Change only the category of {descriptor} from {old} to {new}."


def make_target_prompt(source: dict[str, Any], intervention: dict[str, Any], measure: str) -> str:
    geometry = []
    field = intervention["field"]
    if field == "location":
        geometry.append(
            "The displacement must be obvious across a different quadrant of the frame; do not return the source image unchanged."
        )
    if field == "relations":
        new_predicate = str(intervention["new_value"]).split(":", 1)[0]
        geometry.append(
            "The container rim stays visible and every moved instance remains countable."
            if new_predicate == "inside"
            else "Maintain a clear empty gap between the separated objects."
        )
    if field == "attributes.occlusion_state":
        geometry.append("Both objects remain visible and recognizable.")
    if field == "attributes.occlusion_state":
        if str(intervention["old_value"]) == "unoccluded":
            geometry.append("The source relationship has a clear gap and no overlap; both objects are visible.")
        else:
            geometry.append("The source relationship keeps both objects visible with approximately one third of the target outline occluded.")
    return (
        target_action(source, intervention)
        + " Preserve every other entity identity, count, position, category, attribute, relation, background, lighting, camera, and composition exactly. "
        + "For a count change, add or remove only individual instances in the same original group; never create a second group or row. "
        + "For a move or containment change, the original position must be empty and the object must appear exactly once at the new position. "
        + "For an occlusion change, both objects remain visible and recognizable. "
        + humanize(measure, source)
        + " " + " ".join(geometry)
        + " Create a native 1024x1024 square full-bleed image with one and only one atomic visual change, no crop, frame, border, blank margin, collage, no readable text, no text, logo, watermark, or UI."
    )


def append_layout(cot: dict[str, Any], clauses: list[str]) -> None:
    if not clauses:
        return
    base = str(cot["background"].get("layout_anchor", "")).strip().rstrip(".")
    additions = [clause.strip().rstrip(".") for clause in clauses if clause.strip()]
    merged = ". ".join(dict.fromkeys([base] + additions))
    cot["background"]["layout_anchor"] = merged + "."


def update_intervention(original: dict[str, Any], source: dict[str, Any], pair_id: str) -> dict[str, Any]:
    intervention = copy.deepcopy(original["intervention"])
    if pair_id in INTERVENTION_OVERRIDES:
        intervention.update(INTERVENTION_OVERRIDES[pair_id])
    if pair_id == "g1000_bookshelf_study__penjar_folder_trophy__absolute_vertical_001":
        intervention["old_value"] = "lower_right"
    if pair_id == "g1000_closet_wardrobe__beanie_hatbox_scarf__1_to_2_002":
        intervention["entity_id"] = "beanie_1"
    if pair_id == "g1000_toolshed_garage__hammer_oilcan_basket__functional_state_002":
        intervention["entity_id"] = "toolbox_1"
    if get_field(source, intervention["entity_id"], intervention["field"]) != intervention["old_value"]:
        raise ValueError(
            f"{pair_id}: revised source does not match intervention old_value: "
            f"{get_field(source, intervention['entity_id'], intervention['field'])!r} != {intervention['old_value']!r}"
        )
    return intervention


def make_variant_groups(original_by_id: dict[str, dict[str, Any]], plan_by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    pair_to_group: dict[str, str] = {}
    for pair_id, plan in plan_by_id.items():
        if plan.get("repair_scope") != "source+target":
            continue
        if pair_id in SPLIT_VARIANT_PAIR_IDS:
            key = f"split::{pair_id}"
            groups[key].add(pair_id)
            pair_to_group[pair_id] = key
            continue
        variant_key = str(plan.get("source_variant_key") or f"single::{pair_id}")
        groups[variant_key].update(str(value) for value in plan.get("source_variant_member_pair_ids", [pair_id]))
        pair_to_group[pair_id] = variant_key
    # Remove deliberately split pairs from their old shared variant.
    for split_pair in SPLIT_VARIANT_PAIR_IDS:
        for key in list(groups):
            if key != f"split::{split_pair}":
                groups[key].discard(split_pair)
    for key in list(groups):
        groups[key] = {pair_id for pair_id in groups[key] if pair_id in original_by_id}
        if not groups[key]:
            del groups[key]
    for key, members in groups.items():
        for pair_id in members:
            pair_to_group[pair_id] = key
    return groups, pair_to_group


def group_direct_pairs(group: set[str], plan_by_id: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        pair_id
        for pair_id in group
        if pair_id in SPECS and plan_by_id.get(pair_id, {}).get("repair_scope") == "source+target"
    )


def apply_common_revision(cot: dict[str, Any], direct_pair_ids: list[str]) -> list[dict[str, Any]]:
    applied: list[dict[str, Any]] = []
    for pair_id in direct_pair_ids:
        for patch in COMMON_ENTITY_PATCHES.get(pair_id, []):
            apply_patch(cot, patch)
            applied.append({"pair_id": pair_id, **patch})
    return applied


def build_source_variant(
    base_record: dict[str, Any],
    group: set[str],
    direct_pair_ids: list[str],
    group_key: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = copy.deepcopy(base_record["source"]["cot"])
    applied_common = apply_common_revision(source, direct_pair_ids)
    source_only_applied: list[dict[str, Any]] = []
    for pair_id in direct_pair_ids:
        for patch in SOURCE_ONLY_PATCHES.get(pair_id, []):
            apply_patch(source, patch)
            source_only_applied.append({"pair_id": pair_id, **patch})
    for pair_id in direct_pair_ids:
        if pair_id == "g1000_warehouse_shelf__bag_quota__3_to_2_001":
            set_field(source, "bag_1", "count", 3)
            source_only_applied.append({"pair_id": pair_id, "entity": "bag_1", "field": "count", "value": 3, "reason": "revised old state requested by user"})
        if pair_id == "g1000_living_room_play__n1_rattle__3_to_2_001":
            set_field(source, "rattle_1", "count", 3)
            source_only_applied.append({"pair_id": pair_id, "entity": "rattle_1", "field": "count", "value": 3, "reason": "revised old state requested by user"})
    clauses = [LAYOUT_REQUIREMENTS[pair_id] for pair_id in direct_pair_ids if pair_id in LAYOUT_REQUIREMENTS]
    append_layout(source, clauses)
    revisions = applied_common + source_only_applied
    return source, revisions


def source_registry_record(original_registry: dict[str, Any], scene_id: str, source_cot: dict[str, Any], prompt: str, original_scene_id: str, revisions: list[dict[str, Any]]) -> dict[str, Any]:
    record = copy.deepcopy(original_registry)
    record["scene_id"] = scene_id
    record["cot"] = copy.deepcopy(source_cot)
    record["cot_text"] = strict_render(source_cot)
    record["prompt"] = prompt
    record["image"] = f"source/_by_scene/cf_source_{scene_id}__source.png"
    record["scene_hash"] = sha_json({"cot": source_cot, "prompt": prompt})
    record["source_revision"] = {
        "schema_version": "strict_atomic_source_revision_v1",
        "audit_round": 2,
        "original_source_scene_id": original_scene_id,
        "revisions": revisions,
        "status": "pending_generation",
    }
    record["source_quality"] = {"accepted": False, "review_status": "pending_round2_manual_audit", "review_notes": ""}
    return record


def build_row(
    original: dict[str, Any],
    pair_id: str,
    source_cot: dict[str, Any],
    revised_scene_id: str,
    source_revisions: list[dict[str, Any]],
    source_regenerated: bool,
    source_prompt_text: str | None = None,
) -> dict[str, Any]:
    row = copy.deepcopy(original)
    intervention = update_intervention(original, source_cot, pair_id)
    target_cot = apply_intervention(source_cot, intervention)
    row["source_scene_id"] = revised_scene_id
    row["source"]["cot"] = source_cot
    row["source"]["cot_text"] = strict_render(source_cot)
    row["source"]["prompt"] = source_prompt_text or (
        base_prompt(source_cot)
        + " "
        + humanize(LAYOUT_REQUIREMENTS.get(pair_id, "Keep every listed object clearly separated and fully visible."), source_cot)
        + " "
        + source_geometry_clause(intervention)
    )
    row["target"]["cot"] = target_cot
    row["target"]["cot_text"] = strict_render(target_cot)
    measure = SPECS.get(pair_id, {}).get("target_extra", "")
    row["target"]["prompt"] = make_target_prompt(source_cot, intervention, measure)
    row["intervention"] = intervention
    row["preservation"] = preservation_for(source_cot, intervention)
    row["source"]["image"] = (
        f"source/_by_scene/cf_source_{revised_scene_id}__source.png"
        if source_regenerated
        else f"source/{pair_id}__source.png"
    )
    row["target"]["image"] = f"target/{pair_id}__target.png"
    row["target"]["mask"] = f"masks/{pair_id}__mask.png"
    row["repair_round2"] = {
        "schema_version": "strict_atomic_cot_revision_v1",
        "audit_round": 2,
        "status": "ready_for_generation",
        "original_source_scene_id": original["source_scene_id"],
        "revised_source_scene_id": revised_scene_id,
        "source_regenerated": source_regenerated,
        "baseline_revisions": source_revisions,
        "atomic_change": {
            "entity_id": intervention["entity_id"],
            "field": intervention["field"],
            "old_value": intervention["old_value"],
            "new_value": intervention["new_value"],
            "allowed_to_change": row["preservation"]["allowed_to_change"],
            "changed_atom_count": 1,
        },
        "source_cot_sha256": sha_json(source_cot),
        "target_cot_sha256": sha_json(target_cot),
        "source_revisions": source_revisions,
        "generation_status": "pending_generation",
        "audit_status": "blocked_until_round2_manual_audit",
        "fusion_status": "not_ready",
    }
    return row


def atom_diff(source: dict[str, Any], target: dict[str, Any]) -> list[tuple[str, str]]:
    source_atoms = strict_render(source).split("; ")
    target_atoms = strict_render(target).split("; ")
    if len(source_atoms) != len(target_atoms):
        return [(f"length={len(source_atoms)}", f"length={len(target_atoms)}")]
    return [(left, right) for left, right in zip(source_atoms, target_atoms) if left != right]


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    pair_id = row["pair_id"]
    source = row["source"]["cot"]
    target = row["target"]["cot"]
    intervention = row["intervention"]
    if row["source"]["cot_text"] != strict_render(source):
        errors.append(f"{pair_id}: source cot_text is not canonical")
    if row["target"]["cot_text"] != strict_render(target):
        errors.append(f"{pair_id}: target cot_text is not canonical")
    diffs = atom_diff(source, target)
    if len(diffs) != 1:
        errors.append(f"{pair_id}: changed_atom_count={len(diffs)} diffs={diffs}")
    else:
        old_atom, new_atom = diffs[0]
        expected_key = atom_key(intervention["entity_id"], intervention["field"])
        if old_atom.split("=", 1)[0] != expected_key or new_atom.split("=", 1)[0] != expected_key:
            errors.append(f"{pair_id}: atomic key mismatch {diffs} != {expected_key}")
        if old_atom.split("=", 1)[1] != str(intervention["old_value"]):
            errors.append(f"{pair_id}: atomic old value mismatch")
        if new_atom.split("=", 1)[1] != str(intervention["new_value"]):
            errors.append(f"{pair_id}: atomic new value mismatch")
    if get_field(source, intervention["entity_id"], intervention["field"]) != intervention["old_value"]:
        errors.append(f"{pair_id}: source does not match intervention old value")
    expected_target = apply_intervention(source, intervention)
    if expected_target != target:
        errors.append(f"{pair_id}: target is not deterministic application of intervention")
    preservation = row["preservation"]
    allowed = preservation.get("allowed_to_change", [])
    expected_allowed = [f"{intervention['entity_id']}.{intervention['field']}"]
    if allowed != expected_allowed:
        errors.append(f"{pair_id}: allowed_to_change={allowed} expected={expected_allowed}")
    if set(allowed) & set(preservation.get("must_preserve", [])):
        errors.append(f"{pair_id}: preservation overlap")
    for item in source["entities"] + target["entities"]:
        if item["location"] not in LOCATIONS:
            errors.append(f"{pair_id}: invalid location {item['location']}")
        if not 1 <= int(item["count"]) <= 5:
            errors.append(f"{pair_id}: invalid count {item['count']}")
    if "repair_round2" not in row or row["repair_round2"]["atomic_change"]["changed_atom_count"] != 1:
        errors.append(f"{pair_id}: missing atomic revision metadata")
    return errors


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise RuntimeError(f"output directory is non-empty; refusing to overwrite: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    original_rows = load_jsonl(ORIGINAL_MANIFEST)
    original_by_id = {str(row["pair_id"]): row for row in original_rows}
    plan_rows = load_jsonl(ROUND2_PLAN)
    plan_by_id = {str(row["pair_id"]): row for row in plan_rows}
    missing_direct = sorted(set(SPECS) - set(original_by_id))
    if missing_direct:
        raise RuntimeError(f"direct pair missing from original manifest: {missing_direct}")
    registry_by_scene = {str(row["scene_id"]): row for row in load_jsonl(ORIGINAL_REGISTRY)}
    groups, pair_to_group = make_variant_groups(original_by_id, plan_by_id)
    for group_key in list(groups):
        if not group_direct_pairs(groups[group_key], plan_by_id):
            for pair_id in groups[group_key]:
                pair_to_group.pop(pair_id, None)
            del groups[group_key]

    # Build source variants first. A split pair gets a private revised scene id.
    source_cots: dict[str, dict[str, Any]] = {}
    source_revisions: dict[str, list[dict[str, Any]]] = {}
    revised_scene_ids: dict[str, str] = {}
    source_prompts: dict[str, str] = {}
    registry_rows: list[dict[str, Any]] = []
    for group_key, members in sorted(groups.items()):
        base_pair = sorted(members)[0]
        base_record = original_by_id[base_pair]
        direct_pair_ids = group_direct_pairs(members, plan_by_id)
        source_cot, revisions = build_source_variant(base_record, members, direct_pair_ids, group_key)
        original_scene_id = str(base_record["source_scene_id"])
        revised_scene_id = f"{original_scene_id}__r2_{sha_json(source_cot)[:12]}"
        source_cots[group_key] = source_cot
        source_revisions[group_key] = revisions
        for pair_id in members:
            revised_scene_ids[pair_id] = revised_scene_id
        registry_source = registry_by_scene.get(original_scene_id)
        if registry_source is None:
            raise RuntimeError(f"source registry missing original scene: {original_scene_id}")
        source_clauses = [
            humanize(LAYOUT_REQUIREMENTS[pid], source_cot)
            for pid in direct_pair_ids
            if LAYOUT_REQUIREMENTS.get(pid)
        ]
        source_clauses.extend(
            source_geometry_clause(update_intervention(original_by_id[pid], source_cot, pid))
            for pid in direct_pair_ids
        )
        prompt = base_prompt(source_cot) + " " + " ".join(dict.fromkeys(clause for clause in source_clauses if clause))
        source_prompts[group_key] = prompt
        registry_rows.append(source_registry_record(registry_source, revised_scene_id, source_cot, prompt, original_scene_id, revisions))

    # Direct target-only rows stay on their prior source baseline; all rows in a
    # regenerated variant use the revised source COT and are included together.
    ready_ids: set[str] = set()
    for pair_id in SPECS:
        ready_ids.add(pair_id)
    for group_key, members in groups.items():
        ready_ids.update(members)
    # The user-requested split variants intentionally do not pull old shared
    # siblings into the revised source package.
    for split_pair in SPLIT_VARIANT_PAIR_IDS:
        old_group = pair_to_group.get(split_pair)
        if old_group:
            ready_ids.discard(split_pair)
            ready_ids.add(split_pair)
    ready_ids = {pair_id for pair_id in ready_ids if pair_id in original_by_id}

    revised_rows: list[dict[str, Any]] = []
    plan_output: list[dict[str, Any]] = []
    errors: list[str] = []
    for pair_id in sorted(ready_ids):
        original = original_by_id[pair_id]
        group_key = pair_to_group.get(pair_id)
        source_regenerated = group_key is not None
        if source_regenerated:
            source_cot = copy.deepcopy(source_cots[group_key])
            revisions = copy.deepcopy(source_revisions[group_key])
            revised_scene_id = revised_scene_ids[pair_id]
        else:
            source_cot = copy.deepcopy(original["source"]["cot"])
            revisions = []
            revised_scene_id = str(original["source_scene_id"])
        row = build_row(
            original,
            pair_id,
            source_cot,
            revised_scene_id,
            revisions,
            source_regenerated,
            source_prompts.get(group_key) if source_regenerated else None,
        )
        errors.extend(validate_row(row))
        revised_rows.append(row)
        plan_output.append(
            {
                "schema_version": "strict_atomic_cot_revision_v1",
                "pair_id": pair_id,
                "status": "ready_for_generation",
                "original_source_scene_id": original["source_scene_id"],
                "revised_source_scene_id": revised_scene_id,
                "source_regenerated": source_regenerated,
                "source_variant_key": group_key,
                "source_variant_members": sorted(groups[group_key]) if group_key else [pair_id],
                "baseline_revisions": revisions,
                "intervention": row["intervention"],
                "preservation": row["preservation"],
                "source_cot": row["source"]["cot"],
                "target_cot": row["target"]["cot"],
                "source_cot_text": row["source"]["cot_text"],
                "target_cot_text": row["target"]["cot_text"],
                "source_prompt": row["source"]["prompt"],
                "target_prompt": row["target"]["prompt"],
                "atomic_change": row["repair_round2"]["atomic_change"],
                "generation_status": "pending_generation",
                "fusion_status": "not_ready",
            }
        )

    if errors:
        raise RuntimeError("strict atomic validation failed:\n" + "\n".join(errors[:100]))

    blocked_rows: list[dict[str, Any]] = []
    # Keep an explicit audit record for any old shared sibling that was not
    # pulled into a split variant. It remains reasonable/untouched rather than
    # being silently regenerated from an incompatible source.
    split_siblings = set()
    for split_pair in SPLIT_VARIANT_PAIR_IDS:
        for plan in plan_by_id.values():
            if split_pair in plan.get("source_variant_member_pair_ids", []) and plan.get("pair_id") != split_pair:
                split_siblings.add(str(plan["pair_id"]))
    for pair_id in sorted(split_siblings - ready_ids):
        if pair_id in original_by_id:
            blocked_rows.append(
                {
                    "schema_version": "strict_atomic_cot_revision_v1",
                    "pair_id": pair_id,
                    "status": "default_reasonable_retained_after_source_variant_split",
                    "reason": "用户指定的不同 old count/target object 需要私有 revised source；该 sibling 不得复用新 source。",
                    "original_source_scene_id": original_by_id[pair_id]["source_scene_id"],
                    "fusion_status": "not_ready",
                }
            )
    write_jsonl(REVISED_MANIFEST, revised_rows)
    write_jsonl(REVISED_REGISTRY, registry_rows)
    write_jsonl(COT_PLAN, plan_output)
    write_jsonl(BLOCKED, blocked_rows)

    original_hash = sha_json(original_rows)
    summary = {
        "schema_version": "strict_atomic_cot_revision_summary_v1",
        "audit_round": 2,
        "original_manifest": str(ORIGINAL_MANIFEST),
        "original_manifest_sha256": hashlib.sha256(ORIGINAL_MANIFEST.read_bytes()).hexdigest(),
        "output_dir": str(OUTPUT_DIR),
        "direct_problem_count": len(SPECS),
        "revised_manifest_record_count": len(revised_rows),
        "revised_source_variant_count": len(registry_rows),
        "revised_source_scene_count": len({row["revised_source_scene_id"] for row in plan_output}),
        "source_regenerated_record_count": sum(row["source_regenerated"] for row in plan_output),
        "target_regenerated_record_count": len(revised_rows),
        "strict_atomic_validation_errors": len(errors),
        "split_variant_pair_ids": sorted(SPLIT_VARIANT_PAIR_IDS),
        "retained_split_sibling_count": len(blocked_rows),
        "user_semantic_revisions": {
            "beanie": "scarf_1 renamed to beanie_1 and category revised to knitted beanie; count remains the only source->target atom.",
            "bag_quota": "intervention revised from can_2 to bag_1 with source count 3 and target count 2; can group is preserved.",
            "rattle_3_to_2": "intervention revised from cube_3 to rattle_1 with source count 3 and target count 2; cube group is preserved.",
            "toolbox": "oilcan_1 renamed to toolbox_1 and category revised to compact hinged metal toolbox; functional_state remains the only source->target atom.",
            "single_instance_containment": "plants, flower, and milk counts revised to 1 in both source and target before their relation/location atom changes.",
        },
        "atomic_rule": "每条 revised source/target COT 的 canonical render 必须只有一个不同 atom；intervention.entity_id + field 必须等于该 atom；preservation.allowed_to_change 只能包含同一个字段。",
        "files": {
            "revised_manifest": str(REVISED_MANIFEST),
            "revised_source_registry": str(REVISED_REGISTRY),
            "cot_revision_plan": str(COT_PLAN),
            "tracking": str(TRACKING),
            "blocked_or_retained": str(BLOCKED),
        },
        "status_policy": "仅 revised_manifest 中的记录允许进入下一步独立 source/target 生图；所有记录在分页人工审计通过前均不得融合1000-pair。",
    }
    tracking_rows = []
    for row in plan_output:
        tracking_rows.append(
            {
                "schema_version": "strict_atomic_cot_tracking_v1",
                "audit_round": 2,
                "pair_id": row["pair_id"],
                "status": row["status"],
                "source_scene_id_original": row["original_source_scene_id"],
                "source_scene_id_revised": row["revised_source_scene_id"],
                "source_variant_key": row["source_variant_key"],
                "source_regenerated": row["source_regenerated"],
                "cot_revision_status": "revised_structured_cot_ready",
                "atomic_changed_atom_count": row["atomic_change"]["changed_atom_count"],
                "intervention": row["intervention"],
                "preservation_allowed_to_change": row["preservation"]["allowed_to_change"],
                "generation_status": "pending_generation",
                "replacement_status": "blocked_until_round2_audit",
                "fusion_status": "not_ready",
            }
        )
    tracking_rows.extend(
        {
            "schema_version": "strict_atomic_cot_tracking_v1",
            "audit_round": 2,
            "pair_id": row["pair_id"],
            "status": row["status"],
            "source_scene_id_original": row.get("original_source_scene_id"),
            "source_scene_id_revised": None,
            "source_variant_key": None,
            "source_regenerated": False,
            "cot_revision_status": "not_regenerated_due_to_variant_split",
            "atomic_changed_atom_count": None,
            "intervention": None,
            "preservation_allowed_to_change": None,
            "generation_status": "not_required_round2",
            "replacement_status": "blocked_until_round2_audit",
            "fusion_status": "not_ready",
        }
        for row in blocked_rows
    )
    write_jsonl(TRACKING, tracking_rows)
    summary["tracking_record_count"] = len(tracking_rows)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
