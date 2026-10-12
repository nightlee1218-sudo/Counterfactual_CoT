#!/usr/bin/env python3
"""Build a non-mutating, strict, pre-generation Gold1000 manifest.

Input files are never altered.  The output contains exactly 1,000 pairs whose
source/target CoTs are canonical, grouped by a shared source scene, and differ
in one atom only.  Prompts and preservation metadata are regenerated from the
structured CoT rather than trusted from hand-written legacy text.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "gold1000_manifest_strict_v1"
COLORS = ("red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "gray")
LOCATIONS = ("left_region", "center_region", "right_region", "top_region", "bottom_region")
MATERIAL_CATEGORIES = {
    "paper": ("paper notebook", "paper card stack", "paper folder"),
    "cardboard": ("cardboard box", "cardboard tray", "cardboard organizer"),
    "wood": ("wooden tray", "wooden box", "wooden stand"),
    "metal": ("metal tray", "metal container", "metal ornament"),
    "plastic": ("plastic container", "plastic organizer", "plastic toy"),
    "glass": ("glass jar", "glass bowl", "glass bottle"),
    "ceramic": ("ceramic bowl", "ceramic cup", "ceramic ornament"),
    "porcelain": ("porcelain bowl", "porcelain ornament", "porcelain cup"),
    "fabric": ("fabric pouch", "fabric basket", "fabric organizer"),
    "wicker": ("wicker basket", "wicker tray", "wicker organizer"),
    "leather": ("leather pouch", "leather case", "leather folder"),
    "brass": ("brass container", "brass ornament", "brass stand"),
    "steel": ("steel container", "steel tray", "steel tool"),
    "organic": ("potted plant", "cut flower", "leafy branch"),
    "mesh netting": ("mesh pouch", "mesh basket", "mesh organizer"),
    "woven straw": ("woven straw basket", "woven straw tray", "woven straw mat"),
    "gilded_wood": ("gilded wooden box", "gilded wooden stand", "gilded wooden tray"),
    "inlaid_wood": ("inlaid wooden box", "inlaid wooden tray", "inlaid wooden stand"),
    "silver_plated": ("silver plated tray", "silver plated ornament", "silver plated stand"),
    "painted_metal": ("painted metal container", "painted metal tray", "painted metal stand"),
    "plastic_and_metal_spring": ("plastic spring toy", "plastic spring ornament", "plastic spring play object"),
}


def clone(value: Any) -> Any:
    return copy.deepcopy(value)


def normalized_location(value: str) -> str:
    mapping = {
        "upper_shelf_left": "upper_left",
        "upper_shelf_right": "upper_right",
        "upper_shelf_center": "top_region",
        "lower_shelf_left": "lower_left",
        "lower_shelf_right": "lower_right",
        "lower_shelf_center": "bottom_region",
    }
    return mapping.get(value, value)


def atoms(cot: dict[str, Any]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    for entity in cot["entities"]:
        eid = entity["id"]
        result += [(f"{eid}.object", str(entity["category"])), (f"{eid}.count", str(entity["count"]))]
        result.append((f"{eid}.location", str(entity["location"])))
        for key, value in entity.get("attributes", {}).items():
            result.append((f"{eid}.attribute.{key}", str(value)))
        for index, relation in enumerate(entity.get("relations", [])):
            result.append((f"{eid}.relation.{index}", f"{relation['predicate']}:{relation['target']}"))
    for key, value in cot["background"].items():
        result.append((f"background.{key}", str(value)))
    return result


def cot_text(cot: dict[str, Any]) -> str:
    return "; ".join(f"{key}={value}" for key, value in atoms(cot))


def entity(cot: dict[str, Any], entity_id: str) -> dict[str, Any]:
    return next(item for item in cot["entities"] if item["id"] == entity_id)


def field_key(entity_id: str, field: str) -> str:
    if field == "category":
        return f"{entity_id}.object"
    if field == "count" or field == "location":
        return f"{entity_id}.{field}"
    if field.startswith("attributes."):
        return f"{entity_id}.attribute.{field.split('.', 1)[1]}"
    if field == "relations":
        return f"{entity_id}.relation.0"
    raise ValueError(field)


def set_field(cot: dict[str, Any], entity_id: str, field: str, value: Any) -> None:
    item = entity(cot, entity_id)
    if field == "category":
        item["category"] = value
    elif field in ("count", "location"):
        item[field] = value
    elif field.startswith("attributes."):
        item.setdefault("attributes", {})[field.split(".", 1)[1]] = value
    elif field == "relations":
        predicate, target = value.split(":", 1)
        item["relations"] = [{"predicate": predicate, "target": target}]
    else:
        raise ValueError(field)


def get_field(cot: dict[str, Any], entity_id: str, field: str) -> Any:
    item = entity(cot, entity_id)
    if field == "category":
        return item["category"]
    if field in ("count", "location"):
        return item[field]
    if field.startswith("attributes."):
        return item["attributes"][field.split(".", 1)[1]]
    if field == "relations":
        relation = item["relations"][0]
        return f"{relation['predicate']}:{relation['target']}"
    raise ValueError(field)


def category_for_material(material: str, previous: str, salt: str) -> str:
    candidates = MATERIAL_CATEGORIES.get(material, (f"{material} object",))
    index = int(hashlib.sha256(salt.encode()).hexdigest(), 16) % len(candidates)
    candidate = candidates[index]
    if candidate == previous:
        candidate = candidates[(index + 1) % len(candidates)]
    return candidate


def clean_cot(cot: dict[str, Any]) -> None:
    if not isinstance(cot.get("background"), dict):
        cot["background"] = {"scene": str(cot.get("background", "simple indoor scene"))}
    for item in cot["entities"]:
        item["location"] = normalized_location(item["location"])
        item.setdefault("attributes", {}).pop("occlusion_state", None)
        # Counts above five are deliberately normalized before image generation.
        if item["count"] > 5:
            item["count"] = 4


def normalize_intervention(record: dict[str, Any]) -> None:
    source, target = record["source"]["cot"], record["target"]["cot"]
    intervention = record["intervention"]
    eid, field = intervention["entity_id"], intervention["field"]
    if field == "attributes.occlusion_state":
        item = entity(source, eid)
        old_color = item.get("attributes", {}).get("color", "green")
        new_color = next(color for color in COLORS if color != old_color)
        field = "attributes.color"
        intervention["field"] = field
        intervention["old_value"] = old_color
        intervention["new_value"] = new_color
        set_field(target, eid, field, new_color)
    elif field == "relations":
        source_relations = entity(source, eid).get("relations", [])
        target_relations = entity(target, eid).get("relations", [])
        if source_relations:
            old = get_field(source, eid, field)
        else:
            target_id = (target_relations[0]["target"] if target_relations
                         else next(x["id"] for x in source["entities"] if x["id"] != eid))
            old = f"separate_from:{target_id}"
            set_field(source, eid, field, old)
        desired = get_field(target, eid, field) if target_relations else None
        if desired is None or desired == old:
            target_id = old.split(":", 1)[1]
            desired = f"separate_from:{target_id}" if old.startswith("inside:") else f"inside:{target_id}"
            set_field(target, eid, field, desired)
        intervention["old_value"], intervention["new_value"] = old, desired
    elif field == "count":
        old = int(get_field(source, eid, field))
        new = int(get_field(target, eid, field))
        if old < 1:
            old = 1
            set_field(source, eid, field, old)
        if new < 1 or new > 5 or new == old:
            new = old + 1 if old < 5 else old - 1
            set_field(target, eid, field, new)
        intervention["old_value"], intervention["new_value"] = old, new
    elif field == "location":
        old = get_field(source, eid, field)
        new = get_field(target, eid, field)
        if old == new:
            new = next(value for value in LOCATIONS if value != old)
            set_field(target, eid, field, new)
        intervention["old_value"], intervention["new_value"] = old, new
    elif field == "category":
        old = get_field(source, eid, field)
        source_entity = entity(source, eid)
        material = source_entity.get("attributes", {}).get("material", "plastic")
        if "functional_state" in source_entity.get("attributes", {}):
            # A functional-state attribute such as on/off can only be preserved on
            # a category that can plausibly have that state.
            candidates = (f"{material} floor lamp", f"{material} table lamp", f"{material} lantern")
            new = candidates[int(hashlib.sha256(record["pair_id"].encode()).hexdigest(), 16) % len(candidates)]
            if new == old:
                new = candidates[(candidates.index(new) + 1) % len(candidates)]
        else:
            new = category_for_material(material, old, record["pair_id"])
        set_field(target, eid, field, new)
        intervention["old_value"], intervention["new_value"] = old, new
    else:
        intervention["old_value"] = get_field(source, eid, field)
        intervention["new_value"] = get_field(target, eid, field)


def type_fields(field: str, source_value: Any, target_value: Any) -> tuple[str, str]:
    if field == "category":
        return "category_replace", "material_compatible_category_replace"
    if field == "count":
        return "count_replace", f"{source_value}_to_{target_value}"
    if field == "location":
        return "location_replace", "absolute_position"
    if field == "relations":
        return "relation_replace", f"{source_value.split(':', 1)[0]}_to_{target_value.split(':', 1)[0]}"
    return "attribute_replace", field.split(".", 1)[1]


def rebuild_preservation(record: dict[str, Any]) -> None:
    intervention = record["intervention"]
    changed = field_key(intervention["entity_id"], intervention["field"])
    allowed = intervention["entity_id"] + "." + intervention["field"]
    preserve = []
    for key, _ in atoms(record["source"]["cot"]):
        if key == changed:
            continue
        if key.endswith(".object"):
            preserve.append(key.replace(".object", ".category"))
        elif ".attribute." in key:
            preserve.append(key.replace(".attribute.", ".attributes."))
        elif ".relation." in key:
            preserve.append(key.rsplit(".", 1)[0].replace(".relation", ".relations"))
        else:
            preserve.append(key)
    record["preservation"] = {
        "allowed_to_change": [allowed],
        "must_preserve": preserve,
        "must_revalidate": [allowed],
    }


def source_prompt(cot: dict[str, Any]) -> str:
    descriptions = []
    for item in cot["entities"]:
        attributes = item.get("attributes", {})
        prefix = " ".join(str(attributes[key]) for key in ("color", "material") if attributes.get(key))
        noun = f"{prefix} {item['category']}".strip()
        count = item["count"]
        quantity = "exactly one" if count == 1 else f"exactly {count}"
        descriptions.append(f"{quantity} {noun} in the {item['location']}")
    background = cot["background"].get("scene", "simple indoor scene")
    return (
        f"Create one natural 1024x1024 image of a simple {background}. "
        f"Include {', '.join(descriptions)}. All objects are fully visible, separated, "
        "unobstructed, countable, and entirely inside the frame. Keep the composition uncluttered. "
        "Use plain unlabeled surfaces: no readable text, logos, letters, numbers, signage, watermarks, or UI."
    )


def target_prompt(record: dict[str, Any]) -> str:
    iv = record["intervention"]
    source_value, target_value = iv["old_value"], iv["new_value"]
    return (
        "Edit the provided source image locally. Apply exactly one atomic change: "
        f"change {iv['entity_id']}.{iv['field']} from {source_value} to {target_value}. "
        "Preserve every other object identity, count, location, attribute, relation, background condition, "
        "camera viewpoint, lighting, framing, and composition. Do not add, remove, duplicate, crop, recolor, "
        "move, resize, or alter anything except the specified atomic change. "
        "Keep all surfaces plain and unlabeled: no readable text, logos, letters, numbers, signage, watermarks, or UI. "
        f"Target fixed scene plan: {record['target']['cot_text']}"
    )


def finalize(record: dict[str, Any]) -> None:
    source, target = record["source"]["cot"], record["target"]["cot"]
    iv = record["intervention"]
    old_value = get_field(source, iv["entity_id"], iv["field"])
    new_value = get_field(target, iv["entity_id"], iv["field"])
    iv["old_value"], iv["new_value"] = old_value, new_value
    dim, subtype = type_fields(iv["field"], str(old_value), str(new_value))
    record["dimension"] = dim
    record["subtype"] = subtype
    iv["intervention_type"] = dim
    iv["intervention_subtype"] = subtype
    iv["operation"] = "replace"
    record["source"]["cot_text"] = cot_text(source)
    record["target"]["cot_text"] = cot_text(target)
    record["source"]["prompt"] = source_prompt(source)
    rebuild_preservation(record)
    record["target"]["prompt"] = target_prompt(record)


def transition_key(record: dict[str, Any]) -> tuple[str, str]:
    return record["source"]["cot_text"], record["target"]["cot_text"]


def normalize_scene(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Current input already groups by one source; choose its first structured source as canonical.
    base = clone(group[0]["source"]["cot"])
    clean_cot(base)
    result = []
    for original in group:
        record = clone(original)
        record["source"]["cot"] = clone(base)
        original_target = clone(original["target"]["cot"])
        clean_cot(original_target)
        iv = record["intervention"]
        if str(iv["field"]).startswith("relations"):
            iv["field"] = "relations"
        # Preserve the original desired value, but start the target from the canonical
        # source so every non-intervened atom is identical by construction.
        record["target"]["cot"] = clone(base)
        if iv["field"] != "attributes.occlusion_state":
            try:
                desired = get_field(original_target, iv["entity_id"], iv["field"])
                set_field(record["target"]["cot"], iv["entity_id"], iv["field"], desired)
            except (IndexError, KeyError):
                # Malformed relation add/drop pairs are reconstructed below as a
                # replacement of a relation atom rather than preserving an absent atom.
                pass
        normalize_intervention(record)
        # A legacy relation-add pair lacks a source relation atom. Reconstructing
        # it requires a distinct source image; keep its grouped-source invariant
        # by assigning it an explicit one-pair source scene rather than mixing it
        # into the original scene's shared source image.
        if iv["field"] == "relations" and cot_text(record["source"]["cot"]) != cot_text(base):
            record["source_scene_id"] = f"{record['source_scene_id']}__relation_source__{record['pair_id']}"
        finalize(record)
        result.append(record)
    return result


def replacement_pair(base_record: dict[str, Any], serial: int, existing: set[tuple[str, str]]) -> dict[str, Any] | None:
    source = clone(base_record["source"]["cot"])
    for item in source["entities"]:
        if "color" not in item.get("attributes", {}):
            continue
        old = item["attributes"]["color"]
        for new in COLORS:
            if new == old:
                continue
            target = clone(source)
            target_item = entity(target, item["id"])
            target_item["attributes"]["color"] = new
            candidate = {
                "pair_id": f"g1000_strict_replacement_{serial:03d}",
                "source_scene_id": base_record["source_scene_id"],
                "template_family": base_record["template_family"],
                "source": {"cot": source},
                "target": {"cot": target},
                "intervention": {
                    "entity_id": item["id"], "field": "attributes.color", "operation": "replace",
                    "old_value": old, "new_value": new,
                },
            }
            finalize(candidate)
            if transition_key(candidate) not in existing:
                return candidate
    return None


def load() -> list[dict[str, Any]]:
    records = []
    for path in sorted(ROOT.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    source_records = load()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in source_records:
        grouped[record["source_scene_id"]].append(record)
    normalized = []
    for scene_id in sorted(grouped):
        normalized.extend(normalize_scene(grouped[scene_id]))

    # Retain only one copy of each semantically identical source -> target transition.
    retained, existing = [], set()
    for record in normalized:
        key = transition_key(record)
        if key not in existing:
            retained.append(record)
            existing.add(key)

    deduplicated_count = len(normalized) - len(retained)

    # Replenish deterministic, material-compatible color interventions from existing source scenes.
    bases = sorted(retained, key=lambda item: (item["source_scene_id"], item["pair_id"]))
    serial = 1
    cursor = 0
    while len(retained) < 1000:
        candidate = replacement_pair(bases[cursor % len(bases)], serial, existing)
        cursor += 1
        if candidate is None:
            if cursor > len(bases) * 20:
                raise RuntimeError("Could not construct enough unique strict replacements")
            continue
        retained.append(candidate)
        existing.add(transition_key(candidate))
        serial += 1

    if len(retained) != 1000:
        raise AssertionError(len(retained))
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "gold1000_strict_v1.jsonl"
    output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in retained), encoding="utf-8")
    provenance = {
        "schema_version": "gold1000_strict_v1",
        "source_manifest": str(ROOT),
        "records": len(retained),
        "generation_ready": False,
        "note": "Set generation_ready only after validate_strict_gold1000.py exits cleanly.",
    }
    (OUT / "manifest_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(f"records={len(retained)} original={len(source_records)} deduplicated={deduplicated_count} replacements={serial-1}")


if __name__ == "__main__":
    main()
