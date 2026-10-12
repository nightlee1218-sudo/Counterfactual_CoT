#!/usr/bin/env python3
"""Fail-fast semantic audit for `gold1000_strict_v1.jsonl`. Read-only."""
from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "gold1000_strict_v1.jsonl"
LOCATIONS = {"left_region", "center_region", "right_region", "top_region", "bottom_region", "upper_left", "upper_right", "lower_left", "lower_right"}
MATERIAL_TOKENS = {
    "paper": ("paper",), "cardboard": ("cardboard",), "wood": ("wood",), "metal": ("metal",),
    "plastic": ("plastic",), "glass": ("glass",), "ceramic": ("ceramic",), "porcelain": ("porcelain",),
    "fabric": ("fabric",), "wicker": ("wicker",), "leather": ("leather",), "brass": ("brass",),
    "steel": ("steel",), "organic": ("plant", "flower", "leaf"), "mesh netting": ("mesh",),
    "woven straw": ("woven", "straw"), "gilded_wood": ("gilded", "wood"), "inlaid_wood": ("inlaid", "wood"),
    "silver_plated": ("silver",), "painted_metal": ("painted", "metal"),
    "plastic_and_metal_spring": ("plastic", "spring"),
}


def render(cot: dict[str, Any]) -> str:
    result = []
    for entity in cot["entities"]:
        eid = entity["id"]
        result += [f"{eid}.object={entity['category']}", f"{eid}.count={entity['count']}", f"{eid}.location={entity['location']}"]
        for key, value in entity.get("attributes", {}).items():
            result.append(f"{eid}.attribute.{key}={value}")
        for index, relation in enumerate(entity.get("relations", [])):
            result.append(f"{eid}.relation.{index}={relation['predicate']}:{relation['target']}")
    for key, value in cot["background"].items():
        result.append(f"background.{key}={value}")
    return "; ".join(result)


def atom_key(entity_id: str, field: str) -> str:
    if field == "category": return f"{entity_id}.object"
    if field in {"count", "location"}: return f"{entity_id}.{field}"
    if field.startswith("attributes."): return f"{entity_id}.attribute.{field.split('.', 1)[1]}"
    if field == "relations": return f"{entity_id}.relation.0"
    raise ValueError(field)


def main() -> None:
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    def fail(message: str) -> None:
        if len(errors) < 100:
            errors.append(message)

    ids = [record["pair_id"] for record in records]
    if len(records) != 1000: fail(f"records={len(records)}, expected=1000")
    if len(ids) != len(set(ids)): fail("pair_id is not unique")

    scenes: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    transitions = set()
    for record in records:
        pid = record["pair_id"]
        scenes[record["source_scene_id"]].append(record)
        source, target, iv = record["source"], record["target"], record["intervention"]
        for side in (source, target):
            if render(side["cot"]) != side["cot_text"]:
                fail(f"{pid}: noncanonical cot_text")
            if not isinstance(side["cot"].get("background"), dict):
                fail(f"{pid}: background is not object")
            for entity in side["cot"]["entities"]:
                if entity["location"] not in LOCATIONS:
                    fail(f"{pid}: invalid location {entity['location']}")
                if not 1 <= entity["count"] <= 5:
                    fail(f"{pid}: count outside [1,5]")
                if "occlusion_state" in entity.get("attributes", {}):
                    fail(f"{pid}: occlusion_state is prohibited")
        source_atoms = [part.strip() for part in source["cot_text"].split(";")]
        target_atoms = [part.strip() for part in target["cot_text"].split(";")]
        differences = [(a, b) for a, b in zip(source_atoms, target_atoms) if a != b]
        if len(source_atoms) != len(target_atoms) or len(differences) != 1:
            fail(f"{pid}: does not have exactly one atom replacement")
            continue
        source_ids = [item["id"] for item in source["cot"]["entities"]]
        target_ids = [item["id"] for item in target["cot"]["entities"]]
        if source_ids != target_ids:
            fail(f"{pid}: entity ids differ")
        expected_key = atom_key(iv["entity_id"], iv["field"])
        actual_key = differences[0][0].split("=", 1)[0]
        old, new = differences[0][0].split("=", 1)[1], differences[0][1].split("=", 1)[1]
        if actual_key != expected_key or str(iv["old_value"]) != old or str(iv["new_value"]) != new:
            fail(f"{pid}: intervention does not exactly describe changed atom")
        if record["dimension"] != iv["intervention_type"] or record["subtype"] != iv["intervention_subtype"]:
            fail(f"{pid}: metadata enum mismatch")
        if iv["field"] == "category":
            changed = next(item for item in target["cot"]["entities"] if item["id"] == iv["entity_id"])
            material = changed.get("attributes", {}).get("material")
            tokens = MATERIAL_TOKENS.get(material, ())
            category = changed["category"].lower()
            compatible = any(token in category for token in tokens) if material == "organic" else all(token in category for token in tokens)
            if tokens and not compatible:
                fail(f"{pid}: category/material incompatible: {category} vs {material}")
            if "functional_state" in changed.get("attributes", {}) and not any(token in category for token in ("lamp", "lantern")):
                fail(f"{pid}: functional_state incompatible category: {category}")
        if f"Target fixed scene plan: {target['cot_text']}" not in target["prompt"]:
            fail(f"{pid}: target prompt lacks exact target plan")
        transitions.add((source["cot_text"], target["cot_text"]))

    for scene_id, group in scenes.items():
        if len({record["source"]["cot_text"] for record in group}) != 1:
            fail(f"{scene_id}: inconsistent source cot")
        if len({record["source"]["prompt"] for record in group}) != 1:
            fail(f"{scene_id}: inconsistent source prompt")
    if len(transitions) != len(records):
        fail(f"unique transitions={len(transitions)}, expected={len(records)}")

    report = {
        "records": len(records), "unique_pair_ids": len(set(ids)), "unique_source_scenes": len(scenes),
        "unique_transitions": len(transitions), "errors": len(errors), "examples": errors,
        "generation_ready": not errors,
    }
    (ROOT / "strict_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
