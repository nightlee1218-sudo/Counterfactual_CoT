#!/usr/bin/env python3
"""Read-only release audit for `gold1000_framework_strict_v4.jsonl`."""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "gold1000_framework_strict_v4.jsonl"
QUOTAS = {
    "same_affordance_replace": 80, "shape_divergent_replace": 85, "affordance_divergent_replace": 85,
    "color": 45, "material": 50, "pattern": 50, "occlusion_state": 55, "functional_state": 50,
    "1_to_2": 60, "2_to_1": 60, "2_to_3": 70, "3_to_2": 60,
    "absolute_horizontal": 75, "absolute_vertical": 75, "containment": 100,
}
DIMENSION_BY_SUBTYPE = {
    **{k: "category_replacement" for k in ("same_affordance_replace", "shape_divergent_replace", "affordance_divergent_replace")},
    **{k: "attribute_intervention" for k in ("color", "material", "pattern", "occlusion_state", "functional_state")},
    **{k: "instance_cardinality_intervention" for k in ("1_to_2", "2_to_1", "2_to_3", "3_to_2")},
    **{k: "spatial_geometric_intervention" for k in ("absolute_horizontal", "absolute_vertical", "containment")},
}
LOCATIONS = {"left_region", "center_region", "right_region", "top_region", "bottom_region", "upper_left", "upper_right", "lower_left", "lower_right"}
OFFICIAL = {
    "kitchen_counter", "bathroom_counter", "office_desk", "bedroom_nightstand",
    "bookshelf_study", "closet_wardrobe", "warehouse_shelf",
    "grocery_display", "flower_shop", "toolshed_garage", "antique_curio_shop",
    "city_street", "parking_lot", "playground", "beach_scene",
    "living_room_play", "classroom", "dining_table", "pantry_shelf_closeup", "picnic_blanket",
}
DOMAINS = {
    "indoor_counter": {"kitchen_counter", "bathroom_counter", "office_desk", "bedroom_nightstand"},
    "furniture_storage": {"bookshelf_study", "closet_wardrobe", "warehouse_shelf"},
    "commercial_display": {"grocery_display", "flower_shop", "toolshed_garage", "antique_curio_shop"},
    "outdoor_open": {"city_street", "parking_lot", "playground", "beach_scene"},
    "indoor_activity": {"living_room_play", "classroom", "dining_table"},
    "dense_closeup": {"pantry_shelf_closeup", "picnic_blanket"},
}
SOURCE_REUSE_CAP = 3
ENTITY_COUNT_TARGETS = {1: 200, 2: 500, 3: 300}
SHARED_PAIR_TARGET = 279
CATEGORY_FAMILIES = {
    "same_affordance_replace": {"grocery_display", "classroom", "dining_table", "flower_shop", "toolshed_garage", "warehouse_shelf"},
    "shape_divergent_replace": {"city_street", "parking_lot", "playground"},
    "affordance_divergent_replace": {"bookshelf_study", "antique_curio_shop"},
}
ATTACHMENT_BOUND_CATEGORY_TERMS = {"wall-mounted", "ceiling-mounted", "built-in"}
MATERIAL_CATEGORY_WORDS = {
    "wood", "wooden", "metal", "steel", "brass", "copper", "iron", "aluminum", "plastic", "glass",
    "ceramic", "porcelain", "paper", "cardboard", "fabric", "cotton", "linen", "cloth", "wicker", "rattan",
    "clay", "rubber", "foam", "leather", "stone", "marble", "wool", "canvas", "silk", "velvet", "denim",
    "wire", "straw",
}
CONTAINER_TERMS = {"box", "basket", "bowl", "cup", "jar", "tray", "bin", "drawer", "case", "holder", "crate", "vase", "pot", "bucket", "container", "pouch", "carton", "tin", "sack", "cage", "sleeve", "bag", "pack"}
COLLECTIVE_TERMS = {"set", "stack", "row", "pair", "bunch", "collection", "assortment", "group"}
GENERIC_CATEGORY_TARGETS = {"display object", "storage stand", "metal ornament", "ceramic ornament", "storage vessel"}
FORBIDDEN_COLLECTIVE_CATEGORIES = {
    "greeting card stack", "folded pajama set", "orange stack", "bread roll set", "canned soup set",
    "potted plant row", "milk carton set", "yogurt cup row", "butter block row", "tomato row",
    "banana bunch", "potato set", "cardboard box stack", "stacked tire set", "safety helmet set",
    "grain sack stack", "egg set", "croissant row", "glass jar row", "tulip bunch", "peony bunch",
    "sunflower bunch", "magazine stack", "vinyl record stack", "rolled sock pair set",
}
UNNATURAL_CATEGORY_COMBINATIONS = (
    frozenset({"collectible mug", "magazine folder"}),
    frozenset({"metal bookend", "webcam"}),
)
MATERIAL_ALIASES = {
    "wood": {"wood", "wooden"}, "painted wood": {"wood", "wooden"}, "gilded_wood": {"gilded", "wood", "wooden"},
    "metal": {"metal"}, "brass": {"brass"}, "steel": {"steel"}, "plastic": {"plastic"}, "glass": {"glass"},
    "ceramic": {"ceramic"}, "porcelain": {"porcelain"}, "paper": {"paper", "cardboard"},
    "fabric": {"fabric", "cotton", "linen", "cloth"}, "cotton": {"cotton", "fabric"},
    "linen": {"linen", "fabric"}, "wicker": {"wicker"}, "woven rattan": {"woven", "rattan"},
    "woven straw": {"woven", "straw"}, "metal wire": {"metal", "wire"},
    "clay": {"clay"}, "rubber": {"rubber"}, "foam": {"foam"}, "leather": {"leather"}, "plush": {"plush"},
}
EXPLICIT_MATERIAL_WORDS = MATERIAL_CATEGORY_WORDS


def category_visual_minimal_compatible(category: str, material: str) -> bool:
    words = set(category.lower().replace("-", " ").split())
    explicit = words & EXPLICIT_MATERIAL_WORDS
    return (
        not any(term in category.lower() for term in ATTACHMENT_BOUND_CATEGORY_TERMS)
        and (not explicit or bool(explicit & MATERIAL_ALIASES.get(material, {material})) )
    )


def get_entity(cot: dict[str, Any], entity_id: str) -> dict[str, Any]:
    return next(item for item in cot["entities"] if item["id"] == entity_id)


def visual_role(category: str) -> str | None:
    text = category.lower()
    if "bookend" in text:
        return "bookend"
    for token, role in (
        ("lamp", "lamp"), ("vase", "vase"), ("clock", "clock"),
        ("notebook", "book"), ("storybook", "book"), ("textbook", "book"),
        ("paperback", "book"), ("hardcover", "book"), ("closed book", "book"),
    ):
        if token in text:
            return role
    return None


def valid_functional_states(category: str) -> set[str]:
    text = category.lower()
    if any(word in text for word in ("lamp", "clock", "drill", "powered digital")):
        return {"off", "on"}
    if "soap" in text or "towel" in text:
        return {"dry", "wet"}
    if "stapler" in text:
        return {"unloaded", "loaded"}
    if any(word in text for word in ("book", "notebook", "box", "toolbox", "drawer", "crate", "can", "case")):
        return {"closed", "open"}
    if any(word in text for word in ("wrench", "hammer", "screwdriver", "saw")):
        return {"clean", "rusty"}
    return {"empty", "full"}


def render(cot: dict[str, Any]) -> str:
    parts: list[str] = []
    for entity in cot["entities"]:
        eid = entity["id"]
        parts += [f"{eid}.object={entity['category']}", f"{eid}.count={entity['count']}", f"{eid}.location={entity['location']}"]
        parts += [f"{eid}.attribute.{key}={value}" for key, value in entity.get("attributes", {}).items()]
        parts += [f"{eid}.relation.{index}={relation['predicate']}:{relation['target']}" for index, relation in enumerate(entity.get("relations", []))]
    return "; ".join(parts + [f"background.{key}={value}" for key, value in cot["background"].items()])


def atom_key(entity_id: str, field: str) -> str:
    if field == "category": return f"{entity_id}.object"
    if field in {"count", "location"}: return f"{entity_id}.{field}"
    if field.startswith("attributes."): return f"{entity_id}.attribute.{field.split('.', 1)[1]}"
    if field == "relations": return f"{entity_id}.relation.0"
    raise ValueError(field)


def preservation_key(key: str) -> str:
    return key.replace(".category", ".object").replace(".attributes.", ".attribute.").replace(".relations", ".relation.0")


def run() -> dict[str, Any]:
    records = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    failures: list[str] = []
    def fail(message: str) -> None:
        if len(failures) < 100: failures.append(message)

    ids = [r["pair_id"] for r in records]
    if len(records) != 1000: fail(f"record_count={len(records)}")
    if len(set(ids)) != len(ids): fail("duplicate_pair_id")
    subtypes = collections.Counter(r["subtype"] for r in records)
    if subtypes != collections.Counter(QUOTAS): fail(f"quota={dict(subtypes)}")
    families = {r["template_family"] for r in records}
    if families != OFFICIAL: fail(f"official_families={sorted(families)}")
    for name, members in DOMAINS.items():
        count = sum(r["template_family"] in members for r in records)
        if count > 250: fail(f"domain_over_25pct:{name}={count}")

    transitions, scenes = set(), collections.defaultdict(list)
    forbidden_prompt = ("readable text", "label", "logo", "watermark", "sign", "ui")
    for record in records:
        pid, source, target, iv = record["pair_id"], record["source"], record["target"], record["intervention"]
        scenes[record["source_scene_id"]].append(record)
        if record["dimension"] != DIMENSION_BY_SUBTYPE[record["subtype"]] or iv.get("intervention_type") != record["dimension"] or iv.get("intervention_subtype") != record["subtype"]:
            fail(f"{pid}: taxonomy")
        expected_field_by_attribute_subtype = {
            "color": "attributes.color", "material": "attributes.material", "pattern": "attributes.pattern",
            "occlusion_state": "attributes.occlusion_state", "functional_state": "attributes.functional_state",
        }
        if record["subtype"] in expected_field_by_attribute_subtype and iv["field"] != expected_field_by_attribute_subtype[record["subtype"]]:
            fail(f"{pid}: subtype_field_mismatch")
        for side in (source, target):
            if render(side["cot"]) != side["cot_text"]: fail(f"{pid}: noncanonical_{'source' if side is source else 'target'}_cot")
            if not isinstance(side["cot"].get("background"), dict): fail(f"{pid}: background_schema")
            entity_ids = {entity["id"] for entity in side["cot"]["entities"]}
            for entity in side["cot"]["entities"]:
                if entity["location"] not in LOCATIONS or "shelf" in entity["location"]: fail(f"{pid}: location={entity['location']}")
                if not 1 <= entity["count"] <= 5: fail(f"{pid}: count={entity['count']}")
                for relation in entity.get("relations", []):
                    if relation["target"] not in entity_ids or relation["target"] == entity["id"]:
                        fail(f"{pid}: dangling_or_self_relation")
                if entity["category"] in FORBIDDEN_COLLECTIVE_CATEGORIES:
                    fail(f"{pid}: collective_category={entity['category']}")
                # A static occlusion_state is valid on non-occlusion pairs: it is
                # part of the shared source/target scene and preservation contract.
            categories = {entity["category"] for entity in side["cot"]["entities"]}
            roles = [role for category in categories if (role := visual_role(category)) is not None]
            if len(roles) != len(set(roles)):
                fail(f"{pid}: duplicate_visual_role")
            if any(forbidden <= categories for forbidden in UNNATURAL_CATEGORY_COMBINATIONS):
                fail(f"{pid}: unnatural_category_combination")
            if "open wicker bottle holder" in categories and "wine bottle" not in categories:
                fail(f"{pid}: bottle_holder_without_bottle")
            if any("bookend" in category for category in categories) and not any("book" in category and "bookend" not in category for category in categories):
                fail(f"{pid}: bookend_without_book")
            if "webcam" in categories and "computer monitor" not in categories:
                fail(f"{pid}: webcam_without_monitor")
            if any(word in side["prompt"].lower() for word in forbidden_prompt) and "no readable text" not in side["prompt"].lower(): fail(f"{pid}: prompt_text_risk")
        sa = [x.strip() for x in source["cot_text"].split(";") if x.strip()]
        ta = [x.strip() for x in target["cot_text"].split(";") if x.strip()]
        diffs = [] if len(sa) != len(ta) else [(a, b) for a, b in zip(sa, ta) if a != b]
        if len(diffs) != 1:
            fail(f"{pid}: not_one_atom")
            continue
        old_atom, new_atom = diffs[0]
        expected = atom_key(iv["entity_id"], iv["field"])
        if old_atom.split("=", 1)[0] != expected: fail(f"{pid}: changed_key")
        if str(iv["old_value"]) != old_atom.split("=", 1)[1] or str(iv["new_value"]) != new_atom.split("=", 1)[1]: fail(f"{pid}: intervention_values")
        source_entity = get_entity(source["cot"], iv["entity_id"])
        target_entity = get_entity(target["cot"], iv["entity_id"])
        if iv["field"] == "category":
            material = str(target_entity.get("attributes", {}).get("material", ""))
            if not category_visual_minimal_compatible(str(iv["new_value"]), material):
                fail(f"{pid}: category_visual_incompatible")
            if record["template_family"] not in CATEGORY_FAMILIES[record["subtype"]]:
                fail(f"{pid}: category_family_outside_plan")
            if str(iv["new_value"]).lower() in GENERIC_CATEGORY_TARGETS:
                fail(f"{pid}: generic_category_target")
            if set(re.findall(r"[a-z]+", str(iv["new_value"]).lower())) & COLLECTIVE_TERMS:
                fail(f"{pid}: category_implies_collective_count")
        if iv["field"] == "category" and "functional_state" in target_entity.get("attributes", {}):
            state = str(target_entity["attributes"]["functional_state"])
            if state not in valid_functional_states(target_entity["category"]):
                fail(f"{pid}: target_category_functional_state_incompatible")
        if record["subtype"] == "material":
            words = set(re.findall(r"[a-z]+", source_entity["category"].lower()))
            if words & MATERIAL_CATEGORY_WORDS:
                fail(f"{pid}: material_word_baked_into_category")
            target_material = str(iv["new_value"]).lower()
            color = str(source_entity.get("attributes", {}).get("color", "")).lower()
            if color in {"clear", "transparent", "translucent"} and target_material not in {"glass", "clear plastic", "transparent plastic"}:
                fail(f"{pid}: transparent_color_material_conflict")
        if record["subtype"] == "functional_state":
            if {str(iv["old_value"]), str(iv["new_value"])} != valid_functional_states(source_entity["category"]):
                fail(f"{pid}: invalid_functional_state")
        if iv["field"] == "count":
            words = set(re.findall(r"[a-z]+", source_entity["category"].lower()))
            if words & COLLECTIVE_TERMS:
                fail(f"{pid}: ambiguous_collective_count")
        if record["subtype"] == "containment":
            source_relation = source_entity.get("relations", [])
            target_relation = target_entity.get("relations", [])
            if len(source_relation) != 1 or len(target_relation) != 1:
                fail(f"{pid}: containment_relation_count")
            else:
                source_container = get_entity(source["cot"], source_relation[0]["target"])
                target_container = get_entity(target["cot"], target_relation[0]["target"])
                if source_relation[0]["target"] != target_relation[0]["target"] or {source_relation[0]["predicate"], target_relation[0]["predicate"]} != {"inside", "separate_from"}:
                    fail(f"{pid}: containment_relation_mapping")
                if source_entity["location"] != source_container["location"] or target_entity["location"] != target_container["location"]:
                    fail(f"{pid}: containment_location_mismatch")
                if not any(term in source_container["category"].lower() for term in CONTAINER_TERMS):
                    fail(f"{pid}: containment_non_container")
                inverse_conflicts = [
                    relation for relation in target_container.get("relations", [])
                    if relation["target"] == iv["entity_id"] and relation["predicate"] in {"inside", "separate_from"}
                ]
                if inverse_conflicts:
                    fail(f"{pid}: containment_inverse_relation_conflict")
            new_predicate = str(iv["new_value"]).split(":", 1)[0]
            required = "rim stays visible" if new_predicate == "inside" else "clear empty gap"
            if required not in target["prompt"]:
                fail(f"{pid}: containment_prompt_geometry")
        if record["subtype"] == "occlusion_state":
            source_refs = [relation for relation in source_entity.get("relations", []) if relation["predicate"] == "occlusion_reference"]
            target_refs = [relation for relation in target_entity.get("relations", []) if relation["predicate"] == "occlusion_reference"]
            states = {str(iv["old_value"]), str(iv["new_value"])}
            if len(source_refs) != 1 or source_refs != target_refs or "unoccluded" not in states or not any(value.startswith("partially_occluded_by_") for value in states):
                fail(f"{pid}: invalid_occlusion_geometry")
            elif source_refs[0]["target"] not in {item["id"] for item in source["cot"]["entities"]}:
                fail(f"{pid}: dangling_occlusion_reference")
            else:
                reference = get_entity(source["cot"], source_refs[0]["target"])
                expected_suffix = re.sub(r"[^a-z0-9]+", "_", reference["category"].lower()).strip("_")
                hidden_states = [value for value in states if value.startswith("partially_occluded_by_")]
                if source_entity["location"] != reference["location"]:
                    fail(f"{pid}: occlusion_location_mismatch")
                if hidden_states != [f"partially_occluded_by_{expected_suffix}"]:
                    fail(f"{pid}: occlusion_reference_state_mismatch")
            if not (
                "both objects remain visible" in target["prompt"] or "Both objects remain visible" in target["prompt"]
            ):
                fail(f"{pid}: occlusion_prompt_geometry")
            if "occlusion reference" in source["prompt"].lower():
                fail(f"{pid}: occlusion_prompt_schema_leak")
        if record["subtype"] in {"color", "material", "pattern", "occlusion_state", "functional_state"}:
            old_text = str(iv["old_value"]).replace("_", " ").lower()
            if old_text not in source["prompt"].lower():
                fail(f"{pid}: source_prompt_missing_attribute_state")
        allowed = {preservation_key(x) for x in record["preservation"]["allowed_to_change"]}
        kept = {preservation_key(x) for x in record["preservation"]["must_preserve"]}
        if expected not in allowed or expected in kept or allowed & kept: fail(f"{pid}: preservation")
        if re.search(r"\b[a-z]+_\d+\b", source["prompt"]) or "_region" in source["prompt"]:
            fail(f"{pid}: source_prompt_internal_token")
        if re.search(r"\b[a-z]+_\d+\b", target["prompt"]) or "_region" in target["prompt"] or "Target fixed scene plan" in target["prompt"]:
            fail(f"{pid}: target_prompt_internal_token")
        if re.search(r"made of\s+at\b", source["prompt"]) or re.search(r"\b(\w+)\s+\1\b", source["prompt"].lower()):
            fail(f"{pid}: source_prompt_malformed")
        if re.search(r"\b(\w+)\s+\1\b", target["prompt"].lower()):
            fail(f"{pid}: target_prompt_malformed")
        for side in ("source", "target"):
            names = [str(entity["category"]) for entity in record[side]["cot"]["entities"]]
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    left = set(re.findall(r"[a-z]+", names[i].lower()))
                    right = set(re.findall(r"[a-z]+", names[j].lower()))
                    left_head = [t for t in re.findall(r"[a-z]+", names[i].lower()) if t not in {"a", "of", "with", "and"}]
                    right_head = [t for t in re.findall(r"[a-z]+", names[j].lower()) if t not in {"a", "of", "with", "and"}]
                    nested = left and right and (left <= right or right <= left)
                    same_head = bool(left_head) and left_head[-1] == right_head[-1]
                    if nested or same_head:
                        tag = "nested_category_pair" if nested else "duplicate_head_noun"
                        fail(f"{pid}: {tag}:{names[i]}|{names[j]}")
        if record["subtype"] in {"absolute_horizontal", "absolute_vertical"}:
            if "support surface" not in source["prompt"] and "shelf levels" not in source["prompt"]:
                fail(f"{pid}: spatial_source_missing_physical_anchor")
            prompt_lower = target["prompt"].lower()
            has_displacement = (
                "third of the frame" in prompt_lower
                or "half of the frame" in prompt_lower
                or "quadrant of the frame" in prompt_lower
            ) and "do not return the source image unchanged" in prompt_lower
            if not has_displacement and ("previous" not in prompt_lower or "rest" not in prompt_lower):
                fail(f"{pid}: spatial_target_missing_observable_displacement")
        if record["subtype"] == "occlusion_state":
            source_hidden = str(iv["old_value"]).startswith("partially_occluded_by_")
            if source_hidden and ("both objects are visible" not in source["prompt"] or "one third" not in source["prompt"]):
                fail(f"{pid}: occlusion_source_missing_visible_overlap")
            if not source_hidden and "clear gap and no overlap" not in source["prompt"]:
                fail(f"{pid}: occlusion_source_missing_unoccluded_gap")
            if "both objects remain visible" not in target["prompt"] and "Both objects remain visible" not in target["prompt"]:
                fail(f"{pid}: occlusion_target_missing_visible_overlap")
        if record["subtype"] == "material":
            if "surface appearance" not in source["prompt"] or "surface material" not in target["prompt"]:
                fail(f"{pid}: material_missing_visual_cue")
        if str(iv["new_value"]) == "cardboard display stand":
            fail(f"{pid}: ambiguous_category_target")
        if str(iv["new_value"]) == "single open-front cardboard letter tray" and "one clearly visible compartment" not in target["prompt"]:
            fail(f"{pid}: letter_tray_missing_shape_definition")
        scene_text = str(source["cot"]["background"].get("scene", ""))
        if scene_text.count(", under ") > 1 or "plain" in scene_text.lower() and "backdrop" in scene_text.lower():
            fail(f"{pid}: synthetic_background_conflict")
        if record["template_family"] in {"city_street", "parking_lot", "playground", "beach_scene"} and "indoor light" in scene_text.lower():
            fail(f"{pid}: outdoor_scene_indoor_light")
        entities = source["cot"]["entities"]
        if sum(int(item["count"]) for item in entities) > 8:
            fail(f"{pid}: scene_instance_overload")
        location_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for item in entities:
            location_groups[str(item["location"])].append(item)
        if any(len(group) >= 3 or sum(int(item["count"]) for item in group) >= 7 for group in location_groups.values()):
            fail(f"{pid}: scene_region_overload")
        target_entities = target["cot"]["entities"]
        if sum(int(item["count"]) for item in target_entities) > 8:
            fail(f"{pid}: target_scene_instance_overload")
        target_location_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for item in target_entities:
            target_location_groups[str(item["location"])].append(item)
        if any(len(group) >= 3 or sum(int(item["count"]) for item in group) >= 7 for group in target_location_groups.values()):
            fail(f"{pid}: target_scene_region_overload")
        transitions.add((source["cot_text"], target["cot_text"]))
    if len(transitions) != 1000: fail(f"unique_transitions={len(transitions)}")
    entity_histogram = collections.Counter(len(record["source"]["cot"]["entities"]) for record in records)
    if entity_histogram != collections.Counter(ENTITY_COUNT_TARGETS):
        fail(f"entity_count_distribution={dict(entity_histogram)}")
    source_signatures: dict[tuple[str, str], str] = {}
    for scene_id, group in scenes.items():
        if len(group) > SOURCE_REUSE_CAP:
            fail(f"{scene_id}: source_reuse={len(group)}")
        if len({x["source"]["cot_text"] for x in group}) != 1 or len({x["source"]["prompt"] for x in group}) != 1:
            fail(f"{scene_id}: source_inconsistent")
        signature = (group[0]["source"]["cot_text"], group[0]["source"]["prompt"])
        previous = source_signatures.get(signature)
        if previous is not None and previous != scene_id:
            fail(f"duplicate_source_plan_across_ids:{previous}:{scene_id}")
        source_signatures[signature] = scene_id
    reuse_histogram = collections.Counter(len(group) for group in scenes.values())
    shared_pairs = sum(size * count for size, count in reuse_histogram.items() if size > 1)
    if shared_pairs != SHARED_PAIR_TARGET:
        fail(f"shared_pairs={shared_pairs}")
    return {"records": len(records), "unique_pair_ids": len(set(ids)), "unique_transitions": len(transitions), "source_scenes": len(scenes), "subtype_counts": dict(subtypes), "domain_counts": {name: sum(r["template_family"] in members for r in records) for name, members in DOMAINS.items()}, "entity_count_distribution": dict(entity_histogram), "source_reuse_distribution": dict(reuse_histogram), "shared_pairs": shared_pairs, "errors": len(failures), "examples": failures, "generation_ready": not failures}

if __name__ == "__main__":
    report = run()
    (ROOT / "framework_v4_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]: raise SystemExit(1)
