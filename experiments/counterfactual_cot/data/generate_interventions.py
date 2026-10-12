from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .render_prompts import scene_to_prompt
from .schema import Intervention, SceneGraph, apply_intervention, load_jsonl, render_cot_text, write_jsonl

OBJECT_REPLACEMENTS = {
    "banana": ["apple", "orange", "pear"],
    "apple": ["banana", "orange", "lemon"],
    "cup": ["bowl", "bottle", "mug"],
    "ball": ["box", "cube", "toy car"],
    "car": ["van", "taxi", "pickup truck"],
    "dog": ["cat", "rabbit", "fox"],
    "chair": ["stool", "armchair", "bench"],
    "flower": ["sunflower", "tulip", "rose"],
    "book": ["notebook", "magazine", "newspaper"],
    "bird": ["duck", "seagull", "parrot"],
    "bicycle": ["scooter", "motorcycle", "tricycle"],
}

ATTRIBUTE_REPLACEMENTS = {
    "color": {
        "red": ["blue", "green", "yellow"],
        "blue": ["red", "green", "yellow"],
        "yellow": ["red", "green", "blue"],
        "green": ["red", "blue", "yellow"],
    }
}

LOCATIONS = ["upper-left", "upper-right", "center", "lower-left", "lower-right"]


def _scene_atom_paths(scene: SceneGraph) -> List[str]:
    paths: List[str] = []
    for entity in scene.entities:
        paths.extend([f"{entity.id}.category", f"{entity.id}.count", f"{entity.id}.location"])
        paths.extend(f"{entity.id}.attributes.{key}" for key in sorted(entity.attributes))
        if entity.relations:
            paths.append(f"{entity.id}.relations")
    paths.extend(f"background.{key}" for key in sorted(scene.background))
    return paths


def _preservation_for(scene: SceneGraph, intervention: Intervention) -> Dict[str, List[str]]:
    allowed = f"{intervention.entity_id}.{intervention.field}"
    must_preserve = [path for path in _scene_atom_paths(scene) if path != allowed]
    must_revalidate = [f"{intervention.entity_id}.attributes"] if intervention.field == "category" else []
    return {
        "must_preserve": must_preserve,
        "allowed_to_change": [allowed],
        "must_revalidate": must_revalidate,
    }


def generate_for_scene(record: Dict[str, Any], max_per_scene: int, allow_unreviewed_source: bool = False) -> Iterable[Dict[str, Any]]:
    source_scene = SceneGraph.from_dict(record["cot"])
    source_prompt = str(record.get("prompt", "")).strip()
    scene_id = str(record.get("scene_id") or record.get("sample_id") or record.get("id") or "").strip()
    if not scene_id:
        raise ValueError("source record is missing scene_id/sample_id/id")
    if not str(record.get("image", "")).strip():
        raise ValueError(f"source scene {scene_id} has no generated image")
    source_quality = record.get("source_quality", {})
    if not allow_unreviewed_source and not source_quality.get("accepted", False):
        return
    candidates: List[Intervention] = []

    for entity in source_scene.entities:
        for new_category in OBJECT_REPLACEMENTS.get(entity.category, []):
            candidates.append(Intervention("replace", "object_replace", entity.id, "category", entity.category, new_category))
        for attr_key, attr_value in sorted(entity.attributes.items()):
            for new_value in ATTRIBUTE_REPLACEMENTS.get(attr_key, {}).get(str(attr_value), []):
                candidates.append(Intervention("replace", "attribute_replace", entity.id, f"attributes.{attr_key}", attr_value, new_value))
        if entity.count is not None:
            for new_count in sorted({max(1, entity.count - 1), entity.count + 1}):
                if new_count != entity.count:
                    candidates.append(Intervention("replace", "count_change", entity.id, "count", entity.count, new_count))
        if entity.location:
            for new_location in LOCATIONS:
                if new_location != entity.location:
                    candidates.append(Intervention("replace", "spatial_relation_change", entity.id, "location", entity.location, new_location))

    metadata = record.get("source_metadata") or record.get("metadata") or {}
    primary_type = str(metadata.get("primary_intervention_type", "")).strip()
    primary_entity_id = str(metadata.get("primary_entity_id", "")).strip()
    if primary_type:
        candidates = [candidate for candidate in candidates if candidate.intervention_type == primary_type]
    if primary_entity_id:
        candidates = [candidate for candidate in candidates if candidate.entity_id == primary_entity_id]
    if (primary_type or primary_entity_id) and not candidates:
        raise ValueError(
            f"source scene {scene_id} has no intervention candidate for "
            f"type={primary_type!r}, entity={primary_entity_id!r}"
        )

    for local_id, intervention in enumerate(itertools.islice(candidates, max_per_scene)):
        target_scene = apply_intervention(source_scene, intervention)
        target_prompt = str(record.get("target_prompt") or "").strip() or scene_to_prompt(target_scene)
        pair_id = f"cf_{scene_id}_{local_id:04d}"
        yield {
            "schema_version": "1.0",
            "pair_id": pair_id,
            "scene_id": scene_id,
            "template_family": record.get("template_family", "unknown"),
            "source": {
                "prompt": source_prompt,
                "cot": source_scene.to_dict(),
                "cot_text": record.get("cot_text") or render_cot_text(source_scene),
                "image": record.get("image", ""),
                "generation": record.get("source_generation", {}),
                "quality": source_quality,
            },
            "intervention": intervention.to_dict(),
            "target": {
                "prompt": target_prompt,
                "cot": target_scene.to_dict(),
                "cot_text": render_cot_text(target_scene),
                "image": "",
                "mask": "",
            },
            "preservation": _preservation_for(source_scene, intervention),
            "generation": {"target_generation_mode": "pending", "request_ids": [], "parameters": {}},
            "quality": {
                "target_success": 0.0,
                "preservation_score": 0.0,
                "binding_score": 0.0,
                "locality_score": 0.0,
                "image_quality_score": 0.0,
                "accepted": False,
                "review_status": "pending",
            },
            "split": "unassigned",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic counterfactual interventions.")
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-per-scene", type=int, default=8)
    parser.add_argument(
        "--allow-unreviewed-source",
        action="store_true",
        help="Testing only: allow intervention pairs before source_quality.accepted=true.",
    )
    args = parser.parse_args()

    records = load_jsonl(args.source_registry)
    pairs = []
    for record in records:
        pairs.extend(generate_for_scene(record, args.max_per_scene, args.allow_unreviewed_source))
    write_jsonl(args.output, pairs)
    print(f"Wrote {len(pairs)} intervention pairs to {args.output}")


if __name__ == "__main__":
    main()