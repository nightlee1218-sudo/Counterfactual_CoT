from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

from .render_prompts import scene_to_prompt
from .schema import SceneGraph, write_jsonl

INTERVENTION_TYPES = (
    "object_replace",
    "attribute_replace",
    "count_change",
    "spatial_relation_change",
)

COLORS = ("red", "blue", "yellow", "green")
COUNTS = (1, 2, 3, 4)
LOCATIONS = ("upper-left", "upper-right", "center", "lower-left", "lower-right")

SCENE_TEMPLATES: List[Dict[str, Any]] = [
    {
        "family": "home_kitchen",
        "primary": "cup",
        "context": {"id": "plate_1", "category": "plate", "count": 1, "location": "center"},
        "background": {"scene": "bright home kitchen", "surface": "wooden counter", "lighting": "soft daylight"},
        "relation": "beside",
    },
    {
        "family": "living_room_play",
        "primary": "ball",
        "context": {"id": "sofa_1", "category": "sofa", "count": 1, "location": "upper-right"},
        "background": {"scene": "cozy living room", "floor": "light rug", "lighting": "warm indoor light"},
        "relation": "in_front_of",
    },
    {
        "family": "grocery_display",
        "primary": "apple",
        "context": {"id": "basket_1", "category": "wicker basket", "count": 1, "location": "center"},
        "background": {"scene": "grocery produce display", "surface": "wooden shelf", "lighting": "even store lighting"},
        "relation": "beside",
    },
    {
        "family": "city_street",
        "primary": "car",
        "context": {"id": "lamp_1", "category": "street lamp", "count": 1, "location": "upper-left"},
        "background": {"scene": "quiet city street", "weather": "clear", "time": "early evening"},
        "relation": "near",
    },
    {
        "family": "garden_animals",
        "primary": "dog",
        "context": {"id": "bench_1", "category": "park bench", "count": 1, "location": "upper-right"},
        "background": {"scene": "green public garden", "ground": "short grass", "weather": "sunny"},
        "relation": "in_front_of",
    },
    {
        "family": "classroom",
        "primary": "chair",
        "context": {"id": "desk_1", "category": "school desk", "count": 1, "location": "center"},
        "background": {"scene": "tidy classroom", "floor": "pale wooden floor", "lighting": "daylight"},
        "relation": "beside",
    },
    {
        "family": "flower_shop",
        "primary": "flower",
        "context": {"id": "vase_1", "category": "glass vase", "count": 1, "location": "center"},
        "background": {"scene": "small flower shop", "surface": "white display table", "lighting": "soft natural light"},
        "relation": "inside",
    },
    {
        "family": "library_table",
        "primary": "book",
        "context": {"id": "lamp_1", "category": "reading lamp", "count": 1, "location": "upper-right"},
        "background": {"scene": "quiet library", "surface": "dark wooden table", "lighting": "focused reading light"},
        "relation": "beside",
    },
    {
        "family": "lakeside_birds",
        "primary": "bird",
        "context": {"id": "rock_1", "category": "large rock", "count": 1, "location": "center"},
        "background": {"scene": "calm lakeside", "weather": "clear", "lighting": "morning light"},
        "relation": "on",
    },
    {
        "family": "seaside_path",
        "primary": "bicycle",
        "context": {"id": "sign_1", "category": "wooden sign", "count": 1, "location": "upper-left"},
        "background": {"scene": "seaside path", "weather": "sunny", "background": "blue ocean"},
        "relation": "near",
    },
]


def _entity_id(category: str) -> str:
    return category.replace(" ", "_") + "_1"


def build_scene(template: Dict[str, Any], intervention_type: str, variant: int) -> Dict[str, Any]:
    category = str(template["primary"])
    primary_id = _entity_id(category)
    color = COLORS[variant % len(COLORS)]
    count = COUNTS[(variant + 1) % len(COUNTS)]
    location = LOCATIONS[(variant + 2) % len(LOCATIONS)]
    context = dict(template["context"])
    if template["relation"] in {"inside", "on"}:
        context["location"] = location
    primary = {
        "id": primary_id,
        "category": category,
        "count": count,
        "attributes": {"color": color},
        "location": location,
        "relations": [{"predicate": template["relation"], "target": context["id"]}],
    }
    scene = SceneGraph.from_dict(
        {
            "schema_version": "1.0",
            "entities": [primary, context],
            "background": dict(template["background"]),
        }
    )
    return {
        "schema_version": "1.0",
        "scene_id": f"source_{template['family']}_{intervention_type}",
        "template_family": template["family"],
        "prompt": scene_to_prompt(scene),
        "cot": scene.to_dict(),
        "metadata": {
            "construction": "controlled_template_v1",
            "primary_entity_id": primary_id,
            "primary_intervention_type": intervention_type,
            "requires_source_review": True,
        },
    }


def generate_scenes(seed: int = 20260817) -> List[Dict[str, Any]]:
    scenes: List[Dict[str, Any]] = []
    for template_index, template in enumerate(SCENE_TEMPLATES):
        for intervention_index, intervention_type in enumerate(INTERVENTION_TYPES):
            scenes.append(build_scene(template, intervention_type, template_index + intervention_index))
    random.Random(seed).shuffle(scenes)
    return scenes


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled and diverse source Prompt/CoT candidates.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--max-scenes", type=int, default=None)
    args = parser.parse_args()

    scenes = generate_scenes(args.seed)
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("--max-scenes must be positive")
        scenes = scenes[: args.max_scenes]
    write_jsonl(args.output, scenes)
    counts = {kind: 0 for kind in INTERVENTION_TYPES}
    for scene in scenes:
        counts[scene["metadata"]["primary_intervention_type"]] += 1
    print(f"Wrote {len(scenes)} source candidates to {args.output}")
    print("Intervention targets: " + ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
