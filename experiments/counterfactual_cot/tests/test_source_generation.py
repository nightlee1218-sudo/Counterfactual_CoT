from collections import Counter

from experiments.counterfactual_cot.data.generate_interventions import generate_for_scene
from experiments.counterfactual_cot.data.generate_source_scenes import INTERVENTION_TYPES, generate_scenes
from experiments.counterfactual_cot.data.schema import SceneGraph, render_cot_text


def test_source_generation_is_balanced_and_valid():
    scenes = generate_scenes()
    assert len(scenes) == 40
    assert len({scene["scene_id"] for scene in scenes}) == 40
    assert len({scene["template_family"] for scene in scenes}) == 10
    counts = Counter(scene["metadata"]["primary_intervention_type"] for scene in scenes)
    assert counts == Counter({kind: 10 for kind in INTERVENTION_TYPES})
    for record in scenes:
        scene = SceneGraph.from_dict(record["cot"])
        assert record["prompt"].strip()
        assert ".relation.0=" in render_cot_text(scene)


def test_each_source_yields_its_declared_intervention():
    for record in generate_scenes():
        registry_record = dict(record)
        registry_record["image"] = f"source/{record['scene_id']}.png"
        registry_record["source_quality"] = {"accepted": True}
        pairs = list(generate_for_scene(registry_record, max_per_scene=1))
        assert len(pairs) == 1
        assert pairs[0]["intervention"]["intervention_type"] == record["metadata"]["primary_intervention_type"]
        assert pairs[0]["intervention"]["entity_id"] == record["metadata"]["primary_entity_id"]
