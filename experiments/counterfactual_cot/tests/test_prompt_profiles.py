import pytest

from experiments.counterfactual_cot.data.prompt_profiles import (
    CORE_PROFILE_IDS,
    PROFILES,
    render_prompt_profile,
)
from experiments.counterfactual_cot.data.render_prompts import paired_prompt


PAIR = {
    "source": {"prompt": "source prompt left", "cot_text": "ball.location=left"},
    "target": {"prompt": "target prompt right", "cot_text": "ball.location=right"},
    "intervention": {
        "intervention_type": "spatial_position_change",
        "entity_id": "ball",
        "field": "location",
        "old_value": "left",
        "new_value": "right",
    },
    "preservation": {
        "must_preserve": ["ball.count", "background"],
        "allowed_to_change": ["ball.location"],
    },
}


def test_legacy_profile_is_bit_exact_with_existing_renderer():
    assert render_prompt_profile(PAIR, "legacy_paired_v1") == paired_prompt(
        PAIR, include_source_image=True
    )


def test_cot_core_contains_only_intended_semantic_components():
    prompt = render_prompt_profile(PAIR, "cot_core_v1")
    assert "Source Prompt: source prompt left" in prompt
    assert "Target CoT: ball.location=right" in prompt
    assert "Source CoT:" not in prompt
    assert "Intervention:" not in prompt
    assert "Target Prompt:" not in prompt
    assert "Preserve:" not in prompt


def test_full_paired_and_target_cot_ablation_differ_only_by_target_cot_line():
    full = render_prompt_profile(PAIR, "full_paired_v2")
    ablated = render_prompt_profile(PAIR, "full_without_target_cot_v1")
    assert "Target Prompt: target prompt right" in full
    assert "Target Prompt: target prompt right" in ablated
    target_cot_line = "Target CoT: ball.location=right\n"
    assert target_cot_line in full
    assert full.replace(target_cot_line, "") == ablated


def test_core_profile_contract_is_complete():
    assert CORE_PROFILE_IDS == (
        "legacy_paired_v1",
        "cot_core_v1",
        "full_paired_v2",
        "target_prompt_only_v1",
        "delta_only_v2",
        "full_without_target_cot_v1",
    )
    assert set(PROFILES) >= set(CORE_PROFILE_IDS)


@pytest.mark.parametrize(
    ("intervention", "expected", "forbidden"),
    [
        (
            {"intervention_type": "attribute_replace", "entity_id": "book_1", "field": "attributes.color", "old_value": "red", "new_value": "blue"},
            ("Change the color of the book from red to blue.", "must not retain the old red color"),
            ("Change book_1.location", "exactly one instance"),
        ),
        (
            {"intervention_type": "count_change", "entity_id": "bicycle_1", "field": "count", "old_value": 1, "new_value": 2},
            ("Change the number of bicycle instances from 1 to 2.", "exactly 2 bicycle instances"),
            ("exactly one instance", "Move the bicycle"),
        ),
        (
            {"intervention_type": "object_replace", "entity_id": "chair_1", "field": "category", "old_value": "chair", "new_value": "globe"},
            ("Replace the chair with globe.", "do not retain the old chair"),
            ("Change chair_1.location",),
        ),
        (
            {"intervention_type": "spatial_position_change", "entity_id": "car_1", "field": "location", "old_value": "left_region", "new_value": "right_region"},
            ("Move the car from left region to right region.", "Remove the car from left region"),
            ("Change car_1.location",),
        ),
        (
            {"intervention_type": "spatial_relation_change", "entity_id": "cup_1", "field": "relations.0.predicate", "old_value": "beside", "new_value": "in_front_of"},
            ("relation of the cup to the plate from beside to in front of", "preserve both objects and their counts"),
            ("Change cup_1.location",),
        ),
    ],
)
def test_delta_only_v2_is_intervention_type_aware(intervention, expected, forbidden):
    pair = {
        "source": {
            "prompt": "source",
            "cot_text": "book_1.object=book; bicycle_1.object=bicycle; chair_1.object=chair; car_1.object=car; cup_1.object=cup; cup_1.relation.0=beside:plate_1; plate_1.object=plate",
            "scene_graph": {
                "entities": [
                    {"id": "book_1", "category": "book"},
                    {"id": "bicycle_1", "category": "bicycle"},
                    {"id": "chair_1", "category": "chair"},
                    {"id": "car_1", "category": "car"},
                    {"id": "cup_1", "category": "cup", "relations": [{"predicate": "beside", "target_id": "plate_1"}]},
                    {"id": "plate_1", "category": "plate"},
                ]
            },
        },
        "target": {"prompt": "target", "cot_text": "target cot"},
        "intervention": intervention,
        "preservation": {"must_preserve": ["background"]},
    }
    prompt = render_prompt_profile(pair, "delta_only_v2")
    for fragment in expected:
        assert fragment in prompt
    for fragment in forbidden:
        assert fragment not in prompt
