from experiments.counterfactual_cot.data.schema import Intervention, SceneGraph, apply_intervention


def test_apply_object_replace():
    source = SceneGraph.from_dict(
        {"entities": [{"id": "obj_1", "category": "banana", "count": 2}], "background": {}}
    )
    intervention = Intervention.from_dict(
        {
            "operation": "replace",
            "intervention_type": "object_replace",
            "entity_id": "obj_1",
            "field": "category",
            "old_value": "banana",
            "new_value": "apple",
        }
    )
    target = apply_intervention(source, intervention)
    assert target.entities[0].category == "apple"
    assert target.entities[0].count == 2
