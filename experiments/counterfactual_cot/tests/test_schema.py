from experiments.counterfactual_cot.data.schema import SceneGraph, render_cot_text


def test_scene_graph_roundtrip():
    scene = SceneGraph.from_dict(
        {
            "entities": [
                {"id": "obj_1", "category": "banana", "count": 2, "location": "upper-left"}
            ],
            "background": {"surface": "wooden table"},
        }
    )
    assert scene.to_dict()["entities"][0]["category"] == "banana"
    assert "obj_1.object=banana" in render_cot_text(scene)
