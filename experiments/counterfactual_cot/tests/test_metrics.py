from experiments.counterfactual_cot.evaluation.paired_bootstrap import bootstrap


def test_bootstrap_smoke():
    result = bootstrap({"s1": (1.0, 0.0), "s2": (0.5, 0.5)}, num_samples=100, seed=0)
    assert result["num_scenes"] == 2
    assert "mean_diff" in result
