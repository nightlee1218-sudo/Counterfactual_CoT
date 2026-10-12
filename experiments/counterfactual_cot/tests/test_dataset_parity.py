from pathlib import Path

import yaml

from experiments.counterfactual_cot.datasets.bagel_manifest_dataset import CounterfactualCoTDataset


ROOT = Path(__file__).resolve().parents[1]


def _load_group(config_name: str):
    return yaml.safe_load((ROOT / "configs" / config_name).read_text(encoding="utf-8"))["counterfactual_cot"]


def test_independent_and_paired_configs_keep_same_data_budget_contract():
    independent = _load_group("independent_cf_sft.yaml")
    paired = _load_group("paired_cf_mvp.yaml")

    assert independent["dataset_names"] == ["independent_train"]
    assert paired["dataset_names"] == ["paired_train"]
    assert independent["num_used_data"] == paired["num_used_data"]
    assert independent["image_transform_args"] == paired["image_transform_args"]
    assert independent["vit_image_transform_args"] == paired["vit_image_transform_args"]
    assert independent["weight"] == paired["weight"] == 1.0


def test_training_registry_maps_independent_and_paired_to_same_train_manifest():
    train_source = (ROOT / "training" / "train.py").read_text(encoding="utf-8")

    assert '"independent_train"' in train_source
    assert '"paired_train"' in train_source
    assert '"manifest_path": paths["train"]' in train_source
    assert '"variant": "independent_cf"' in train_source
    assert '"variant": "paired_cf_mvp"' in train_source


def test_standard_and_independent_are_not_identical_training_views(monkeypatch):
    pair = {
        "source": {"prompt": "source prompt", "cot_text": "source cot", "image": "source.png"},
        "target": {"prompt": "target prompt", "cot_text": "target cot", "image": "target.png"},
    }

    calls = []

    def fake_standard_sample(self, pair_arg, side):
        calls.append((self.variant, "standard", side))
        return {}, side, "standard_cot"

    def fake_independent_sample(self, pair_arg):
        calls.append((self.variant, "independent_edit", "target"))
        return {}, "target", "independent_cf"

    monkeypatch.setattr(CounterfactualCoTDataset, "_standard_sample", fake_standard_sample)
    monkeypatch.setattr(CounterfactualCoTDataset, "_independent_edit_sample", fake_independent_sample)

    standard = object.__new__(CounterfactualCoTDataset)
    standard.variant = "standard_cot"
    independent = object.__new__(CounterfactualCoTDataset)
    independent.variant = "independent_cf"

    assert CounterfactualCoTDataset._make_sample(standard, pair, sample_index=0, epoch=0)[1] == "source"
    assert CounterfactualCoTDataset._make_sample(independent, pair, sample_index=0, epoch=0)[1] == "target"
    assert calls == [
        ("standard_cot", "standard", "source"),
        ("independent_cf", "independent_edit", "target"),
    ]
