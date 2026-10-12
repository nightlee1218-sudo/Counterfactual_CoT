from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from data import dataset_info  # noqa: E402
from experiments.counterfactual_cot.datasets import CounterfactualCoTDataset  # noqa: E402
from train.pretrain_unified_navit import main as bagel_train_main  # noqa: E402

EXPERIMENT_ROOT = BAGEL_ROOT / "experiments" / "counterfactual_cot"
DEFAULT_MANIFEST_DIR = EXPERIMENT_ROOT / "manifests"


def _manifest_paths() -> Dict[str, str]:
    train_manifest = os.environ.get("CFCOT_MANIFEST_TRAIN", str(DEFAULT_MANIFEST_DIR / "manifest_train.jsonl"))
    val_manifest = os.environ.get("CFCOT_MANIFEST_VAL", str(DEFAULT_MANIFEST_DIR / "manifest_val.jsonl"))
    test_manifest = os.environ.get("CFCOT_MANIFEST_TEST", str(DEFAULT_MANIFEST_DIR / "manifest_test.jsonl"))
    return {"train": train_manifest, "val": val_manifest, "test": test_manifest}


def register_counterfactual_dataset() -> None:
    """Register experiment datasets without editing BAGEL core registry files."""
    paths = _manifest_paths()
    dataset_info.DATASET_REGISTRY["counterfactual_cot"] = CounterfactualCoTDataset
    dataset_info.DATASET_INFO["counterfactual_cot"] = {
        "standard_train": {
            "manifest_path": paths["train"],
            "variant": "standard_cot",
            "training_seed": int(os.environ.get("CFCOT_TRAINING_SEED", "3407")),
            "num_total_samples": 0,
        },
        "independent_train": {
            "manifest_path": paths["train"],
            "variant": "independent_cf",
            "training_seed": int(os.environ.get("CFCOT_TRAINING_SEED", "3407")),
            "num_total_samples": 0,
        },
        "paired_train": {
            "manifest_path": paths["train"],
            "variant": "paired_cf_mvp",
            "training_seed": int(os.environ.get("CFCOT_TRAINING_SEED", "3407")),
            "num_total_samples": 0,
        },
        "paired_generation_train": {
            "manifest_path": paths["train"],
            "variant": "paired_cf_generation",
            "training_seed": int(os.environ.get("CFCOT_TRAINING_SEED", "3407")),
            "num_total_samples": 0,
        },
        "val": {
            "manifest_path": paths["val"],
            "variant": "paired_cf_mvp",
            "training_seed": int(os.environ.get("CFCOT_TRAINING_SEED", "3407")),
            "num_total_samples": 0,
        },
        "test": {
            "manifest_path": paths["test"],
            "variant": "paired_cf_mvp",
            "training_seed": int(os.environ.get("CFCOT_TRAINING_SEED", "3407")),
            "num_total_samples": 0,
        },
    }


def main() -> None:
    register_counterfactual_dataset()
    bagel_train_main()


if __name__ == "__main__":
    main()
