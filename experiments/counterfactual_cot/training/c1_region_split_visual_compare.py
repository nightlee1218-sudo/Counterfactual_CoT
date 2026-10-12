"""Strict visual matrix: full-image C1 LoRA versus region-split C1 LoRA.

This configures the audited fresh-worker visual driver with two LoRA
architectures and frozen Step000/400/500 checkpoints. Step400/500 generation is
gated on bit-exact Step000 outputs for all preregistered seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.counterfactual_cot.training import c1_single_pair_visual_compare as driver


MODULE_NAME = "experiments.counterfactual_cot.training.c1_region_split_visual_compare"
FULL_IMAGE_ARCH = "full_image_c1_lora"
REGION_SPLIT_ARCH = "region_split_c1_lora"
ARCHITECTURES = (FULL_IMAGE_ARCH, REGION_SPLIT_ARCH)
STEPS = (0, 400, 500)
FORMAL_OUTPUT_DIR = REPO_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_c1_region_split_visual_compare_v1"
)
PREFLIGHT_OUTPUT_DIR = REPO_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_c1_region_split_visual_compare_preflight_v1"
)

FULL_IMAGE_CHECKPOINT_DIR = REPO_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_c1_single_ball_p4d_v1/checkpoints"
)
REGION_SPLIT_CHECKPOINT_DIR = REPO_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_c1_single_ball_region_split_v1_500steps/checkpoints"
)
CHECKPOINTS: Dict[Tuple[str, int], Dict[str, Any]] = {
    (FULL_IMAGE_ARCH, 0): {
        "path": FULL_IMAGE_CHECKPOINT_DIR / "step_000_trainable.pt",
        "size_bytes": 135_587_762,
        "sha256": "0af1669d6ad50f7354b850917401a6862970221f63c786311f12fb5aad924001",
    },
    (FULL_IMAGE_ARCH, 400): {
        "path": FULL_IMAGE_CHECKPOINT_DIR / "step_400_trainable.pt",
        "size_bytes": 135_587_762,
        "sha256": "58a69f9b3a1dd34569c56ebe0131ec2e0a4edd36d87f1018fc6889be59620995",
    },
    (FULL_IMAGE_ARCH, 500): {
        "path": FULL_IMAGE_CHECKPOINT_DIR / "step_500_trainable.pt",
        "size_bytes": 135_587_762,
        "sha256": "ca15aef699e9255e305a865fb3ea113531f101f021f60a58c87a92fed4a602ac",
    },
    (REGION_SPLIT_ARCH, 0): {
        "path": REGION_SPLIT_CHECKPOINT_DIR / "step_000_trainable.pt",
        "size_bytes": 135_587_762,
        "sha256": "9658b383e277fc9ffec65985f29e4ffa18747f8f437c56e584ec400531659dc8",
    },
    (REGION_SPLIT_ARCH, 400): {
        "path": REGION_SPLIT_CHECKPOINT_DIR / "step_400_trainable.pt",
        "size_bytes": 135_587_762,
        "sha256": "4535d99ccdc009ff369074897a95fec39313f612d50d822ec7893493aeaff4d8",
    },
    (REGION_SPLIT_ARCH, 500): {
        "path": REGION_SPLIT_CHECKPOINT_DIR / "step_500_trainable.pt",
        "size_bytes": 135_587_762,
        "sha256": "98ec12e8330b744394a3c688faaf1302485af504d080d9f8a4c016b80d6dd740",
    },
}

_ORIGINAL_STRICT_LOAD = driver._strict_load_trainable
_ORIGINAL_VERIFY_FROZEN_INPUTS = driver._verify_frozen_inputs
_ORIGINAL_VALIDATE_MODEL_OUTPUT_PATH = driver._validate_model_output_path
_ORIGINAL_RUN_STATE_PATH = driver._run_state_path


def _strict_load_lora(
    payload: Any,
    model: Any,
    architecture: str,
    torch: Any,
) -> Dict[str, Any]:
    if architecture not in ARCHITECTURES:
        raise RuntimeError(f"Unexpected LoRA architecture: {architecture!r}")
    report = _ORIGINAL_STRICT_LOAD(payload, model, "c1_lora", torch)
    report["requested_architecture"] = architecture
    report["loader_contract"] = "c1_lora_14_bfloat16_plus_48_float32"
    return report


def _verify_frozen_inputs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bindings, pair = _ORIGINAL_VERIFY_FROZEN_INPUTS()
    engine_path = Path(driver.__file__).resolve()
    bindings["driver_engine"] = {
        "path": str(engine_path),
        "size_bytes": engine_path.stat().st_size,
        "sha256": driver._sha256_file(engine_path),
    }
    bindings["matrix_contract"] = {
        "architectures": list(ARCHITECTURES),
        "steps": list(STEPS),
        "seeds": list(driver.SEEDS),
        "step000_gate": "bit_exact_png_and_rgb_pixels_for_every_seed",
        "later_phase_gate": "Step400/500 forbidden until Step000 passes",
    }
    return bindings, pair


def _validate_model_output_path(
    raw_path: Any,
    architecture: str,
    step: int,
    seed: int,
    *,
    output_root: Path | None = None,
) -> Path:
    return _ORIGINAL_VALIDATE_MODEL_OUTPUT_PATH(
        raw_path,
        architecture,
        step,
        seed,
        output_root=FORMAL_OUTPUT_DIR if output_root is None else output_root,
    )


def _run_state_path(output_root: Path | None = None) -> Path:
    return _ORIGINAL_RUN_STATE_PATH(
        FORMAL_OUTPUT_DIR if output_root is None else output_root
    )


def _compare_step000(worker_runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from PIL import Image, ImageChops

    if not isinstance(worker_runs, (list, tuple)) or len(worker_runs) != 2:
        raise RuntimeError("Step000 requires exactly two LoRA worker runs")
    runs_by_architecture: Dict[str, Dict[str, Any]] = {}
    for run in worker_runs:
        if not isinstance(run, dict):
            raise RuntimeError("Each Step000 worker run must be a dict")
        architecture = run.get("architecture")
        if architecture not in ARCHITECTURES:
            raise RuntimeError(f"Unexpected Step000 architecture: {architecture!r}")
        if architecture in runs_by_architecture:
            raise RuntimeError(f"Duplicate Step000 architecture: {architecture}")
        runs_by_architecture[architecture] = run
    if set(runs_by_architecture) != set(ARCHITECTURES):
        raise RuntimeError(
            f"Step000 architecture set mismatch: {sorted(runs_by_architecture)}"
        )

    full_image = driver._index_worker_outputs(
        runs_by_architecture[FULL_IMAGE_ARCH], FULL_IMAGE_ARCH, 0
    )
    region_split = driver._index_worker_outputs(
        runs_by_architecture[REGION_SPLIT_ARCH], REGION_SPLIT_ARCH, 0
    )
    metadata_fields = (
        "size_bytes",
        "png_sha256",
        "rgb_pixel_sha256",
        "dimensions",
        "shape",
        "native_mode",
        "mode",
    )
    comparisons: List[Dict[str, Any]] = []
    for seed in driver.SEEDS:
        left = full_image[seed]
        right = region_split[seed]
        left_path = driver._validate_model_output_path(
            left["path"], FULL_IMAGE_ARCH, 0, seed
        )
        right_path = driver._validate_model_output_path(
            right["path"], REGION_SPLIT_ARCH, 0, seed
        )
        left_actual = driver._image_record(left_path)
        right_actual = driver._image_record(right_path)
        left_report_matches_file = all(
            left.get(field) == left_actual[field] for field in metadata_fields
        )
        right_report_matches_file = all(
            right.get(field) == right_actual[field] for field in metadata_fields
        )
        with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
            if left_image.mode != driver.IMAGE_MODE or right_image.mode != driver.IMAGE_MODE:
                raise RuntimeError(
                    f"Step000 native mode mismatch at seed={seed}: "
                    f"full_image={left_image.mode}, region_split={right_image.mode}"
                )
            left_rgb = left_image.convert("RGB")
            right_rgb = right_image.convert("RGB")
            shape_equal = left_rgb.size == right_rgb.size
            max_abs_diff = None
            if shape_equal:
                extrema = ImageChops.difference(left_rgb, right_rgb).getextrema()
                max_abs_diff = max(channel_max for _, channel_max in extrema)
        png_equal = left_actual["png_sha256"] == right_actual["png_sha256"]
        pixel_equal = (
            left_actual["rgb_pixel_sha256"] == right_actual["rgb_pixel_sha256"]
        )
        passed = (
            left_report_matches_file
            and right_report_matches_file
            and png_equal
            and pixel_equal
            and shape_equal
            and max_abs_diff == 0
        )
        comparisons.append(
            {
                "seed": seed,
                "full_image_output": left_actual,
                "region_split_output": right_actual,
                "full_image_report_matches_file": left_report_matches_file,
                "region_split_report_matches_file": right_report_matches_file,
                "png_sha256_equal": png_equal,
                "rgb_pixel_sha256_equal": pixel_equal,
                "shape_equal": shape_equal,
                "pixel_max_abs_diff": max_abs_diff,
                "passed": passed,
            }
        )
    return comparisons


def _configure_driver() -> None:
    driver.SCRIPT_PATH = SCRIPT_PATH
    driver.MODULE_NAME = MODULE_NAME
    driver.FORMAL_OUTPUT_DIR = FORMAL_OUTPUT_DIR
    driver.PREFLIGHT_OUTPUT_DIR = PREFLIGHT_OUTPUT_DIR
    driver.ARCHITECTURES = ARCHITECTURES
    driver.STEPS = STEPS
    driver.CHECKPOINTS = CHECKPOINTS
    driver.EXPECTED_OUTPUT_COUNT = len(ARCHITECTURES) * len(STEPS) * len(driver.SEEDS)
    driver._strict_load_trainable = _strict_load_lora
    driver._verify_frozen_inputs = _verify_frozen_inputs
    driver._validate_model_output_path = _validate_model_output_path
    driver._run_state_path = _run_state_path
    driver._compare_step000 = _compare_step000


def main() -> int:
    _configure_driver()
    return driver.main()


if __name__ == "__main__":
    raise SystemExit(main())
