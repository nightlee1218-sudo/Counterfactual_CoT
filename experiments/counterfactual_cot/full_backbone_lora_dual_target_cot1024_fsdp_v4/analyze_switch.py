#!/usr/bin/env python3
"""Analyze same-seed left/right Target-CoT switch behavior at v4 step 0 and 500."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.full_backbone_lora_dual_target_cot1024_fsdp_v4.dual_target_data import (  # noqa: E402
    BRANCHES,
)
from experiments.counterfactual_cot.full_backbone_lora_v3_strict_step0_step500_control.analyze_controls import (  # noqa: E402
    EXPECTED_SOURCE_SHA256,
    EXPECTED_TARGET_SHA256,
    _atomic_json,
    _build_masks,
    _load_rgb,
    _mae,
    _red_stats,
    _sha256,
)

DEFAULT_OUTPUT = BAGEL_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "cf_singlepair_fullL0-27_lora_r8_dualTargetCoT1024_fsdpFullShard_overfit_v4"
)


def _largest_red_component(image: np.ndarray) -> Dict[str, Any] | None:
    hsv = np.asarray(Image.fromarray(image).convert("HSV"))
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((hue < 15) | (hue > 240)) & (saturation > 110) & (value > 80)
    mask[:520] = False
    seen = np.zeros_like(mask, dtype=bool)
    best: List[Tuple[int, int]] = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        points: List[Tuple[int, int]] = []
        while stack:
            current_y, current_x = stack.pop()
            points.append((current_y, current_x))
            for delta_y, delta_x in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_y, next_x = current_y + delta_y, current_x + delta_x
                if 0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x] and not seen[next_y, next_x]:
                    seen[next_y, next_x] = True
                    stack.append((next_y, next_x))
        if len(points) > len(best):
            best = points
    if not best:
        return None
    ys = np.asarray([point[0] for point in best])
    xs = np.asarray([point[1] for point in best])
    return {
        "pixels": len(best),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "bbox_ltrb": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    eval_dir = args.output_dir / "evaluation_dual_target_cot_switch"
    reports: Dict[int, Dict[str, Any]] = {}
    for step in (0, 500):
        report_path = eval_dir / f"step_{step:06d}/evaluation_report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"Missing evaluation report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "completed" or int(report.get("step", -1)) != step:
            raise RuntimeError(f"Invalid evaluation report: {report_path}")
        reports[step] = report
    if reports[0]["prompt_manifest"] != reports[500]["prompt_manifest"]:
        raise RuntimeError("Step 0 and step 500 prompt manifests differ")
    if reports[0]["seeds"] != reports[500]["seeds"]:
        raise RuntimeError("Step 0 and step 500 seed sets differ")
    if reports[0]["generation_parameters"] != reports[500]["generation_parameters"]:
        raise RuntimeError("Step 0 and step 500 generation parameters differ")

    source_path = Path(reports[0]["source_visualization"])
    target_path = Path(reports[0]["target_visualization"])
    if _sha256(source_path) != EXPECTED_SOURCE_SHA256 or _sha256(target_path) != EXPECTED_TARGET_SHA256:
        raise RuntimeError("Reference image hash mismatch")
    source = _load_rgb(source_path)
    target = _load_rgb(target_path)
    masks = _build_masks(source, target)
    source_red = _red_stats(source, masks)
    target_red = _red_stats(target, masks)
    old_source = int(source_red["red_pixels_old_roi"])
    old_target = int(target_red["red_pixels_old_roi"])
    new_source = int(source_red["red_pixels_new_roi"])
    new_target = int(target_red["red_pixels_new_roi"])
    old_reference = max(1, old_source - old_target)
    new_reference = max(1, new_target - new_source)

    rows: List[Dict[str, Any]] = []
    for step in (0, 500):
        for pair_row in reports[step]["pairs"]:
            seed = int(pair_row["seed"])
            images = {branch: _load_rgb(Path(pair_row[f"{branch}_path"])) for branch in BRANCHES}
            left_red = _red_stats(images["left"], masks)
            right_red = _red_stats(images["right"], masks)
            left_old_retention = (left_red["red_pixels_old_roi"] - old_target) / old_reference
            left_new_addition = (left_red["red_pixels_new_roi"] - new_source) / new_reference
            right_old_removal = (old_source - right_red["red_pixels_old_roi"]) / old_reference
            right_new_addition = (right_red["red_pixels_new_roi"] - new_source) / new_reference
            left_component = _largest_red_component(images["left"])
            right_component = _largest_red_component(images["right"])
            left_success = bool(left_old_retention >= 0.50 and left_new_addition <= 0.25)
            right_success = bool(right_old_removal >= 0.50 and right_new_addition >= 0.50)
            strict_left = bool(left_success and left_component is not None and left_component["centroid_x"] < 270)
            strict_right = bool(right_success and right_component is not None and right_component["centroid_x"] >= 810)
            row = {
                "step": step,
                "seed": seed,
                "left_path": pair_row["left_path"],
                "right_path": pair_row["right_path"],
                "left_old_retention_progress": left_old_retention,
                "left_new_addition_progress": left_new_addition,
                "right_old_removal_progress": right_old_removal,
                "right_new_addition_progress": right_new_addition,
                "left_proxy_success": left_success,
                "right_proxy_success": right_success,
                "strict_left_position_success": strict_left,
                "strict_right_position_success": strict_right,
                "paired_switch_success": bool(strict_left and strict_right),
                "left_component": left_component,
                "right_component": right_component,
                "left_target_mae": _mae(images["left"], source, masks["old_edit"] | masks["new_edit"]),
                "right_target_mae": _mae(images["right"], target, masks["old_edit"] | masks["new_edit"]),
                "left_preserve_source_mae": _mae(images["left"], source, masks["preserve"]),
                "right_preserve_source_mae": _mae(images["right"], source, masks["preserve"]),
                "paired_output_mae": float(np.abs(images["left"].astype(np.float32) - images["right"].astype(np.float32)).mean() / 255.0),
            }
            rows.append(row)

    summaries: Dict[str, Any] = {}
    for step in (0, 500):
        selected = [row for row in rows if row["step"] == step]
        summaries[str(step)] = {
            "num_seeds": len(selected),
            "left_proxy_successes": sum(row["left_proxy_success"] for row in selected),
            "right_proxy_successes": sum(row["right_proxy_success"] for row in selected),
            "strict_left_position_successes": sum(row["strict_left_position_success"] for row in selected),
            "strict_right_position_successes": sum(row["strict_right_position_success"] for row in selected),
            "paired_switch_successes": sum(row["paired_switch_success"] for row in selected),
            "mean_paired_output_mae": sum(row["paired_output_mae"] for row in selected) / len(selected),
            "mean_left_target_mae": sum(row["left_target_mae"] for row in selected) / len(selected),
            "mean_right_target_mae": sum(row["right_target_mae"] for row in selected) / len(selected),
            "mean_left_preserve_source_mae": sum(row["left_preserve_source_mae"] for row in selected) / len(selected),
            "mean_right_preserve_source_mae": sum(row["right_preserve_source_mae"] for row in selected) / len(selected),
        }
    report = {
        "status": "completed",
        "definition": {
            "left_condition": "same source + Source Prompt + Source CoT(left) + Target CoT(left)",
            "right_condition": "same source + Source Prompt + Source CoT(left) + Target CoT(right)",
            "paired_switch_success": "same seed produces strict left under left CoT and strict right under right CoT",
            "strict_left": "left proxy succeeds and largest lower-image red component centroid x < 270",
            "strict_right": "right proxy succeeds and largest lower-image red component centroid x >= 810",
            "warning": "Automatic red and centroid proxies require visual review for exact object count and semantic quality.",
        },
        "reference_red_stats": {"source": source_red, "target": target_red},
        "summaries": summaries,
        "rows": rows,
    }
    analysis_dir = args.output_dir / "evaluation_dual_target_cot_switch/analysis"
    _atomic_json(analysis_dir / "switch_metrics.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
