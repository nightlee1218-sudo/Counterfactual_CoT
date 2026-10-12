#!/usr/bin/env python3
"""Paired analysis for three prompt/CoT conditions at v3 step 0 and step 500."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.full_backbone_lora_v3_prompt_cot_order_control.prompt_conditions import (  # noqa: E402
    CONDITION_IDS,
)
from experiments.counterfactual_cot.full_backbone_lora_v3_strict_step0_step500_control.analyze_controls import (  # noqa: E402
    EXPECTED_SOURCE_SHA256,
    EXPECTED_TARGET_SHA256,
    _atomic_json,
    _blind_contact_sheet,
    _build_masks,
    _contact_sheet,
    _load_rgb,
    _mae,
    _red_stats,
    _sha256,
    _tile,
    _validate_receipts,
)

DEFAULT_OUTPUT = BAGEL_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "cf_singlepair_fullL0-27_lora_r8_promptCotOrder_strict_step0_step500_control_v1"
)
SEEDS = (1234, 2234, 3234)
A = "target_prompt_then_source_cot_v1"
B = "source_cot_then_target_prompt_v1"
C = "target_prompt_then_target_cot_v1"
METRICS = (
    "old_target_mae",
    "new_target_mae",
    "preserve_source_mae",
    "old_red_removal_progress",
    "new_red_addition_progress",
)
LOWER_IS_BETTER = {"old_target_mae", "new_target_mae", "preserve_source_mae"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generation_map(report: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    result: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in report.get("generations", []):
        key = (str(row["profile"]), int(row["seed"]))
        if key in result or float(row.get("cfg_img_scale", -1)) != 2.0:
            raise RuntimeError(f"Invalid generation row: {key}")
        result[key] = row
    expected = {(condition, seed) for condition in CONDITION_IDS for seed in SEEDS}
    if set(result) != expected:
        raise RuntimeError(f"Generation matrix mismatch: {set(result) ^ expected}")
    return result


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _condition_comparison_sheet(
    rows_by_key: Dict[Tuple[str, int], Dict[str, Any]],
    source: Path,
    target: Path,
    output: Path,
    step: int,
) -> None:
    size, label_height, columns = 256, 58, 5
    canvas = Image.new("RGB", (columns * size, len(SEEDS) * (size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    labels = ("SOURCE", "GOLD", "A: TP->SC", "B: SC->TP", "C: TP->TC")
    for row_index, seed in enumerate(SEEDS):
        paths = [source, target] + [Path(rows_by_key[(condition, seed)][f"step{step}_path"]) for condition in CONDITION_IDS]
        y = row_index * (size + label_height)
        for column, (path, label) in enumerate(zip(paths, labels)):
            canvas.paste(_tile(path, size), (column * size, y))
            draw.text((column * size + 5, y + size + 4), label, fill="black")
        draw.text((5, y + size + 27), f"step={step} seed={seed}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol_path = args.output_dir / "control_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "completed":
        raise RuntimeError(f"Protocol incomplete: {protocol.get('status')}")
    locked = protocol.get("locked_generation", {})
    if tuple(locked.get("conditions", [])) != CONDITION_IDS or tuple(locked.get("seeds", [])) != SEEDS:
        raise RuntimeError("Protocol conditions/seeds differ from the locked design")
    receipts = _validate_receipts(protocol, args.output_dir)
    reports: Dict[int, Dict[str, Any]] = {}
    maps: Dict[int, Dict[Tuple[str, int], Dict[str, Any]]] = {}
    expected_text = {condition: protocol["conditions"][condition]["text"] for condition in CONDITION_IDS}
    for step in (0, 500):
        path = args.output_dir / "generations" / f"step_{step:06d}" / "evaluation_report.json"
        if receipts[step]["report"]["sha256"] != _sha256(path):
            raise RuntimeError(f"Report changed after receipt: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "completed" or int(report.get("step", -1)) != step:
            raise RuntimeError(f"Report status/step mismatch: {path}")
        if report.get("prompt_manifest") != expected_text:
            raise RuntimeError(f"Prompt text mismatch at step {step}")
        reports[step] = report
        maps[step] = _generation_map(report)
    if reports[0].get("base_load_report") != reports[500].get("base_load_report"):
        raise RuntimeError("Base checkpoint load reports differ")
    source_path = Path(reports[0]["source_visualization"])
    target_path = Path(reports[0]["target_visualization"])
    if _sha256(source_path) != EXPECTED_SOURCE_SHA256 or _sha256(target_path) != EXPECTED_TARGET_SHA256:
        raise RuntimeError("Reference image hash mismatch")
    if _sha256(Path(reports[500]["source_visualization"])) != EXPECTED_SOURCE_SHA256 or _sha256(Path(reports[500]["target_visualization"])) != EXPECTED_TARGET_SHA256:
        raise RuntimeError("Step-500 references differ from step 0")
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
    if old_source <= old_target or new_target <= new_source:
        raise RuntimeError("Reference red direction is invalid")

    rows: List[Dict[str, Any]] = []
    rows_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for condition in CONDITION_IDS:
        for seed in SEEDS:
            row: Dict[str, Any] = {
                "profile": condition,
                "condition": condition,
                "seed": seed,
                "prompt_sha256": protocol["conditions"][condition]["sha256"],
            }
            for step in (0, 500):
                path = Path(maps[step][(condition, seed)]["path"])
                image = _load_rgb(path)
                prefix = f"step{step}_"
                row[prefix + "path"] = str(path)
                row[prefix + "sha256"] = _sha256(path)
                row[prefix + "old_target_mae"] = _mae(image, target, masks["old_edit"])
                row[prefix + "new_target_mae"] = _mae(image, target, masks["new_edit"])
                row[prefix + "preserve_source_mae"] = _mae(image, source, masks["preserve"])
                red = _red_stats(image, masks)
                row[prefix + "old_red_pixels"] = red["red_pixels_old_roi"]
                row[prefix + "new_red_pixels"] = red["red_pixels_new_roi"]
                row[prefix + "old_red_removal_progress"] = (old_source - red["red_pixels_old_roi"]) / old_reference
                row[prefix + "new_red_addition_progress"] = (red["red_pixels_new_roi"] - new_source) / new_reference
                row[prefix + "proxy_success"] = bool(
                    row[prefix + "old_red_removal_progress"] >= 0.5
                    and row[prefix + "new_red_addition_progress"] >= 0.5
                )
            for metric in METRICS:
                before, after = row[f"step0_{metric}"], row[f"step500_{metric}"]
                row[f"training_delta_{metric}"] = after - before
                row[f"training_improvement_{metric}"] = before - after if metric in LOWER_IS_BETTER else after - before
            row["old_target_mae_improvement_pct"] = 100.0 * (
                row["step0_old_target_mae"] - row["step500_old_target_mae"]
            ) / max(row["step0_old_target_mae"], 1e-12)
            row["new_target_mae_improvement_pct"] = 100.0 * (
                row["step0_new_target_mae"] - row["step500_new_target_mae"]
            ) / max(row["step0_new_target_mae"], 1e-12)
            row["preserve_source_mae_change"] = row["step500_preserve_source_mae"] - row["step0_preserve_source_mae"]
            rows.append(row)
            rows_by_key[(condition, seed)] = row

    summaries: Dict[str, Any] = {}
    for condition in CONDITION_IDS:
        selected = [row for row in rows if row["condition"] == condition]
        summaries[condition] = {
            "step0_proxy_successes": sum(bool(row["step0_proxy_success"]) for row in selected),
            "step500_proxy_successes": sum(bool(row["step500_proxy_success"]) for row in selected),
            **{f"mean_training_improvement_{metric}": _mean(selected, f"training_improvement_{metric}") for metric in METRICS},
        }

    contrasts: List[Dict[str, Any]] = []
    for contrast_name, left, right, interpretation in (
        ("B_minus_A_order_effect", A, B, "Same Target Prompt and Source CoT; only field order changes."),
        ("C_minus_A_target_cot_alignment_effect", A, C, "Target Prompt remains first; second field changes from conflicting Source CoT to aligned Target CoT."),
    ):
        for seed in SEEDS:
            item: Dict[str, Any] = {
                "contrast": contrast_name,
                "left": left,
                "right": right,
                "seed": seed,
                "interpretation": interpretation,
            }
            for metric in METRICS:
                step0_delta = rows_by_key[(right, seed)][f"step0_{metric}"] - rows_by_key[(left, seed)][f"step0_{metric}"]
                step500_delta = rows_by_key[(right, seed)][f"step500_{metric}"] - rows_by_key[(left, seed)][f"step500_{metric}"]
                raw_did = step500_delta - step0_delta
                direction = -1.0 if metric in LOWER_IS_BETTER else 1.0
                item[f"step0_{metric}_right_minus_left"] = step0_delta
                item[f"step500_{metric}_right_minus_left"] = step500_delta
                item[f"raw_difference_in_differences_{metric}"] = raw_did
                item[f"step0_improvement_oriented_{metric}_right_minus_left"] = direction * step0_delta
                item[f"step500_improvement_oriented_{metric}_right_minus_left"] = direction * step500_delta
                item[f"improvement_oriented_difference_in_differences_{metric}"] = direction * raw_did
            contrasts.append(item)

    analysis_dir = args.output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "completed",
        "run_id": protocol["run_id"],
        "created_at": _utc_now(),
        "research_questions": protocol["research_questions"],
        "condition_definitions": protocol["conditions"],
        "interpretation_limits": {
            "A_vs_B": "Strict order comparison because A and B contain identical labeled fields and differ only in order.",
            "A_vs_C": "Joint label-and-content alignment effect, not a pure CoT-content effect.",
            "C": "Both Target Prompt and Target CoT request right; success cannot identify which field caused the edit.",
            "target_prompt": "The target prompt itself contains an explicit move-only instruction, so success may reflect action-language following.",
            "automatic_metrics": "Red proxy and pixel MAE require blinded visual confirmation.",
            "contrast_sign": (
                "Raw contrasts are right-minus-left. For MAE, negative is better; for red progress, positive is better. "
                "Improvement-oriented contrasts multiply MAE differences by -1 so positive always means the right condition is better."
            ),
            "difference_in_differences": (
                "(right-minus-left at step500) minus (right-minus-left at step0); only three seeds, so this is descriptive, not a significance claim."
            ),
            "checkpoint_ancestry": protocol.get("ancestry_scope"),
        },
        "reference_red_stats": {"source": source_red, "target": target_red},
        "summaries": summaries,
        "pairs": rows,
        "contrasts": contrasts,
    }
    _atomic_json(analysis_dir / "paired_metrics.json", report)
    with (analysis_dir / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (analysis_dir / "contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)

    for condition in CONDITION_IDS:
        _contact_sheet(rows, source_path, target_path, analysis_dir / "diagnostic_conditions" / f"{condition}.jpg", condition)
    _condition_comparison_sheet(rows_by_key, source_path, target_path, analysis_dir / "diagnostic_step0_conditions.jpg", 0)
    _condition_comparison_sheet(rows_by_key, source_path, target_path, analysis_dir / "diagnostic_step500_conditions.jpg", 500)
    blind_key = _blind_contact_sheet(rows, source_path, target_path, analysis_dir / "blind_review" / "blind_ab_contact_sheet.jpg", protocol["run_id"])
    criteria = (
        "old_location_ball_absent", "new_location_single_red_ball_present", "exactly_one_ball_total",
        "sofa_and_background_preserved", "output_clear_not_blurry", "overall_edit_success",
    )
    manual = {
        "instructions": "Score A/B without opening blind_key.json or paired metrics; decode only after scores are frozen.",
        "criteria": list(criteria),
        "reviews": [
            {
                "blind_id": item["blind_id"], "condition": item["profile"], "seed": item["seed"],
                "candidate_a": {criterion: None for criterion in criteria},
                "candidate_b": {criterion: None for criterion in criteria}, "notes": "",
            }
            for item in blind_key
        ],
    }
    manual_path = analysis_dir / "blind_review" / "manual_review_template.json"
    if not manual_path.exists():
        _atomic_json(manual_path, manual)
    key_path = analysis_dir / "blind_review" / "blind_key.json"
    key_payload = {"do_not_open_before_review": True, "mapping": blind_key}
    if key_path.exists() and json.loads(key_path.read_text(encoding="utf-8")) != key_payload:
        raise RuntimeError("Existing blind key differs from the deterministic run mapping")
    if not key_path.exists():
        _atomic_json(key_path, key_payload)
    print(json.dumps({"status": "completed", "analysis_dir": str(analysis_dir), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
