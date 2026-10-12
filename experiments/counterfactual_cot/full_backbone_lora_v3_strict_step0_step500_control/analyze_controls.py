#!/usr/bin/env python3
"""Analyze paired step-0/step-500 generations with transparent proxy metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

IMAGE_SIZE = 1024
OLD_ROI = (40, 590, 270, 850)
NEW_ROI = (810, 610, 1024, 860)
CANDIDATE_THRESHOLD = 24
CANDIDATE_DILATION_PX = 9
EXPECTED_SOURCE_SHA256 = "71d756241d84721b7885c68bd4da650bf4873277d48be363fb6990d74ace1539"
EXPECTED_TARGET_SHA256 = "8500c4401b5dd17ca7aa06d78bf9f3e1dbab3ceb67904c0a2903e034477f9dd1"
DEFAULT_OUTPUT = Path("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs") / (
    "cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_"
    "strict_step0_step500_control_v1"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"Expected 1024x1024 image, got {image.size}: {path}")
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _roi_mask(roi: Tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    left, top, right, bottom = roi
    mask[top:bottom, left:right] = True
    return mask


def _build_masks(source: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    difference = np.max(np.abs(source.astype(np.int16) - target.astype(np.int16)), axis=2)
    candidate = Image.fromarray((difference > CANDIDATE_THRESHOLD).astype(np.uint8) * 255, mode="L")
    candidate = candidate.filter(ImageFilter.MaxFilter(CANDIDATE_DILATION_PX))
    candidate = candidate.filter(ImageFilter.MedianFilter(3))
    candidate_mask = np.asarray(candidate, dtype=np.uint8) > 0
    old_roi = _roi_mask(OLD_ROI)
    new_roi = _roi_mask(NEW_ROI)
    old_edit = candidate_mask & old_roi
    new_edit = candidate_mask & new_roi
    if not old_edit.any() or not new_edit.any() or np.any(old_edit & new_edit):
        raise RuntimeError("Invalid reconstructed old/new edit masks")
    return {
        "old_edit": old_edit,
        "new_edit": new_edit,
        "preserve": ~(old_edit | new_edit),
        "old_roi": old_roi,
        "new_roi": new_roi,
    }


def _mae(image: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    values = np.abs(image.astype(np.float32) - reference.astype(np.float32))[mask]
    return float(values.mean() / 255.0)


def _red_mask(image: np.ndarray) -> np.ndarray:
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    return (red >= 110) & (red - green >= 35) & (red - blue >= 35) & (red >= 1.35 * green) & (red >= 1.35 * blue)


def _red_stats(image: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
    red = _red_mask(image)
    result: Dict[str, Any] = {}
    for label in ("old_roi", "new_roi"):
        selected = red & masks[label]
        ys, xs = np.nonzero(selected)
        result[f"red_pixels_{label}"] = int(selected.sum())
        result[f"red_centroid_{label}"] = [round(float(xs.mean()), 2), round(float(ys.mean()), 2)] if len(xs) else None
    return result


def _generation_map(report: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    mapped: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in report.get("generations", []):
        if float(row.get("cfg_img_scale", -1)) != 2.0:
            continue
        key = (str(row["profile"]), int(row["seed"]))
        if key in mapped:
            raise RuntimeError(f"Duplicate generation key: {key}")
        mapped[key] = row
    return mapped


def _validate_receipts(protocol: Dict[str, Any], output_dir: Path) -> Dict[int, Dict[str, Any]]:
    receipts: Dict[int, Dict[str, Any]] = {}
    run_id = protocol.get("run_id")
    if not run_id:
        raise RuntimeError("Protocol has no run_id")
    for step in (0, 500):
        recorded = protocol.get("generation_receipts", {}).get(str(step))
        if not recorded:
            raise RuntimeError(f"Protocol has no receipt for step {step}")
        receipt_path = Path(recorded["path"]).resolve()
        expected_receipt_path = (output_dir / "receipts" / f"step_{step:06d}.json").resolve()
        if receipt_path != expected_receipt_path or _sha256(receipt_path) != recorded["sha256"]:
            raise RuntimeError(f"Receipt identity mismatch at step {step}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        checkpoint = protocol["checkpoint_identities"][str(step)]
        if receipt.get("run_id") != run_id or int(receipt.get("step", -1)) != step:
            raise RuntimeError(f"Receipt run/step mismatch at step {step}")
        if receipt.get("checkpoint_sha256") != checkpoint.get("sha256"):
            raise RuntimeError(f"Receipt/checkpoint hash mismatch at step {step}")
        report_identity = receipt.get("report", {})
        report_path = Path(report_identity.get("path", "")).resolve()
        expected_report = (output_dir / "generations" / f"step_{step:06d}" / "evaluation_report.json").resolve()
        if report_path != expected_report or _sha256(report_path) != report_identity.get("sha256"):
            raise RuntimeError(f"Report receipt mismatch at step {step}")
        for identity in receipt.get("files", []):
            path = Path(identity.get("path", ""))
            if not path.is_file() or path.stat().st_size != int(identity.get("size_bytes", -1)) or _sha256(path) != identity.get("sha256"):
                raise RuntimeError(f"Generated file receipt mismatch at step {step}: {path}")
        if receipt.get("runtime_identities") != protocol.get("runtime_identities"):
            raise RuntimeError(f"Runtime identity mismatch at step {step}")
        receipts[step] = receipt
    return receipts


def _mean(rows: Iterable[Dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return float(sum(values) / len(values))


def _tile(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        tile = image.convert("RGB")
        tile.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(tile, ((size - tile.width) // 2, (size - tile.height) // 2))
    return canvas


def _contact_sheet(
    rows: List[Dict[str, Any]],
    source_path: Path,
    target_path: Path,
    output: Path,
    profile: str | None = None,
) -> None:
    selected = [row for row in rows if profile is None or row["profile"] == profile]
    if not selected:
        return
    size, label_height, columns = 256, 82, 4
    canvas = Image.new("RGB", (columns * size, len(selected) * (size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(selected):
        y = index * (size + label_height)
        paths = (source_path, target_path, Path(row["step0_path"]), Path(row["step500_path"]))
        headers = ("SOURCE", "GOLD TARGET", "STEP 0", "STEP 500")
        for column, (path, header) in enumerate(zip(paths, headers)):
            canvas.paste(_tile(path, size), (column * size, y))
            draw.text((column * size + 6, y + size + 4), header, fill="black")
        label = (
            f"{row['profile']} seed={row['seed']} | proxy {row['step0_proxy_success']} -> "
            f"{row['step500_proxy_success']} | target edit MAE improvement: "
            f"old={row['old_target_mae_improvement_pct']:.1f}% new={row['new_target_mae_improvement_pct']:.1f}%"
        )
        draw.text((6, y + size + 27), label, fill="black")
        draw.text(
            (6, y + size + 50),
            f"preserve source MAE: {row['step0_preserve_source_mae']:.4f} -> {row['step500_preserve_source_mae']:.4f}",
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def _blind_contact_sheet(
    rows: List[Dict[str, Any]],
    source_path: Path,
    target_path: Path,
    output: Path,
    run_id: str,
) -> List[Dict[str, Any]]:
    size, label_height, columns = 256, 58, 4
    canvas = Image.new("RGB", (columns * size, len(rows) * (size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    key: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        blind_id = f"B{index + 1:03d}"
        swap = int(hashlib.sha256(f"{run_id}|{row['profile']}|{row['seed']}".encode()).hexdigest(), 16) % 2 == 1
        candidates = [("step500", Path(row["step500_path"])), ("step0", Path(row["step0_path"]))] if swap else [
            ("step0", Path(row["step0_path"])), ("step500", Path(row["step500_path"]))
        ]
        y = index * (size + label_height)
        paths = (source_path, target_path, candidates[0][1], candidates[1][1])
        headers = ("SOURCE", "GOLD TARGET", "CANDIDATE A", "CANDIDATE B")
        for column, (path, header) in enumerate(zip(paths, headers)):
            canvas.paste(_tile(path, size), (column * size, y))
            draw.text((column * size + 6, y + size + 4), header, fill="black")
        draw.text((6, y + size + 28), f"{blind_id} | profile={row['profile']} | seed={row['seed']}", fill="black")
        key.append({
            "blind_id": blind_id,
            "profile": row["profile"],
            "seed": row["seed"],
            "candidate_a": candidates[0][0],
            "candidate_b": candidates[1][0],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)
    return key


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    protocol_path = args.output_dir / "control_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Run generate_controls.py first: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "completed":
        raise RuntimeError(f"Generation protocol is incomplete: {protocol.get('status')}")
    receipts = _validate_receipts(protocol, args.output_dir)
    locked = protocol.get("locked_generation", {})
    expected_profiles = (
        "cot_core_v1", "legacy_paired_v1", "full_paired_v2", "full_without_target_cot_v1",
        "target_prompt_only_v1", "delta_only_v2",
    )
    expected_seeds = (1234, 2234, 3234)
    if tuple(locked.get("profiles", [])) != expected_profiles or tuple(locked.get("seeds", [])) != expected_seeds:
        raise RuntimeError("Protocol profile/seed matrix is not the locked strict design")
    reports: Dict[int, Dict[str, Any]] = {}
    for step in (0, 500):
        path = args.output_dir / "generations" / f"step_{step:06d}" / "evaluation_report.json"
        reports[step] = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = protocol["checkpoint_identities"][str(step)]
        if reports[step].get("status") != "completed" or int(reports[step].get("step", -1)) != step:
            raise RuntimeError(f"Report status/step mismatch: {path}")
        if Path(reports[step].get("checkpoint", "")).resolve() != Path(checkpoint["path"]):
            raise RuntimeError(f"Report checkpoint path mismatch: {path}")
        if receipts[step]["report"]["sha256"] != _sha256(path):
            raise RuntimeError(f"Report changed after receipt creation: {path}")
    if reports[0].get("prompt_manifest") != reports[500].get("prompt_manifest"):
        raise RuntimeError("Prompt text differs between step 0 and step 500")
    if reports[0].get("base_load_report") != reports[500].get("base_load_report"):
        raise RuntimeError("Base checkpoint load reports differ between step 0 and step 500")
    step0 = _generation_map(reports[0])
    step500 = _generation_map(reports[500])
    if set(step0) != set(step500):
        raise RuntimeError(f"Generation matrix mismatch: only0={set(step0)-set(step500)}, only500={set(step500)-set(step0)}")
    source_path = Path(reports[0]["source_visualization"])
    target_path = Path(reports[0]["target_visualization"])
    source = _load_rgb(source_path)
    target = _load_rgb(target_path)
    if _sha256(source_path) != EXPECTED_SOURCE_SHA256 or _sha256(target_path) != EXPECTED_TARGET_SHA256:
        raise RuntimeError("Source or target visualization hash mismatch")
    if _sha256(Path(reports[500]["source_visualization"])) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Step-500 source differs from step-0 source")
    if _sha256(Path(reports[500]["target_visualization"])) != EXPECTED_TARGET_SHA256:
        raise RuntimeError("Step-500 target differs from step-0 target")
    masks = _build_masks(source, target)
    source_red = _red_stats(source, masks)
    target_red = _red_stats(target, masks)
    old_source_red = int(source_red["red_pixels_old_roi"])
    old_target_red = int(target_red["red_pixels_old_roi"])
    new_source_red = int(source_red["red_pixels_new_roi"])
    new_target_red = int(target_red["red_pixels_new_roi"])
    old_removal_reference = max(1, old_source_red - old_target_red)
    new_addition_reference = max(1, new_target_red - new_source_red)
    if old_source_red <= old_target_red or new_target_red <= new_source_red:
        raise RuntimeError("Reference red-pixel direction does not match left-to-right movement")
    rows: List[Dict[str, Any]] = []
    for profile, seed in sorted(step0):
        paths = {0: Path(step0[(profile, seed)]["path"]), 500: Path(step500[(profile, seed)]["path"])}
        images = {step: _load_rgb(path) for step, path in paths.items()}
        row: Dict[str, Any] = {
            "profile": profile,
            "seed": seed,
            "step0_path": str(paths[0]),
            "step500_path": str(paths[500]),
            "step0_sha256": _sha256(paths[0]),
            "step500_sha256": _sha256(paths[500]),
        }
        for step in (0, 500):
            image = images[step]
            prefix = f"step{step}_"
            row[prefix + "old_target_mae"] = _mae(image, target, masks["old_edit"])
            row[prefix + "new_target_mae"] = _mae(image, target, masks["new_edit"])
            row[prefix + "preserve_target_mae"] = _mae(image, target, masks["preserve"])
            row[prefix + "preserve_source_mae"] = _mae(image, source, masks["preserve"])
            red = _red_stats(image, masks)
            row[prefix + "old_red_pixels"] = red["red_pixels_old_roi"]
            row[prefix + "new_red_pixels"] = red["red_pixels_new_roi"]
            row[prefix + "old_red_removal_progress"] = (
                old_source_red - red["red_pixels_old_roi"]
            ) / old_removal_reference
            row[prefix + "new_red_addition_progress"] = (
                red["red_pixels_new_roi"] - new_source_red
            ) / new_addition_reference
            row[prefix + "proxy_success"] = bool(
                row[prefix + "old_red_removal_progress"] >= 0.50
                and row[prefix + "new_red_addition_progress"] >= 0.50
            )
        for region in ("old", "new"):
            before = row[f"step0_{region}_target_mae"]
            after = row[f"step500_{region}_target_mae"]
            row[f"{region}_target_mae_improvement_pct"] = 100.0 * (before - after) / max(before, 1e-12)
        row["preserve_source_mae_change"] = row["step500_preserve_source_mae"] - row["step0_preserve_source_mae"]
        rows.append(row)
    summaries: Dict[str, Any] = {}
    for profile in sorted({row["profile"] for row in rows}):
        current = [row for row in rows if row["profile"] == profile]
        summaries[profile] = {
            "num_pairs": len(current),
            "step0_proxy_successes": sum(bool(row["step0_proxy_success"]) for row in current),
            "step500_proxy_successes": sum(bool(row["step500_proxy_success"]) for row in current),
            "mean_old_target_mae_improvement_pct": _mean(current, "old_target_mae_improvement_pct"),
            "mean_new_target_mae_improvement_pct": _mean(current, "new_target_mae_improvement_pct"),
            "mean_step0_preserve_source_mae": _mean(current, "step0_preserve_source_mae"),
            "mean_step500_preserve_source_mae": _mean(current, "step500_preserve_source_mae"),
            "mean_preserve_source_mae_change": _mean(current, "preserve_source_mae_change"),
        }
    analysis_dir = args.output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "completed",
        "run_id": protocol["run_id"],
        "created_at": _utc_now(),
        "checkpoint_sha256": {
            step: protocol["checkpoint_identities"][step]["sha256"] for step in ("0", "500")
        },
        "interpretation_scope": {
            "step0": "same v3 run before optimizer updates; close to zero-update behavior, not independently loaded pure base",
            "step500": "same v3 run after 500 optimizer updates",
            "paired_claim": "differences are attributable to the two checkpoint states under hash-locked generation settings",
            "ancestry_limit": "legacy checkpoints lack a run UUID; same-run ancestry is supported by experiment/schema/history/LoRA-state/timestamp consistency, not cryptographically proven",
            "not_proven": "a red-pixel proxy success is not proof of semantic correctness; blinded manual review remains primary",
        },
        "metric_definitions": {
            "old_new_target_mae": "RGB MAE to golden target on deterministic changed-pixel masks inside audited old/new ROIs",
            "preserve_source_mae": "RGB MAE to source outside old/new changed-pixel masks; lower is better preservation",
            "red_proxy": "R>=110, R-G>=35, R-B>=35, R>=1.35G/B inside audited ROIs",
            "red_progress": "source=0 and golden target=1 for each of old-region removal and new-region addition; values may fall outside [0,1]",
            "proxy_success": "both old red-removal progress and new red-addition progress reach at least 0.50",
            "warning": "Pixel MAE is sensitive to small spatial shifts and must be interpreted with contact sheets/manual review",
        },
        "reference_red_stats": {"source": source_red, "target": target_red},
        "profiles": summaries,
        "pairs": rows,
    }
    _atomic_json(analysis_dir / "paired_metrics.json", report)
    fieldnames = list(rows[0])
    with (analysis_dir / "paired_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    blind_key = _blind_contact_sheet(
        rows,
        source_path,
        target_path,
        analysis_dir / "blind_review" / "blind_ab_contact_sheet.jpg",
        protocol["run_id"],
    )
    criteria = (
        "old_location_ball_absent", "new_location_single_red_ball_present", "exactly_one_ball_total",
        "sofa_and_background_preserved", "output_clear_not_blurry", "overall_edit_success",
    )
    manual = {
        "instructions": (
            "Use blind_ab_contact_sheet.jpg without opening blind_key.json or paired metrics. "
            "Score candidates A and B independently; only decode checkpoint identity after all entries are frozen."
        ),
        "criteria": list(criteria),
        "reviews": [
            {
                "blind_id": item["blind_id"],
                "profile": item["profile"],
                "seed": item["seed"],
                "candidate_a": {criterion: None for criterion in criteria},
                "candidate_b": {criterion: None for criterion in criteria},
                "notes": "",
            }
            for item in blind_key
        ],
    }
    manual_path = analysis_dir / "blind_review" / "manual_review_template.json"
    if not manual_path.exists():
        _atomic_json(manual_path, manual)
    _atomic_json(analysis_dir / "blind_review" / "blind_key.json", {"do_not_open_before_review": True, "mapping": blind_key})
    for profile in summaries:
        _contact_sheet(rows, source_path, target_path, analysis_dir / "diagnostic_contact_sheets" / f"{profile}.jpg", profile)
    _contact_sheet(rows, source_path, target_path, analysis_dir / "diagnostic_all_profiles_contact_sheet.jpg")
    print(json.dumps({"status": "completed", "analysis_dir": str(analysis_dir), "profiles": summaries}, indent=2))


if __name__ == "__main__":
    main()
