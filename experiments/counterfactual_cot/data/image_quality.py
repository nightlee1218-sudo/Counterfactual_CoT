from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


DEFAULT_MIN_CHANGED_RATIO = 0.004
# An occlusion edit only overlaps two already-adjacent objects, so its footprint
# is legitimately smaller than a relocation or a category swap.
SUBTYPE_MIN_CHANGED_RATIO = {"occlusion_state": 0.0015}


def min_changed_ratio_for(subtype: str | None) -> float:
    return SUBTYPE_MIN_CHANGED_RATIO.get(str(subtype), DEFAULT_MIN_CHANGED_RATIO)


def inspect_pair_images(
    source_path: Path,
    target_path: Path,
    mask_path: Path,
    min_changed_ratio: float = DEFAULT_MIN_CHANGED_RATIO,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"passed": False, "failures": []}
    result["min_changed_ratio"] = min_changed_ratio
    for label, path in (("source", source_path), ("target", target_path), ("mask", mask_path)):
        if not path.exists():
            result["failures"].append(f"missing_{label}")
    if result["failures"]:
        return result

    try:
        with Image.open(source_path) as image:
            source = image.convert("RGB")
        with Image.open(target_path) as image:
            target = image.convert("RGB")
        with Image.open(mask_path) as image:
            mask = image.convert("L")
    except Exception as exc:
        result["failures"].append(f"unreadable_image:{type(exc).__name__}")
        return result

    result["source_size"] = list(source.size)
    result["target_size"] = list(target.size)
    if source.size != target.size:
        result["failures"].append("size_mismatch")
    if mask.size != target.size:
        result["failures"].append("mask_size_mismatch")
    if result["failures"]:
        return result

    source_array = np.asarray(source)
    target_array = np.asarray(target)
    mask_array = np.asarray(mask)
    difference = np.abs(source_array.astype(np.int16) - target_array.astype(np.int16))
    result["mean_absolute_difference"] = float(difference.mean())
    result["changed_pixel_ratio"] = float((difference.max(axis=2) > 8).mean())
    result["mask_area_ratio"] = float((mask_array > 0).mean())
    if np.array_equal(source_array, target_array):
        result["failures"].append("source_target_identical")
    # Cheap triage without a VLM: an edit that changes almost nothing is as
    # unusable as one that re-renders the whole frame.
    if result["changed_pixel_ratio"] < min_changed_ratio:
        result["failures"].append("no_visible_change")
    if result["changed_pixel_ratio"] > 0.55:
        result["failures"].append("global_rerender_suspected")
    if result["mask_area_ratio"] <= 0.0:
        result["failures"].append("mask_empty")
    if result["mask_area_ratio"] >= 1.0:
        result["failures"].append("mask_full_image")
    if not 0.001 <= result["mask_area_ratio"] <= 0.50:
        result["failures"].append("mask_area_outside_recommended_range")
    result["passed"] = not result["failures"]
    return result


def apply_local_check_to_quality(quality: Dict[str, Any], check: Dict[str, Any]) -> Dict[str, Any]:
    """Record local checks without inventing semantic TS/PS/BA/LS/IQ scores."""
    updated = dict(quality)
    updated["accepted"] = False
    updated["review_status"] = "auto_checked" if check.get("passed") else "auto_failed"
    updated["local_checks"] = check
    return updated
