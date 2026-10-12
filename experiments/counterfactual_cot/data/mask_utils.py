from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageFilter


def create_candidate_mask(
    source_path: Path,
    target_path: Path,
    output_path: Path,
    threshold: int = 24,
    dilation_px: int = 9,
    min_area_ratio: float = 0.001,
    max_area_ratio: float = 0.50,
) -> Dict[str, Any]:
    """Create a review-only pixel-difference mask.

    The mask is deliberately called a candidate mask: generative editors can
    introduce small global changes, so it must not be treated as gold spatial
    supervision without human review.
    """
    with Image.open(source_path) as source_image:
        source = source_image.convert("RGB")
    with Image.open(target_path) as target_image:
        target = target_image.convert("RGB")
    if source.size != target.size:
        raise ValueError(f"Source/target size mismatch: {source.size} != {target.size}")

    source_array = np.asarray(source, dtype=np.int16)
    target_array = np.asarray(target, dtype=np.int16)
    difference = np.max(np.abs(source_array - target_array), axis=2)
    binary = (difference > int(threshold)).astype(np.uint8) * 255
    mask = Image.fromarray(binary, mode="L")
    if dilation_px > 1:
        kernel = int(dilation_px)
        if kernel % 2 == 0:
            kernel += 1
        mask = mask.filter(ImageFilter.MaxFilter(kernel))
        mask = mask.filter(ImageFilter.MedianFilter(3))

    mask_array = np.asarray(mask) > 0
    area_ratio = float(mask_array.mean())
    if area_ratio <= 0.0:
        raise ValueError("Candidate mask is empty; source and target are effectively identical")
    if area_ratio >= 1.0:
        raise ValueError("Candidate mask covers the whole image")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output_path, format="PNG")
    return {
        "area_ratio": area_ratio,
        "within_recommended_range": min_area_ratio <= area_ratio <= max_area_ratio,
        "threshold": int(threshold),
        "dilation_px": int(dilation_px),
        "candidate_only": True,
    }
