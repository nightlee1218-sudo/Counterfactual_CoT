"""Audited old/new/preserve target-flow objective for the locked C1 pair."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

IMAGE_SIZE = 1024
TOKEN_GRID = (64, 64)
EXPECTED_TARGET_TOKENS = 4096
OLD_ROI = (40, 590, 270, 850)
NEW_ROI = (810, 610, 1024, 860)
CANDIDATE_THRESHOLD = 24
CANDIDATE_DILATION_PX = 9
SOURCE_SHA256 = "71d756241d84721b7885c68bd4da650bf4873277d48be363fb6990d74ace1539"
TARGET_SHA256 = "8500c4401b5dd17ca7aa06d78bf9f3e1dbab3ceb67904c0a2903e034477f9dd1"
MASK_SHA256 = "1e3cc4e4c0f1cc45b6c2052e674104711838490bac1b7b0a86cc333bff677887"
LOSS_WEIGHTS = {"old": 0.25, "new": 0.25, "preserve": 0.50}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_region_objective(
    source_path: Path,
    target_path: Path,
    mask_path: Path,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    expected_hashes = {
        "source": SOURCE_SHA256,
        "target": TARGET_SHA256,
        "mask": MASK_SHA256,
    }
    paths = {"source": source_path, "target": target_path, "mask": mask_path}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing locked {label} asset: {path}")
        actual = _sha256(path)
        if actual != expected_hashes[label]:
            raise RuntimeError(f"Locked {label} SHA256 mismatch: {actual} != {expected_hashes[label]}")

    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.int16)
        source_size = image.size
    with Image.open(target_path) as image:
        target = np.asarray(image.convert("RGB"), dtype=np.int16)
        target_size = image.size
    with Image.open(mask_path) as image:
        if image.mode != "L":
            raise RuntimeError(f"Candidate mask must be L mode, got {image.mode!r}")
        candidate = np.asarray(image, dtype=np.uint8)
        mask_size = image.size
    if source_size != (IMAGE_SIZE, IMAGE_SIZE) or target_size != source_size or mask_size != source_size:
        raise RuntimeError(f"Expected aligned 1024 assets, got source={source_size}, target={target_size}, mask={mask_size}")

    expected_binary = (np.max(np.abs(source - target), axis=2) > CANDIDATE_THRESHOLD).astype(np.uint8) * 255
    reconstructed = Image.fromarray(expected_binary, mode="L")
    reconstructed = reconstructed.filter(ImageFilter.MaxFilter(CANDIDATE_DILATION_PX))
    reconstructed = reconstructed.filter(ImageFilter.MedianFilter(3))
    if not np.array_equal(candidate, np.asarray(reconstructed, dtype=np.uint8)):
        raise RuntimeError("Candidate mask no longer matches deterministic threshold/dilation reconstruction")

    candidate_binary = candidate > 0
    old_pixels = np.zeros_like(candidate_binary)
    new_pixels = np.zeros_like(candidate_binary)
    left, top, right, bottom = OLD_ROI
    old_pixels[top:bottom, left:right] = candidate_binary[top:bottom, left:right]
    left, top, right, bottom = NEW_ROI
    new_pixels[top:bottom, left:right] = candidate_binary[top:bottom, left:right]
    if np.any(old_pixels & new_pixels):
        raise RuntimeError("Old and new edit masks overlap")
    if not old_pixels.any() or not new_pixels.any():
        raise RuntimeError("Both old and new edit masks must be non-empty")

    old = F.adaptive_avg_pool2d(torch.from_numpy(old_pixels.astype(np.float32))[None, None], TOKEN_GRID).reshape(-1)
    new = F.adaptive_avg_pool2d(torch.from_numpy(new_pixels.astype(np.float32))[None, None], TOKEN_GRID).reshape(-1)
    preserve = 1.0 - old - new
    weights = {"old": old.contiguous(), "new": new.contiguous(), "preserve": preserve.contiguous()}
    for label, tensor in weights.items():
        if tensor.numel() != EXPECTED_TARGET_TOKENS or not torch.isfinite(tensor).all():
            raise RuntimeError(f"Invalid {label} region token weights")
        if float(tensor.min()) < -1e-7 or float(tensor.max()) > 1.0 + 1e-7 or float(tensor.sum()) <= 0:
            raise RuntimeError(f"Out-of-range or empty {label} region token weights")
    partition_error = float(torch.max(torch.abs(old + new + preserve - 1.0)))
    if partition_error > 1e-6:
        raise RuntimeError(f"Region token weights do not partition unity: {partition_error}")

    masses = {label: float(tensor.sum()) for label, tensor in weights.items()}
    spec = {
        "objective": "old_new_preserve_target_flow_mse",
        "formula": "0.25*old_mse + 0.25*new_mse + 0.50*preserve_mse",
        "loss_weights": LOSS_WEIGHTS,
        "preserve_target": "gold_target_flow_velocity_not_source_latent",
        "source_path": str(source_path.resolve()),
        "target_path": str(target_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "sha256": expected_hashes,
        "candidate_mask_reconstruction": {
            "threshold": CANDIDATE_THRESHOLD,
            "dilation_px": CANDIDATE_DILATION_PX,
            "median_filter_px": 3,
            "pixel_exact_match": True,
        },
        "old_roi_ltrb": list(OLD_ROI),
        "new_roi_ltrb": list(NEW_ROI),
        "old_pixels": int(old_pixels.sum()),
        "new_pixels": int(new_pixels.sum()),
        "preserve_pixels": int(IMAGE_SIZE * IMAGE_SIZE - old_pixels.sum() - new_pixels.sum()),
        "token_grid_hw": list(TOKEN_GRID),
        "expected_target_tokens": EXPECTED_TARGET_TOKENS,
        "token_masses": masses,
        "token_mass_ratios": {label: mass / EXPECTED_TARGET_TOKENS for label, mass in masses.items()},
        "partition_max_error": partition_error,
    }
    return spec, weights


def region_forward_loss(
    model: torch.nn.Module,
    data: Dict[str, Any],
    weights_cpu: Dict[str, torch.Tensor],
    autocast_context: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    with autocast_context(model):
        output = model(**data)
    mse = output.get("mse")
    if mse is None:
        raise RuntimeError("Model returned no diffusion MSE")
    if mse.ndim != 2 or tuple(mse.shape) != (EXPECTED_TARGET_TOKENS, 64):
        raise RuntimeError(f"Expected target-flow MSE [{EXPECTED_TARGET_TOKENS}, 64], got {list(mse.shape)}")
    per_token = mse.float().mean(dim=-1)
    region_mses: Dict[str, torch.Tensor] = {}
    device_weights: Dict[str, torch.Tensor] = {}
    for label, weights in weights_cpu.items():
        current = weights.to(device=per_token.device, dtype=per_token.dtype)
        device_weights[label] = current
        region_mses[label] = (per_token * current).sum() / current.sum()
    native = per_token.mean()
    recomposed = sum(
        (device_weights[label].sum() / EXPECTED_TARGET_TOKENS) * region_mses[label]
        for label in ("old", "new", "preserve")
    )
    recomposition_error = torch.abs(recomposed - native)
    tolerance = 2e-6 * torch.maximum(native.detach().abs(), torch.ones_like(native))
    if recomposition_error.detach() > tolerance:
        raise RuntimeError(f"Region partition failed to recompose native MSE: {float(recomposition_error)}")
    region_loss = sum(LOSS_WEIGHTS[label] * region_mses[label] for label in ("old", "new", "preserve"))
    ce = output.get("ce")
    ce_loss = ce.float().mean() if ce is not None else torch.zeros_like(region_loss)
    total = region_loss + ce_loss
    scalars = {
        "total": float(total.detach().cpu()),
        "region_mse": float(region_loss.detach().cpu()),
        "native_mse": float(native.detach().cpu()),
        "old_mse": float(region_mses["old"].detach().cpu()),
        "new_mse": float(region_mses["new"].detach().cpu()),
        "preserve_mse": float(region_mses["preserve"].detach().cpu()),
        "weighted_old_mse": float((LOSS_WEIGHTS["old"] * region_mses["old"]).detach().cpu()),
        "weighted_new_mse": float((LOSS_WEIGHTS["new"] * region_mses["new"]).detach().cpu()),
        "weighted_preserve_mse": float((LOSS_WEIGHTS["preserve"] * region_mses["preserve"]).detach().cpu()),
        "native_recomposition_error": float(recomposition_error.detach().cpu()),
        "ce": float(ce_loss.detach().cpu()),
    }
    return total, scalars
