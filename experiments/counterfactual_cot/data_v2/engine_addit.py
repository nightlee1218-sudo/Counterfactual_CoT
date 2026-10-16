"""Add-it reference engine wrapper for data_v2 (training-free object insertion).

Wraps NVlabs Add-it (github.com/NVlabs/addit — FLUX backbone + latent
blending + SAM2 mask refinement) as an ALTERNATIVE crop-edit engine with the
same interface as the nanobanana engine in composite_target.py:

    edit_crop(crop: PIL.Image, prompt: str, seed: int) -> PIL.Image

Engine selection:  engine = nanobanana | addit  (see build_engine_client);
pilot runs can execute BOTH engines in parallel, each producing its own
target set under eng_<name>/ for cross-engine comparison.

Real inference is environment-gated — NO model weights are downloaded here:
  * ADDIT_WEIGHTS_DIR   path to local FLUX + SAM2 checkpoint dir
  * ADDIT_CODE_DIR      path to a local checkout of the addit repo
  (both unset, or missing on disk  ->  deterministic mock engine;
   set but unusable                    ->  informative RuntimeError)

The mock engine is content-deterministic and deliberately produces a
different signature than composite_target.MockEditClient so dual-engine
selftests can verify that the two engine outputs stay distinguishable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_rng  # noqa: E402

WEIGHTS_ENV = "ADDIT_WEIGHTS_DIR"
CODE_ENV = "ADDIT_CODE_DIR"


# ---------------------------------------------------------------------------
# Mock engine
# ---------------------------------------------------------------------------

class MockAdditEditClient:
    """Offline stand-in: deterministic channel-shift edit of the crop center.

    Signature differs from composite_target.MockEditClient (which adds a
    brightness disk) so selftests can tell the two engines apart.
    """

    name = "addit"

    def edit_crop(self, crop: Image.Image, prompt: str, seed: int) -> Image.Image:
        rng = make_rng(seed)
        arr = np.asarray(crop.convert("RGB")).copy()
        h, w = arr.shape[:2]
        cy, cx = h // 2, w // 2
        r = max(2, min(h, w) // 5)
        yy, xx = np.ogrid[:h, :w]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        # deterministic per-channel shifts (hue-like change, engine signature)
        shifts = [rng.randint(-70, -20), rng.randint(20, 70), rng.randint(-40, 40)]
        for c, s in enumerate(shifts):
            arr[..., c][disk] = np.clip(arr[..., c][disk].astype(np.int32) + s, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Real engine (lazy, environment-gated)
# ---------------------------------------------------------------------------

class AdditEditClient:
    """Real Add-it engine. Requires local weights + code (never downloads)."""

    name = "addit"

    def __init__(self, weights_dir: Optional[str] = None, code_dir: Optional[str] = None,
                 device: str = "auto", num_steps: int = 28, guidance: float = 3.5):
        weights = weights_dir or os.environ.get(WEIGHTS_ENV, "")
        code = code_dir or os.environ.get(CODE_ENV, "")
        self.weights_dir = Path(weights) if weights else None
        self.code_dir = Path(code) if code else None
        self.device = device
        self.num_steps = num_steps
        self.guidance = guidance
        self._pipeline = None

        if self.weights_dir is None or not self.weights_dir.is_dir():
            raise RuntimeError(
                f"{WEIGHTS_ENV} unset or not a directory: {weights!r} — "
                "the real Add-it engine needs local FLUX/SAM2 weights "
                "(the pipeline never downloads them; unset the var to use the mock)")
        if self.code_dir is None or not self.code_dir.is_dir():
            raise RuntimeError(
                f"{CODE_ENV} unset or not a directory: {code!r} — "
                "point it at a local checkout of github.com/NVlabs/addit")

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        import importlib.util
        import sys as _sys
        # import the addit inference entrypoint from the local checkout
        entry = Path(self.code_dir) / "addit" / "pipeline_addit.py"
        if not entry.exists():  # repo layout fallback
            entry = Path(self.code_dir) / "pipeline_addit.py"
        if not entry.exists():
            raise RuntimeError(f"cannot find pipeline_addit.py under {self.code_dir}")
        spec = importlib.util.spec_from_file_location("addit_pipeline_addit", entry)
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["addit_pipeline_addit"] = mod
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        self._pipeline = mod  # module-level build/infer functions
        return self._pipeline

    def edit_crop(self, crop: Image.Image, prompt: str, seed: int) -> Image.Image:
        """Add-it edit of the crop (object add/replace inside the crop only).

        Add-it's core strength is object *addition* with latent blending; for
        data_v2 attribute/category edits we feed the crop, the edit prompt and
        the (mock-free) SAM2-refined mask back as the paste alpha source.
        """
        mod = self._load()
        import torch
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        result = mod.infer(
            image=crop.convert("RGB"),
            prompt=prompt,
            weights_dir=str(self.weights_dir),
            device=device,
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance,
            seed=seed,
        )
        if isinstance(result, Image.Image):
            return result.convert("RGB")
        if isinstance(result, (tuple, list)) and result and isinstance(result[0], Image.Image):
            return result[0].convert("RGB")
        if isinstance(result, np.ndarray):
            return Image.fromarray(result).convert("RGB")
        raise RuntimeError(f"unexpected addit infer() return type: {type(result)!r}")


# ---------------------------------------------------------------------------
# Factory (env-gated: missing weights -> mock)
# ---------------------------------------------------------------------------

def addit_real_available(weights_dir: Optional[str] = None) -> bool:
    weights = weights_dir or os.environ.get(WEIGHTS_ENV, "")
    return bool(weights) and Path(weights).is_dir()


def build_addit_client(weights_dir: Optional[str] = None, code_dir: Optional[str] = None,
                       force_real: bool = False):
    """Return the real Add-it client when weights exist, else the mock.

    force_real=True (or env DATA_V2_ENGINE_ADDIT=real) demands the real
    engine and raises instead of silently falling back to the mock.
    """
    want_real = force_real or os.environ.get("DATA_V2_ENGINE_ADDIT", "") == "real"
    if addit_real_available(weights_dir) or want_real:
        return AdditEditClient(weights_dir=weights_dir, code_dir=code_dir)
    return MockAdditEditClient()


def build_engine_client(engine: str, **kwargs: Any):
    """engine = nanobanana | addit (unified factory for run_pilot)."""
    if engine == "addit":
        return build_addit_client(**kwargs)
    if engine == "nanobanana":
        from composite_target import build_edit_client
        return build_edit_client("nanobanana", **kwargs)
    raise ValueError(f"unknown engine: {engine!r} (nanobanana|addit)")
