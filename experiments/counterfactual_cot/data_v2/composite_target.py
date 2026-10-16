"""data_v2 target compositing — background is pixel-identical to source.

Pipeline per pair (mode=edit):
    1. read source image + object bbox from the plan
    2. crop the object with ~10% context margin (clamped to image bounds)
    3. edit ONLY the crop via the nanobanana image-edit API
       (prompt keeps lighting/viewpoint; key from env NANOBANANA_KEY;
        retries with backoff; on repeated failure retry with a new seed)
    4. matte the new object: SAM2 / rembg alpha, unioned with the crop diff
    5. paste back onto the ORIGINAL source pixel grid with
       cv2.seamlessClone (Poisson, mask = matte boundary dilated);
       pixels outside the pasted mask are forced byte-identical to source

mode=move (relative_position):
    matte object A from the source, cv2.inpaint its old location, paste A at
    the new position relative to object B (left_of / on_top_of), Poisson-blend.

No real API call happens under the default mock provider; the nanobanana
provider is only enabled explicitly (run_pilot.sh gates it behind
DATA_V2_PROVIDER=nanobanana + NANOBANANA_KEY).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    append_jsonl, atomic_write_bytes, clamp_bbox, context_crop_bbox, dilate_mask,
    load_jsonl, make_rng, normalize_bbox, write_jsonl,
)

Array = np.ndarray

# ---------------------------------------------------------------------------
# Edit clients (step 3)
# ---------------------------------------------------------------------------

class EditClient(ABC):
    """Abstract crop editor: crop PIL image + prompt -> edited crop PIL image."""

    @abstractmethod
    def edit_crop(self, crop: Image.Image, prompt: str, seed: int) -> Image.Image: ...


class MockEditClient(EditClient):
    """Deterministic offline editor for selftests: recolors the crop's center."""

    def edit_crop(self, crop: Image.Image, prompt: str, seed: int) -> Image.Image:
        rng = make_rng(seed)
        arr = np.asarray(crop.convert("RGB")).copy()
        h, w = arr.shape[:2]
        cy, cx = h // 2, w // 2
        r = max(1, min(h, w) // 6)
        yy, xx = np.ogrid[:h, :w]
        disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        shift = rng.randint(28, 80)
        arr[disk] = np.clip(arr[disk].astype(np.int32) + shift, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class NanoBananaEditClient(EditClient):
    """nanobanana crop editor. API key: env NANOBANANA_KEY (never hardcoded).

    ``transport`` is injectable for tests (callable(payload_dict) -> response_dict).
    On transient failure the request is retried with backoff; when retries are
    exhausted the seed is changed once and the whole cycle repeats.
    """

    def __init__(self, endpoint: str = "http://v2.open.venus.oa.com/llmproxy/chat/completions",
                 model: str = "gemini-3-pro-image",
                 api_key_env: str = "NANOBANANA_KEY",
                 timeout_s: int = 180, max_retries: int = 4,
                 retry_backoff_s: float = 3.0,
                 transport: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None):
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key and transport is None:
            raise ValueError(f"Missing API key in environment variable {api_key_env}")
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self._transport = transport
        self.seeds_used: List[int] = []

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._transport is not None:
            return self._transport(payload)
        import requests
        resp = requests.post(
            self.endpoint,
            json=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_image(response: Dict[str, Any]) -> bytes:
        import base64
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"empty choices in edit response: {str(response)[:200]}")
        message = choices[0].get("message") or {}
        images = message.get("images") or []
        if not images:
            content = message.get("content")
            if isinstance(content, list):
                images = [p.get("image_url", {}).get("url", "") for p in content
                          if isinstance(p, dict) and p.get("type") == "image_url"]
        if not images:
            raise RuntimeError("no image in edit response")
        value = images[0]
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        return base64.b64decode(value)

    def edit_crop(self, crop: Image.Image, prompt: str, seed: int) -> Image.Image:
        import base64
        import io
        buf = io.BytesIO()
        crop.convert("RGB").save(buf, format="PNG")
        data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        seeds = [seed, seed + 10_000]  # one seed change on hard failure
        self.seeds_used = list(seeds)
        last_error: Optional[Exception] = None
        for attempt_seed in seeds:
            payload = {
                "model": self.model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": f"{prompt}\n[seed={attempt_seed}]"},
                    ],
                }],
                "seed": attempt_seed,
                "modalities": ["image", "text"],
            }
            for attempt in range(self.max_retries):
                try:
                    raw = self._extract_image(self._post(payload))
                    return Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception as exc:  # noqa: BLE001 - retry any transport/API error
                    last_error = exc
                    time.sleep(min(self.retry_backoff_s * (2 ** attempt), 60))
        raise RuntimeError(f"nanobanana edit failed after retries (seeds={seeds}): {last_error}")


def build_edit_client(provider: str, **kwargs: Any) -> EditClient:
    if provider == "mock":
        return MockEditClient()
    if provider == "nanobanana":
        return NanoBananaEditClient(**kwargs)
    if provider == "addit":
        # Add-it reference engine (env-gated real, mock fallback); duck-typed
        from engine_addit import build_addit_client
        client = build_addit_client(**kwargs)
        return client  # type: ignore[return-value]
    raise ValueError(f"unknown edit provider: {provider!r} (mock|nanobanana|addit)")


# ---------------------------------------------------------------------------
# Matting backends (step 4)
# ---------------------------------------------------------------------------

class MattingBackend(ABC):
    """Abstract matte extractor: object RGB crop -> uint8 alpha (H, W)."""

    @abstractmethod
    def extract_alpha(self, crop: Image.Image) -> Array: ...


class MockMatting(MattingBackend):
    """Center-disk alpha for selftests."""

    def extract_alpha(self, crop: Image.Image) -> Array:
        arr = np.asarray(crop.convert("RGB"))
        h, w = arr.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        r = max(1, min(h, w) // 4)
        return (((yy - h // 2) ** 2 + (xx - w // 2) ** 2) <= r * r).astype(np.uint8) * 255


class RembgMatting(MattingBackend):
    """rembg general model (lazy import; returns 255 inside the salient object)."""

    def __init__(self) -> None:
        from rembg import new_session  # noqa: F401  (import check)
        self._session = None

    def extract_alpha(self, crop: Image.Image) -> Array:
        from rembg import new_session, remove
        if self._session is None:
            self._session = new_session("u2net")
        out = remove(crop.convert("RGBA"), session=self._session)
        alpha = np.asarray(out)[..., 3]
        return alpha


class Sam2Matting(MattingBackend):
    """SAM2 point-prompt matte at the crop center (lazy import)."""

    def __init__(self, checkpoint: Optional[str] = None) -> None:
        self.checkpoint = checkpoint

    def extract_alpha(self, crop: Image.Image) -> Array:
        try:
            from sam2.build_sam import build_sam2_hiera
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError("sam2 not installed; use rembg or grabcut") from exc
        model = build_sam2_hiera("tiny", ckpt_path=self.checkpoint)
        predictor = SAM2ImagePredictor(model)
        predictor.set_image(np.asarray(crop.convert("RGB")))
        h, w = crop.height, crop.width
        masks, scores, _ = predictor.predict(
            point_coords=np.array([[w // 2, h // 2]]),
            point_labels=np.array([1]), multimask_output=False)
        best = int(np.argmax(scores))
        return (masks[best].astype(np.uint8)) * 255


class GrabCutMatting(MattingBackend):
    """cv2 GrabCut fallback with a generous center rect (no extra deps)."""

    def extract_alpha(self, crop: Image.Image) -> Array:
        import cv2
        arr = np.asarray(crop.convert("RGB"))
        h, w = arr.shape[:2]
        rect = (w // 10, h // 10, w * 8 // 10, h * 8 // 10)
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        mask = np.zeros((h, w), np.uint8)
        cv2.grabCut(arr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def build_matting(name: str) -> MattingBackend:
    if name == "mock":
        return MockMatting()
    if name == "rembg":
        return RembgMatting()
    if name == "sam2":
        return Sam2Matting()
    if name == "grabcut":
        return GrabCutMatting()
    if name == "auto":  # first available real backend
        for cls in (RembgMatting, Sam2Matting, GrabCutMatting):
            try:
                return cls()
            except Exception:  # noqa: BLE001
                continue
        return GrabCutMatting()
    raise ValueError(f"unknown matting backend: {name!r}")


# ---------------------------------------------------------------------------
# Compositing core (step 5)
# ---------------------------------------------------------------------------

def diff_mask(a: Array, b: Array, tol: int = 10) -> Array:
    """Boolean mask of pixels differing more than tol (0..255) in any channel."""
    return np.any(np.abs(a.astype(np.int16) - b.astype(np.int16)) > tol, axis=-1)


def composite_edit(source_rgb: Array, bbox: Tuple[int, int, int, int],
                   edited_crop: Array, alpha: Array,
                   context_frac: float = 0.10, dilate_px: int = 9
                   ) -> Tuple[Array, Array]:
    """Paste an edited crop back; returns (target_rgb, paste_mask bool).

    paste_mask marks the ONLY region allowed to differ from the source.
    Outside it the output is byte-identical to the source (enforced after
    seamlessClone, which only blends inside the mask anyway).
    """
    import cv2
    h, w = source_rgb.shape[:2]
    x1, y1, x2, y2 = context_crop_bbox(bbox, w, h, context_frac)
    ch, cw = y2 - y1, x2 - x1
    if edited_crop.shape[0] != ch or edited_crop.shape[1] != cw:
        edited_crop = cv2.resize(edited_crop, (cw, ch), interpolation=cv2.INTER_LANCZOS4)
    if alpha.shape[0] != ch or alpha.shape[1] != cw:
        alpha = cv2.resize(alpha, (cw, ch), interpolation=cv2.INTER_NEAREST)

    orig_crop = source_rgb[y1:y2, x1:x2]
    changed = diff_mask(orig_crop, edited_crop) | (alpha > 127)
    paste_mask_small = dilate_mask(changed, ksize=max(3, dilate_px))

    paste_mask = np.zeros((h, w), bool)
    paste_mask[y1:y2, x1:x2] = paste_mask_small
    if not paste_mask.any():
        return source_rgb.copy(), paste_mask

    edited_full = source_rgb.copy()
    edited_full[y1:y2, x1:x2] = edited_crop

    # seamlessClone wants an inclusive interior mask with a margin from the border
    clone_mask = paste_mask.astype(np.uint8) * 255
    ys, xs = np.nonzero(paste_mask)
    cx, cy = int(xs.mean()), int(ys.mean())
    if (cx <= 0 or cy <= 0 or cx >= w - 1 or cy >= h - 1
            or not clone_mask[cy - 1:cy + 2, cx - 1:cx + 2].any()):
        target = edited_full  # mask touches the border or degenerate center: direct paste
    else:
        try:
            target = cv2.seamlessClone(edited_full, source_rgb, clone_mask,
                                       (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:  # border-touching / degenerate mask: direct paste
            target = edited_full

    # red-line guarantee: outside paste_mask the pixels ARE the source pixels
    target = target.copy()
    target[~paste_mask] = source_rgb[~paste_mask]
    return target, paste_mask


def composite_move(source_rgb: Array, bbox_a: Tuple[int, int, int, int],
                   bbox_b: Tuple[int, int, int, int], kind: str,
                   matting: MattingBackend, dilate_px: int = 9,
                   pad: int = 6
                   ) -> Tuple[Array, Array]:
    """Move object A relative to B: matte A, inpaint hole, paste at new spot."""
    import cv2
    h, w = source_rgb.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox_a, w, h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return source_rgb.copy(), np.zeros((h, w), bool)

    crop = Image.fromarray(source_rgb[y1:y2, x1:x2])
    alpha = matting.extract_alpha(crop)
    obj_mask_small = alpha > 127

    # object pixels ( RGBA ) lifted from the source
    obj = source_rgb[y1:y2, x1:x2].copy()

    # new bbox for A relative to B
    bw = x2 - x1
    bh = y2 - y1
    bx1, by1, bx2, by2 = clamp_bbox(bbox_b, w, h)
    if kind == "left_of":
        nx2 = max(bx1 - pad, bw)          # right edge of A just left of B
        nx1 = nx2 - bw
        ny1 = (by1 + by2) // 2 - bh // 2  # vertically centered on B
        ny2 = ny1 + bh
    elif kind == "on_top_of":
        ny2 = min(by1 + max(4, (by2 - by1) // 3), h)  # A resting on top of B
        ny1 = max(0, ny2 - bh)
        nx1 = (bx1 + bx2) // 2 - bw // 2
        nx2 = nx1 + bw
    else:
        raise ValueError(f"unknown relative_position kind: {kind!r}")
    nx1, ny1, nx2, ny2 = clamp_bbox((nx1, ny1, nx2, ny2), w, h)
    if nx2 - nx1 < 2 or ny2 - ny1 < 2:
        return source_rgb.copy(), np.zeros((h, w), bool)

    # 1) erase A at the old location (inpaint), 2) paste A at the new location
    work = source_rgb.copy()
    hole = np.zeros((h, w), np.uint8)
    hole[y1:y2, x1:x2][obj_mask_small] = 255
    hole = (dilate_mask(hole > 0, ksize=max(3, dilate_px))).astype(np.uint8) * 255
    if hole.any():
        work = cv2.inpaint(work, hole, 5, cv2.INPAINT_TELEA)

    # paste region: resized object centered in the new bbox
    obj_resized = cv2.resize(obj, (nx2 - nx1, ny2 - ny1), interpolation=cv2.INTER_LANCZOS4)
    alpha_resized = cv2.resize(alpha, (nx2 - nx1, ny2 - ny1), interpolation=cv2.INTER_NEAREST)
    paste_mask = np.zeros((h, w), bool)
    new_small = alpha_resized > 127
    if not new_small.any():
        new_small = np.ones_like(new_small)
    paste_mask[ny1:ny2, nx1:nx2] = new_small
    paste_mask = dilate_mask(paste_mask, ksize=max(3, dilate_px))
    # the old location also legitimately changed (inpainting)
    old_region = np.zeros((h, w), bool)
    old_region[y1:y2, x1:x2] = obj_mask_small
    paste_mask |= dilate_mask(old_region, ksize=max(3, dilate_px))

    moved = work.copy()
    moved[ny1:ny2, nx1:nx2][new_small] = obj_resized[new_small]

    ys, xs = np.nonzero(paste_mask)
    cx, cy = int(xs.mean()), int(ys.mean())
    if 0 < cx < w - 1 and 0 < cy < h - 1 and paste_mask[cy - 1:cy + 2, cx - 1:cx + 2].any():
        clone_mask = paste_mask.astype(np.uint8) * 255
        try:
            target = cv2.seamlessClone(moved, work, clone_mask, (cx, cy), cv2.NORMAL_CLONE)
        except cv2.error:  # border-touching mask: direct paste
            target = moved
    else:
        target = moved
    target = target.copy()
    target[~paste_mask] = source_rgb[~paste_mask]
    return target, paste_mask


# ---------------------------------------------------------------------------
# Pair processing (idempotent, ledger-resumable)
# ---------------------------------------------------------------------------

def _relative_kind(plan: Dict[str, Any]) -> Tuple[str, str, Tuple[int, int, int, int]]:
    iv = plan["intervention"]
    rel = iv["new_value"][0]
    eid_a, eid_b = iv["entity_id"], rel["target"]
    bbox_a = normalize_bbox(plan["objects"][eid_a])
    bbox_b = normalize_bbox(plan["objects"][eid_b])
    if bbox_a is None or bbox_b is None:
        raise ValueError(f"{plan['pair_id']}: missing bbox for move")
    return rel["type"], bbox_a, bbox_b


def process_pair(plan: Dict[str, Any], image_root: Path, out_dir: Path,
                 edit_client: EditClient, matting: MattingBackend,
                 seed: int = 1234) -> Dict[str, Any]:
    """Composite one plan -> target.png + mask.png; returns ledger record."""
    pair_id = plan["pair_id"]
    target_rel = f"targets/{pair_id}__target.png"
    mask_rel = f"masks/{pair_id}__mask.png"
    target_path = out_dir / target_rel
    mask_path = out_dir / mask_rel

    src_path = image_root / plan["source"]["image"]
    source_rgb = np.asarray(Image.open(src_path).convert("RGB")).copy()
    h, w = source_rgb.shape[:2]

    mode = plan["compositing"]["mode"]
    if mode == "move":
        kind, bbox_a, bbox_b = _relative_kind(plan)
        target, paste_mask = composite_move(
            source_rgb, bbox_a, bbox_b, kind, matting,
            dilate_px=int(plan["compositing"].get("dilate_px", 9)))
        edit_prompt = plan["compositing"]["edit_prompt"]
    else:
        bbox = normalize_bbox(plan["compositing"]["crop_bbox"])
        if bbox is None:
            raise ValueError(f"{pair_id}: crop_bbox missing")
        x1, y1, x2, y2 = context_crop_bbox(bbox, w, h,
                                           plan["compositing"].get("context_frac", 0.10))
        crop = Image.fromarray(source_rgb[y1:y2, x1:x2])
        edited = edit_client.edit_crop(crop, plan["compositing"]["edit_prompt"], seed)
        alpha = matting.extract_alpha(edited)
        target, paste_mask = composite_edit(
            source_rgb, bbox, np.asarray(edited), alpha,
            context_frac=plan["compositing"].get("context_frac", 0.10))
        edit_prompt = plan["compositing"]["edit_prompt"]

    import io
    buf = io.BytesIO()
    Image.fromarray(target).save(buf, format="PNG", compress_level=3)
    atomic_write_bytes(target_path, buf.getvalue())
    mask_u8 = (paste_mask.astype(np.uint8)) * 255
    buf = io.BytesIO()
    Image.fromarray(mask_u8).save(buf, format="PNG", compress_level=3)
    atomic_write_bytes(mask_path, buf.getvalue())

    record = {
        "pair_id": pair_id,
        "subtype": plan["subtype"],
        "mode": mode,
        "target": target_rel,
        "mask": mask_rel,
        "changed_px": int(paste_mask.sum()),
        "edit_prompt": edit_prompt,
        "status": "ok",
    }
    return record


def run(plans_path: Path, image_root: Path, out_dir: Path,
        provider: str = "mock", matting_name: str = "mock",
        limit: int = 0, seed: int = 1234,
        edit_client: Optional[EditClient] = None,
        matting: Optional[MattingBackend] = None) -> Dict[str, Any]:
    plans = load_jsonl(plans_path)
    if limit:
        plans = plans[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "targets").mkdir(exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)
    ledger_path = out_dir / "ledger.jsonl"

    done: Dict[str, Dict[str, Any]] = {}
    if ledger_path.exists():
        for row in load_jsonl(ledger_path):
            if row.get("status") == "ok":
                done[row["pair_id"]] = row

    edit_client = edit_client or build_edit_client(provider)
    matting = matting or build_matting(matting_name)

    results: List[Dict[str, Any]] = []
    for plan in plans:
        pid = plan["pair_id"]
        if pid in done and (out_dir / done[pid]["target"]).exists():
            results.append({**done[pid], "status": "skipped_idempotent"})
            continue
        try:
            record = process_pair(plan, image_root, out_dir, edit_client, matting, seed)
            append_jsonl(ledger_path, record)
            results.append(record)
        except Exception as exc:  # noqa: BLE001 - ledger everything, keep going
            record = {"pair_id": pid, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            append_jsonl(ledger_path, record)
            results.append(record)

    ok_rows = [r for r in results if r["status"] == "ok"]
    skip_rows = [r for r in results if r["status"] == "skipped_idempotent"]
    fail_rows = [r for r in results if r["status"] == "failed"]

    # materialize the completed-pair manifest (plans + asset paths)
    by_id = {r["pair_id"]: r for r in results if r["status"] in ("ok", "skipped_idempotent")}
    manifest_rows = []
    for plan in plans:
        pid = plan["pair_id"]
        rec = by_id.get(pid)
        if rec is None:
            continue
        row = dict(plan)
        row["target"]["image"] = rec["target"]
        row["target"]["mask"] = rec["mask"]
        row["compositing"]["changed_px"] = rec.get("changed_px", 0)
        manifest_rows.append(row)
    write_jsonl(out_dir / "manifest.jsonl", manifest_rows)

    return {"total": len(results), "ok": len(ok_rows), "skipped": len(skip_rows),
            "failed": len(fail_rows), "failures": fail_rows[:10]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--provider", default="mock", choices=["mock", "nanobanana", "addit"])
    ap.add_argument("--matting", default="mock",
                    choices=["mock", "auto", "rembg", "sam2", "grabcut"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.provider == "nanobanana" and not os.environ.get("NANOBANANA_KEY"):
        raise SystemExit("nanobanana provider requires env NANOBANANA_KEY (refusing to run)")

    summary = run(args.plans, args.image_root, args.out_dir,
                  provider=args.provider, matting_name=args.matting,
                  limit=args.limit, seed=args.seed)
    import json
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
