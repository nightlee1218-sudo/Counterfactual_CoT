"""data_v2 integrity guard loop (Anywhere-Repaper-style post-composite checks).

Two per-pair checks on every composited target:

  1. blend-overflow: pixels OUTSIDE the allowed paste mask must be
     byte-identical to the source (tolerance 2/255).  Any nonzero diff means
     the Poisson blend or a future code path leaked past the mask —
     the pair is marked needs_fix (never silently accepted).
  2. foreground erosion: matte the object region of the TARGET crop and
     require a minimum IoU with the paste-mask region.  A low IoU means the
     background ate into the object (or the object vanished) — mark retry.

The guard is cheap (numpy + matting) and injectable for selftests: matting
backends come from composite_target (mock/rembg/sam2/grabcut/auto).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    append_jsonl, context_crop_bbox, load_jsonl, normalize_bbox, write_json_atomic,
)
import composite_target as ct  # noqa: E402

DIFF_TOL = 2          # /255; PNG is lossless, compositing forces source outside mask
MIN_FG_IOU = 0.15     # matte-vs-paste-region IoU floor for edit mode


def check_pair(record: Dict[str, Any], image_root: Path, manifest_dir: Path,
               matting: Optional[Any] = None) -> Dict[str, Any]:
    """One manifest row -> integrity row (never raises)."""
    row: Dict[str, Any] = {"pair_id": record.get("pair_id"),
                           "subtype": record.get("subtype"), "status": "error"}
    try:
        src = np.asarray(Image.open(image_root / record["source"]["image"]).convert("RGB"))
        tgt = np.asarray(Image.open(manifest_dir / record["target"]["image"]).convert("RGB"))
        m = Image.open(manifest_dir / record["target"]["mask"]).convert("L")
        mask = np.asarray(m) > 127
        if src.shape != tgt.shape or mask.shape != src.shape[:2]:
            row["error"] = "shape mismatch between source/target/mask"
            return row

        # ---- check 1: blend overflow ------------------------------------
        keep = ~mask
        diff_out = np.any(np.abs(tgt.astype(np.int16) - src.astype(np.int16))
                          > DIFF_TOL, axis=-1)
        overflow_px = int((diff_out & keep).sum())
        row["overflow_px"] = overflow_px
        row["overflow_frac"] = float(overflow_px / max(int(keep.sum()), 1))

        # ---- check 2: foreground erosion ---------------------------------
        fg_iou = None
        mode = record.get("compositing", {}).get("mode")
        if mode == "edit" and matting is not None:
            h, w = src.shape[:2]
            bbox = normalize_bbox(record["compositing"].get("crop_bbox"))
            if bbox is not None:
                x1, y1, x2, y2 = context_crop_bbox(
                    bbox, w, h, record["compositing"].get("context_frac", 0.10))
                if x2 - x1 >= 4 and y2 - y1 >= 4:
                    tgt_crop = Image.fromarray(tgt[y1:y2, x1:x2])
                    try:
                        alpha = matting.extract_alpha(tgt_crop)
                        if alpha.shape[0] != y2 - y1 or alpha.shape[1] != x2 - x1:
                            alpha = np.asarray(Image.fromarray(alpha).resize(
                                (x2 - x1, y2 - y1)))  # nearest-ish; acceptable for guard
                        matte = alpha > 127
                        paste_region = mask[y1:y2, x1:x2]
                        inter = int((matte & paste_region).sum())
                        union = int((matte | paste_region).sum())
                        fg_iou = float(inter / union) if union else None
                    except Exception:  # noqa: BLE001 - matting failure is not a data error
                        fg_iou = None
        row["fg_iou"] = fg_iou

        overflow_bad = overflow_px > 0
        erosion_bad = fg_iou is not None and fg_iou < MIN_FG_IOU
        if overflow_bad or erosion_bad:
            row["status"] = "needs_fix"
            row["reasons"] = ([f"blend_overflow:{overflow_px}px"]
                              if overflow_bad else []) + \
                             ([f"foreground_eroded:iou={fg_iou:.3f}"]
                              if erosion_bad else [])
        else:
            row["status"] = "ok"
            row["reasons"] = []
        row["error"] = None
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_guard(manifest_path: Path, image_root: Path, out_json: Path,
              out_jsonl: Optional[Path] = None,
              matting_name: str = "mock",
              matting: Optional[Any] = None) -> Dict[str, Any]:
    matting = matting or ct.build_matting(matting_name)
    records = load_jsonl(manifest_path)
    rows = [check_pair(r, image_root, manifest_path.parent, matting)
            for r in records]
    ok_rows = [r for r in rows if r["status"] == "ok"]
    fix_rows = [r for r in rows if r["status"] == "needs_fix"]
    ious = [r["fg_iou"] for r in rows if r.get("fg_iou") is not None]
    report = {
        "schema_version": "counterfactual_data_v2_integrity/1.0",
        "manifest": str(manifest_path),
        "gates": {"max_overflow_px": 0, "min_fg_iou": MIN_FG_IOU},
        "summary": {
            "n_pairs": len(rows),
            "n_ok": len(ok_rows),
            "n_needs_fix": len(fix_rows),
            "n_error": sum(1 for r in rows if r["status"] == "error"),
            "overflow_pairs": sum(1 for r in rows if r.get("overflow_px", 0) > 0),
            "fg_iou": {"n": len(ious),
                       "mean": float(np.mean(ious)) if ious else None,
                       "min": float(np.min(ious)) if ious else None},
        },
        "rows": rows,
    }
    write_json_atomic(out_json, report)
    if out_jsonl:
        for r in rows:
            append_jsonl(out_jsonl, r)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--matting", default="mock",
                    choices=["mock", "auto", "rembg", "sam2", "grabcut"])
    args = ap.parse_args()
    report = run_guard(args.manifest, args.image_root, args.out,
                       matting_name=args.matting)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["summary"]["n_needs_fix"] or report["summary"]["n_error"]:
        print(f"INTEGRITY: {report['summary']['n_needs_fix']} needs_fix, "
              f"{report['summary']['n_error']} errors (see {args.out})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
