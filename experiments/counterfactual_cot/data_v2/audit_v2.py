"""data_v2 acceptance gate (audit_v2).

Verifies, per completed pair, the construction invariant of the compositing
pipeline — **the target background is pixel-identical to the source**:

  * bg_hist_l1 <= 0.05   histogram L1 between source/target outside the
                         allowed paste mask (definition replicated from
                         data_audit/style_audit.py). By construction this is
                         ~0; the gate is a regression check (e.g. someone
                         reintroduces full-image regeneration).
  * changed_out_frac == 0  fraction of pixels that differ from the source
                         OUTSIDE the allowed mask. Must be exactly 0.
  * mask IoU spot-check   sampled IoU between the actual changed-pixel mask
                         and the intervened object bbox region (edit mode).

Outputs an audit report JSON and exits nonzero when any gate fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import make_rng, normalize_bbox  # noqa: E402

HIST_BINS = 32  # must match data_audit/style_audit.py
BG_HIST_L1_MAX = 0.05
CHANGED_OUT_FRAC_MAX = 0.0
DIFF_TOL = 2  # /255 tolerance for "changed pixel" (PNG is lossless; 2 is generous)


def _hist_l1(chan_a: np.ndarray, chan_b: np.ndarray) -> float:
    total = 0.0
    for c in range(3):
        ha, _ = np.histogram(chan_a[..., c], bins=HIST_BINS, range=(0.0, 1.0))
        hb, _ = np.histogram(chan_b[..., c], bins=HIST_BINS, range=(0.0, 1.0))
        total += abs(ha / max(chan_a[..., c].size, 1)
                     - hb / max(chan_b[..., c].size, 1)).sum()
    return float(total / 3.0)


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path: Path) -> Optional[np.ndarray]:
    if not path or not Path(path).exists():
        return None
    m = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    return m > 127


def audit_pair(record: Dict[str, Any], image_root: Path, manifest_dir: Path,
               iou_check: bool = False) -> Dict[str, Any]:
    """One manifest row -> one audit row. Never raises; errors go in row.

    The source image resolves under ``image_root``; the generated target and
    mask resolve under ``manifest_dir`` (the directory of the manifest, which
    is the composite out-dir holding targets/ and masks/).
    """
    row: Dict[str, Any] = {
        "pair_id": record.get("pair_id"),
        "subtype": record.get("subtype"),
        "status": "error",
    }
    try:
        src_path = image_root / record["source"]["image"]
        tgt_path = manifest_dir / record["target"]["image"]
        mask_path = manifest_dir / record["target"].get("mask", "")
        src = _load_rgb(src_path)
        tgt = _load_rgb(tgt_path)
        if src.shape != tgt.shape:
            row["error"] = f"size mismatch: {src.shape} vs {tgt.shape}"
            return row
        mask = _load_mask(mask_path) if mask_path else None
        if mask is None or mask.shape != src.shape[:2]:
            row["error"] = "missing/invalid allowed mask"
            return row

        keep = ~mask
        row["bg_hist_l1"] = _hist_l1(src[keep], tgt[keep]) if keep.any() else None

        changed = np.any(np.abs(tgt - src) > DIFF_TOL / 255.0, axis=-1)
        n_out = int((changed & keep).sum())
        row["changed_out_px"] = n_out
        row["changed_out_frac"] = float(n_out / max(int(keep.sum()), 1))
        row["changed_in_px"] = int((changed & mask).sum())
        row["mask_frac"] = float(mask.mean())
        row["changed_frac"] = float(changed.mean())

        if iou_check:
            bbox = normalize_bbox(record.get("compositing", {}).get("crop_bbox"))
            if bbox is not None and record.get("compositing", {}).get("mode") == "edit":
                h, w = src.shape[:2]
                x1, y1, x2, y2 = bbox
                region = np.zeros((h, w), bool)
                region[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
                inter = int((changed & region).sum())
                union = int((changed | region).sum())
                row["bbox_iou"] = float(inter / union) if union else None

        row["error"] = None
        row["status"] = "ok"
        row["pass"] = (
            row["bg_hist_l1"] is not None
            and row["bg_hist_l1"] <= BG_HIST_L1_MAX
            and row["changed_out_frac"] <= CHANGED_OUT_FRAC_MAX
            and row["changed_in_px"] > 0  # the intervention must actually change something
        )
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_audit(manifest_path: Path, image_root: Path, out_json: Path,
              sample_iou: int = 0, seed: int = 1234) -> Dict[str, Any]:
    from common import load_jsonl, write_json_atomic
    records = load_jsonl(manifest_path)
    rng = make_rng(seed)
    iou_ids = set()
    if sample_iou > 0 and records:
        idx = list(range(len(records)))
        rng.shuffle(idx)
        iou_ids = {records[i]["pair_id"] for i in idx[:sample_iou]}

    rows = [audit_pair(r, image_root, manifest_path.parent, iou_check=r["pair_id"] in iou_ids)
            for r in records]

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    pass_rows = [r for r in ok_rows if r.get("pass")]
    bg_vals = [r["bg_hist_l1"] for r in ok_rows if r.get("bg_hist_l1") is not None]
    out_fracs = [r["changed_out_frac"] for r in ok_rows]
    ious = [r["bbox_iou"] for r in ok_rows if r.get("bbox_iou") is not None]

    report = {
        "schema_version": "counterfactual_data_v2_audit/1.0",
        "manifest": str(manifest_path),
        "image_root": str(image_root),
        "gates": {"bg_hist_l1_max": BG_HIST_L1_MAX,
                  "changed_out_frac_max": CHANGED_OUT_FRAC_MAX},
        "summary": {
            "n_records": len(rows),
            "n_ok": len(ok_rows),
            "n_error": len(rows) - len(ok_rows),
            "n_pass": len(pass_rows),
            "n_fail_gate": len(ok_rows) - len(pass_rows),
            "bg_hist_l1": {"max": max(bg_vals) if bg_vals else None,
                           "mean": float(np.mean(bg_vals)) if bg_vals else None},
            "changed_out_frac": {"max": max(out_fracs) if out_fracs else None,
                                 "n_nonzero": sum(1 for v in out_fracs if v > 0)},
            "bbox_iou_sampled": {"n": len(ious),
                                 "mean": float(np.mean(ious)) if ious else None,
                                 "min": min(ious) if ious else None},
        },
        "all_pass": (len(rows) > 0 and len(pass_rows) == len(rows)),
        "rows": rows,
    }
    write_json_atomic(out_json, report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sample-iou", type=int, default=0,
                    help="spot-check bbox IoU on N random pairs (0 = off)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    report = run_audit(args.manifest, args.image_root, args.out,
                       sample_iou=args.sample_iou, seed=args.seed)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if not report["all_pass"]:
        print(f"AUDIT FAILED: {report['summary']['n_fail_gate']} gate failures, "
              f"{report['summary']['n_error']} errors (see {args.out})", file=sys.stderr)
        sys.exit(1)
    print(f"AUDIT PASS: {report['summary']['n_pass']}/{report['summary']['n_records']} pairs")


if __name__ == "__main__":
    main()
