#!/usr/bin/env python3
"""Style-drift audit for the final991_automask_v5 pairs (target vs source).

Quantifies how much the TARGET image drifted from the SOURCE image beyond
the intended edit — the failure mode where the edit model re-renders the
whole scene instead of doing a local, background-consistent change.  Every
metric is computed twice: globally and OUTSIDE the difference mask ("bg_"
prefix), because the masked region is *supposed* to change while the
background should stay pixel-consistent (the target prompts demand it).

Metrics per pair (pure PIL + numpy, no torch):
  sat_mean / val_mean / contrast / hist_l1          per image (source, target)
  sat_delta / val_delta / contrast_delta / hist_l1   target - source, global
  bg_*                                              same, outside-mask pixels only

  * sat      HSV-style saturation, mean of (max-min)/max over RGB in [0,1]
  * val      brightness, mean of per-pixel max-RGB in [0,1]
  * contrast std of grayscale (Rec.601 weights) in [0,1]
  * hist_l1  mean over RGB channels of the L1 distance between 16-bin,
             sum-normalized histograms (0 = identical distribution)

Rows -> <out-dir>/style_audit.jsonl (sorted by split, pair_id; deterministic).
Aggregates -> <out-dir>/style_summary.json (percentiles per metric).

Thresholds/flagging happen in report.py (report-time decision, so the raw
rows stay stable while v8 filter rules are being tuned).

Usage (login node or GPU instance, CPU only):
  python3 style_audit.py --splits-dir <...>/artifacts/splits
  python3 style_audit.py --splits val test --workers 16
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFCOT = HERE.parent
DEFAULT_SPLITS_DIR = CFCOT / "final991_automask_v5_data/artifacts/splits"
DEFAULT_OUT_DIR = CFCOT / "outputs/data_audit_v1"
HIST_BINS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="audit only the first N pairs (0 = all)")
    parser.add_argument("--workers", type=int, default=0, help="process workers (0 = min(8, cpus))")
    return parser.parse_args()


def load_records(splits_dir: Path, splits: list[str], limit: int) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for split in splits:
        path = splits_dir / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append((split, json.loads(line)))
    if limit:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------
# per-pair metric core (runs inside workers; numpy/PIL imported lazily)
# ---------------------------------------------------------------------------
def _image_stats(img, mask=None):
    """img: HxWx3 float32 [0,1]; mask: optional HxW bool of pixels to KEEP."""
    import numpy as np
    if mask is not None:
        if not mask.any():
            return None
        px = img[mask]
        gray = (px[:, 0] * 0.299 + px[:, 1] * 0.587 + px[:, 2] * 0.114)
    else:
        px = img.reshape(-1, 3)
        gray = (img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114).reshape(-1)
    mx = px.max(axis=1)
    mn = px.min(axis=1)
    # pure-black pixels (max RGB == 0) have undefined saturation: exclude
    # them from the mean instead of dividing a (N,) numerator by a (K,)
    # boolean-filtered denominator — the old form crashed every image that
    # contained a single black pixel (522/991 real pairs, 20261015 fix)
    nz = mx > 0
    sat = (mx[nz] - mn[nz]) / mx[nz] if nz.any() else mx[:0]
    return {
        "sat_mean": float(sat.mean()) if len(sat) else 0.0,
        "val_mean": float(mx.mean()),
        "contrast": float(gray.std()),
    }


def _pair_stats(source_img, target_img, mask):
    s = _image_stats(source_img)
    t = _image_stats(target_img)
    s_bg = _image_stats(source_img, ~mask) if mask is not None else None
    t_bg = _image_stats(target_img, ~mask) if mask is not None else None
    out = {}
    for key, short in (("sat_mean", "sat"), ("val_mean", "val"),
                       ("contrast", "contrast")):
        out[f"{key}_src"], out[f"{key}_tgt"] = s[key], t[key]
        out[f"{short}_delta"] = t[key] - s[key]
        if s_bg is not None and t_bg is not None:
            out[f"bg_{short}_src"], out[f"bg_{short}_tgt"] = s_bg[key], t_bg[key]
            out[f"bg_{short}_delta"] = t_bg[key] - s_bg[key]
    return out


def _hist_l1_pair(source_img, target_img, mask):
    """Direct |h_src - h_tgt| L1 (mean over channels), global + outside-mask."""
    import numpy as np

    def l1(chan_a, chan_b):
        total = 0.0
        for c in range(3):
            ha, _ = np.histogram(chan_a[..., c], bins=HIST_BINS, range=(0.0, 1.0))
            hb, _ = np.histogram(chan_b[..., c], bins=HIST_BINS, range=(0.0, 1.0))
            total += abs(ha / max(chan_a[..., c].size, 1)
                         - hb / max(chan_b[..., c].size, 1)).sum()
        return float(total / 3.0)

    out = {"hist_l1": l1(source_img, target_img)}
    if mask is not None:
        keep = ~mask
        if keep.any():
            out["bg_hist_l1"] = l1(source_img[keep], target_img[keep])
        else:
            out["bg_hist_l1"] = None
    return out


def audit_pair(split: str, record: dict) -> dict:
    """One pair -> one row. Never raises; load problems become row['error']."""
    import numpy as np
    from PIL import Image

    row = {"pair_id": record["pair_id"], "split": split,
           "field": record["intervention"]["field"],
           "intervention_type": record["intervention_type"]}
    try:
        src = np.asarray(Image.open(record["source"]["image"]).convert("RGB"),
                         dtype=np.float32) / 255.0
        tgt = np.asarray(Image.open(record["target"]["image"]).convert("RGB"),
                         dtype=np.float32) / 255.0
        if src.shape != tgt.shape:
            row["error"] = f"size mismatch: {src.shape} vs {tgt.shape}"
            return row
        mask = None
        mask_path = record.get("automatic_difference_mask", {}).get("path")
        if mask_path:
            m = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
            mask = m > 127
            if mask.shape != src.shape[:2]:
                mask = None  # report global-only metrics; integrity audit flags it
        row.update(_pair_stats(src, tgt, mask))
        row.update(_hist_l1_pair(src, tgt, mask))
        row["error"] = None
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _work(job):
    split, record = job
    return audit_pair(split, record)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> None:
    args = parse_args()
    if not args.splits_dir.is_dir():
        sys.exit(f"splits dir not found: {args.splits_dir} (pass --splits-dir)")
    jobs = load_records(args.splits_dir, args.splits, args.limit)
    workers = args.workers or min(8, __import__("os").cpu_count() or 1)
    rows = []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_work, jobs, chunksize=8))
    else:
        rows = [_work(j) for j in jobs]
    rows.sort(key=lambda r: (r["split"], r["pair_id"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "style_audit.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    ok_rows = [r for r in rows if not r["error"]]
    summary = {"n_pairs": len(rows), "n_errors": len(rows) - len(ok_rows),
               "splits": args.splits, "metric_percentiles": {}}
    for metric in ("hist_l1", "bg_hist_l1", "sat_delta", "val_delta", "contrast_delta",
                   "bg_sat_delta", "bg_val_delta", "bg_contrast_delta"):
        vals = [r[metric] for r in ok_rows if isinstance(r.get(metric), float)]
        if vals:
            summary["metric_percentiles"][metric] = {
                "p50": percentile(vals, 0.50), "p90": percentile(vals, 0.90),
                "p99": percentile(vals, 0.99), "max": max(vals)}
    (args.out_dir / "style_summary.json").write_text(json.dumps(summary, indent=1),
                                                    encoding="utf-8")
    print(json.dumps({"event": "style_audit_done", "rows": len(rows),
                      "errors": summary["n_errors"], "out": str(out_path)}))


if __name__ == "__main__":
    main()
