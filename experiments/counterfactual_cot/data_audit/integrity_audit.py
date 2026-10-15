#!/usr/bin/env python3
"""Integrity audit for the final991_automask_v5 pairs (CPU only).

Per pair, verifies that the annotation actually matches the pixels on disk:

  files        source/target/mask exist; sizes equal the recorded geometry
  checksums    sha256(source/target) vs source.image_sha256 /
               target.image_sha256, sha256(mask) vs mask.sha256
  mask         binary {0,255}; nonzero-pixel count == the recorded
               automatic_difference_mask.geometry.nonzero_pixels (exact);
               coverage fraction within [--min-coverage, --max-coverage]
  change mask  recomputed as max-channel |target - source| > threshold
               (the recorded source_candidate_mask.threshold, default 24):
               changed_in_frac (recall: did the edit happen inside the mask),
               changed_out_frac (background churn), precision_in (how
               concentrated the change is), IoU(change, mask), plus a
               sensitive 8-threshold variant and mask-decomposed MAD
  prompt       target.prompt != source.prompt; the intervention new_value
               is referenced by target.prompt (token-overlap rule with
               field-aware synonyms, calibrated to 991/991 on the real
               splits); recorded quality.local_checks.passed cross-referenced

Rows -> <out-dir>/integrity_audit.jsonl (sorted by split, pair_id), each:
  {...metrics, "flags": [...], "recorded": {...echoes...}, "error": None}

Flags (thresholds are CLI knobs, defaults in parentheses):
  missing_source_image / missing_target_image / missing_difference_mask
  image_size_mismatch / mask_size_mismatch
  source_sha_mismatch / target_sha_mismatch / mask_sha_mismatch
  mask_not_binary / mask_nonzero_mismatch / mask_too_small (<0.001) /
  mask_too_large (>0.8)
  no_change_in_mask (changed_in_frac@8 < 0.02 — mask region did not change)
  bg_churn (changed_out_frac@24 > 0.02 — background changed with the edit)
  low_change_concentration (precision_in@24 < 0.5)
  identical_prompts / new_value_unreferenced / recorded_local_check_failed
  audit_error (unexpected exception; see row["error"])

Usage:
  python3 integrity_audit.py --splits-dir <...>/artifacts/splits
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFCOT = HERE.parent
DEFAULT_SPLITS_DIR = CFCOT / "final991_automask_v5_data/artifacts/splits"
DEFAULT_OUT_DIR = CFCOT / "outputs/data_audit_v1"
DEFAULT_MIN_COVERAGE = 0.001
DEFAULT_MAX_COVERAGE = 0.8
DEFAULT_SENSITIVE_THR = 8

# ---------------------------------------------------------------------------
# prompt-atomicity: value -> is it referenced by the target prompt?
# Calibrated against the real 991 splits (selftest + full run: 991/991 pass
# with ratio >= 0.5).  Field-aware synonyms handle the structured values:
#   location "top_region" -> "upper", "bottom_region" -> "lower"
#   occlusion "unoccluded" -> "fully visible"/"nothing overlaps"/"separate"
#   relations "separate_from:X" -> beside/outside/next to/apart
#   count digits -> also accept the spelled-out word ("exactly TWO")
#   ":entity_N" ids -> the trailing index digit is dropped
# ---------------------------------------------------------------------------
NUM_WORDS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
             "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
             "10": "ten", "11": "eleven", "12": "twelve"}
TOKEN_ALTS = {
    "top": ["upper", "top"],
    "bottom": ["lower", "bottom"],
    "unoccluded": ["fully visible", "unoccluded", "nothing overlaps",
                   "no overlap", "separate"],
    "separate": ["separate", "beside", "outside", "next to", "apart"],
}
FUNCTION_WORDS = {"from", "of", "by", "to", "in", "on", "at", "the", "a",
                  "an", "and", "with", "for"}


def value_tokens(value) -> list[str]:
    v = str(value).lower()
    structured = ":" in v
    toks = []
    for t in re.split(r"[^a-z0-9]+", v):
        if t.endswith("_region"):
            t = t[: -len("_region")]
        if not t or t in FUNCTION_WORDS:
            continue
        if structured and t.isdigit():
            continue
        toks.append(t)
    return toks


def token_hit(token: str, prompt_lower: str) -> bool:
    for alt in TOKEN_ALTS.get(token, [token]):
        if alt in prompt_lower:
            return True
    if token in NUM_WORDS and NUM_WORDS[token] in prompt_lower:
        return True
    return False


def new_value_ref_ratio(new_value, target_prompt: str) -> float:
    sig = [t for t in value_tokens(new_value) if len(t) >= 3 or t.isdigit()]
    if not sig:
        return 1.0
    p = target_prompt.lower()
    return sum(1 for t in sig if token_hit(t, p)) / len(sig)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE)
    parser.add_argument("--max-coverage", type=float, default=DEFAULT_MAX_COVERAGE)
    parser.add_argument("--sensitive-thr", type=int, default=DEFAULT_SENSITIVE_THR)
    parser.add_argument("--min-changed-in", type=float, default=0.02,
                        help="no_change_in_mask when changed_in_frac@{sensitive} below this")
    parser.add_argument("--max-changed-out", type=float, default=0.02,
                        help="bg_churn when changed_out_frac@recorded-thr above this")
    parser.add_argument("--min-precision", type=float, default=0.5,
                        help="low_change_concentration when precision_in below this")
    parser.add_argument("--no-sha", action="store_true",
                        help="skip the (I/O heavy) checksum verification")
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
# per-pair core (runs inside workers; options passed explicitly)
# ---------------------------------------------------------------------------
def _echo(record: dict) -> dict:
    lc = record.get("quality", {}).get("local_checks", {})
    geom = record.get("automatic_difference_mask", {}).get("geometry", {})
    return {
        "nonzero_pixels": geom.get("nonzero_pixels"),
        "accepted": record.get("quality", {}).get("accepted"),
        "local_checks_passed": lc.get("passed"),
        "local_check_failures": lc.get("failures"),
        "changed_pixel_ratio": lc.get("changed_pixel_ratio"),
        "mask_area_ratio": lc.get("mask_area_ratio"),
    }


def audit_pair(split: str, record: dict, opts: dict) -> dict:
    import numpy as np
    from PIL import Image

    flags: list[str] = []
    row = {"pair_id": record["pair_id"], "split": split,
           "field": record["intervention"]["field"],
           "intervention_type": record["intervention_type"]}
    try:
        src_path = Path(record["source"]["image"])
        tgt_path = Path(record["target"]["image"])
        mask_path = Path(record["automatic_difference_mask"]["path"])
        for tag, p in (("source_image", src_path), ("target_image", tgt_path),
                       ("difference_mask", mask_path)):
            if not p.exists():
                flags.append(f"missing_{tag}")
        if flags:
            row.update({"flags": flags, "error": "missing_files",
                        "recorded": _echo(record)})
            return row

        geom = record["automatic_difference_mask"].get("geometry", {})
        thr = int(record["automatic_difference_mask"]
                  .get("source_candidate_mask", {}).get("threshold", 24))

        src = np.asarray(Image.open(src_path).convert("RGB"), dtype=np.uint8)
        tgt = np.asarray(Image.open(tgt_path).convert("RGB"), dtype=np.uint8)
        mask_u8 = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if (geom.get("height"), geom.get("width")) != tuple(src.shape[:2]) \
                or tgt.shape != src.shape:
            flags.append("image_size_mismatch")
        if mask_u8.shape != src.shape[:2]:
            flags.append("mask_size_mismatch")

        if not opts["no_sha"]:
            if sha256_file(src_path) != record["source"].get("image_sha256"):
                flags.append("source_sha_mismatch")
            if sha256_file(tgt_path) != record["target"].get("image_sha256"):
                flags.append("target_sha_mismatch")
            if sha256_file(mask_path) != record["automatic_difference_mask"].get("sha256"):
                flags.append("mask_sha_mismatch")

        # mask integrity
        uniq = set(int(v) for v in np.unique(mask_u8))
        if not uniq <= {0, 255}:
            flags.append("mask_not_binary")
        mask = mask_u8 > 127
        n_px = int(mask.sum())
        recorded_nz = geom.get("nonzero_pixels")
        if recorded_nz is not None and n_px != int(recorded_nz):
            flags.append("mask_nonzero_mismatch")
        coverage = n_px / mask.size if mask.size else 0.0
        if coverage < opts["min_coverage"]:
            flags.append("mask_too_small")
        if coverage > opts["max_coverage"]:
            flags.append("mask_too_large")

        # change-mask consistency (only meaningful when shapes align)
        metrics: dict = {"mask_coverage": coverage, "mask_nonzero_pixels": n_px,
                         "change_threshold": thr}
        if mask.shape == tgt.shape[:2] and tgt.shape == src.shape:
            diff = np.abs(tgt.astype(np.int16) - src.astype(np.int16)).max(axis=2)
            change = diff > thr
            change_s = diff > opts["sensitive_thr"]
            n_mask, n_out = n_px, int(mask.size) - n_px
            metrics["changed_in_frac"] = float((change & mask).sum() / n_mask) if n_mask else None
            metrics["changed_in_frac_sensitive"] = float((change_s & mask).sum() / n_mask) if n_mask else None
            metrics["changed_out_frac"] = float((change & ~mask).sum() / n_out) if n_out else None
            n_change = int(change.sum())
            metrics["changed_frac"] = n_change / mask.size
            metrics["precision_in"] = float((change & mask).sum() / n_change) if n_change else None
            union = int((change | mask).sum())
            metrics["iou_change_mask"] = float((change & mask).sum() / union) if union else None
            metrics["mad_in"] = float(diff[mask].mean()) if n_mask else None
            metrics["mad_out"] = float(diff[~mask].mean()) if n_out else None
            cin_s = metrics["changed_in_frac_sensitive"]
            if cin_s is not None and cin_s < opts["min_changed_in"]:
                flags.append("no_change_in_mask")
            cout = metrics["changed_out_frac"]
            if cout is not None and cout > opts["max_changed_out"]:
                flags.append("bg_churn")
            if n_change and metrics["precision_in"] is not None \
                    and metrics["precision_in"] < opts["min_precision"]:
                flags.append("low_change_concentration")

        # prompt atomicity
        s_prompt = record["source"].get("prompt", "")
        t_prompt = record["target"].get("prompt", "")
        if s_prompt == t_prompt:
            flags.append("identical_prompts")
        ratio = new_value_ref_ratio(record["intervention"]["new_value"], t_prompt)
        metrics["new_value_ref_ratio"] = ratio
        if ratio < 0.5:
            flags.append("new_value_unreferenced")
        lc = record.get("quality", {}).get("local_checks", {})
        if lc and lc.get("passed") is False:
            flags.append("recorded_local_check_failed")

        row.update(metrics)
        row["flags"] = flags
        row["error"] = None
        row["recorded"] = _echo(record)
    except Exception as exc:  # noqa: BLE001
        row.update({"flags": flags + ["audit_error"],
                    "error": f"{type(exc).__name__}: {exc}",
                    "recorded": _echo(record)})
    return row


def _work(job):
    split, record, opts = job
    return audit_pair(split, record, opts)


def main() -> None:
    args = parse_args()
    if not args.splits_dir.is_dir():
        sys.exit(f"splits dir not found: {args.splits_dir} (pass --splits-dir)")
    opts = {"no_sha": args.no_sha, "min_coverage": args.min_coverage,
            "max_coverage": args.max_coverage, "sensitive_thr": args.sensitive_thr,
            "min_changed_in": args.min_changed_in, "max_changed_out": args.max_changed_out,
            "min_precision": args.min_precision}
    jobs = [(split, rec, opts)
            for split, rec in load_records(args.splits_dir, args.splits, args.limit)]
    workers = args.workers or min(8, __import__("os").cpu_count() or 1)
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_work, jobs, chunksize=4))
    else:
        rows = [_work(j) for j in jobs]
    rows.sort(key=lambda r: (r["split"], r["pair_id"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "integrity_audit.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")

    flag_counts: dict[str, int] = {}
    for r in rows:
        for fl in r.get("flags", []):
            flag_counts[fl] = flag_counts.get(fl, 0) + 1
    print(json.dumps({"event": "integrity_audit_done", "rows": len(rows),
                      "pairs_with_flags": sum(1 for r in rows if r.get("flags")),
                      "flag_counts": dict(sorted(flag_counts.items())),
                      "out": str(out_path)}))


if __name__ == "__main__":
    main()
