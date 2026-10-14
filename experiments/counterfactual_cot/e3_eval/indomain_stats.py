#!/usr/bin/env python3
"""E3 eval shared stats helpers + Module-A (in-domain, 95 pairs) three-way CLI.

This module is deliberately PURE (stdlib only — no torch/numpy) so the CPU
selftest and every stats component can import it anywhere.

Recipe (byte-identical to e1_general_t2i/stats_paired.py::paired_bootstrap /
final991_eval_v1 Module A): per-item paired diff, 20000 bootstrap resamples
seeded 1234, 2.5/97.5 percentile CI, significant iff the CI excludes 0,
win/loss/tie counts, optional scipy Wilcoxon signed-rank on the nonzero
diffs.  The bootstrap function body below is copied verbatim from E1 so
three-way numbers are directly comparable with the E1 stats_summary.json
replication pair (final991 - base).

Three model pairs everywhere:
  v6 - final991     (E3 question: alignment loss on top of Module-A SFT)
  v6 - base         (total effect)
  final991 - base   (replication sanity check vs final991_eval_v1/E1)

CLI (Module A): reads indomain_{model}_{split}.jsonl rows written by
final991_eval_v1/score_indomain.py (base/final991, copied here by
run_e3_eval.sh) and by e3_eval/score_indomain_threeway.py (v6), then writes
  indomain_stats_threeway.json
  indomain_table.md
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFCOT = HERE.parent
DEFAULT_SCORES_DIR = CFCOT / "outputs/e3_eval/scores_indomain"

MODEL_ORDER = ["base", "final991", "v6"]
DEFAULT_PAIRS = [("v6", "final991"), ("v6", "base"), ("final991", "base")]

# (field, better) — polarity only annotates direction; significance is CI != 0
IND_METRICS = [
    ("tgt_mse", "lower"),
    ("src_mse", "lower"),
    ("tgt_psnr", "higher"),
    ("tgt_ssim", "higher"),
    ("tgt_lpips", "lower"),
    ("clip_src", "higher"),
    ("clip_tgt", "higher"),
    ("dino_tgt", "higher"),
    ("in_mse", "lower"),
    ("out_mse", "lower"),
    ("strength", "ratio"),      # gen_change / gold_change inside mask: ~1 is ideal
    ("bg_drift", "lower"),
    ("directional", "higher"),  # clip_tgt - clip_tgt_vs_src
]


# ---------------------------------------------------------------------------
# shared pure helpers
# ---------------------------------------------------------------------------
def load_rows(path: Path, key: str = "pair_id") -> dict[str, dict]:
    """jsonl -> {key: row}; missing file returns {} (callers decide tolerance)."""
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            rows[rec[key]] = rec
    return rows


def paired_bootstrap(diffs: list[float], n_boot: int, seed: int = 1234) -> tuple[float, float, float]:
    """Verbatim from e1_general_t2i/stats_paired.py (returns lo, hi, mean)."""
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[int(0.975 * len(means))]
    return lo, hi, statistics.fmean(diffs)


def wilcoxon_p(diffs: list[float]) -> float | None:
    """Signed-rank p on nonzero diffs (scipy if importable, else None)."""
    nz = [d for d in diffs if d != 0]
    if len(nz) < 5:
        return None
    try:
        from scipy.stats import wilcoxon

        _stat, p = wilcoxon(nz)
        return float(p)
    except Exception:
        return None


def pair_stats(rows_x: dict[str, dict], rows_y: dict[str, dict], field: str,
               ids: list[str], n_boot: int = 20000, seed: int = 1234) -> dict:
    """Paired stats of (y - x) over `ids`; None/non-finite values dropped."""
    diffs: list[float] = []
    x_vals: list[float] = []
    y_vals: list[float] = []
    dropped = 0
    for i in ids:
        vx = rows_x.get(i, {}).get(field)
        vy = rows_y.get(i, {}).get(field)
        if vx is None or vy is None:
            dropped += 1
            continue
        d = vy - vx
        if not (isinstance(d, float) and math.isfinite(d)):
            dropped += 1
            continue
        diffs.append(d)
        x_vals.append(vx)
        y_vals.append(vy)
    if not diffs:
        return {"n": 0, "n_dropped": dropped}
    lo, hi, mean_diff = paired_bootstrap(diffs, n_boot, seed)
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    entry = {
        "n": len(diffs),
        "n_dropped": dropped,
        "x_mean": statistics.fmean(x_vals),
        "y_mean": statistics.fmean(y_vals),
        "mean_diff_y_minus_x": mean_diff,
        "ci95": [lo, hi],
        "significant": bool(lo > 0 or hi < 0),
        "wins/losses/ties": [wins, losses, len(diffs) - wins - losses],
        "wilcoxon_p": wilcoxon_p(diffs),
    }
    return entry


def available_pairs(models: list[str],
                    pairs: list[tuple[str, str]] = DEFAULT_PAIRS) -> list[tuple[str, str]]:
    """Ordered pairs restricted to models actually present in the data."""
    return [(a, b) for (a, b) in pairs if a in models and b in models]


def pair_label(y: str, x: str) -> str:
    return f"{y}-{x}"


def fmt_entry(e: dict | None, prec: int = 4) -> str:
    if not e or e.get("n", 0) == 0:
        return "n/a"
    txt = (f"{e['mean_diff_y_minus_x']:+.{prec}f} "
           f"[{e['ci95'][0]:+.{prec}f}, {e['ci95'][1]:+.{prec}f}]")
    if e["significant"]:
        txt += " SIG"
    p = e.get("wilcoxon_p")
    if p is not None:
        txt += f" (p={p:.4g})"
    w = e["wins/losses/ties"]
    txt += f" {w[0]}/{w[1]}/{w[2]}"
    return txt


# ---------------------------------------------------------------------------
# Module-A three-way analysis
# ---------------------------------------------------------------------------
def analyze_split(rows_by_model: dict[str, dict[str, dict]], metrics: list[tuple[str, str]],
                  n_boot: int) -> dict:
    """rows_by_model: {model: {pair_id: row}} for ONE split (already loaded)."""
    models = [m for m in MODEL_ORDER if m in rows_by_model and rows_by_model[m]]
    if len(models) < 2:
        return {"models": models, "skipped": True}
    ids = None
    for m in models:
        s = set(rows_by_model[m])
        ids = s if ids is None else (ids & s)
    ids = sorted(ids)
    out: dict = {"models": models, "n_pairs": len(ids), "model_means": {}, "pairs": {}}
    for field, _better in metrics:
        mm = {}
        for m in models:
            vals = [rows_by_model[m][i][field] for i in ids
                    if isinstance(rows_by_model[m][i].get(field), float)]
            if vals:
                mm[m] = statistics.fmean(vals)
        out["model_means"][field] = mm
    for (y, x) in available_pairs(models):
        plist: dict[str, dict] = {}
        for field, _better in metrics:
            plist[field] = pair_stats(rows_by_model[x], rows_by_model[y], field, ids, n_boot)
        out["pairs"][pair_label(y, x)] = {"diff": f"{y} - {x}", "metrics": plist}
    return out


def markdown_report(summary: dict, out: Path) -> None:
    models = summary.get("models", [])
    lines = ["# E3 Module A (in-domain, 95 counterfactual pairs): three-way comparison", "",
             f"Models: {', '.join(models)}. Paired diff = column model - row model; "
             f"bootstrap 95% CI ({summary['bootstrap_resamples']} resamples, seed 1234); "
             f"SIG = CI excludes 0. Non-finite values (e.g. PSNR=inf) are dropped per metric.", ""]
    for split, data in summary["splits"].items():
        if data.get("skipped"):
            lines += [f"## {split}: skipped (need >= 2 models with rows)", ""]
            continue
        lines += [f"## split: {split} (n={data['n_pairs']} paired pairs)", ""]
        cols = [pair_label(y, x) for (y, x) in available_pairs(data["models"])]
        lines.append("| metric | " + " | ".join(f"{m} mean" for m in data["models"])
                     + " | " + " | ".join(cols) + " |")
        lines.append("|---" * (1 + len(data["models"]) + len(cols)) + "|")
        for field, _better in IND_METRICS:
            means = data["model_means"].get(field, {})
            mcells = [f"{means[m]:.4f}" if m in means else "n/a" for m in data["models"]]
            pcells = [fmt_entry(data["pairs"][c]["metrics"][field]) if c in data["pairs"] else "n/a"
                      for c in cols]
            lines.append(f"| {field} | " + " | ".join(mcells) + " | "
                         + " | ".join(pcells) + " |")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER)
    parser.add_argument("--splits", nargs="*", default=["val", "test"])
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--out-prefix", default="indomain_stats_threeway")
    args = parser.parse_args()

    rows_all: dict[str, dict[str, dict[str, dict]]] = {}
    for split in args.splits:
        per_model: dict[str, dict[str, dict]] = {}
        for m in args.models:
            rows = load_rows(args.scores_dir / f"indomain_{m}_{split}.jsonl")
            if rows:
                per_model[m] = rows
            else:
                print(f"[warn] no rows: indomain_{m}_{split}.jsonl")
        rows_all[split] = per_model
    # pooled = union of splits
    pooled: dict[str, dict[str, dict]] = {}
    for m in args.models:
        merged: dict[str, dict] = {}
        for s in args.splits:
            for pid, row in rows_all[s].get(m, {}).items():
                merged[pid] = row
        if merged:
            pooled[m] = merged
    rows_all["pooled"] = pooled

    summary = {
        "experiment": "e3_module_a_indomain_threeway",
        "models": [m for m in MODEL_ORDER if any(m in rows_all[s] for s in rows_all)],
        "bootstrap_resamples": args.bootstrap,
        "splits": {s: analyze_split(rows_all[s], IND_METRICS, args.bootstrap)
                   for s in list(args.splits) + ["pooled"]},
        "metrics": {f: b for f, b in IND_METRICS},
    }
    out_json = args.scores_dir / f"{args.out_prefix}.json"
    out_md = args.scores_dir / "indomain_table.md"
    args.scores_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    markdown_report(summary, out_md)
    print(json.dumps({"event": "indomain_stats_done", "json": str(out_json), "md": str(out_md)}))


if __name__ == "__main__":
    main()
