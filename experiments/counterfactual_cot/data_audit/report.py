#!/usr/bin/env python3
"""Data-audit report: merges style + integrity rows into the v8 filter input.

Reads <audit-dir>/style_audit.jsonl and <audit-dir>/integrity_audit.jsonl
(either may be absent -> reported as missing, exit stays 0), applies the
style-drift thresholds (integrity flags were already computed at audit time),
and writes into the same directory:

  problem_pairs.jsonl   one row per pair with any problem:
                        {pair_id, split, field, intervention_type,
                         problems: [...], metrics: {...key numbers...}}
  audit_report.md       human report: totals, per-flag counts, metric
                        percentile tables, worst offenders, v8 guidance
  audit_summary.json    machine-readable aggregates for the v8 build

Style flags evaluated here (thresholds are report-time decisions so the raw
rows stay stable while the v8 filter rules are tuned):

  bg_style_drift      bg_hist_l1 > --bg-hist-l1         (default 0.05)
  global_style_drift  hist_l1 > --global-hist-l1        (default 0.10)
  sat_shift           |bg_sat_delta| > --bg-sat-delta   (default 0.03)
  contrast_shift      |bg_contrast_delta| > --bg-contrast-delta (default 0.03)
  brightness_shift    |bg_val_delta| > --bg-val-delta   (default 0.03)

Usage:
  python3 report.py --audit-dir <dir> [--bg-hist-l1 0.05 ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFCOT = HERE.parent
DEFAULT_AUDIT_DIR = CFCOT / "outputs/data_audit_v1"

STYLE_FLAGS = [
    ("bg_style_drift", "bg_hist_l1", "bg_hist_l1", 0.05, "above"),
    ("global_style_drift", "hist_l1", "global_hist_l1", 0.10, "above"),
    ("sat_shift", "bg_sat_delta", "bg_sat_delta", 0.03, "abs_above"),
    ("contrast_shift", "bg_contrast_delta", "bg_contrast_delta", 0.03, "abs_above"),
    ("brightness_shift", "bg_val_delta", "bg_val_delta", 0.03, "abs_above"),
]
PROBLEM_METRICS = ["mask_coverage", "changed_in_frac", "changed_in_frac_sensitive",
                   "changed_out_frac", "precision_in", "iou_change_mask",
                   "mad_in", "mad_out", "hist_l1", "bg_hist_l1",
                   "bg_sat_delta", "bg_val_delta", "bg_contrast_delta",
                   "new_value_ref_ratio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    for _flag, _metric, knob, default, _mode in STYLE_FLAGS:
        parser.add_argument(f"--{knob.replace('_', '-')}", type=float, default=default,
                            help=f"threshold for {_flag} (default {default})")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def style_flags_for(row: dict, args: argparse.Namespace) -> list[str]:
    flags = []
    for name, metric, knob, _default, mode in STYLE_FLAGS:
        val = row.get(metric)
        thr = getattr(args, knob)
        if val is None:
            continue
        hit = val > thr if mode == "above" else abs(val) > thr
        if hit:
            flags.append(name)
    return flags


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def fmt(x, prec=4):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{x:.{prec}f}"


def main() -> None:
    args = parse_args()
    audit_dir = args.audit_dir
    style_rows = load_rows(audit_dir / "style_audit.jsonl")
    integ_rows = load_rows(audit_dir / "integrity_audit.jsonl")
    if not style_rows and not integ_rows:
        sys.exit(f"no audit rows under {audit_dir} (run style_audit.py / "
                 "integrity_audit.py first)")

    style_by_id = {r["pair_id"]: r for r in style_rows}
    integ_by_id = {r["pair_id"]: r for r in integ_rows}
    all_ids = sorted(set(style_by_id) | set(integ_by_id),
                     key=lambda pid: (integ_by_id.get(pid, style_by_id.get(pid, {}))
                                      .get("split", "zz"), pid))

    problems: list[dict] = []
    flag_counts: dict[str, int] = {}
    split_problems: dict[str, int] = {}
    clean = 0
    for pid in all_ids:
        s, i = style_by_id.get(pid), integ_by_id.get(pid)
        flags: list[str] = []
        if s and not s.get("error"):
            flags += style_flags_for(s, args)
        if i:
            flags += list(i.get("flags", []))
        flags = sorted(set(flags))
        if flags:
            for fl in flags:
                flag_counts[fl] = flag_counts.get(fl, 0) + 1
            base = i or s
            metrics = {}
            for src in (i, s):
                if src:
                    for m in PROBLEM_METRICS:
                        if isinstance(src.get(m), float):
                            metrics[m] = src[m]
            sp = base.get("split", "?")
            problems.append({"pair_id": pid, "split": sp, "field": base.get("field"),
                             "intervention_type": base.get("intervention_type"),
                             "problems": flags, "metrics": metrics})
            split_problems[sp] = split_problems.get(sp, 0) + 1
        else:
            clean += 1

    # ---- outputs ----------------------------------------------------------
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "problem_pairs.jsonl").open("w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p, sort_keys=True) + "\n")

    n_errors = sum(1 for r in style_rows if r.get("error")) + \
        sum(1 for r in integ_rows if r.get("error"))
    summary = {
        "n_pairs": len(all_ids), "n_clean": clean,
        "n_problem_pairs": len(problems),
        "n_rows": {"style": len(style_rows), "integrity": len(integ_rows)},
        "n_row_errors": n_errors,
        "flag_counts": dict(sorted(flag_counts.items())),
        "problems_by_split": dict(sorted(split_problems.items())),
        "thresholds": {name: getattr(args, knob)
                       for name, _m, knob, _d, _mo in STYLE_FLAGS},
        "problem_pair_ids": [p["pair_id"] for p in problems],
        "metric_percentiles": {},
    }
    for metric in ("hist_l1", "bg_hist_l1", "bg_sat_delta", "bg_val_delta",
                   "bg_contrast_delta"):
        vals = [r[metric] for r in style_rows
                if not r.get("error") and isinstance(r.get(metric), float)]
        if vals:
            summary["metric_percentiles"][metric] = {
                "p50": percentile(vals, 0.50), "p90": percentile(vals, 0.90),
                "p99": percentile(vals, 0.99), "max": max(vals)}
    for metric in ("mask_coverage", "changed_in_frac", "changed_in_frac_sensitive",
                   "changed_out_frac", "precision_in", "iou_change_mask"):
        vals = [r[metric] for r in integ_rows if isinstance(r.get(metric), float)]
        if vals:
            summary["metric_percentiles"][f"integrity.{metric}"] = {
                "p50": percentile(vals, 0.50), "p90": percentile(vals, 0.90),
                "p99": percentile(vals, 0.99), "max": max(vals)}
    (audit_dir / "audit_summary.json").write_text(json.dumps(summary, indent=1),
                                                  encoding="utf-8")

    # ---- markdown ---------------------------------------------------------
    lines = ["# final991_automask_v5 data audit report", "",
             f"Pairs audited: **{len(all_ids)}** "
             f"(style rows: {len(style_rows)}, integrity rows: {len(integ_rows)}; "
             f"row errors: {n_errors}). Clean: **{clean}**; with problems: "
             f"**{len(problems)}**.", ""]
    if not style_rows:
        lines += [f"(style_audit.jsonl missing under {audit_dir} — style "
                  "columns skipped)", ""]
    if not integ_rows:
        lines += [f"(integrity_audit.jsonl missing under {audit_dir} — "
                  "integrity flags skipped)", ""]

    lines += ["## Problem counts by flag", "", "| flag | pairs |", "|---|---|"]
    for fl, n in sorted(flag_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {fl} | {n} |")
    if not flag_counts:
        lines.append("| (none) | 0 |")
    lines += ["", "## Problems by split", "", "| split | problem pairs |",
              "|---|---|"]
    for sp, n in sorted(split_problems.items()):
        lines.append(f"| {sp} | {n} |")
    lines.append("")

    lines += ["## Metric percentiles (all audited pairs)", ""]
    for source, metrics in (("style", ("hist_l1", "bg_hist_l1", "bg_sat_delta",
                                       "bg_val_delta", "bg_contrast_delta")),
                            ("integrity", ("integrity.mask_coverage",
                                           "integrity.changed_in_frac",
                                           "integrity.changed_in_frac_sensitive",
                                           "integrity.changed_out_frac",
                                           "integrity.precision_in",
                                           "integrity.iou_change_mask"))):
        present = [m for m in metrics if m in summary["metric_percentiles"]]
        if not present:
            continue
        lines += [f"### {source}", "", "| metric | p50 | p90 | p99 | max |",
                  "|---|---|---|---|---|"]
        for m in present:
            e = summary["metric_percentiles"][m]
            lines.append(f"| {m} | {fmt(e['p50'])} | {fmt(e['p90'])} "
                         f"| {fmt(e['p99'])} | {fmt(e['max'])} |")
        lines.append("")

    lines += ["## Worst offenders", ""]
    for title, rowsel, metric, reverse in (
            ("background style drift (bg_hist_l1)", style_rows, "bg_hist_l1", True),
            ("background churn (changed_out_frac)", integ_rows, "changed_out_frac", True),
            ("weakest edits (changed_in_frac@8, ascending)", integ_rows,
             "changed_in_frac_sensitive", False)):
        vals = [(r.get(metric), r["pair_id"]) for r in rowsel
                if isinstance(r.get(metric), float)]
        if not vals:
            continue
        vals.sort(reverse=reverse)
        lines += [f"**Top 10 by {title}:**", ""]
        for v, pid in vals[:10]:
            lines.append(f"- {pid}: {fmt(v)}")
        lines.append("")

    lines += ["## v8 filter guidance", "",
              "Thresholds used by this report (regenerate with different values",
              "via the CLI knobs; the raw audit rows stay unchanged):", "",
              "| flag | metric | threshold |", "|---|---|---|"]
    for name, metric, knob, _default, mode in STYLE_FLAGS:
        cmp_op = ">" if mode == "above" else "|x| >"
        lines.append(f"| {name} | {metric} | {cmp_op} {getattr(args, knob)} |")
    lines += ["",
              "Suggested v8 filter (initial; calibrate on the training-loss",
              "response): drop pairs with any hard integrity flag",
              "(`bg_churn`, `no_change_in_mask`, `mask_too_small`,",
              "`mask_too_large`, `mask_nonzero_mismatch`, `*_sha_mismatch`,",
              "`missing_*`, `low_change_concentration`, `new_value_unreferenced`,",
              "`recorded_local_check_failed`, `audit_error`); treat the style",
              "flags (bg_style_drift / sat_shift / ...) as a soft tier —",
              "inspect the worst-offender lists before dropping.", "",
              "Machine-readable inputs for the v8 build:",
              "`problem_pairs.jsonl` (per-pair flags + metrics) and",
              "`audit_summary.json` (aggregates).", ""]
    (audit_dir / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"event": "report_done", "pairs": len(all_ids),
                      "problem_pairs": len(problems),
                      "flag_counts": dict(sorted(flag_counts.items())),
                      "md": str(audit_dir / "audit_report.md")}))


if __name__ == "__main__":
    main()
