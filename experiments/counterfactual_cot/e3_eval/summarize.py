#!/usr/bin/env python3
"""E3 unified summary: three-way (base / final991 / v6) x metrics + significance.

Collects the three component stats files produced by run_e3_eval.sh:
  <e3-root>/scores_t2i/stats_threeway.json            (E1 general T2I, CLIP-T)
  <e3-root>/scores_indomain/indomain_stats_threeway.json (Module A, 95 pairs)
  <e3-root>/probe_stats/probe_stats_threeway.json      (E2 token probe, 51 pairs)
and writes a single markdown table + condensed json:
  <e3-root>/summary_table.md
  <e3-root>/summary.json

Missing components are tolerated and reported (partial runs / smoke).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from indomain_stats import DEFAULT_PAIRS, MODEL_ORDER, fmt_entry

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent / "outputs/e3_eval"

ALL_PAIRS = [f"{y}-{x}" for (y, x) in DEFAULT_PAIRS]


def load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def cell(entry: dict | None, prec: int = 4) -> str:
    return fmt_entry(entry, prec) if entry else "n/a"


def pair_cols(present_pairs: list[str]) -> list[str]:
    return [p for p in ALL_PAIRS if p in present_pairs]


def section_e1(data: dict) -> list[str]:
    clip = data.get("clipT") or {}
    lines = ["## E1 general T2I (T2I-CompBench 3x150, CLIP-T ViT-L/14)", ""]
    if clip.get("skipped") or "ALL" not in clip:
        lines += ["(E1 rows missing)", ""]
        return lines
    cols = pair_cols(list(clip["ALL"].get("pairs", {}).keys()))
    lines += ["| subset | n | " + " | ".join(cols) + " |",
              "|" + "---|" * (2 + len(cols))]
    for subset in [k for k in clip if k not in ("models", "skipped")]:
        e = clip[subset]
        cells = [cell(e["pairs"].get(c)) for c in cols]
        lines.append(f"| {subset} | {e['n']} | " + " | ".join(cells) + " |")
    lines.append("")
    if data.get("vqa") and not data["vqa"].get("skipped"):
        lines += ["### VQA faithfulness (when vqa rows exist)", ""]
        vqa = data["vqa"]
        for subset in [k for k in vqa if k not in ("models", "skipped")]:
            e = vqa[subset]
            cells = [cell(e["pairs"].get(c), 3) for c in cols]
            lines.append(f"- {subset} (n={e['n']}): " + "; ".join(
                f"{c} {t}" for c, t in zip(cols, cells) if t != "n/a"))
        lines.append("")
    return lines


def section_indomain(data: dict) -> list[str]:
    lines = ["## Module A in-domain (95 counterfactual pairs, pooled val+test)", ""]
    splits = data.get("splits", {})
    pooled = splits.get("pooled", {})
    if pooled.get("skipped") or "pairs" not in pooled:
        lines += ["(Module-A rows missing)", ""]
        return lines
    cols = pair_cols(list(pooled["pairs"].keys()))
    lines += ["| metric (better) | " + " | ".join(f"{m}" for m in pooled["models"])
              + " | " + " | ".join(cols) + " |",
              "|" + "---|" * (1 + len(pooled["models"]) + len(cols))]
    for metric, better in data.get("metrics", {}).items():
        means = pooled["model_means"].get(metric, {})
        mcells = [f"{means[m]:.4f}" if m in means else "n/a" for m in pooled["models"]]
        cells = [cell(pooled["pairs"][c]["metrics"].get(metric)) for c in cols]
        lines.append(f"| {metric} ({better}) | " + " | ".join(mcells) + " | "
                     + " | ".join(cells) + " |")
    lines.append("")
    for s in ("val", "test"):
        d = splits.get(s, {})
        if not d.get("skipped") and "pairs" in d:
            lines.append(f"- split {s}: n={d['n_pairs']} (see indomain_stats_threeway.json)")
    lines.append("")
    return lines


def section_probe(data: dict) -> list[str]:
    lines = ["## E2 token probe (51 val pairs)", ""]
    late = data.get("late")
    if late:
        cols = pair_cols(list(late["stats"].get("flow_auroc", {}).keys()))
        lines += [f"Late-step AUROC (step {late.get('late_step')}, "
                  f"layer L{late.get('layer', late.get('late_layer'))}, "
                  f"n={late.get('n_pairs_with_flow')}):", "",
                  "| pair | diff [95% CI] |", "|---|---|"]
        for c in cols:
            e = late["stats"]["flow_auroc"].get(c)
            lines.append(f"| {c} | {cell(e)} |")
        lines.append("")
    else:
        lines += ["(E2 probe rows missing)", ""]
    lines += ["Best layer per metric (largest mean paired diff):", "",
              "| metric | " + " | ".join(ALL_PAIRS) + " |",
              "|" + "---|" * (1 + len(ALL_PAIRS))]
    for metric, per_pair in data.get("metrics", {}).items():
        cells = []
        for c in ALL_PAIRS:
            d = per_pair.get(c)
            if d and d.get("best") and d["best"].get("n", 0) > 0:
                cells.append(f"L{d['best_layer']} {d['best']['mean_diff_y_minus_x']:+.4f}"
                             f"{' SIG' if d['best']['significant'] else ''}")
            else:
                cells.append("n/a")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    chk = data.get("npz_check")
    if chk:
        lines.append(f"npz cross-check (tol {chk['tol']}): " + "; ".join(
            f"{m}: max|Δ|={c['max_abs_diff']:.4f}" if c["max_abs_diff"] is not None
            else f"{m}: no npz" for m, c in chk["per_model"].items()))
        lines.append("")
    return lines


def condense(root: Path) -> dict:
    out: dict = {"models": MODEL_ORDER,
                 "pairs": ALL_PAIRS,
                 "components": {}}
    e1 = load_json(root / "scores_t2i" / "stats_threeway.json")
    if e1:
        clip = e1.get("clipT", {})
        out["components"]["e1_clipt"] = {
            s: {"n": e.get("n"), "pairs": e.get("pairs")}
            for s, e in clip.items() if s not in ("models", "skipped")}
    ind = load_json(root / "scores_indomain" / "indomain_stats_threeway.json")
    if ind:
        out["components"]["module_a"] = {
            s: {"n": d.get("n_pairs"), "pairs": d.get("pairs")}
            for s, d in ind.get("splits", {}).items() if not d.get("skipped")}
    pr = load_json(root / "probe_stats" / "probe_stats_threeway.json")
    if pr:
        out["components"]["e2_probe"] = {
            "n_pairs": pr.get("n_pairs"),
            "late": pr.get("late"),
            "best": {m: {p: {"layer": d.get("best_layer"), "diff": (d.get("best") or {}).get(
                "mean_diff_y_minus_x")} for p, d in pp.items()}
                for m, pp in pr.get("metrics", {}).items()}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e3-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.e3_root
    e1 = load_json(root / "scores_t2i" / "stats_threeway.json")
    ind = load_json(root / "scores_indomain" / "indomain_stats_threeway.json")
    pr = load_json(root / "probe_stats" / "probe_stats_threeway.json")

    lines = ["# E3 three-way evaluation summary: base / final991 / v6", "",
             "v6 = final991 SFT + token-region alignment loss (2973 steps, latest.pt, LoRA "
             "direct-load). Pairs: v6-final991 / v6-base / final991-base (replication). "
             "Bootstrap 95% CI (20000 resamples, seed 1234); SIG = CI excludes 0; "
             "optional Wilcoxon signed-rank p in parentheses.", ""]
    lines += section_e1(e1) if e1 else ["## E1 general T2I", "", "(missing)", ""]
    lines += section_indomain(ind) if ind else ["## Module A in-domain", "", "(missing)", ""]
    lines += section_probe(pr) if pr else ["## E2 token probe", "", "(missing)", ""]
    lines += ["---", "Generated by e3_eval/summarize.py; per-component tables: "
              "scores_t2i/comparison_table.md, indomain_table.md, probe_table.md"]

    root.mkdir(parents=True, exist_ok=True)
    (root / "summary_table.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(condense(root), indent=1), encoding="utf-8")
    print(json.dumps({"event": "summary_done",
                      "md": str(root / "summary_table.md"),
                      "json": str(root / "summary.json"),
                      "components_present": [n for n, ok in
                                             (("e1", e1), ("indomain", ind), ("probe", pr)) if ok]}))


if __name__ == "__main__":
    main()
