#!/usr/bin/env python3
"""E1 general-T2I three-way stats: CLIP-T per-item paired comparison.

Reads clipT_{base,final991,v6}.jsonl (base/final991 rows are copies of the
E1 run; v6 rows come from e1_general_t2i/score_clip.py --models v6 on the
e3_eval v6 generation) and computes the same paired bootstrap recipe as
e1_general_t2i/stats_paired.py (per-item diff, 20000 resamples seed 1234,
2.5/97.5 CI, significant iff CI excludes 0, optional Wilcoxon), for all
three model pairs and per T2I-CompBench subset.

VQA faithfulness rows (vqa_{model}.jsonl) are merged when present for at
least two models (binary match rate), mirroring stats_paired.py.

Outputs (under --scores-dir):
  stats_threeway.json
  comparison_table.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from indomain_stats import (DEFAULT_PAIRS, MODEL_ORDER, available_pairs, fmt_entry,
                            load_rows, pair_label, pair_stats)

HERE = Path(__file__).resolve().parent
DEFAULT_SCORES_DIR = HERE.parent / "outputs/e3_eval/scores_t2i"


def analyze_subsets(rows_by_model: dict[str, dict[str, dict]], field: str,
                    n_boot: int) -> dict:
    models = [m for m in MODEL_ORDER if m in rows_by_model and rows_by_model[m]]
    if len(models) < 2:
        return {"models": models, "skipped": True}
    common = None
    for m in models:
        common = set(rows_by_model[m]) if common is None else (common & set(rows_by_model[m]))
    subsets = sorted({r["subset"] for m in models for r in rows_by_model[m].values()})
    result: dict = {"models": models}
    for subset in subsets + ["ALL"]:
        ids = [i for i in sorted(common)
               if subset == "ALL" or rows_by_model[models[0]][i].get("subset") == subset]
        if not ids:
            continue
        entry: dict = {"n": len(ids), "model_means": {}, "pairs": {}}
        for m in models:
            vals = [rows_by_model[m][i][field] for i in ids]
            entry["model_means"][m] = statistics.fmean(vals)
        for (y, x) in available_pairs(models):
            entry["pairs"][pair_label(y, x)] = pair_stats(
                rows_by_model[x], rows_by_model[y], field, ids, n_boot)
        result[subset] = entry
    return result


def markdown(clip: dict, vqa: dict | None, out: Path, n_boot: int) -> None:
    lines = ["# E3 / E1 general T2I: three-way comparison (CLIP-T, ViT-L/14)", "",
             f"Paired per-item diff = column model - row model; bootstrap 95% CI "
             f"({n_boot} resamples, seed 1234). Positive diff = better text-image "
             f"alignment. SIG = CI excludes 0.", ""]
    cols = [pair_label(y, x) for (y, x) in available_pairs(clip.get("models", []))]

    def section(title: str, data: dict) -> None:
        if data.get("skipped"):
            lines.extend([f"## {title}: skipped (need >= 2 models)", ""])
            return
        lines.extend([f"## {title}", "",
                      "| subset | n | " + " | ".join(f"{m} mean" for m in data["models"])
                      + " | " + " | ".join(cols) + " |",
                      "|" + "---|" * (2 + len(data["models"]) + len(cols))])
        for subset in [k for k in data if k not in ("models", "skipped")]:
            e = data[subset]
            pcells = [fmt_entry(e["pairs"][c]) if c in e["pairs"] else "n/a" for c in cols]
            mcells = [f"{e['model_means'][m]:.4f}" if m in e["model_means"] else "n/a"
                      for m in data["models"]]
            lines.append(f"| {subset} | {e['n']} | " + " | ".join(mcells) + " | "
                         + " | ".join(pcells) + " |")
        lines.append("")

    section("CLIP-T cosine", clip)
    if vqa:
        section("VQA faithfulness (match rate)", vqa)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER)
    parser.add_argument("--field", default="clip_cos")
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()

    rows: dict[str, dict[str, dict]] = {}
    for m in args.models:
        r = load_rows(args.scores_dir / f"clipT_{m}.jsonl", key="item_id")
        if r:
            rows[m] = r
        else:
            print(f"[warn] no rows: clipT_{m}.jsonl under {args.scores_dir}")
    clip_result = analyze_subsets(rows, args.field, args.bootstrap)

    vqa_result = None
    vqa_rows: dict[str, dict[str, dict]] = {}
    for m in args.models:
        r = load_rows(args.scores_dir / f"vqa_{m}.jsonl", key="item_id")
        if r:
            for row in r.values():
                row["vqa_match_f"] = 1.0 if row.get("vqa_match") else 0.0
            vqa_rows[m] = r
    if len(vqa_rows) >= 2:
        vqa_result = analyze_subsets(vqa_rows, "vqa_match_f", args.bootstrap)

    summary = {
        "experiment": "e3_e1_general_t2i_threeway",
        "sampling": {"seed": 1234, "num_timesteps": 25, "cfg_text": 4.0, "cfg_img": 2.0,
                     "shift": 3.0, "resolution": 1024,
                     "prompts": "T2I-CompBench 3 subsets x 150"},
        "bootstrap_resamples": args.bootstrap,
        "clipT": clip_result,
        "vqa": vqa_result,
    }
    args.scores_dir.mkdir(parents=True, exist_ok=True)
    (args.scores_dir / "stats_threeway.json").write_text(json.dumps(summary, indent=1),
                                                         encoding="utf-8")
    markdown(clip_result, vqa_result, args.scores_dir / "comparison_table.md", args.bootstrap)
    print(json.dumps({"event": "e1_stats_done",
                      "json": str(args.scores_dir / "stats_threeway.json")}))
    all_entry = clip_result.get("ALL", {}).get("pairs", {}).get("v6-final991")
    if all_entry:
        print(json.dumps({"pair": "v6-final991", "subset": "ALL", **all_entry}, indent=2))


if __name__ == "__main__":
    main()
