#!/usr/bin/env python3
"""E1 paired statistics: base vs final991 per-item paired differences + bootstrap 95% CI.

Method mirrors final991_eval_v1 (Module A): per-item paired diff
(final991 - base), 20000 bootstrap resamples of the item set, 2.5/97.5
percentile CI. Paired win/loss/tie counts and optional Wilcoxon signed-rank
(scipy if importable). VQA match-rate comparison is merged when
scores/vqa_*.jsonl exist.

Outputs:
  scores/stats_summary.json
  scores/comparison_table.md
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_scores(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            rows[rec["item_id"]] = rec
    return rows


def paired_bootstrap(diffs: list[float], n_boot: int, seed: int = 1234) -> tuple[float, float, float]:
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


def analyze(metric_name: str, base_rows: dict[str, dict], final_rows: dict[str, dict],
            field: str, subsets: list[str], n_boot: int) -> dict:
    result: dict = {"metric": metric_name}
    overall: dict[str, list[float]] = {}
    for subset in subsets + ["ALL"]:
        ids = [i for i in base_rows if i in final_rows and (subset == "ALL" or base_rows[i]["subset"] == subset)]
        diffs = [final_rows[i][field] - base_rows[i][field] for i in ids]
        if not diffs:
            continue
        lo, hi, mean_diff = paired_bootstrap(diffs, n_boot)
        wins = sum(1 for d in diffs if d > 0)
        losses = sum(1 for d in diffs if d < 0)
        ties = len(diffs) - wins - losses
        entry = {
            "n": len(diffs),
            "base_mean": statistics.fmean([base_rows[i][field] for i in ids]),
            "final991_mean": statistics.fmean([final_rows[i][field] for i in ids]),
            "mean_diff_final_minus_base": mean_diff,
            "ci95": [lo, hi],
            "significant": (lo > 0) or (hi < 0),
            "wins/losses/ties": [wins, losses, ties],
        }
        try:
            from scipy.stats import wilcoxon

            if wins + losses > 0:
                stat, p = wilcoxon([final_rows[i][field] for i in ids], [base_rows[i][field] for i in ids])
                entry["wilcoxon_p"] = float(p)
        except Exception:
            pass
        result[subset] = entry
        overall[subset] = diffs
    return result


def markdown_table(clip_result: dict, vqa_result: dict | None, out: Path) -> None:
    lines = ["# E1 General T2I: base vs final991 (LoRA direct-load)", "",
             f"Metric primary: CLIP-T cosine (CLIP ViT-L/14). Paired per-item diff = final991 - base; "
             f"bootstrap 95% CI ({20000} resamples). Positive diff means final991 better.", ""]
    header = "| subset | n | base | final991 | diff (95% CI) | win/loss/tie | sig? |"
    lines += [header, "|---|---|---|---|---|---|---|"]
    for subset, e in clip_result.items():
        if subset == "metric":
            continue
        lines.append(
            f"| {subset} | {e['n']} | {e['base_mean']:.4f} | {e['final991_mean']:.4f} "
            f"| {e['mean_diff_final_minus_base']:+.4f} [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] "
            f"| {e['wins/losses/ties'][0]}/{e['wins/losses/ties'][1]}/{e['wins/losses/ties'][2]} "
            f"| {'YES' if e['significant'] else 'no'} |")
    lines.append("")
    if vqa_result:
        lines += ["## VQA faithfulness (Qwen3-VL, capped subset)", "",
                  "| subset | n | base acc | final991 acc | diff |", "|---|---|---|---|---|"]
        for subset, e in vqa_result.items():
            if subset == "metric":
                continue
            lines.append(f"| {subset} | {e['n']} | {e['base_mean']:.3f} | {e['final991_mean']:.3f} "
                         f"| {e['mean_diff_final_minus_base']:+.3f} |")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, default=HERE / "../outputs/e1_general_t2i/scores")
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()

    base_rows = load_scores(args.scores_dir / "clipT_base.jsonl")
    final_rows = load_scores(args.scores_dir / "clipT_final991.jsonl")
    if not base_rows or not final_rows:
        raise SystemExit(f"missing CLIP-T score files under {args.scores_dir}")
    subsets = sorted({r["subset"] for r in base_rows.values()})
    clip_result = analyze("clip_cos", base_rows, final_rows, "clip_cos", subsets, args.bootstrap)

    vqa_result = None
    vqa_base = load_scores(args.scores_dir / "vqa_base.jsonl")
    vqa_final = load_scores(args.scores_dir / "vqa_final991.jsonl")
    if vqa_base and vqa_final:
        # binary metric: paired diff of 0/1 match
        for rows in (vqa_base, vqa_final):
            for r in rows.values():
                r["vqa_match_f"] = 1.0 if r["vqa_match"] else 0.0
        vqa_result = analyze("vqa_match", vqa_base, vqa_final, "vqa_match_f", subsets, args.bootstrap)

    summary = {
        "experiment": "e1_general_t2i",
        "comparison": "final991 (LoRA direct-load, train2973 latest.pt) vs base (zero-LoRA)",
        "sampling": {"seed": 1234, "num_timesteps": 25, "cfg_text": 4.0, "cfg_img": 2.0, "shift": 3.0, "resolution": 1024},
        "bootstrap_resamples": args.bootstrap,
        "clipT": clip_result,
        "vqa": vqa_result,
    }
    (args.scores_dir / "stats_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_table(clip_result, vqa_result, args.scores_dir / "comparison_table.md")
    print(json.dumps({"event": "stats_done", "out": str(args.scores_dir / "stats_summary.json")}))
    print(json.dumps(clip_result.get("ALL", {}), indent=2))


if __name__ == "__main__":
    main()
