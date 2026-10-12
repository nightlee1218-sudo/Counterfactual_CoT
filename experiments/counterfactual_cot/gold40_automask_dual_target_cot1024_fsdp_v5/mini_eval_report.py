#!/usr/bin/env python3
"""Build the paired BASE vs Gold40-V5 delta report for the mini benchmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CAPABILITY = {
    "gedit": {
        "edit_success": "目标定位/编辑执行",
        "target_grounding": "单目标编辑(正确作用对象)",
        "preservation": "多对象保持",
        "locality": "局部性(不改无关区域)",
    },
    "wise": {
        "prompt_following": "指令执行/prompt following",
        "factual_correctness": "事实/常识正确性",
        "multi_element": "多元素场景",
    },
    "geneval": {
        "object_correctness": "类别正确",
        "count_correctness": "计数",
        "spatial_correctness": "空间关系",
        "attribute_correctness": "属性修改",
    },
}
BENCH_LABEL = {"gedit": "GEdit-mini", "wise": "WISE-mini", "geneval": "GenEval-mini"}


def load(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = row
    return out


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    per_bench: dict[str, Any] = {}
    capability: dict[str, dict[str, float]] = {}
    win_stats: dict[str, dict[str, float]] = {}

    for bench in ("gedit", "wise", "geneval"):
        base = load(args.root / f"{bench}_base_scores.jsonl")
        gold = load(args.root / f"{bench}_gold40_scores.jsonl")
        if not base and not gold:
            continue
        keys = list(CAPABILITY[bench])
        ids = sorted(set(base) & set(gold))

        def overall(row: dict[str, Any]) -> float:
            return mean([float(row[k]) for k in keys])

        base_overall = [overall(base[i]) for i in ids]
        gold_overall = [overall(gold[i]) for i in ids]
        wins = sum(1 for i in ids if overall(gold[i]) > overall(base[i]))
        losses = sum(1 for i in ids if overall(gold[i]) < overall(base[i]))
        ties = len(ids) - wins - losses
        per_bench[bench] = {
            "n": len(ids),
            "base": round(mean(base_overall), 4),
            "gold40": round(mean(gold_overall), 4),
            "delta": round(mean(gold_overall) - mean(base_overall), 4),
            "win_rate": round(wins / len(ids), 4) if ids else 0.0,
            "wins": wins, "losses": losses, "ties": ties,
        }
        win_stats[bench] = {"wins": wins, "losses": losses, "ties": ties}
        for key in keys:
            label = CAPABILITY[bench][key]
            capability[label] = {
                "base": round(mean([float(base[i][key]) for i in ids]), 4),
                "gold40": round(mean([float(gold[i][key]) for i in ids]), 4),
                "delta": round(
                    mean([float(gold[i][key]) for i in ids]) - mean([float(base[i][key]) for i in ids]), 4
                ),
                "bench": bench,
            }

    def verdict(row: dict[str, Any]) -> str:
        if row["delta"] >= 0.10 and row["win_rate"] >= 0.60:
            return "初步有提升"
        if row["delta"] <= -0.10:
            return "可能退化"
        return "暂无明确提升"

    lines = ["# Mini-Eval: BASE vs Gold40-V5 (step960)", ""]
    lines.append("| Bench | n | BASE | Gold40-V5 | Δ | Gold40胜出比例 | W/T/L | 判定 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for bench, row in per_bench.items():
        s = win_stats[bench]
        lines.append(
            f"| {BENCH_LABEL[bench]} | {row['n']} | {row['base']:.3f} | {row['gold40']:.3f} | "
            f"{row['delta']:+.3f} | {row['win_rate']*100:.0f}% | {s['wins']}/{s['ties']}/{s['losses']} | {verdict(row)} |"
        )
    lines += ["", "## 能力拆分", "", "| 能力 | Bench | BASE | Gold40-V5 | Δ |", "|---|---|---|---|---|"]
    for label, row in capability.items():
        lines.append(f"| {label} | {BENCH_LABEL[row['bench']]} | {row['base']:.3f} | {row['gold40']:.3f} | {row['delta']:+.3f} |")
    lines += [
        "",
        "## 判定规则",
        "- 初步有提升: Δ >= +0.10 且 Gold40 胜出比例 >= 60%",
        "- 可能退化: Δ <= -0.10",
        "- 其他: 暂无明确提升",
        "",
        "## 配置锁定",
        "seed=1234, resolution=1024, num_timesteps=25, cfg_text=4.0, cfg_img=2.0, timestep_shift=3.0;",
        "BASE 与 Gold40 使用同一推理脚本，唯一差异为是否 strict 加载 latest.pt 的 trainable_state_dict。",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out.with_suffix(".json")).write_text(
        json.dumps({"per_bench": per_bench, "capability": capability}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
