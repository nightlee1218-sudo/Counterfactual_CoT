#!/usr/bin/env python3
"""Paired BASE vs Gold40 analysis for mini-eval v2 (official evaluators).

Input layout (produced by run_mini_eval_v2.sh)::

    ROOT/scores/{bench}_{variant}[_official]_s{seed}.jsonl

Everything is normalised to 0-1 before comparison so a single judgement rule
applies across benchmarks:
    gedit    : VIE-Score SC / 10           (sub: edit_success, overedit_penalty)
    wise     : WiScore / 2                 (sub: consistency, realism, aesthetic)
    geneval  : official correct flag       (sub: per tag)

Statistics
----------
* paired per-item delta, clustered bootstrap over ``item_id`` (seeds collapse
  into their item) -> 95% CI
* Benjamini-Hochberg FDR across the primary test family
* sigma_seed: how much the delta moves between seeds (sampling noise floor)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

NAME_RE = re.compile(r"^(gedit|wise|geneval)_(?P<variant>.+?)_(?P<official>official_)?s(?P<seed>\d+)\.jsonl$")

BENCH_LABEL = {"gedit": "GEdit-36", "wise": "WISE-36", "geneval": "GenEval-36"}
VARIANT_LABEL = {
    "base": "BASE",
    "gold40": "Gold40-V5",
    "gold40_lora0": "Gold40-LoRA0",
    "final991": "Final991-V5",
}

PRIMARY = {"gedit": "sc", "wise": "wiscore", "geneval": "correct"}
SCALE = {"gedit": 10.0, "wise": 2.0, "geneval": 1.0}
SUBSCORES = {
    "gedit": [("sc_edit_success", "编辑成功", 10.0), ("sc_overedit_penalty", "低过度编辑", 10.0)],
    "wise": [
        ("consistency", "一致性", 2.0),
        ("realism", "真实感", 2.0),
        ("aesthetic_quality", "美学质量", 2.0),
    ],
    "geneval": [],
}
BREAKDOWN_FIELD = {"gedit": "task_type", "wise": "group", "geneval": "tag"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalise(bench: str, row: dict[str, Any], key: str, scale: float) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    value = float(value)
    if bench == "wise" and value == 9.9:
        return None
    return value / scale


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(pairs: dict[str, tuple[float, float]], num_samples: int, seed: int) -> dict[str, float]:
    """Cluster bootstrap over items; pairs[item] = (base_value, gold_value)."""
    keys = sorted(pairs)
    rng = random.Random(seed)
    observed = mean([pairs[k][1] - pairs[k][0] for k in keys])
    draws: list[float] = []
    for _ in range(num_samples):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        draws.append(mean([pairs[k][1] - pairs[k][0] for k in pick]))
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return {"observed": observed, "ci_low": low, "ci_high": high, "n_items": len(keys)}


def benjamini_hochberg(p_values: dict[str, float], fdr: float) -> dict[str, bool]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, bool] = {}
    max_k = 0
    for i, (name, p) in enumerate(items, start=1):
        if p <= fdr * i / m:
            max_k = i
    for i, (name, _p) in enumerate(items, start=1):
        out[name] = i <= max_k
    return out


def ci_to_p(delta: float, ci_low: float, ci_high: float) -> float:
    """Two-sided p-value from a bootstrap CI, using a normal approximation.

    A 95% CI spans 2*1.96 sigma, so sigma = (ci_high - ci_low) / 3.92.
    """
    sigma = (ci_high - ci_low) / 3.92
    if sigma <= 0:
        return 1e-6
    z = abs(delta) / sigma
    return max(1e-6, math.erfc(z / math.sqrt(2.0)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method-a", default="base")
    parser.add_argument("--method-b", default="gold40")
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--fdr", type=float, default=0.1)
    parser.add_argument("--min-effect", type=float, default=0.03)
    args = parser.parse_args()

    scores_dir = args.root / "scores"
    if not scores_dir.is_dir():
        raise RuntimeError(f"missing scores dir: {scores_dir}")

    # (bench, variant, official, seed) -> {item_id: row}
    data: dict[tuple[str, str, bool, int], dict[str, dict[str, Any]]] = {}
    for path in sorted(scores_dir.glob("*.jsonl")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        bench = match.group(1)
        variant = match.group("variant")
        official = bool(match.group("official"))
        seed = int(match.group("seed"))
        data[(bench, variant, official, seed)] = {r["id"]: r for r in read_jsonl(path)}

    if not data:
        raise RuntimeError("no score files matched; check the naming convention")

    def item_value(bench: str, variant: str, official: bool, seed: int, item: str,
                   key: str | None = None, scale: float | None = None) -> float | None:
        rows = data.get((bench, variant, official, seed))
        if rows is None or item not in rows:
            return None
        key = key or PRIMARY[bench]
        scale = scale if scale is not None else SCALE[bench]
        return normalise(bench, rows[item], key, scale)

    report: dict[str, Any] = {"per_bench": {}, "capability": {}, "breakdown": {}, "seed_noise": {}}
    tests: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}

    for bench in ("gedit", "wise", "geneval"):
        seeds = sorted({s for (b, _v, _o, s) in data if b == bench and not _o})
        if not seeds:
            continue
        a_rows = data.get((bench, args.method_a, False, seeds[0]))
        b_rows = data.get((bench, args.method_b, False, seeds[0]))
        if not a_rows or not b_rows:
            print(f"[warn] skipping {bench}: missing {args.method_a} or {args.method_b}", flush=True)
            continue
        items = sorted(set(a_rows) & set(b_rows))
        if not items:
            continue

        # --- primary, collapsing seeds into the item cluster ---
        pairs: dict[str, tuple[float, float]] = {}
        for item in items:
            a_vals = [v for s in seeds if (v := item_value(bench, args.method_a, False, s, item)) is not None]
            b_vals = [v for s in seeds if (v := item_value(bench, args.method_b, False, s, item)) is not None]
            if a_vals and b_vals:
                pairs[item] = (mean(a_vals), mean(b_vals))

        rng = random.Random(args.seed + len(bench))
        keys = sorted(pairs)
        observed = mean([pairs[k][1] - pairs[k][0] for k in keys])
        draws = []
        for _ in range(args.num_samples):
            pick = [keys[rng.randrange(len(keys))] for _ in keys]
            draws.append(mean([pairs[k][1] - pairs[k][0] for k in pick]))
        draws.sort()
        ci_low = draws[int(0.025 * (len(draws) - 1))]
        ci_high = draws[int(0.975 * (len(draws) - 1))]

        wins = sum(1 for k in keys if pairs[k][1] > pairs[k][0])
        losses = sum(1 for k in keys if pairs[k][1] < pairs[k][0])
        ties = len(keys) - wins - losses
        report["per_bench"][bench] = {
            "n_items": len(keys),
            "seeds": seeds,
            "a": round(mean([pairs[k][0] for k in keys]), 4),
            "b": round(mean([pairs[k][1] for k in keys]), 4),
            "delta": round(observed, 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "win_rate": round(wins / len(keys), 4),
            "wins": wins, "ties": ties, "losses": losses,
        }
        tests[bench] = {"ci_low": ci_low, "ci_high": ci_high, "delta": observed}

        # --- seed noise (only meaningful when >1 seed) ---
        if len(seeds) > 1:
            per_seed_delta: dict[int, list[float]] = defaultdict(list)
            for seed in seeds:
                for item in keys:
                    a = item_value(bench, args.method_a, False, seed, item)
                    b = item_value(bench, args.method_b, False, seed, item)
                    if a is not None and b is not None:
                        per_seed_delta[seed].append(b - a)
            seed_means = {s: mean(v) for s, v in per_seed_delta.items()}
            itemwise_sd = []
            for item in keys:
                vals = []
                for seed in seeds:
                    a = item_value(bench, args.method_a, False, seed, item)
                    b = item_value(bench, args.method_b, False, seed, item)
                    if a is not None and b is not None:
                        vals.append(b - a)
                if len(vals) > 1:
                    mu = mean(vals)
                    itemwise_sd.append(math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)))
            report["seed_noise"][bench] = {
                "per_seed_delta": {str(s): round(v, 4) for s, v in seed_means.items()},
                "itemwise_sd_mean": round(mean(itemwise_sd), 4) if itemwise_sd else None,
            }

        # --- sub-scores ---
        for key, label, scale in SUBSCORES[bench]:
            sub_pairs = {}
            for item in keys:
                a_vals = [v for s in seeds if (v := item_value(bench, args.method_a, False, s, item, key, scale)) is not None]
                b_vals = [v for s in seeds if (v := item_value(bench, args.method_b, False, s, item, key, scale)) is not None]
                if a_vals and b_vals:
                    sub_pairs[item] = (mean(a_vals), mean(b_vals))
            if not sub_pairs:
                continue
            stat = bootstrap_ci(sub_pairs, args.num_samples, args.seed)
            name = f"{bench}::{key}"
            report["capability"][name] = {
                "label": label, "bench": bench, "scale": scale,
                "a": round(mean([sub_pairs[k][0] for k in sub_pairs]), 4),
                "b": round(mean([sub_pairs[k][1] for k in sub_pairs]), 4),
                "delta": round(stat["observed"], 4),
                "ci_low": round(stat["ci_low"], 4),
                "ci_high": round(stat["ci_high"], 4),
                "raw_delta": round(stat["observed"] * scale, 4),
            }
            tests[name] = {"ci_low": stat["ci_low"], "ci_high": stat["ci_high"], "delta": stat["observed"]}

        # --- breakdown by task_type / group / tag ---
        field = BREAKDOWN_FIELD[bench]
        groups: dict[str, list[str]] = defaultdict(list)
        for item in keys:
            row = a_rows.get(item) or {}
            group = row.get(field) or (row.get("meta") or {}).get(field)
            if group is None:
                continue
            groups[str(group)].append(item)
        for group, group_items in sorted(groups.items()):
            sub_pairs = {}
            for item in group_items:
                a_vals = [v for s in seeds if (v := item_value(bench, args.method_a, False, s, item)) is not None]
                b_vals = [v for s in seeds if (v := item_value(bench, args.method_b, False, s, item)) is not None]
                if a_vals and b_vals:
                    sub_pairs[item] = (mean(a_vals), mean(b_vals))
            if not sub_pairs:
                continue
            stat = bootstrap_ci(sub_pairs, args.num_samples, args.seed)
            name = f"{bench}::{field}={group}"
            report["breakdown"][name] = {
                "bench": bench, "group": group, "n_items": len(sub_pairs),
                "a": round(mean([sub_pairs[k][0] for k in sub_pairs]), 4),
                "b": round(mean([sub_pairs[k][1] for k in sub_pairs]), 4),
                "delta": round(stat["observed"], 4),
                "ci_low": round(stat["ci_low"], 4),
                "ci_high": round(stat["ci_high"], 4),
            }
            tests[name] = {"ci_low": stat["ci_low"], "ci_high": stat["ci_high"], "delta": stat["observed"]}

    p_values = {name: ci_to_p(t["delta"], t["ci_low"], t["ci_high"]) for name, t in tests.items()}
    passed = benjamini_hochberg(p_values, args.fdr)

    def verdict(name: str, t: dict[str, Any]) -> str:
        if t["ci_low"] > 0 and t["delta"] >= args.min_effect and passed.get(name, False):
            return "显著提升"
        if t["ci_high"] < 0 and passed.get(name, False):
            return "显著退化"
        if t["ci_low"] > 0 or t["ci_high"] < 0:
            return "方向一致但未过 FDR"
        return "无定论"

    for name, t in tests.items():
        t["p_value"] = round(p_values[name], 6)
        t["significant_fdr"] = bool(passed.get(name, False))
        t["verdict"] = verdict(name, t)

    # ---------------- ablation ----------------
    ablation: dict[str, Any] = {}
    for bench in ("gedit",):
        for official in (False, True):
            seeds = sorted({s for (b, _v, o, s) in data if b == bench and o == official})
            if not seeds:
                continue
            seed = seeds[0]
            variants = sorted({v for (b, v, o, s) in data if b == bench and o == official and s == seed})
            if args.method_a not in variants or args.method_b not in variants:
                continue
            items = sorted(set(data[(bench, args.method_a, official, seed)]))
            rows = {}
            for variant in variants:
                vals = []
                for item in items:
                    v = item_value(bench, variant, official, seed, item)
                    if v is not None:
                        vals.append(v)
                rows[variant] = round(mean(vals), 4)
            key = "official_sampling" if official else "default_sampling"
            ablation[f"{bench}::{key}"] = {
                "seed": seed,
                "variants": {VARIANT_LABEL.get(v, v): x for v, x in rows.items()},
                "delta_gold40_minus_base": round(
                    rows.get(args.method_b, 0.0) - rows.get(args.method_a, 0.0), 4
                ),
            }
    report["ablation"] = ablation

    # ---------------- render ----------------
    lines = ["# Mini-Eval v2: BASE vs Gold40-V5 (official evaluators)", ""]
    lines.append("| Bench | n | BASE | Gold40-V5 | Δ | 95% CI | 胜出 | W/T/L | p | 判定 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for bench in ("gedit", "wise", "geneval"):
        if bench not in report["per_bench"]:
            continue
        r = report["per_bench"][bench]
        t = tests[bench]
        lines.append(
            f"| {BENCH_LABEL[bench]} | {r['n_items']} | {r['a']:.3f} | {r['b']:.3f} | "
            f"{r['delta']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['win_rate']*100:.0f}% | "
            f"{r['wins']}/{r['ties']}/{r['losses']} | {t['p_value']:.3f} | {t['verdict']} |"
        )

    lines += ["", "## 子分数（capability）", "", "| Bench | 能力 | BASE | Gold40-V5 | Δ | 原始尺度Δ | 判定 |", "|---|---|---|---|---|---|---|"]
    for name, r in report["capability"].items():
        t = tests[name]
        lines.append(
            f"| {BENCH_LABEL[r['bench']]} | {r['label']} | {r['a']:.3f} | {r['b']:.3f} | "
            f"{r['delta']:+.3f} | {r['raw_delta']:+.2f} | {t['verdict']} |"
        )

    lines += ["", "## 分类型拆分", "", "| Bench | 类型 | n | BASE | Gold40-V5 | Δ | 95% CI | 判定 |", "|---|---|---|---|---|---|---|---|"]
    for name, r in report["breakdown"].items():
        t = tests[name]
        lines.append(
            f"| {BENCH_LABEL[r['bench']]} | {r['group']} | {r['n_items']} | {r['a']:.3f} | {r['b']:.3f} | "
            f"{r['delta']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {t['verdict']} |"
        )

    if report["seed_noise"]:
        lines += ["", "## 采样噪声（跨 seed）", "", "| Bench | 各 seed 的 Δ | item 级 SD |", "|---|---|---|"]
        for bench, r in report["seed_noise"].items():
            seeds_str = ", ".join(f"s{s}: {v:+.3f}" for s, v in r["per_seed_delta"].items())
            lines.append(f"| {BENCH_LABEL[bench]} | {seeds_str} | {r['itemwise_sd_mean']} |")

    if ablation:
        lines += ["", "## 消融（GEdit，单 seed）", "", "| 采样设置 | 各变体均分 | Gold40−BASE |", "|---|---|---|"]
        for key, r in ablation.items():
            variants_str = ", ".join(f"{k}={v:.3f}" for k, v in r["variants"].items())
            lines.append(f"| {key.split('::')[1]} (seed {r['seed']}) | {variants_str} | {r['delta_gold40_minus_base']:+.3f} |")

    lines += [
        "",
        "## 判定规则（归一化 0-1 尺度）",
        f"- 显著提升: CI 下限 > 0 且 Δ >= {args.min_effect} 且通过 BH FDR {args.fdr}",
        "- 显著退化: CI 上限 < 0 且通过 BH FDR",
        "- 方向一致但未过 FDR: CI 不跨 0 但未通过多重比较校正",
        "- 无定论: CI 跨 0",
        "",
        "## 评测器",
        "- GEdit: 官方 VIE-Score（仅 SC 分支，0-10，除以 10 归一）",
        "- WISE: 仓库官方 rubric（0-2，WiScore = 0.7C+0.2R+0.1A，除以 2 归一）",
        "- GenEval: 官方 Mask2Former + CLIP 确定性评测（0/1）",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps({**report, "tests": tests}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
