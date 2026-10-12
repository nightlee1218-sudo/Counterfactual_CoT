#!/usr/bin/env python3
"""E2 analysis: per-layer metric tables + base vs final991 paired bootstrap stats.

Reads probe_manifest.jsonl from both models, builds:
  scores/e2_layer_table.md          per-layer metric table (both models, diff + CI)
  scores/e2_stats_summary.json      machine-readable full stats
  scores/e2_layer_curves.csv        layer-wise means for plotting

Paired method mirrors final991_eval_v1 (Module A): per-pair diff
(final991 - base) per layer, 20000 bootstrap resamples, 2.5/97.5 percentile CI.

Usage:
  python3 analyze.py --probe-root ../outputs/e2_token_probe
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent

# metrics pulled into the layer table; positive = final991 stronger binding
METRICS = [
    ("norm_ratio_vae", "P1 vae density ratio in/out (prefill)"),
    ("norm_ratio_vit", "P1 vit density ratio in/out (prefill)"),
    ("norm_ratio_visual", "P1 visual density ratio in/out (prefill)"),
    ("mass_visual", "P1 attn mass on all visual keys (prefill)"),
    ("flow_log2_contrast", "P2 log2 in/out contrast (flow)"),
    ("flow_mass_in_frac", "P2 in-mask mass fraction (flow)"),
    ("flow_auroc", "P2 AUROC map vs mask (flow)"),
    ("margin_vae", "P3 hidden margin cos (vae)"),
    ("margin_vit", "P3 hidden margin cos (vit)"),
    ("margin_lat", "P3 hidden margin cos (latent)"),
]


def load_manifest(path: Path) -> dict[str, dict]:
    """pair_id -> {'per_layer': {L: row}, ...} for successful pairs."""
    pairs: dict[str, dict] = {}
    if not path.exists():
        raise SystemExit(f"missing probe manifest: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") == "ok":
            pairs[rec["pair_id"]] = rec
    if not pairs:
        raise SystemExit(f"no successful pairs in {path}")
    return pairs


def paired_bootstrap(diffs: list[float], n_boot: int = 20000, seed: int = 1234):
    rng = random.Random(seed)
    n = len(diffs)
    mean = statistics.fmean(diffs)
    boots = []
    for _ in range(n_boot):
        s = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.fmean(s))
    boots.sort()
    return mean, boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path,
                        default=HERE / "../outputs/e2_token_probe")
    parser.add_argument("--out-dir", type=Path,
                        default=HERE / "../outputs/e2_token_probe/scores")
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()

    base = load_manifest(args.probe_root / "probe_base" / "probe_manifest.jsonl")
    final = load_manifest(args.probe_root / "probe_final991" / "probe_manifest.jsonl")
    common = sorted(set(base) & set(final))
    print(f"paired pairs: {len(common)} (base {len(base)}, final991 {len(final)})")
    layers = sorted({int(L) for rec in final.values() for L in rec["per_layer"]})
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-layer, per-metric stats --------------------------------------
    stats: dict[str, dict[int, dict]] = {m: {} for m, _ in METRICS}
    for metric, _desc in METRICS:
        for layer in layers:
            b_vals, f_vals = [], []
            for pid in common:
                b = base[pid]["per_layer"].get(str(layer), {}).get(metric)
                f = final[pid]["per_layer"].get(str(layer), {}).get(metric)
                if b is not None and f is not None:
                    b_vals.append(b)
                    f_vals.append(f)
            if len(b_vals) < 5:
                continue
            diffs = [f - b for f, b in zip(f_vals, b_vals)]
            mean_d, lo, hi = paired_bootstrap(diffs, args.bootstrap)
            wins = sum(1 for d in diffs if d > 0)
            stats[metric][layer] = {
                "n": len(diffs),
                "base_mean": statistics.fmean(b_vals),
                "final_mean": statistics.fmean(f_vals),
                "diff_mean": mean_d, "ci95": [lo, hi],
                "significant": bool(lo > 0 or hi < 0),
                "wins": wins, "losses": len(diffs) - wins,
            }

    # ---- aggregate views ---------------------------------------------------
    summary = {
        "experiment": "e2_token_probe",
        "comparison": "final991 (LoRA direct-load) vs base (zero-LoRA)",
        "n_pairs": len(common),
        "bootstrap": args.bootstrap,
        "metrics": {},
    }
    lines = ["# E2 token-localization probe: base vs final991", "",
             f"Paired items: {len(common)} val pairs. Diff = final991 - base; "
             f"bootstrap 95% CI ({args.bootstrap} resamples). All layers scanned "
             f"(early/mid/late included; not just the last layer).", ""]
    for metric, desc in METRICS:
        if not stats[metric]:
            continue
        rows = stats[metric]
        best_layer = max(rows, key=lambda L: rows[L]["diff_mean"])
        best = rows[best_layer]
        summary["metrics"][metric] = {
            "desc": desc,
            "best_layer": best_layer, "best": best,
            "layers": {str(L): v for L, v in rows.items()},
        }
        lines += [f"## {metric}", f"*{desc}*", "",
                  f"- best layer (largest diff): **L{best_layer}** "
                  f"base {best['base_mean']:.4f} -> final {best['final_mean']:.4f}, "
                  f"diff {best['diff_mean']:+.4f} [{best['ci95'][0]:+.4f}, {best['ci95'][1]:+.4f}]"
                  f"{' **SIG**' if best['significant'] else ''} "
                  f"({best['wins']}/{best['losses']} wins/losses, n={best['n']})", ""]
        lines += ["| layer | n | base | final991 | diff | 95% CI | sig | w/l |",
                  "|---|---|---|---|---|---|---|---|"]
        for L in sorted(rows):
            r = rows[L]
            lines.append(
                f"| L{L} | {r['n']} | {r['base_mean']:.4f} | {r['final_mean']:.4f} "
                f"| {r['diff_mean']:+.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] "
                f"| {'YES' if r['significant'] else ''} | {r['wins']}/{r['losses']} |")
        lines.append("")

    # ---- layer curves csv ---------------------------------------------------
    csv_lines = ["metric,layer,n,base_mean,final_mean,diff,ci_lo,ci_hi,significant"]
    for metric, _ in METRICS:
        for L in sorted(stats[metric]):
            r = stats[metric][L]
            csv_lines.append(
                f"{metric},{L},{r['n']},{r['base_mean']:.6f},{r['final_mean']:.6f},"
                f"{r['diff_mean']:.6f},{r['ci95'][0]:.6f},{r['ci95'][1]:.6f},"
                f"{int(r['significant'])}")
    (args.out_dir / "e2_layer_curves.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    (args.out_dir / "e2_stats_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "e2_layer_table.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out_dir / 'e2_layer_table.md'}")

    # console digest
    for metric in ("norm_ratio_visual", "flow_log2_contrast", "flow_auroc", "margin_vae"):
        rows = stats.get(metric) or {}
        if not rows:
            continue
        best_layer = max(rows, key=lambda L: rows[L]["diff_mean"])
        r = rows[best_layer]
        print(f"[{metric}] best L{best_layer}: {r['base_mean']:.4f} -> {r['final_mean']:.4f} "
              f"(diff {r['diff_mean']:+.4f}, CI [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}], "
              f"sig={r['significant']})")


if __name__ == "__main__":
    main()
