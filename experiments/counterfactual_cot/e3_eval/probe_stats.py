#!/usr/bin/env python3
"""E2 token-probe three-way stats: best-layer paired diffs + late-step AUROC.

Reads probe_manifest.jsonl from the existing E2 runs
(outputs/e2_token_probe/probe_{base,final991}) and the new E3 v6 probe
(outputs/e3_eval/probe_v6, produced through model_shim.py -> probe.py with
the same --probe-steps 0 11 23 defaults), then computes the E2 recipe
(per-pair diff, 20000 bootstrap resamples seed 1234, 2.5/97.5 CI, SIG iff
CI excludes 0) for all three model pairs on every per-layer metric.

Flow metrics exist only at the captured (layer, step) combos — with the
probe.py defaults step_idx is shared between layer and step, so the on-disk
keys are flow_map_L0_S0 / L11_S11 / L23_S23.  "Late-step AUROC" below is the
flow AUROC at the LARGEST captured step (t ~ 0.115, near the end of the
denoising trajectory), read from per_layer[late_layer]["flow_auroc"] of the
manifests; --verify-npz recomputes it from the per-pair npz maps
(head-mean of flow_map_L{L}_S{late}, AUROC vs the mask flags rebuilt from
the val records) as an f16-rounding cross-check.  The auroc/mask helpers
are verbatim copies from e2_token_probe/probe.py so this module imports
without the heavy BAGEL stack.

Outputs (under --out-dir):
  probe_stats_threeway.json
  probe_table.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from indomain_stats import available_pairs, fmt_entry, ordered_models, pair_label, pair_stats

HERE = Path(__file__).resolve().parent
CFCOT = HERE.parent
PROBE_ROOT = CFCOT / "outputs/e2_token_probe"
DEFAULT_OUT_DIR = CFCOT / "outputs/e3_eval/probe_stats"
DEFAULT_VAL = CFCOT / "final991_automask_v5_data/artifacts/splits/val.jsonl"

# verbatim metric list from e2_token_probe/analyze.py
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

IMAGE_SIZE = 1024
LAT_DOWNSAMPLE = 16  # probe.py: lat_side = image_size // model.latent_downsample


# ---------------------------------------------------------------------------
# manifest loading (mirrors e2_token_probe/analyze.py::load_manifest)
# ---------------------------------------------------------------------------
def load_manifest(path: Path) -> dict[str, dict]:
    pairs: dict[str, dict] = {}
    if not path.exists():
        return pairs
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") == "ok":
            pairs[rec["pair_id"]] = rec
    return pairs


def layer_values(rec: dict, layer: int, metric: str) -> float | None:
    row = rec.get("per_layer", {}).get(str(layer), {})
    return row.get(metric)


# ---------------------------------------------------------------------------
# verbatim copies from e2_token_probe/probe.py (kept importable without torch)
# ---------------------------------------------------------------------------
def auroc(scores, labels) -> float:  # noqa: ANN001 (np arrays)
    import numpy as np
    pos = labels.astype(bool)
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mask_grid_fractions(mask_path: Path, grid_hw):  # noqa: ANN001
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    if mask.shape[:2] != (1024, 1024):
        mask = np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8)).resize((1024, 1024), Image.NEAREST),
            dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
    h, w = grid_hw
    t = torch.from_numpy(mask)[None, None]
    return F.interpolate(t, size=(h, w), mode="area")[0, 0].numpy()


def in_mask_flags(frac, threshold: float = 0.5):  # noqa: ANN001
    import numpy as np
    return (frac >= threshold).reshape(-1)


def npz_recompute_auroc(npz_path: Path, late_layer: int, late_step: int,
                        lat_in) -> float:  # noqa: ANN001
    import numpy as np
    with np.load(npz_path) as z:
        key = f"flow_map_L{late_layer}_S{late_step}"
        if key not in z.files:
            cands = [k for k in z.files if k.startswith("flow_map_L") and k.endswith(f"_S{late_step}")]
            if not cands:
                return float("nan")
            key = cands[0]
        head_mean = z[key].astype(np.float32).mean(axis=0)
    return auroc(head_mean, lat_in.astype("int8"))


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def detect_late(manifests: dict[str, dict[str, dict]]) -> dict:
    """Late step = max captured probe step; late layer = that layer under the
    probe.py L==S convention (fallback: max layer carrying flow metrics)."""
    late_step = -1
    for recs in manifests.values():
        for rec in recs.values():
            for s in rec.get("probe_steps", {}):
                late_step = max(late_step, int(s))
            break  # one record per manifest is enough (config is per-run)
    flow_layers: set[int] = set()
    for recs in manifests.values():
        for rec in recs.values():
            for L, row in rec.get("per_layer", {}).items():
                if "flow_auroc" in row:
                    flow_layers.add(int(L))
            break
    late_layer = late_step if late_step in flow_layers else (max(flow_layers) if flow_layers else -1)
    return {"late_step": late_step, "late_layer": late_layer}


def analyze_probe(manifests: dict[str, dict[str, dict]], n_boot: int, min_n: int) -> dict:
    models = ordered_models([m for m, recs in manifests.items() if recs])
    common = None
    for m in models:
        common = set(manifests[m]) if common is None else (common & set(manifests[m]))
    common = sorted(common or [])
    layers = sorted({int(L) for m in models for rec in manifests[m].values()
                     for L in rec.get("per_layer", {})})
    result: dict = {"models": models, "n_pairs": len(common), "metrics": {}, "pairs_list":
                    [pair_label(y, x) for (y, x) in available_pairs(models)]}

    for metric, _desc in METRICS:
        per_pair: dict[str, dict] = {}
        for (y, x) in available_pairs(models):
            rows: dict[int, dict] = {}
            for L in layers:
                ids = [pid for pid in common
                       if layer_values(manifests[x][pid], L, metric) is not None
                       and layer_values(manifests[y][pid], L, metric) is not None]
                if len(ids) < min_n:
                    continue
                rows[L] = pair_stats(
                    {pid: {"v": layer_values(manifests[x][pid], L, metric)} for pid in ids},
                    {pid: {"v": layer_values(manifests[y][pid], L, metric)} for pid in ids},
                    "v", ids, n_boot)
            best_layer = None
            if rows:
                best_layer = max(rows, key=lambda L: (rows[L]["mean_diff_y_minus_x"], -L))
            per_pair[pair_label(y, x)] = {"best_layer": best_layer,
                                          "best": rows.get(best_layer),
                                          "layers": {str(L): v for L, v in rows.items()}}
        result["metrics"][metric] = per_pair

    late = detect_late(manifests)
    if late["late_layer"] >= 0:
        L = late["late_layer"]
        late_metrics: dict[str, dict] = {}
        n_late = 0
        for field in ("flow_auroc", "flow_log2_contrast", "flow_mass_in_frac"):
            ids = [pid for pid in common
                   if all(layer_values(manifests[m][pid], L, field) is not None for m in models)]
            if len(ids) < min_n:
                continue
            n_late = max(n_late, len(ids))
            late_metrics[field] = {}
            for (y, x) in available_pairs(models):
                late_metrics[field][pair_label(y, x)] = pair_stats(
                    {pid: {"v": layer_values(manifests[x][pid], L, field)} for pid in ids},
                    {pid: {"v": layer_values(manifests[y][pid], L, field)} for pid in ids},
                    "v", ids, n_boot)
        result["late"] = {**late, "layer": L, "stats": late_metrics,
                          "n_pairs_with_flow": n_late}
    return result


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------
def markdown(result: dict, out: Path, n_boot: int) -> None:
    models = result["models"]
    lines = ["# E3 / E2 token probe: three-way comparison (51 val pairs)", "",
             f"Paired diff = column model - row model; bootstrap 95% CI "
             f"({n_boot} resamples, seed 1234). SIG = CI excludes 0. "
             f"Best layer = layer with the largest mean paired diff.", ""]
    if "late" in result:
        late = result["late"]
        lines += [f"## Late-step AUROC (step {late['late_step']}, layer L{late['layer']}, "
                  f"n={late['n_pairs_with_flow']})", "",
                  "| pair | diff | 95% CI | sig | w/l/t | wilcoxon p |",
                  "|---|---|---|---|---|---|"]
        for pair, e in late["stats"].get("flow_auroc", {}).items():
            if not e or e.get("n", 0) == 0:
                continue
            lines.append(f"| {pair} | {e['mean_diff_y_minus_x']:+.4f} "
                         f"| [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] "
                         f"| {'YES' if e['significant'] else ''} "
                         f"| {e['wins/losses/ties'][0]}/{e['wins/losses/ties'][1]}/{e['wins/losses/ties'][2]} "
                         f"| {e['wilcoxon_p'] if e['wilcoxon_p'] is not None else 'n/a'} |")
        lines.append("")
    for metric, desc in METRICS:
        lines += [f"## {metric}", f"*{desc}*", ""]
        for pair, data in result["metrics"][metric].items():
            b = data["best"]
            if b and b.get("n", 0) > 0:
                lines.append(f"- {pair}: best **L{data['best_layer']}** "
                             f"{fmt_entry(b, 4)}")
        lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=PROBE_ROOT / "probe_base")
    parser.add_argument("--final991-dir", type=Path, default=PROBE_ROOT / "probe_final991")
    parser.add_argument("--v6-dir", type=Path, default=CFCOT / "outputs/e3_eval/probe_v6")
    parser.add_argument("--v7-dir", type=Path, default=CFCOT / "outputs/e3_eval/probe_v7",
                        help="v7 probe dir (e3_align_v7 model); skipped when absent")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--min-n", type=int, default=5)
    parser.add_argument("--verify-npz", action="store_true",
                        help="recompute late AUROC from maps/*.npz (needs torch+PIL)")
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--npz-tol", type=float, default=0.02)
    args = parser.parse_args()

    dirs = {"base": args.base_dir, "final991": args.final991_dir,
            "v6": args.v6_dir, "v7": args.v7_dir}
    manifests: dict[str, dict[str, dict]] = {}
    for m, d in dirs.items():
        man = load_manifest(d / "probe_manifest.jsonl")
        if man:
            manifests[m] = man
        else:
            print(f"[warn] no ok pairs in {d}/probe_manifest.jsonl")
    if len(manifests) < 2:
        raise SystemExit("probe_stats: need >= 2 probe manifests with ok pairs")

    result = analyze_probe(manifests, args.bootstrap, args.min_n)
    result["experiment"] = "e3_e2_token_probe_threeway"

    if args.verify_npz and "late" in result:
        late = result["late"]
        common = sorted(set.intersection(*[set(v) for v in manifests.values()]))
        records = {}
        for line in args.val.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                records[rec["pair_id"]] = rec
        lat_side = IMAGE_SIZE // LAT_DOWNSAMPLE
        per_model: dict[str, dict] = {}
        for m, d in dirs.items():
            if m not in manifests:
                continue
            devs, checked = [], 0
            for pid in common:
                npz = d / "maps" / f"{pid}.npz"
                rec = records.get(pid)
                if not npz.exists() or rec is None:
                    continue
                lat_in = in_mask_flags(mask_grid_fractions(
                    Path(rec["automatic_difference_mask"]["path"]), (lat_side, lat_side)))
                got = npz_recompute_auroc(npz, late["late_layer"], late["late_step"], lat_in)
                want = layer_values(manifests[m][pid], late["late_layer"], "flow_auroc")
                if want is None or got != got:  # nan guard
                    continue
                devs.append(abs(got - want))
                checked += 1
            per_model[m] = {"n_checked": checked,
                            "max_abs_diff": max(devs) if devs else None}
        result["npz_check"] = {"tol": args.npz_tol, "per_model": per_model}
        for m, c in per_model.items():
            if c["max_abs_diff"] is not None and c["max_abs_diff"] > args.npz_tol:
                print(f"[warn] npz vs manifest late AUROC drift for {m}: "
                      f"{c['max_abs_diff']:.4f} > tol {args.npz_tol}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / "probe_stats_threeway.json"
    out_json.write_text(json.dumps(result, indent=1), encoding="utf-8")
    markdown(result, args.out_dir / "probe_table.md", args.bootstrap)
    print(json.dumps({"event": "probe_stats_done", "json": str(out_json),
                      "n_pairs": result["n_pairs"]}))


if __name__ == "__main__":
    main()
