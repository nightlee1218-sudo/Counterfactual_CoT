#!/usr/bin/env python3
"""E3 eval CPU selftest (no GPU, no BAGEL stack, no heavy imports).

Covers:
  1. paired_bootstrap / pair_stats math (determinism, CI, significance,
     non-finite dropping)
  2. wilcoxon fallback
  3. indomain_stats CLI on synthetic Module-A rows (3-way + replication)
  4. stats_threeway CLI on synthetic CLIP-T rows (subsets + ALL)
  5. probe_stats CLI on synthetic probe manifests (best layer + late AUROC)
  6. probe npz cross-check path (skipped if torch/PIL unavailable)
  7. model_shim dry-run + real injection on a toy script
  8. score_indomain_threeway --help (lazy-import check)
  9. summarize on synthetic stats files
 10. v6 generator copy guard audit (20261014 smoke fix)
 11. four-way (base/final991/v6/v7) auto-detection + pair enumeration
 12. vqa_scorer: T2I-CompBench templates, answer matching, real-prompt
     parse coverage (450/450 when the E1 prompts dir is reachable), mock
     CLI pipeline + stats_threeway VQA merge
 13. run_e3_eval.sh v7 wiring: bash syntax, V7_CKPT resolution, v7 stage
     functions, idempotent skips, E1_VQA default-on, v6-preserving smoke

Usage:  python3 selftest_e3_eval.py
Exit 0 only if every test passes.
"""
from __future__ import annotations

import importlib.util
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASS, FAIL = 0, 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True, text=True,
                         cwd=str(HERE))


# ---------------------------------------------------------------------------
# 1-2: pure stats math
# ---------------------------------------------------------------------------
def test_pair_stats(tmp: Path) -> None:
    istats = load_module("istats_selftest", HERE / "indomain_stats.py")
    diffs = [0.02 + 0.01 * ((i % 5) - 2) / 2 for i in range(30)]
    lo, hi, mean = istats.paired_bootstrap(diffs, 2000, 1234)
    check("bootstrap_deterministic", abs(mean - sum(diffs) / len(diffs)) < 1e-12)
    check("bootstrap_ci_order", lo <= mean <= hi)
    lo2, hi2, _ = istats.paired_bootstrap(diffs, 2000, 1234)
    check("bootstrap_seed_reproducible", (lo, hi) == (lo2, hi2))
    rows_x = {f"p{i:03d}": {"v": 1.0 + 0.01 * (i % 7)} for i in range(20)}
    rows_y = {k: {"v": v["v"] + 0.05} for k, v in rows_x.items()}
    e = istats.pair_stats(rows_x, rows_y, "v", sorted(rows_x), 500)
    check("pair_stats_sig_pos", e["significant"] and e["mean_diff_y_minus_x"] > 0)
    check("pair_stats_wlt", e["wins/losses/ties"] == [20, 0, 0])
    # non-finite + None dropping (json reads inf as float('inf'))
    rows_x["pInf"] = {"v": float("inf")}
    rows_y["pInf"] = {"v": 1.0}
    rows_x["pNone"] = {"v": None}
    rows_y["pNone"] = {"v": 2.0}
    e2 = istats.pair_stats(rows_x, rows_y, "v", sorted(rows_x), 500)
    check("pair_stats_drops_nonfinite", e2["n"] == 20 and e2["n_dropped"] == 2)
    # ties only -> not significant
    e3 = istats.pair_stats(rows_x, {k: dict(v) for k, v in rows_x.items()}, "v",
                           sorted(rows_x)[:10], 200)
    check("pair_stats_ties_not_sig", not e3["significant"] and e3["n"] == 10)


def test_wilcoxon() -> None:
    istats = load_module("istats_selftest2", HERE / "indomain_stats.py")
    try:
        import scipy  # noqa: F401
        have_scipy = True
    except Exception:
        have_scipy = False
    p = istats.wilcoxon_p([0.1 * (i + 1) for i in range(20)])
    if have_scipy:
        check("wilcoxon_strong_effect", p is not None and p < 0.05, f"p={p}")
    else:
        check("wilcoxon_no_scipy_none", p is None)
    check("wilcoxon_too_few", istats.wilcoxon_p([1.0, 2.0]) is None)


# ---------------------------------------------------------------------------
# 3: Module-A stats CLI
# ---------------------------------------------------------------------------
def test_indomain_stats_cli(tmp: Path) -> None:
    rng = random.Random(7)
    scores = tmp / "scores_indomain"
    scores.mkdir(parents=True)
    n = 40
    for model, lift in (("base", 0.0), ("final991", 0.03), ("v6", 0.04)):
        for split in ("val", "test"):
            rows = []
            for i in range(n):
                shared = rng.random() * 0.1
                row = {"pair_id": f"{split}_p{i:03d}", "model": model, "split": split}
                for m in ("tgt_mse", "src_mse", "in_mse", "out_mse", "bg_drift"):
                    row[m] = 0.1 + shared - lift * (0.5 if m == "tgt_mse" else 0.2)
                for m in ("tgt_ssim", "clip_tgt", "dino_tgt", "clip_src"):
                    row[m] = 0.5 + shared + lift
                row["tgt_psnr"] = 20.0 + shared + lift
                row["tgt_lpips"] = 0.3 - shared - lift
                row["strength"] = 1.0 + shared * 0.5
                row["clip_tgt_vs_src"] = 0.4 + shared
                row["directional"] = row["clip_tgt"] - row["clip_tgt_vs_src"]
                if i == 0 and model == "v6" and split == "val":
                    row["tgt_psnr"] = float("inf")  # non-finite dropping
                    row["dino_tgt"] = None
                rows.append(row)
            with (scores / f"indomain_{model}_{split}.jsonl").open("w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
    proc = run_cli(["indomain_stats.py", "--scores-dir", str(scores),
                    "--splits", "val", "test", "--bootstrap", "300"])
    check("indomain_cli_exit0", proc.returncode == 0, proc.stderr[-400:])
    data = json.loads((scores / "indomain_stats_threeway.json").read_text())
    pooled = data["splits"]["pooled"]
    check("indomain_models3", pooled["models"] == ["base", "final991", "v6"])
    e = pooled["pairs"]["v6-final991"]["metrics"]["tgt_ssim"]
    check("indomain_v6_sig_pos", e["significant"] and e["mean_diff_y_minus_x"] > 0)
    e_inf = pooled["pairs"]["v6-final991"]["metrics"]["tgt_psnr"]
    check("indomain_inf_dropped", e_inf["n"] == 79 and e_inf["n_dropped"] == 1,
          f"n={e_inf['n']} dropped={e_inf['n_dropped']}")
    # replication pair matches a direct helper computation
    istats = load_module("istats_selftest3", HERE / "indomain_stats.py")
    rows_b = {r["pair_id"]: r for split in ("val", "test")
              for r in (json.loads(l) for l in
                        (scores / f"indomain_base_{split}.jsonl").read_text().splitlines())}
    rows_f = {r["pair_id"]: r for split in ("val", "test")
              for r in (json.loads(l) for l in
                        (scores / f"indomain_final991_{split}.jsonl").read_text().splitlines())}
    ids = sorted(set(rows_b) & set(rows_f))
    direct = istats.pair_stats(rows_b, rows_f, "tgt_ssim", ids, 300)  # same n_boot as CLI
    rep = pooled["pairs"]["final991-base"]["metrics"]["tgt_ssim"]
    check("indomain_replication_match",
          abs(rep["ci95"][0] - direct["ci95"][0]) < 1e-9
          and abs(rep["mean_diff_y_minus_x"] - direct["mean_diff_y_minus_x"]) < 1e-9)
    check("indomain_md_written", (scores / "indomain_table.md").exists())


# ---------------------------------------------------------------------------
# 4: E1 CLIP-T stats CLI
# ---------------------------------------------------------------------------
def test_stats_threeway_cli(tmp: Path) -> None:
    rng = random.Random(11)
    scores = tmp / "scores_t2i"
    scores.mkdir(parents=True)
    for model, lift in (("base", 0.0), ("final991", 0.012), ("v6", 0.018)):
        rows = []
        for i in range(60):
            subset = "attribute_binding" if i % 2 == 0 else "counting"
            shared = rng.random() * 0.02
            rows.append({"item_id": f"it{i:03d}", "subset": subset, "model": model,
                         "clip_cos": 0.30 + shared + lift + 0.004 * rng.random()})
        with (scores / f"clipT_{model}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    proc = run_cli(["stats_threeway.py", "--scores-dir", str(scores),
                    "--bootstrap", "300"])
    check("e1_cli_exit0", proc.returncode == 0, proc.stderr[-400:])
    data = json.loads((scores / "stats_threeway.json").read_text())
    all_e = data["clipT"]["ALL"]
    check("e1_all_n60", all_e["n"] == 60)
    check("e1_subsets_present",
          {"attribute_binding", "counting"} <= set(data["clipT"].keys()))
    e = all_e["pairs"]["v6-final991"]
    check("e1_v6_sig_pos", e["significant"] and e["mean_diff_y_minus_x"] > 0)
    rep = all_e["pairs"]["final991-base"]
    check("e1_replication_pos_sig", rep["significant"] and rep["mean_diff_y_minus_x"] > 0)
    check("e1_vqa_skipped", data["vqa"] is None)
    check("e1_md_written", (scores / "comparison_table.md").exists())
    # tolerance: drop v6 -> still runs on 2 models
    (scores / "clipT_v6.jsonl").unlink()
    proc2 = run_cli(["stats_threeway.py", "--scores-dir", str(scores),
                     "--bootstrap", "200"])
    data2 = json.loads((scores / "stats_threeway.json").read_text())
    check("e1_two_model_fallback",
          data2["clipT"]["models"] == ["base", "final991"]
          and "v6-final991" not in data2["clipT"]["ALL"]["pairs"])


# ---------------------------------------------------------------------------
# 5-6: probe stats CLI (+npz cross-check)
# ---------------------------------------------------------------------------
def synth_probe_dir(root: Path, model: str, lift: float, rng: random.Random,
                    layers=(3, 7, 23), n_pairs=16) -> Path:
    d = root / f"probe_{model}"
    (d / "maps").mkdir(parents=True)
    rows = []
    for i in range(n_pairs):
        pid = f"pr{i:03d}"
        per_layer = {}
        for L in layers:
            base_val = 0.4 + 0.1 * rng.random()
            row = {m: base_val + lift for m in
                   ("norm_ratio_vae", "norm_ratio_vit", "norm_ratio_visual", "mass_visual",
                    "margin_vae", "margin_vit", "margin_lat")}
            if L >= 7:
                for fm in ("flow_log2_contrast", "flow_mass_in_frac", "flow_auroc"):
                    row[fm] = base_val + lift
            per_layer[str(L)] = row
        rows.append({"pair_id": pid, "status": "ok", "n_latent": 256,
                     "n_latent_in": 128,
                     "probe_steps": {"0": 1.0, "11": 0.5, "23": 0.1},
                     "per_layer": per_layer})
    with (d / "probe_manifest.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return d


def test_probe_stats_cli(tmp: Path) -> None:
    pstats = load_module("pstats_selftest", HERE / "probe_stats.py")
    rng = random.Random(13)
    d_base = synth_probe_dir(tmp, "base", 0.0, rng)
    d_final = synth_probe_dir(tmp, "final991", 0.05, rng)
    d_v6 = synth_probe_dir(tmp, "v6", 0.08, rng)
    out = tmp / "probe_stats"
    argv = ["probe_stats.py", "--base-dir", str(d_base), "--final991-dir", str(d_final),
            "--v6-dir", str(d_v6), "--out-dir", str(out), "--bootstrap", "300",
            "--min-n", "5"]
    proc = run_cli(argv)
    check("probe_cli_exit0", proc.returncode == 0, proc.stderr[-400:])
    data = json.loads((out / "probe_stats_threeway.json").read_text())
    check("probe_models3", data["models"] == ["base", "final991", "v6"])
    check("probe_npairs16", data["n_pairs"] == 16)
    late = data["late"]
    check("probe_late_detection", late["late_step"] == 23 and late["late_layer"] == 23)
    e = late["stats"]["flow_auroc"]["v6-final991"]
    check("probe_late_sig_pos", e["significant"] and e["mean_diff_y_minus_x"] > 0)
    # best layer on a metric present at all layers; lift equal everywhere ->
    # best layer should exist and be positive
    best = data["metrics"]["norm_ratio_vae"]["v6-final991"]["best"]
    check("probe_best_layer_pos", best is not None and best["mean_diff_y_minus_x"] > 0)
    # auroc pure function on synthetic arrays
    import numpy as np
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.95, 0.05, 0.85])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    check("auroc_perfect", abs(pstats.auroc(scores, labels) - 1.0) < 1e-9)
    tie_scores = np.zeros(10)
    tie_labels = np.array([0, 1] * 5, dtype=np.int8)
    check("auroc_ties_half", abs(pstats.auroc(tie_scores, tie_labels) - 0.5) < 1e-9)


def test_probe_npz_verify(tmp: Path) -> None:
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        print("SKIP probe_npz_verify (torch/PIL unavailable)")
        return
    import numpy as np
    from PIL import Image, ImageDraw

    pstats = load_module("pstats_selftest2", HERE / "probe_stats.py")
    lat_side = pstats.IMAGE_SIZE // pstats.LAT_DOWNSAMPLE  # 64
    n_lat = lat_side * lat_side
    val = tmp / "val.jsonl"
    records = []
    d_map = {}
    for i in range(6):
        pid = f"vr{i:02d}"
        mask = Image.new("L", (512, 512), 0)
        ImageDraw.Draw(mask).rectangle([0, 0, 256, 512], fill=255)
        mask_path = tmp / f"mask_{pid}.png"
        mask.save(mask_path)
        records.append({"pair_id": pid,
                        "automatic_difference_mask": {"path": str(mask_path)}})
        lat_in = pstats.in_mask_flags(
            pstats.mask_grid_fractions(mask_path, (lat_side, lat_side)))
        rng = np.random.RandomState(i)
        head_mean = rng.rand(n_lat).astype(np.float32) * 0.1
        head_mean[lat_in] += 0.8          # strong in-mask separation
        maps = np.stack([head_mean, head_mean * 0.9]).astype(np.float16)
        d_map[pid] = (maps, lat_in)
    with val.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    lifts = {"base": 0.0, "final991": 0.03, "v6": 0.05}
    rng = random.Random(21)
    dirs = {}
    for model, lift in lifts.items():
        d = tmp / f"vp_{model}"
        (d / "maps").mkdir(parents=True)
        for pid, (maps, lat_in) in d_map.items():
            # manifest late AUROC must equal the npz-recomputed value (no lift
            # here) so the cross-check drift stays ~0; the lift is applied to
            # the other two late metrics only.
            auroc_val = pstats.auroc(maps.astype(np.float32).mean(axis=0),
                                     lat_in.astype("int8"))
            row = {"pair_id": pid, "status": "ok",
                   "probe_steps": {"0": 1.0, "11": 0.5, "23": 0.1},
                   "per_layer": {"23": {"flow_auroc": auroc_val,
                                        "flow_log2_contrast": 1.0 + lift,
                                        "flow_mass_in_frac": 0.5 + lift}}}
            with (d / "probe_manifest.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            np.savez_compressed(d / "maps" / f"{pid}.npz", **{"flow_map_L23_S23": maps})
        dirs[model] = d
    out = tmp / "vp_out"
    proc = run_cli(["probe_stats.py", "--base-dir", str(dirs["base"]),
                    "--final991-dir", str(dirs["final991"]),
                    "--v6-dir", str(dirs["v6"]), "--out-dir", str(out),
                    "--bootstrap", "300", "--min-n", "5",
                    "--verify-npz", "--val", str(val), "--npz-tol", "0.02"])
    check("probe_verify_exit0", proc.returncode == 0, proc.stderr[-400:])
    data = json.loads((out / "probe_stats_threeway.json").read_text())
    chk = data.get("npz_check", {}).get("per_model", {})
    ok_drift = all(c.get("max_abs_diff") is not None and c["max_abs_diff"] < 0.02
                   and c["n_checked"] == 6 for c in chk.values())
    check("probe_npz_drift_small", bool(chk) and ok_drift, json.dumps(chk)[:300])


# ---------------------------------------------------------------------------
# 7: shim
# ---------------------------------------------------------------------------
def test_shim(tmp: Path) -> None:
    toy = tmp / "toy_gen.py"
    toy.write_text(
        "import argparse, json\n"
        "MODELS = {'base': {'trainable_checkpoint': None, 'skip_trainable_reload': True}}\n"
        "def main():\n"
        "    p = argparse.ArgumentParser()\n"
        "    p.add_argument('--model', required=True, choices=list(MODELS))\n"
        "    p.add_argument('--out-dir', required=True)\n"
        "    a = p.parse_args()\n"
        "    m = MODELS[a.model]\n"
        "    print(json.dumps({'ran': True, 'model': a.model,\n"
        "                      'ckpt': str(m['trainable_checkpoint'])}))\n")
    ckpt = tmp / "latest.pt"
    ckpt.write_bytes(b"fake-ckpt")
    # dry run
    proc = subprocess.run([sys.executable, str(HERE / "model_shim.py"),
                           "--script", str(toy), "--alias", "v6", "--ckpt", str(ckpt),
                           "--dry-run", "--", "--model", "v6", "--out-dir", str(tmp)],
                          capture_output=True, text=True)
    check("shim_dryrun_exit0", proc.returncode == 0, proc.stderr[-300:])
    evt = json.loads(proc.stdout.strip().splitlines()[-1])
    check("shim_dryrun_registered", "v6" in evt["registered"] and "base" in evt["registered"])
    check("shim_dryrun_argv", evt["forwarded_argv"] == ["--model", "v6", "--out-dir", str(tmp)])
    # real injection: toy main() accepts --model v6 through the injected alias
    proc2 = subprocess.run([sys.executable, str(HERE / "model_shim.py"),
                            "--script", str(toy), "--alias", "v6", "--ckpt", str(ckpt),
                            "--", "--model", "v6", "--out-dir", str(tmp)],
                           capture_output=True, text=True)
    check("shim_realrun_exit0", proc2.returncode == 0, proc2.stderr[-300:])
    ran = json.loads(proc2.stdout.strip().splitlines()[-1])
    check("shim_realrun_model_v6", ran.get("model") == "v6" and ran.get("ckpt") == str(ckpt))
    # missing checkpoint rejected
    proc3 = subprocess.run([sys.executable, str(HERE / "model_shim.py"),
                            "--script", str(toy), "--alias", "v6", "--ckpt", str(tmp / "nope.pt"),
                            "--dry-run"], capture_output=True, text=True)
    check("shim_missing_ckpt_rejected", proc3.returncode != 0)


# ---------------------------------------------------------------------------
# 8: lazy-import check
# ---------------------------------------------------------------------------
def test_score_lazy_import() -> None:
    proc = subprocess.run([sys.executable, str(HERE / "score_indomain_threeway.py"),
                          "--help"], capture_output=True, text=True, cwd=str(HERE))
    check("score_cli_help", proc.returncode == 0 and "--gen-v6-dir" in proc.stdout,
          proc.stderr[-300:])
    src = (HERE / "score_indomain_threeway.py").read_text()
    check("score_module_top_clean",
          "import torch" not in src.split("def build_ctx")[0]
          and "import open_clip" not in src.split("def build_ctx")[0])


# ---------------------------------------------------------------------------
# 9: summarize
# ---------------------------------------------------------------------------
def test_summarize(tmp: Path) -> None:
    root = tmp / "e3root"
    (root / "scores_t2i").mkdir(parents=True)
    (root / "scores_indomain").mkdir(parents=True)
    (root / "probe_stats").mkdir(parents=True)
    pair = {"mean_diff_y_minus_x": 0.01, "ci95": [0.005, 0.02], "significant": True,
            "wins/losses/ties": [3, 1, 0], "wilcoxon_p": 0.04, "n": 4, "n_dropped": 0,
            "x_mean": 0.3, "y_mean": 0.31}
    (root / "scores_t2i" / "stats_threeway.json").write_text(json.dumps({
        "clipT": {"models": ["base", "final991", "v6"],
                  "ALL": {"n": 450, "model_means": {}, "pairs": {"v6-final991": pair}}}}))
    (root / "scores_indomain" / "indomain_stats_threeway.json").write_text(json.dumps({
        "models": ["base", "final991", "v6"],
        "metrics": {"tgt_ssim": "higher"},
        "splits": {"pooled": {"models": ["base", "final991", "v6"], "n_pairs": 95,
                              "model_means": {"tgt_ssim": {}},
                              "pairs": {"v6-final991": {"diff": "v6 - final991",
                                                        "metrics": {"tgt_ssim": pair}}}}}}))
    (root / "probe_stats" / "probe_stats_threeway.json").write_text(json.dumps({
        "models": ["base", "final991", "v6"], "n_pairs": 51,
        "late": {"late_step": 23, "late_layer": 23, "n_pairs_with_flow": 51,
                 "stats": {"flow_auroc": {"v6-final991": pair}}},
        "metrics": {"norm_ratio_vae": {"v6-final991": {"best_layer": 13, "best": pair}}}}))
    proc = run_cli(["summarize.py", "--e3-root", str(root)])
    check("summarize_exit0", proc.returncode == 0, proc.stderr[-400:])
    md = (root / "summary_table.md").read_text()
    check("summarize_sections",
          all(s in md for s in ("E1 general T2I", "Module A in-domain", "E2 token probe",
                                "Late-step AUROC", "v6-final991")))
    sj = json.loads((root / "summary.json").read_text())
    check("summarize_json_components", set(sj["components"]) ==
          {"e1_clipt", "module_a", "e2_probe"})
    # missing component tolerance
    (root / "probe_stats" / "probe_stats_threeway.json").unlink()
    proc2 = run_cli(["summarize.py", "--e3-root", str(root)])
    check("summarize_tolerant", proc2.returncode == 0)
    md2 = (root / "summary_table.md").read_text()
    check("summarize_missing_marked", "(missing)" in md2)


# ---------------------------------------------------------------------------
# 10: v6 generator copy guard audit (20261014 smoke fix)
# ---------------------------------------------------------------------------
def test_guard_copy() -> None:
    copy_path = HERE / "gen_same_seed_states_v6.py"
    check("guard_copy_exists", copy_path.is_file())
    if not copy_path.is_file():
        return
    copy_src = copy_path.read_text(encoding="utf-8")
    check("guard_copy_accepts_v6_label", '"gold40_automask_v6_align_fsdp"' in copy_src)
    check("guard_copy_accepts_v7_label", '"gold40_automask_v7_align_fsdp"' in copy_src)
    check("guard_copy_keeps_v5_label", '"gold40_automask_v5_paired_fsdp"' in copy_src)
    orig_path = HERE.parent / "gold40_automask_dual_target_cot1024_fsdp_v5" / \
        "generate_same_seed_states.py"
    if not orig_path.is_file():
        print(f"SKIP guard_copy_vs_original diff audit (no {orig_path})")
        return
    orig_src = orig_path.read_text(encoding="utf-8")
    check("guard_v6_label_absent_from_original",
          "gold40_automask_v6_align_fsdp" not in orig_src)
    # line-by-line diff: only the guard lines may differ
    o_lines = orig_src.splitlines()
    c_lines = copy_src.splitlines()
    diffs = [(i, o, c) for i, (o, c) in enumerate(zip(o_lines, c_lines)) if o != c]
    tail_same = len(o_lines) == len(c_lines)
    check("guard_copy_sameline_count", tail_same,
          f"orig {len(o_lines)} vs copy {len(c_lines)}")
    check("guard_copy_diff_lines_minimal", len(diffs) <= 4,
          f"diffs at {[i + 1 for i, _, _ in diffs]}")
    check("guard_copy_diff_only_in_guard",
          all("automask" in c.lower() and "automask" in o.lower() for _, o, c in diffs),
          "; ".join(repr(c)[:80] for _, _, c in diffs))
    check("guard_copy_root_resolution_unchanged",
          "Path(__file__).resolve().parents[3]" in copy_src)
    # the E1 / E2 chains load the v6 checkpoint WITHOUT any format-label
    # guard (build_inference_model -> load_trainable_strict is key-strict;
    # generate_t2i only reads payload["step"]) — assert that stays true.
    for script in ("e2_token_probe/probe.py", "e1_general_t2i/generate_t2i.py"):
        s = (HERE.parent / script)
        src = s.read_text(encoding="utf-8") if s.is_file() else ""
        check(f"no_format_guard:{script.split('/')[-1]}",
              '.get("format")' not in src and ".get('format')" not in src)


# ---------------------------------------------------------------------------
# 11: four-way (base/final991/v6/v7) auto-detection + pair enumeration
# ---------------------------------------------------------------------------
def test_four_way_models(tmp: Path) -> None:
    istats = load_module("istats_4way", HERE / "indomain_stats.py")
    # backward-compat: the 3-model pair order is unchanged
    check("pairs_3way_compat",
          istats.pairs_for(["base", "final991", "v6"])
          == [("v6", "final991"), ("v6", "base"), ("final991", "base")])
    # 4 models -> 6 ordered pairs, v7 pairs appended after the v6 three-way
    check("pairs_4way_count",
          istats.pairs_for(["base", "final991", "v6", "v7"])
          == [("v6", "final991"), ("v6", "base"), ("final991", "base"),
              ("v7", "v6"), ("v7", "final991"), ("v7", "base")])
    # indomain CLI auto-detects 4 models and emits all 6 pairs
    scores = tmp / "scores_indomain_4w"
    scores.mkdir(parents=True)
    n = 40
    for model, lift in (("base", 0.0), ("final991", 0.03), ("v6", 0.01), ("v7", 0.04)):
        rows = []
        for i in range(n):
            shared = (i % 50) / 500.0  # identical across models: pure paired design
            rows.append({"pair_id": f"p{i:03d}", "model": model, "split": "val",
                         "tgt_ssim": 0.5 + shared + lift,
                         "tgt_mse": 0.1 + shared - lift,
                         "in_mse": 0.1 + shared - lift, "out_mse": 0.1 + shared,
                         "tgt_psnr": 20.0 + lift, "tgt_lpips": 0.3 - lift,
                         "src_mse": 0.1 + shared, "clip_src": 0.5 + shared,
                         "clip_tgt": 0.5 + shared + lift, "dino_tgt": 0.4 + lift,
                         "strength": 1.0, "bg_drift": 0.05, "directional": 0.1 + lift})
        with (scores / f"indomain_{model}_val.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    proc = run_cli(["indomain_stats.py", "--scores-dir", str(scores),
                    "--splits", "val", "--bootstrap", "300"])
    check("fourway_indomain_cli_exit0", proc.returncode == 0, proc.stderr[-400:])
    data = json.loads((scores / "indomain_stats_threeway.json").read_text())
    pooled = data["splits"]["pooled"]
    check("fourway_models_detected", pooled["models"] == ["base", "final991", "v6", "v7"])
    check("fourway_six_pairs",
          set(pooled["pairs"]) == {"v6-final991", "v6-base", "final991-base",
                                   "v7-v6", "v7-final991", "v7-base"})
    e = pooled["pairs"]["v7-final991"]["metrics"]["tgt_ssim"]
    check("fourway_v7_sig_pos", e["significant"] and e["mean_diff_y_minus_x"] > 0)
    md = (scores / "indomain_table.md").read_text()
    check("fourway_md_columns", all(c in md for c in ("v7-v6", "v7-final991", "v7-base")))
    # E1 CLIP-T: auto-detect 4 models from clipT_*.jsonl
    rng = random.Random(19)
    t2i = tmp / "scores_t2i_4w"
    t2i.mkdir(parents=True)
    for model, lift in (("base", 0.0), ("final991", 0.02), ("v6", 0.012), ("v7", 0.03)):
        with (t2i / f"clipT_{model}.jsonl").open("w") as f:
            for i in range(50):
                f.write(json.dumps({"item_id": f"it{i:03d}", "subset": "counting",
                                    "clip_cos": 0.30 + 0.01 * rng.random() + lift}) + "\n")
    proc = run_cli(["stats_threeway.py", "--scores-dir", str(t2i), "--bootstrap", "300"])
    check("fourway_e1_cli_exit0", proc.returncode == 0, proc.stderr[-400:])
    e1 = json.loads((t2i / "stats_threeway.json").read_text())["clipT"]
    check("fourway_e1_models", e1["models"] == ["base", "final991", "v6", "v7"])
    check("fourway_e1_pairs", set(e1["ALL"]["pairs"]) ==
          {"v6-final991", "v6-base", "final991-base", "v7-v6", "v7-final991", "v7-base"})


# ---------------------------------------------------------------------------
# 12: vqa_scorer (local BLIP-2 T2I-CompBench-style scorer)
# ---------------------------------------------------------------------------
def locate_e1_prompts() -> Path | None:
    """The 450-prompt dir: sibling in the merged tree, else the known
    server-side BAGEL copy (both live under /group)."""
    for cand in (HERE.parent / "e1_general_t2i" / "prompts",
                 Path("/group/40016/niightli/BAGEL/experiments/counterfactual_cot"
                      "/e1_general_t2i/prompts")):
        if (cand / "attribute_binding.jsonl").is_file():
            return cand
    return None


def test_vqa_scorer(tmp: Path) -> None:
    vqa = load_module("vqa_selftest", HERE / "vqa_scorer.py")

    # lazy-import contract (same as score_indomain_threeway)
    proc = subprocess.run([sys.executable, str(HERE / "vqa_scorer.py"), "--help"],
                          capture_output=True, text=True, cwd=str(HERE))
    check("vqa_cli_help", proc.returncode == 0 and "--mock-acc" in proc.stdout,
          proc.stderr[-300:])
    src = (HERE / "vqa_scorer.py").read_text(encoding="utf-8")
    check("vqa_module_top_clean",
          "import torch" not in src.split("def build_blip2_ctx")[0]
          and "import transformers" not in src.split("def build_blip2_ctx")[0])
    bad = subprocess.run([sys.executable, str(HERE / "vqa_scorer.py"),
                          "--models", "nonsense"], capture_output=True, text=True,
                         cwd=str(HERE))
    check("vqa_bad_model_rejected", bad.returncode != 0)

    # answer matching
    check("vqa_yesno_variants",
          vqa.parse_yes_no("Yes.") is True and vqa.parse_yes_no("no,") is False
          and vqa.parse_yes_no("No") is False and vqa.parse_yes_no("YES") is True
          and vqa.parse_yes_no("There is no dog") is False
          and vqa.parse_yes_no("maybe") is None)
    check("vqa_count_variants",
          vqa.parse_count("3") == 3 and vqa.parse_count("Three cups") == 3
          and vqa.parse_count("There are seven birds") == 7
          and vqa.parse_count("unknown") is None)
    check("vqa_answer_ok",
          vqa.answer_ok("yes", True) and not vqa.answer_ok("no", True)
          and vqa.answer_ok("four boys", 4) and not vqa.answer_ok("five", 4))

    # templates: attribute binding + swapped negatives
    q = vqa.build_questions({"prompt": "a red lipstick and a pink blush",
                             "subset": "attribute_binding", "source_category": "color"},
                            negatives=True)
    check("vqa_attr_pos", [x["q"] for x in q[:2]] ==
          ["Is there a red lipstick in the image?", "Is there a pink blush in the image?"]
          and all(x["expected"] is True for x in q[:2]))
    check("vqa_attr_swaps", sorted(x["q"] for x in q[2:]) ==
          ["Is there a pink lipstick in the image?", "Is there a red blush in the image?"]
          and all(x["expected"] is False for x in q[2:]))
    qnn = vqa.build_questions({"prompt": "a red lipstick and a pink blush",
                               "subset": "attribute_binding", "source_category": "color"},
                              negatives=False)
    check("vqa_no_negatives_flag", len(qnn) == 2)
    # complex sentences: "glass dinner party" binding dropped, multi-word
    # objects, of-extension, plural template
    q = vqa.build_questions({"prompt": "The wooden salad bowl and metallic tongs mix "
                              "the fluffy greens for the glass dinner party.",
                              "subset": "attribute_binding", "source_category": "texture"},
                            negatives=True)
    check("vqa_attr_complex", [x["q"] for x in q if x["expected"] is True] ==
          ["Is there a wooden salad bowl in the image?",
           "Are there metallic tongs in the image?",
           "Are there fluffy greens in the image?"])
    q = vqa.build_questions({"prompt": "a rectangular slice of cake",
                             "subset": "attribute_binding", "source_category": "shape"},
                            negatives=True)
    check("vqa_attr_of_ext", q[0]["q"] == "Is there a rectangular slice of cake in the image?")
    q = vqa.build_questions({"prompt": "a fabric pants and a leather bag",
                             "subset": "attribute_binding", "source_category": "texture"},
                            negatives=True)
    check("vqa_attr_plural_neg", "Are there leather pants in the image?" in
          [x["q"] for x in q if x["expected"] is False])

    # counting: narrative-tail suppression vs keep-prepositions
    def counted(prompt):
        q = vqa.build_questions({"prompt": prompt, "subset": "counting",
                                 "source_category": "numeracy"}, negatives=True)
        return {x["q"]: x["expected"] for x in q}
    c = counted("six people gathered for a picnic")
    check("vqa_count_tail_suppressed",
          c == {"How many people are there in the image?": 6})
    c = counted("three trains carried three cups of tea")
    check("vqa_count_carried_kept",
          c == {"How many trains are there in the image?": 3,
                "How many cups are there in the image?": 3})
    c = counted("one frog jumped near three suitcases")
    check("vqa_count_near_kept",
          c == {"How many frogs are there in the image?": 1,
                "How many suitcases are there in the image?": 3})
    c = counted("two sheep, four dogs, three candles, and three bees created a warm atmosphere")
    check("vqa_count_atmosphere_suppressed", sorted(c.values()) == [2, 3, 3, 4]
          and len(c) == 4 and not any("atmosphere" in k for k in c))
    c = counted("one man and two mice")
    check("vqa_count_plurals",
          c == {"How many men are there in the image?": 1,
                "How many mice are there in the image?": 2})
    c = counted("a computer, a microphone, and two flags completed the setup")
    check("vqa_count_indefinite_articles", sorted(c.values()) == [1, 1, 2] and len(c) == 3)

    # spatial: mirror negatives, symmetric relations, grammar fixes
    q = vqa.build_questions({"prompt": "a bird on the right of a man",
                             "subset": "spatial_relations", "source_category": "spatial"},
                            negatives=True)
    check("vqa_spatial_mirror", [x["q"] for x in q] ==
          ["Is there a bird on the right of a man in the image?",
           "Is there a bird on the left of a man in the image?"]
          and [x["expected"] for x in q] == [True, False])
    q = vqa.build_questions({"prompt": "a person next to a chicken",
                             "subset": "spatial_relations", "source_category": "spatial"},
                            negatives=True)
    check("vqa_spatial_symmetric", len(q) == 1
          and q[0]["q"] == "Is there a person next to a chicken in the image?")
    q = vqa.build_questions({"prompt": "a airplane on the right of a rabbit",
                             "subset": "spatial_relations", "source_category": "spatial"},
                            negatives=True)
    check("vqa_spatial_article_fix",
          q[0]["q"] == "Is there an airplane on the right of a rabbit in the image?")
    q = vqa.build_questions({"prompt": "a horse on side of a bicycle",
                             "subset": "spatial_relations", "source_category": "spatial"},
                            negatives=True)
    check("vqa_spatial_side_norm",
          q[0]["q"] == "Is there a horse on the side of a bicycle in the image?")

    # real-prompt coverage: 450/450 must parse (skip when prompts unreachable)
    prompts_dir = locate_e1_prompts()
    if prompts_dir is None:
        print("SKIP vqa_real_coverage (no E1 prompts dir reachable)")
    else:
        fails: list[str] = []
        n = 0
        for fname in ("attribute_binding", "counting", "spatial_relations"):
            for line in (prompts_dir / f"{fname}.jsonl").read_text(
                    encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                n += 1
                rec = json.loads(line)
                try:
                    qs = vqa.build_questions(rec, negatives=True)
                    assert qs and all(x["q"] and x["expected"] is not None for x in qs)
                except Exception as exc:  # noqa: BLE001
                    fails.append(f"{rec['item_id']}: {exc!r}")
        check("vqa_real_coverage_450", n == 450 and not fails,
              f"n={n} fails={fails[:3]}")

    # mock CLI pipeline: two models -> vqa rows -> stats_threeway merge
    gen = tmp / "vqa_gen"
    (gen / "gen_t2i" / "v6").mkdir(parents=True)
    (gen / "legacy" / "base").mkdir(parents=True)
    items = [
        {"item_id": "attribute_binding__color__0000", "prompt": "a red lipstick and a pink blush",
         "source_category": "color", "subset": "attribute_binding", "output_path": "/nonexistent/a.png"},
        {"item_id": "counting__numeracy__0000", "prompt": "four boys and three phones",
         "source_category": "numeracy", "subset": "counting", "output_path": "/nonexistent/b.png"},
        {"item_id": "spatial_relations__spatial__0000", "prompt": "a bird on the right of a man",
         "source_category": "spatial", "subset": "spatial_relations", "output_path": "/nonexistent/c.png"},
    ]
    for d in (gen / "gen_t2i" / "v6", gen / "legacy" / "base"):
        with (d / "generation_manifest.jsonl").open("w") as f:
            for r in items:
                f.write(json.dumps(r) + "\n")
    scores = tmp / "vqa_scores"
    base_cmd = [sys.executable, str(HERE / "vqa_scorer.py"), "--mock",
                "--gen-legacy-root", str(gen / "legacy"),
                "--gen-e3-root", str(gen / "gen_t2i")]
    p_oracle = subprocess.run(base_cmd + ["--models", "base", "--out-dir", str(scores)],
                               capture_output=True, text=True, cwd=str(HERE))
    p_flip = subprocess.run(base_cmd + ["--models", "v6", "--mock-acc", "0.0",
                                        "--out-dir", str(scores)],
                            capture_output=True, text=True, cwd=str(HERE))
    check("vqa_mock_cli_exit0", p_oracle.returncode == 0 and p_flip.returncode == 0,
          (p_oracle.stderr[-300:] + p_flip.stderr[-300:]))
    rows_b = [json.loads(l) for l in (scores / "vqa_base.jsonl").read_text().splitlines()]
    rows_v = [json.loads(l) for l in (scores / "vqa_v6.jsonl").read_text().splitlines()]
    check("vqa_mock_oracle_rows", len(rows_b) == 3 and all(r["vqa_match"] is True
          for r in rows_b) and all(r["mode"] == "mock" for r in rows_b))
    check("vqa_mock_flip_rows", len(rows_v) == 3 and all(r["vqa_match"] is False
          and r["n_correct"] == 0 for r in rows_v))
    check("vqa_row_contract", all(set(("item_id", "subset", "model", "vqa_match",
          "n_questions", "n_correct", "answers")) <= set(r) for r in rows_b + rows_v))
    # resumable: rerun adds nothing
    p_again = subprocess.run(base_cmd + ["--models", "base", "--out-dir", str(scores)],
                             capture_output=True, text=True, cwd=str(HERE))
    rows_b2 = [json.loads(l) for l in (scores / "vqa_base.jsonl").read_text().splitlines()]
    check("vqa_resumable_no_dup", p_again.returncode == 0 and len(rows_b2) == 3)
    # stats_threeway merges the vqa rows automatically
    p_stats = subprocess.run([sys.executable, "stats_threeway.py", "--scores-dir",
                             str(scores), "--bootstrap", "300"],
                             capture_output=True, text=True, cwd=str(HERE))
    check("vqa_stats_exit0", p_stats.returncode == 0, p_stats.stderr[-300:])
    data = json.loads((scores / "stats_threeway.json").read_text())
    vqa_block = data.get("vqa") or {}
    check("vqa_stats_models", vqa_block.get("models") == ["base", "v6"])
    entry = vqa_block.get("ALL", {})
    check("vqa_stats_means", entry.get("model_means") == {"base": 1.0, "v6": 0.0})
    pair = entry.get("pairs", {}).get("v6-base")
    check("vqa_stats_pair_sig", pair is not None
          and abs(pair["mean_diff_y_minus_x"] + 1.0) < 1e-12
          and pair["significant"])
    check("vqa_stats_md_section", "VQA faithfulness (match rate)" in
          (scores / "comparison_table.md").read_text())


# ---------------------------------------------------------------------------
# 13: launcher v7 wiring (four-way run_e3_eval.sh, added 20261015)
# ---------------------------------------------------------------------------
def test_launcher_v7_wiring() -> None:
    sh_path = HERE / "run_e3_eval.sh"
    check("launcher_exists", sh_path.is_file())
    if not sh_path.is_file():
        return
    proc = subprocess.run(["bash", "-n", str(sh_path)], capture_output=True, text=True)
    check("launcher_bash_syntax", proc.returncode == 0, proc.stderr[-300:])
    src = sh_path.read_text(encoding="utf-8")

    # v7 checkpoint: knob + resolver + default run name + model_ckpt gate
    check("launcher_v7_ckpt_var", 'V7_CKPT="${V7_CKPT:-}"' in src)
    check("launcher_v7_resolver",
          "resolve_v7_ckpt()" in src
          and "gold40_automask_v7_align_fsdp_full_shard" in src
          and "*v7*/checkpoints/latest.pt" in src)
    check("launcher_model_ckpt_gate", "model_ckpt()" in src
          and "checkpoint missing" in src)

    # v7 stage functions, isomorphic to the v6 ones
    for fn in ("e1_v7_gen", "e1_v7_score", "indomain_v7_gen",
               "indomain_v7_score", "probe_v7"):
        check(f"launcher_has_{fn}", f"{fn}()" in src)
    for fn in ("e1_gen", "e1_score", "indomain_gen", "indomain_score", "probe_run"):
        check(f"launcher_generic_{fn}", f"{fn}() {{ # " in src)
    check("launcher_v7_shim_alias", '--alias "$m"' in src)
    check("launcher_indomain_v7_dirs", '--gen-v7-dir "$o/gen_indomain/v7"' in src)
    check("launcher_probe_stats_v7dir", '--v7-dir "$o/probe_v7"' in src)

    # idempotent four-way: skip markers + model loops + v6-preserving smoke
    check("launcher_skip_markers", src.count("skip (idempotent)") >= 4)
    check("launcher_model_loops",
          '${E1_MODELS:-v6 v7}' in src and '${INDO_MODELS:-v6 v7}' in src)
    check("launcher_smoke_stays_v6",
          'E1_MODELS=v6 chain_gpu0' in src and 'INDO_MODELS=v6' in src)
    check("launcher_run_checks_v7", "clipT_$m.jsonl" in src
          and "probe_$m/probe_manifest.jsonl" in src)
    check("launcher_status_lists_v7", "gen_t2i/$m/generation_manifest.jsonl" in src)

    # VQA default ON in the chain, opt-out preserved, smoke opts out
    check("launcher_vqa_default_on", '[ "${E1_VQA:-1}" = "1" ]' in src)
    check("launcher_smoke_vqa_off", "E1_VQA=0" in src)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="e3_eval_selftest_"))
    try:
        test_pair_stats(tmp)
        test_wilcoxon()
        test_indomain_stats_cli(tmp)
        test_stats_threeway_cli(tmp)
        test_probe_stats_cli(tmp)
        test_probe_npz_verify(tmp)
        test_shim(tmp)
        test_score_lazy_import()
        test_summarize(tmp)
        test_guard_copy()
        test_four_way_models(tmp)
        test_vqa_scorer(tmp)
        test_launcher_v7_wiring()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAIL:
        print(f"FAILED: {FAIL} ({', '.join(FAILURES)})")
        sys.exit(1)
    print(f"ALL PASS ({PASS} checks)")


if __name__ == "__main__":
    main()
