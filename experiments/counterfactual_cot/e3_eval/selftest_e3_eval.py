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
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAIL:
        print(f"FAILED: {FAIL} ({', '.join(FAILURES)})")
        sys.exit(1)
    print(f"ALL PASS ({PASS} checks)")


if __name__ == "__main__":
    main()
