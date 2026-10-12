#!/usr/bin/env python3
"""Final991-V5 vs BASE full evaluation driver (modules A-F).

Usage:
  python3 run_eval.py --stage plan        # print the full execution plan
  python3 run_eval.py --stage gen-a       # module A: dual-state generation, both models, 3 seeds
  python3 run_eval.py --stage gen-b       # module B: prompt-profile ablation (7 profiles x val)
  python3 run_eval.py --stage gen-c       # module C: sampling sensitivity + merge equivalence
  python3 run_eval.py --stage score       # local metrics: CLIP-I / DINOv2 / LPIPS / MSE + mask decomposition
  python3 run_eval.py --stage stats       # paired stats + bootstrap CI + per-field breakdown
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BAGEL = Path("/group/40016/niightli/BAGEL")
V5 = BAGEL / "experiments/counterfactual_cot/gold40_automask_dual_target_cot1024_fsdp_v5"
SPLITS = BAGEL / "experiments/counterfactual_cot/final991_automask_v5_data/artifacts/splits"
OUT = BAGEL / "experiments/counterfactual_cot/outputs/final991_eval_v1"
MODELS = {
    "base": {
        "llm": Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT"),
        "ckpt": None,  # --skip-trainable-reload
    },
    "final991": {
        "llm": Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT"),
        "ckpt": Path("/dockerdata/niightli/BAGEL/models/final991_automask_v5_latest.pt"),
    },
}
SEEDS = [1234, 2234, 3234]
GEN_KWARGS = dict(num_timesteps=25, cfg_text=4.0, cfg_img=2.0, shift=3.0)


def gen_cmd(model: str, manifest: Path, outdir: Path, seeds: list[int], extra: list[str] | None = None) -> str:
    m = MODELS[model]
    cmd = [
        "python3", str(V5 / "generate_same_seed_states.py"),
        "--manifest", str(manifest),
        "--output-dir", str(outdir),
        "--llm-path", str(m["llm"]),
        "--vae-path", str(m["llm"] / "ae.safetensors"),
        "--base-checkpoint-path", str(m["llm"] / "ema.safetensors"),
        "--seeds", *[str(s) for s in seeds],
        "--num-timesteps", str(GEN_KWARGS["num_timesteps"]),
        "--cfg-text-scale", str(GEN_KWARGS["cfg_text"]),
        "--cfg-img-scale", str(GEN_KWARGS["cfg_img"]),
        "--timestep-shift", str(GEN_KWARGS["shift"]),
        "--overwrite",
    ]
    if m["ckpt"] is None:
        cmd.append("--skip-trainable-reload")
    else:
        cmd += ["--trainable-checkpoint", str(m["ckpt"])]
    if extra:
        cmd += extra
    return " ".join(cmd)


def stage_plan() -> None:
    total_imgs = 0
    print("=== Final991-V5 vs BASE evaluation plan ===")
    print(f"output root: {OUT}")
    for split in ("val", "test"):
        n = sum(1 for _ in open(SPLITS / f"{split}.jsonl"))
        print(f"  {split}: {n} pairs")
    print(f"seeds: {SEEDS}")
    print()
    print("Module A: dual-state generation")
    for model in MODELS:
        for split in ("val", "test"):
            n = sum(1 for _ in open(SPLITS / f"{split}.jsonl"))
            imgs = n * 2 * len(SEEDS)
            total_imgs += imgs
            print(f"  A/{model}/{split}: {n} pairs x 2 states x {len(SEEDS)} seeds = {imgs} imgs -> {OUT}/gen/{model}/{split}")
    print("Module B: prompt ablation (val only, seed 1234)")
    profiles = ["cot_core_v1", "full_without_target_cot_v1", "target_prompt_only_v1",
                "delta_only_v2", "full_paired_v2", "legacy_paired_v1", "wrong_cot_v1"]
    for model in MODELS:
        n = sum(1 for _ in open(SPLITS / "val.jsonl"))
        imgs = n * len(profiles)
        total_imgs += imgs
        print(f"  B/{model}: {n} pairs x {len(profiles)} profiles = {imgs} imgs -> {OUT}/gen/{model}/ablation")
    print("Module C: sampling sensitivity (16 val pairs) + merge equivalence (4 pairs)")
    for model in MODELS:
        imgs = 16 * 2 + 4 * 2
        total_imgs += imgs
        print(f"  C/{model}: {imgs} imgs -> {OUT}/gen/{model}/robustness")
    print(f"Module D: WISE-36 (reuse GEdit/GenEval from 20260918)")
    print(f"Module E: VLM failure attribution (worst-30 target_state cases)")
    print(f"Module F: BASE vs Final991 paired comparison everywhere")
    print(f"\nTotal generation: ~{total_imgs} images (~{total_imgs * 25 / 3600:.1f}h single-GPU, half on 2 GPUs)")


def run_bg(cmd: str, log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w")
    return subprocess.Popen(["bash", "-c", cmd], stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


def stage_gen_a() -> None:
    """Two GPUs, one model each; val then test."""
    jobs = []
    for gpu, model in enumerate(("base", "final991")):
        for split in ("val", "test"):
            outdir = OUT / f"gen/{model}/{split}"
            cmd = f"CUDA_VISIBLE_DEVICES={gpu} " + gen_cmd(model, SPLITS / f"{split}.jsonl", outdir, SEEDS)
            jobs.append((f"A-{model}-{split}", cmd, OUT / f"logs/A-{model}-{split}.log"))
    # serialize per GPU: base runs val->test sequentially, same for final991
    for gpu, model in enumerate(("base", "final991")):
        cmds = []
        for split in ("val", "test"):
            outdir = OUT / f"gen/{model}/{split}"
            cmds.append(f"CUDA_VISIBLE_DEVICES={gpu} " + gen_cmd(model, SPLITS / f"{split}.jsonl", outdir, SEEDS))
        script = " && ".join(cmds)
        log = OUT / f"logs/A-{model}.log"
        proc = run_bg(script, log)
        print(f"[A] launched {model} on gpu{gpu}, log: {log}, pid {proc.pid}")


def stage_gen_b() -> None:
    """Ablation: 7 profiles x val x seed 1234, both models. Uses custom prompt rendering."""
    script = str(Path(__file__).parent / "gen_ablation.py")
    for gpu, model in enumerate(("base", "final991")):
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} python3 {script} --model {model} --out {OUT}/gen/{model}/ablation"
        proc = run_bg(cmd, OUT / f"logs/B-{model}.log")
        print(f"[B] launched {model} on gpu{gpu}, log: {OUT}/logs/B-{model}.log, pid {proc.pid}")


def stage_gen_c() -> None:
    """Sampling sensitivity: 16 val pairs x {25step/4.0,2.0} vs {50step/8.0,1.5}; merge equivalence 4 pairs."""
    # subset manifests
    sub = OUT / "manifests"
    sub.mkdir(parents=True, exist_ok=True)
    val_lines = open(SPLITS / "val.jsonl").readlines()
    (sub / "val16.jsonl").write_text("".join(val_lines[:16]))
    (sub / "val4.jsonl").write_text("".join(val_lines[:4]))
    for gpu, model in enumerate(("base", "final991")):
        # sensitivity: alt sampling on 16 pairs, seed 1234
        outdir = OUT / f"gen/{model}/robustness/alt_sampling"
        cmd = f"CUDA_VISIBLE_DEVICES={gpu} " + gen_cmd(
            model, sub / "val16.jsonl", outdir, [1234],
            extra=["--num-timesteps", "50", "--cfg-text-scale", "8.0", "--cfg-img-scale", "1.5"],
        )
        proc = run_bg(cmd, OUT / f"logs/C-{model}-alt.log")
        print(f"[C] alt-sampling {model} gpu{gpu} pid {proc.pid}")
    # merge equivalence: final991 merged model vs LoRA-loaded, 4 pairs, gpu0 after A finishes
    # (launched manually in stage order; see stats stage)


def stage_score() -> None:
    script = str(Path(__file__).parent / "score_indomain.py")
    proc = run_bg(f"python3 {script} --root {OUT}", OUT / "logs/score.log")
    print(f"[score] pid {proc.pid}")


def stage_stats() -> None:
    script = str(Path(__file__).parent / "stats.py")
    proc = run_bg(f"python3 {script} --root {OUT}", OUT / "logs/stats.log")
    print(f"[stats] pid {proc.pid}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["plan", "gen-a", "gen-b", "gen-c", "score", "stats"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"plan": stage_plan, "gen-a": stage_gen_a, "gen-b": stage_gen_b,
     "gen-c": stage_gen_c, "score": stage_score, "stats": stage_stats}[args.stage]()


if __name__ == "__main__":
    main()
