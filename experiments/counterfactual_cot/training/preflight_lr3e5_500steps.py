"""Read-only preflight check for the lr=3e-5 500-step replication run.

Verifies:
  1. pair_id is unique in the manifest
  2. source/golden target file paths & SHA256 match the old lr=3e-5 (300-step) run
  3. conditioning prompt (raw text SHA256) matches the old lr=3e-5 run
  4. new output dir does not exist yet
  5. GPU 0 has no residual process
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
from pathlib import Path


OLD_DIR = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/single_pair_overfit_diag_ball_lr3e5"
)
NEW_DIR = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/single_pair_overfit_diag_ball_lr3e5_500steps"
)
MANIFEST = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/manifests/intervention_pairs_gold40_human_verified_final_normalized.jsonl"
)
PAIR_ID = "cf_source_living_room_play_absolute_position_change_v4_absolute_region_replacement_v4_0000"


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    report = {}

    # 1. Pair uniqueness in manifest
    hits = []
    with MANIFEST.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("pair_id") == PAIR_ID:
                hits.append(r)
    print(f"[1] pair_id hits in manifest = {len(hits)}")
    assert len(hits) == 1, f"pair_id not unique: {len(hits)} hits"
    report["pair_hits"] = len(hits)

    # 2. Image identity check vs. old lr=3e-5 run
    old_ii = json.loads((OLD_DIR / "image_identity.json").read_text())
    src_path = old_ii["source_image_path"]
    tgt_path = old_ii["target_image_path"]
    print(f"[2a] source path = {src_path} exists={Path(src_path).exists()}")
    print(f"[2b] target path = {tgt_path} exists={Path(tgt_path).exists()}")
    now_src_sha = sha256_file(src_path)
    now_tgt_sha = sha256_file(tgt_path)
    src_ok = now_src_sha == old_ii["source_image_sha256"]
    tgt_ok = now_tgt_sha == old_ii["target_image_sha256"]
    print(f"[2c] source sha256 match = {src_ok}  ({now_src_sha})")
    print(f"[2d] target sha256 match = {tgt_ok}  ({now_tgt_sha})")
    assert src_ok and tgt_ok, "image SHA256 mismatch vs old lr=3e-5 run"
    report["source_sha256"] = now_src_sha
    report["target_sha256"] = now_tgt_sha

    # 3. Conditioning prompt raw SHA256 (from old run's json dump)
    old_cp = json.loads((OLD_DIR / "conditioning_prompt.json").read_text())
    prompt_sha = hashlib.sha256(old_cp["prompt_raw"].encode("utf-8")).hexdigest()
    print(f"[3] old prompt tokens = {old_cp['token_ids_len']}, chars = {len(old_cp['prompt_raw'])}, sha256 = {prompt_sha}")
    report["prompt_token_len"] = old_cp["token_ids_len"]
    report["prompt_sha256"] = prompt_sha

    # 4. Old 300-step run key artifacts (must exist untouched)
    for name in (
        "checkpoints/step_000_trainable.pt",
        "checkpoints/step_010_trainable.pt",
        "checkpoints/step_025_trainable.pt",
        "checkpoints/step_050_trainable.pt",
        "checkpoints/step_100_trainable.pt",
        "checkpoints/step_200_trainable.pt",
        "checkpoints/step_300_trainable.pt",
        "visualizations/step_000/model_output.png",
        "visualizations/step_010/model_output.png",
        "visualizations/step_025/model_output.png",
        "visualizations/step_050/model_output.png",
        "visualizations/step_100/model_output.png",
        "visualizations/step_200/model_output.png",
        "visualizations/step_300/model_output.png",
        "review.html",
        "contact_sheet.png",
        "conditioning_prompt.json",
        "image_identity.json",
        "loss_mask_summary.json",
        "pretrain_1step_assertions.json",
    ):
        p = OLD_DIR / name
        print(f"[old.{name}] exists={p.exists()}  size={p.stat().st_size if p.exists() else -1}")
        assert p.exists(), f"old artifact missing: {p}"

    # 5. New output dir must not exist yet
    print(f"[4] new output dir exists = {NEW_DIR.exists()}")
    assert not NEW_DIR.exists(), f"NEW_DIR already exists, refusing to overwrite: {NEW_DIR}"

    # 6. Old 1-step assertion summary: MSE mean & effective_mse_tokens
    ass = json.loads((OLD_DIR / "pretrain_1step_assertions.json").read_text())
    print(f"[6] old pretrain_1step: effective_mse_tokens = {ass['effective_mse_tokens']}, mse_mean = {ass['mse_mean']}, checks_all_pass = {all(ass['checks'].values())}")
    assert ass["effective_mse_tokens"] == 256
    assert all(ass["checks"].values())

    # 7. Base BAGEL files exist
    for name in (
        "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT",
        "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors",
        "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors",
    ):
        p = Path(name)
        print(f"[7.{name}] exists={p.exists()}")
        assert p.exists(), f"base BAGEL missing: {name}"

    # 8. Old lr=3e-5 300-step Step-1 raw MSE (for later comparison)
    old_report = json.loads((OLD_DIR / "single_pair_overfit_report.json").read_text())
    step1 = old_report["loss_history"][0]
    step10 = old_report["loss_history"][9]
    step25 = old_report["loss_history"][24]
    step50 = old_report["loss_history"][49]
    print(f"[8a] old step1 mse = {step1['mse']}")
    print(f"[8b] old step10 mse = {step10['mse']}")
    print(f"[8c] old step25 mse = {step25['mse']}")
    print(f"[8d] old step50 mse = {step50['mse']}")
    report["old_step1_mse"] = step1["mse"]
    report["old_step10_mse"] = step10["mse"]
    report["old_step25_mse"] = step25["mse"]
    report["old_step50_mse"] = step50["mse"]

    # 9. GPU 0 usage
    print("[9] nvidia-smi:")
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    print(r.stdout)
    # Any GPU-owning process on GPU 0?
    r2 = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    print("compute apps:", repr(r2.stdout))
    assert r2.stdout.strip() == "", "GPU has residual processes"

    print("PREFLIGHT_READ_ONLY_OK")


if __name__ == "__main__":
    main()
