#!/usr/bin/env bash
# Launch the lr=3e-5 500-step replication run for the single-pair overfit
# diagnostic. Everything except lr / steps / save_steps / output_dir must
# stay identical to the old lr=3e-5 300-step run.
set -euo pipefail

OUT_DIR="/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/single_pair_overfit_diag_ball_lr3e5_500steps"

if [ -e "${OUT_DIR}" ]; then
  echo "ERROR: output dir already exists: ${OUT_DIR}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/logs"

REPO_ROOT="/group/40016/niightli/BAGEL"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec python3 -u \
  "${REPO_ROOT}/experiments/counterfactual_cot/training/single_pair_overfit_diag.py" \
  --output-dir "${OUT_DIR}" \
  --pair-id "cf_source_living_room_play_absolute_position_change_v4_absolute_region_replacement_v4_0000" \
  --manifest "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/manifests/intervention_pairs_gold40_human_verified_final_normalized.jsonl" \
  --llm-path "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT" \
  --vae-path "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors" \
  --checkpoint-path "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors" \
  --variant paired_cf_mvp \
  --steps 500 \
  --save-steps 0 10 25 50 100 200 300 400 500 \
  --image-size 256 \
  --num-timesteps 25 \
  --precision bf16 \
  --lr 3e-5 \
  --max-grad-norm 1.0 \
  --seed 1234 \
  --generation-seed 1234 \
  --expected-num-tokens 1536 \
  --max-num-tokens-per-sample 1536 \
  --max-num-tokens 1536 \
  --trainable-name-regex '^(time_embedder|latent_pos_embed|vit_pos_embed|vae2llm|llm2vae|connector)\.' \
  2>&1 | tee "${OUT_DIR}/logs/run.log"
