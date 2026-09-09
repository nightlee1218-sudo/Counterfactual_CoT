#!/usr/bin/env bash
set -euo pipefail

ROOT=/group/40016/niightli/BAGEL
EXP_DIR="$ROOT/experiments/counterfactual_cot/full_backbone_lora_region1024_fsdp_v3"
MODEL_DIR=/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT
OUTPUT_DIR="$ROOT/experiments/counterfactual_cot/outputs/cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_overfit_v3"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

cd "$ROOT"
python "$EXP_DIR/evaluate.py" \
  --llm-path "$MODEL_DIR" \
  --vae-path "$MODEL_DIR/ae.safetensors" \
  --checkpoint-path "$MODEL_DIR/ema.safetensors" \
  --output-dir "$OUTPUT_DIR" \
  --resume-checkpoint "$OUTPUT_DIR/checkpoints/step_000500.pt" \
  2>&1 | tee -a "$OUTPUT_DIR/evaluate_console.log"
