#!/usr/bin/env bash
set -euo pipefail

ROOT=/group/40016/niightli/BAGEL
EXP_DIR="$ROOT/experiments/counterfactual_cot/full_backbone_lora_v3_strict_step0_step500_control"
OUTPUT_DIR="$ROOT/experiments/counterfactual_cot/outputs/cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_strict_step0_step500_control_v1"
GPU_ID="${GPU_ID:-0}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"
mkdir -p "$OUTPUT_DIR"

python "$EXP_DIR/generate_controls.py" \
  --output-dir "$OUTPUT_DIR" \
  --gpu-id "$GPU_ID" \
  2>&1 | tee -a "$OUTPUT_DIR/generation_console.log"

python "$EXP_DIR/analyze_controls.py" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee -a "$OUTPUT_DIR/analysis_console.log"
