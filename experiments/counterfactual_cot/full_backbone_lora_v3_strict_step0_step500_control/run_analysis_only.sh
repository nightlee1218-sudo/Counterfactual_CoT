#!/usr/bin/env bash
set -euo pipefail

ROOT=/group/40016/niightli/BAGEL
EXP_DIR="$ROOT/experiments/counterfactual_cot/full_backbone_lora_v3_strict_step0_step500_control"
OUTPUT_DIR="$ROOT/experiments/counterfactual_cot/outputs/cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_strict_step0_step500_control_v1"

cd "$ROOT"
python "$EXP_DIR/analyze_controls.py" --output-dir "$OUTPUT_DIR" \
  2>&1 | tee -a "$OUTPUT_DIR/analysis_console.log"
