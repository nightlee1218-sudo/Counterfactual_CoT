#!/usr/bin/env bash
set -euo pipefail
# Formal run: 2973 macro steps (~3 epochs over the 896-pair train split).
# One macro step = source_state + target_state branches, each backward scaled 0.5,
# shared pack seed / raw timestep / controlled noise, then a single AdamW update.
ROOT=/group/40016/niightli/BAGEL
EXP_DIR="$ROOT/experiments/counterfactual_cot/gold40_automask_dual_target_cot1024_fsdp_v5"
MODEL_DIR="${MODEL_DIR:-/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT}"
if [ ! -f "$MODEL_DIR/ema.safetensors" ] || [ ! -f "$MODEL_DIR/ae.safetensors" ] || [ ! -f "$MODEL_DIR/llm_config.json" ]; then
  echo "[fatal] MODEL_DIR incomplete: $MODEL_DIR (override with MODEL_DIR=...)" >&2; exit 2
fi
MANIFEST="${MANIFEST:-$ROOT/experiments/counterfactual_cot/final991_automask_v5_data/artifacts/splits/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/counterfactual_cot/outputs/final991_automask_v5_train2973}"
STEPS="${STEPS:-2973}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
RESUME="${RESUME:-auto}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
cd "$ROOT"
torchrun --standalone --nproc_per_node=2 "$EXP_DIR/train.py" \
  --manifest "$MANIFEST" \
  --llm-path "$MODEL_DIR" \
  --vae-path "$MODEL_DIR/ae.safetensors" \
  --checkpoint-path "$MODEL_DIR/ema.safetensors" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" --checkpoint-every "$CHECKPOINT_EVERY" --resume "$RESUME" --activation-checkpointing
