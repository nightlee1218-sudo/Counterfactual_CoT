#!/usr/bin/env bash
set -euo pipefail
ROOT=/group/40016/niightli/BAGEL
EXP_DIR="$ROOT/experiments/counterfactual_cot/gold40_dual_target_cot1024_fsdp_v4"
MODEL_DIR="${MODEL_DIR:-/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT}"
MANIFEST="${MANIFEST:?Set MANIFEST to an approved reviewed Gold40 train6 JSONL}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/experiments/counterfactual_cot/outputs/gold40_dual_target_smoke_v4}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
cd "$ROOT"
torchrun --standalone --nproc_per_node=2 "$EXP_DIR/train.py" --manifest "$MANIFEST" --llm-path "$MODEL_DIR" --vae-path "$MODEL_DIR/ae.safetensors" --checkpoint-path "$MODEL_DIR/ema.safetensors" --output-dir "$OUTPUT_DIR" --steps 2 --checkpoint-every 1 --activation-checkpointing
