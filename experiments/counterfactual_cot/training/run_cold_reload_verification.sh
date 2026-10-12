#!/usr/bin/env bash
# Cold-reload validation orchestrator.
#
# Runs two independent GPU verifier subprocesses (A then B), each on the SAME
# saved step_002 trainable checkpoint, then a pure-CPU comparator that produces
# cold_reload_verification_report.json.
#
# Rules honored:
#   * Never re-runs training.
#   * Never touches step_002_trainable.pt (read-only).
#   * Verifier A and B are separate processes; between them we wait for exit
#     and confirm no residual GPU allocation for the current user.
#   * All *generated artefacts* live under /group (never /dockerdata).
#   * Base weights (ema.safetensors, ae.safetensors) MUST live under
#     /dockerdata -- the exact same frozen paths that the training run used.
#     No auto-fallback to /group is permitted. If /dockerdata is missing or
#     the base sha256 does not match the training-side sha256, the whole
#     verification is aborted.
#
# Env override examples (rarely needed):
#   TRAIN_DIR=/path/to/output OUTPUT_ROOT=/path/to/cold_reload \
#   bash run_cold_reload_verification.sh

set -euo pipefail

REPO_ROOT="/group/40016/niightli/BAGEL"
TRAINING_DIR="${TRAIN_DIR:-$REPO_ROOT/experiments/counterfactual_cot/outputs/lora_2step_smoke_lr3e5_L24-27_r8}"
TRAINABLE_CKPT="${TRAINABLE_CKPT:-$TRAINING_DIR/checkpoints/step_002_trainable.pt}"
TRAINING_REPORT="${TRAINING_REPORT:-$TRAINING_DIR/lora_2step_smoke_report.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$TRAINING_DIR/cold_reload}"

# --------------------------------------------------------------------------
# Base weights are FROZEN under /dockerdata (the same paths training used).
# No auto-fallback to /group is permitted. If /dockerdata is missing or the
# sha256 differs from the training-side base, the orchestrator must fail hard.
# --------------------------------------------------------------------------
LLM_PATH="${LLM_PATH:-/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT}"
VAE_PATH="${VAE_PATH:-$LLM_PATH/ae.safetensors}"
CKPT_PATH="${CKPT_PATH:-$LLM_PATH/ema.safetensors}"

case "$LLM_PATH" in
    /dockerdata/*) : ;;
    *)
        echo "[orchestrator][fatal] LLM_PATH must live under /dockerdata (got: $LLM_PATH)" >&2
        exit 2 ;;
esac
case "$CKPT_PATH" in
    /dockerdata/*) : ;;
    *)
        echo "[orchestrator][fatal] CKPT_PATH must live under /dockerdata (got: $CKPT_PATH)" >&2
        exit 2 ;;
esac

if [[ ! -d "$LLM_PATH" ]]; then
    echo "[orchestrator][fatal] BAGEL-7B-MoT model directory missing: $LLM_PATH" >&2
    exit 2
fi
if [[ ! -f "$CKPT_PATH" ]]; then
    echo "[orchestrator][fatal] base ema.safetensors missing at: $CKPT_PATH" >&2
    exit 2
fi
if [[ ! -f "$VAE_PATH" ]]; then
    echo "[orchestrator][fatal] base ae.safetensors missing at: $VAE_PATH" >&2
    exit 2
fi

echo "[orchestrator] llm_path           = $LLM_PATH"
echo "[orchestrator] vae_path           = $VAE_PATH"
echo "[orchestrator] base_ckpt_path     = $CKPT_PATH"

VERIF_A_DIR="$OUTPUT_ROOT/verifier_A"
VERIF_B_DIR="$OUTPUT_ROOT/verifier_B"
LOG_A="$OUTPUT_ROOT/verifier_A.log"
LOG_B="$OUTPUT_ROOT/verifier_B.log"
REPORT="$OUTPUT_ROOT/cold_reload_verification_report.json"

mkdir -p "$VERIF_A_DIR" "$VERIF_B_DIR"

echo "[orchestrator] training_dir       = $TRAINING_DIR"
echo "[orchestrator] trainable_ckpt     = $TRAINABLE_CKPT"
echo "[orchestrator] training_report    = $TRAINING_REPORT"
echo "[orchestrator] output_root        = $OUTPUT_ROOT"

if [[ ! -f "$TRAINABLE_CKPT" ]]; then
    echo "[orchestrator][fatal] trainable checkpoint missing: $TRAINABLE_CKPT" >&2
    exit 2
fi
if [[ ! -f "$TRAINING_REPORT" ]]; then
    echo "[orchestrator][fatal] training report missing: $TRAINING_REPORT" >&2
    exit 2
fi

# Guardrail: refuse to write anywhere under /dockerdata.
case "$OUTPUT_ROOT" in
    /dockerdata*)
        echo "[orchestrator][fatal] OUTPUT_ROOT under /dockerdata is not allowed: $OUTPUT_ROOT" >&2
        exit 2 ;;
esac

_gpu_alloc_summary() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "[orchestrator] nvidia-smi (memory & processes):"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits || true
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits || true
    else
        echo "[orchestrator] nvidia-smi not available"
    fi
}

_wait_for_gpu_release() {
    local max_seconds="${1:-30}"
    local elapsed=0
    while (( elapsed < max_seconds )); do
        # Look for any lingering compute process from this user with the
        # verifier script name.
        local mine
        mine="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null | grep -c "cold_reload_verifier" || true)"
        if [[ "$mine" == "0" ]]; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo "[orchestrator][warn] verifier process still listed by nvidia-smi after ${max_seconds}s"
    return 0
}

pushd "$REPO_ROOT" >/dev/null

VERIFIER="experiments/counterfactual_cot/training/cold_reload_verifier.py"
COMPARER="experiments/counterfactual_cot/training/cold_reload_compare.py"

########################################
# Verifier A
########################################
echo "[orchestrator] ============ Verifier A ============"
_gpu_alloc_summary

set +e
python "$VERIFIER" \
    --label A \
    --trainable-checkpoint "$TRAINABLE_CKPT" \
    --output-dir "$VERIF_A_DIR" \
    --llm-path "$LLM_PATH" \
    --vae-path "$VAE_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    2>&1 | tee "$LOG_A"
rc_a="${PIPESTATUS[0]}"
set -e
echo "[orchestrator] Verifier A exit code = $rc_a"

echo "[orchestrator] waiting for GPU release before Verifier B ..."
_wait_for_gpu_release 30
_gpu_alloc_summary

########################################
# Verifier B (only if A wrote its report; we still want to run B for the
# comparison even if A's checks failed, to know whether B agrees).
########################################
echo "[orchestrator] ============ Verifier B ============"

set +e
python "$VERIFIER" \
    --label B \
    --trainable-checkpoint "$TRAINABLE_CKPT" \
    --output-dir "$VERIF_B_DIR" \
    --llm-path "$LLM_PATH" \
    --vae-path "$VAE_PATH" \
    --checkpoint-path "$CKPT_PATH" \
    2>&1 | tee "$LOG_B"
rc_b="${PIPESTATUS[0]}"
set -e
echo "[orchestrator] Verifier B exit code = $rc_b"

echo "[orchestrator] waiting for GPU release before comparator ..."
_wait_for_gpu_release 30
_gpu_alloc_summary

########################################
# Comparator (CPU only)
########################################
echo "[orchestrator] ============ Comparator (CPU) ============"

set +e
python "$COMPARER" \
    --verifier-a-dir "$VERIF_A_DIR" \
    --verifier-b-dir "$VERIF_B_DIR" \
    --training-report "$TRAINING_REPORT" \
    --output "$REPORT" \
    --allow-training-smoke-failed-status \
    2>&1 | tee "$OUTPUT_ROOT/compare.log"
rc_c="${PIPESTATUS[0]}"
set -e

echo "[orchestrator] Comparator exit code = $rc_c"
echo "[orchestrator] Verifier A rc=$rc_a  B rc=$rc_b  Comparator rc=$rc_c"
echo "[orchestrator] Report: $REPORT"

popd >/dev/null

# The comparator exit code is authoritative for the overall gate.
exit "$rc_c"
