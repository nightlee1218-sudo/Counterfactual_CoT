#!/usr/bin/env bash
# E3 v7: AutoMask dual SFT + thrust-fixed alignment loss — launcher.
#
# Usage on the 2x A100 GPU machine (mounts /group):
#   bash /group/40016/niightli/BAGEL/experiments/counterfactual_cot/e3_align_v7/run_v7.sh
# Optional stages:
#   bash run_v7.sh selftest   # CPU mock suite (no GPU needed)
#   bash run_v7.sh preflight  # GPU machine: full manifest/objective audit, no training
#   bash run_v7.sh smoke     # 2 optimizer steps into a throwaway output dir
#   bash run_v7.sh train     # full 2973-step run in the background + watchdog
#   bash run_v7.sh status    # tail the last training log lines + progress file
#
# v7 defaults: lambda=0.3, tau=0.1 (margin scale 1/tau=10), strength-reg on
# (w_s=0.05, replaces v6's edit-weight rebalancing; region weights back to
# 0.5/0.5), plateau-guard on (1000-step window around 0.69+-0.005 -> ALERT +
# emergency checkpoint + stop), align layers 13/6/16/14, overshoot debias on,
# general replay off.
# Override with EXTRA_ARGS, e.g.:
#   EXTRA_ARGS="--align-tau 0.05 --strength-reg off" bash run_v7.sh smoke
#   EXTRA_ARGS="--init-trainable-from <final991 latest.pt>" bash run_v7.sh train
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFCOT="$(cd "$HERE/.." && pwd)"
OUT="$CFCOT/outputs/gold40_automask_v7_align_fsdp_full_shard"
SMOKE_OUT="$CFCOT/outputs/gold40_automask_v7_align_smoke"
PROGRESS="$CFCOT/wb_e3_progress_v7.txt"
PY=python3
MODEL_DIR="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT"
MANIFEST="$CFCOT/final991_automask_v5_data/artifacts/splits/train.jsonl"
STEPS="${E3_STEPS:-2973}"
STAGE="${1:-train}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$OUT/logs" 2>/dev/null || true

write_progress() {
  local tmp="$PROGRESS.tmp"
  { echo "updated: $(date '+%F %T') (stage=$1)"; shift
    for line in "$@"; do echo "$line"; done; } > "$tmp" && mv "$tmp" "$PROGRESS"
}

die() { write_progress "ERROR" "error: $*"; echo "[E3 FATAL] $*" >&2; exit 1; }

gpu_preflight() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found (need the 2x A100 machine)"
  nvidia-smi --L | head -2
  [ -f "$MODEL_DIR/ema.safetensors" ] && [ -f "$MODEL_DIR/ae.safetensors" ] || die "BAGEL weights missing: $MODEL_DIR"
  [ -f "$MANIFEST" ] || die "train manifest missing: $MANIFEST"
}

launch() { # out_dir steps extra...
  local out_dir="$1"; shift
  local steps="$1"; shift
  mkdir -p "$out_dir" "$out_dir/logs"
  local log
  log="$out_dir/logs/train_$(date '+%Y%m%d_%H%M%S').log"
  echo "[E3] torchrun 2-GPU -> $log"
  nohup $PY -m torch.distributed.run --standalone --nproc_per_node=2 \
    "$HERE/train_v7.py" --activation-checkpointing \
      --manifest "$MANIFEST" --output-dir "$out_dir" --steps "$steps" \
      --progress-file "$PROGRESS" $EXTRA_ARGS "$@" \
    > "$log" 2>&1 &
  echo $! > "$out_dir/train.pid"
  echo "[E3] pid=$(cat "$out_dir/train.pid") log=$log"
}

watch_train() { # out_dir
  local out_dir="$1"
  local pid
  pid=$(cat "$out_dir/train.pid" 2>/dev/null) || die "no train.pid under $out_dir"
  while kill -0 "$pid" 2>/dev/null; do
    # plateau ALERT path: the trainer writes stage=ALERT into the progress
    # file itself; stop watchdog overwrites so the alert survives.
    if [ -f "$out_dir/final_report.json" ] \
      && grep -q '"status": "plateau_alert_stopped"' "$out_dir/final_report.json" 2>/dev/null; then
      break
    fi
    local log last step peak align margin strength
    log=$(ls -t "$out_dir/logs/"train_*.log 2>/dev/null | head -1)
    last=$(grep '"event": "macro_step"' "$log" 2>/dev/null | tail -1)
    step=$(echo "$last" | grep -o '"step": [0-9]*' | cut -d' ' -f2 | tail -1)
    peak=$(echo "$last" | grep -o '"gpu_peak_reserved_gib": [0-9.]*' | cut -d' ' -f2 | tail -1)
    margin=$(echo "$last" | grep -o '"target_state_align_margin": [-0-9.e]*' | cut -d' ' -f2 | tail -1)
    align=$(echo "$last" | grep -o '"target_state_align_loss": [0-9.e]*' | cut -d' ' -f2 | tail -1)
    strength=$(echo "$last" | grep -o '"target_state_strength_est": [-0-9.e]*' | cut -d' ' -f2 | tail -1)
    write_progress "train" \
      "step: ${step:-0}/${STEPS}  peak_gib: ${peak:-n/a}" \
      "target align: loss=${align:-n/a} margin=${margin:-n/a}" \
      "strength_est: ${strength:-n/a}" \
      "log: $log"
    sleep 30
  done
  local log
  log=$(ls -t "$out_dir/logs/"train_*.log 2>/dev/null | head -1)
  if grep -q '"status": "plateau_alert_stopped"' "$out_dir/final_report.json" 2>/dev/null; then
    # the trainer already wrote stage=ALERT into $PROGRESS; surface it and
    # stop without overwriting the alert block.
    echo "[E3] PLATEAU ALERT — training stopped by plateau_guard; see $PROGRESS"
    echo "[E3] alert details: $out_dir/final_report.json (plateau_alert) + emergency latest.pt"
    tail -20 "$log" 2>/dev/null
    return 0
  fi
  if grep -q '"status": "completed"' "$out_dir/final_report.json" 2>/dev/null; then
    write_progress "DONE" "final_report: $out_dir/final_report.json" "log: $log"
    echo "[E3] TRAIN DONE"
  else
    write_progress "ERROR" "trainer exited without final_report; inspect $log"
    tail -20 "$log" 2>/dev/null
    return 1
  fi
}

stage_selftest() {
  write_progress "selftest" "CPU mock suite running ..."
  $PY "$HERE/selftest_v7.py" 2>&1 | tee "$HERE/selftest_last.log" | grep -E "^(PASS|FAIL|ALL)" \
    || die "selftest failed; see $HERE/selftest_last.log"
  write_progress "selftest_done" "all CPU mock tests passed"
}

stage_preflight() {
  gpu_preflight
  write_progress "preflight" "manifest + objective audit running ..."
  $PY "$HERE/train_v7.py" --preflight-only --manifest "$MANIFEST" --output-dir "$OUT" $EXTRA_ARGS \
    2>&1 | tee "$OUT/logs/preflight.log" | tail -5 \
    || die "preflight failed; see $OUT/logs/preflight.log"
  echo "[E3] preflight OK -> $OUT/preflight.json"
}

stage_smoke() {
  gpu_preflight
  write_progress "smoke" "2 optimizer steps, fresh output dir ..."
  launch "$SMOKE_OUT" 2 --checkpoint-every 2 --resume none
  watch_train "$SMOKE_OUT" || die "smoke failed"
  echo "[E3] smoke artifacts: $SMOKE_OUT/final_report.json"
}

stage_train() {
  gpu_preflight
  if [ -f "$OUT/final_report.json" ] && grep -q '"status": "completed"' "$OUT/final_report.json"; then
    echo "[E3] already complete, skip"; write_progress "DONE" "final_report: $OUT/final_report.json"; return 0
  fi
  launch "$OUT" "$STEPS" --checkpoint-every 200
  watch_train "$OUT" || die "training failed"
}

stage_status() {
  echo "== progress file =="; cat "$PROGRESS" 2>/dev/null || echo "(none)"
  local log
  log=$(ls -t "$OUT/logs/"train_*.log 2>/dev/null | head -1)
  if [ -n "${log:-}" ]; then
    echo "== last 5 macro steps ($log) =="
    grep '"event": "macro_step"' "$log" | tail -5
  else
    echo "(no train log under $OUT/logs)"
  fi
}

case "$STAGE" in
  selftest) stage_selftest ;;
  preflight) stage_preflight ;;
  smoke) stage_smoke ;;
  train) stage_train ;;
  status) stage_status ;;
  *) die "unknown stage: $STAGE (selftest|preflight|smoke|train|status)" ;;
esac
