#!/usr/bin/env bash
# E2 token-localization probe: base vs final991 (LoRA direct-load) — launcher.
#
# Usage on the 2x A100 GPU machine (mounts /group):
#   bash /group/40016/niightli/BAGEL/experiments/counterfactual_cot/e2_token_probe/run_e2.sh
# Optional stages:
#   bash run_e2.sh smoke    # 1 pair, 1 probe step on GPU1 (final991) — pipeline check
#   bash run_e2.sh probe|analyze|viz|all
#
# Stages:
#   probe  : GPU0=base, GPU1=final991 in parallel (each ~27GB VRAM).
#            Faithful 25-step CFG-4.0/2.0 generation; capture at steps 0/11/23.
#   analyze: paired bootstrap stats per layer (needs both probes done)
#   viz    : attention heatmap grids overlaid with the edit mask
# Watchdog rewrites ../wb_e2_progress.txt every 30s.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFCOT="$(cd "$HERE/.." && pwd)"
OUT="$CFCOT/outputs/e2_token_probe"
PROGRESS="$CFCOT/wb_e2_progress.txt"
PY=python3
MODEL_DIR="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT"
CKPT="$CFCOT/outputs/final991_automask_v5_train2973/checkpoints/latest.pt"
STAGE="${1:-all}"

mkdir -p "$OUT/logs"

write_progress() {
  local tmp="$PROGRESS.tmp"
  { echo "updated: $(date '+%F %T') (stage=$1)"; shift
    for line in "$@"; do echo "$line"; done; } > "$tmp" && mv "$tmp" "$PROGRESS"
}

die() { write_progress "ERROR" "error: $*"; echo "[E2 FATAL] $*" >&2; exit 1; }

preflight() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found (need GPU machine)"
  nvidia-smi -L | head -2
  [ -f "$MODEL_DIR/ema.safetensors" ] && [ -f "$MODEL_DIR/ae.safetensors" ] || die "BAGEL weights missing: $MODEL_DIR"
  [ -f "$CKPT" ] || die "final991 checkpoint missing: $CKPT"
  [ -f "$CFCOT/final991_automask_v5_data/artifacts/splits/val.jsonl" ] || die "val split missing"
  echo "[E2] preflight OK"
}

pairs_done() { # manifest
  [ -f "$1" ] && grep -c '"status": "ok"' "$1" 2>/dev/null || echo 0
}

watch_probe() { # waits for both pids
  local pid0=$1 pid1=$2
  while kill -0 "$pid0" 2>/dev/null || kill -0 "$pid1" 2>/dev/null; do
    local b f bt ft
    b=$(pairs_done "$OUT/probe_base/probe_manifest.jsonl")
    f=$(pairs_done "$OUT/probe_final991/probe_manifest.jsonl")
    bt=$(grep -c '"pair_id"' "$CFCOT/final991_automask_v5_data/artifacts/splits/val.jsonl" 2>/dev/null || echo 51)
    write_progress "probe" \
      "base     : ${b}/${bt} pairs" \
      "final991 : ${f}/${bt} pairs" \
      "logs: $OUT/logs/probe_base.log , $OUT/logs/probe_final991.log"
    sleep 30
  done
  local ok=1
  grep -q '"event": "all_done"' "$OUT/logs/probe_base.log" 2>/dev/null || ok=0
  grep -q '"event": "all_done"' "$OUT/logs/probe_final991.log" 2>/dev/null || ok=0
  if [ "$ok" = 1 ]; then
    write_progress "probe_done" "both models finished probing"
  else
    write_progress "ERROR" "probe ended without all_done; check $OUT/logs/"
    return 1
  fi
}

stage_smoke() {
  echo "[E2] smoke: final991 on GPU1, 1 pair, single probe step"
  CUDA_VISIBLE_DEVICES=1 $PY "$HERE/probe.py" --model final991 \
    --out-dir "$OUT/smoke_final991" --limit 1 --probe-steps 11 \
    2>&1 | tee "$OUT/logs/smoke.log" | grep -E "event|Error|error" | tail -10
  ls -la "$OUT/smoke_final991/maps/" 2>/dev/null | head -4
}

stage_probe() {
  if grep -q '"event": "all_done"' "$OUT/logs/probe_base.log" 2>/dev/null \
     && grep -q '"event": "all_done"' "$OUT/logs/probe_final991.log" 2>/dev/null; then
    echo "[E2] probe already complete, skip"; return 0
  fi
  echo "[E2] launching dual-GPU probing (GPU0=base GPU1=final991) ..."
  CUDA_VISIBLE_DEVICES=0 nohup $PY "$HERE/probe.py" --model base \
    --out-dir "$OUT/probe_base" > "$OUT/logs/probe_base.log" 2>&1 &
  PID0=$!
  CUDA_VISIBLE_DEVICES=1 nohup $PY "$HERE/probe.py" --model final991 \
    --out-dir "$OUT/probe_final991" > "$OUT/logs/probe_final991.log" 2>&1 &
  PID1=$!
  echo "[E2] probe pids: base=$PID0 final991=$PID1"
  watch_probe "$PID0" "$PID1"
}

stage_analyze() {
  write_progress "analyze" "paired bootstrap stats running ..."
  $PY "$HERE/analyze.py" --probe-root "$OUT" --out-dir "$OUT/scores" \
    > "$OUT/logs/analyze.log" 2>&1 || die "analyze failed; see $OUT/logs/analyze.log"
  echo "[E2] analyze done -> $OUT/scores/e2_layer_table.md"
}

stage_viz() {
  write_progress "viz" "attention heatmap grids ..."
  $PY "$HERE/visualize.py" --probe-root "$OUT" --out-dir "$OUT/figures" \
    > "$OUT/logs/viz.log" 2>&1 || echo "[E2] viz failed (non-fatal); see log"
  write_progress "DONE" \
    "results: $OUT/scores/e2_layer_table.md , $OUT/scores/e2_stats_summary.json" \
    "figures: $OUT/figures/{base,final991}/*_grid.png"
  echo "[E2] ALL DONE"
}

preflight
case "$STAGE" in
  smoke) stage_smoke ;;
  probe) stage_probe ;;
  analyze) stage_analyze ;;
  viz) stage_viz ;;
  all)
    stage_probe || die "probe failed"
    stage_analyze
    stage_viz
    ;;
  *) die "unknown stage: $STAGE" ;;
esac
