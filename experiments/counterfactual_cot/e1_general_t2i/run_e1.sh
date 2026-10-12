#!/usr/bin/env bash
# E1 general-T2I benchmark: base vs final991 (LoRA direct-load) — single-command launcher.
#
# Usage on the 2x A100 GPU machine (which mounts /group):
#   bash /group/40016/niightli/BAGEL/experiments/counterfactual_cot/e1_general_t2i/run_e1.sh
# Optional stages:
#   bash run_e1.sh smoke   # 2 items on GPU1 (final991) to validate the pipeline
#   bash run_e1.sh gen|score|stats|sheet|all
#
# Stages (all resumable):
#   gen    : GPU0=base, GPU1=final991, seed 1234, 25 steps, CFG 4.0/2.0, 1024px
#   score  : CLIP-T (ViT-L/14, local weights) via geneval_eval venv python
#   stats  : per-item paired diff + bootstrap 95% CI (20k resamples)
#   sheet  : contact sheets per subset + overview
#   vqa    : optional Qwen3-VL faithfulness on capped subset (auto-skipped w/o token)
# Watchdog rewrites ../wb_e1_progress.txt every 30s.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFCOT="$(cd "$HERE/.." && pwd)"
OUT="$CFCOT/outputs/e1_general_t2i"
PROGRESS="$CFCOT/wb_e1_progress.txt"
PY_SYS=python3
PY_CLIP="$CFCOT/../../venvs/geneval_eval/bin/python"
MODEL_DIR="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT"
CKPT="$CFCOT/outputs/final991_automask_v5_train2973/checkpoints/latest.pt"
SEED=1234
STAGE="${1:-all}"

mkdir -p "$OUT/logs"

write_progress() { # stage, extra_json_line
  local tmp="$PROGRESS.tmp"
  {
    echo "updated: $(date '+%F %T') (stage=$1)"
    shift
    for line in "$@"; do echo "$line"; done
  } > "$tmp" && mv "$tmp" "$PROGRESS"
}

die() { write_progress "ERROR" "error: $*"; echo "[E1 FATAL] $*" >&2; exit 1; }

preflight() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found (need GPU machine)"
  nvidia-smi -L | head -2
  [ -f "$MODEL_DIR/ema.safetensors" ] && [ -f "$MODEL_DIR/ae.safetensors" ] || die "BAGEL weights missing: $MODEL_DIR"
  [ -f "$CKPT" ] || die "final991 checkpoint missing: $CKPT"
  [ -x "$PY_CLIP" ] || die "open_clip venv python missing: $PY_CLIP"
  ls "$HERE/prompts/"*.jsonl >/dev/null || die "prompts missing; run build_prompts.py first"
  echo "[E1] preflight OK"
}

parse_last_event() { # logfile, field
  grep '"event": "generated"' "$1" 2>/dev/null | tail -1 | $PY_SYS -c "import json,sys; r=json.loads(sys.stdin.read() or '{}'); print(r.get('$2','?'))" 2>/dev/null || echo "?"
}

watch_gen() { # waits for both gen pids; updates progress
  local pid0=$1 pid1=$2
  while kill -0 "$pid0" 2>/dev/null || kill -0 "$pid1" 2>/dev/null; do
    local b_done b_total b_eta f_done f_total f_eta
    b_done=$(parse_last_event "$OUT/logs/gen_base.log" done)
    b_total=$(parse_last_event "$OUT/logs/gen_base.log" total)
    b_eta=$(parse_last_event "$OUT/logs/gen_base.log" eta_min)
    f_done=$(parse_last_event "$OUT/logs/gen_final991.log" done)
    f_total=$(parse_last_event "$OUT/logs/gen_final991.log" total)
    f_eta=$(parse_last_event "$OUT/logs/gen_final991.log" eta_min)
    write_progress "gen" \
      "base     : ${b_done}/${b_total}  eta ${b_eta} min" \
      "final991 : ${f_done}/${f_total}  eta ${f_eta} min" \
      "logs: $OUT/logs/gen_base.log , $OUT/logs/gen_final991.log"
    sleep 30
  done
  local ok=1
  [ -f "$OUT/gen/base/generation_report.json" ] || ok=0
  [ -f "$OUT/gen/final991/generation_report.json" ] || ok=0
  if [ "$ok" = 1 ]; then
    write_progress "gen_done" "both models finished generation"
  else
    write_progress "ERROR" "generation ended without reports; check $OUT/logs/"
    echo "[E1] generation incomplete — see logs" >&2
    return 1
  fi
}

stage_gen() {
  if [ -f "$OUT/gen/base/generation_report.json" ] && [ -f "$OUT/gen/final991/generation_report.json" ]; then
    echo "[E1] gen already complete, skip"; return 0
  fi
  echo "[E1] launching dual-GPU generation (GPU0=base GPU1=final991) ..."
  CUDA_VISIBLE_DEVICES=0 nohup $PY_SYS "$HERE/generate_t2i.py" --model base \
    --output-dir "$OUT/gen/base" --seed $SEED \
    > "$OUT/logs/gen_base.log" 2>&1 &
  PID0=$!
  CUDA_VISIBLE_DEVICES=1 nohup $PY_SYS "$HERE/generate_t2i.py" --model final991 \
    --output-dir "$OUT/gen/final991" --seed $SEED \
    > "$OUT/logs/gen_final991.log" 2>&1 &
  PID1=$!
  echo "[E1] gen pids: base=$PID0 final991=$PID1"
  watch_gen "$PID0" "$PID1"
}

stage_smoke() {
  echo "[E1] smoke: final991 on GPU1, 2 items"
  CUDA_VISIBLE_DEVICES=1 $PY_SYS "$HERE/generate_t2i.py" --model final991 \
    --output-dir "$OUT/smoke/final991" --seed $SEED --max-items 2 \
    2>&1 | tee "$OUT/logs/smoke.log" | tail -5
  ls -la "$OUT/smoke/final991/images/"*/*.png
}

stage_score() {
  if [ -f "$OUT/scores/clipT_summary.json" ] && grep -q final991 "$OUT/scores/clipT_summary.json" 2>/dev/null; then
    echo "[E1] clip score exists, skip"; return 0
  fi
  write_progress "score" "CLIP-T scoring (ViT-L/14) running ..."
  CUDA_VISIBLE_DEVICES=0 $PY_CLIP "$HERE/score_clip.py" --gen-root "$OUT/gen" --out-dir "$OUT/scores" \
    > "$OUT/logs/score_clip.log" 2>&1 || die "CLIP scoring failed; see $OUT/logs/score_clip.log"
  echo "[E1] CLIP-T done"
}

stage_vqa() {
  if [ -z "${ENV_VENUS_OPENAPI_SECRET_ID:-}" ]; then
    echo "[E1] VQA skipped (no ENV_VENUS_OPENAPI_SECRET_ID)"; return 0
  fi
  echo "[E1] VQA (capped 60/model, rate-limit tolerant) ..."
  $PY_SYS "$HERE/score_vqa.py" --gen-root "$OUT/gen" --out-dir "$OUT/scores" \
    > "$OUT/logs/score_vqa.log" 2>&1 || echo "[E1] VQA failed (non-fatal); see log"
}

stage_stats() {
  write_progress "stats" "paired bootstrap stats running ..."
  $PY_CLIP "$HERE/stats_paired.py" --scores-dir "$OUT/scores" \
    > "$OUT/logs/stats.log" 2>&1 || die "stats failed; see $OUT/logs/stats.log"
  echo "[E1] stats done"
}

stage_sheet() {
  write_progress "sheet" "contact sheets ..."
  $PY_CLIP "$HERE/contact_sheet.py" --gen-root "$OUT/gen" --out-dir "$OUT/contact_sheets" \
    > "$OUT/logs/sheet.log" 2>&1 || echo "[E1] contact sheet failed (non-fatal); see log"
  write_progress "DONE" \
    "results: $OUT/scores/comparison_table.md , $OUT/scores/stats_summary.json" \
    "sheets : $OUT/contact_sheets/"
  echo "[E1] ALL DONE — see $OUT/scores/comparison_table.md"
}

preflight
case "$STAGE" in
  smoke) stage_smoke ;;
  gen) stage_gen ;;
  score) stage_score ;;
  vqa) stage_vqa ;;
  stats) stage_stats ;;
  sheet) stage_sheet ;;
  all)
    stage_gen || die "generation failed"
    stage_score
    stage_vqa
    stage_stats
    stage_sheet
    ;;
  *) die "unknown stage: $STAGE" ;;
esac
