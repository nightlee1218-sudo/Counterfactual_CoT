#!/usr/bin/env bash
# data_v2 pilot entrypoint: plans -> dual-engine composite -> integrity ->
# verify -> audit gate -> merged pilot_report.json.
#
# Pipeline per engine (engines: nanobanana + addit by default, each writing
# its own target set under eng_<name>/ for cross-engine comparison):
#   stage 1  gen_intervention  (idempotent: skip when plans.jsonl has N rows)
#   stage 2  composite_target  (resumable: ledger ok rows + assets are kept)
#   stage 3  integrity_guard   (blend-overflow == 0 + foreground erosion IoU)
#   stage 4  verify_checklist  (M3/T2I-Copilot VLM checklist, regen flagging)
#   stage 5  audit_v2          (bg_hist_l1 <= 0.05, changed_out_frac == 0)
#   stage 6  pilot_report.py   (merge per-engine stats -> pilot_report.json)
#
# Safety / red lines:
#   * default engines are MOCK (no network, no weight downloads)
#   * real nanobanana needs DATA_V2_PROVIDER=nanobanana AND NANOBANANA_KEY
#   * real Add-it needs ADDIT_WEIGHTS_DIR (+ADDIT_CODE_DIR); missing -> mock
#   * real BLIP-2 verify needs DATA_V2_VLM=blip2 (local HF cache only)
#
# Usage:
#   bash run_pilot.sh [--n 200] [--seed 1234] [--engines "nanobanana addit"]
#                     [--matting mock|auto|rembg|sam2|grabcut]
#                     [--threshold 0.8]
# Env overrides: SOURCE_MANIFEST IMAGE_ROOT PILOT_DIR LOG_DIR
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python3

# ---- defaults -------------------------------------------------------------
N="${PILOT_N:-200}"
SEED="${PILOT_SEED:-1234}"
ENGINES="${DATA_V2_ENGINES:-nanobanana addit}"
MATTING="${DATA_V2_MATTING:-mock}"
VERIFY_THRESHOLD="${DATA_V2_VERIFY_THRESHOLD:-0.8}"
VLM="${DATA_V2_VLM:-mock}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$HERE/../../gold1000_framework_strict_v4/v4_manifests/source_registry.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-$HERE/../../images_gold}"
PILOT_DIR="${PILOT_DIR:-$HERE/pilot_200}"
LOG_DIR="${LOG_DIR:-$PILOT_DIR/logs}"

# ---- arg passthrough ------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --n) N="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --engines) ENGINES="$2"; shift 2 ;;
    --matting) MATTING="$2"; shift 2 ;;
    --threshold) VERIFY_THRESHOLD="$2"; shift 2 ;;
    *) echo "[run_pilot] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- guards ---------------------------------------------------------------
ENGINE_LIST=($ENGINES)
for ENG in "${ENGINE_LIST[@]}"; do
  case "$ENG" in
    nanobanana|addit|mock) ;;
    *) echo "[run_pilot] FATAL: unknown engine '$ENG' (nanobanana|addit)" >&2; exit 1 ;;
  esac
done

# per-engine provider resolution: real only when explicitly requested AND env ready
provider_for() {
  case "$1" in
    nanobanana)
      if [ "${DATA_V2_PROVIDER:-}" = "nanobanana" ]; then
        if [ -z "${NANOBANANA_KEY:-}" ]; then
          echo "[run_pilot] FATAL: DATA_V2_PROVIDER=nanobanana but NANOBANANA_KEY is empty" >&2
          exit 1
        fi
        echo "nanobanana"
      else
        echo "mock"   # nanobanana mock engine
      fi ;;
    addit) echo "addit" ;;  # engine_addit falls back to mock when weights missing
    *) echo "$1" ;;
  esac
}

[ -f "$SOURCE_MANIFEST" ] || { echo "[run_pilot] FATAL: source manifest missing: $SOURCE_MANIFEST" >&2; exit 1; }
[ -d "$IMAGE_ROOT" ] || { echo "[run_pilot] FATAL: image root missing: $IMAGE_ROOT" >&2; exit 1; }

mkdir -p "$PILOT_DIR" "$LOG_DIR"
STAMP="$(date '+%Y%m%d_%H%M%S')"
PROGRESS="$PILOT_DIR/pilot_progress.txt"
PLANS="$PILOT_DIR/plans.jsonl"
LOG="$LOG_DIR/pilot_${STAMP}.log"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG" >> "$PROGRESS"; }

# ---- stage 1: intervention plans (idempotent) ------------------------------
PLAN_ROWS=0
[ -f "$PLANS" ] && PLAN_ROWS=$(wc -l < "$PLANS" | tr -d ' ')
if [ "$PLAN_ROWS" -ge "$N" ]; then
  say "stage1 plans: complete ($PLAN_ROWS rows), skip (idempotent)"
else
  say "stage1 plans: generating $N plans (had $PLAN_ROWS rows)"
  if ! $PY "$HERE/gen_intervention.py" \
      --source-manifest "$SOURCE_MANIFEST" --out "$PLANS" \
      --n-plans "$N" --seed "$SEED" >> "$LOG" 2>&1; then
    say "stage1 plans: FAILED (see $LOG)"; exit 1
  fi
  say "stage1 plans: done ($(wc -l < "$PLANS" | tr -d ' ') rows)"
fi

# ---- stages 2-5 per engine --------------------------------------------------
FAIL=0
for ENG in "${ENGINE_LIST[@]}"; do
  ENG_DIR="$PILOT_DIR/eng_$ENG"
  PROVIDER="$(provider_for "$ENG")"
  say "engine=$ENG: provider=$PROVIDER matting=$MATTING out=$ENG_DIR"

  # stage 2: composite (resumable)
  if $PY "$HERE/composite_target.py" \
      --plans "$PLANS" --image-root "$IMAGE_ROOT" --out-dir "$ENG_DIR" \
      --provider "$PROVIDER" --matting "$MATTING" --seed "$SEED" \
      >> "$LOG" 2>&1; then
    say "engine=$ENG stage2 composite: ok"
  else
    say "engine=$ENG stage2 composite: some pairs FAILED (ledger has details)"
    FAIL=1
  fi

  # stage 3: integrity guard
  if $PY "$HERE/integrity_guard.py" \
      --manifest "$ENG_DIR/manifest.jsonl" --image-root "$IMAGE_ROOT" \
      --out "$ENG_DIR/integrity.json" --matting "$MATTING" \
      >> "$LOG" 2>&1; then
    say "engine=$ENG stage3 integrity: ok"
  else
    say "engine=$ENG stage3 integrity: NEEDS_FIX flagged (see $ENG_DIR/integrity.json)"
    FAIL=1
  fi

  # stage 4: semantic verify (checklist VLM)
  if $PY "$HERE/verify_checklist.py" \
      --manifest "$ENG_DIR/manifest.jsonl" --image-root "$IMAGE_ROOT" \
      --out "$ENG_DIR/verify.json" --threshold "$VERIFY_THRESHOLD" --vlm "$VLM" \
      >> "$LOG" 2>&1; then
    say "engine=$ENG stage4 verify: all pairs pass threshold"
  else
    say "engine=$ENG stage4 verify: regen flagged (see $ENG_DIR/verify.json)"
    FAIL=1
  fi

  # stage 5: audit gate
  if $PY "$HERE/audit_v2.py" \
      --manifest "$ENG_DIR/manifest.jsonl" --image-root "$IMAGE_ROOT" \
      --out "$ENG_DIR/audit_report.json" --sample-iou 20 --seed "$SEED" \
      >> "$LOG" 2>&1; then
    say "engine=$ENG stage5 audit: PASS"
  else
    say "engine=$ENG stage5 audit: FAILED (see $ENG_DIR/audit_report.json)"
    FAIL=1
  fi
done

# ---- stage 6: merged pilot report ------------------------------------------
$PY "$HERE/pilot_report.py" --pilot-dir "$PILOT_DIR" --engines "${ENGINE_LIST[@]}" \
  >> "$LOG" 2>&1 || say "stage6 pilot_report: FAILED"
say "stage6 pilot_report: $PILOT_DIR/pilot_report.json"

if [ "$FAIL" = "0" ]; then
  say "PILOT DONE (all gates passed for ${#ENGINE_LIST[@]} engine(s))"
  exit 0
else
  say "PILOT DONE WITH GATE FAILURES (see pilot_report.json + per-engine reports)"
  exit 1
fi
