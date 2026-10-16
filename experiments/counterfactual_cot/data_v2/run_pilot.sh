#!/usr/bin/env bash
# data_v2 200-pair pilot entrypoint: plan -> composite -> audit.
#
# Idempotent + resumable:
#   * gen_intervention  : skipped when the plans file already has N rows
#   * composite_target  : skips pairs with a prior "ok" ledger row + assets
#   * audit_v2          : cheap, always re-run
#
# Safety: the default provider is MOCK (no network, no real API calls). Real
# nanobanana calls require BOTH  DATA_V2_PROVIDER=nanobanana  AND a non-empty
# NANOBANANA_KEY in the environment — the pilot pipeline must be validated
# end-to-end with the mock provider first.
#
# Usage:
#   bash run_pilot.sh [--n 200] [--seed 1234] [--provider mock|nanobanana]
#                     [--matting mock|auto|rembg|sam2|grabcut]
# Env overrides: SOURCE_MANIFEST  IMAGE_ROOT  PILOT_DIR  LOG_DIR
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python3

# ---- defaults -------------------------------------------------------------
N="${PILOT_N:-200}"
SEED="${PILOT_SEED:-1234}"
PROVIDER="${DATA_V2_PROVIDER:-mock}"
MATTING="${DATA_V2_MATTING:-mock}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-$HERE/../../gold1000_framework_strict_v4/v4_manifests/source_registry.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-$HERE/../../images_gold}"
PILOT_DIR="${PILOT_DIR:-$HERE/pilot_200}"
LOG_DIR="${LOG_DIR:-$PILOT_DIR/logs}"

# ---- arg passthrough ------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    --n) N="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --provider) PROVIDER="$2"; shift 2 ;;
    --matting) MATTING="$2"; shift 2 ;;
    *) echo "[run_pilot] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- guards ---------------------------------------------------------------
if [ "$PROVIDER" = "nanobanana" ]; then
  if [ -z "${NANOBANANA_KEY:-}" ]; then
    echo "[run_pilot] FATAL: PROVIDER=nanobanana but NANOBANANA_KEY is empty" >&2
    exit 1
  fi
  echo "[run_pilot] REAL API MODE (nanobanana); key present, proceeding"
else
  echo "[run_pilot] mock mode (no real API calls)"
fi
[ -f "$SOURCE_MANIFEST" ] || { echo "[run_pilot] FATAL: source manifest missing: $SOURCE_MANIFEST" >&2; exit 1; }
[ -d "$IMAGE_ROOT" ] || { echo "[run_pilot] FATAL: image root missing: $IMAGE_ROOT" >&2; exit 1; }

mkdir -p "$PILOT_DIR" "$LOG_DIR"
STAMP="$(date '+%Y%m%d_%H%M%S')"
PROGRESS="$PILOT_DIR/pilot_progress.txt"
PLANS="$PILOT_DIR/plans.jsonl"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG_DIR/pilot_${STAMP}.log" >> "$PROGRESS"; }

# ---- stage 1: intervention plans (idempotent) ------------------------------
PLAN_ROWS=0
[ -f "$PLANS" ] && PLAN_ROWS=$(wc -l < "$PLANS" | tr -d ' ')
if [ "$PLAN_ROWS" -ge "$N" ]; then
  say "stage1 plans: complete ($PLAN_ROWS rows), skip (idempotent)"
else
  say "stage1 plans: generating $N plans (had $PLAN_ROWS rows)"
  if ! $PY "$HERE/gen_intervention.py" \
      --source-manifest "$SOURCE_MANIFEST" --out "$PLANS" \
      --n-plans "$N" --seed "$SEED" >> "$LOG_DIR/pilot_${STAMP}.log" 2>&1; then
    say "stage1 plans: FAILED (see $LOG_DIR/pilot_${STAMP}.log)"; exit 1
  fi
  say "stage1 plans: done ($(wc -l < "$PLANS" | tr -d ' ') rows)"
fi

# ---- stage 2: compositing (resumable) --------------------------------------
say "stage2 composite: provider=$PROVIDER matting=$MATTING"
if ! $PY "$HERE/composite_target.py" \
    --plans "$PLANS" --image-root "$IMAGE_ROOT" --out-dir "$PILOT_DIR" \
    --provider "$PROVIDER" --matting "$MATTING" --seed "$SEED" \
    >> "$LOG_DIR/pilot_${STAMP}.log" 2>&1; then
  say "stage2 composite: some pairs FAILED (see ledger.jsonl + log) — continuing to audit"
fi
say "stage2 composite: ledger at $PILOT_DIR/ledger.jsonl"

# ---- stage 3: audit gate ----------------------------------------------------
say "stage3 audit: running gates (bg_hist_l1<=0.05, changed_out_frac==0, IoU spot-check 20)"
if $PY "$HERE/audit_v2.py" \
    --manifest "$PILOT_DIR/manifest.jsonl" --image-root "$IMAGE_ROOT" \
    --out "$PILOT_DIR/audit_report.json" --sample-iou 20 --seed "$SEED" \
    >> "$LOG_DIR/pilot_${STAMP}.log" 2>&1; then
  say "stage3 audit: PASS (report: $PILOT_DIR/audit_report.json)"
  say "PILOT DONE (all gates passed)"
  exit 0
else
  say "stage3 audit: FAILED (report: $PILOT_DIR/audit_report.json)"
  say "PILOT DONE WITH GATE FAILURES"
  exit 1
fi
