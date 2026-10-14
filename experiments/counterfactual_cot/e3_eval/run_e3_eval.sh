#!/usr/bin/env bash
# E3 three-way evaluation launcher: base / final991 / v6 (align-loss model).
#
# Usage on the 2x A100 GPU machine (mounts /group):
#   bash /group/40016/niightli/BAGEL/experiments/counterfactual_cot/e3_eval/run_e3_eval.sh
# Stages:
#   selftest   # CPU: 46-check mock suite (no GPU needed)
#   preflight  # GPU machine: env + reusable-artifact audit, no generation
#   smoke      # tiny v6-only pipeline (E1 6 imgs / A-module 2 pairs / probe 2) + stats
#   run        # full three-way: v6 generation+scoring on 2 GPUs in parallel, stats + summary
#   status     # progress file + artifact inventory + last log lines
# Granular foreground stages (single GPU):
#   e1-gen | e1-score | indomain-gen | indomain-score | probe | stats | summary
#
# Only v6 is generated/scored/probed; base/final991 rows come from the
# existing E1 / final991_eval_v1 / E2 artifacts (copied into OUT, never rescored).
# Override knobs: V6_CKPT MODEL_DIR EVAL_PY N_BOOT GPU0 GPU1
#                 E1_MAX_ITEMS INDO_LIMIT PROBE_LIMIT PROBE_VERIFY_NPZ
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFCOT="$(cd "$HERE/.." && pwd)"
OUT="$CFCOT/outputs/e3_eval"
SMOKE_OUT="$CFCOT/outputs/e3_eval_smoke"
PROGRESS="$CFCOT/wb_e3_eval_progress.txt"
STAGE="${1:-run}"

PY=python3
EVAL_PY="${EVAL_PY:-/group/40016/niightli/BAGEL/venvs/geneval_eval/bin/python}"
MODEL_DIR="${MODEL_DIR:-/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT}"
V6_CKPT="${V6_CKPT:-$CFCOT/outputs/gold40_automask_v6_align_fsdp_full_shard/checkpoints/latest.pt}"
SPLITS="$CFCOT/final991_automask_v5_data/artifacts/splits"
E1_DIR="$CFCOT/e1_general_t2i"
E1_PROMPTS="$E1_DIR/prompts"
V5GEN="$CFCOT/gold40_automask_dual_target_cot1024_fsdp_v5/generate_same_seed_states.py"
# v6 A-module generation goes through the e3_eval-side copy of the v5
# generator: byte-identical except the checkpoint-format guard also accepts
# the v6 label 'gold40_automask_v6_align_fsdp' (the v5 original must stay
# untouched as the final991 reference generator).
V6GEN="$HERE/gen_same_seed_states_v6.py"
E2_PROBE="$CFCOT/e2_token_probe/probe.py"
E1_SCORES_SRC="$CFCOT/outputs/e1_general_t2i/scores"
IND_SCORES_SRC="$CFCOT/outputs/final991_eval_v1/scores"
PROBE_ROOT="$CFCOT/outputs/e2_token_probe"
N_BOOT="${N_BOOT:-20000}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
PROBE_VERIFY_NPZ="${PROBE_VERIFY_NPZ:-1}"

mkdir -p "$OUT/logs" 2>/dev/null || true

write_progress() {
  local tmp="$PROGRESS.tmp"
  { echo "updated: $(date '+%F %T') (stage=$1)"; shift
    for line in "$@"; do echo "$line"; done; } > "$tmp" && mv "$tmp" "$PROGRESS"
}

die() { write_progress "ERROR" "error: $*"; echo "[E3-EVAL FATAL] $*" >&2; exit 1; }

count_lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

# ---------------------------------------------------------------------------
# reusable artifact copy (idempotent)
# ---------------------------------------------------------------------------
copy_reference_scores() { # out_root
  local o="$1"
  mkdir -p "$o/scores_t2i" "$o/scores_indomain"
  for f in clipT_base.jsonl clipT_final991.jsonl; do
    [ -f "$E1_SCORES_SRC/$f" ] && cp -f "$E1_SCORES_SRC/$f" "$o/scores_t2i/$f"
  done
  for f in indomain_base_val.jsonl indomain_base_test.jsonl \
           indomain_final991_val.jsonl indomain_final991_test.jsonl; do
    [ -f "$IND_SCORES_SRC/$f" ] && cp -f "$IND_SCORES_SRC/$f" "$o/scores_indomain/$f"
  done
}

# ---------------------------------------------------------------------------
# generation / scoring steps
# ---------------------------------------------------------------------------
e1_v6_gen() { # out_root [max_items]
  local o="$1" extra=()
  [ "${2:-0}" != "0" ] && extra+=(--max-items "$2")
  CUDA_VISIBLE_DEVICES=$GPU0 $PY "$HERE/model_shim.py" \
    --script "$E1_DIR/generate_t2i.py" --alias v6 --ckpt "$V6_CKPT" -- \
    --model v6 --prompts-dir "$E1_PROMPTS" --output-dir "$o/gen_t2i/v6" \
    --llm-path "$MODEL_DIR" --seed 1234 "${extra[@]}"
}

e1_v6_score() { # out_root
  local o="$1"
  CUDA_VISIBLE_DEVICES=$GPU0 $EVAL_PY "$E1_DIR/score_clip.py" \
    --gen-root "$o/gen_t2i" --out-dir "$o/scores_t2i" --models v6
}

indomain_v6_gen() { # out_root splits_dir splits... (passed as array via globals)
  local o="$1" splits_dir="$2"; shift 2
  local split pairs expected manifest
  for split in "$@"; do
    manifest="$o/gen_indomain/v6/$split/generation_manifest.jsonl"
    pairs=$(count_lines "$splits_dir/$split.jsonl")
    expected=$((pairs * 2 * INDO_SEEDS_N))
    if [ "$(count_lines "$manifest")" -ge "$expected" ] && [ "$expected" -gt 0 ]; then
      echo "[indomain-gen] $split complete ($expected rows), skip"; continue
    fi
    local ow=()
    [ -e "$o/gen_indomain/v6/$split" ] && [ -n "$(ls -A "$o/gen_indomain/v6/$split" 2>/dev/null)" ] \
      && ow+=(--overwrite)
    echo "[indomain-gen] $split: $pairs pairs x 2 states x $INDO_SEEDS_N seeds"
    CUDA_VISIBLE_DEVICES=$GPU1 $PY "$V6GEN" \
      --manifest "$splits_dir/$split.jsonl" \
      --output-dir "$o/gen_indomain/v6/$split" \
      --llm-path "$MODEL_DIR" --vae-path "$MODEL_DIR/ae.safetensors" \
      --base-checkpoint-path "$MODEL_DIR/ema.safetensors" \
      --trainable-checkpoint "$V6_CKPT" \
      --seeds $INDO_SEEDS \
      --num-timesteps 25 --cfg-text-scale 4.0 --cfg-img-scale 2.0 --timestep-shift 3.0 \
      "${ow[@]}"
  done
}

indomain_v6_score() { # out_root splits_dir splits...
  local o="$1" splits_dir="$2"; shift 2
  CUDA_VISIBLE_DEVICES=$GPU1 $EVAL_PY "$HERE/score_indomain_threeway.py" \
    --models v6 --splits $* --splits-dir "$splits_dir" \
    --gen-v6-dir "$o/gen_indomain/v6" --out-dir "$o/scores_indomain" --seeds $INDO_SEEDS
}

probe_v6() { # out_root limit probe_steps...
  local o="$1" limit="$2"; shift 2
  local extra=()
  [ "$limit" != "0" ] && extra+=(--limit "$limit")
  CUDA_VISIBLE_DEVICES=$GPU1 $PY "$HERE/model_shim.py" \
    --script "$E2_PROBE" --alias v6 --ckpt "$V6_CKPT" -- \
    --model v6 --out-dir "$o/probe_v6" --probe-steps "$@" "${extra[@]}"
}

stats_all() { # out_root probe_min_n
  local o="$1" min_n="${2:-5}"
  $PY "$HERE/stats_threeway.py" --scores-dir "$o/scores_t2i" --bootstrap "$N_BOOT" \
    || die "E1 stats failed"
  $PY "$HERE/indomain_stats.py" --scores-dir "$o/scores_indomain" \
    --splits val test --bootstrap "$N_BOOT" \
    || die "Module-A stats failed"
  local pv=()
  [ "$PROBE_VERIFY_NPZ" = "1" ] && pv+=(--verify-npz)
  $PY "$HERE/probe_stats.py" --v6-dir "$o/probe_v6" --out-dir "$o/probe_stats" \
    --bootstrap "$N_BOOT" --min-n "$min_n" "${pv[@]}" \
    || die "E2 probe stats failed"
  $PY "$HERE/summarize.py" --e3-root "$o" || die "summary failed"
}

# ---------------------------------------------------------------------------
# GPU chains (background) + watcher
# ---------------------------------------------------------------------------
INDO_SEEDS="${INDO_SEEDS:-1234 2234 3234}"
INDO_SEEDS_N=$(echo $INDO_SEEDS | wc -w | tr -d ' ')

chain_gpu0() { # out_root e1_max_items
  e1_v6_gen "$1" "${2:-0}" && copy_reference_scores "$1" && e1_v6_score "$1"
}

chain_gpu1() { # out_root splits_dir splits... (probe args via globals)
  local o="$1" splits_dir="$2"; shift 2
  indomain_v6_gen "$o" "$splits_dir" "$@" \
    && probe_v6 "$o" "${PROBE_LIMIT:-0}" $PROBE_STEPS \
    && copy_reference_scores "$o" \
    && indomain_v6_score "$o" "$splits_dir" "$@"
}
PROBE_STEPS="${PROBE_STEPS:-0 11 23}"

last_event() { grep -E '"event":' "$1" 2>/dev/null | tail -1 | cut -c1-160; }

watch_chains() { # out_root pid0 pid1
  local o="$1" p0="$2" p1="$3" i=0
  while kill -0 "$p0" 2>/dev/null || kill -0 "$p1" 2>/dev/null; do
    write_progress "${STAGE}" \
      "gpu0 (E1 v6 gen+score): $(last_event "$o/logs/gpu0.log")" \
      "gpu1 (A gen -> probe -> score): $(last_event "$o/logs/gpu1.log")"
    sleep 60
  done
}

# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
gpu_preflight() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found (need the 2x A100 machine)"
  nvidia-smi --L | head -2
  [ -f "$MODEL_DIR/ema.safetensors" ] && [ -f "$MODEL_DIR/ae.safetensors" ] \
    || die "BAGEL weights missing: $MODEL_DIR"
  [ -f "$V6_CKPT" ] || die "v6 checkpoint missing: $V6_CKPT"
  [ -x "$EVAL_PY" ] || die "scoring python missing: $EVAL_PY"
  "$EVAL_PY" -c "import torch, open_clip" 2>/dev/null \
    || die "$EVAL_PY cannot import torch+open_clip"
}

artifact_preflight() {
  local fail=0
  [ "$(count_lines "$E1_PROMPTS/attribute_binding.jsonl")" = 150 ] \
    && [ "$(count_lines "$E1_PROMPTS/counting.jsonl")" = 150 ] \
    && [ "$(count_lines "$E1_PROMPTS/spatial_relations.jsonl")" = 150 ] \
    || { echo "[FAIL] E1 prompts != 3x150 under $E1_PROMPTS"; fail=1; }
  [ "$(count_lines "$SPLITS/val.jsonl")" = 51 ] && [ "$(count_lines "$SPLITS/test.jsonl")" = 44 ] \
    || { echo "[FAIL] splits != 51/44 under $SPLITS"; fail=1; }
  [ "$(count_lines "$E1_SCORES_SRC/clipT_base.jsonl")" = 450 ] \
    && [ "$(count_lines "$E1_SCORES_SRC/clipT_final991.jsonl")" = 450 ] \
    || { echo "[FAIL] E1 clipT rows != 450"; fail=1; }
  [ -f "$IND_SCORES_SRC/indomain_base_val.jsonl" ] \
    && [ -f "$IND_SCORES_SRC/indomain_final991_test.jsonl" ] \
    || { echo "[FAIL] Module-A reference rows missing under $IND_SCORES_SRC"; fail=1; }
  [ "$(grep -c '"status": "ok"' "$PROBE_ROOT/probe_base/probe_manifest.jsonl" 2>/dev/null)" = 51 ] \
    && [ "$(grep -c '"status": "ok"' "$PROBE_ROOT/probe_final991/probe_manifest.jsonl" 2>/dev/null)" = 51 ] \
    || { echo "[FAIL] E2 probe manifests != 51 ok pairs"; fail=1; }
  grep -q '"status": "completed"' \
    "$CFCOT/outputs/gold40_automask_v6_align_fsdp_full_shard/final_report.json" 2>/dev/null \
    || echo "[warn] v6 final_report.json not 'completed' — checkpoint may be partial"
  local free_gb
  free_gb=$(df -Pk "$CFCOT" | awk 'NR==2 {print int($4/1048576)}')
  [ "${free_gb:-999}" -lt 50 ] && echo "[warn] only ${free_gb}GB free on $(df -P "$CFCOT" | awk 'NR==1{print $6}')"
  [ "$fail" = 0 ] || die "artifact preflight failed (reference rows incomplete)"
  echo "[E3-EVAL] preflight OK: v6 ckpt + reusable base/final991 artifacts present"
}

stage_selftest() {
  write_progress "selftest" "CPU mock suite running ..."
  $PY "$HERE/selftest_e3_eval.py" 2>&1 | tee "$HERE/selftest_last.log" | grep -E "^(PASS|FAIL|SKIP|ALL)" \
    || die "selftest failed; see $HERE/selftest_last.log"
  write_progress "selftest_done" "all CPU mock tests passed"
}

stage_preflight() {
  gpu_preflight
  artifact_preflight
  $PY "$HERE/model_shim.py" --script "$E2_PROBE" --alias v6 --ckpt "$V6_CKPT" --dry-run \
    -- --model v6 --limit 1 2>/dev/null | tail -1 \
    || die "shim dry-run against probe.py failed"
  write_progress "preflight_done" "env + artifacts + shim OK"
}

stage_smoke() {
  gpu_preflight
  artifact_preflight
  local o="$SMOKE_OUT"
  rm -rf "$o"; mkdir -p "$o/logs" "$o/splits" "$o/manifests"
  head -2 "$SPLITS/val.jsonl" > "$o/splits/val.jsonl"
  copy_reference_scores "$o"
  write_progress "smoke" "v6-only tiny pipeline starting (E1 6 / A 2x2x2 / probe 2)"
  ( chain_gpu0 "$o" 6 ) > "$o/logs/gpu0.log" 2>&1 & local p0=$!
  ( INDO_SEEDS="1234 2234" INDO_SEEDS_N=2 PROBE_LIMIT=2 \
    chain_gpu1 "$o" "$o/splits" val ) > "$o/logs/gpu1.log" 2>&1 & local p1=$!
  echo $p0 > "$o/gpu0.pid"; echo $p1 > "$o/gpu1.pid"
  watch_chains "$o" "$p0" "$p1"
  wait "$p0" || { tail -20 "$o/logs/gpu0.log"; die "smoke gpu0 chain failed"; }
  wait "$p1" || { tail -20 "$o/logs/gpu1.log"; die "smoke gpu1 chain failed"; }
  stats_all "$o" 2 || die "smoke stats failed"
  write_progress "DONE" "smoke complete: $o/summary_table.md"
  echo "[E3-EVAL] smoke summary: $o/summary_table.md"
}

stage_run() {
  gpu_preflight
  artifact_preflight
  local o="$OUT"
  mkdir -p "$o/logs"
  copy_reference_scores "$o"
  write_progress "run" "full three-way run starting (gpu0: E1 v6; gpu1: A+probe v6)"
  ( chain_gpu0 "$o" 0 ) > "$o/logs/gpu0.log" 2>&1 & local p0=$!
  ( chain_gpu1 "$o" "$SPLITS" val test ) > "$o/logs/gpu1.log" 2>&1 & local p1=$!
  echo $p0 > "$o/gpu0.pid"; echo $p1 > "$o/gpu1.pid"
  watch_chains "$o" "$p0" "$p1"
  wait "$p0" || { tail -20 "$o/logs/gpu0.log"; die "gpu0 chain failed"; }
  wait "$p1" || { tail -20 "$o/logs/gpu1.log"; die "gpu1 chain failed"; }
  for f in scores_t2i/clipT_v6.jsonl scores_indomain/indomain_v6_val.jsonl \
           scores_indomain/indomain_v6_test.jsonl probe_v6/probe_manifest.jsonl; do
    [ -s "$o/$f" ] || die "missing artifact after chains: $o/$f (see $o/logs/)"
  done
  stats_all "$o" 5
  write_progress "DONE" "three-way eval complete: $o/summary_table.md"
  echo "[E3-EVAL] summary: $o/summary_table.md"
}

stage_status() {
  echo "== progress file =="; cat "$PROGRESS" 2>/dev/null || echo "(none)"
  local o out_root
  for out_root in "$OUT" "$SMOKE_OUT"; do
    echo "== artifacts under $out_root =="
    for f in gen_t2i/v6/generation_manifest.jsonl scores_t2i/clipT_v6.jsonl \
             gen_indomain/v6/val/generation_manifest.jsonl \
             gen_indomain/v6/test/generation_manifest.jsonl \
             scores_indomain/indomain_v6_val.jsonl scores_indomain/indomain_v6_test.jsonl \
             probe_v6/probe_manifest.jsonl probe_stats/probe_stats_threeway.json \
             scores_t2i/stats_threeway.json scores_indomain/indomain_stats_threeway.json \
             summary_table.md; do
      if [ -f "$out_root/$f" ]; then
        echo "  [ok] $f ($(count_lines "$out_root/$f") lines)"
      else
        echo "  [..] $f"
      fi
    done
    if [ -f "$out_root/probe_v6/probe_manifest.jsonl" ]; then
      echo "  probe_v6 ok pairs: $(grep -c '"status": "ok"' "$out_root/probe_v6/probe_manifest.jsonl")"
    fi
    for lg in gpu0 gpu1; do
      if [ -f "$out_root/logs/$lg.log" ]; then
        echo "== last event ($lg) =="; last_event "$out_root/logs/$lg.log"
      fi
    done
  done
}

# granular foreground stages -------------------------------------------------
stage_e1_gen()   { gpu_preflight; e1_v6_gen "$OUT" "${E1_MAX_ITEMS:-0}"; }
stage_e1_score() { e1_v6_score "$OUT"; }
stage_indomain_gen() {
  gpu_preflight
  indomain_v6_gen "$OUT" "$SPLITS" val test
}
stage_indomain_score() { indomain_v6_score "$OUT" "$SPLITS" val test; }
stage_probe()    { gpu_preflight; probe_v6 "$OUT" "${PROBE_LIMIT:-0}" $PROBE_STEPS; }
stage_stats()    { copy_reference_scores "$OUT"; stats_all "$OUT" 5; }
stage_summary()  { $PY "$HERE/summarize.py" --e3-root "$OUT"; }

case "$STAGE" in
  selftest) stage_selftest ;;
  preflight) stage_preflight ;;
  smoke) stage_smoke ;;
  run) stage_run ;;
  status) stage_status ;;
  e1-gen) stage_e1_gen ;;
  e1-score) stage_e1_score ;;
  indomain-gen) stage_indomain_gen ;;
  indomain-score) stage_indomain_score ;;
  probe) stage_probe ;;
  stats) stage_stats ;;
  summary) stage_summary ;;
  *) die "unknown stage: $STAGE (selftest|preflight|smoke|run|status|e1-gen|e1-score|indomain-gen|indomain-score|probe|stats|summary)" ;;
esac
