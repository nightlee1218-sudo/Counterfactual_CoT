#!/usr/bin/env bash
# E3 four-way evaluation launcher: base / final991 / v6 / v7 (align-loss models).
#
# Usage on the 2x A100 GPU machine (mounts /group):
#   bash /group/40016/niightli/BAGEL/experiments/counterfactual_cot/e3_eval/run_e3_eval.sh
# Stages:
#   selftest   # CPU: 100-check mock suite (no GPU needed)
#   preflight  # GPU machine: env + reusable-artifact audit, no generation
#   smoke      # tiny v6-only pipeline (E1 6 imgs / A-module 2 pairs / probe 2) + stats
#   run        # full four-way: v6+v7 generation+scoring on 2 GPUs in parallel,
#              #   idempotent (complete v6 artifacts from the three-way run are
#              #   skipped, never regenerated), then VQA (default on) + stats + summary
#   status     # progress file + artifact inventory + last log lines
# Granular foreground stages (single GPU; E1_MODELS/INDO_MODELS select the
# model set, default "v6 v7"):
#   e1-gen | e1-score | vqa | indomain-gen | indomain-score | probe | stats | summary
#
# base/final991 rows always come from the existing E1 / final991_eval_v1 / E2
# artifacts (copied into OUT, never rescored). v7 rows are generated through
# the same shims/generators as v6 (gen_same_seed_states_v6.py format guard
# accepts the v7 label; model_shim.py is alias-agnostic).
# Override knobs: V6_CKPT V7_CKPT MODEL_DIR EVAL_PY N_BOOT GPU0 GPU1
#                 E1_MAX_ITEMS E1_MODELS INDO_MODELS INDO_SEEDS
#                 PROBE_LIMIT PROBE_STEPS PROBE_VERIFY_NPZ
#                 VQA_MODELS E1_VQA (vqa stage: local BLIP-2, see vqa_scorer.py;
#                 E1_VQA defaults to 1 — set E1_VQA=0 to skip)
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
# v7: resolved on demand (resolve_v7_ckpt) — explicit override wins, else
# run_v7.sh's default run name, else the newest *v7* run dir under outputs/.
V7_CKPT="${V7_CKPT:-}"
V7_RUN_DIR_DEFAULT="gold40_automask_v7_align_fsdp_full_shard"
SPLITS="$CFCOT/final991_automask_v5_data/artifacts/splits"
E1_DIR="$CFCOT/e1_general_t2i"
E1_PROMPTS="$E1_DIR/prompts"
V5GEN="$CFCOT/gold40_automask_dual_target_cot1024_fsdp_v5/generate_same_seed_states.py"
# v6/v7 A-module generation goes through the e3_eval-side copy of the v5
# generator: byte-identical except the checkpoint-format guard also accepts
# the v6/v7 labels 'gold40_automask_v6_align_fsdp' /
# 'gold40_automask_v7_align_fsdp' (the v5 original must stay untouched as
# the final991 reference generator).
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

count_probe_ok() { # probe_manifest -> ok-pair count (0 when absent)
  [ -f "$1" ] && grep -c '"status": "ok"' "$1" || echo 0
}

# ---------------------------------------------------------------------------
# v7 checkpoint resolution (side effect: sets V7_CKPT; empty when unresolved)
# ---------------------------------------------------------------------------
resolve_v7_ckpt() {
  if [ -n "$V7_CKPT" ]; then return 0; fi
  local cand="$CFCOT/outputs/$V7_RUN_DIR_DEFAULT/checkpoints/latest.pt"
  if [ -f "$cand" ]; then V7_CKPT="$cand"; return 0; fi
  cand="$(ls -1dt "$CFCOT"/outputs/*v7*/checkpoints/latest.pt 2>/dev/null | head -1 || true)"
  if [ -n "$cand" ]; then V7_CKPT="$cand"; fi
}

# model -> global MODEL_CKPT (dies when the checkpoint is missing)
model_ckpt() {
  if [ "$1" = "v7" ]; then
    resolve_v7_ckpt
    MODEL_CKPT="${V7_CKPT:-}"
  else
    MODEL_CKPT="$V6_CKPT"
  fi
  [ -n "$MODEL_CKPT" ] && [ -f "$MODEL_CKPT" ] \
    || die "$1 checkpoint missing: ${MODEL_CKPT:-<unresolved>} (set V6_CKPT / V7_CKPT)"
}

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
# generation / scoring steps (model-generic: v6 | v7, idempotent skips)
# ---------------------------------------------------------------------------
e1_expected_rows() { # max_items -> 3 x min(150, max_items) (0 = all -> 450)
  local per=150
  if [ "$1" != "0" ] && [ "$1" -lt 150 ]; then per=$1; fi
  echo $((3 * per))
}

e1_gen() { # out_root model [max_items]
  local o="$1" m="$2" extra=() expected
  [ "${3:-0}" != "0" ] && extra+=(--max-items "$3")
  expected="$(e1_expected_rows "${3:-0}")"
  if [ "$(count_lines "$o/gen_t2i/$m/generation_manifest.jsonl")" -ge "$expected" ]; then
    echo "[e1-gen] $m complete ($expected rows), skip (idempotent)"; return 0
  fi
  model_ckpt "$m"
  CUDA_VISIBLE_DEVICES=$GPU0 $PY "$HERE/model_shim.py" \
    --script "$E1_DIR/generate_t2i.py" --alias "$m" --ckpt "$MODEL_CKPT" -- \
    --model "$m" --prompts-dir "$E1_PROMPTS" --output-dir "$o/gen_t2i/$m" \
    --llm-path "$MODEL_DIR" --seed 1234 "${extra[@]}"
}

e1_score() { # out_root model [max_items]
  local o="$1" m="$2" expected
  expected="$(e1_expected_rows "${3:-0}")"
  if [ "$(count_lines "$o/scores_t2i/clipT_$m.jsonl")" -ge "$expected" ]; then
    echo "[e1-score] $m clipT complete ($expected rows), skip (idempotent)"; return 0
  fi
  CUDA_VISIBLE_DEVICES=$GPU0 $EVAL_PY "$E1_DIR/score_clip.py" \
    --gen-root "$o/gen_t2i" --out-dir "$o/scores_t2i" --models "$m"
}

e1_v6_gen()   { e1_gen "$1" v6 "${2:-0}"; }
e1_v7_gen()   { e1_gen "$1" v7 "${2:-0}"; }
e1_v6_score() { e1_score "$1" v6; }
e1_v7_score() { e1_score "$1" v7; }

# Local BLIP-2 VQA scoring (pure-local replacement of the 403'd API scorer).
# Auto-detects every model with an E1 gen manifest under $o/gen_t2i plus the
# reference base/final991 generations; VQA_MODELS narrows it when set.
e1_vqa_score() { # out_root
  local o="$1" models="${VQA_MODELS:-}"
  local vm=()
  [ -n "$models" ] && vm+=(--models $models)
  CUDA_VISIBLE_DEVICES=$GPU0 $EVAL_PY "$HERE/vqa_scorer.py" \
    --gen-e3-root "$o/gen_t2i" --out-dir "$o/scores_t2i" "${vm[@]}"
}

indomain_gen() { # out_root model splits_dir splits...
  local o="$1" m="$2" splits_dir="$3"; shift 3
  local split pairs expected manifest
  for split in "$@"; do
    manifest="$o/gen_indomain/$m/$split/generation_manifest.jsonl"
    pairs=$(count_lines "$splits_dir/$split.jsonl")
    expected=$((pairs * 2 * INDO_SEEDS_N))
    if [ "$(count_lines "$manifest")" -ge "$expected" ] && [ "$expected" -gt 0 ]; then
      echo "[indomain-gen] $m/$split complete ($expected rows), skip (idempotent)"; continue
    fi
    local ow=()
    [ -e "$o/gen_indomain/$m/$split" ] && [ -n "$(ls -A "$o/gen_indomain/$m/$split" 2>/dev/null)" ] \
      && ow+=(--overwrite)
    model_ckpt "$m"
    echo "[indomain-gen] $m/$split: $pairs pairs x 2 states x $INDO_SEEDS_N seeds"
    CUDA_VISIBLE_DEVICES=$GPU1 $PY "$V6GEN" \
      --manifest "$splits_dir/$split.jsonl" \
      --output-dir "$o/gen_indomain/$m/$split" \
      --llm-path "$MODEL_DIR" --vae-path "$MODEL_DIR/ae.safetensors" \
      --base-checkpoint-path "$MODEL_DIR/ema.safetensors" \
      --trainable-checkpoint "$MODEL_CKPT" \
      --seeds $INDO_SEEDS \
      --num-timesteps 25 --cfg-text-scale 4.0 --cfg-img-scale 2.0 --timestep-shift 3.0 \
      "${ow[@]}"
  done
}

indomain_score() { # out_root model splits_dir splits...
  local o="$1" m="$2" splits_dir="$3"; shift 3
  CUDA_VISIBLE_DEVICES=$GPU1 $EVAL_PY "$HERE/score_indomain_threeway.py" \
    --models "$m" --splits "$@" --splits-dir "$splits_dir" \
    --gen-v6-dir "$o/gen_indomain/v6" --gen-v7-dir "$o/gen_indomain/v7" \
    --out-dir "$o/scores_indomain" --seeds $INDO_SEEDS
}

indomain_v6_gen()   { indomain_gen "$1" v6 "$2" "${@:3}"; }
indomain_v7_gen()   { indomain_gen "$1" v7 "$2" "${@:3}"; }
indomain_v6_score() { indomain_score "$1" v6 "$2" "${@:3}"; }
indomain_v7_score() { indomain_score "$1" v7 "$2" "${@:3}"; }

probe_run() { # out_root model limit probe_steps...
  local o="$1" m="$2" limit="$3"; shift 3
  local extra=() expected manifest
  expected=$(count_lines "$SPLITS/val.jsonl")
  [ "$limit" != "0" ] && expected=$limit
  manifest="$o/probe_$m/probe_manifest.jsonl"
  if [ "$expected" -gt 0 ] && [ "$(count_probe_ok "$manifest")" -ge "$expected" ]; then
    echo "[probe] $m complete ($expected ok pairs), skip (idempotent)"; return 0
  fi
  [ "$limit" != "0" ] && extra+=(--limit "$limit")
  model_ckpt "$m"
  CUDA_VISIBLE_DEVICES=$GPU1 $PY "$HERE/model_shim.py" \
    --script "$E2_PROBE" --alias "$m" --ckpt "$MODEL_CKPT" -- \
    --model "$m" --out-dir "$o/probe_$m" --probe-steps "$@" "${extra[@]}"
}

probe_v6() { probe_run "$1" v6 "$2" "${@:3}"; }
probe_v7() { probe_run "$1" v7 "$2" "${@:3}"; }

stats_all() { # out_root probe_min_n
  local o="$1" min_n="${2:-5}"
  $PY "$HERE/stats_threeway.py" --scores-dir "$o/scores_t2i" --bootstrap "$N_BOOT" \
    || die "E1 stats failed"
  $PY "$HERE/indomain_stats.py" --scores-dir "$o/scores_indomain" \
    --splits val test --bootstrap "$N_BOOT" \
    || die "Module-A stats failed"
  local pv=()
  [ "$PROBE_VERIFY_NPZ" = "1" ] && pv+=(--verify-npz)
  $PY "$HERE/probe_stats.py" --v6-dir "$o/probe_v6" --v7-dir "$o/probe_v7" \
    --out-dir "$o/probe_stats" \
    --bootstrap "$N_BOOT" --min-n "$min_n" "${pv[@]}" \
    || die "E2 probe stats failed"
  $PY "$HERE/summarize.py" --e3-root "$o" || die "summary failed"
}

# ---------------------------------------------------------------------------
# GPU chains (background) + watcher
# ---------------------------------------------------------------------------
INDO_SEEDS="${INDO_SEEDS:-1234 2234 3234}"
INDO_SEEDS_N=$(echo $INDO_SEEDS | wc -w | tr -d ' ')

chain_gpu0() { # out_root e1_max_items (models from E1_MODELS, default "v6 v7")
  local o="$1" max="${2:-0}" m
  for m in ${E1_MODELS:-v6 v7}; do
    e1_gen "$o" "$m" "$max" || return 1
  done
  copy_reference_scores "$o" || return 1
  for m in ${E1_MODELS:-v6 v7}; do
    e1_score "$o" "$m" "$max" || return 1
  done
  # local BLIP-2 VQA pass on the same GPU, default ON (four-way: every model
  # with a gen manifest — base/final991/v6/v7; append-resumable; stats merges
  # the vqa rows automatically once >= 2 models have them)
  if [ "${E1_VQA:-1}" = "1" ]; then e1_vqa_score "$o" || return 1; fi
  return 0
}

chain_gpu1() { # out_root splits_dir splits... (probe args + models via globals)
  local o="$1" splits_dir="$2"; shift 2
  local m
  for m in ${INDO_MODELS:-v6 v7}; do
    indomain_gen "$o" "$m" "$splits_dir" "$@" || return 1
  done
  for m in ${INDO_MODELS:-v6 v7}; do
    probe_run "$o" "$m" "${PROBE_LIMIT:-0}" $PROBE_STEPS || return 1
  done
  copy_reference_scores "$o" || return 1
  for m in ${INDO_MODELS:-v6 v7}; do
    indomain_score "$o" "$m" "$splits_dir" "$@" || return 1
  done
}
PROBE_STEPS="${PROBE_STEPS:-0 11 23}"

last_event() { grep -E '"event":' "$1" 2>/dev/null | tail -1 | cut -c1-160; }

watch_chains() { # out_root pid0 pid1
  local o="$1" p0="$2" p1="$3" i=0
  while kill -0 "$p0" 2>/dev/null || kill -0 "$p1" 2>/dev/null; do
    write_progress "${STAGE}" \
      "gpu0 (E1 gen+score [${E1_MODELS:-v6 v7}] -> VQA): $(last_event "$o/logs/gpu0.log")" \
      "gpu1 (A gen -> probe -> score [${INDO_MODELS:-v6 v7}]): $(last_event "$o/logs/gpu1.log")"
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
  resolve_v7_ckpt
  if [ -n "$V7_CKPT" ] && [ -f "$V7_CKPT" ]; then
    echo "[E3-EVAL] v7 ckpt: $V7_CKPT"
  else
    V7_CKPT=""
    echo "[warn] no v7 checkpoint resolved (set V7_CKPT) — v7 stages will fail"
  fi
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
  [ "$(count_probe_ok "$PROBE_ROOT/probe_base/probe_manifest.jsonl")" = 51 ] \
    && [ "$(count_probe_ok "$PROBE_ROOT/probe_final991/probe_manifest.jsonl")" = 51 ] \
    || { echo "[FAIL] E2 probe manifests != 51 ok pairs"; fail=1; }
  grep -q '"status": "completed"' \
    "$CFCOT/outputs/gold40_automask_v6_align_fsdp_full_shard/final_report.json" 2>/dev/null \
    || echo "[warn] v6 final_report.json not 'completed' — checkpoint may be partial"
  resolve_v7_ckpt
  if [ -n "$V7_CKPT" ]; then
    grep -q '"status": "completed"' \
      "$(dirname "$(dirname "$V7_CKPT")")/final_report.json" 2>/dev/null \
      || echo "[warn] v7 final_report.json not 'completed' — checkpoint may be partial"
  fi
  local free_gb
  free_gb=$(df -Pk "$CFCOT" | awk 'NR==2 {print int($4/1048576)}')
  [ "${free_gb:-999}" -lt 50 ] && echo "[warn] only ${free_gb}GB free on $(df -P "$CFCOT" | awk 'NR==1{print $6}')"
  [ "$fail" = 0 ] || die "artifact preflight failed (reference rows incomplete)"
  echo "[E3-EVAL] preflight OK: v6/v7 ckpts + reusable base/final991 artifacts present"
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
  resolve_v7_ckpt
  if [ -n "$V7_CKPT" ]; then
    $PY "$HERE/model_shim.py" --script "$E2_PROBE" --alias v7 --ckpt "$V7_CKPT" --dry-run \
      -- --model v7 --limit 1 2>/dev/null | tail -1 \
      || die "shim dry-run (v7) against probe.py failed"
  fi
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
  ( E1_VQA=0 E1_MODELS=v6 chain_gpu0 "$o" 6 ) > "$o/logs/gpu0.log" 2>&1 & local p0=$!
  ( INDO_MODELS=v6 INDO_SEEDS="1234 2234" INDO_SEEDS_N=2 PROBE_LIMIT=2 \
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
  write_progress "run" "full four-way run starting (gpu0: E1 [${E1_MODELS:-v6 v7}] + VQA; gpu1: A+probe [${INDO_MODELS:-v6 v7}]); complete v6 artifacts are skipped"
  ( chain_gpu0 "$o" 0 ) > "$o/logs/gpu0.log" 2>&1 & local p0=$!
  ( chain_gpu1 "$o" "$SPLITS" val test ) > "$o/logs/gpu1.log" 2>&1 & local p1=$!
  echo $p0 > "$o/gpu0.pid"; echo $p1 > "$o/gpu1.pid"
  watch_chains "$o" "$p0" "$p1"
  wait "$p0" || { tail -20 "$o/logs/gpu0.log"; die "gpu0 chain failed"; }
  wait "$p1" || { tail -20 "$o/logs/gpu1.log"; die "gpu1 chain failed"; }
  local m f
  for f in scores_t2i/clipT_base.jsonl scores_t2i/clipT_final991.jsonl; do
    [ -s "$o/$f" ] || die "missing artifact after chains: $o/$f (see $o/logs/)"
  done
  for m in ${E1_MODELS:-v6 v7}; do
    [ -s "$o/scores_t2i/clipT_$m.jsonl" ] \
      || die "missing artifact after chains: $o/scores_t2i/clipT_$m.jsonl (see $o/logs/)"
  done
  for m in ${INDO_MODELS:-v6 v7}; do
    for f in "scores_indomain/indomain_${m}_val.jsonl" \
             "scores_indomain/indomain_${m}_test.jsonl" \
             "probe_$m/probe_manifest.jsonl"; do
      [ -s "$o/$f" ] || die "missing artifact after chains: $o/$f (see $o/logs/)"
    done
  done
  stats_all "$o" 5
  write_progress "DONE" "four-way eval complete: $o/summary_table.md"
  echo "[E3-EVAL] summary: $o/summary_table.md"
}

stage_status() {
  echo "== progress file =="; cat "$PROGRESS" 2>/dev/null || echo "(none)"
  local o out_root m
  for out_root in "$OUT" "$SMOKE_OUT"; do
    echo "== artifacts under $out_root =="
    for m in v6 v7; do
      for f in "gen_t2i/$m/generation_manifest.jsonl" "scores_t2i/clipT_$m.jsonl" \
               "gen_indomain/$m/val/generation_manifest.jsonl" \
               "gen_indomain/$m/test/generation_manifest.jsonl" \
               "scores_indomain/indomain_${m}_val.jsonl" \
               "scores_indomain/indomain_${m}_test.jsonl" \
               "probe_$m/probe_manifest.jsonl"; do
        if [ -f "$out_root/$f" ]; then
          echo "  [ok] $f ($(count_lines "$out_root/$f") lines)"
        else
          echo "  [..] $f"
        fi
      done
      if [ -f "$out_root/probe_$m/probe_manifest.jsonl" ]; then
        echo "  probe_$m ok pairs: $(count_probe_ok "$out_root/probe_$m/probe_manifest.jsonl")"
      fi
    done
    for f in probe_stats/probe_stats_threeway.json \
             scores_t2i/stats_threeway.json scores_indomain/indomain_stats_threeway.json \
             summary_table.md; do
      if [ -f "$out_root/$f" ]; then
        echo "  [ok] $f ($(count_lines "$out_root/$f") lines)"
      else
        echo "  [..] $f"
      fi
    done
    for vf in "$out_root"/scores_t2i/vqa_*.jsonl; do
      if [ -f "$vf" ]; then
        echo "  [ok] scores_t2i/$(basename "$vf") ($(count_lines "$vf") lines)"
      fi
    done
    for lg in gpu0 gpu1; do
      if [ -f "$out_root/logs/$lg.log" ]; then
        echo "== last event ($lg) =="; last_event "$out_root/logs/$lg.log"
      fi
    done
  done
}

# granular foreground stages -------------------------------------------------
stage_e1_gen() {
  gpu_preflight
  local m
  for m in ${E1_MODELS:-v6 v7}; do e1_gen "$OUT" "$m" "${E1_MAX_ITEMS:-0}"; done
}
stage_e1_score() {
  local m
  for m in ${E1_MODELS:-v6 v7}; do e1_score "$OUT" "$m" "${E1_MAX_ITEMS:-0}"; done
}
stage_vqa()      { e1_vqa_score "$OUT"; }
stage_indomain_gen() {
  gpu_preflight
  local m
  for m in ${INDO_MODELS:-v6 v7}; do indomain_gen "$OUT" "$m" "$SPLITS" val test; done
}
stage_indomain_score() {
  local m
  for m in ${INDO_MODELS:-v6 v7}; do indomain_score "$OUT" "$m" "$SPLITS" val test; done
}
stage_probe() {
  gpu_preflight
  local m
  for m in ${INDO_MODELS:-v6 v7}; do probe_run "$OUT" "$m" "${PROBE_LIMIT:-0}" $PROBE_STEPS; done
}
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
  vqa) stage_vqa ;;
  indomain-gen) stage_indomain_gen ;;
  indomain-score) stage_indomain_score ;;
  probe) stage_probe ;;
  stats) stage_stats ;;
  summary) stage_summary ;;
  *) die "unknown stage: $STAGE (selftest|preflight|smoke|run|status|e1-gen|e1-score|vqa|indomain-gen|indomain-score|probe|stats|summary)" ;;
esac
