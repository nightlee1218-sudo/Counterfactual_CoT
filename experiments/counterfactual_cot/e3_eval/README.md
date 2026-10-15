# E3 evaluation: base / final991 / v6 (/ v7)

E3 v6 (= Module-A SFT + token-region alignment loss, 2973 steps,
`outputs/gold40_automask_v6_align_fsdp_full_shard/checkpoints/latest.pt`,
checkpoint keys identical to v5 → LoRA direct-load) evaluated against both
baselines on three axes, then unified into one summary table:

1. **E1 general T2I** — T2I-CompBench 3 subsets x 150 prompts, CLIP-T
   (ViT-L/14), per-item paired bootstrap + Wilcoxon.
2. **Module A in-domain** — the 95 counterfactual pairs (val 51 + test 44),
   3 seeds x 2 states, 13 metrics (pixel / CLIP-I / DINOv2 / LPIPS / mask
   decomposition / directionality).
3. **E2 token probe** — 51 val pairs, per-layer binding metrics, best-layer
   paired diffs + late-step AUROC.

Model pairs everywhere: `v6-final991` (alignment-loss marginal effect),
`v6-base` (total effect), `final991-base` (replication of the already-known
Module-A / E1 / E2 result, as a sanity anchor).

## Files

| file | role |
|---|---|
| `run_e3_eval.sh` | launcher: `selftest\|preflight\|smoke\|run\|status` + granular `e1-gen\|e1-score\|indomain-gen\|indomain-score\|probe\|stats\|summary` (run_v6.sh style, progress file `wb_e3_eval_progress.txt`) |
| `model_shim.py` | injects `MODELS["v6"] = {trainable_checkpoint: <ckpt>}` into e1/e2 scripts at import time (see below) |
| `gen_same_seed_states_v6.py` | e3_eval-side copy of the v5 A-module generator; byte-identical except the checkpoint-format guard also accepts the v6 label (see "Guard fix" below) |
| `indomain_stats.py` | shared PURE stats helpers (bootstrap/wilcoxon/pair_stats) + Module-A three-way CLI |
| `score_indomain_threeway.py` | Module-A scoring, metric code line-identical to `final991_eval_v1/score_indomain.py`; lazy heavy imports, `--models`, skip-if-exists |
| `stats_threeway.py` | E1 CLIP-T three-way CLI (subsets + ALL, optional VQA merge) |
| `vqa_scorer.py` | local BLIP-2 VQA scorer: T2I-CompBench-style questions per subset + negative controls; writes `vqa_{model}.jsonl`; `--mock` CPU selftest mode (see below) |
| `probe_stats.py` | E2 three-way CLI: 10 metrics x layers, best layer, late-step AUROC, optional npz cross-check |
| `summarize.py` | unified `summary_table.md` + `summary.json` from the three stats files |
| `selftest_e3_eval.py` | CPU mock suite (100 checks; no GPU/BAGEL stack) |

## What is reused vs. newly computed

| component | base / final991 | v6 |
|---|---|---|
| E1 generation | existing `outputs/e1_general_t2i/gen/{base,final991}` | new: `gen_t2i/v6` via shim -> `generate_t2i.py` |
| E1 CLIP-T rows | copied `clipT_{base,final991}.jsonl` | new: `score_clip.py --models v6` |
| A gen (95x2x3) | existing `outputs/final991_eval_v1/gen/...` | new: `gen_same_seed_states_v6.py --trainable-checkpoint <v6 latest.pt>` (e3_eval-side copy, guard relaxed; no shim needed) |
| A rows | copied `indomain_{base,final991}_{val,test}.jsonl` | new: `score_indomain_threeway.py --models v6` |
| E2 probe | existing `outputs/e2_token_probe/probe_{base,final991}` (51 ok pairs) | new: shim -> `probe.py`, same `--probe-steps 0 11 23` |
| stats | recomputed from the copied rows (replication pair) | — |

Only v6 is generated/scored/probed; the reference rows are never rescored,
so the replication pair stays comparable with the historical runs.

## The import-shim (why e1/e2 files are untouched)

`e1_general_t2i/generate_t2i.py` and `e2_token_probe/probe.py` live on other
branches (`feature/e1-general-t2i`, `exp/e2-token-probe`) and resolve
`--model` through a module-level `MODELS` dict (`choices=list(MODELS)`
inside parse). `model_shim.py` loads the script with
`importlib.util.spec_from_file_location`, registers
`MODELS["v6"] = {"trainable_checkpoint": <latest.pt>, "skip_trainable_reload": False}`,
rewrites `sys.argv` and calls `module.main()` — adding v6 without touching
any file on those branches:

```bash
python3 model_shim.py --script ../e2_token_probe/probe.py \
    --alias v6 --ckpt $V6_CKPT -- --model v6 --out-dir $OUT/probe_v6
```

`--dry-run` prints the injected entry and exits (used by preflight and the
selftest). The v5 A-module generator already takes
`--trainable-checkpoint` directly, so no shim is needed there.

## Output layout (`outputs/e3_eval/`, smoke: `outputs/e3_eval_smoke/`)

```
gen_t2i/v6/                       E1 v6 generation (450 imgs)
scores_t2i/clipT_{base,final991,v6}.jsonl + stats_threeway.json + comparison_table.md
gen_indomain/{v6,v7}/{val,test}/       A-module v6/v7 generation (570 imgs each)
scores_indomain/indomain_{model}_{split}.jsonl + indomain_stats_threeway.json + indomain_table.md
probe_{v6,v7}/                          E2 v6/v7 probe (manifest + maps/*.npz)
probe_stats/probe_stats_threeway.json + probe_table.md
summary_table.md + summary.json   unified N-way summary (models with rows)
```

## N-way models (v7 support, added 20261014)

Every stats CLI auto-detects the models that actually have rows
(`indomain_stats.py` / `stats_threeway.py`: `--models` default None;
`probe_stats.py`: `--v7-dir`, skipped when absent;
`score_indomain_threeway.py`: `--gen-v7-dir`).  With v7 rows present the
comparisons extend to six pairs — the v6 three-way order is preserved
byte-identically, then `v7-v6`, `v7-final991`, `v7-base` are appended
(`indomain_stats.pairs_for`; future models enumerate as later-minus-earlier).
v7 generation for the A-module uses the same `gen_same_seed_states_v6.py`
copy (its format guard accepts `gold40_automask_v7_align_fsdp` — same keys,
label only); E1/probe v7 runs go through `model_shim.py --alias v7 --ckpt
<v7 latest.pt>` exactly like v6.

## VQA scorer (local BLIP-2, added 20261015)

`vqa_scorer.py` is the pure-local replacement of the E1 API-based scorer
(`e1_general_t2i/score_vqa.py`, venus llmproxy -> HTTP 403 on the eval
instance). It answers T2I-CompBench-style questions per generated image with
`Salesforce/blip2-flan-t5-xl` (transformers 4.52.4; weight pull precedent:
ViT-g-14 / dinov2 / lpips already came from HF on the GPU instance;
`--local-vqa-dir` points at an offline copy if needed; /group has no VQA
weights — BAGEL-7B-MoT only — which is why the model is not pre-cached).

Templates per subset (paper convention) + sensitivity extensions (default ON,
`--no-negatives` off):

| subset | questions per image | negative controls |
|---|---|---|
| attribute_binding (color/shape/texture) | `Is there a {mod} {obj} in the image?` per binding | swapped-modifier checks between the first two bindings (attribute leakage) |
| counting | `How many {plural obj} are there in the image?`, answer parsed to int, exact match | none (paper convention) |
| spatial_relations | `Is there a {objA} {relation} a {objB} in the image?` | mirrored relation for right/left/top/bottom (yes to both directions fails) |

Prompts are parsed from the generation manifests (flat text; modifier
vocabularies extracted from the real 450 prompts; counting narration tails
like "for a picnic" suppressed, keep-prepositions ("with/near/alongside") and
listed verbs ("carried") keep post-verb counts; the selftest asserts
**450/450 parse coverage** on the real prompt files plus template/answer
spot checks). Output rows keep the e1 `score_vqa.py` contract
(`item_id/subset/source_category/model/vqa_match/...`, append-resumable),
so `stats_threeway.py` merges them automatically (binary match rate, >= 2
models required) and `summarize.py` surfaces the VQA block.

```bash
bash run_e3_eval.sh vqa            # auto-detects models with E1 gen manifests
VQA_MODELS="base final991 v6" bash run_e3_eval.sh vqa
E1_VQA=1 bash run_e3_eval.sh run  # or hook into the full run (gpu0 chain)
```

`--mock` (with `--mock-acc`) is a deterministic oracle answerer used by the
selftest: no torch/transformers/PIL, no image reads. BLIP-2 caveats: yes/no
bias and weak counting are known (T2I-CompBench itself notes VQA-based
counting underestimates); the paired per-item design keeps model comparisons
fair, CLIP-T remains the primary E1 metric.

## Stats recipe (all components, identical)

Per-item paired diff (y − x), 20000 bootstrap resamples seeded 1234,
2.5/97.5 percentile CI, significant iff the CI excludes 0, win/loss/tie,
optional scipy Wilcoxon signed-rank on nonzero diffs. The bootstrap body is
copied verbatim from `e1_general_t2i/stats_paired.py`, so the replication
pair reproduces the historical numbers exactly (validated: E1 CLIP-T
diff+CI match `stats_summary.json` to 1e-12; Module-A deltas match
`final991_eval_v1/scores/summary.json` to 2e-16; E2 all 10 best-layer
selections match `e2_stats_summary.json`). Non-finite values (e.g.
PSNR=inf) are dropped per metric with `n_dropped` recorded.

Note: `final991_eval_v1` used `random.seed(0)` for its bootstrap; e3_eval
unifies on seed 1234, so replication CIs may differ marginally from the
old file while deltas are identical.

## E2 flow keys and "late-step AUROC"

`probe.py` captures flow maps at (layer, step) combos where the shared
counter coincides (defaults → `flow_map_L0_S0`, `L11_S11`, `L23_S23` in
`maps/<pair>.npz`, shape (n_heads, 4096) f16). The late step is the largest
captured step (t≈0.115, near the trajectory end); **late-step AUROC** =
`per_layer["23"]["flow_auroc"]` from the manifests, compared three-way.
`--verify-npz` recomputes it from the npz maps (head-mean, tie-corrected
AUROC vs mask flags rebuilt from the val records — helpers verbatim from
probe.py). Validated on the existing runs: max manifest-vs-npz drift
4.8e-05 (f16 rounding), tolerance 0.02.

## Guard fix (20261014 smoke finding)

The first GPU smoke run passed the whole GPU0 chain (E1 v6 generation +
CLIP-T) but the A-module v6 generation died at
`generate_same_seed_states.py:181`: the v5 driver guards
`payload["format"] == "gold40_automask_v5_paired_fsdp"`, while the v6
trainer writes the label `gold40_automask_v6_align_fsdp` (keys identical,
label different). Fix (e3_eval-side; the v5 script stays untouched as the
final991 reference generator): `gen_same_seed_states_v6.py` is a
byte-identical copy whose guard accepts BOTH labels — verified by diff (only
lines 181-182 differ) and enforced by a selftest audit (≤4 diff lines, all
inside the guard). `run_e3_eval.sh indomain-gen` now calls the copy. The
`BAGEL_ROOT = Path(__file__).resolve().parents[3]` line resolves to the
BAGEL root from `e3_eval/` as well (same tree depth), so every other line —
defaults, sys.path, model build — behaves identically.

Guard-risk audit of the other v6 paths (none found): `probe.py` loads the
checkpoint only through `build_inference_model` -> `load_trainable_strict`
(key-strict, no format check) and never touches `payload["format"]`;
`generate_t2i.py` reads only `payload.get("step")`. The GPU0 smoke run
passing end-to-end independently confirms both paths.

## Running (GPU instance, 2x A100)

```bash
bash run_e3_eval.sh selftest   # CPU only
bash run_e3_eval.sh preflight  # env + reusable-artifact audit + shim dry-run (v6 + v7 when resolvable)
bash run_e3_eval.sh smoke      # v6-only tiny pipeline (~30 min) + stats
bash run_e3_eval.sh run        # full four-way: gpu0 E1 gen+score (v6 skipped when complete, v7 generated) -> VQA (default on);
                               #   gpu1 A gen -> probe -> score (same skip rule); then stats + summary
bash run_e3_eval.sh status
```

Env knobs: `V6_CKPT V7_CKPT MODEL_DIR EVAL_PY N_BOOT GPU0 GPU1 E1_MAX_ITEMS
E1_MODELS INDO_MODELS INDO_SEEDS PROBE_LIMIT PROBE_STEPS PROBE_VERIFY_NPZ
VQA_MODELS E1_VQA`. `V7_CKPT` defaults to the resolver chain (explicit
override -> `outputs/gold40_automask_v7_align_fsdp_full_shard/checkpoints/latest.pt`
-> newest `outputs/*v7*/checkpoints/latest.pt`); `E1_MODELS`/`INDO_MODELS`
default to `v6 v7` (the granular stages loop over them); `E1_VQA` defaults
to 1 (BLIP-2 pass runs after E1 scoring; `E1_VQA=0` opts out).
Scoring uses
`/group/40016/niightli/BAGEL/venvs/geneval_eval/bin/python` (torch +
open_clip ViT-g-14 + dinov2 + lpips + transformers for the local BLIP-2
VQA scorer); generation/probing use system `python3` with the BAGEL stack.

Resumability / idempotent four-way: E1 gen and the probe skip already-done
items by manifest, and the launcher additionally skips the whole stage when
the previous three-way run's artifacts are complete (E1 manifest >= 3x150
rows, `clipT_{model}.jsonl` >= 450 rows, probe >= 51 ok pairs, A-module
splits >= pairs x 2 x seeds rows) — a four-way `run` after the completed
three-way run regenerates only v7 (plus the four-way VQA pass, which is
append-resumable per row). The v5 A-module generator is NOT resumable
(non-empty dir without `--overwrite` aborts), so `indomain-gen` auto-passes
`--overwrite` only when a split is incomplete.

## Limitations / notes

- Runtime requires the merged server tree (sibling `e1_general_t2i/`,
  `e2_token_probe/`, `final991_automask_v5_data/`, v5 trainer dir); on the
  `exp/e3-align-v6` branch alone only `e3_eval/` exists — the e1/e2 dirs
  live on their own branches by design.
- Model weights default to `/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT`
  (GPU-instance local); override with `MODEL_DIR` if needed.
- `dinov2`/`lpips` load failures degrade to null columns (stats drop them
  per metric) instead of crashing, unlike the original scorer.
- VQA faithfulness (E1) is merged only if `vqa_{model}.jsonl` rows exist;
  produce them with `run_e3_eval.sh vqa` (local BLIP-2, `vqa_scorer.py`). The
  historical E1 API-based rows never materialized (HTTP 403), so the VQA
  section appears once the local scorer runs.
- The author delivered the launcher only; GPU evaluation itself was NOT
  run from the authoring session (per task scope).
