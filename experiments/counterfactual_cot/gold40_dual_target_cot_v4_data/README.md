# Gold40 Dual Target-CoT v4 Data Preparation

This directory prepares the next-stage Gold40 dataset for v4-style paired Target-CoT training.

## Output semantics

Each original Gold40 pair becomes one paired macro-example with two branches:

```text
source_state branch:
  Source Image = S
  Source Prompt = P_source
  Source CoT = C_source
  Target CoT = C_source
  Target Image = S

target_state branch:
  Source Image = S
  Source Prompt = P_source
  Source CoT = C_source
  Target CoT = C_target
  Target Image = T
```

The paired manifest has 40 macro-examples. The flattened branch manifest has 80 records, but future training must keep the two records of each `macro_pair_id` together in one paired macro-step.

## Canonical source policy

The script uses structured scenes and current target semantics from:

```text
gold40_final/manifest/intervention_pairs_gold40_final.jsonl
```

and source/target image paths plus SHA256 from:

```text
gold40_final/manifest/asset_index.jsonl
```

It also records the SHA256 and semantic delta against the frozen v3 semantic reference:

```text
reruns/nanobanana_v4_absolute_region/manifests/intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl
```

Stored `cot_text` is never trusted directly. It is re-rendered from structured `source.cot` / `target.cot` after validating that applying the intervention reproduces the target scene and changes exactly one scene atom.

## Dynamic masks

For every pair, the script creates a dynamic RGB-difference **candidate** mask:

```text
max-channel RGB difference > 24
→ dilation 9 px
→ median filter 3 px
```

It audits source/target SHA256, image decode/size, semantic transition, mask geometry, area ratio, bounding box and border contact. These masks are candidate supervision only, not gold segmentation masks.

The generated report deliberately marks all records:

```text
candidate_ready_manual_review_required
```

Do not train Gold40 from these masks until every pair has an explicit human decision in a later `review_decisions.jsonl` file.

## Human-review gate

`build_gold40_reviewed_manifest.py` never creates approvals or ROI masks. A human reviewer must supply exactly one JSONL decision per candidate `pair_id` (all 40). Each ROI is a **1024×1024 binary mask in the original image coordinate system**; nonzero pixels are included.

Use up to three mutually disjoint semantic masks:

- `source_changed_roi`: source-only support, e.g. the object's old location after a move.
- `target_changed_roi`: target-only support, e.g. the object's new location after a move.
- `shared_changed_roi`: the same spatial support belongs to both states, e.g. color/attribute replacement, object replacement at the same location, or a deleted object replaced by background.

For same-location edits, provide `shared_changed_roi` rather than arbitrarily cutting one object into source and target halves. For movement edits, provide source-only and target-only masks; add shared only for genuinely shared redraw/occlusion areas. The provided masks must be non-empty collectively, pixel-disjoint, match the candidate-mask dimensions, cover every candidate-mask pixel, and provide support for both states (shared satisfies both).

```json
{"pair_id":"...","approved":true,"reviewer":"reviewer-id","reviewed_at":"2026-09-10T12:00:00+00:00","notes":"Attribute replacement at one location; shared ROI covers the entire edited object.","shared_changed_roi":"review_rois/...__shared.png"}
```

```json
{"pair_id":"...","approved":true,"reviewer":"reviewer-id","reviewed_at":"2026-09-10T12:00:00+00:00","notes":"Object moved from old position to new position.","source_changed_roi":"review_rois/...__source_only.png","target_changed_roi":"review_rois/...__target_only.png"}
```

`approved` must be the JSON boolean `true`; `reviewer` and `notes` must be non-empty; `reviewed_at` must be timezone-aware ISO-8601. ROI values are binary mask paths (or objects with `path`/`mask_path`); relative paths resolve from the decisions file. Do not draw the full image, background texture drift, or candidate-mask dilation halo unless that area is necessary for the semantic intervention.

```bash
cd /group/40016/niightli/BAGEL
python experiments/counterfactual_cot/gold40_dual_target_cot_v4_data/build_gold40_reviewed_manifest.py \
  --review-decisions experiments/counterfactual_cot/gold40_dual_target_cot_v4_data/artifacts/review_decisions.jsonl
```

A successful run writes `artifacts/gold40_source_target_paired_reviewed_v4.jsonl` plus `artifacts/gold40_human_review_audit_v4.json`. On a failed check it writes a failed audit only; it does not create or replace the reviewed manifest. Use `--overwrite` only when replacing a successful reviewed manifest.

## Run

```bash
cd /group/40016/niightli/BAGEL
python experiments/counterfactual_cot/gold40_dual_target_cot_v4_data/prepare_gold40_dual_target_data.py
```

Outputs are written under:

```text
experiments/counterfactual_cot/gold40_dual_target_cot_v4_data/artifacts/
```

Key files:

- `gold40_source_target_paired_candidate_v4.jsonl`: 40 nested paired macro-examples.
- `gold40_branch_records_candidate_v4.jsonl`: 80 flattened branch records.
- `dynamic_masks/`: 40 generated candidate masks.
- `gold40_dynamic_mask_audit_v4.json`: all semantic, asset and mask audits.
- `review/gold40_dynamic_mask_contact_sheet.jpg`: source / red-mask overlay / target review gallery.

## Reviewed source-scene split

Only run the split builder after `build_gold40_reviewed_manifest.py` has produced the
approved paired manifest. It rejects candidate/unreviewed records and requires every
record to retain the complete nested candidate payload plus top-level
`source_scene_id`, `intervention_type`, `review.status=approved`, and approved dynamic
region boxes. It assigns whole `source_scene_id` groups, never individual pairs, so a
scene cannot appear in more than one split. With exactly 10 source scenes, its default
allocation is 6/2/2 train/val/test scenes; the deterministic seed also resolves ties
while minimizing intervention-type imbalance.

```bash
cd /group/40016/niightli/BAGEL
python experiments/counterfactual_cot/gold40_dual_target_cot_v4_data/build_gold40_source_scene_splits.py \
  --seed 20260910 \
  --smoke-train-pairs 8
```

The default input is `artifacts/gold40_source_target_paired_reviewed_v4.jsonl`; outputs
are `artifacts/splits/train.jsonl`, `val.jsonl`, `test.jsonl`, `split_audit.json`, and
(if requested) `smoke_train_8_pairs.jsonl`. Current Gold40 has 10 source scenes with 4
pairs each, so a leakage-free smoke set must contain 4, 8, 12, ... pairs; the originally
suggested 6-pair smoke would split a scene and is therefore intentionally rejected.

## Freeze the single-pair v4 proof

Before Gold40 training, archive the existing proof-of-concept without modifying it:

```bash
cd /group/40016/niightli/BAGEL
python experiments/counterfactual_cot/gold40_dual_target_cot_v4_data/archive_single_pair_v4.py
```

It hashes the step 0/500 artifacts, checkpoints, logs, and switch report into
`single_pair_v4_provenance.json`. It deliberately labels the evidence as single-pair
only, not generalization.

## Next experimental sequence

1. Review each dynamic candidate mask and approve/reject it explicitly.
2. Build the strict source-scene split from the approved paired manifest.
3. Begin with a small paired smoke subset after approval.
4. Train source-state/target-state macro-steps, not independently shuffled branch records.
5. Evaluate same-seed source-CoT/target-CoT switching on held-out source scenes.

## Same-seed evaluation audit

`audit_gold40_switch.py` is independent of training. It joins three manifests by
`(pair_id, seed, state)` and writes a JSON report with every pair/seed row, per-pair,
overall, and per-`intervention_type` percentile bootstrap 95% confidence intervals.
Overall/type summaries resample `pair_id` clusters while retaining all same-seed
observations; each pair summary resamples that pair's same-seed observations.

The generation-result JSONL must contain one record per branch, for example:

```json
{"pair_id":"example_0001","seed":1234,"state":"source_state","output_path":"/outputs/example_0001/source_1234.png"}
{"pair_id":"example_0001","seed":1234,"state":"target_state","output_path":"/outputs/example_0001/target_1234.png"}
```

The automated-audit-label JSONL must contain the same keys and an explicit boolean
`state_correct`. This is the only input used for source-state, target-state, and
same-seed CoT Switch Accuracy; the tool never infers semantic correctness from image
pixels:

```json
{"pair_id":"example_0001","seed":1234,"state":"source_state","state_correct":true}
{"pair_id":"example_0001","seed":1234,"state":"target_state","state_correct":true}
```

The reviewed paired manifest contains source/target reference images, intervention type,
and an approved `reviewed_mask`: the union of the two human-reviewed change ROIs.
The gate writes this mask under `artifacts/reviewed_allowed_change_masks/` with a SHA256.
The current candidate manifest is intentionally unreviewed, so it is rejected by default.
`--allow-unreviewed-candidate-masks` is only a diagnostic override and marks all mask
MAE evidence accordingly.

```bash
cd /group/40016/niightli/BAGEL
python experiments/counterfactual_cot/gold40_dual_target_cot_v4_data/audit_gold40_switch.py \
  --generation-manifest /path/to/generation_results.jsonl \
  --paired-manifest /path/to/reviewed_gold40_pairs.jsonl \
  --automated-audit-labels /path/to/automated_audit_labels.jsonl \
  --output /path/to/gold40_switch_audit.json \
  --bootstrap-samples 10000 --bootstrap-seed 0
```

The report includes state-specific ROI reference MAE and source-reference
outside-mask preservation MAE, both normalized to `[0, 1]`. These are pixel
preservation diagnostics, not semantic-success substitutes.
