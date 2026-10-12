"""Judge v1 Calibration Plan — freeze and audit.

This script:
  1. Freezes the 10-pair stratified calibration set (pre-registered by
     pair_id lexicographic order within each intervention type).
  2. Freezes the Judge v1 prompt template, model version, decoding
     parameters, input ordering, blinding scheme, failure/retry rules,
     human scoring form, and disagreement adjudication rules.
  3. Writes two read-only output files:
       judge_v1_calibration_plan.json   (machine-readable frozen plan)
       judge_v1_calibration_plan_audit.json  (audit report)
  4. Does NOT call any Judge.
  5. Does NOT produce any TS/PS/BA/LS/IQ scores.
  6. Does NOT modify any frozen asset.

Entry point:
  python -m experiments.counterfactual_cot.evaluation.judge_v1_calibration_plan
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/group/40016/niightli/BAGEL").resolve()
EVAL_DIR = REPO_ROOT / "experiments/counterfactual_cot/evaluation"
OUT_DIR = EVAL_DIR / "judge_v1_calibration_plan"

V3_MANIFEST = (
    REPO_ROOT / "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/manifests/"
    "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl"
)
V3_MANIFEST_SHA_EXPECT = "5960e1eebb4d7a394a7b7cba2b2c17c686d184756b637110ef8b43397e172061"

JUDGE_V1_PY = EVAL_DIR / "judge_v1.py"
JUDGE_V1_PY_SHA_EXPECT = "c2bd7e849f5508cc026c6ef7a3d5a3552353e2a919e3a2c79d6ad1aa62961d3b"

PRECHECK_REPORT = EVAL_DIR / "judge_v1_precheck/judge_v1_precheck_report.json"
PRECHECK_REPORT_SHA_EXPECT = "8e7b9bdb51b724ca84cee40cc7486dcd164da138c7edd2fe5be38f67899531db"

# ---------------------------------------------------------------------------
# Pre-registered calibration set (pair_id lexicographic order within type)
# ---------------------------------------------------------------------------

# Selection rule: first 2 pair_ids (lex order) per type for object/attribute/
# count/spatial_position; ALL 2 for spatial_relation.
CALIBRATION_SET: List[Dict[str, Any]] = [
    # object_replace × 2
    {
        "pair_id": "cf_source_city_street_object_replace_0000",
        "intervention_type": "object_replace",
        "calibration_index": 0,
        "selection_rule": "lex_order_rank_0_of_object_replace",
        "intervention": {"entity_id": "car_1", "field": "category",
                         "old_value": "car", "new_value": "van"},
        "num_must_preserve_atoms": 4,
        "metadata_warning": False,
    },
    {
        "pair_id": "cf_source_classroom_object_replace_0000",
        "intervention_type": "object_replace",
        "calibration_index": 1,
        "selection_rule": "lex_order_rank_1_of_object_replace",
        "intervention": {"entity_id": "chair_1", "field": "category",
                         "old_value": "chair", "new_value": "stool"},
        "num_must_preserve_atoms": 4,
        "metadata_warning": False,
    },
    # attribute_replace × 2
    {
        "pair_id": "cf_source_city_street_attribute_replace_0000",
        "intervention_type": "attribute_replace",
        "calibration_index": 2,
        "selection_rule": "lex_order_rank_0_of_attribute_replace",
        "intervention": {"entity_id": "car_1", "field": "attributes.color",
                         "old_value": "red", "new_value": "blue"},
        "num_must_preserve_atoms": 4,
        "metadata_warning": False,
    },
    {
        "pair_id": "cf_source_classroom_attribute_replace_0000",
        "intervention_type": "attribute_replace",
        "calibration_index": 3,
        "selection_rule": "lex_order_rank_1_of_attribute_replace",
        "intervention": {"entity_id": "chair_1", "field": "attributes.color",
                         "old_value": "yellow", "new_value": "red"},
        "num_must_preserve_atoms": 4,
        "metadata_warning": False,
    },
    # count_change × 2
    {
        "pair_id": "cf_source_city_street_count_change_0000",
        "intervention_type": "count_change",
        "calibration_index": 4,
        "selection_rule": "lex_order_rank_0_of_count_change",
        "intervention": {"entity_id": "car_1", "field": "count",
                         "old_value": "3", "new_value": "2"},
        "num_must_preserve_atoms": 4,
        "metadata_warning": False,
    },
    {
        "pair_id": "cf_source_classroom_count_change_0000",
        "intervention_type": "count_change",
        "calibration_index": 5,
        "selection_rule": "lex_order_rank_1_of_count_change",
        "intervention": {"entity_id": "chair_1", "field": "count",
                         "old_value": "1", "new_value": "2"},
        "num_must_preserve_atoms": 4,
        "metadata_warning": False,
    },
    # spatial_position_change × 2
    {
        "pair_id": "cf_source_city_street_absolute_position_change_v4_absolute_region_review_fix_v6_short_0000",
        "intervention_type": "spatial_position_change",
        "calibration_index": 6,
        "selection_rule": "lex_order_rank_0_of_spatial_position_change",
        "intervention": {"entity_id": "car_1", "field": "location",
                         "old_value": "left_region", "new_value": "right_region"},
        "num_must_preserve_atoms": 4,
        "metadata_warning": False,
        "reporting_note": "spatial_position_change — must be reported separately from spatial_relation_change",
    },
    {
        "pair_id": "cf_source_classroom_absolute_position_change_v4_absolute_region_replacement_v4_0000",
        "intervention_type": "spatial_position_change",
        "calibration_index": 7,
        "selection_rule": "lex_order_rank_1_of_spatial_position_change",
        "intervention": {"entity_id": "chair_1", "field": "location",
                         "old_value": "right_region", "new_value": "left_region"},
        "num_must_preserve_atoms": 4,
        "metadata_warning": False,
        "reporting_note": "spatial_position_change — must be reported separately from spatial_relation_change",
    },
    # spatial_relation_change × 2 (ALL)
    {
        "pair_id": "cf_source_home_kitchen_spatial_relation_change_0000",
        "intervention_type": "spatial_relation_change",
        "calibration_index": 8,
        "selection_rule": "all_spatial_relation_change_lex_rank_0",
        "intervention": {"entity_id": "cup_1", "field": "relations.0.predicate",
                         "old_value": "beside", "new_value": "in_front_of"},
        "num_must_preserve_atoms": 4,
        "metadata_warning": False,
        "reporting_note": "spatial_relation_change — MUST be reported as separate category from spatial_position_change (8 pairs)",
    },
    {
        "pair_id": "cf_source_lakeside_birds_spatial_relation_change_0000",
        "intervention_type": "spatial_relation_change",
        "calibration_index": 9,
        "selection_rule": "all_spatial_relation_change_lex_rank_1",
        "intervention": {"entity_id": "bird_1", "field": "relations.0.predicate",
                         "old_value": "on", "new_value": "beside"},
        "num_must_preserve_atoms": 4,
        "metadata_warning": False,
        "reporting_note": "spatial_relation_change — MUST be reported as separate category from spatial_position_change (8 pairs)",
    },
]

CALIBRATION_PAIR_IDS = [c["pair_id"] for c in CALIBRATION_SET]

# ---------------------------------------------------------------------------
# Frozen Judge v1 prompt template
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
You are a blind image-editing evaluator. You do NOT know which model or \
training condition produced the generated image.

## Task
Score the generated image on five metrics given:
  - SOURCE IMAGE: the original unedited image
  - TARGET IMAGE: the reference edited image (ground truth)
  - GENERATED IMAGE: the image to evaluate
  - INTERVENTION: a structured description of the required edit

## Intervention
Entity: {entity_id}
Field: {field}
Required change: {old_value} → {new_value}

## Preservation requirements
The following atoms must remain unchanged from the source image:
{must_preserve_list}

## Scoring rubric

### TS — Target-atom Success (binary: 0 or 1)
Does the generated image reflect the new value ({new_value}) for \
{entity_id}.{field}?
- 1 = yes, the change is clearly visible and correct
- 0 = no, the change is absent, wrong, or ambiguous

### PS — Preservation Score (ratio 0.0–1.0)
For each atom in the preservation list above, is it unchanged relative \
to the source image?
Report the fraction of preserved atoms: (num_preserved) / (num_total).
Evaluate each atom independently. Do NOT penalize for the intended change.

### BA — Binding Accuracy (ratio 0.0–1.0)
Is the observed change bound to the correct entity ({entity_id})?
If there are multiple instances of the same category, is the change \
applied to the correct one?
- 1.0 = change is on the correct entity only
- 0.0 = change is on the wrong entity or spread across all instances

### LS — Locality Score (ratio 0.0–1.0)
Is the spatial extent of the change confined to what is required by the \
intervention?
Use ONLY the source image, target image, generated image, and the \
structured intervention description to assess locality.
DO NOT use any pixel-difference mask as ground truth.
- 1.0 = change is perfectly localized
- 0.0 = change bleeds into unrelated regions

### IQ — Image Quality (ratio 0.0–1.0)
Is the generated image clear, complete, and free of severe artifacts?
- 1.0 = no visible artifacts, natural appearance
- 0.0 = severe distortions, tears, checkerboard patterns, or catastrophic \
composition failure

## Output format (JSON only, no other text)
{{
  "TS": <0 or 1>,
  "PS": <float 0.0–1.0>,
  "BA": <float 0.0–1.0>,
  "LS": <float 0.0–1.0>,
  "IQ": <float 0.0–1.0>,
  "TS_rationale": "<one sentence>",
  "PS_rationale": "<one sentence, list which atoms passed/failed>",
  "BA_rationale": "<one sentence>",
  "LS_rationale": "<one sentence>",
  "IQ_rationale": "<one sentence>",
  "PS_atom_decisions": {{
    "<atom_name>": <true|false>,
    ...
  }}
}}

## Hard rules
- Output ONLY the JSON object above. No markdown, no preamble.
- If you cannot determine a score, output null for that field and add \
  a "score_status": "not_scored" key. Do NOT output 0 as a placeholder.
- Do NOT mention model names, training steps, LoRA, or any condition label.
- Do NOT use any pixel-difference mask as LS ground truth.
"""

# ---------------------------------------------------------------------------
# Frozen model + decoding parameters
# ---------------------------------------------------------------------------

FROZEN_MODEL_CONFIG: Dict[str, Any] = {
    "model_family": "GPT-4o",
    "model_version": "gpt-4o-2024-11-20",
    "model_version_note": (
        "Exact version string must be pinned at execution time. "
        "If gpt-4o-2024-11-20 is unavailable, execution must STOP and "
        "await re-authorization with the new version string."
    ),
    "api_provider": "OpenAI",
    "input_modalities": ["image", "text"],
    "decoding": {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "seed": 42,
        "response_format": "json_object",
    },
    "decoding_note": (
        "temperature=0.0 and seed=42 are required for reproducibility. "
        "Any deviation requires re-authorization."
    ),
}

# ---------------------------------------------------------------------------
# Frozen input ordering and blinding scheme
# ---------------------------------------------------------------------------

FROZEN_INPUT_ORDER: Dict[str, Any] = {
    "image_order_in_prompt": ["source_image", "target_image", "generated_image"],
    "image_order_note": (
        "Source image is always first, target image second, generated image "
        "third. This order is fixed and must not vary across pairs or conditions."
    ),
    "blinding_scheme": {
        "condition_label_hidden": True,
        "hidden_fields": [
            "condition_name (A/B/A_step000/A_step400/B_step000/B_step400)",
            "model_name or architecture",
            "training_step_number",
            "LoRA_presence_or_absence",
            "output_directory_side_labels",
        ],
        "visible_to_judge": [
            "source_image",
            "target_image",
            "generated_image",
            "intervention (entity_id, field, old_value, new_value)",
            "preservation.must_preserve atom list",
        ],
        "blinding_implementation": (
            "The judge runner must strip all condition metadata from the "
            "prompt and image filenames before submission. Generated image "
            "filenames must be replaced with a neutral identifier "
            "(e.g., 'generated_image.png') before passing to the Judge API."
        ),
    },
    "pair_order_in_batch": "manifest_order_frozen (Gold40 v3 manifest line order)",
    "condition_order_in_batch": (
        "Conditions are submitted in separate batches; the Judge never sees "
        "two conditions for the same pair in the same API call."
    ),
}

# ---------------------------------------------------------------------------
# Failure and retry rules
# ---------------------------------------------------------------------------

FROZEN_FAILURE_RETRY_RULES: Dict[str, Any] = {
    "api_error_retry": {
        "max_retries": 2,
        "retry_delay_seconds": 10,
        "on_third_failure": "mark_pair_as_not_scored and continue",
    },
    "malformed_json_retry": {
        "max_retries": 1,
        "on_second_failure": "mark_pair_as_not_scored and continue",
    },
    "out_of_range_score_handling": (
        "If any score is outside its allowed range (TS not in {0,1}; "
        "PS/BA/LS/IQ not in [0.0, 1.0]), mark that metric as not_scored "
        "for that pair. Do NOT clamp or impute."
    ),
    "null_placeholder_prohibition": (
        "A score of 0 MUST NOT be used as a placeholder for a missing score. "
        "Missing scores must be null or score_status=not_scored."
    ),
    "no_auto_parameter_adjustment": (
        "On any failure, do NOT adjust temperature, seed, model version, "
        "prompt, or any other parameter. Stop and report."
    ),
    "no_rerun_after_seeing_results": (
        "Once any score is produced for a pair, that pair cannot be re-run "
        "to improve its score. Re-runs are only allowed for API/JSON errors."
    ),
}

# ---------------------------------------------------------------------------
# Human scoring form (calibration round)
# ---------------------------------------------------------------------------

HUMAN_SCORING_FORM: Dict[str, Any] = {
    "purpose": (
        "Human annotators independently score the 10 calibration pairs "
        "to establish ground-truth labels for Judge calibration."
    ),
    "annotator_requirements": {
        "num_annotators": 2,
        "annotator_blinding": "same blinding scheme as Judge (condition hidden)",
        "annotator_independence": "annotators must not discuss scores before submission",
    },
    "per_pair_fields": [
        {
            "field": "TS",
            "type": "binary",
            "allowed_values": [0, 1],
            "instruction": "Is the required change (old_value → new_value) clearly visible and correct?",
        },
        {
            "field": "PS",
            "type": "ratio_0_to_1_two_decimal",
            "instruction": "For each must_preserve atom, mark true/false. PS = num_true / num_total.",
            "sub_field": "PS_atom_decisions (one entry per must_preserve atom)",
        },
        {
            "field": "BA",
            "type": "ratio_0_to_1_two_decimal",
            "instruction": "Is the change bound to the correct entity_id? 1.0=correct, 0.0=wrong entity.",
        },
        {
            "field": "LS",
            "type": "ratio_0_to_1_two_decimal",
            "instruction": (
                "Is the change spatially confined to the required region? "
                "Use source/target/generated images and intervention description only. "
                "DO NOT use pixel-difference masks."
            ),
        },
        {
            "field": "IQ",
            "type": "ratio_0_to_1_two_decimal",
            "instruction": "Is the image clear, complete, and free of severe artifacts?",
        },
        {
            "field": "rationale",
            "type": "free_text",
            "instruction": "One sentence per metric explaining the score.",
        },
        {
            "field": "score_status",
            "type": "enum",
            "allowed_values": ["scored", "not_scored"],
            "instruction": "If unable to score any metric, set score_status=not_scored for that metric.",
        },
    ],
    "submission_format": "JSON matching Judge output format exactly",
    "null_placeholder_prohibition": "Do NOT fill 0 for an unscored field. Use null.",
}

# ---------------------------------------------------------------------------
# Disagreement adjudication rules
# ---------------------------------------------------------------------------

FROZEN_ADJUDICATION_RULES: Dict[str, Any] = {
    "calibration_phase_adjudication": {
        "ts_disagreement": {
            "trigger": "annotator_1.TS != annotator_2.TS",
            "resolution": "third_annotator_tiebreak",
            "fallback": "mark_as_disputed_exclude_from_calibration_anchor",
        },
        "continuous_metric_disagreement": {
            "trigger": "abs(annotator_1.X - annotator_2.X) > 0.3 for X in {PS, BA, LS, IQ}",
            "resolution": "discussion_round_then_consensus_or_third_annotator",
            "fallback": "use_mean_of_two_annotators_and_flag_as_low_confidence",
        },
        "judge_vs_human_disagreement": {
            "trigger": (
                "abs(judge.X - human_consensus.X) > 0.3 for any metric X, "
                "or judge.TS != human_consensus.TS"
            ),
            "resolution": "human_consensus_takes_precedence",
            "action": (
                "Flag pair as calibration_failure. "
                "Do NOT adjust Judge prompt or parameters mid-batch. "
                "Accumulate failures and review after full calibration batch."
            ),
        },
    },
    "formal_scoring_phase_adjudication": {
        "spot_check_rate": "10% random sample of formal 40-pair results reviewed by human",
        "dispute_trigger": "human_spot_check disagrees with Judge on TS for any pair",
        "dispute_resolution": "human_overrides_Judge_for_that_pair",
        "no_retroactive_prompt_change": (
            "Prompt, model, and decoding parameters are frozen for the entire "
            "formal scoring batch. No changes after seeing any results."
        ),
    },
    "spatial_relation_vs_position_adjudication": {
        "rule": (
            "spatial_relation_change (2 pairs) and spatial_position_change "
            "(8 pairs) must be adjudicated and reported as separate categories. "
            "Disagreements in one category do not affect the other."
        ),
    },
}

# ---------------------------------------------------------------------------
# Reporting structure
# ---------------------------------------------------------------------------

FROZEN_REPORTING_STRUCTURE: Dict[str, Any] = {
    "formal_evaluation_set": "all_40_pairs (Gold40 v3 full set)",
    "formal_evaluation_note": (
        "Gold40 v3 is the TRAINING SET. All results are overfit/capacity "
        "diagnostic only. NOT generalization performance."
    ),
    "reporting_groups": {
        "calibration_10": {
            "description": "10 calibration pairs (pre-registered, scored first)",
            "pair_ids": CALIBRATION_PAIR_IDS,
            "reported_separately": True,
        },
        "non_calibration_30": {
            "description": "Remaining 30 pairs (not used in calibration)",
            "reported_separately": True,
            "note": "Scored in the same formal batch; reported separately to distinguish calibration from non-calibration.",
        },
        "full_40": {
            "description": "All 40 pairs combined",
            "reported_separately": True,
        },
    },
    "intervention_type_reporting": {
        "attribute_replace": {"n": 10, "reported_separately": True},
        "count_change": {"n": 10, "reported_separately": True},
        "object_replace": {"n": 10, "reported_separately": True},
        "spatial_position_change": {
            "n": 8,
            "reported_separately": True,
            "note": "MUST NOT be merged with spatial_relation_change",
        },
        "spatial_relation_change": {
            "n": 2,
            "reported_separately": True,
            "note": "MUST NOT be merged with spatial_position_change",
        },
        "merge_prohibition": (
            "spatial_position_change (8) and spatial_relation_change (2) "
            "MUST NEVER be reported as a single '10 spatial' number."
        ),
    },
    "conditions_reported": ["A_step000", "A_step400", "B_step000", "B_step400"],
    "condition_comparison_note": (
        "step000 conditions are pre-training baselines. "
        "step400 conditions are post-training. "
        "A vs B comparison is the primary experimental contrast. "
        "MSE is a training diagnostic only; TS/PS/BA/LS/IQ are the "
        "grounding effectiveness metrics."
    ),
    "grounded_success_threshold": {
        "condition": "TS==1 AND PS>=0.8 AND BA>=0.8 AND LS>=0.7 AND IQ>=0.6",
        "note": "Pairs with any not_scored metric are excluded from grounded_success rate.",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    failures: List[str] = []

    # 1. Verify upstream anchors
    manifest_sha = sha256_file(V3_MANIFEST)
    if manifest_sha != V3_MANIFEST_SHA_EXPECT:
        failures.append(f"manifest_sha_mismatch: expected={V3_MANIFEST_SHA_EXPECT} observed={manifest_sha}")

    judge_v1_sha = sha256_file(JUDGE_V1_PY)
    if judge_v1_sha != JUDGE_V1_PY_SHA_EXPECT:
        failures.append(f"judge_v1_py_sha_mismatch: expected={JUDGE_V1_PY_SHA_EXPECT} observed={judge_v1_sha}")

    precheck_sha = sha256_file(PRECHECK_REPORT)
    if precheck_sha != PRECHECK_REPORT_SHA_EXPECT:
        failures.append(f"precheck_report_sha_mismatch: expected={PRECHECK_REPORT_SHA_EXPECT} observed={precheck_sha}")

    # 2. Verify calibration pair_ids exist in manifest
    manifest_ids = set()
    with open(V3_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_ids.add(json.loads(line)["pair_id"])
    for c in CALIBRATION_SET:
        if c["pair_id"] not in manifest_ids:
            failures.append(f"calibration_pair_not_in_manifest: {c['pair_id']}")

    # 3. Verify calibration set size and type distribution
    type_counts: Dict[str, int] = {}
    for c in CALIBRATION_SET:
        t = c["intervention_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    expected_type_counts = {
        "object_replace": 2,
        "attribute_replace": 2,
        "count_change": 2,
        "spatial_position_change": 2,
        "spatial_relation_change": 2,
    }
    for t, expected in expected_type_counts.items():
        observed = type_counts.get(t, 0)
        if observed != expected:
            failures.append(f"calibration_type_count_mismatch: {t} expected={expected} observed={observed}")
    if len(CALIBRATION_SET) != 10:
        failures.append(f"calibration_set_size={len(CALIBRATION_SET)} (expected 10)")

    # 4. Verify no calibration pair has metadata_warning=True
    # (all 10 selected pairs are clean; metadata warnings are on pumpkin and flower_shop
    # which are NOT in the calibration set)
    for c in CALIBRATION_SET:
        if c.get("metadata_warning"):
            failures.append(f"calibration_pair_has_metadata_warning: {c['pair_id']}")

    # 5. Verify calibration indices are 0-9 unique
    indices = [c["calibration_index"] for c in CALIBRATION_SET]
    if sorted(indices) != list(range(10)):
        failures.append(f"calibration_indices_not_0_to_9: {indices}")

    # 6. Verify prompt template contains required prohibition text
    if "DO NOT use any pixel-difference mask" not in JUDGE_PROMPT_TEMPLATE:
        failures.append("prompt_template_missing_mask_prohibition")
    if "null" not in JUDGE_PROMPT_TEMPLATE or "not_scored" not in JUDGE_PROMPT_TEMPLATE:
        failures.append("prompt_template_missing_null_not_scored_rule")
    if "Do NOT mention model names" not in JUDGE_PROMPT_TEMPLATE:
        failures.append("prompt_template_missing_blinding_instruction")

    # 7. Verify model version is pinned
    if FROZEN_MODEL_CONFIG["model_version"] == "":
        failures.append("model_version_not_pinned")

    # 8. Verify temperature=0.0 (reproducibility)
    if FROZEN_MODEL_CONFIG["decoding"]["temperature"] != 0.0:
        failures.append(f"temperature_not_zero: {FROZEN_MODEL_CONFIG['decoding']['temperature']}")

    # 9. Verify spatial reporting separation rule is present
    merge_prohibition = FROZEN_REPORTING_STRUCTURE["intervention_type_reporting"]["merge_prohibition"]
    if "spatial_position_change" not in merge_prohibition or "spatial_relation_change" not in merge_prohibition:
        failures.append("spatial_merge_prohibition_missing")

    # 10. Assemble calibration plan
    plan: Dict[str, Any] = {
        "plan_id": "judge_v1_calibration_plan_v1",
        "plan_version": "v1",
        "created_at_iso": now_iso(),
        "plan_is_frozen": True,
        "amendment_policy": (
            "Any change to calibration set, prompt, model version, decoding "
            "params, blinding scheme, or adjudication rules requires a new "
            "plan version (v2+) and MUST NOT overwrite v1."
        ),
        "upstream_anchors": {
            "v3_manifest_sha256_expected": V3_MANIFEST_SHA_EXPECT,
            "v3_manifest_sha256_observed": manifest_sha,
            "judge_v1_py_sha256_expected": JUDGE_V1_PY_SHA_EXPECT,
            "judge_v1_py_sha256_observed": judge_v1_sha,
            "precheck_report_sha256_expected": PRECHECK_REPORT_SHA_EXPECT,
            "precheck_report_sha256_observed": precheck_sha,
        },
        "calibration_set": {
            "num_pairs": len(CALIBRATION_SET),
            "selection_rule": (
                "Lexicographic order of pair_id within each intervention type. "
                "First 2 pairs for object_replace, attribute_replace, count_change, "
                "spatial_position_change. ALL 2 pairs for spatial_relation_change."
            ),
            "pairs": CALIBRATION_SET,
            "calibration_pair_ids": CALIBRATION_PAIR_IDS,
            "type_distribution": type_counts,
            "metadata_warning_pairs_excluded": [
                "cf_source_grocery_display_absolute_position_change_v4_absolute_region_pumpkin_in_basket_fix_v5_0000",
                "cf_source_flower_shop_absolute_position_change_v4_absolute_region_wide_table_fix_v6_0000",
            ],
            "metadata_warning_pairs_note": (
                "The two metadata_warning pairs (pumpkin, flower_shop) are NOT "
                "in the calibration set. They are in the formal 40-pair set and "
                "will be scored in the non-calibration-30 group."
            ),
        },
        "judge_prompt_template": JUDGE_PROMPT_TEMPLATE,
        "frozen_model_config": FROZEN_MODEL_CONFIG,
        "frozen_input_order": FROZEN_INPUT_ORDER,
        "frozen_failure_retry_rules": FROZEN_FAILURE_RETRY_RULES,
        "human_scoring_form": HUMAN_SCORING_FORM,
        "frozen_adjudication_rules": FROZEN_ADJUDICATION_RULES,
        "frozen_reporting_structure": FROZEN_REPORTING_STRUCTURE,
        "declarations": [
            "No Judge invoked in this step.",
            "No TS/PS/BA/LS/IQ scores produced.",
            "No model loaded.",
            "No GPU forward executed.",
            "No frozen asset modified.",
            "This plan is read-only once written.",
        ],
    }

    # 11. Write plan
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = OUT_DIR / "judge_v1_calibration_plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plan_sha = sha256_file(plan_path)

    # 12. Write audit report
    overall = "passed" if len(failures) == 0 else "failed"
    audit: Dict[str, Any] = {
        "audit_type": "judge_v1_calibration_plan_freeze_audit",
        "created_at_iso": now_iso(),
        "overall_status": overall,
        "total_failures": len(failures),
        "all_failures": failures,
        "calibration_plan_path": str(plan_path),
        "calibration_plan_sha256": plan_sha,
        "calibration_plan_size_bytes": plan_path.stat().st_size,
        "checks_performed": [
            "manifest_sha256_verified",
            "judge_v1_py_sha256_verified",
            "precheck_report_sha256_verified",
            "all_10_calibration_pair_ids_in_manifest",
            "calibration_type_distribution_correct",
            "calibration_set_size_equals_10",
            "no_metadata_warning_pairs_in_calibration_set",
            "calibration_indices_0_to_9_unique",
            "prompt_template_contains_mask_prohibition",
            "prompt_template_contains_null_not_scored_rule",
            "prompt_template_contains_blinding_instruction",
            "model_version_pinned",
            "temperature_equals_zero",
            "spatial_merge_prohibition_present",
        ],
        "declarations": [
            "No Judge invoked.",
            "No TS/PS/BA/LS/IQ scores produced.",
            "No model loaded.",
            "No GPU forward executed.",
            "No frozen asset modified.",
        ],
    }
    audit_path = OUT_DIR / "judge_v1_calibration_plan_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    audit_sha = sha256_file(audit_path)

    # 13. Print summary
    summary = {
        "overall_status": overall,
        "total_failures": len(failures),
        "all_failures": failures,
        "calibration_plan_path": str(plan_path),
        "calibration_plan_size_bytes": plan_path.stat().st_size,
        "calibration_plan_sha256": plan_sha,
        "audit_path": str(audit_path),
        "audit_size_bytes": audit_path.stat().st_size,
        "audit_sha256": audit_sha,
    }
    print(json.dumps(summary, indent=2))
    return 0 if overall == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
