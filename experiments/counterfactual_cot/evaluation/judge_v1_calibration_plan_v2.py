"""Judge v1 Calibration Plan v2 — fills GAP-1 through GAP-4 from v1 audit.

GAP-1: Explicitly declares 10 pair × 4 conditions = 40 blind items;
       freezes handling of A_step000/B_step000 bit-exact repeat.
GAP-2: Freezes anonymous item_id generation rule, global execution order
       (manifest × condition, fixed seed), and condition-mapping custody.
GAP-3: Freezes API endpoint (OpenAI direct), image encoding (base64),
       detail=high, and 3-image-per-call layout.
GAP-4: Freezes quantitative calibration pass threshold:
       max_calibration_failures_to_proceed=2 / 10,
       TS Cohen's κ ≥ 0.6 (Judge vs human consensus),
       explicit block condition.

Outputs (MUST NOT overwrite v1):
  judge_v1_calibration_plan_v2/judge_v1_calibration_plan_v2.json
  judge_v1_calibration_plan_v2/judge_v1_calibration_plan_v2_audit.json

No Judge invoked. No scores produced. No frozen asset modified.
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
OUT_DIR = EVAL_DIR / "judge_v1_calibration_plan_v2"

V1_PLAN = EVAL_DIR / "judge_v1_calibration_plan/judge_v1_calibration_plan.json"
V1_PLAN_SHA_EXPECT = "7042af29510809353238ed6fee0cc0563a4c711b4b4182cfc9832803300c643e"

V1_AUDIT = EVAL_DIR / "judge_v1_calibration_plan/judge_v1_calibration_plan_audit.json"
V1_AUDIT_SHA_EXPECT = "d05bd2ec9717afd9028840def0d64242c1d05e5500cf00616798fb5b34607815"

JUDGE_V1_PY = EVAL_DIR / "judge_v1.py"
JUDGE_V1_PY_SHA_EXPECT = "c2bd7e849f5508cc026c6ef7a3d5a3552353e2a919e3a2c79d6ad1aa62961d3b"

V3_MANIFEST_SHA_EXPECT = "5960e1eebb4d7a394a7b7cba2b2c17c686d184756b637110ef8b43397e172061"
STEP000_GATE_SHA_EXPECT = "213564895e789b532d4e53d833a17ab31ef80973d40096a951c58bd7215bf570"

STEP000_GATE_REPORT = (
    REPO_ROOT / "experiments/counterfactual_cot/outputs/"
    "gold40_v3_ab_aligned_inference_nt25_protocol_v1/"
    "step000_alignment_gate_report.json"
)

# ---------------------------------------------------------------------------
# Calibration pair IDs (inherited from v1, unchanged)
# ---------------------------------------------------------------------------

CALIBRATION_PAIR_IDS: List[str] = [
    "cf_source_city_street_object_replace_0000",
    "cf_source_classroom_object_replace_0000",
    "cf_source_city_street_attribute_replace_0000",
    "cf_source_classroom_attribute_replace_0000",
    "cf_source_city_street_count_change_0000",
    "cf_source_classroom_count_change_0000",
    "cf_source_city_street_absolute_position_change_v4_absolute_region_review_fix_v6_short_0000",
    "cf_source_classroom_absolute_position_change_v4_absolute_region_replacement_v4_0000",
    "cf_source_home_kitchen_spatial_relation_change_0000",
    "cf_source_lakeside_birds_spatial_relation_change_0000",
]

CONDITIONS: List[str] = ["A_step000", "A_step400", "B_step000", "B_step400"]

# ---------------------------------------------------------------------------
# GAP-1: 40-item declaration + Step000 bit-exact repeat handling
# ---------------------------------------------------------------------------

GAP1_40_ITEM_DECLARATION: Dict[str, Any] = {
    "calibration_total_items": 40,
    "derivation": "10 calibration pairs × 4 conditions = 40 independent blind items",
    "conditions": CONDITIONS,
    "num_calibration_pairs": 10,
    "step000_bit_exact_repeat_handling": {
        "fact": (
            "A_step000 and B_step000 produce pixel-identical generated images "
            "for all 40 pairs (proven by step000 alignment gate, "
            f"SHA256={STEP000_GATE_SHA_EXPECT}, 17/17 gates passed, "
            "40/40 pairs pixel_max_abs_diff=0)."
        ),
        "submission_rule": (
            "A_step000 and B_step000 items for the same pair are submitted as "
            "two INDEPENDENT blind items with different anonymous item_ids. "
            "The Judge receives the same generated image twice (under different "
            "item_ids) without knowing they are identical."
        ),
        "expected_judge_behavior": (
            "Because the generated image is pixel-identical, a well-calibrated "
            "Judge should produce identical scores for A_step000 and B_step000 "
            "items of the same pair. Any discrepancy is recorded as a "
            "step000_consistency_check_failure for that pair."
        ),
        "step000_consistency_check": {
            "trigger": (
                "For any calibration pair p: "
                "judge_score(A_step000, p) != judge_score(B_step000, p) "
                "for any metric in {TS, PS, BA, LS, IQ}"
            ),
            "action": (
                "Flag as step000_consistency_failure for that pair and metric. "
                "Count toward calibration_failure total (see GAP-4 threshold). "
                "Do NOT suppress or average the two scores."
            ),
            "note": (
                "step000 consistency failures indicate Judge non-determinism or "
                "sensitivity to item ordering. They are diagnostic signals, not "
                "grounds for re-running with different parameters."
            ),
        },
        "step000_gate_report_sha256": STEP000_GATE_SHA_EXPECT,
    },
}

# ---------------------------------------------------------------------------
# GAP-2: Anonymous item_id, execution order, condition mapping custody
# ---------------------------------------------------------------------------

# Item ordering: manifest_order (0-39) × condition_order (A_step000, A_step400,
# B_step000, B_step400). Fixed — no randomization.
# item_id = f"item_{(pair_manifest_index * 4 + condition_index):03d}"
# e.g. pair[0] × A_step000 → item_000, pair[0] × A_step400 → item_001, ...
# pair[9] × B_step400 → item_039

def _build_item_table() -> List[Dict[str, Any]]:
    """Build the frozen 40-item table with anonymous item_ids."""
    table = []
    for pi, pair_id in enumerate(CALIBRATION_PAIR_IDS):
        for ci, condition in enumerate(CONDITIONS):
            item_index = pi * len(CONDITIONS) + ci
            item_id = f"item_{item_index:03d}"
            table.append({
                "item_id": item_id,
                "item_index": item_index,
                "pair_manifest_index": pi,
                "condition_index": ci,
                # The following fields are in the CUSTODY MAP (not sent to Judge)
                "_custody_pair_id": pair_id,
                "_custody_condition": condition,
            })
    return table


ITEM_TABLE = _build_item_table()

# Split into blind manifest (sent to Judge runner) and custody map (sealed)
BLIND_ITEM_MANIFEST: List[Dict[str, Any]] = [
    {"item_id": r["item_id"], "item_index": r["item_index"]}
    for r in ITEM_TABLE
]

CUSTODY_MAP: List[Dict[str, Any]] = [
    {
        "item_id": r["item_id"],
        "pair_id": r["_custody_pair_id"],
        "condition": r["_custody_condition"],
        "pair_manifest_index": r["pair_manifest_index"],
        "condition_index": r["condition_index"],
    }
    for r in ITEM_TABLE
]

# SHA256 of the custody map JSON (sorted, compact) — seals the mapping
_CUSTODY_MAP_BYTES = json.dumps(
    CUSTODY_MAP, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode("utf-8")
CUSTODY_MAP_SHA256 = hashlib.sha256(_CUSTODY_MAP_BYTES).hexdigest()

GAP2_ITEM_ID_AND_ORDER: Dict[str, Any] = {
    "item_id_generation_rule": (
        "item_id = f'item_{(pair_manifest_index * 4 + condition_index):03d}' "
        "where pair_manifest_index ∈ [0,9] (calibration pairs in calibration_set order) "
        "and condition_index ∈ [0,3] for [A_step000, A_step400, B_step000, B_step400]."
    ),
    "item_id_format": "item_NNN (zero-padded 3 digits, 000–039)",
    "global_execution_order": "fixed: item_000 → item_001 → ... → item_039 (no randomization)",
    "execution_order_rationale": (
        "Fixed order (not randomized) because: "
        "(1) the Judge is blinded to condition labels, so order bias is not a concern; "
        "(2) fixed order ensures exact reproducibility without a random seed; "
        "(3) step000 consistency check requires A_step000 and B_step000 items "
        "for the same pair to be submitted in a known relative order."
    ),
    "blind_item_manifest": BLIND_ITEM_MANIFEST,
    "blind_item_manifest_note": (
        "The blind_item_manifest contains ONLY item_id and item_index. "
        "It is the only information passed to the Judge runner. "
        "pair_id and condition are NOT included."
    ),
    "custody_map_sha256": CUSTODY_MAP_SHA256,
    "custody_map_note": (
        "The custody map (item_id → pair_id + condition) is sealed by its SHA256. "
        "It must be stored separately from the Judge runner and must NOT be "
        "accessible to the Judge runner during scoring. "
        "Deblinding occurs ONLY after all 40 items are scored and results are locked."
    ),
    "custody_map_storage": (
        "custody_map is written to "
        "judge_v1_calibration_plan_v2/judge_v1_custody_map.json "
        "and its SHA256 is recorded in this plan. "
        "The Judge runner receives only blind_item_manifest."
    ),
    "deblinding_timing": (
        "Deblinding (joining scores with custody_map) occurs ONLY after: "
        "(1) all 40 items are scored, "
        "(2) scores are locked (no further API calls), "
        "(3) the custody_map SHA256 is verified against the value in this plan."
    ),
}

# ---------------------------------------------------------------------------
# GAP-3: API endpoint, image encoding, detail parameter
# ---------------------------------------------------------------------------

GAP3_API_CONFIG: Dict[str, Any] = {
    "api_provider": "OpenAI",
    "api_type": "direct (not Azure, not proxy)",
    "api_endpoint": "https://api.openai.com/v1/chat/completions",
    "api_endpoint_note": (
        "Must use this exact endpoint. If a proxy or Azure endpoint is used, "
        "execution must STOP and await re-authorization."
    ),
    "model_version": "gpt-4o-2024-11-20",
    "image_encoding": "base64",
    "image_encoding_note": (
        "All images (source, target, generated) are encoded as base64 strings "
        "and embedded in the messages array as image_url content blocks with "
        "url='data:image/png;base64,<base64_string>'. "
        "No external URLs. No file uploads."
    ),
    "image_detail_parameter": "high",
    "image_detail_rationale": (
        "detail='high' is required so that 1024×1024 images are processed at "
        "full resolution (tiled into 512×512 patches). detail='low' would "
        "reduce images to 85-token thumbnails, making LS and IQ assessment "
        "unreliable. detail='auto' is non-deterministic and forbidden."
    ),
    "images_per_api_call": 3,
    "images_per_call_layout": (
        "Each API call contains exactly 3 images in a single messages array: "
        "[source_image (base64, detail=high), "
        " target_image (base64, detail=high), "
        " generated_image (base64, detail=high)]. "
        "All 3 images and the text prompt are in ONE call. "
        "No multi-turn, no separate calls per image."
    ),
    "image_size_note": (
        "Generated images are 1024×1024 PNG (from Protocol v1 visual_image_size=1024). "
        "Source and target images vary in size but are passed as-is (no resizing). "
        "detail='high' handles variable sizes correctly."
    ),
    "request_structure": {
        "model": "gpt-4o-2024-11-20",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<SOURCE_B64>", "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<TARGET_B64>", "detail": "high"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<GENERATED_B64>", "detail": "high"}},
                    {"type": "text", "text": "<FILLED_PROMPT_TEMPLATE>"},
                ],
            }
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1024,
        "seed": 42,
        "response_format": {"type": "json_object"},
    },
    "filename_anonymization": (
        "Before base64 encoding, generated image filenames are renamed to "
        "'generated_image.png' in the prompt context. "
        "Source and target filenames are renamed to 'source_image.png' and "
        "'target_image.png'. No pair_id, condition, or step number appears "
        "in any filename or prompt text."
    ),
}

# ---------------------------------------------------------------------------
# GAP-4: Quantitative calibration pass threshold
# ---------------------------------------------------------------------------

GAP4_CALIBRATION_THRESHOLD: Dict[str, Any] = {
    "calibration_failure_definition": (
        "A calibration_failure is counted for a pair p when: "
        "abs(judge.X(p) - human_consensus.X(p)) > 0.3 for any metric X ∈ {PS, BA, LS, IQ}, "
        "OR judge.TS(p) != human_consensus.TS(p), "
        "OR step000_consistency_failure for pair p (A_step000 score != B_step000 score "
        "for any metric)."
    ),
    "max_calibration_failures_to_proceed": 2,
    "max_calibration_failures_note": (
        "Out of 10 calibration pairs, at most 2 may be calibration_failures "
        "before proceeding to formal 40-pair scoring. "
        "If calibration_failures > 2, execution MUST STOP. "
        "Do NOT proceed to formal scoring. Await re-authorization."
    ),
    "ts_cohen_kappa_threshold": {
        "metric": "Cohen's κ (Judge TS vs human consensus TS, across 10 calibration pairs)",
        "minimum_required": 0.6,
        "computation": (
            "κ = (p_o - p_e) / (1 - p_e) where p_o = observed agreement rate "
            "and p_e = expected agreement by chance. "
            "Computed over the 10 calibration pairs where both Judge and human "
            "produced a TS score (not_scored pairs excluded from κ computation)."
        ),
        "block_condition": (
            "If κ < 0.6, execution MUST STOP regardless of calibration_failure count. "
            "Do NOT proceed to formal scoring. Await re-authorization."
        ),
    },
    "combined_pass_condition": (
        "BOTH of the following must hold to proceed to formal 40-pair scoring: "
        "(1) calibration_failures ≤ 2 out of 10 pairs, "
        "(2) TS Cohen's κ ≥ 0.6 (Judge vs human consensus). "
        "Failure of EITHER condition blocks formal scoring."
    ),
    "block_action": (
        "On block: preserve all calibration scores, human annotations, and "
        "disagreement records. Do NOT delete or modify any scored item. "
        "Do NOT adjust prompt, model, temperature, or any parameter. "
        "Report block reason and await explicit re-authorization."
    ),
    "not_scored_handling_in_threshold": (
        "Pairs where Judge or human returned not_scored for any metric are "
        "counted as calibration_failures for that metric. "
        "They are excluded from κ computation but included in failure count."
    ),
    "step000_consistency_in_threshold": (
        "step000_consistency_failures (A_step000 score != B_step000 score for "
        "same pair) count toward calibration_failure total. "
        "Maximum 1 step000_consistency_failure is allowed within the "
        "max_calibration_failures_to_proceed=2 budget."
    ),
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

    # G01: v1 plan SHA unchanged
    v1_sha_observed = sha256_file(V1_PLAN)
    g01 = v1_sha_observed == V1_PLAN_SHA_EXPECT
    if not g01:
        failures.append(f"G01_v1_plan_sha_mismatch: expected={V1_PLAN_SHA_EXPECT} observed={v1_sha_observed}")

    # G02: v1 audit SHA unchanged
    v1_audit_sha_observed = sha256_file(V1_AUDIT)
    g02 = v1_audit_sha_observed == V1_AUDIT_SHA_EXPECT
    if not g02:
        failures.append(f"G02_v1_audit_sha_mismatch: expected={V1_AUDIT_SHA_EXPECT} observed={v1_audit_sha_observed}")

    # G03: judge_v1.py SHA unchanged
    judge_v1_sha_observed = sha256_file(JUDGE_V1_PY)
    g03 = judge_v1_sha_observed == JUDGE_V1_PY_SHA_EXPECT
    if not g03:
        failures.append(f"G03_judge_v1_py_sha_mismatch: expected={JUDGE_V1_PY_SHA_EXPECT} observed={judge_v1_sha_observed}")

    # G04: step000 gate report SHA
    step000_sha_observed = sha256_file(STEP000_GATE_REPORT)
    g04 = step000_sha_observed == STEP000_GATE_SHA_EXPECT
    if not g04:
        failures.append(f"G04_step000_gate_sha_mismatch: expected={STEP000_GATE_SHA_EXPECT} observed={step000_sha_observed}")

    # G05: 40 items declared
    g05 = GAP1_40_ITEM_DECLARATION["calibration_total_items"] == 40
    if not g05:
        failures.append("G05_total_items_not_40")

    # G06: step000 bit-exact repeat handling present
    g06 = "step000_consistency_check" in GAP1_40_ITEM_DECLARATION["step000_bit_exact_repeat_handling"]
    if not g06:
        failures.append("G06_step000_consistency_check_missing")

    # G07: item_id table has exactly 40 entries
    g07 = len(ITEM_TABLE) == 40
    if not g07:
        failures.append(f"G07_item_table_size={len(ITEM_TABLE)} (expected 40)")

    # G08: item_ids are item_000 to item_039, unique
    expected_ids = {f"item_{i:03d}" for i in range(40)}
    observed_ids = {r["item_id"] for r in ITEM_TABLE}
    g08 = expected_ids == observed_ids
    if not g08:
        failures.append(f"G08_item_id_set_mismatch: missing={expected_ids-observed_ids} extra={observed_ids-expected_ids}")

    # G09: custody map SHA256 is deterministic and recorded
    g09 = len(CUSTODY_MAP_SHA256) == 64 and CUSTODY_MAP_SHA256.isalnum()
    if not g09:
        failures.append("G09_custody_map_sha256_invalid")

    # G10: API endpoint is OpenAI direct
    g10 = GAP3_API_CONFIG["api_endpoint"] == "https://api.openai.com/v1/chat/completions"
    if not g10:
        failures.append("G10_api_endpoint_not_openai_direct")

    # G11: image encoding is base64
    g11 = GAP3_API_CONFIG["image_encoding"] == "base64"
    if not g11:
        failures.append("G11_image_encoding_not_base64")

    # G12: detail=high
    g12 = GAP3_API_CONFIG["image_detail_parameter"] == "high"
    if not g12:
        failures.append("G12_detail_not_high")

    # G13: 3 images per call
    g13 = GAP3_API_CONFIG["images_per_api_call"] == 3
    if not g13:
        failures.append("G13_images_per_call_not_3")

    # G14: max_calibration_failures_to_proceed == 2
    g14 = GAP4_CALIBRATION_THRESHOLD["max_calibration_failures_to_proceed"] == 2
    if not g14:
        failures.append("G14_max_calibration_failures_not_2")

    # G15: TS Cohen's κ threshold >= 0.6
    g15 = GAP4_CALIBRATION_THRESHOLD["ts_cohen_kappa_threshold"]["minimum_required"] == 0.6
    if not g15:
        failures.append("G15_kappa_threshold_not_0.6")

    # G16: block condition explicitly stated
    g16 = "MUST STOP" in GAP4_CALIBRATION_THRESHOLD["combined_pass_condition"] or \
          "blocks formal scoring" in GAP4_CALIBRATION_THRESHOLD["combined_pass_condition"]
    if not g16:
        failures.append("G16_block_condition_not_explicit")

    # G17: v2 output dir does not overwrite v1
    v1_dir = EVAL_DIR / "judge_v1_calibration_plan"
    g17 = OUT_DIR.resolve() != v1_dir.resolve()
    if not g17:
        failures.append("G17_v2_output_dir_would_overwrite_v1")

    # Assemble v2 plan
    plan_v2: Dict[str, Any] = {
        "plan_id": "judge_v1_calibration_plan_v2",
        "plan_version": "v2",
        "created_at_iso": now_iso(),
        "plan_is_frozen": True,
        "supersedes": "judge_v1_calibration_plan_v1",
        "v1_plan_sha256": V1_PLAN_SHA_EXPECT,
        "amendment_policy": (
            "v1 is preserved unchanged. v2 adds GAP-1 through GAP-4 fixes. "
            "Any further change requires v3+. v2 MUST NOT be overwritten."
        ),
        "upstream_anchors": {
            "v1_plan_sha256_expected": V1_PLAN_SHA_EXPECT,
            "v1_plan_sha256_observed": v1_sha_observed,
            "v1_audit_sha256_expected": V1_AUDIT_SHA_EXPECT,
            "v1_audit_sha256_observed": v1_audit_sha_observed,
            "judge_v1_py_sha256_expected": JUDGE_V1_PY_SHA_EXPECT,
            "judge_v1_py_sha256_observed": judge_v1_sha_observed,
            "v3_manifest_sha256": V3_MANIFEST_SHA_EXPECT,
            "step000_gate_report_sha256_expected": STEP000_GATE_SHA_EXPECT,
            "step000_gate_report_sha256_observed": step000_sha_observed,
        },
        "gap1_40_item_declaration": GAP1_40_ITEM_DECLARATION,
        "gap2_item_id_and_order": GAP2_ITEM_ID_AND_ORDER,
        "gap3_api_config": GAP3_API_CONFIG,
        "gap4_calibration_threshold": GAP4_CALIBRATION_THRESHOLD,
        "declarations": [
            "No Judge invoked.",
            "No TS/PS/BA/LS/IQ scores produced.",
            "No model loaded.",
            "No GPU forward executed.",
            "v1 plan and audit files are NOT modified.",
            "This plan is read-only once written.",
        ],
    }

    # Write v2 plan
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = OUT_DIR / "judge_v1_calibration_plan_v2.json"
    plan_path.write_text(
        json.dumps(plan_v2, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plan_sha = sha256_file(plan_path)

    # Write custody map (sealed separately)
    custody_path = OUT_DIR / "judge_v1_custody_map.json"
    custody_path.write_text(
        json.dumps(
            {"custody_map": CUSTODY_MAP, "custody_map_sha256": CUSTODY_MAP_SHA256,
             "created_at_iso": now_iso(),
             "note": (
                 "This file must NOT be accessible to the Judge runner during scoring. "
                 "Deblind only after all 40 items are scored and locked."
             )},
            ensure_ascii=False, indent=2, sort_keys=True,
        ),
        encoding="utf-8",
    )
    custody_sha = sha256_file(custody_path)

    # Write audit
    overall = "passed" if len(failures) == 0 else "failed"
    gate_results = {
        "G01_v1_plan_sha_unchanged": g01,
        "G02_v1_audit_sha_unchanged": g02,
        "G03_judge_v1_py_sha_unchanged": g03,
        "G04_step000_gate_sha_verified": g04,
        "G05_total_items_equals_40": g05,
        "G06_step000_consistency_check_present": g06,
        "G07_item_table_size_equals_40": g07,
        "G08_item_ids_item000_to_item039_unique": g08,
        "G09_custody_map_sha256_valid": g09,
        "G10_api_endpoint_openai_direct": g10,
        "G11_image_encoding_base64": g11,
        "G12_detail_equals_high": g12,
        "G13_images_per_call_equals_3": g13,
        "G14_max_calibration_failures_equals_2": g14,
        "G15_ts_kappa_threshold_equals_0_6": g15,
        "G16_block_condition_explicit": g16,
        "G17_v2_dir_does_not_overwrite_v1": g17,
    }
    audit: Dict[str, Any] = {
        "audit_type": "judge_v1_calibration_plan_v2_freeze_audit",
        "created_at_iso": now_iso(),
        "overall_status": overall,
        "total_failures": len(failures),
        "all_failures": failures,
        "gate_results": gate_results,
        "num_gates": len(gate_results),
        "num_gates_passed": sum(1 for v in gate_results.values() if v),
        "v2_plan_path": str(plan_path),
        "v2_plan_sha256": plan_sha,
        "v2_plan_size_bytes": plan_path.stat().st_size,
        "custody_map_path": str(custody_path),
        "custody_map_sha256": custody_sha,
        "custody_map_size_bytes": custody_path.stat().st_size,
        "v1_plan_sha256_verified_unchanged": g01,
        "declarations": [
            "No Judge invoked.",
            "No TS/PS/BA/LS/IQ scores produced.",
            "No model loaded.",
            "No GPU forward executed.",
            "v1 plan and audit files are NOT modified.",
        ],
    }
    audit_path = OUT_DIR / "judge_v1_calibration_plan_v2_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    audit_sha = sha256_file(audit_path)

    summary = {
        "overall_status": overall,
        "total_failures": len(failures),
        "all_failures": failures,
        "gate_results": gate_results,
        "v2_plan_path": str(plan_path),
        "v2_plan_size_bytes": plan_path.stat().st_size,
        "v2_plan_sha256": plan_sha,
        "audit_path": str(audit_path),
        "audit_size_bytes": audit_path.stat().st_size,
        "audit_sha256": audit_sha,
        "custody_map_path": str(custody_path),
        "custody_map_sha256": custody_sha,
        "v1_plan_sha256_unchanged": g01,
    }
    print(json.dumps(summary, indent=2))
    return 0 if overall == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
