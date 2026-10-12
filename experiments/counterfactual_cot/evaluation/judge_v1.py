"""Gold40 v3 Grounding Evaluation Judge v1 — Implementation + Input Pre-check.

This module provides:

1. JudgeV1Protocol  — frozen rubric (TS/PS/BA/LS/IQ, Grounded Success threshold,
                      null/not_scored rules, candidate_only mask prohibition).
2. InputPrecheck    — read-only audit of all 160 generated PNGs across the four
                      conditions, resolving source/target image paths from the
                      Gold40 v3 manifest and verifying SHA256 against the
                      gold40_v3_asset_and_manifest_provenance.json index.

This round:
  - Implements the Judge v1 rubric and pre-check.
  - Does NOT call any VLM/LLM Judge.
  - Does NOT produce any TS/PS/BA/LS/IQ scores.
  - Does NOT modify any frozen asset.

Entry point:
  python -m experiments.counterfactual_cot.evaluation.judge_v1 --mode precheck

Outputs:
  judge_v1_precheck_report.json   (in PRECHECK_OUT_DIR)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/group/40016/niightli/BAGEL").resolve()
EVAL_DIR = REPO_ROOT / "experiments/counterfactual_cot/evaluation"
INFER_ROOT = REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_ab_aligned_inference_nt25_protocol_v1"
AUDIT_DIR = REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_ab_400steps_final_engineering_audit"
PRECHECK_OUT_DIR = EVAL_DIR / "judge_v1_precheck"

V3_MANIFEST = (
    REPO_ROOT / "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/manifests/"
    "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl"
)
V3_MANIFEST_SHA_EXPECT = "5960e1eebb4d7a394a7b7cba2b2c17c686d184756b637110ef8b43397e172061"

PROVENANCE_JSON = (
    REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3/"
    "gold40_v3_asset_and_manifest_provenance.json"
)

# Four conditions: (label, verifier_json_path)
CONDITIONS: List[Tuple[str, Path]] = [
    ("A_step000", INFER_ROOT / "verifier_A/verifier_A.json"),
    ("B_step000", INFER_ROOT / "verifier_B/verifier_B.json"),
    ("A_step400", INFER_ROOT / "verifier_A_step400/verifier_A.json"),
    ("B_step400", INFER_ROOT / "verifier_B_step400/verifier_B.json"),
]

# ---------------------------------------------------------------------------
# Frozen rubric (Protocol v1)
# ---------------------------------------------------------------------------

JUDGE_V1_RUBRIC: Dict[str, Any] = {
    "protocol_id": "gold40_v3_grounding_evaluation_protocol_v1",
    "judge_version": "v1",
    "metrics": {
        "TS": {
            "name": "Target-atom Success",
            "definition": (
                "For the pair's structured intervention (entity_id, field, "
                "old_value → new_value), does the generated image reflect "
                "new_value on entity_id at field?"
            ),
            "score_type": "binary",
            "allowed_values": [0, 1],
            "not_scored_marker": "null_or_score_status_not_scored",
        },
        "PS": {
            "name": "Preservation Score",
            "definition": (
                "Fraction of the pair's preservation.must_preserve atoms "
                "that remain unchanged in the generated image relative to "
                "the source image."
            ),
            "score_type": "ratio_0_to_1",
            "denominator": "len(preservation.must_preserve)",
            "not_scored_marker": "null_or_score_status_not_scored",
        },
        "BA": {
            "name": "Binding Accuracy",
            "definition": (
                "Fraction of the observed change that is bound to the "
                "correct entity_id (not to another instance of the same "
                "category)."
            ),
            "score_type": "ratio_0_to_1",
            "not_scored_marker": "null_or_score_status_not_scored",
        },
        "LS": {
            "name": "Locality Score",
            "definition": (
                "Extent to which the observed change is spatially confined "
                "to the region required by the intervention."
            ),
            "score_type": "ratio_0_to_1",
            "evidence_sources_allowed": [
                "source_image",
                "target_image",
                "generated_image",
                "structured_intervention_requirements",
            ],
            "evidence_sources_forbidden": [
                "candidate_only_difference_masks_as_gold_supervision",
                "candidate_only_difference_masks_as_ls_ground_truth",
            ],
            "candidate_only_mask_prohibition": (
                "candidate_only difference masks (auto-generated pixel diffs "
                "between source and target) MUST NOT be used as Gold "
                "supervision or as LS ground truth. LS may only be scored "
                "from source/target/generated images plus the structured "
                "intervention requirements."
            ),
            "not_scored_marker": "null_or_score_status_not_scored",
        },
        "IQ": {
            "name": "Image Quality",
            "definition": (
                "Image is clear, complete, and free of severe artifacts "
                "(no severe distortions, tears, checkerboard patterns, "
                "extreme blur, or catastrophic composition failures)."
            ),
            "score_type": "ratio_0_to_1",
            "not_scored_marker": "null_or_score_status_not_scored",
        },
    },
    "grounded_success": {
        "condition": "TS == 1 AND PS >= 0.8 AND BA >= 0.8 AND LS >= 0.7 AND IQ >= 0.6",
        "thresholds": {"TS": 1, "PS": 0.8, "BA": 0.8, "LS": 0.7, "IQ": 0.6},
        "note": (
            "All five predicates must hold. Any missing score → pair is "
            "'not_scored' for grounded_success; NOT counted as success or "
            "failure until fully scored under a reviewed judge pass."
        ),
    },
    "null_not_scored_rules": [
        "Unrated fields MUST be null or score_status='not_scored'.",
        "Do NOT fill 0 as a placeholder for an unscored field.",
        "A pair with any null/not_scored metric is 'not_scored' for "
        "grounded_success, not a failure.",
    ],
    "category_reporting_rules": [
        "attribute_replace (10 pairs) — report separately",
        "count_change (10 pairs) — report separately",
        "object_replace (10 pairs) — report separately",
        "spatial_position_change (8 pairs) — report separately",
        "spatial_relation_change (2 pairs) — report separately",
        "NEVER merge spatial_position_change (8) and spatial_relation_change (2) "
        "into a single '10 spatial' number.",
    ],
    "judge_execution_rules": [
        "Judge generator and data provider must be decoupled.",
        "Judge must be condition-blinded (A/B/step_number hidden).",
        "Judge output must include: original rationale text, atom-level "
        "decisions, structured TS/PS/BA/LS/IQ scores.",
        "Pre-execution freeze required: prompt template, model+version, "
        "decoding params, thresholds, human-review arbitration plan.",
        "No scoring before all freeze requirements are met.",
    ],
    "evaluation_scope": "Gold40 training-set overfit/capacity diagnostic",
    "training_set_boundary": (
        "Gold40 v3 is the TRAINING SET. Results can ONLY be interpreted as "
        "overfit/capacity diagnostic. NOT generalization performance."
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


def _load_manifest() -> List[Dict[str, Any]]:
    records = []
    with open(V3_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_provenance_index(prov_path: Path) -> Dict[str, Dict[str, Any]]:
    """Build {pair_id -> {source: {resolved_path, sha256}, target: {...}, mask: {...}}}."""
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    index: Dict[str, Dict[str, Any]] = {}
    for entry in prov.get("assets", []):
        pid = entry.get("pair_id")
        kind = entry.get("kind")  # source / target / mask
        if pid and kind in ("source", "target", "mask"):
            if pid not in index:
                index[pid] = {}
            index[pid][kind] = {
                "resolved_path": entry.get("resolved_path"),
                "sha256": entry.get("sha256"),
                "size_bytes": entry.get("size_bytes"),
                "exists_in_provenance": entry.get("exists", False),
            }
    return index


def _resolve_image_path(pair: Dict[str, Any], side: str) -> Optional[Path]:
    """Resolve source or target image absolute path from manifest record."""
    root = pair.get("final_image_root")
    img_rel = (pair.get(side) or {}).get("image")
    if root and img_rel:
        return Path(root) / img_rel
    return None


# ---------------------------------------------------------------------------
# Input pre-check
# ---------------------------------------------------------------------------

def run_precheck() -> Dict[str, Any]:
    """
    Read-only pre-check of all 160 generated PNGs across 4 conditions.
    Resolves source/target paths from manifest + provenance index.
    Does NOT call any Judge. Does NOT produce scores.
    """
    failures: List[str] = []
    warnings: List[str] = []

    # 1. Verify manifest SHA
    manifest_sha_observed = sha256_file(V3_MANIFEST)
    manifest_sha_ok = manifest_sha_observed == V3_MANIFEST_SHA_EXPECT
    if not manifest_sha_ok:
        failures.append(
            f"manifest_sha_mismatch: expected={V3_MANIFEST_SHA_EXPECT} "
            f"observed={manifest_sha_observed}"
        )

    # 2. Load manifest
    records = _load_manifest()
    if len(records) != 40:
        failures.append(f"manifest_pair_count={len(records)} (expected 40)")

    # 3. Load provenance index
    prov_index = _build_provenance_index(PROVENANCE_JSON)

    # 4. Build per-pair manifest index
    manifest_index: Dict[str, Dict[str, Any]] = {r["pair_id"]: r for r in records}

    # 5. Per-pair source/target resolution
    pair_asset_checks: List[Dict[str, Any]] = []
    for pair in records:
        pid = pair["pair_id"]
        iv = pair.get("intervention") or {}
        pres = pair.get("preservation") or {}
        must_preserve = pres.get("must_preserve") or []

        # Resolve source path from manifest
        src_path = _resolve_image_path(pair, "source")
        tgt_path = _resolve_image_path(pair, "target")

        # Get provenance SHA
        prov_src = (prov_index.get(pid) or {}).get("source") or {}
        prov_tgt = (prov_index.get(pid) or {}).get("target") or {}
        prov_mask = (prov_index.get(pid) or {}).get("mask") or {}

        src_exists = src_path is not None and src_path.exists()
        tgt_exists = tgt_path is not None and tgt_path.exists()

        # Verify SHA if file exists
        src_sha_observed = sha256_file(src_path) if src_exists else None
        tgt_sha_observed = sha256_file(tgt_path) if tgt_exists else None
        src_sha_expected = prov_src.get("sha256")
        tgt_sha_expected = prov_tgt.get("sha256")

        src_sha_ok = (src_sha_observed == src_sha_expected) if (src_sha_observed and src_sha_expected) else None
        tgt_sha_ok = (tgt_sha_observed == tgt_sha_expected) if (tgt_sha_observed and tgt_sha_expected) else None

        pair_failures = []
        if not src_exists:
            pair_failures.append(f"source_image_missing: {src_path}")
        elif src_sha_ok is False:
            pair_failures.append(f"source_sha_mismatch: expected={src_sha_expected} observed={src_sha_observed}")
        if not tgt_exists:
            pair_failures.append(f"target_image_missing: {tgt_path}")
        elif tgt_sha_ok is False:
            pair_failures.append(f"target_sha_mismatch: expected={tgt_sha_expected} observed={tgt_sha_observed}")

        # Mask: record but do NOT use as LS ground truth (prohibition)
        mask_path_str = prov_mask.get("resolved_path")
        mask_exists = Path(mask_path_str).exists() if mask_path_str else False

        pair_asset_checks.append({
            "pair_id": pid,
            "intervention_type": iv.get("intervention_type"),
            "intervention_field": iv.get("field"),
            "entity_id": iv.get("entity_id"),
            "old_value": iv.get("old_value"),
            "new_value": iv.get("new_value"),
            "num_must_preserve_atoms": len(must_preserve),
            "must_preserve_atoms": must_preserve,
            "source_image_path": str(src_path) if src_path else None,
            "source_image_exists": src_exists,
            "source_image_sha256_expected": src_sha_expected,
            "source_image_sha256_observed": src_sha_observed,
            "source_image_sha256_ok": src_sha_ok,
            "target_image_path": str(tgt_path) if tgt_path else None,
            "target_image_exists": tgt_exists,
            "target_image_sha256_expected": tgt_sha_expected,
            "target_image_sha256_observed": tgt_sha_observed,
            "target_image_sha256_ok": tgt_sha_ok,
            "mask_path": mask_path_str,
            "mask_exists": mask_exists,
            "mask_usage_note": (
                "candidate_only mask MUST NOT be used as Gold supervision "
                "or LS ground truth per Protocol v1 prohibition."
            ),
            "pair_failures": pair_failures,
            "pair_passed": len(pair_failures) == 0,
        })
        failures.extend([f"pair {pid}: {pf}" for pf in pair_failures])

    # 6. Per-condition PNG check (all 160 PNGs)
    condition_checks: Dict[str, Any] = {}
    for label, json_path in CONDITIONS:
        cond_failures = []
        if not json_path.exists():
            cond_failures.append(f"verifier_json_missing: {json_path}")
            condition_checks[label] = {
                "passed": False, "failures": cond_failures,
                "num_pngs_checked": 0, "num_pngs_ok": 0,
            }
            failures.extend(cond_failures)
            continue

        verifier = json.loads(json_path.read_text(encoding="utf-8"))
        per_pair = verifier.get("per_pair") or []
        png_checks = []
        for pp in per_pair:
            pid = pp.get("pair_id")
            png_path_str = pp.get("output_path")
            png_size = pp.get("output_png_size_bytes", 0)
            png_sha = pp.get("output_png_sha256")
            png_path = Path(png_path_str) if png_path_str else None
            png_exists = png_path is not None and png_path.exists()
            # Re-verify PNG SHA from disk
            png_sha_disk = sha256_file(png_path) if png_exists else None
            png_sha_ok = (png_sha_disk == png_sha) if (png_sha_disk and png_sha) else None

            pf = []
            if not png_exists:
                pf.append("png_missing")
            elif png_size == 0:
                pf.append("png_zero_bytes")
            elif png_sha_ok is False:
                pf.append(f"png_sha_mismatch_disk={png_sha_disk}_recorded={png_sha}")

            png_checks.append({
                "pair_id": pid,
                "output_path": png_path_str,
                "output_png_size_bytes": png_size,
                "output_png_sha256_recorded": png_sha,
                "output_png_sha256_disk": png_sha_disk,
                "output_png_sha256_verified": png_sha_ok,
                "png_exists": png_exists,
                "pair_failures": pf,
                "passed": len(pf) == 0,
            })
            if pf:
                cond_failures.append(f"{label} pair {pid}: {pf}")

        num_ok = sum(1 for c in png_checks if c["passed"])
        condition_checks[label] = {
            "verifier_json_path": str(json_path),
            "num_pngs_checked": len(png_checks),
            "num_pngs_ok": num_ok,
            "num_pngs_failed": len(png_checks) - num_ok,
            "passed": len(cond_failures) == 0,
            "failures": cond_failures,
            "per_png": png_checks,
        }
        failures.extend(cond_failures)

    # 7. Evaluation-adaptation warnings (from Protocol v1)
    eval_adaptation_warnings = [
        {
            "kind": "region_label_non_whitelist",
            "pair_id": "cf_source_grocery_display_absolute_position_change_v4_absolute_region_pumpkin_in_basket_fix_v5_0000",
            "note": "pumpkin uses new_value='center' (non-whitelist region label). Judge must accept 'center' as valid per v3 schema normalization.",
        },
        {
            "kind": "schema_normalization",
            "pair_id": "cf_source_flower_shop_absolute_position_change_v4_absolute_region_wide_table_fix_v6_0000",
            "note": "flower_shop had support_context removed in v3 schema normalization. Pair content intact.",
        },
        {
            "kind": "spatial_relation_change_separate_category",
            "pair_ids": [
                "cf_source_lakeside_birds_spatial_relation_change_0000",
                "cf_source_home_kitchen_spatial_relation_change_0000",
            ],
            "note": "Must be reported as separate category from spatial_position_change (8 pairs). Do NOT merge.",
        },
    ]

    # 8. Summary
    num_pairs_asset_ok = sum(1 for p in pair_asset_checks if p["pair_passed"])
    overall = "passed" if len(failures) == 0 else "failed"

    report = {
        "report_type": "judge_v1_input_precheck",
        "created_at_iso": now_iso(),
        "overall_status": overall,
        "total_failures": len(failures),
        "all_failures": failures,
        "warnings": warnings,
        "manifest_sha256_expected": V3_MANIFEST_SHA_EXPECT,
        "manifest_sha256_observed": manifest_sha_observed,
        "manifest_sha256_ok": manifest_sha_ok,
        "num_pairs_in_manifest": len(records),
        "num_pairs_asset_ok": num_pairs_asset_ok,
        "num_pairs_asset_failed": len(pair_asset_checks) - num_pairs_asset_ok,
        "conditions_checked": [label for label, _ in CONDITIONS],
        "total_pngs_expected": 40 * len(CONDITIONS),
        "total_pngs_ok": sum(
            c.get("num_pngs_ok", 0) for c in condition_checks.values()
        ),
        "judge_v1_rubric": JUDGE_V1_RUBRIC,
        "eval_adaptation_warnings": eval_adaptation_warnings,
        "pair_asset_checks": pair_asset_checks,
        "condition_png_checks": condition_checks,
        "declarations": [
            "No Judge invoked.",
            "No TS/PS/BA/LS/IQ scores produced.",
            "No model loaded.",
            "No GPU forward executed.",
            "No frozen asset modified.",
            "candidate_only masks recorded but NOT used as Gold or LS ground truth.",
            "This is a pre-check only; formal scoring requires separate authorization.",
        ],
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["precheck"], required=True)
    args = parser.parse_args()

    if args.mode == "precheck":
        print("[judge_v1] running input pre-check ...", flush=True)
        report = run_precheck()
        PRECHECK_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PRECHECK_OUT_DIR / "judge_v1_precheck_report.json"
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report_sha = sha256_file(out_path)
        summary = {
            "overall_status": report["overall_status"],
            "total_failures": report["total_failures"],
            "all_failures": report["all_failures"],
            "num_pairs_asset_ok": report["num_pairs_asset_ok"],
            "total_pngs_ok": report["total_pngs_ok"],
            "report_path": str(out_path),
            "report_size_bytes": out_path.stat().st_size,
            "report_sha256": report_sha,
        }
        print(json.dumps(summary, indent=2))
        return 0 if report["overall_status"] == "passed" else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
