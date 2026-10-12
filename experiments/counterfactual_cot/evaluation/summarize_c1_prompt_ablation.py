#!/usr/bin/env python3
"""Validate, unblind, and summarize a completed C1 prompt-ablation review."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

DEFAULT_ROOT = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/"
    "c1_prompt_ablation_v1"
)
BOOLEAN_FIELDS: Tuple[str, ...] = (
    "right_ball_present",
    "left_ball_present",
    "ball_color_preserved",
    "sofa_position_preserved",
    "background_preserved",
    "viewpoint_preserved",
    "lighting_preserved",
    "image_quality_valid",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_ROOT / "blind_manual_review.json")
    parser.add_argument("--key-file", type=Path, default=DEFAULT_ROOT / "blind_review_key.json")
    parser.add_argument("--generation-report", type=Path, default=DEFAULT_ROOT / "prompt_ablation_report.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "prompt_ablation_summary.json")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite summary: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_scores(item: Mapping[str, Any], item_id: str) -> List[str]:
    failures: List[str] = []
    for field in BOOLEAN_FIELDS:
        if type(item.get(field)) is not bool:
            failures.append(f"{item_id}.{field} must be bool, not {item.get(field)!r}")
    if type(item.get("ball_count")) is not int or item.get("ball_count", -1) < 0:
        failures.append(f"{item_id}.ball_count must be a non-negative int")
    return failures


def _score(item: Mapping[str, Any]) -> Dict[str, bool]:
    edit_success = bool(
        item["right_ball_present"]
        and not item["left_ball_present"]
        and item["ball_count"] == 1
    )
    preserve_success = bool(
        item["ball_color_preserved"]
        and item["sofa_position_preserved"]
        and item["background_preserved"]
        and item["viewpoint_preserved"]
        and item["lighting_preserved"]
    )
    return {
        "edit_success": edit_success,
        "preserve_success": preserve_success,
        "full_success": edit_success and preserve_success and bool(item["image_quality_valid"]),
    }


def _aggregate(items: List[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(items)
    scored = [_score(item) for item in items]
    metrics: Dict[str, Any] = {"count": count}
    for key in ("edit_success", "preserve_success", "full_success"):
        successes = sum(int(row[key]) for row in scored)
        metrics[f"{key}_count"] = successes
        metrics[f"{key}_rate"] = successes / count if count else None
    for key in ("right_ball_present", "left_ball_present", "image_quality_valid"):
        successes = sum(int(bool(item[key])) for item in items)
        metrics[f"{key}_count"] = successes
        metrics[f"{key}_rate"] = successes / count if count else None
    metrics["exactly_one_ball_count"] = sum(int(item["ball_count"] == 1) for item in items)
    metrics["exactly_one_ball_rate"] = metrics["exactly_one_ball_count"] / count if count else None
    return metrics


def _contrast(
    groups: Mapping[Tuple[str, str], List[Mapping[str, Any]]],
    checkpoint_id: str,
    profile_a: str,
    profile_b: str,
) -> Dict[str, Any]:
    a = _aggregate(groups.get((checkpoint_id, profile_a), []))
    b = _aggregate(groups.get((checkpoint_id, profile_b), []))
    return {
        "checkpoint_id": checkpoint_id,
        "profile_a": profile_a,
        "profile_b": profile_b,
        "a": a,
        "b": b,
        "edit_success_rate_delta_a_minus_b": (
            a["edit_success_rate"] - b["edit_success_rate"]
            if a["edit_success_rate"] is not None and b["edit_success_rate"] is not None
            else None
        ),
    }


def _expected_generation_items(report: Mapping[str, Any]) -> Dict[Tuple[str, str, int], Dict[str, str]]:
    expected: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    for job in report.get("jobs", []):
        checkpoint_id = job.get("checkpoint_id")
        profile_id = job.get("profile_id")
        for output in job.get("worker_result", {}).get("outputs", []):
            identity = (checkpoint_id, profile_id, output.get("seed"))
            expected[identity] = {
                "image_path": str(Path(output["path"]).resolve()),
                "image_png_sha256": output["png_sha256"],
            }
    return expected


def main() -> int:
    args = _parse_args()
    paths = [args.review_file.resolve(), args.key_file.resolve(), args.generation_report.resolve()]
    output = args.output.resolve()
    if output in paths:
        raise RuntimeError("Summary output must not overwrite an input artifact")
    review, key, report = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    failures: List[str] = []
    if report.get("status") != "passed" or report.get("kind") != "c1_prompt_ablation_report":
        failures.append("generation report is not a passed c1_prompt_ablation_report")
    if review.get("kind") != "c1_prompt_ablation_blind_manual_review":
        failures.append("review kind mismatch")
    if key.get("kind") != "c1_prompt_ablation_blind_key":
        failures.append("blind key kind mismatch")
    for field in ("pair_id", "run_id", "generation_report_sha256", "expected_item_count"):
        if review.get(field) != key.get(field):
            failures.append(f"review/key {field} mismatch")
    if review.get("pair_id") != report.get("pair_id") or review.get("run_id") != report.get("run_id"):
        failures.append("review identity does not match generation report")
    if review.get("generation_report_sha256") != _sha256(paths[2]):
        failures.append("generation report SHA256 binding mismatch")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        failures.append("reviewer must be filled")
    if not isinstance(review.get("reviewed_at_utc"), str) or not review["reviewed_at_utc"].strip():
        failures.append("reviewed_at_utc must be filled")

    review_items = review.get("items") if isinstance(review.get("items"), list) else []
    key_items = key.get("items") if isinstance(key.get("items"), list) else []
    review_by_id = {item.get("item_id"): item for item in review_items if isinstance(item, dict)}
    key_by_id = {item.get("item_id"): item for item in key_items if isinstance(item, dict)}
    if len(review_by_id) != len(review_items) or len(key_by_id) != len(key_items):
        failures.append("duplicate or invalid item_id")
    expected_count = review.get("expected_item_count")
    if len(review_items) != expected_count or len(key_items) != expected_count:
        failures.append("review/key item count is incomplete")
    if set(review_by_id) != set(key_by_id):
        failures.append("review/key item_id sets differ")

    expected_generation = _expected_generation_items(report)
    observed_generation: Dict[Tuple[str, str, int], Dict[str, str]] = {}
    merged_items: List[Dict[str, Any]] = []
    for item_id, review_item in review_by_id.items():
        key_item = key_by_id.get(item_id, {})
        failures.extend(_validate_scores(review_item, str(item_id)))
        for field in ("image_path", "image_png_sha256"):
            if review_item.get(field) != key_item.get(field):
                failures.append(f"{item_id}.{field} differs from blind key")
        image_path = Path(str(review_item.get("image_path", ""))).resolve()
        if not image_path.is_file() or _sha256(image_path) != review_item.get("image_png_sha256"):
            failures.append(f"{item_id} image missing or SHA256 mismatch")
        identity = (key_item.get("checkpoint_id"), key_item.get("profile_id"), key_item.get("seed"))
        observed_generation[identity] = {
            "image_path": str(image_path),
            "image_png_sha256": str(review_item.get("image_png_sha256")),
        }
        merged_items.append({**review_item, **key_item})
    if observed_generation != expected_generation:
        failures.append("review identities/paths/hashes do not exactly cover generation report")
    if failures:
        print(json.dumps({"status": "incomplete", "failures": failures}, ensure_ascii=False, indent=2))
        return 2

    groups: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for item in merged_items:
        groups[(item["checkpoint_id"], item["profile_id"])].append(item)
    group_metrics = {
        f"{checkpoint_id}/{profile_id}": _aggregate(group_items)
        for (checkpoint_id, profile_id), group_items in sorted(groups.items())
    }
    checkpoint_ids = sorted({item["checkpoint_id"] for item in merged_items})
    profiles = {item["profile_id"] for item in merged_items}
    contrast_pairs = [
        ("cot_core_v1", "legacy_paired_v1"),
        ("full_paired_v2", "full_without_target_cot_v1"),
        ("target_prompt_only_v1", "cot_core_v1"),
        ("delta_only_v2", "legacy_paired_v1"),
    ]
    if {
        "conflict_prompt_right_cot_left_v1",
        "conflict_prompt_left_cot_right_v1",
    }.issubset(profiles):
        contrast_pairs.append(
            ("conflict_prompt_left_cot_right_v1", "conflict_prompt_right_cot_left_v1")
        )
    contrasts = [
        _contrast(groups, checkpoint_id, profile_a, profile_b)
        for checkpoint_id in checkpoint_ids
        for profile_a, profile_b in contrast_pairs
        if profile_a in profiles and profile_b in profiles
    ]
    summary = {
        "schema_version": 1,
        "kind": "c1_prompt_ablation_summary",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "pair_id": report["pair_id"],
        "run_id": report["run_id"],
        "artifact_sha256": {
            "review": _sha256(paths[0]),
            "blind_key": _sha256(paths[1]),
            "generation_report": _sha256(paths[2]),
        },
        "reviewer": review["reviewer"],
        "reviewed_at_utc": review["reviewed_at_utc"],
        "item_count": len(merged_items),
        "success_rule": "right_ball_present AND NOT left_ball_present AND ball_count == 1",
        "group_metrics": group_metrics,
        "planned_contrasts": contrasts,
        "interpretation_guardrails": [
            "full_paired_v2 success alone demonstrates capability, not Target-CoT causality",
            "prompt-only results describe responsiveness of legacy-trained checkpoints, not retraining outcomes",
            "Target-CoT causality requires the conflict suite, especially prompt-left/cot-right",
            "report status passed means review completeness, not semantic task success",
        ],
    }
    _write_new_json(output, summary)
    print(json.dumps({"status": "passed", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
