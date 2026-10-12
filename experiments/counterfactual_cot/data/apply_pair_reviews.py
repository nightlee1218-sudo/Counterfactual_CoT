"""Strict write-back of human pair-review CSV decisions into a generated manifest.

Reads the review CSV exported by ``review_export.export_review`` after a human
reviewer filled in the five semantic scores and the final ``accepted`` verdict,
then produces a reviewed manifest plus a machine-readable summary.

Strict rules enforced by this tool:

1. Every CSV ``pair_id`` must exist in the input manifest; duplicates are fatal.
2. Every non-``generation_failed`` manifest pair must carry a decision row
   (``--allow-partial`` is only for intermediate progress checks).
3. All five scores must be present and within [0, 1].
4. ``accepted=true`` requires every threshold to be met:
   ``target_success == 1.0``, ``preservation_score >= 0.8``,
   ``binding_score >= 0.8``, ``locality_score >= 0.7``,
   ``image_quality_score >= 0.6`` (deliberate overrides need
   ``--allow-threshold-override`` and are recorded in the summary).
5. A pair whose source scene was rejected (``--source-review-csv``) can never
   be accepted.
6. ``generation_failed`` pairs can never be accepted.
7. The input manifest is never overwritten in place.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .schema import load_jsonl, write_jsonl

TRUE_VALUES = {"1", "true", "yes", "y", "accepted"}
FALSE_VALUES = {"0", "false", "no", "n", "rejected", ""}
SCORE_FIELDS = (
    "target_success",
    "preservation_score",
    "binding_score",
    "locality_score",
    "image_quality_score",
)
THRESHOLDS = {
    "preservation_score": 0.80,
    "binding_score": 0.80,
    "locality_score": 0.70,
    "image_quality_score": 0.60,
}
TARGET_SUCCESS_REQUIRED = 1.0
REQUIRED_COLUMNS = ("pair_id", *SCORE_FIELDS, "accepted", "review_notes")


class ReviewValidationError(RuntimeError):
    """The pair review CSV cannot be safely written back."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_bool(value: str, key: str, column: str) -> bool:
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    raise ReviewValidationError(f"{key}: column {column!r} must be true/false, got {value!r}")


def _parse_score(value: str, pair_id: str, column: str, *, allow_blank: bool) -> float:
    text = str(value).strip()
    if not text:
        if allow_blank:
            return 0.0
        raise ReviewValidationError(f"{pair_id}: column {column!r} is empty; every reviewed pair needs a score")
    try:
        score = float(text)
    except ValueError as exc:
        raise ReviewValidationError(f"{pair_id}: column {column!r} is not a number: {value!r}") from exc
    if not 0.0 <= score <= 1.0:
        raise ReviewValidationError(f"{pair_id}: column {column!r} must be in [0, 1], got {score}")
    return score


def _threshold_failures(scores: Dict[str, float]) -> List[str]:
    failures: List[str] = []
    if scores["target_success"] < TARGET_SUCCESS_REQUIRED:
        failures.append("target_success<1.0")
    for field, minimum in THRESHOLDS.items():
        if scores[field] < minimum:
            failures.append(f"{field}<{minimum:.2f}")
    return failures


def _load_source_decisions(path: Path) -> Dict[str, bool]:
    decisions: Dict[str, bool] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scene_id = str(row.get("scene_id", "")).strip()
            if not scene_id:
                continue
            if scene_id in decisions:
                raise ReviewValidationError(f"duplicate scene_id in source review CSV: {scene_id}")
            decisions[scene_id] = _parse_bool(str(row.get("accepted", "")), scene_id, "accepted")
    if not decisions:
        raise ReviewValidationError(f"source review CSV has no decisions: {path}")
    return decisions


def _ratio_map(records: List[Dict[str, Any]], origin: Dict[str, str], key_fn) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"accepted": 0, "total": 0})
    for record in records:
        key = str(key_fn(record))
        stats[key]["total"] += 1
        if record.get("quality", {}).get("accepted") is True:
            stats[key]["accepted"] += 1
    return dict(sorted(stats.items()))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="gold1000.generated.jsonl")
    parser.add_argument("--review-csv", required=True, type=Path, help="Filled review_full_1000/review.csv")
    parser.add_argument("--output-manifest", required=True, type=Path, help="e.g. gold1000.reviewed.jsonl")
    parser.add_argument("--source-review-csv", type=Path, default=None, help="Filled source_review_strict_v3.csv for scene-level rejection enforcement")
    parser.add_argument("--summary", type=Path, default=None, help="Summary JSON path; defaults next to output manifest")
    parser.add_argument("--allow-threshold-override", action="store_true", help="Permit accepted=true below thresholds; recorded in summary")
    parser.add_argument("--allow-partial", action="store_true", help="Permit undecided pairs for progress checks; they keep original quality")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_manifest.resolve() == args.manifest.resolve():
        raise ReviewValidationError("--output-manifest must not overwrite the input manifest")

    records = load_jsonl(args.manifest)
    by_pair: Dict[str, Dict[str, Any]] = {}
    for record in records:
        pair_id = str(record.get("pair_id", ""))
        if not pair_id:
            raise ReviewValidationError("manifest contains a record without pair_id")
        if pair_id in by_pair:
            raise ReviewValidationError(f"duplicate pair_id in manifest: {pair_id}")
        by_pair[pair_id] = record

    source_decisions = _load_source_decisions(args.source_review_csv) if args.source_review_csv else {}
    if source_decisions:
        scenes_in_manifest = {str(record.get("source_scene_id", "")) for record in records}
        uncovered = sorted(scenes_in_manifest - set(source_decisions))
        if uncovered:
            raise ReviewValidationError(
                f"source review CSV does not cover {len(uncovered)} source scenes (first 5): {uncovered[:5]}"
            )

    generation_failed = {
        pair_id for pair_id, record in by_pair.items()
        if record.get("quality", {}).get("review_status") == "generation_failed"
    }

    with args.review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing_columns:
            raise ReviewValidationError(f"review CSV is missing required columns: {missing_columns}")
        rows = list(reader)

    decisions: Dict[str, Dict[str, str]] = {}
    for row in rows:
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            continue
        if pair_id in decisions:
            raise ReviewValidationError(f"duplicate pair_id in review CSV: {pair_id}")
        decisions[pair_id] = row

    unknown = sorted(set(decisions) - set(by_pair))
    if unknown:
        raise ReviewValidationError(f"review CSV contains unknown pair_ids (first 5): {unknown[:5]}")

    if not args.allow_partial:
        missing = sorted(set(by_pair) - generation_failed - set(decisions))
        if missing:
            raise ReviewValidationError(
                f"review CSV is missing decisions for {len(missing)} pairs (first 5): {missing[:5]}; "
                "pass --allow-partial only for intermediate progress checks"
            )

    origin = {
        pair_id: str(record.get("quality", {}).get("review_status", "unknown"))
        for pair_id, record in by_pair.items()
    }

    accepted_count = rejected_count = 0
    overrides: List[Dict[str, Any]] = []
    blocked_by_source: List[str] = []
    for pair_id, row in sorted(decisions.items()):
        record = by_pair[pair_id]
        quality = record.setdefault("quality", {})
        is_generation_failed = pair_id in generation_failed
        accepted = _parse_bool(str(row.get("accepted", "")), pair_id, "accepted")
        scores = {
            field: _parse_score(str(row.get(field, "")), pair_id, field, allow_blank=is_generation_failed)
            for field in SCORE_FIELDS
        }
        source_scene_id = str(record.get("source_scene_id", ""))
        source_rejected = bool(source_decisions) and not source_decisions.get(source_scene_id, True)

        if accepted:
            if is_generation_failed:
                raise ReviewValidationError(
                    f"{pair_id}: cannot accept a generation_failed pair (no target image or mask)"
                )
            failures = _threshold_failures(scores)
            if failures:
                if not args.allow_threshold_override:
                    raise ReviewValidationError(
                        f"{pair_id}: accepted=true but thresholds unmet: {', '.join(failures)}; "
                        "lower the scores, set accepted=false, or pass --allow-threshold-override deliberately"
                    )
                overrides.append({"pair_id": pair_id, "failures": failures})
            if source_rejected:
                raise ReviewValidationError(
                    f"{pair_id}: accepted=true but its source scene {source_scene_id} was rejected; "
                    "reject this pair (or re-review the source) instead"
                )
        if source_rejected:
            blocked_by_source.append(pair_id)

        quality.update(scores)
        quality["accepted"] = False if is_generation_failed else accepted
        quality["review_notes"] = str(row.get("review_notes", "")).strip()
        quality["review_status"] = (
            "generation_failed" if is_generation_failed
            else ("review_accepted" if accepted else "review_rejected")
        )
        if not is_generation_failed:
            accepted_count += int(accepted)
            rejected_count += int(not accepted)

    pending = sorted(set(by_pair) - set(decisions))
    write_jsonl(args.output_manifest, records)

    summary = {
        "created_at": _utc_now(),
        "status": "complete" if not pending else "partial",
        "inputs": {
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "review_csv": str(args.review_csv),
            "review_csv_sha256": _sha256(args.review_csv),
            "source_review_csv": str(args.source_review_csv) if args.source_review_csv else None,
            "source_review_csv_sha256": _sha256(args.source_review_csv) if args.source_review_csv else None,
        },
        "counts": {
            "total_pairs": len(records),
            "decided": len(decisions),
            "pending": len(pending),
            "accepted": accepted_count,
            "rejected": rejected_count,
            "generation_failed": len(generation_failed),
            "pairs_with_rejected_source": len(blocked_by_source),
        },
        "thresholds": {"target_success": TARGET_SUCCESS_REQUIRED, **THRESHOLDS},
        "threshold_overrides": overrides,
        "acceptance_by_intervention_type": _ratio_map(records, origin, lambda r: r.get("intervention", {}).get("intervention_type", "unknown")),
        "acceptance_by_subtype": _ratio_map(records, origin, lambda r: r.get("subtype", "unknown")),
        "acceptance_by_auto_status": {
            auto_status: {
                "accepted": sum(1 for r in records if origin.get(r.get("pair_id", "")) == auto_status and r.get("quality", {}).get("accepted") is True),
                "total": sum(1 for r in records if origin.get(r.get("pair_id", "")) == auto_status),
            }
            for auto_status in sorted(set(origin.values()))
        },
        "output_manifest": str(args.output_manifest),
        "output_manifest_sha256": _sha256(args.output_manifest),
    }
    summary_path = args.summary or (args.output_manifest.parent / (args.output_manifest.stem + ".summary.json"))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps({
        "status": summary["status"],
        "decided": len(decisions),
        "pending": len(pending),
        "accepted": accepted_count,
        "rejected": rejected_count,
        "threshold_overrides": len(overrides),
        "output_manifest": str(args.output_manifest),
        "summary": str(summary_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
