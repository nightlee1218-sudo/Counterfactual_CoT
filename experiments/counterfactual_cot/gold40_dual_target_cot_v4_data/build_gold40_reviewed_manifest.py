#!/usr/bin/env python3
"""Build a training-ready Gold40 paired manifest from explicit human decisions.

This gate never creates review decisions. It accepts a candidate manifest only when all
40 candidate pairs have explicit approved human decisions and mutually disjoint
source-only, target-only, and optional shared changed-region masks whose union covers
the candidate difference mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = SCRIPT_DIR / "artifacts"
DEFAULT_CANDIDATE_MANIFEST = DEFAULT_ARTIFACTS / "gold40_source_target_paired_candidate_v4.jsonl"
DEFAULT_MASK_AUDIT = DEFAULT_ARTIFACTS / "gold40_dynamic_mask_audit_v4.json"
DEFAULT_REVIEW_DECISIONS = DEFAULT_ARTIFACTS / "review_decisions.jsonl"
DEFAULT_OUTPUT_MANIFEST = DEFAULT_ARTIFACTS / "gold40_source_target_paired_reviewed_v4.jsonl"
DEFAULT_OUTPUT_AUDIT = DEFAULT_ARTIFACTS / "gold40_human_review_audit_v4.json"
EXPECTED_PAIR_COUNT = 40


class ValidationError(RuntimeError):
    """A review artifact is malformed, incomplete, or not training-ready."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_binary_mask(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    Image.fromarray((pixels.astype(np.uint8) * 255), mode="L").save(temporary)
    os.replace(temporary, path)


def _read_jsonl(path: Path, label: str) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise ValidationError(f"{label} does not exist or is not a file: {path}")
    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"{label} line {line_number} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValidationError(f"{label} line {line_number} must be a JSON object")
        records.append(record)
    if not records:
        raise ValidationError(f"{label} is empty: {path}")
    return records


def _require_text(record: Mapping[str, Any], field: str, pair_id: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{pair_id}: required non-empty string field {field!r} is missing")
    return value.strip()


def _parse_reviewed_at(value: str, pair_id: str) -> str:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValidationError(f"{pair_id}: reviewed_at must be ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValidationError(f"{pair_id}: reviewed_at must include an explicit timezone")
    return value


def _index_by_pair_id(records: Iterable[Dict[str, Any]], label: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        pair_id = record.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValidationError(f"{label} has a record without a non-empty pair_id")
        if pair_id in indexed:
            raise ValidationError(f"{label} contains duplicate pair_id: {pair_id}")
        indexed[pair_id] = record
    return indexed


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base_dir / path


def _optional_roi_path(decision: Mapping[str, Any], field: str, decisions_dir: Path, pair_id: str) -> Path | None:
    value = decision.get(field)
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return _resolve_path(value.strip(), decisions_dir)
    if isinstance(value, dict):
        for key in ("path", "mask_path"):
            nested = value.get(key)
            if isinstance(nested, str) and nested.strip():
                return _resolve_path(nested.strip(), decisions_dir)
    raise ValidationError(
        f"{pair_id}: {field} must be omitted or a non-empty mask path/object containing path/mask_path"
    )


def _load_binary_mask(path: Path, label: str) -> np.ndarray:
    if not path.is_file():
        raise ValidationError(f"{label} does not exist or is not a file: {path}")
    try:
        with Image.open(path) as image:
            values = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Unable to decode {label}: {path}: {exc}") from exc
    if values.ndim != 2 or not values.size:
        raise ValidationError(f"{label} is not a non-empty 2D image: {path}")
    return values


def _validate_candidate_record(record: Mapping[str, Any], pair_id: str) -> Mapping[str, Any]:
    if record.get("macro_pair_id") != f"dual_cot::{pair_id}":
        raise ValidationError(f"{pair_id}: macro_pair_id does not match the paired-record convention")
    branches = record.get("paired_branches")
    if not isinstance(branches, list) or len(branches) != 2:
        raise ValidationError(f"{pair_id}: paired_branches must contain exactly source_state and target_state")
    branch_ids = {branch.get("branch_id") for branch in branches if isinstance(branch, dict)}
    if branch_ids != {"source_state", "target_state"}:
        raise ValidationError(f"{pair_id}: paired_branches must contain source_state and target_state exactly once")
    mask = record.get("dynamic_candidate_mask")
    if not isinstance(mask, dict):
        raise ValidationError(f"{pair_id}: dynamic_candidate_mask is missing")
    if mask.get("candidate_only") is not True or mask.get("manual_review_required") is not True:
        raise ValidationError(f"{pair_id}: manifest record is not marked as a human-review candidate")
    if mask.get("review_status") != "unreviewed":
        raise ValidationError(f"{pair_id}: candidate manifest review_status must be 'unreviewed'")
    if not isinstance(mask.get("path"), str) or not mask["path"].strip():
        raise ValidationError(f"{pair_id}: candidate mask path is missing")
    if not isinstance(mask.get("sha256"), str) or len(mask["sha256"]) != 64:
        raise ValidationError(f"{pair_id}: candidate mask sha256 is missing or malformed")
    return mask


def _validate_mask_audit(
    audit: Mapping[str, Any], candidate_by_id: Mapping[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    if audit.get("status") != "candidate_ready_manual_review_required":
        raise ValidationError("mask audit status must be candidate_ready_manual_review_required")
    pair_audits = audit.get("pairs")
    if not isinstance(pair_audits, list):
        raise ValidationError("mask audit pairs must be a list")
    audit_by_id = _index_by_pair_id(pair_audits, "mask audit")
    if set(audit_by_id) != set(candidate_by_id):
        missing = sorted(set(candidate_by_id) - set(audit_by_id))
        extra = sorted(set(audit_by_id) - set(candidate_by_id))
        raise ValidationError(f"mask audit pair IDs differ from candidate manifest; missing={missing}, extra={extra}")
    for pair_id, candidate in candidate_by_id.items():
        audit_record = audit_by_id[pair_id]
        if audit_record.get("status") != "candidate_ready_manual_review_required":
            raise ValidationError(f"{pair_id}: mask audit is not candidate_ready_manual_review_required")
        audit_mask = audit_record.get("mask")
        candidate_mask = candidate["dynamic_candidate_mask"]
        if not isinstance(audit_mask, dict):
            raise ValidationError(f"{pair_id}: mask audit has no mask object")
        for field in ("path", "sha256"):
            if audit_mask.get(field) != candidate_mask.get(field):
                raise ValidationError(f"{pair_id}: audit mask {field} does not match candidate manifest")
    return audit_by_id


def _validate_decision(
    decision: Mapping[str, Any], pair_id: str, decisions_dir: Path
) -> Tuple[Dict[str, Any], Dict[str, Path | None]]:
    if decision.get("approved") is not True:
        raise ValidationError(f"{pair_id}: approved must be the JSON boolean true")
    reviewer = _require_text(decision, "reviewer", pair_id)
    reviewed_at = _parse_reviewed_at(_require_text(decision, "reviewed_at", pair_id), pair_id)
    notes = _require_text(decision, "notes", pair_id)
    roi_paths = {
        "source_changed_roi": _optional_roi_path(decision, "source_changed_roi", decisions_dir, pair_id),
        "target_changed_roi": _optional_roi_path(decision, "target_changed_roi", decisions_dir, pair_id),
        "shared_changed_roi": _optional_roi_path(decision, "shared_changed_roi", decisions_dir, pair_id),
    }
    if not any(roi_paths.values()):
        raise ValidationError(f"{pair_id}: provide at least one changed-region mask")
    sanitized = {
        "pair_id": pair_id,
        "approved": True,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "notes": notes,
        **{field: str(path) for field, path in roi_paths.items() if path is not None},
    }
    return sanitized, roi_paths


def _reviewed_record(
    candidate: Mapping[str, Any], decision: Mapping[str, Any], roi_audit: Mapping[str, Any]
) -> Dict[str, Any]:
    reviewed = json.loads(json.dumps(candidate))
    source_scene_id = candidate.get("source_scene_id") or candidate.get("scene_id") or candidate.get("template_family")
    if not isinstance(source_scene_id, str) or not source_scene_id:
        raise ValidationError(f"{candidate.get('pair_id')}: cannot derive source_scene_id")
    intervention = candidate.get("intervention")
    if not isinstance(intervention, Mapping) or not isinstance(intervention.get("intervention_type"), str):
        raise ValidationError(f"{candidate.get('pair_id')}: intervention.intervention_type is missing")
    reviewed["source_scene_id"] = source_scene_id
    reviewed["intervention_type"] = intervention["intervention_type"]
    reviewed["dynamic_candidate_mask"]["review_status"] = "approved"
    reviewed["review"] = {
        "approved": True,
        "status": "approved",
        "reviewer": decision["reviewer"],
        "reviewed_at": decision["reviewed_at"],
        "notes": decision["notes"],
    }
    reviewed["human_review"] = {
        **reviewed["review"],
        "source_changed_roi": roi_audit.get("source_changed_roi"),
        "target_changed_roi": roi_audit.get("target_changed_roi"),
        "shared_changed_roi": roi_audit.get("shared_changed_roi"),
        "approved_allowed_change_mask": roi_audit["approved_allowed_change_mask"],
        "candidate_mask_coverage": roi_audit["candidate_mask_coverage"],
    }
    reviewed["reviewed_mask"] = {
        **roi_audit["approved_allowed_change_mask"],
        "status": "approved",
        "kind": "human_reviewed_source_target_union",
    }
    reviewed["dynamic_regions"] = {
        "source_changed_roi": roi_audit.get("source_changed_roi"),
        "target_changed_roi": roi_audit.get("target_changed_roi"),
        "shared_changed_roi": roi_audit.get("shared_changed_roi"),
        "source_changed_roi_ltrb": (roi_audit.get("source_changed_roi") or {}).get("bbox_ltrb"),
        "target_changed_roi_ltrb": (roi_audit.get("target_changed_roi") or {}).get("bbox_ltrb"),
        "shared_changed_roi_ltrb": (roi_audit.get("shared_changed_roi") or {}).get("bbox_ltrb"),
        "kind": "human_reviewed_source_only_target_only_shared_binary_masks",
    }
    reviewed["training_readiness"] = {
        "ready_for_training": True,
        "reason": "All required human review fields and changed-region coverage checks passed.",
    }
    return reviewed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--mask-audit", type=Path, default=DEFAULT_MASK_AUDIT)
    parser.add_argument("--review-decisions", type=Path, default=DEFAULT_REVIEW_DECISIONS)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_OUTPUT_AUDIT)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing reviewed manifest.")
    return parser.parse_args()


def _validate_output_paths(args: argparse.Namespace) -> None:
    input_paths = {
        args.candidate_manifest.resolve(),
        args.mask_audit.resolve(),
        args.review_decisions.resolve(),
    }
    outputs = {
        "output_manifest": args.output_manifest.resolve(),
        "output_audit": args.output_audit.resolve(),
    }
    if outputs["output_manifest"] == outputs["output_audit"]:
        raise ValidationError("--output-manifest and --output-audit must be different files")
    for label, path in outputs.items():
        if path in input_paths:
            raise ValidationError(f"{label} must not overwrite an input artifact: {path}")


def main() -> None:
    args = _parse_args()
    _validate_output_paths(args)
    output_audit: Dict[str, Any] = {
        "created_at": _utc_now(),
        "status": "failed",
        "training_readiness": {"ready_for_training": False},
        "inputs": {
            "candidate_manifest": str(args.candidate_manifest),
            "mask_audit": str(args.mask_audit),
            "review_decisions": str(args.review_decisions),
        },
    }
    try:
        candidate_records = _read_jsonl(args.candidate_manifest, "candidate manifest")
        candidate_by_id = _index_by_pair_id(candidate_records, "candidate manifest")
        if len(candidate_by_id) != EXPECTED_PAIR_COUNT:
            raise ValidationError(f"candidate manifest must contain exactly {EXPECTED_PAIR_COUNT} unique pairs")
        for pair_id, record in candidate_by_id.items():
            _validate_candidate_record(record, pair_id)

        if not args.mask_audit.is_file():
            raise ValidationError(f"mask audit does not exist or is not a file: {args.mask_audit}")
        try:
            mask_audit = json.loads(args.mask_audit.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"mask audit is not valid JSON: {exc}") from exc
        if not isinstance(mask_audit, dict):
            raise ValidationError("mask audit root must be a JSON object")
        audit_by_id = _validate_mask_audit(mask_audit, candidate_by_id)

        decision_records = _read_jsonl(args.review_decisions, "review decisions")
        decisions_by_id = _index_by_pair_id(decision_records, "review decisions")
        if set(decisions_by_id) != set(candidate_by_id):
            missing = sorted(set(candidate_by_id) - set(decisions_by_id))
            extra = sorted(set(decisions_by_id) - set(candidate_by_id))
            raise ValidationError(f"review decision pair IDs differ from candidates; missing={missing}, extra={extra}")

        reviewed_records: List[Dict[str, Any]] = []
        pair_audits: List[Dict[str, Any]] = []
        for pair_id in sorted(candidate_by_id):
            candidate = candidate_by_id[pair_id]
            candidate_mask = candidate["dynamic_candidate_mask"]
            candidate_path = Path(candidate_mask["path"])
            observed_sha256 = _sha256(candidate_path)
            if observed_sha256 != candidate_mask["sha256"]:
                raise ValidationError(f"{pair_id}: candidate mask SHA256 differs from manifest")
            if audit_by_id[pair_id]["mask"].get("sha256") != observed_sha256:
                raise ValidationError(f"{pair_id}: candidate mask SHA256 differs from mask audit")

            decision, roi_paths = _validate_decision(decisions_by_id[pair_id], pair_id, args.review_decisions.parent)
            candidate_pixels = _load_binary_mask(candidate_path, f"{pair_id} candidate mask")
            roi_pixels = {
                field: (
                    _load_binary_mask(path, f"{pair_id} {field}")
                    if path is not None
                    else np.zeros_like(candidate_pixels, dtype=bool)
                )
                for field, path in roi_paths.items()
            }
            for field, pixels in roi_pixels.items():
                if pixels.shape != candidate_pixels.shape:
                    raise ValidationError(f"{pair_id}: {field} shape must match candidate mask shape {candidate_pixels.shape}")
            if not any(pixels.any() for pixels in roi_pixels.values()):
                raise ValidationError(f"{pair_id}: at least one changed-region ROI must be non-empty")
            categories = tuple(roi_pixels)
            overlap_pixels = sum(
                int(np.logical_and(roi_pixels[categories[i]], roi_pixels[categories[j]]).sum())
                for i in range(len(categories))
                for j in range(i + 1, len(categories))
            )
            if overlap_pixels:
                raise ValidationError(f"{pair_id}: source-only, target-only, and shared ROIs overlap by {overlap_pixels} pixels")
            source_support = np.logical_or(roi_pixels["source_changed_roi"], roi_pixels["shared_changed_roi"])
            target_support = np.logical_or(roi_pixels["target_changed_roi"], roi_pixels["shared_changed_roi"])
            if not source_support.any() or not target_support.any():
                raise ValidationError(
                    f"{pair_id}: each state needs changed-region support; use shared_changed_roi for same-location replacement"
                )
            union_pixels = np.logical_or.reduce(tuple(roi_pixels.values()))
            uncovered_pixels = int(np.logical_and(candidate_pixels, ~union_pixels).sum())
            if uncovered_pixels:
                raise ValidationError(f"{pair_id}: changed ROIs fail to cover {uncovered_pixels} candidate-mask pixels")

            candidate_count = int(candidate_pixels.sum())
            approved_mask_path = args.output_manifest.parent / "reviewed_allowed_change_masks" / f"{pair_id}.png"
            _write_binary_mask(approved_mask_path, union_pixels)

            def bbox_ltrb(values: np.ndarray, label: str) -> List[int]:
                ys, xs = np.nonzero(values)
                if not len(xs):
                    raise ValidationError(f"{pair_id}: {label} is empty")
                return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

            def roi_metadata(field: str) -> Dict[str, Any] | None:
                path, pixels = roi_paths[field], roi_pixels[field]
                if path is None:
                    return None
                if not pixels.any():
                    raise ValidationError(f"{pair_id}: provided {field} is empty")
                return {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "pixels": int(pixels.sum()),
                    "bbox_ltrb": bbox_ltrb(pixels, field),
                }

            roi_audit = {
                "approved_allowed_change_mask": {
                    "path": str(approved_mask_path),
                    "sha256": _sha256(approved_mask_path),
                    "pixels": int(union_pixels.sum()),
                    "bbox_ltrb": bbox_ltrb(union_pixels, "approved_allowed_change_mask"),
                },
                **{field: roi_metadata(field) for field in categories},
                "candidate_mask_coverage": {
                    "candidate_pixels": candidate_count,
                    "covered_candidate_pixels": candidate_count - uncovered_pixels,
                    "uncovered_candidate_pixels": uncovered_pixels,
                    "coverage_ratio": 1.0,
                    "source_target_shared_overlap_pixels": 0,
                    "union_pixels": int(union_pixels.sum()),
                },
            }
            reviewed_records.append(_reviewed_record(candidate, decision, roi_audit))
            pair_audits.append({"pair_id": pair_id, "status": "approved", **decision, **roi_audit})

        if args.output_manifest.exists() and not args.overwrite:
            raise ValidationError(f"reviewed manifest already exists: {args.output_manifest}; rerun with --overwrite")
        output_audit.update({
            "status": "training_ready",
            "training_readiness": {"ready_for_training": True},
            "inputs": {
                **output_audit["inputs"],
                "candidate_manifest_sha256": _sha256(args.candidate_manifest),
                "mask_audit_sha256": _sha256(args.mask_audit),
                "review_decisions_sha256": _sha256(args.review_decisions),
            },
            "output": {"reviewed_paired_manifest": str(args.output_manifest)},
            "counts": {"candidate_pairs": EXPECTED_PAIR_COUNT, "approved_pairs": len(reviewed_records)},
            "pairs": pair_audits,
        })
        _write_jsonl(args.output_manifest, reviewed_records)
        _atomic_json(args.output_audit, output_audit)
        print(json.dumps({"status": "training_ready", "paired_records": len(reviewed_records), "output_manifest": str(args.output_manifest)}, indent=2))
    except Exception as exc:  # noqa: BLE001
        output_audit["error"] = str(exc)
        _atomic_json(args.output_audit, output_audit)
        raise


if __name__ == "__main__":
    main()
