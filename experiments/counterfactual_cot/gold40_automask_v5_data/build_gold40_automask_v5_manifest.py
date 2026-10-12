#!/usr/bin/env python3
"""Build the Gold40 AutoMask-v5 training manifest from v4 candidate artifacts.

This adapter deliberately does not read, require, create, or infer human review data.
Every output mask is explicitly labelled as an automatic source/target RGB-difference
mask, not as a reviewed region-of-interest mask.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
V4_ARTIFACTS = SCRIPT_DIR.parent / "gold40_dual_target_cot_v4_data" / "artifacts"
DEFAULT_CANDIDATE_MANIFEST = V4_ARTIFACTS / "gold40_source_target_paired_candidate_v4.jsonl"
DEFAULT_OUTPUT_MANIFEST = SCRIPT_DIR / "artifacts" / "gold40_source_target_automask_v5.jsonl"
DEFAULT_OUTPUT_AUDIT = SCRIPT_DIR / "artifacts" / "gold40_automask_v5_audit.json"
EXPECTED_PAIR_COUNT = 40
SCHEMA_VERSION = "gold40_automask_v5"
MASK_PROVENANCE = {
    "kind": "automatic_rgb_difference_mask",
    "semantic_annotation": False,
    "state_specific": False,
}


class ValidationError(RuntimeError):
    """The candidate-derived manifest is not safe to label AutoMask-v5."""


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
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        raise ValidationError(f"candidate manifest does not exist or is not a file: {path}")
    records: list[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"candidate manifest line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValidationError(f"candidate manifest line {line_number} must be a JSON object")
        records.append(record)
    if not records:
        raise ValidationError(f"candidate manifest is empty: {path}")
    return records


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be non-empty text")
    return value.strip()


def _digest(value: Any, label: str) -> str:
    value = _text(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValidationError(f"{label} must be a SHA256 hex digest")
    return value.lower()


def _existing_file(value: Any, label: str) -> Path:
    path = Path(_text(value, label))
    if not path.is_file():
        raise ValidationError(f"{label} does not exist or is not a file: {path}")
    return path


def _validate_sha256(path: Path, declared: Any, label: str) -> str:
    expected = _digest(declared, f"{label}.sha256")
    observed = _sha256(path)
    if observed != expected:
        raise ValidationError(f"{label} SHA256 differs from manifest: {path}")
    return observed


def _validate_mask(path: Path, pair_id: str) -> Dict[str, Any]:
    try:
        with Image.open(path) as image:
            values = image.convert("L")
            if values.size != (1024, 1024):
                raise ValidationError(f"{pair_id}: automatic mask must be 1024x1024, got {values.size}")
            nonzero = sum(value > 0 for value in values.getdata())
    except (OSError, ValueError) as exc:
        raise ValidationError(f"{pair_id}: cannot decode automatic mask {path}: {exc}") from exc
    if not nonzero:
        raise ValidationError(f"{pair_id}: automatic mask is empty: {path}")
    return {"width": 1024, "height": 1024, "nonzero_pixels": nonzero}


def _validate_image(path: Path, pair_id: str, label: str) -> None:
    try:
        with Image.open(path) as image:
            if image.size != (1024, 1024):
                raise ValidationError(f"{pair_id}: {label} must be 1024x1024, got {image.size}")
    except (OSError, ValueError) as exc:
        raise ValidationError(f"{pair_id}: cannot decode {label} image {path}: {exc}") from exc


def _branch_map(record: Mapping[str, Any], pair_id: str) -> Dict[str, Mapping[str, Any]]:
    branches = record.get("paired_branches")
    if not isinstance(branches, list) or len(branches) != 2:
        raise ValidationError(f"{pair_id}: paired_branches must contain exactly two branches")
    mapped: Dict[str, Mapping[str, Any]] = {}
    for branch in branches:
        if not isinstance(branch, Mapping):
            raise ValidationError(f"{pair_id}: paired_branches entries must be objects")
        branch_id = _text(branch.get("branch_id"), f"{pair_id}.paired_branches.branch_id")
        if branch_id in mapped:
            raise ValidationError(f"{pair_id}: duplicate branch_id {branch_id}")
        mapped[branch_id] = branch
    if set(mapped) != {"source_state", "target_state"}:
        raise ValidationError(f"{pair_id}: branches must be source_state and target_state exactly once")
    return mapped


def _validate_branch_consistency(record: Mapping[str, Any], pair_id: str) -> Dict[str, Dict[str, str]]:
    source, target = record.get("source"), record.get("target")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise ValidationError(f"{pair_id}: source and target objects are required")
    branches = _branch_map(record, pair_id)
    states = (("source_state", "source", source), ("target_state", "target", target))
    result: Dict[str, Dict[str, str]] = {}
    for branch_id, expected_state, endpoint in states:
        branch = branches[branch_id]
        if branch.get("target_state") != expected_state:
            raise ValidationError(f"{pair_id}: {branch_id}.target_state must be {expected_state!r}")
        image_path = _existing_file(endpoint.get("image"), f"{pair_id}.{expected_state}.image")
        image_sha256 = _validate_sha256(image_path, endpoint.get("image_sha256"), f"{pair_id}.{expected_state}.image")
        _validate_image(image_path, pair_id, expected_state)
        if _text(branch.get("target_image"), f"{pair_id}.{branch_id}.target_image") != str(image_path):
            raise ValidationError(f"{pair_id}: {branch_id}.target_image differs from {expected_state}.image")
        if _digest(branch.get("target_image_sha256"), f"{pair_id}.{branch_id}.target_image") != image_sha256:
            raise ValidationError(f"{pair_id}: {branch_id}.target_image_sha256 differs from {expected_state}.image_sha256")
        if _text(branch.get("target_cot_text"), f"{pair_id}.{branch_id}.target_cot_text") != _text(endpoint.get("cot_text"), f"{pair_id}.{expected_state}.cot_text"):
            raise ValidationError(f"{pair_id}: {branch_id}.target_cot_text differs from {expected_state}.cot_text")
        if branch.get("target_cot") != endpoint.get("cot"):
            raise ValidationError(f"{pair_id}: {branch_id}.target_cot differs from {expected_state}.cot")
        result[expected_state] = {"path": str(image_path), "sha256": image_sha256}
    return result


def _automatic_mask_path(candidate_mask: Mapping[str, Any], automatic_mask_root: Path | None, pair_id: str) -> Path:
    manifest_path = _existing_file(candidate_mask.get("path"), f"{pair_id}.dynamic_candidate_mask.path")
    if automatic_mask_root is None:
        return manifest_path
    mask_path = automatic_mask_root / f"{pair_id}.png"
    if not mask_path.is_file():
        raise ValidationError(f"{pair_id}: automatic mask is missing from --automatic-mask-root: {mask_path}")
    if mask_path.resolve() != manifest_path.resolve():
        raise ValidationError(
            f"{pair_id}: --automatic-mask-root mask must be the candidate manifest mask; "
            f"found {mask_path}, manifest declares {manifest_path}"
        )
    return mask_path


def _adapt_record(record: Mapping[str, Any], automatic_mask_root: Path | None) -> tuple[Dict[str, Any], Dict[str, Any]]:
    pair_id = _text(record.get("pair_id"), "pair_id")
    if record.get("macro_pair_id") != f"dual_cot::{pair_id}":
        raise ValidationError(f"{pair_id}: macro_pair_id must equal dual_cot::<pair_id>")
    forbidden_review_fields = ("review", "human_review", "reviewed_mask", "dynamic_regions")
    present = [field for field in forbidden_review_fields if field in record]
    if present:
        raise ValidationError(f"{pair_id}: human-reviewed schema fields are forbidden in AutoMask-v5 input: {present}")
    template_family = _text(record.get("template_family"), f"{pair_id}.template_family")
    intervention = record.get("intervention")
    if not isinstance(intervention, Mapping):
        raise ValidationError(f"{pair_id}: intervention object is required")
    intervention_type = _text(intervention.get("intervention_type"), f"{pair_id}.intervention.intervention_type")
    candidate_mask = record.get("dynamic_candidate_mask")
    if not isinstance(candidate_mask, Mapping):
        raise ValidationError(f"{pair_id}: dynamic_candidate_mask is required")
    if candidate_mask.get("candidate_only") is not True or candidate_mask.get("manual_review_required") is not True:
        raise ValidationError(f"{pair_id}: input must be the original unreviewed v4 candidate schema")
    if candidate_mask.get("review_status") != "unreviewed":
        raise ValidationError(f"{pair_id}: input candidate review_status must be 'unreviewed'")
    source_target = _validate_branch_consistency(record, pair_id)
    mask_path = _automatic_mask_path(candidate_mask, automatic_mask_root, pair_id)
    mask_sha256 = _validate_sha256(mask_path, candidate_mask.get("sha256"), f"{pair_id}.dynamic_candidate_mask")
    mask_geometry = _validate_mask(mask_path, pair_id)

    adapted = copy.deepcopy(dict(record))
    adapted.pop("dynamic_candidate_mask", None)
    adapted["schema_version"] = SCHEMA_VERSION
    adapted["source_scene_id"] = template_family
    adapted["intervention_type"] = intervention_type
    adapted["mask_provenance"] = MASK_PROVENANCE
    adapted["automatic_difference_mask"] = {
        "path": str(mask_path),
        "sha256": mask_sha256,
        "geometry": mask_geometry,
        "source_candidate_mask": {
            "threshold": candidate_mask.get("threshold"),
            "dilation_px": candidate_mask.get("dilation_px"),
            "input_field": "dynamic_candidate_mask",
        },
    }
    adapted["training_readiness"] = {
        "ready_for_training": True,
        "mode": "automatic_difference_mask",
        "basis": "candidate_derived_automatic_mask",
        "human_review_used": False,
        "mask_provenance": MASK_PROVENANCE,
    }
    adapted["provenance"] = {
        **(copy.deepcopy(record.get("provenance")) if isinstance(record.get("provenance"), Mapping) else {}),
        "adapter": Path(__file__).name,
        "input_kind": "v4_unreviewed_candidate_manifest",
        "mask_provenance": MASK_PROVENANCE,
        "human_review_used": False,
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "pair_id": pair_id,
        "source_scene_id": template_family,
        "intervention_type": intervention_type,
        "status": "training_ready_candidate_derived",
        "mask_provenance": MASK_PROVENANCE,
        "source": source_target["source"],
        "target": source_target["target"],
        "mask": {"path": str(mask_path), "sha256": mask_sha256, **mask_geometry},
        "candidate_paired_branches_consistent": True,
    }
    return adapted, audit


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument(
        "--automatic-mask-root",
        type=Path,
        help="Optional v4 dynamic_masks directory; each <pair_id>.png must exactly match the manifest path.",
    )
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_OUTPUT_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_manifest.resolve() == args.output_audit.resolve():
        raise ValidationError("--output-manifest and --output-audit must differ")
    input_paths = {args.candidate_manifest.resolve()}
    if args.automatic_mask_root is not None:
        if not args.automatic_mask_root.is_dir():
            raise ValidationError(f"--automatic-mask-root is not a directory: {args.automatic_mask_root}")
        input_paths.add(args.automatic_mask_root.resolve())
    if args.output_manifest.resolve() in input_paths or args.output_audit.resolve() in input_paths:
        raise ValidationError("output paths must not overwrite input artifacts")
    existing = [path for path in (args.output_manifest, args.output_audit) if path.exists()]
    if existing and not args.overwrite:
        raise ValidationError(f"output already exists; pass --overwrite to replace: {existing}")

    audit: Dict[str, Any] = {
        "created_at": _utc_now(),
        "status": "failed",
        "training_readiness": {"ready_for_training": False},
        "inputs": {"candidate_manifest": str(args.candidate_manifest), "automatic_mask_root": str(args.automatic_mask_root) if args.automatic_mask_root else None},
        "mask_provenance": MASK_PROVENANCE,
        "human_review_used": False,
    }
    try:
        candidates = _read_jsonl(args.candidate_manifest)
        adapted_records: list[Dict[str, Any]] = []
        pair_audits: list[Dict[str, Any]] = []
        seen_pair_ids: set[str] = set()
        for candidate in candidates:
            adapted, pair_audit = _adapt_record(candidate, args.automatic_mask_root)
            if pair_audit["pair_id"] in seen_pair_ids:
                raise ValidationError(f"candidate manifest contains duplicate pair_id: {pair_audit['pair_id']}")
            seen_pair_ids.add(pair_audit["pair_id"])
            adapted_records.append(adapted)
            pair_audits.append(pair_audit)
        if len(seen_pair_ids) != EXPECTED_PAIR_COUNT:
            raise ValidationError(f"candidate manifest must contain exactly {EXPECTED_PAIR_COUNT} unique pairs; got {len(seen_pair_ids)}")

        adapted_records.sort(key=lambda item: item["pair_id"])
        pair_audits.sort(key=lambda item: item["pair_id"])
        _write_jsonl(args.output_manifest, adapted_records)
        audit.update({
            "status": "training_ready_candidate_derived",
            "training_readiness": {
                "ready_for_training": True,
                "basis": "candidate_derived_automatic_mask",
                "human_review_used": False,
            },
            "inputs": {**audit["inputs"], "candidate_manifest_sha256": _sha256(args.candidate_manifest)},
            "output": {"manifest": str(args.output_manifest), "manifest_sha256": _sha256(args.output_manifest)},
            "counts": {
                "unique_pairs": len(adapted_records),
                "source_scenes": len({item["source_scene_id"] for item in adapted_records}),
                "intervention_types": dict(sorted(Counter(item["intervention_type"] for item in adapted_records).items())),
            },
            "pairs": pair_audits,
        })
        _atomic_json(args.output_audit, audit)
        print(json.dumps({"status": audit["status"], "paired_records": len(adapted_records), "output_manifest": str(args.output_manifest)}, indent=2))
    except Exception as exc:  # noqa: BLE001
        audit["error"] = str(exc)
        _atomic_json(args.output_audit, audit)
        raise


if __name__ == "__main__":
    main()
