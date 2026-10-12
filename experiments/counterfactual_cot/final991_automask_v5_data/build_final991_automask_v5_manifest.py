#!/usr/bin/env python3
"""Adapt the 991-pair final audited dataset to the Gold40 AutoMask-v5 paired contract.

This adapter is a data-side converter only.  The v5 trainer
(``gold40_automask_dual_target_cot1024_fsdp_v5/train.py``), its paired loader and the
AutoMask objective are consumed unchanged: the dual-branch macro step, the 0.5+0.5
backward scaling, the shared pack seed / raw timestep / controlled noise and the
single optimizer step per macro step are all properties of that trainer, not of the
manifest.

Input  : final_training_manifest_20260928.jsonl (991 audited pairs, absolute paths)
Output : AutoMask-v5 ready JSONL + audit json

The output mask payload stays an automatic RGB-difference mask, so
``training_readiness.human_review_used`` is false by contract; the pair-level human
audit is recorded separately under ``provenance.human_pair_audit``.
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
DEFAULT_INPUT = (
    SCRIPT_DIR.parent
    / "gold1000_framework_strict_v4"
    / "final_training_manifest_20260928.jsonl"
)
DEFAULT_OUTPUT_MANIFEST = SCRIPT_DIR / "artifacts" / "final991_source_target_automask_v5.jsonl"
DEFAULT_OUTPUT_AUDIT = SCRIPT_DIR / "artifacts" / "final991_automask_v5_audit.json"

SCHEMA_VERSION = "gold40_automask_v5"
MASK_PROVENANCE = {
    "kind": "automatic_rgb_difference_mask",
    "semantic_annotation": False,
    "state_specific": False,
}
STATE_IDS = ("source_state", "target_state")
FORBIDDEN_FIELDS = ("review", "human_review", "reviewed_mask", "dynamic_regions", "dynamic_candidate_mask")


class ValidationError(RuntimeError):
    """The audited dataset cannot be safely expressed as an AutoMask-v5 record."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    records: list[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValidationError(f"{path}:{line_no}: record must be an object")
            records.append(value)
    if not records:
        raise ValidationError(f"input manifest is empty: {path}")
    return records


def _atomic_write_json(path: Path, payload: Any) -> None:
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


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be non-empty text")
    return value.strip()


def _validate_image(path: Path, pair_id: str, label: str) -> str:
    if not path.is_file():
        raise ValidationError(f"{pair_id}: missing {label} image {path}")
    try:
        with Image.open(path) as image:
            if image.size != (1024, 1024):
                raise ValidationError(f"{pair_id}: {label} must be 1024x1024, got {image.size}")
    except (OSError, ValueError) as exc:
        raise ValidationError(f"{pair_id}: cannot decode {label} image {path}: {exc}") from exc
    return _sha256(path)


def _validate_mask(path: Path, pair_id: str) -> Dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"{pair_id}: missing automatic mask {path}")
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


def _adapt_record(record: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    pair_id = _text(record.get("pair_id"), "pair_id")
    assets = record.get("training_assets")
    if not isinstance(assets, Mapping):
        raise ValidationError(f"{pair_id}: training_assets block is required")

    source_path = Path(_text(assets.get("source_abs"), f"{pair_id}.source_abs"))
    target_path = Path(_text(assets.get("target_abs"), f"{pair_id}.target_abs"))
    mask_path = Path(_text(assets.get("mask_abs"), f"{pair_id}.mask_abs"))

    source_sha = _validate_image(source_path, pair_id, "source")
    target_sha = _validate_image(target_path, pair_id, "target")
    mask_geometry = _validate_mask(mask_path, pair_id)
    mask_sha = _sha256(mask_path)

    source = record.get("source")
    target = record.get("target")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise ValidationError(f"{pair_id}: source/target blocks are required")
    source_cot_text = _text(source.get("cot_text"), f"{pair_id}.source.cot_text")
    target_cot_text = _text(target.get("cot_text"), f"{pair_id}.target.cot_text")
    _text(source.get("prompt"), f"{pair_id}.source.prompt")
    _text(target.get("prompt"), f"{pair_id}.target.prompt")
    if not isinstance(source.get("cot"), Mapping) or not isinstance(target.get("cot"), Mapping):
        raise ValidationError(f"{pair_id}: source.cot/target.cot objects are required")

    intervention = record.get("intervention")
    if not isinstance(intervention, Mapping):
        raise ValidationError(f"{pair_id}: intervention object is required")
    intervention_type = _text(intervention.get("intervention_type"), f"{pair_id}.intervention_type")
    source_scene_id = _text(record.get("source_scene_id"), f"{pair_id}.source_scene_id")

    adapted = copy.deepcopy(dict(record))
    for field in FORBIDDEN_FIELDS:
        adapted.pop(field, None)
    adapted.pop("training_assets", None)

    adapted["schema_version"] = SCHEMA_VERSION
    adapted["macro_pair_id"] = f"dual_cot::{pair_id}"
    adapted["source_scene_id"] = source_scene_id
    adapted["intervention_type"] = intervention_type
    adapted["source"] = {
        **{k: v for k, v in source.items() if k in ("cot", "cot_text", "prompt")},
        "image": str(source_path),
        "image_sha256": source_sha,
    }
    adapted["target"] = {
        **{k: v for k, v in target.items() if k in ("cot", "cot_text", "prompt")},
        "image": str(target_path),
        "image_sha256": target_sha,
    }
    adapted["mask_provenance"] = MASK_PROVENANCE
    adapted["automatic_difference_mask"] = {
        "path": str(mask_path),
        "sha256": mask_sha,
        "geometry": mask_geometry,
        "source_candidate_mask": {
            "input_field": "final991_training_mask",
            "threshold": 24,
            "dilation_px": None,
        },
    }
    adapted["training_readiness"] = {
        "ready_for_training": True,
        "mode": "automatic_difference_mask",
        "basis": "audited_dataset_automatic_mask",
        "human_review_used": False,
        "mask_provenance": MASK_PROVENANCE,
    }
    adapted["paired_branches"] = [
        {
            "branch_id": "source_state",
            "target_state": "source",
            "target_image": str(source_path),
            "target_image_sha256": source_sha,
            "target_cot_text": source_cot_text,
            "target_cot": copy.deepcopy(source["cot"]),
        },
        {
            "branch_id": "target_state",
            "target_state": "target",
            "target_image": str(target_path),
            "target_image_sha256": target_sha,
            "target_cot_text": target_cot_text,
            "target_cot": copy.deepcopy(target["cot"]),
        },
    ]
    adapted["provenance"] = {
        "adapter": Path(__file__).name,
        "input_kind": "final991_human_audited_manifest_20260928",
        "input_manifest": "gold1000_framework_strict_v4/final_training_manifest_20260928.jsonl",
        "asset_batch": assets.get("asset_batch", ""),
        "human_pair_audit": {
            "date": "2026-09-28",
            "scope": "pair-level accept/delete across four audit rounds",
            "mask_audited": False,
            "note": "pair accepted by human audit; mask remains an automatic RGB-difference mask",
        },
        "mask_provenance": MASK_PROVENANCE,
    }

    audit = {
        "pair_id": pair_id,
        "schema_version": SCHEMA_VERSION,
        "source_scene_id": source_scene_id,
        "intervention_type": intervention_type,
        "status": "training_ready_audited_dataset",
        "asset_batch": assets.get("asset_batch", ""),
        "source": {"image": str(source_path), "sha256": source_sha},
        "target": {"image": str(target_path), "sha256": target_sha},
        "mask": {"path": str(mask_path), "sha256": mask_sha, **mask_geometry},
    }
    return adapted, audit


def build(input_path: Path, output_manifest: Path, output_audit: Path, overwrite: bool) -> Dict[str, Any]:
    if output_manifest.exists() or output_audit.exists():
        if not overwrite:
            raise ValidationError(f"output already exists; pass --overwrite: {output_manifest}")
    records_in = _read_jsonl(input_path)
    seen: set[str] = set()
    adapted_records: list[Dict[str, Any]] = []
    audits: list[Dict[str, Any]] = []
    for record in records_in:
        pair_id = str(record.get("pair_id", ""))
        if pair_id in seen:
            raise ValidationError(f"duplicate pair_id in input: {pair_id}")
        seen.add(pair_id)
        adapted, audit = _adapt_record(record)
        adapted_records.append(adapted)
        audits.append(audit)

    _write_jsonl(output_manifest, adapted_records)
    summary = {
        "created_at": _utc_now(),
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "input_manifest": str(input_path),
        "record_count": len(adapted_records),
        "source_scene_count": len({a["source_scene_id"] for a in audits}),
        "asset_batch_counts": dict(Counter(a["asset_batch"] for a in audits)),
        "intervention_type_counts": dict(Counter(a["intervention_type"] for a in audits)),
        "mask_provenance": MASK_PROVENANCE,
        "training_readiness": {"ready_for_training": True, "human_review_used": False},
        "output_manifest": str(output_manifest),
        "output_audit": str(output_audit),
    }
    _atomic_write_json(output_audit, {**summary, "records": audits})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_OUTPUT_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.input_manifest, args.output_manifest, args.output_audit, args.overwrite),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
