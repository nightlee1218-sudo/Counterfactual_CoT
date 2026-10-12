#!/usr/bin/env python3
"""Build auditable Gold40 source-state/target-state paired data for v4-style training.

This script is intentionally data preparation only. It generates dynamic *candidate*
masks from source/target RGB differences and marks every pair as requiring human mask
review before a future training job is allowed to consume the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.data.mask_utils import create_candidate_mask  # noqa: E402
from experiments.counterfactual_cot.data.schema import (  # noqa: E402
    Intervention,
    SceneGraph,
    changed_scene_paths,
    render_cot_text,
)
from experiments.counterfactual_cot.training.gold40_full_manifest_audit import (  # noqa: E402
    _apply_intervention_extended,
    _normalize_atom,
    _normalize_field,
)

V3_MANIFEST = BAGEL_ROOT / (
    "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/manifests/"
    "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl"
)
FINAL_MANIFEST = BAGEL_ROOT / (
    "experiments/counterfactual_cot/gold40_final/manifest/intervention_pairs_gold40_final.jsonl"
)
ASSET_INDEX = BAGEL_ROOT / "experiments/counterfactual_cot/gold40_final/manifest/asset_index.jsonl"
DEFAULT_OUTPUT = BAGEL_ROOT / "experiments/counterfactual_cot/gold40_dual_target_cot_v4_data/artifacts"
MASK_THRESHOLD = 24
MASK_DILATION_PX = 9
IMAGE_SIZE = (1024, 1024)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _mask_geometry(mask_path: Path) -> Dict[str, Any]:
    with Image.open(mask_path) as image:
        values = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    if values.shape != (1024, 1024):
        raise RuntimeError(f"Candidate mask has unexpected shape {values.shape}: {mask_path}")
    ys, xs = np.nonzero(values)
    if not len(xs):
        raise RuntimeError(f"Candidate mask is empty: {mask_path}")
    height, width = values.shape
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return {
        "pixels": int(values.sum()),
        "area_ratio": float(values.mean()),
        "bbox_ltrb": bbox,
        "bbox_area_ratio": float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (height * width)),
        "touches_left": bool(values[:, 0].any()),
        "touches_right": bool(values[:, -1].any()),
        "touches_top": bool(values[0, :].any()),
        "touches_bottom": bool(values[-1, :].any()),
    }


def _raw_difference_geometry(source_path: Path, target_path: Path) -> Dict[str, Any]:
    with Image.open(source_path) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.int16)
    with Image.open(target_path) as image:
        target = np.asarray(image.convert("RGB"), dtype=np.int16)
    if source.shape != target.shape or source.shape[:2] != (1024, 1024):
        raise RuntimeError(f"Unexpected source/target shapes: {source.shape}, {target.shape}")
    difference = np.max(np.abs(source - target), axis=2)
    changed = difference > MASK_THRESHOLD
    return {
        "raw_changed_pixels": int(changed.sum()),
        "raw_changed_ratio": float(changed.mean()),
        "mean_absolute_rgb_difference": float(np.abs(source - target).mean()),
        "max_channel_difference": int(difference.max()),
    }


def _semantic_audit(record: Dict[str, Any]) -> Dict[str, Any]:
    source_scene = SceneGraph.from_dict(record["source"]["cot"])
    target_scene = SceneGraph.from_dict(record["target"]["cot"])
    intervention = Intervention.from_dict(record["intervention"])
    applied = _apply_intervention_extended(source_scene, intervention)
    if applied.to_dict() != target_scene.to_dict():
        raise RuntimeError("Applying intervention to source scene does not exactly reproduce target scene")
    def endpoint(diff_path: str) -> str:
        if diff_path.startswith("entities["):
            index_text, _, remainder = diff_path[len("entities["):].partition("]")
            try:
                entity = source_scene.entities[int(index_text)]
            except (ValueError, IndexError):
                return diff_path
            if remainder.startswith("."):
                remainder = remainder[1:]
            return _normalize_atom(f"{entity.id}.{remainder}")
        return _normalize_atom(diff_path)

    changed = sorted({endpoint(path) for path in changed_scene_paths(source_scene, target_scene)})
    expected = _normalize_atom(f"{intervention.entity_id}.{_normalize_field(intervention.field)}")
    if changed != [expected]:
        raise RuntimeError(f"Expected exactly one changed atom {expected}, got {changed}")
    source_text = render_cot_text(source_scene)
    target_text = render_cot_text(target_scene)
    return {
        "changed_atoms": changed,
        "expected_changed_atom": expected,
        "source_cot_text_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "target_cot_text_sha256": hashlib.sha256(target_text.encode()).hexdigest(),
        "source_cot_text": source_text,
        "target_cot_text": target_text,
    }


def _asset_map(index_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    mapped = {}
    for record in index_records:
        pair_id = str(record["pair_id"])
        if pair_id in mapped:
            raise RuntimeError(f"Duplicate asset index pair_id: {pair_id}")
        mapped[pair_id] = record
    return mapped


def _candidate_record(
    semantic_record: Dict[str, Any],
    asset_record: Dict[str, Any],
    v3_record: Dict[str, Any],
    mask_path: Path,
    mask_meta: Dict[str, Any],
    raw_meta: Dict[str, Any],
    semantic_meta: Dict[str, Any],
) -> Dict[str, Any]:
    pair_id = str(semantic_record["pair_id"])
    source_asset = asset_record["source"]
    target_asset = asset_record["target"]
    source_path = Path(source_asset["new_path"])
    target_path = Path(target_asset["new_path"])
    if _sha256(source_path) != source_asset["sha256"] or _sha256(target_path) != target_asset["sha256"]:
        raise RuntimeError(f"Asset SHA256 mismatch: {pair_id}")
    semantic_delta_from_v3 = {
        "intervention_changed": semantic_record.get("intervention") != v3_record.get("intervention"),
        "preservation_changed": semantic_record.get("preservation") != v3_record.get("preservation"),
    }
    source_cot = semantic_record["source"]["cot"]
    target_cot = semantic_record["target"]["cot"]
    source_text = semantic_meta.pop("source_cot_text")
    target_text = semantic_meta.pop("target_cot_text")
    source_branch = {
        "branch_id": "source_state",
        "target_state": "source",
        "target_image": str(source_path),
        "target_image_sha256": source_asset["sha256"],
        "target_cot": source_cot,
        "target_cot_text": source_text,
    }
    target_branch = {
        "branch_id": "target_state",
        "target_state": "target",
        "target_image": str(target_path),
        "target_image_sha256": target_asset["sha256"],
        "target_cot": target_cot,
        "target_cot_text": target_text,
    }
    return {
        "pair_id": pair_id,
        "macro_pair_id": f"dual_cot::{pair_id}",
        "template_family": asset_record.get("template_family"),
        "intervention": semantic_record["intervention"],
        "preservation": semantic_record.get("preservation", {}),
        "source": {
            "image": str(source_path),
            "image_sha256": source_asset["sha256"],
            "prompt": semantic_record["source"]["prompt"],
            "cot": source_cot,
            "cot_text": source_text,
        },
        "target": {
            "image": str(target_path),
            "image_sha256": target_asset["sha256"],
            "prompt": semantic_record["target"]["prompt"],
            "cot": target_cot,
            "cot_text": target_text,
        },
        "paired_branches": [source_branch, target_branch],
        "dynamic_candidate_mask": {
            "path": str(mask_path),
            "sha256": _sha256(mask_path),
            "threshold": MASK_THRESHOLD,
            "dilation_px": MASK_DILATION_PX,
            "candidate_only": True,
            "manual_review_required": True,
            "review_status": "unreviewed",
            "mask_geometry": mask_meta,
            "raw_difference": raw_meta,
        },
        "semantic_audit": semantic_meta,
        "provenance": {
            "semantic_manifest": str(FINAL_MANIFEST),
            "asset_index": str(ASSET_INDEX),
            "v3_semantic_reference": str(V3_MANIFEST),
            "semantic_delta_from_v3": semantic_delta_from_v3,
            "asset_source": "gold40_final consolidated source/target files",
        },
    }


def _flat_branch_records(paired: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []
    for branch in paired["paired_branches"]:
        result.append({
            "branch_record_id": f"{paired['macro_pair_id']}::{branch['branch_id']}",
            "macro_pair_id": paired["macro_pair_id"],
            "pair_id": paired["pair_id"],
            "branch_id": branch["branch_id"],
            "template_family": paired["template_family"],
            "intervention": paired["intervention"],
            "source": paired["source"],
            "target": {
                "image": branch["target_image"],
                "image_sha256": branch["target_image_sha256"],
                "cot": branch["target_cot"],
                "cot_text": branch["target_cot_text"],
                "state": branch["target_state"],
            },
            "dynamic_candidate_mask": paired["dynamic_candidate_mask"],
        })
    return result


def _review_gallery(records: List[Dict[str, Any]], output_path: Path) -> None:
    tile, columns, label_height = 128, 2, 18
    rows = (len(records) + columns - 1) // columns
    panel_width = tile * 3
    canvas = Image.new("RGB", (columns * panel_width, rows * (tile + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        column, row = index % columns, index // columns
        origin_x, origin_y = column * panel_width, row * (tile + label_height)
        source_path = Path(record["source"]["image"])
        target_path = Path(record["target"]["image"])
        mask_path = Path(record["dynamic_candidate_mask"]["path"])
        with Image.open(source_path) as image:
            source = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        with Image.open(target_path) as image:
            target = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
        with Image.open(mask_path) as image:
            mask = image.convert("L").resize((tile, tile), Image.Resampling.NEAREST)
        overlay = source.copy()
        red = Image.new("RGB", (tile, tile), (255, 0, 0))
        overlay = Image.composite(red, overlay, mask.point(lambda value: 110 if value else 0))
        canvas.paste(source, (origin_x, origin_y))
        canvas.paste(overlay, (origin_x + tile, origin_y))
        canvas.paste(target, (origin_x + tile * 2, origin_y))
        area = record["dynamic_candidate_mask"]["mask_geometry"]["area_ratio"]
        draw.text((origin_x + 2, origin_y + tile + 2), f"{record['pair_id'][-26:]} | {area:.3f}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    paired_path = output_dir / "gold40_source_target_paired_candidate_v4.jsonl"
    flat_path = output_dir / "gold40_branch_records_candidate_v4.jsonl"
    report_path = output_dir / "gold40_dynamic_mask_audit_v4.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory already contains files: {output_dir}; use --overwrite only after review")
    if args.overwrite and output_dir.exists():
        for path in sorted(output_dir.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    output_dir.mkdir(parents=True, exist_ok=True)

    v3_records = _read_jsonl(V3_MANIFEST)
    final_records = _read_jsonl(FINAL_MANIFEST)
    asset_records = _read_jsonl(ASSET_INDEX)
    v3_by_id = {str(record["pair_id"]): record for record in v3_records}
    final_by_id = {str(record["pair_id"]): record for record in final_records}
    assets_by_id = _asset_map(asset_records)
    if len(v3_by_id) != 40 or len(final_by_id) != 40 or len(assets_by_id) != 40:
        raise RuntimeError("Expected exactly 40 unique records in v3, final, and asset index")
    if set(v3_by_id) != set(final_by_id) or set(v3_by_id) != set(assets_by_id):
        raise RuntimeError("Gold40 ID sets differ across source manifests/assets")

    paired_records: List[Dict[str, Any]] = []
    flat_records: List[Dict[str, Any]] = []
    pair_audits: List[Dict[str, Any]] = []
    for pair_id in sorted(final_by_id):
        semantic_record = final_by_id[pair_id]
        asset_record = assets_by_id[pair_id]
        source_path = Path(asset_record["source"]["new_path"])
        target_path = Path(asset_record["target"]["new_path"])
        mask_path = output_dir / "dynamic_masks" / f"{pair_id}.png"
        try:
            semantic = _semantic_audit(semantic_record)
            mask_generation = create_candidate_mask(
                source_path,
                target_path,
                mask_path,
                threshold=MASK_THRESHOLD,
                dilation_px=MASK_DILATION_PX,
            )
            geometry = _mask_geometry(mask_path)
            raw_difference = _raw_difference_geometry(source_path, target_path)
            warnings = []
            if not mask_generation["within_recommended_range"]:
                warnings.append("candidate_mask_area_outside_recommended_range")
            if geometry["area_ratio"] > 0.25:
                warnings.append("large_candidate_mask_requires_manual_locality_review")
            if any(geometry[key] for key in ("touches_left", "touches_right", "touches_top", "touches_bottom")):
                warnings.append("candidate_mask_touches_image_border")
            paired = _candidate_record(
                semantic_record,
                asset_record,
                v3_by_id[pair_id],
                mask_path,
                {**mask_generation, **geometry},
                raw_difference,
                semantic,
            )
            paired_records.append(paired)
            flat_records.extend(_flat_branch_records(paired))
            pair_audits.append({
                "pair_id": pair_id,
                "status": "candidate_ready_manual_review_required",
                "warnings": warnings,
                "intervention_type": semantic_record["intervention"]["intervention_type"],
                "intervention_field": semantic_record["intervention"]["field"],
                "mask": paired["dynamic_candidate_mask"],
                "semantic_audit": paired["semantic_audit"],
            })
        except Exception as exc:  # noqa: BLE001
            pair_audits.append({"pair_id": pair_id, "status": "failed", "error": repr(exc)})

    failures = [audit for audit in pair_audits if audit["status"] == "failed"]
    if failures:
        _atomic_json(report_path, {
            "status": "failed",
            "created_at": _utc_now(),
            "num_pairs": len(final_by_id),
            "num_failed": len(failures),
            "pairs": pair_audits,
        })
        raise RuntimeError(f"Gold40 paired data preparation failed for {len(failures)} pair(s): {failures[:3]}")
    if len(paired_records) != 40 or len(flat_records) != 80:
        raise RuntimeError(f"Unexpected paired/branch record counts: {len(paired_records)}, {len(flat_records)}")
    _write_jsonl(paired_path, paired_records)
    _write_jsonl(flat_path, flat_records)
    _review_gallery(paired_records, output_dir / "review" / "gold40_dynamic_mask_contact_sheet.jpg")
    warning_counts = Counter(warning for audit in pair_audits for warning in audit["warnings"])
    type_counts = Counter(record["intervention"]["intervention_type"] for record in paired_records)
    report = {
        "status": "candidate_ready_manual_review_required",
        "created_at": _utc_now(),
        "input_manifests": {
            "v3_semantic_reference": {"path": str(V3_MANIFEST), "sha256": _sha256(V3_MANIFEST)},
            "gold40_final_semantic_and_asset_manifest": {"path": str(FINAL_MANIFEST), "sha256": _sha256(FINAL_MANIFEST)},
            "asset_index": {"path": str(ASSET_INDEX), "sha256": _sha256(ASSET_INDEX)},
        },
        "output": {
            "paired_manifest": str(paired_path),
            "branch_records": str(flat_path),
            "dynamic_mask_root": str(output_dir / "dynamic_masks"),
            "review_contact_sheet": str(output_dir / "review" / "gold40_dynamic_mask_contact_sheet.jpg"),
        },
        "counts": {
            "paired_records": len(paired_records),
            "branch_records": len(flat_records),
            "intervention_types": dict(sorted(type_counts.items())),
            "warning_counts": dict(sorted(warning_counts.items())),
        },
        "training_readiness": {
            "ready_for_training": False,
            "reason": "Masks are RGB-difference candidate masks. Every pair must receive explicit human review/approval before v4 Gold40 training.",
            "required_next_artifact": "review_decisions.jsonl with pair_id, approved, reviewer, reviewed_at, and notes",
        },
        "pairs": pair_audits,
    }
    _atomic_json(report_path, report)
    print(json.dumps({"status": report["status"], "paired_records": 40, "branch_records": 80, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
