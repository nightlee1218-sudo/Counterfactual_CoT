from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from .schema import Intervention, SceneGraph, apply_intervention, changed_scene_paths, load_jsonl


REQUIRED_QUALITY_FIELDS = (
    "target_success",
    "preservation_score",
    "binding_score",
    "locality_score",
    "image_quality_score",
    "accepted",
    "review_status",
)


def _resolve(root: Path, path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def _check_image(path: Path, name: str) -> Tuple[int, int]:
    if not path.exists():
        raise ValueError(f"Missing {name}: {path}")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size
    except Exception as exc:
        raise ValueError(f"Unreadable {name}: {path}: {exc}") from exc


def _check_mask(path: Path, target_size: Tuple[int, int]) -> None:
    if not path.exists():
        raise ValueError(f"Missing mask: {path}")
    with Image.open(path) as mask:
        mask = mask.convert("L")
        if mask.size != target_size:
            raise ValueError(f"Mask size {mask.size} does not match target image size {target_size}: {path}")
        extrema = mask.getextrema()
        if extrema == (0, 0):
            raise ValueError(f"Mask is empty: {path}")
        if extrema == (255, 255):
            raise ValueError(f"Mask covers the whole image: {path}")


def validate_pair(pair: Dict[str, Any], root: Path, require_files: bool = True, accepted_only: bool = False) -> List[str]:
    errors: List[str] = []
    pair_id = str(pair.get("pair_id", "<missing>"))
    scene_id = str(pair.get("scene_id", "")).strip()
    if not scene_id:
        errors.append(f"{pair_id}: missing scene_id")
    if not str(pair.get("source", {}).get("prompt", "")).strip():
        errors.append(f"{pair_id}: missing source.prompt")
    if not str(pair.get("target", {}).get("prompt", "")).strip():
        errors.append(f"{pair_id}: missing target.prompt")
    quality = pair.get("quality", {})
    for field in REQUIRED_QUALITY_FIELDS:
        if field not in quality:
            errors.append(f"{pair_id}: missing quality.{field}")
    if not str(quality.get("review_status", "")).strip():
        errors.append(f"{pair_id}: quality.review_status is empty")
    generation = pair.get("generation", {})
    if require_files:
        if generation.get("target_generation_mode") != "edit":
            errors.append(f"{pair_id}: generation.target_generation_mode must be 'edit'")
        for field in ("provider", "model", "target_request_id"):
            if not str(generation.get(field, "")).strip():
                errors.append(f"{pair_id}: missing generation.{field}")
        if not generation.get("request_ids"):
            errors.append(f"{pair_id}: generation.request_ids is empty")
    try:
        source_scene = SceneGraph.from_dict(pair["source"]["cot"])
        target_scene = SceneGraph.from_dict(pair["target"]["cot"])
        intervention = Intervention.from_dict(pair["intervention"])
        expected_target = apply_intervention(source_scene, intervention)
        if expected_target.to_dict() != target_scene.to_dict():
            errors.append(f"{pair_id}: target cot is not deterministic application of intervention")
        changes = changed_scene_paths(source_scene, target_scene)
        allowed = set(pair.get("preservation", {}).get("allowed_to_change", []))
        allowed.add(f"{intervention.entity_id}.{intervention.field}")
        allowed_path_fragments = {item.replace(".", ".") for item in allowed}
        for changed in changes:
            if "schema_version" in changed:
                continue
            if not any(token.split(".", 1)[-1] in changed or token in changed for token in allowed_path_fragments):
                errors.append(f"{pair_id}: unexpected CoT change at {changed}")
    except Exception as exc:
        errors.append(f"{pair_id}: semantic validation failed: {exc}")

    if accepted_only and not pair.get("quality", {}).get("accepted", False):
        errors.append(f"{pair_id}: sample is not accepted")

    if require_files:
        try:
            source_size = _check_image(_resolve(root, pair["source"].get("image", "")), "source image")
            target_size = _check_image(_resolve(root, pair["target"].get("image", "")), "target image")
            if source_size != target_size:
                errors.append(f"{pair_id}: source size {source_size} differs from target size {target_size}")
            mask_path = pair["target"].get("mask", "")
            if mask_path:
                _check_mask(_resolve(root, mask_path), target_size)
            else:
                errors.append(f"{pair_id}: missing target.mask")
        except Exception as exc:
            errors.append(f"{pair_id}: file validation failed: {exc}")
    return errors


def validate_records(records: List[Dict[str, Any]], root: Path, require_files: bool, accepted_only: bool) -> Dict[str, Any]:
    seen_pair_ids = set()
    seen_by_split: Dict[str, set] = {}
    scene_by_split: Dict[str, set] = {}
    errors: List[str] = []
    for pair in records:
        pair_id = str(pair.get("pair_id", ""))
        if not pair_id:
            errors.append("record missing pair_id")
            continue
        if pair_id in seen_pair_ids:
            errors.append(f"duplicate pair_id: {pair_id}")
        seen_pair_ids.add(pair_id)
        split = str(pair.get("split", "unassigned"))
        seen_by_split.setdefault(split, set()).add(pair_id)
        scene_id = str(pair.get("scene_id", "")).strip()
        if scene_id:
            scene_by_split.setdefault(split, set()).add(scene_id)
        errors.extend(validate_pair(pair, root=root, require_files=require_files, accepted_only=accepted_only))
    split_names = list(seen_by_split)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            overlap = seen_by_split[left] & seen_by_split[right]
            if overlap:
                errors.append(f"split overlap {left}/{right}: {sorted(overlap)[:5]}")
            scene_overlap = scene_by_split.get(left, set()) & scene_by_split.get(right, set())
            if scene_overlap:
                errors.append(f"scene_id split leakage {left}/{right}: {sorted(scene_overlap)[:5]}")
    return {"valid": not errors, "num_records": len(records), "num_errors": len(errors), "errors": errors}


def validate_split_files(manifests: List[Path], root: Path | None, require_files: bool, accepted_only: bool) -> Dict[str, Any]:
    errors: List[str] = []
    seen_pair_ids: Dict[str, Path] = {}
    seen_scene_ids: Dict[str, Path] = {}
    total_records = 0
    for manifest in manifests:
        manifest_root = root or manifest.parent
        records = load_jsonl(manifest)
        total_records += len(records)
        report = validate_records(records, manifest_root, require_files, accepted_only)
        errors.extend(f"{manifest.name}: {error}" for error in report["errors"])
        for pair in records:
            pair_id = str(pair.get("pair_id", "")).strip()
            scene_id = str(pair.get("scene_id", "")).strip()
            if pair_id:
                if pair_id in seen_pair_ids:
                    errors.append(f"pair_id appears in multiple manifests: {pair_id} ({seen_pair_ids[pair_id]} and {manifest})")
                seen_pair_ids[pair_id] = manifest
            if scene_id:
                if scene_id in seen_scene_ids and seen_scene_ids[scene_id] != manifest:
                    errors.append(f"scene_id appears in multiple manifests: {scene_id} ({seen_scene_ids[scene_id]} and {manifest})")
                seen_scene_ids[scene_id] = manifest
    return {"valid": not errors, "num_records": total_records, "num_errors": len(errors), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate counterfactual CoT manifest JSONL files.")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifests", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--no-file-check", action="store_true")
    parser.add_argument("--accepted-only", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if bool(args.manifest) == bool(args.manifests):
        raise SystemExit("Pass exactly one of --manifest or --manifests")

    if args.manifests:
        report = validate_split_files(args.manifests, args.root, not args.no_file_check, args.accepted_only)
    else:
        root = args.root or args.manifest.parent
        report = validate_records(load_jsonl(args.manifest), root, not args.no_file_check, args.accepted_only)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["valid"]:
        for error in report["errors"][:50]:
            print(error)
        raise SystemExit(1)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
