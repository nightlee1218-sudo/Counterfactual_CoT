"""Merge generated source-scene images back into the 1000-pair gold1000 manifest.

Input:
    --pairs           gold1000_framework_strict_v4.jsonl (1000 pair records)
    --source-registry output-registry produced by generate_source_images_strict_v3.py
                       (one row per unique source_scene_id, with an "image" field
                       relative to the source generation --image-root)

Output:
    --output          a merged manifest (1000 records) where every pair["source"]["image"]
                       points at a pair-aligned source image named
                       source/<pair_id>__source.png, so that source/target/mask share the
                       same <pair_id> prefix:
                           source/<pair_id>__source.png
                           target/<pair_id>__target.png
                           masks/<pair_id>__mask.png
                       The shared per-scene file stays in source/_by_scene/ and is recorded
                       in pair["source"]["scene_image"]; the per-pair file is a hardlink
                       (fallback: symlink, then copy), so no image bytes are duplicated.

Usage:
    python3 merge_source_images_v4.py \
        --pairs gold1000_framework_strict_v4.jsonl \
        --source-registry source_images_v4/source_registry_generated.jsonl \
        --image-root source_images_v4 \
        --output gold1000_framework_strict_v4.with_source_images.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _materialize_pair_source(scene_path: Path, pair_path: Path, link_mode: str) -> str:
    """Create <pair_id>__source.png pointing at the shared scene image."""
    if not scene_path.exists():
        raise FileNotFoundError(f"shared source image is missing: {scene_path}")
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    if pair_path.exists() or pair_path.is_symlink():
        if pair_path.is_symlink() or pair_path.stat().st_ino != scene_path.stat().st_ino:
            pair_path.unlink()
    if pair_path.exists():
        return link_mode
    if link_mode in {"hardlink", "auto"}:
        try:
            os.link(scene_path, pair_path)
            return "hardlink"
        except OSError:
            if link_mode == "hardlink":
                raise
    if link_mode in {"symlink", "auto"}:
        try:
            pair_path.symlink_to(scene_path.resolve())
            return "symlink"
        except OSError:
            if link_mode == "symlink":
                raise
    shutil.copy2(scene_path, pair_path)
    return "copy"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--image-root",
        required=True,
        type=Path,
        help="Image root that the registry 'image' paths are relative to; per-pair source files are written here.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["auto", "hardlink", "symlink", "copy"],
        default="auto",
        help="How the per-pair source file is attached to the shared scene image (default: auto).",
    )
    parser.add_argument(
        "--require-accepted",
        action="store_true",
        help="Fail if a source scene's source_quality.accepted is not true (use after human review).",
    )
    args = parser.parse_args()

    pairs = load_jsonl(args.pairs)
    registry = load_jsonl(args.source_registry)

    by_scene: Dict[str, Dict[str, Any]] = {}
    duplicate_ids: List[str] = []
    for item in registry:
        scene_id = str(item["scene_id"])
        if scene_id in by_scene:
            duplicate_ids.append(scene_id)
        by_scene[scene_id] = item
    if duplicate_ids:
        raise SystemExit(f"FATAL: duplicate scene_id in source registry: {sorted(set(duplicate_ids))[:10]}")

    pair_scene_ids = {str(pair["source_scene_id"]) for pair in pairs}
    missing_registry = sorted(pair_scene_ids - set(by_scene))
    orphan_registry = sorted(set(by_scene) - pair_scene_ids)
    if missing_registry or orphan_registry:
        raise SystemExit(
            f"FATAL: registry coverage mismatch: missing={missing_registry[:10]}, "
            f"orphan={orphan_registry[:10]}"
        )

    missing_images: List[str] = []
    not_accepted: List[str] = []
    plan_mismatches: List[str] = []
    link_stats: Dict[str, int] = {}
    merged: List[Dict[str, Any]] = []
    for pair in pairs:
        pair = dict(pair)
        pair_id = str(pair["pair_id"])
        scene_id = str(pair["source_scene_id"])
        scene = by_scene[scene_id]
        source = pair["source"]
        if any(scene.get(key) != expected for key, expected in {
            "template_family": pair.get("template_family", "unknown"),
            "prompt": source["prompt"],
            "cot": source["cot"],
            "cot_text": source["cot_text"],
        }.items()):
            plan_mismatches.append(scene_id)
        if not scene.get("image"):
            missing_images.append(scene_id)
        if args.require_accepted and not scene.get("source_quality", {}).get("accepted"):
            not_accepted.append(scene_id)
        if scene.get("image"):
            scene_relative = str(scene["image"])
            scene_path = Path(scene_relative)
            if not scene_path.is_absolute():
                scene_path = args.image_root / scene_path
            pair_relative = f"source/{pair_id}__source.png"
            pair_path = args.image_root / pair_relative
            try:
                used = _materialize_pair_source(scene_path, pair_path, args.link_mode)
            except (FileNotFoundError, OSError) as exc:
                missing_images.append(f"{pair_id}:{scene_id}:{exc}")
            else:
                link_stats[used] = link_stats.get(used, 0) + 1
                pair["source"] = dict(source)
                pair["source"]["image"] = pair_relative
                pair["source"]["scene_image"] = scene_relative
                pair["source"]["source_scene_id"] = scene_id
                merged.append(pair)

    if plan_mismatches or missing_images or not_accepted:
        raise SystemExit(
            f"FATAL: source registry is not merge-ready: plan_mismatches={sorted(set(plan_mismatches))[:10]}, "
            f"missing_images={sorted(set(missing_images))[:10]}, "
            f"not_accepted={sorted(set(not_accepted))[:10]}"
        )
    write_jsonl(args.output, merged)
    print(
        f"Wrote {len(merged)} pairs to {args.output} with verified source images; "
        f"naming=<pair_id>__source/__target/__mask; link_stats={link_stats}"
    )


if __name__ == "__main__":
    main()
