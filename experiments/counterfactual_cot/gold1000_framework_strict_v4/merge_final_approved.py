#!/usr/bin/env python3
"""Safely merge the approved regeneration rounds into the 1000-pair dataset.

Priority is intentional:
  n3_pants approved-source round > v5 source/target round > v4 round > pattern round.
The three retained original pairs and functional_state_005 are never merged.
Use --dry-run before the real merge.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

KEEP_OLD = {
    "g1000_bathroom_counter__rinsecup_soap_towel__functional_state_005",
    "g1000_flower_shop__tulip_pot_ribbon__pattern_006",
    "g1000_flower_shop__tulip_pot_ribbon__pattern_007",
    "g1000_flower_shop__tulip_pot_ribbon__pattern_008",
}
N3_IDS = {
    "g1000_closet_wardrobe__n3_pants__2_to_1_001",
    "g1000_closet_wardrobe__n3_pants__2_to_3_001",
    "g1000_closet_wardrobe__n3_pants__3_to_2_001",
}


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def read_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.open(encoding="utf-8") if line.strip()}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_atomic(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".merge_tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def asset_paths(image_root: Path, pair_id: str) -> Tuple[Path, Path, Path]:
    return (
        image_root / "source" / f"{pair_id}__source.png",
        image_root / "target" / f"{pair_id}__target.png",
        image_root / "masks" / f"{pair_id}__mask.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--v4-sandbox", required=True, type=Path)
    parser.add_argument("--v5-sandbox", required=True, type=Path)
    parser.add_argument("--pattern-sandbox", required=True, type=Path)
    parser.add_argument("--n3-sandbox", required=True, type=Path)
    parser.add_argument("--v4-list", required=True, type=Path)
    parser.add_argument("--v5-list", required=True, type=Path)
    parser.add_argument("--pattern-list", required=True, type=Path)
    parser.add_argument("--n3-list", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    image_root = root / "source_images"
    manifest_path = root / "gold1000.generated.jsonl"
    original_records = {r["pair_id"]: r for r in read_jsonl(manifest_path)}

    # Resolve overlaps by the latest approved round.
    v4 = read_ids(args.v4_list) - KEEP_OLD - N3_IDS
    v5 = read_ids(args.v5_list) - KEEP_OLD - N3_IDS
    pattern = read_ids(args.pattern_list) - KEEP_OLD
    n3 = read_ids(args.n3_list)
    if n3 != N3_IDS:
        raise RuntimeError(f"--n3-list must contain exactly {sorted(N3_IDS)}, got {sorted(n3)}")
    v4 -= v5
    groups = {"v4": (args.v4_sandbox.resolve(), v4), "v5": (args.v5_sandbox.resolve(), v5),
              "pattern": (args.pattern_sandbox.resolve(), pattern),
              "n3": (args.n3_sandbox.resolve(), n3)}

    selected: Dict[str, Tuple[str, Path, dict]] = {}
    for label, (sandbox, pair_ids) in groups.items():
        sb_manifest = {r["pair_id"]: r for r in read_jsonl(sandbox / "manifest.jsonl")}
        for pid in sorted(pair_ids):
            if pid not in original_records:
                raise RuntimeError(f"{label}: unknown pair_id {pid}")
            if pid not in sb_manifest:
                raise RuntimeError(f"{label}: {pid} absent from {sandbox / 'manifest.jsonl'}")
            if pid in selected:
                raise RuntimeError(f"duplicate selected pair {pid}")
            for path in asset_paths(sandbox, pid):
                if not path.is_file():
                    raise RuntimeError(f"{label}: missing asset {path}")
            selected[pid] = (label, sandbox, sb_manifest[pid])

    # Determine which shared source scenes will change. Pattern-only rounds do
    # not change source. For v4/v5, every member of a changed scene must be in
    # the selected set, otherwise source/target consistency would be violated.
    scene_members: Dict[str, set[str]] = defaultdict(set)
    for pid, record in original_records.items():
        scene_members[record["source_scene_id"]].add(pid)
    scene_updates: Dict[str, Path] = {}
    for pid, (label, sandbox, _) in selected.items():
        if label == "pattern":
            continue
        old_source = image_root / "source" / f"{pid}__source.png"
        new_source = sandbox / "source" / f"{pid}__source.png"
        if digest(old_source) == digest(new_source):
            continue
        scene = original_records[pid]["source_scene_id"]
        missing = scene_members[scene] - set(selected)
        if scene == "closet_wardrobe__n3_pants__v0330":
            missing = set()
        if missing:
            raise RuntimeError(f"source scene {scene} changed for {pid}, but unselected members remain: {sorted(missing)}")
        candidate = sandbox / "source" / "_by_scene" / f"cf_source_{scene}__source.png"
        scene_updates[scene] = candidate if candidate.is_file() else new_source

    summary = {
        "merge_order": ["n3", "v5", "v4", "pattern"],
        "selected_counts": {label: len(pair_ids) for label, (_, pair_ids) in groups.items()},
        "selected_total": len(selected),
        "scene_updates": sorted(scene_updates),
        "kept_original": sorted(KEEP_OLD),
        "selected_pairs": sorted(selected),
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = args.backup_dir.resolve() / stamp
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, backup / manifest_path.name)
    backup_assets = set()
    for pid in selected:
        backup_assets.update(asset_paths(image_root, pid))
    for scene in scene_updates:
        backup_assets.add(image_root / "source" / "_by_scene" / f"cf_source_{scene}__source.png")
    for source_path in sorted(backup_assets):
        if source_path.is_file():
            rel = source_path.relative_to(root)
            target = backup / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)

    merged = {pid: copy.deepcopy(record) for pid, record in original_records.items()}
    for pid, (label, sandbox, sb_record) in selected.items():
        # Copy all three assets for v4/v5/n3; pattern source remains original.
        _, old_target, old_mask = asset_paths(image_root, pid)
        if label != "pattern":
            old_source, _, _ = asset_paths(image_root, pid)
            copy_atomic(sandbox / "source" / f"{pid}__source.png", old_source)
        copy_atomic(sandbox / "target" / f"{pid}__target.png", old_target)
        copy_atomic(sandbox / "masks" / f"{pid}__mask.png", old_mask)
        # Preserve the original record identity but import approved CoT/prompt/
        # generation changes from the sandbox record.
        imported = copy.deepcopy(sb_record)
        imported["source"]["image"] = str(Path("source") / f"{pid}__source.png")
        imported["target"]["image"] = str(Path("target") / f"{pid}__target.png")
        imported["target"]["mask"] = str(Path("masks") / f"{pid}__mask.png")
        imported["generation"] = {**imported.get("generation", {}), "sandbox_merged": True, "sandbox_round": label}
        merged[pid] = imported

    for scene, source_path in scene_updates.items():
        copy_atomic(source_path, image_root / "source" / "_by_scene" / f"cf_source_{scene}__source.png")

    tmp_manifest = manifest_path.with_name(manifest_path.name + ".merge_tmp")
    with tmp_manifest.open("w", encoding="utf-8") as handle:
        for pid in original_records:
            handle.write(json.dumps(merged[pid], ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp_manifest, manifest_path)
    summary.update({"backup": str(backup), "manifest_sha256_before": digest(backup / manifest_path.name),
                    "manifest_sha256_after": digest(manifest_path)})
    (args.backup_dir.resolve() / "last_merge.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
