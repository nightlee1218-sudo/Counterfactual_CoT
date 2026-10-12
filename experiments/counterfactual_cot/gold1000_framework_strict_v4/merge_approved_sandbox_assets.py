#!/usr/bin/env python3
"""Safely merge only explicitly approved sandbox assets into the dataset.

The original manifest/assets are backed up first.  No unlisted pair is touched.
This script also handles the approved n3_pants source separately and never
merges the rejected `functional_state_005` or the three retained tulip pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List

KEEP_OLD = {
    "g1000_bathroom_counter__rinsecup_soap_towel__functional_state_005",
    "g1000_flower_shop__tulip_pot_ribbon__pattern_006",
    "g1000_flower_shop__tulip_pot_ribbon__pattern_007",
    "g1000_flower_shop__tulip_pot_ribbon__pattern_008",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def ids(path: Path) -> List[str]:
    return [line.strip() for line in path.open(encoding="utf-8") if line.strip()]


def copy_asset(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".merge_tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True, type=Path)
    p.add_argument("--v4-sandbox", required=True, type=Path)
    p.add_argument("--v5-sandbox", required=True, type=Path)
    p.add_argument("--pattern-sandbox", required=True, type=Path)
    p.add_argument("--n3-sandbox", required=True, type=Path)
    p.add_argument("--v4-list", required=True, type=Path)
    p.add_argument("--v5-list", required=True, type=Path)
    p.add_argument("--pattern-list", required=True, type=Path)
    p.add_argument("--n3-list", required=True, type=Path)
    p.add_argument("--backup-dir", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = args.dataset_root.resolve()
    image_root = root / "source_images"
    manifest_path = root / "gold1000.generated.jsonl"
    old_records = {r["pair_id"]: r for r in load(manifest_path)}

    groups = [
        ("v4", args.v4_sandbox, ids(args.v4_list)),
        ("v5", args.v5_sandbox, ids(args.v5_list)),
        ("pattern", args.pattern_sandbox, ids(args.pattern_list)),
        ("n3", args.n3_sandbox, ids(args.n3_list)),
    ]
    selected: Dict[str, tuple[Path, dict]] = {}
    for label, sandbox, pair_ids in groups:
        sb_records = {r["pair_id"]: r for r in load(sandbox / "manifest.jsonl")}
        for pid in pair_ids:
            if pid in KEEP_OLD:
                raise RuntimeError(f"refusing to merge explicitly retained old pair: {pid}")
            if pid not in old_records:
                raise RuntimeError(f"unknown dataset pair: {pid}")
            if pid not in sb_records:
                raise RuntimeError(f"{label}: pair missing from sandbox manifest: {pid}")
            if pid in selected:
                raise RuntimeError(f"pair selected in multiple groups: {pid}")
            for rel in (Path("source") / f"{pid}__source.png", Path("target") / f"{pid}__target.png", Path("masks") / f"{pid}__mask.png"):
                if not (sandbox / rel).is_file():
                    raise RuntimeError(f"{label}: missing sandbox asset: {sandbox / rel}")
            selected[pid] = (sandbox, sb_records[pid])

    # n3 source is approved separately and must be merged for all three pairs.
    n3_scene = "closet_wardrobe__n3_pants__v0330"
    n3_source = args.n3_sandbox / "source" / "_by_scene" / f"cf_source_{n3_scene}__source.png"
    if not n3_source.is_file():
        raise RuntimeError(f"missing approved n3 source: {n3_source}")
    n3_members = [pid for pid, r in old_records.items() if r["source_scene_id"] == n3_scene]
    if set(n3_members) != set(ids(args.n3_list)):
        raise RuntimeError(f"n3 list must contain exactly shared scene members: {n3_members}")

    summary = {"selected_pairs": sorted(selected), "count": len(selected), "kept_old": sorted(KEEP_OLD), "n3_members": n3_members}
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    args.backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = args.backup_dir / stamp
    backup.mkdir()
    shutil.copy2(manifest_path, backup / manifest_path.name)
    for pid in sorted(selected):
        for rel in (Path("source") / f"{pid}__source.png", Path("target") / f"{pid}__target.png", Path("masks") / f"{pid}__mask.png"):
            src = image_root / rel
            if src.is_file():
                dst = backup / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    before = sha(manifest_path)
    merged_records = {pid: json.loads(json.dumps(r)) for pid, r in old_records.items()}
    changes = []
    for pid, (sandbox, sb_record) in selected.items():
        for rel in (Path("source") / f"{pid}__source.png", Path("target") / f"{pid}__target.png", Path("masks") / f"{pid}__mask.png"):
            copy_asset(sandbox / rel, image_root / rel)
        merged_records[pid]["source"]["image"] = str(Path("source") / f"{pid}__source.png")
        merged_records[pid]["target"]["image"] = str(Path("target") / f"{pid}__target.png")
        merged_records[pid]["target"]["mask"] = str(Path("masks") / f"{pid}__mask.png")
        merged_records[pid]["generation"] = {**merged_records[pid].get("generation", {}), "sandbox_merged": True,
                                               "sandbox_source": str(sandbox)}
        changes.append(pid)

    # Merge the user-approved n3 source and ensure all three shared pair views
    # point to exactly the same bytes; target/mask assets come from the n3 sandbox.
    scene_dst = image_root / "source" / "_by_scene" / f"cf_source_{n3_scene}__source.png"
    copy_asset(n3_source, scene_dst)
    for pid in n3_members:
        copy_asset(args.n3_sandbox / "source" / f"{pid}__source.png", image_root / "source" / f"{pid}__source.png")

    tmp_manifest = manifest_path.with_suffix(".merge_tmp.jsonl")
    with tmp_manifest.open("w", encoding="utf-8") as f:
        for pid in old_records:
            f.write(json.dumps(merged_records[pid], ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp_manifest, manifest_path)
    after = sha(manifest_path)
    report = {**summary, "backup": str(backup), "manifest_sha256_before": before, "manifest_sha256_after": after,
              "changed_pair_count": len(changes), "changed_pairs": sorted(changes)}
    (args.backup_dir / "last_merge.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
