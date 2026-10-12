#!/usr/bin/env python3
"""Create whole-source-scene 90/5/5 train/val/test splits for the final 991 dataset.

Splitting is done by ``source_scene_id`` so that no scene (and therefore no source
image) ever crosses a split boundary; the 991 pairs come from 813 distinct scenes and
some scenes contribute up to four pairs.

A deterministic scene-slice smoke manifest (2 scenes, 8 pairs) is emitted alongside
the splits so the two-rank smoke run can be validated before the formal run.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "artifacts" / "final991_source_target_automask_v5.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "splits"
DEFAULT_AUDIT = SCRIPT_DIR / "artifacts" / "final991_v5_split_audit.json"

SPLIT_NAMES = ("train", "val", "test")
SCENE_RATIOS = {"train": 0.90, "val": 0.05, "test": 0.05}
SMOKE_SCENES = 2
SMOKE_TARGET_PAIRS = 8
DEFAULT_SEED = 20260928
SCHEMA_VERSION = "gold40_automask_v5"


class ValidationError(RuntimeError):
    """The AutoMask-v5 manifest cannot be safely split by source scene."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise ValidationError(f"manifest is empty: {path}")
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


def _stable_scene_order(scene_ids: Iterable[str], seed: int) -> list[str]:
    """Deterministic shuffled scene order derived from scene id + seed."""
    def key(scene_id: str) -> str:
        return hashlib.sha256(f"{seed}:{scene_id}".encode("utf-8")).hexdigest()

    return sorted(set(scene_ids), key=key)


def _allocate(counts: Dict[str, int], total: int) -> Dict[str, int]:
    """Largest-remainder allocation so scene counts sum exactly to total."""
    raw = {name: counts[name] * SCENE_RATIOS[name] for name in SPLIT_NAMES}
    allocated = {name: int(raw[name] // 1) for name in SPLIT_NAMES}
    remainder = total - sum(allocated.values())
    order = sorted(SPLIT_NAMES, key=lambda name: (raw[name] - allocated[name]), reverse=True)
    for index in range(remainder):
        allocated[order[index % len(order)]] += 1
    if sum(allocated.values()) != total:
        raise ValidationError("scene allocation did not sum to the scene total")
    return allocated


def build(input_path: Path, output_dir: Path, audit_path: Path, seed: int, overwrite: bool) -> Dict[str, Any]:
    records = _read_jsonl(input_path)
    by_scene: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = str(record.get("pair_id", ""))
        scene_id = str(record.get("source_scene_id", ""))
        if not pair_id or not scene_id:
            raise ValidationError(f"record missing pair_id or source_scene_id: {record.get('pair_id')}")
        if str(record.get("macro_pair_id")) != f"dual_cot::{pair_id}":
            raise ValidationError(f"{pair_id}: invalid macro_pair_id")
        by_scene[scene_id].append(record)

    scene_ids = _stable_scene_order(by_scene.keys(), seed)
    total_scenes = len(scene_ids)
    allocated = _allocate({name: total_scenes for name in SPLIT_NAMES}, total_scenes)

    assignment: Dict[str, str] = {}
    cursor = 0
    for name in SPLIT_NAMES:
        for scene_id in scene_ids[cursor:cursor + allocated[name]]:
            assignment[scene_id] = name
        cursor += allocated[name]
    if len(assignment) != total_scenes:
        raise ValidationError("every scene must be assigned to exactly one split")

    splits: Dict[str, list[Dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for scene_id, name in assignment.items():
        splits[name].extend(by_scene[scene_id])
    for name in SPLIT_NAMES:
        splits[name].sort(key=lambda record: str(record["pair_id"]))

    # scene disjointness guard
    owned: Dict[str, set] = {name: {r["source_scene_id"] for r in splits[name]} for name in SPLIT_NAMES}
    for left in SPLIT_NAMES:
        for right in SPLIT_NAMES:
            if left < right and owned[left] & owned[right]:
                raise ValidationError(f"scene leakage between {left} and {right}")

    # deterministic smoke slice: whole scenes, closest to the requested pair count
    # Prefer scenes that contribute more pairs, mirroring the gold40 whole-scene
    # 4/8-pair smoke convention; ties break deterministically by hash.
    smoke_scene_ids: list[str] = []
    smoke_records: list[Dict[str, Any]] = []
    candidate_scenes = sorted(
        owned["train"],
        key=lambda scene_id: (-len(by_scene[scene_id]), _stable_scene_order([scene_id], seed + 1)[0]),
    )
    for scene_id in candidate_scenes:
        if len(smoke_records) >= SMOKE_TARGET_PAIRS:
            break
        smoke_scene_ids.append(scene_id)
        smoke_records.extend(sorted(by_scene[scene_id], key=lambda r: str(r["pair_id"])))
    smoke_records = smoke_records[:SMOKE_TARGET_PAIRS]
    smoke_records.sort(key=lambda record: str(record["pair_id"]))

    outputs = {name: output_dir / f"{name}.jsonl" for name in SPLIT_NAMES}
    outputs["smoke_train_8_pairs"] = output_dir / "smoke_train_8_pairs.jsonl"
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        raise ValidationError(f"split outputs already exist; pass --overwrite: {existing}")

    for name in SPLIT_NAMES:
        _write_jsonl(outputs[name], splits[name])
    _write_jsonl(outputs["smoke_train_8_pairs"], smoke_records)

    summary = {
        "created_at": _utc_now(),
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "input_manifest": str(input_path),
        "pair_count": len(records),
        "scene_count": total_scenes,
        "scene_ratios": SCENE_RATIOS,
        "scene_counts": {name: allocated[name] for name in SPLIT_NAMES},
        "pair_counts": {name: len(splits[name]) for name in SPLIT_NAMES},
        "smoke": {
            "scene_ids": smoke_scene_ids,
            "scene_count": len(smoke_scene_ids),
            "pair_count": len(smoke_records),
            "pair_ids": [str(r["pair_id"]) for r in smoke_records],
        },
        "intervention_type_counts_by_split": {
            name: dict(Counter(str(r.get("intervention_type", "")) for r in splits[name])) for name in SPLIT_NAMES
        },
        "asset_batch_counts_by_split": {
            name: dict(Counter(str(r.get("provenance", {}).get("asset_batch", "")) for r in splits[name]))
            for name in SPLIT_NAMES
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    _atomic_write_json(audit_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.input_manifest, args.output_dir, args.output_audit, args.seed, args.overwrite),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
