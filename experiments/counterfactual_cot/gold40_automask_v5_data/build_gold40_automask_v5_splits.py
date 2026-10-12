#!/usr/bin/env python3
"""Create whole-source-scene 6/2/2 Gold40 AutoMask-v5 splits.

The input contract is the candidate-derived automatic-mask schema from
``build_gold40_automask_v5_manifest.py``.  Human-review fields are explicitly
forbidden: this splitter neither consumes nor fabricates reviewed ROI metadata.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "artifacts" / "gold40_source_target_automask_v5.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "splits"
EXPECTED_PAIR_COUNT = 40
EXPECTED_SCENE_COUNT = 10
SCENE_COUNTS = {"train": 6, "val": 2, "test": 2}
SPLIT_NAMES = tuple(SCENE_COUNTS)
SCHEMA_VERSION = "gold40_automask_v5"
MASK_PROVENANCE = {
    "kind": "automatic_rgb_difference_mask",
    "semantic_annotation": False,
    "state_specific": False,
}


class ValidationError(RuntimeError):
    """The AutoMask-v5 manifest cannot be safely split by source scene."""


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
        raise ValidationError(f"AutoMask-v5 manifest does not exist or is not a file: {path}")
    records: list[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"AutoMask-v5 manifest line {line_number} is invalid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValidationError(f"AutoMask-v5 manifest line {line_number} must be an object")
        records.append(record)
    if not records:
        raise ValidationError(f"AutoMask-v5 manifest is empty: {path}")
    return records


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{label} must be non-empty text")
    return value.strip()


def _digest(value: Any, label: str) -> str:
    value = _text(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValidationError(f"{label} must be a SHA256 digest")
    return value.lower()


def _validate_record(record: Mapping[str, Any]) -> tuple[str, str, str]:
    pair_id = _text(record.get("pair_id"), "pair_id")
    if record.get("macro_pair_id") != f"dual_cot::{pair_id}":
        raise ValidationError(f"{pair_id}: malformed macro_pair_id")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValidationError(f"{pair_id}: schema_version must be {SCHEMA_VERSION!r}")
    forbidden = [field for field in ("review", "human_review", "reviewed_mask", "dynamic_regions", "dynamic_candidate_mask") if field in record]
    if forbidden:
        raise ValidationError(f"{pair_id}: reviewed/candidate review schema is forbidden for AutoMask-v5 split input: {forbidden}")
    scene_id = _text(record.get("source_scene_id"), f"{pair_id}.source_scene_id")
    intervention_type = _text(record.get("intervention_type"), f"{pair_id}.intervention_type")
    intervention = record.get("intervention")
    if not isinstance(intervention, Mapping) or intervention.get("intervention_type") != intervention_type:
        raise ValidationError(f"{pair_id}: top-level intervention_type must match intervention.intervention_type")
    if record.get("mask_provenance") != MASK_PROVENANCE:
        raise ValidationError(f"{pair_id}: mask_provenance must be {MASK_PROVENANCE!r}")
    mask = record.get("automatic_difference_mask")
    if not isinstance(mask, Mapping):
        raise ValidationError(f"{pair_id}: automatic_difference_mask is required")
    mask_path = Path(_text(mask.get("path"), f"{pair_id}.automatic_difference_mask.path"))
    if not mask_path.is_file():
        raise ValidationError(f"{pair_id}: automatic mask does not exist: {mask_path}")
    if _sha256(mask_path) != _digest(mask.get("sha256"), f"{pair_id}.automatic_difference_mask.sha256"):
        raise ValidationError(f"{pair_id}: automatic mask SHA256 differs from manifest")
    readiness = record.get("training_readiness")
    if not isinstance(readiness, Mapping) or readiness.get("ready_for_training") is not True:
        raise ValidationError(f"{pair_id}: training_readiness.ready_for_training=true is required")
    if readiness.get("mode") != "automatic_difference_mask":
        raise ValidationError(f"{pair_id}: training_readiness.mode must be 'automatic_difference_mask'")
    if readiness.get("human_review_used") is not False:
        raise ValidationError(f"{pair_id}: training_readiness.human_review_used=false is required")
    return pair_id, scene_id, intervention_type


def _group_records(records: Sequence[Dict[str, Any]]) -> Dict[str, list[Dict[str, Any]]]:
    groups: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    pair_ids: set[str] = set()
    for record in records:
        pair_id, scene_id, _ = _validate_record(record)
        if pair_id in pair_ids:
            raise ValidationError(f"duplicate pair_id: {pair_id}")
        pair_ids.add(pair_id)
        groups[scene_id].append(record)
    if len(pair_ids) != EXPECTED_PAIR_COUNT:
        raise ValidationError(f"AutoMask-v5 manifest must have {EXPECTED_PAIR_COUNT} unique pairs; got {len(pair_ids)}")
    if len(groups) != EXPECTED_SCENE_COUNT:
        raise ValidationError(f"AutoMask-v5 manifest must have {EXPECTED_SCENE_COUNT} source scenes; got {len(groups)}")
    return dict(groups)


def _rank_scenes(scene_ids: Iterable[str], seed: int) -> list[str]:
    return sorted(scene_ids, key=lambda scene_id: hashlib.sha256(f"{seed}:{scene_id}".encode("utf-8")).hexdigest())


def _type_counts(scene_ids: Iterable[str], groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> Counter[str]:
    return Counter(str(record["intervention_type"]) for scene_id in scene_ids for record in groups[scene_id])


def _partition_score(assignments: Mapping[str, Sequence[str]], groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[float, float]:
    all_scenes = [scene_id for scene_ids in assignments.values() for scene_id in scene_ids]
    all_types = _type_counts(all_scenes, groups)
    total_pairs, total_scenes = sum(all_types.values()), len(all_scenes)
    type_error = pair_error = 0.0
    for split_name, scene_ids in assignments.items():
        fraction = len(scene_ids) / total_scenes
        observed = _type_counts(scene_ids, groups)
        type_error += sum(abs(observed[type_name] - count * fraction) for type_name, count in all_types.items())
        pair_error += abs(sum(observed.values()) - total_pairs * fraction)
    return type_error, pair_error


def _assign_scenes(groups: Mapping[str, Sequence[Mapping[str, Any]]], seed: int) -> Dict[str, list[str]]:
    ranked = _rank_scenes(groups, seed)
    best: tuple[tuple[float, float], tuple[tuple[str, ...], ...], Dict[str, list[str]]] | None = None
    scene_set = set(ranked)
    for train in itertools.combinations(ranked, SCENE_COUNTS["train"]):
        remaining = sorted(scene_set - set(train))
        for val in itertools.combinations(remaining, SCENE_COUNTS["val"]):
            test = sorted(scene_set - set(train) - set(val))
            candidate = {"train": list(train), "val": list(val), "test": test}
            tie_break = (tuple(train), tuple(val), tuple(test))
            key = (_partition_score(candidate, groups), tie_break)
            if best is None or key < (best[0], best[1]):
                best = (key[0], key[1], candidate)
    assert best is not None
    return best[2]


def _select_smoke_scenes(train_scenes: Sequence[str], groups: Mapping[str, Sequence[Mapping[str, Any]]], pair_count: int, seed: int) -> list[str]:
    if pair_count not in (4, 8):
        raise ValidationError("--smoke-train-pairs supports only whole-scene 4 or 8 pair smoke manifests")
    ranked = _rank_scenes(train_scenes, seed)
    train_pair_count = sum(len(groups[scene_id]) for scene_id in ranked)
    expected_types = _type_counts(ranked, groups)
    candidates: list[tuple[tuple[float, tuple[str, ...]], list[str]]] = []
    for scene_count in range(1, len(ranked) + 1):
        for subset in itertools.combinations(ranked, scene_count):
            if sum(len(groups[scene_id]) for scene_id in subset) != pair_count:
                continue
            observed = _type_counts(subset, groups)
            type_error = sum(abs(observed[name] - count * pair_count / train_pair_count) for name, count in expected_types.items())
            candidates.append(((type_error, subset), list(subset)))
    if not candidates:
        sizes = sorted({len(groups[scene_id]) for scene_id in ranked})
        raise ValidationError(f"cannot make a {pair_count}-pair whole-scene smoke split; train scene group sizes are {sizes}")
    return min(candidates, key=lambda item: item[0])[1]


def _records(split_name: str, scene_ids: Sequence[str], groups: Mapping[str, Sequence[Mapping[str, Any]]], seed: int, *, smoke: bool = False) -> list[Dict[str, Any]]:
    result: list[Dict[str, Any]] = []
    for scene_id in sorted(scene_ids):
        for record in sorted(groups[scene_id], key=lambda item: str(item["pair_id"])):
            copied = copy.deepcopy(record)
            copied["split"] = split_name
            copied["split_provenance"] = {
                "builder": Path(__file__).name,
                "group_key": "source_scene_id",
                "seed": seed,
                "whole_source_scene": True,
                "human_review_used": False,
                **({"smoke_train": True} if smoke else {}),
            }
            result.append(copied)
    return result


def _split_audit(scene_ids: Sequence[str], groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    records = [record for scene_id in scene_ids for record in groups[scene_id]]
    return {
        "source_scene_ids": sorted(scene_ids),
        "scene_count": len(scene_ids),
        "pair_count": len(records),
        "intervention_type_counts": dict(sorted(Counter(record["intervention_type"] for record in records).items())),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--smoke-train-pairs", type=int, choices=(4, 8), help="Optional exact-size, whole-source-scene train smoke manifest.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_paths = {name: args.output_dir / f"{name}.jsonl" for name in SPLIT_NAMES}
    audit_path = args.output_dir / "split_audit.json"
    smoke_path = args.output_dir / f"smoke_train_{args.smoke_train_pairs}_pairs.jsonl" if args.smoke_train_pairs else None
    planned = [*output_paths.values(), audit_path, *([smoke_path] if smoke_path else [])]
    existing = [path for path in planned if path.exists()]
    if existing and not args.overwrite:
        raise ValidationError(f"split output already exists; pass --overwrite to replace: {existing}")

    records = _read_jsonl(args.input)
    groups = _group_records(records)
    assignments = _assign_scenes(groups, args.seed)
    assigned = [scene_id for scene_ids in assignments.values() for scene_id in scene_ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(groups):
        raise AssertionError("internal source-scene overlap or omission")

    split_records = {name: _records(name, assignments[name], groups, args.seed) for name in SPLIT_NAMES}
    for name, path in output_paths.items():
        _write_jsonl(path, split_records[name])
    smoke_audit: Dict[str, Any] | None = None
    if smoke_path is not None:
        smoke_scenes = _select_smoke_scenes(assignments["train"], groups, args.smoke_train_pairs, args.seed)
        smoke_records = _records("train", smoke_scenes, groups, args.seed, smoke=True)
        if len(smoke_records) != args.smoke_train_pairs:
            raise AssertionError("internal smoke split count mismatch")
        _write_jsonl(smoke_path, smoke_records)
        smoke_audit = {"path": str(smoke_path), **_split_audit(smoke_scenes, groups)}

    audit = {
        "status": "complete",
        "created_at": _utc_now(),
        "input": {"path": str(args.input), "sha256": _sha256(args.input)},
        "schema": {"mask_provenance": MASK_PROVENANCE, "human_review_used": False},
        "parameters": {"group_key": "source_scene_id", "seed": args.seed, "scene_counts": SCENE_COUNTS, "whole_source_scene": True},
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "splits": {name: _split_audit(assignments[name], groups) for name in SPLIT_NAMES},
        "leakage_check": {"passed": True, "unique_source_scenes": len(groups), "assigned_source_scenes": len(set(assigned)), "overlapping_source_scene_ids": []},
        "smoke_train": smoke_audit,
    }
    _atomic_json(audit_path, audit)
    print(json.dumps({"status": "complete", "outputs": audit["outputs"], "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
