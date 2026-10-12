#!/usr/bin/env python3
"""Build leakage-free Gold40 train/val/test splits from an approved paired manifest.

Records are assigned as whole ``source_scene_id`` groups. Candidate or unreviewed
records are rejected before any output is written.
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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "artifacts" / "gold40_source_target_paired_reviewed_v4.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "artifacts" / "splits"
DEFAULT_SEED = 20260910
SPLIT_NAMES = ("train", "val", "test")


class ValidationError(RuntimeError):
    """The reviewed paired manifest cannot safely be used for a grouped split."""


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


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise ValidationError(f"Reviewed manifest does not exist or is not a file: {path}")
    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Reviewed manifest line {line_number} is not valid JSON: {exc}") from exc
        if not isinstance(record, dict):
            raise ValidationError(f"Reviewed manifest line {line_number} must be a JSON object")
        records.append(record)
    if not records:
        raise ValidationError(f"Reviewed manifest is empty: {path}")
    return records


def _required_text(record: Mapping[str, Any], field: str, pair_id: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{pair_id}: missing required non-empty {field!r}")
    return value.strip()


def _valid_reviewed_roi(value: Any, pair_id: str, field: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, Mapping):
        raise ValidationError(f"{pair_id}: dynamic_regions.{field} must be an approved ROI-mask object")
    path, digest = value.get("path"), value.get("sha256")
    if not isinstance(path, str) or not path.strip() or not isinstance(digest, str) or len(digest) != 64:
        raise ValidationError(f"{pair_id}: dynamic_regions.{field} requires path and SHA256")
    return True


def _validate_ready_record(record: Mapping[str, Any]) -> Tuple[str, str, str]:
    pair_id = _required_text(record, "pair_id", "<unknown pair>")
    scene_id = _required_text(record, "source_scene_id", pair_id)
    intervention_type = _required_text(record, "intervention_type", pair_id)
    if record.get("macro_pair_id") != f"dual_cot::{pair_id}":
        raise ValidationError(f"{pair_id}: full paired candidate macro_pair_id is missing or malformed")
    branches = record.get("paired_branches")
    if not isinstance(branches, list) or {branch.get("branch_id") for branch in branches if isinstance(branch, dict)} != {"source_state", "target_state"}:
        raise ValidationError(f"{pair_id}: full paired candidate branches are missing")

    review = record.get("review")
    if not isinstance(review, dict) or review.get("approved") is not True or review.get("status") != "approved":
        raise ValidationError(f"{pair_id}: review.status='approved' and review.approved=true are required; candidate/unreviewed records are forbidden")
    readiness = record.get("training_readiness")
    if not isinstance(readiness, dict) or readiness.get("ready_for_training") is not True:
        raise ValidationError(f"{pair_id}: training_readiness.ready_for_training=true is required")
    mask = record.get("dynamic_candidate_mask")
    if not isinstance(mask, dict) or mask.get("review_status") != "approved":
        raise ValidationError(f"{pair_id}: dynamic_candidate_mask.review_status must be 'approved'; unreviewed candidates are forbidden")
    regions = record.get("dynamic_regions")
    if not isinstance(regions, dict):
        raise ValidationError(f"{pair_id}: dynamic_regions is required")
    source_present = _valid_reviewed_roi(regions.get("source_changed_roi"), pair_id, "source_changed_roi")
    target_present = _valid_reviewed_roi(regions.get("target_changed_roi"), pair_id, "target_changed_roi")
    shared_present = _valid_reviewed_roi(regions.get("shared_changed_roi"), pair_id, "shared_changed_roi")
    if not (source_present or shared_present) or not (target_present or shared_present):
        raise ValidationError(f"{pair_id}: reviewed ROIs must support both source and target states")
    return pair_id, scene_id, intervention_type


def _group_records(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_pair_ids = set()
    for record in records:
        pair_id, scene_id, _ = _validate_ready_record(record)
        if pair_id in seen_pair_ids:
            raise ValidationError(f"Reviewed manifest contains duplicate pair_id: {pair_id}")
        seen_pair_ids.add(pair_id)
        grouped[scene_id].append(record)
    return dict(grouped)


def _split_counts(scene_count: int, requested: Tuple[int | None, int | None, int | None]) -> Dict[str, int]:
    if scene_count < 3:
        raise ValidationError(f"At least three source scenes are required, found {scene_count}")
    if all(value is None for value in requested):
        if scene_count == 10:
            return {"train": 6, "val": 2, "test": 2}
        train = round(scene_count * 0.6)
        val = round(scene_count * 0.2)
        train = max(1, min(train, scene_count - 2))
        val = max(1, min(val, scene_count - train - 1))
        return {"train": train, "val": val, "test": scene_count - train - val}
    if any(value is None for value in requested):
        raise ValidationError("Specify all of --train-scenes, --val-scenes, and --test-scenes together")
    counts = dict(zip(SPLIT_NAMES, requested))
    if any(value is None or value < 1 for value in counts.values()) or sum(counts.values()) != scene_count:
        raise ValidationError(f"Requested scene counts must be positive and sum to {scene_count}: {counts}")
    return {name: int(value) for name, value in counts.items()}


def _ranked_scenes(scene_ids: Iterable[str], seed: int) -> List[str]:
    return sorted(scene_ids, key=lambda scene_id: hashlib.sha256(f"{seed}:{scene_id}".encode("utf-8")).hexdigest())


def _type_counts(scene_ids: Iterable[str], groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> Counter[str]:
    return Counter(
        str(record["intervention_type"])
        for scene_id in scene_ids
        for record in groups[scene_id]
    )


def _partition_score(assignments: Mapping[str, Sequence[str]], groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> Tuple[float, float]:
    all_scenes = [scene_id for scene_ids in assignments.values() for scene_id in scene_ids]
    all_types = _type_counts(all_scenes, groups)
    total_records = sum(all_types.values())
    total_scenes = len(all_scenes)
    type_error = 0.0
    size_error = 0.0
    for split_name, scene_ids in assignments.items():
        scene_fraction = len(scene_ids) / total_scenes
        observed = _type_counts(scene_ids, groups)
        type_error += sum(abs(observed[type_name] - total * scene_fraction) for type_name, total in all_types.items())
        size_error += abs(sum(observed.values()) - total_records * scene_fraction)
    return (type_error, size_error)


def _best_small_partition(
    ranked_scenes: Sequence[str], counts: Mapping[str, int], groups: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, List[str]]:
    best: Tuple[Tuple[float, float], Tuple[Tuple[str, ...], ...], Dict[str, List[str]]] | None = None
    scene_set = set(ranked_scenes)
    for train in itertools.combinations(ranked_scenes, counts["train"]):
        remaining_after_train = sorted(scene_set - set(train))
        for val in itertools.combinations(remaining_after_train, counts["val"]):
            test = sorted(scene_set - set(train) - set(val))
            candidate = {"train": list(train), "val": list(val), "test": test}
            tie_break = (tuple(train), tuple(val), tuple(test))
            key = (_partition_score(candidate, groups), tie_break)
            if best is None or key < (best[0], best[1]):
                best = (key[0], key[1], candidate)
    assert best is not None
    return best[2]


def _greedy_partition(
    ranked_scenes: Sequence[str], counts: Mapping[str, int], groups: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, List[str]]:
    assignments = {name: [] for name in SPLIT_NAMES}
    remaining = dict(counts)
    for scene_id in ranked_scenes:
        candidates = []
        for split_name in SPLIT_NAMES:
            if not remaining[split_name]:
                continue
            trial = {name: list(scene_ids) for name, scene_ids in assignments.items()}
            trial[split_name].append(scene_id)
            candidates.append((_partition_score(trial, groups), -remaining[split_name], split_name))
        _, _, selected = min(candidates)
        assignments[selected].append(scene_id)
        remaining[selected] -= 1
    if any(remaining.values()):
        raise AssertionError(f"Internal split allocation failure: {remaining}")
    return assignments


def _assign_scenes(groups: Mapping[str, Sequence[Mapping[str, Any]]], counts: Mapping[str, int], seed: int) -> Dict[str, List[str]]:
    ranked = _ranked_scenes(groups, seed)
    # Exhaustive search makes the requested ten-scene 6/2/2 plan globally best for the
    # declared type-balance objective; larger datasets use the same deterministic objective greedily.
    if len(ranked) <= 12:
        return _best_small_partition(ranked, counts, groups)
    return _greedy_partition(ranked, counts, groups)


def _select_smoke_scenes(
    train_scenes: Sequence[str], groups: Mapping[str, Sequence[Mapping[str, Any]]], pair_count: int, seed: int
) -> List[str]:
    if pair_count < 1:
        raise ValidationError("--smoke-train-pairs must be a positive integer")
    ranked = _ranked_scenes(train_scenes, seed)
    if len(ranked) > 20:
        raise ValidationError("Smoke selection supports at most 20 training source scenes while preserving whole-scene groups")
    train_records = sum(len(groups[scene_id]) for scene_id in ranked)
    expected_types = _type_counts(ranked, groups)
    choices: List[Tuple[Tuple[float, Tuple[str, ...]], List[str]]] = []
    for count in range(1, len(ranked) + 1):
        for subset_tuple in itertools.combinations(ranked, count):
            if sum(len(groups[scene_id]) for scene_id in subset_tuple) != pair_count:
                continue
            observed = _type_counts(subset_tuple, groups)
            error = sum(abs(observed[type_name] - total * pair_count / train_records) for type_name, total in expected_types.items())
            choices.append(((error, subset_tuple), list(subset_tuple)))
    if not choices:
        sizes = sorted({len(groups[scene_id]) for scene_id in ranked})
        raise ValidationError(
            f"Cannot build an exactly {pair_count}-pair smoke split without splitting a source_scene_id; "
            f"training scene group sizes are {sizes}"
        )
    return min(choices, key=lambda item: item[0])[1]


def _records_for_split(
    split_name: str, scene_ids: Sequence[str], groups: Mapping[str, Sequence[Mapping[str, Any]]], seed: int
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for scene_id in sorted(scene_ids):
        for record in sorted(groups[scene_id], key=lambda item: str(item["pair_id"])):
            copied = copy.deepcopy(record)
            copied["split"] = split_name
            copied["split_provenance"] = {
                "builder": Path(__file__).name,
                "group_key": "source_scene_id",
                "seed": seed,
            }
            records.append(copied)
    return records


def _audit_split(
    scene_ids: Sequence[str], groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    records = [record for scene_id in scene_ids for record in groups[scene_id]]
    return {
        "source_scene_ids": sorted(scene_ids),
        "scene_count": len(scene_ids),
        "pair_count": len(records),
        "intervention_type_counts": dict(sorted(Counter(record["intervention_type"] for record in records).items())),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Approved full paired manifest.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-scenes", type=int)
    parser.add_argument("--val-scenes", type=int)
    parser.add_argument("--test-scenes", type=int)
    parser.add_argument("--smoke-train-pairs", type=int, help="Write an exact-size, whole-source-scene smoke train manifest.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_paths = {name: args.output_dir / f"{name}.jsonl" for name in SPLIT_NAMES}
    audit_path = args.output_dir / "split_audit.json"
    smoke_path = (
        args.output_dir / f"smoke_train_{args.smoke_train_pairs}_pairs.jsonl"
        if args.smoke_train_pairs is not None
        else None
    )
    planned_paths = [*output_paths.values(), audit_path, *( [smoke_path] if smoke_path else [])]
    existing = [path for path in planned_paths if path.exists()]
    if existing and not args.overwrite:
        raise ValidationError(f"Split output already exists; use --overwrite to replace: {existing}")

    records = _read_jsonl(args.input)
    groups = _group_records(records)
    counts = _split_counts(len(groups), (args.train_scenes, args.val_scenes, args.test_scenes))
    assignments = _assign_scenes(groups, counts, args.seed)
    assigned_scenes = [scene_id for split_scenes in assignments.values() for scene_id in split_scenes]
    if len(set(assigned_scenes)) != len(assigned_scenes) or set(assigned_scenes) != set(groups):
        raise AssertionError("Internal scene leakage or incomplete assignment detected")

    split_records = {
        name: _records_for_split(name, assignments[name], groups, args.seed)
        for name in SPLIT_NAMES
    }
    smoke_audit: Dict[str, Any] | None = None
    if smoke_path is not None:
        smoke_scenes = _select_smoke_scenes(assignments["train"], groups, args.smoke_train_pairs, args.seed)
        smoke_records = _records_for_split("train", smoke_scenes, groups, args.seed)
        for record in smoke_records:
            record["split_provenance"]["smoke_train"] = True
        if len(smoke_records) != args.smoke_train_pairs:
            raise AssertionError("Internal smoke selection count mismatch")
        _write_jsonl(smoke_path, smoke_records)
        smoke_audit = {"path": str(smoke_path), **_audit_split(smoke_scenes, groups)}

    for name, path in output_paths.items():
        _write_jsonl(path, split_records[name])
    audit = {
        "status": "complete",
        "created_at": _utc_now(),
        "input": {"path": str(args.input), "sha256": _sha256(args.input)},
        "parameters": {
            "group_key": "source_scene_id",
            "seed": args.seed,
            "requested_scene_counts": counts,
            "balance_objective": "minimize intervention_type deviation from each split's scene fraction",
        },
        "outputs": {name: str(path) for name, path in output_paths.items()},
        "splits": {name: _audit_split(assignments[name], groups) for name in SPLIT_NAMES},
        "leakage_check": {
            "passed": True,
            "unique_source_scenes": len(groups),
            "assigned_source_scenes": len(set(assigned_scenes)),
            "overlapping_source_scene_ids": [],
        },
        "smoke_train": smoke_audit,
    }
    _atomic_json(audit_path, audit)
    print(json.dumps({"status": "complete", "outputs": audit["outputs"], "audit": str(audit_path)}, indent=2))


if __name__ == "__main__":
    main()
