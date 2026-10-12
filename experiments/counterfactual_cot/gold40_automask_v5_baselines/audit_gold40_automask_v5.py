#!/usr/bin/env python3
"""Audit same-seed Gold40 AutoMask-v5 generations without using pixels as semantic labels.

The three inputs are deliberately separate: the v5 automatic-mask manifest supplies
reference images and AutoMask-v5 regions; the generation manifest supplies outputs;
and the semantic-label manifest supplies explicit automated boolean labels.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

PATH_KEYS = ("output_path", "image_path", "generated_image_path", "generated_path", "path")
V5_MARKERS = ("automask-v5", "automask_v5", "auto_mask_v5", "gold40_automask_v5")


def _records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Manifest is empty: {path}")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        value = value.get("records") if isinstance(value, dict) and isinstance(value.get("records"), list) else value
    else:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must be a JSON array/records list or JSONL objects")
    return value


def _path(value: Any, base: Path, context: str, required: bool = True) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        if required:
            raise ValueError(f"{context}: missing path")
        return None
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def _existing_path(row: Mapping[str, Any], base: Path, context: str, keys: Sequence[str] = PATH_KEYS) -> Path:
    for key in keys:
        if row.get(key) is not None:
            candidate = _path(row[key], base, f"{context}.{key}")
            assert candidate is not None
            if not candidate.is_file():
                raise FileNotFoundError(f"{context}: file does not exist: {candidate}")
            return candidate
    raise ValueError(f"{context}: require one of {', '.join(keys)}")


def _state(value: Any, context: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {"source": "source", "source_state": "source", "target": "target", "target_state": "target"}
    if normalized not in aliases:
        raise ValueError(f"{context}: state must be source/source_state or target/target_state, got {value!r}")
    return aliases[normalized]


def _key(row: Mapping[str, Any], context: str) -> tuple[str, int, str]:
    pair_id = str(row.get("pair_id") or row.get("macro_pair_id") or "").strip()
    if not pair_id or row.get("seed") is None:
        raise ValueError(f"{context}: require pair_id and seed")
    try:
        seed = int(row["seed"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: seed must be an integer") from error
    return pair_id, seed, _state(row.get("state") or row.get("source_or_target") or row.get("branch_id"), context)


def _semantic_correct(row: Mapping[str, Any], context: str) -> bool:
    # Explicit labels only. Do not infer this from any visual or pixel metric.
    present = [key for key in ("automated_semantic_correct", "semantic_correct", "state_correct") if key in row]
    if len(present) != 1 or not isinstance(row[present[0]], bool):
        raise ValueError(f"{context}: require exactly one explicit boolean automated_semantic_correct, semantic_correct, or state_correct")
    return row[present[0]]


def _mapping(row: Mapping[str, Any], keys: Sequence[str]) -> Mapping[str, Any] | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            return value
    return None


def _v5_mask(row: Mapping[str, Any], base: Path, context: str) -> tuple[Path, str]:
    candidate = _mapping(row, ("automatic_difference_mask", "auto_mask_v5", "automask_v5", "auto_mask", "automask", "mask")) or row
    marker = " ".join(str(value).lower() for value in (
        row.get("schema_version"), row.get("mask_generator"), row.get("mask_method"),
        candidate.get("schema_version"), candidate.get("generator"), candidate.get("method"), candidate.get("version"),
    ) if value is not None)
    if not any(token in marker for token in V5_MARKERS) and "auto_mask_v5" not in row and "automask_v5" not in row:
        raise ValueError(f"{context}: not an explicit AutoMask-v5 record; include schema_version/mask generator marker or auto_mask_v5")
    for key in ("auto_mask_path", "mask_path", "path", "image_path"):
        value = candidate.get(key) if key in candidate else row.get(key)
        if value is not None:
            path = _path(value, base, f"{context}.{key}")
            assert path is not None
            if not path.is_file():
                raise FileNotFoundError(f"{context}: automatic mask does not exist: {path}")
            return path, str(candidate.get("status") or row.get("auto_mask_status") or "automatic_unreviewed")
    raise ValueError(f"{context}: missing AutoMask-v5 mask path")


def _reference(row: Mapping[str, Any], state: str, base: Path, context: str) -> Path:
    endpoint = row.get(state)
    if isinstance(endpoint, Mapping):
        return _existing_path(endpoint, base, f"{context}.{state}", ("image", "target_image", "path", "image_path"))
    for key in (f"{state}_image", f"{state}_image_path", f"{state}_path"):
        if row.get(key) is not None:
            return _existing_path(row, base, context, (key,))
    branches = row.get("paired_branches")
    if isinstance(branches, list):
        for branch in branches:
            if isinstance(branch, Mapping) and _state(branch.get("state") or branch.get("target_state") or branch.get("branch_id"), context) == state:
                return _existing_path(branch, base, f"{context}.paired_branches", ("target_image", "image", "path", "image_path"))
    raise ValueError(f"{context}: cannot find {state} reference image")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    if not mask.any():
        raise ValueError(f"AutoMask-v5 mask is empty: {path}")
    return mask


def _mae(output: np.ndarray, reference: np.ndarray, selector: np.ndarray, context: str) -> float | None:
    if output.shape != reference.shape or output.shape[:2] != selector.shape:
        raise ValueError(f"{context}: image/mask shape mismatch: output={output.shape}, reference={reference.shape}, mask={selector.shape}")
    return float(np.abs(output - reference)[selector].mean()) if selector.any() else None


def _mean(values: Sequence[float | bool | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return float(sum(usable) / len(usable)) if usable else None


def _metric_row(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, float | bool | None]:
    return {
        "source_accuracy": bool(source["semantic_correct"]),
        "target_accuracy": bool(target["semantic_correct"]),
        "same_seed_switch_accuracy": bool(source["semantic_correct"] and target["semantic_correct"]),
        "source_edit_mae": source["edit_mae"],
        "target_edit_mae": target["edit_mae"],
        "edit_mae": _mean((source["edit_mae"], target["edit_mae"])),
        "source_outside_mae": source["outside_mae"],
        "target_outside_mae": target["outside_mae"],
        "outside_mae": _mean((source["outside_mae"], target["outside_mae"])),
    }


def _bootstrap(rows: Sequence[Mapping[str, Any]], metric: str, samples: int, rng: random.Random, cluster_by: str) -> dict[str, Any]:
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cluster = str(row["pair_id"]) if cluster_by == "pair_id" else f"{row['pair_id']}::{row['seed']}"
        clusters[cluster].append(row)
    values = [row[metric] for row in rows if row.get(metric) is not None]
    result: dict[str, Any] = {"estimate": _mean(values), "n_metric_observations": len(values), "n_pair_seed_observations": len(rows), "n_pairs": len({row['pair_id'] for row in rows}), "bootstrap_cluster": cluster_by, "n_bootstrap_clusters": len(clusters)}
    if not values:
        return {**result, "ci95_low": None, "ci95_high": None}
    ids = sorted(clusters)
    draws: list[float] = []
    for _ in range(samples):
        sampled = [row for cluster in (rng.choice(ids) for _ in ids) for row in clusters[cluster]]
        draw_values = [float(row[metric]) for row in sampled if row.get(metric) is not None]
        draws.append(sum(draw_values) / len(draw_values))
    draws.sort()
    return {**result, "ci95_low": draws[int(.025 * (samples - 1))], "ci95_high": draws[int(.975 * (samples - 1))]}


def _summary(rows: Sequence[Mapping[str, Any]], samples: int, seed: int, cluster_by: str = "pair_id") -> dict[str, Any]:
    metrics = tuple(_metric_row(rows[0], rows[0])) if rows else ()
    rng = random.Random(seed)
    return {"n_pairs": len({row['pair_id'] for row in rows}), "n_pair_seed_observations": len(rows), "metrics": {metric: _bootstrap(rows, metric, samples, rng, cluster_by) for metric in metrics}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-auto-manifest", required=True, type=Path, help="AutoMask-v5 JSON/JSONL with source/target references and automatic mask paths")
    parser.add_argument("--generation-manifest", required=True, type=Path, help="JSON/JSONL: pair_id, seed, state, output_path")
    parser.add_argument("--automated-semantic-label-manifest", required=True, type=Path, help="JSON/JSONL labels keyed by pair_id, seed, state")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")

    generations: dict[tuple[str, int, str], Path] = {}
    for index, row in enumerate(_records(args.generation_manifest), 1):
        context = f"generation {args.generation_manifest}:{index}"
        key = _key(row, context)
        if key in generations:
            raise ValueError(f"Duplicate generation key: {key}")
        generations[key] = _existing_path(row, args.generation_manifest.parent, context)
    labels: dict[tuple[str, int, str], bool] = {}
    for index, row in enumerate(_records(args.automated_semantic_label_manifest), 1):
        context = f"automated semantic label {args.automated_semantic_label_manifest}:{index}"
        key = _key(row, context)
        if key in labels:
            raise ValueError(f"Duplicate semantic-label key: {key}")
        labels[key] = _semantic_correct(row, context)
    if set(generations) != set(labels):
        raise ValueError(f"Generation and semantic-label keys must match; missing_labels={sorted(set(generations) - set(labels))[:10]}, orphan_labels={sorted(set(labels) - set(generations))[:10]}")

    active_pairs = {key[0] for key in generations}
    pairs: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(_records(args.v5_auto_manifest), 1):
        pair_id = str(row.get("pair_id") or row.get("macro_pair_id") or "").strip()
        if pair_id not in active_pairs:
            continue
        if not pair_id or pair_id in pairs:
            raise ValueError(f"v5 auto manifest {args.v5_auto_manifest}:{index}: missing or duplicate pair_id {pair_id!r}")
        context = f"v5 auto manifest {args.v5_auto_manifest}:{index}"
        mask_path, mask_status = _v5_mask(row, args.v5_auto_manifest.parent, context)
        pairs[pair_id] = {"pair_id": pair_id, "source_scene_id": str(row.get("source_scene_id") or "unknown"), "intervention_type": str(row.get("intervention_type") or (row.get("intervention") or {}).get("intervention_type") or "unknown"), "source_path": _reference(row, "source", args.v5_auto_manifest.parent, context), "target_path": _reference(row, "target", args.v5_auto_manifest.parent, context), "mask_path": mask_path, "mask_status": mask_status}
    missing = active_pairs - set(pairs)
    if missing:
        raise ValueError(f"Generation pair_ids absent from v5 auto manifest: {sorted(missing)[:10]}")

    artifacts: dict[str, dict[str, np.ndarray]] = {}
    for pair_id, pair in pairs.items():
        source, target, mask = _load_rgb(pair["source_path"]), _load_rgb(pair["target_path"]), _load_mask(pair["mask_path"])
        if source.shape != target.shape or source.shape[:2] != mask.shape:
            raise ValueError(f"{pair_id}: source/target/mask shapes differ")
        artifacts[pair_id] = {"source": source, "target": target, "mask": mask}
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for (pair_id, seed, state), output_path in generations.items():
        arrays = artifacts[pair_id]
        output = _load_rgb(output_path)
        grouped[(pair_id, seed)][state] = {"pair_id": pair_id, "seed": seed, "state": state, "output_path": str(output_path), "semantic_correct": labels[(pair_id, seed, state)], "edit_mae": _mae(output, arrays[state], arrays["mask"], f"{pair_id}/{seed}/{state}"), "outside_mae": _mae(output, arrays["source"], ~arrays["mask"], f"{pair_id}/{seed}/{state}")}
    rows: list[dict[str, Any]] = []
    for (pair_id, seed), states in sorted(grouped.items()):
        if set(states) != {"source", "target"}:
            raise ValueError(f"pair_id={pair_id}, seed={seed}: both source and target are required for same-seed SwitchAcc")
        pair = pairs[pair_id]
        rows.append({"pair_id": pair_id, "seed": seed, "source_scene_id": pair["source_scene_id"], "intervention_type": pair["intervention_type"], "auto_mask_path": str(pair["mask_path"]), "auto_mask_status": pair["mask_status"], "source": states["source"], "target": states["target"], **_metric_row(states["source"], states["target"])})
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["intervention_type"]].append(row); by_pair[row["pair_id"]].append(row)
    report = {"status": "completed", "schema_version": "gold40_automask_v5_audit_v1", "inputs": {"v5_auto_manifest": str(args.v5_auto_manifest.resolve()), "generation_manifest": str(args.generation_manifest.resolve()), "automated_semantic_label_manifest": str(args.automated_semantic_label_manifest.resolve()), "bootstrap_samples": args.bootstrap_samples, "bootstrap_seed": args.bootstrap_seed}, "definitions": {"source_accuracy": "Mean explicit automated semantic label for source state.", "target_accuracy": "Mean explicit automated semantic label for target state.", "same_seed_switch_accuracy": "Both explicit semantic labels are true for one pair_id and matched seed.", "edit_mae": "RGB MAE inside the AutoMask-v5 region against the corresponding source/target reference, scaled to [0,1].", "outside_mae": "RGB MAE outside the AutoMask-v5 region against the source reference, scaled to [0,1].", "semantic_guard": "No pixel metric is used as a semantic-success label.", "bootstrap": "Percentile 95% CI; overall/type resample pair_id clusters, pair summaries resample pair_id+seed observations."}, "overall": _summary(rows, args.bootstrap_samples, args.bootstrap_seed), "by_intervention_type": {kind: _summary(kind_rows, args.bootstrap_samples, args.bootstrap_seed) for kind, kind_rows in sorted(by_type.items())}, "by_pair": {pair_id: _summary(pair_rows, args.bootstrap_samples, args.bootstrap_seed, "pair_seed") for pair_id, pair_rows in sorted(by_pair.items())}, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output": str(args.output.resolve()), "n_pair_seed_observations": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
