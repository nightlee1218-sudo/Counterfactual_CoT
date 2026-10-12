#!/usr/bin/env python3
"""Audit same-seed Gold40 source/target-CoT generation results.

Semantic success is intentionally read only from an explicit automated-audit label
manifest.  Pixel metrics are reference-image diagnostics, not semantic judges.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

APPROVED_MASK_STATUSES = {"approved", "approve", "accepted", "pass", "passed"}
PATH_KEYS = ("output_path", "image_path", "generated_image_path", "generated_path", "path")


def _read_records(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Manifest is empty: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            raise ValueError(f"JSON manifest must be a list or contain a records list: {path}")
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"All records must be JSON objects: {path}")
    return records


def _resolve_path(value: Any, base_dir: Path, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: missing path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _state(value: Any, context: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {"source": "source", "source_state": "source", "target": "target", "target_state": "target"}
    if normalized not in aliases:
        raise ValueError(f"{context}: state must be source/source_state or target/target_state, got {value!r}")
    return aliases[normalized]


def _required_pair_seed_state(record: Mapping[str, Any], context: str) -> Tuple[str, int, str]:
    pair_id = str(record.get("pair_id") or record.get("macro_pair_id") or "").strip()
    if not pair_id:
        raise ValueError(f"{context}: missing pair_id")
    if record.get("seed") is None:
        raise ValueError(f"{context}: missing seed")
    try:
        seed = int(record["seed"])
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context}: seed must be an integer") from error
    return pair_id, seed, _state(record.get("state") or record.get("source_or_target") or record.get("branch_id"), context)


def _output_path(record: Mapping[str, Any], base_dir: Path, context: str) -> Path:
    for key in PATH_KEYS:
        if record.get(key) is not None:
            path = _resolve_path(record[key], base_dir, f"{context}.{key}")
            if not path.is_file():
                raise FileNotFoundError(f"{context}: output image does not exist: {path}")
            return path
    raise ValueError(f"{context}: provide one of {', '.join(PATH_KEYS)}")


def _label_success(record: Mapping[str, Any], context: str) -> bool:
    # Do not silently substitute a visual proxy for this field.
    if "state_correct" not in record:
        raise ValueError(f"{context}: missing required explicit automated-audit boolean state_correct")
    value = record["state_correct"]
    if not isinstance(value, bool):
        raise ValueError(f"{context}: state_correct must be true or false, not {value!r}")
    return value


def _mapping(record: Mapping[str, Any], keys: Iterable[str]) -> Mapping[str, Any] | None:
    for key in keys:
        candidate = record.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return None


def _first_path(container: Mapping[str, Any], base_dir: Path) -> Path | None:
    for key in ("approved_mask_path", "mask_path", "path", "image_path"):
        if container.get(key) is not None:
            return _resolve_path(container[key], base_dir, f"mask.{key}")
    return None


def _approved_mask(record: Mapping[str, Any], base_dir: Path, allow_candidate: bool, context: str) -> Tuple[Path, str]:
    review = _mapping(record, ("reviewed_mask", "mask_review", "audit_mask", "dynamic_candidate_mask"))
    if review is None:
        review = record
    status = str(
        review.get("status")
        or review.get("review_status")
        or review.get("decision")
        or record.get("mask_review_status")
        or ""
    ).strip().lower()
    path = _first_path(review, base_dir)
    if path is None and review is not record:
        path = _first_path(record, base_dir)
    if path is None:
        raise ValueError(f"{context}: reviewed paired manifest has no mask path")
    if not path.is_file():
        raise FileNotFoundError(f"{context}: mask does not exist: {path}")
    if status not in APPROVED_MASK_STATUSES:
        if not allow_candidate:
            raise ValueError(
                f"{context}: mask status {status!r} is not approved. Supply a reviewed mask or use "
                "--allow-unreviewed-candidate-masks for a clearly labeled diagnostic-only run."
            )
        return path, "unreviewed_candidate_override"
    return path, "approved"


def _image_path(record: Mapping[str, Any], state: str, base_dir: Path, context: str) -> Path:
    endpoint = record.get(state)
    if isinstance(endpoint, Mapping):
        for key in ("image", "target_image", "path"):
            if endpoint.get(key) is not None:
                path = _resolve_path(endpoint[key], base_dir, f"{context}.{state}.{key}")
                if not path.is_file():
                    raise FileNotFoundError(f"{context}: reference image does not exist: {path}")
                return path
    branches = record.get("paired_branches")
    desired_state = "source" if state == "source" else "target"
    if isinstance(branches, list):
        for branch in branches:
            if not isinstance(branch, Mapping):
                continue
            branch_state = _state(branch.get("target_state") or branch.get("state") or branch.get("branch_id"), context)
            if branch_state == desired_state:
                for key in ("target_image", "image", "path"):
                    if branch.get(key) is not None:
                        path = _resolve_path(branch[key], base_dir, f"{context}.paired_branches.{key}")
                        if not path.is_file():
                            raise FileNotFoundError(f"{context}: reference image does not exist: {path}")
                        return path
    raise ValueError(f"{context}: cannot find {state} reference image")


def _intervention_type(record: Mapping[str, Any]) -> str:
    intervention = record.get("intervention")
    if isinstance(intervention, Mapping) and intervention.get("intervention_type"):
        return str(intervention["intervention_type"])
    return str(record.get("intervention_type") or "unknown")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    if not mask.any():
        raise ValueError(f"Mask is empty: {path}")
    return mask


def _mae(output: np.ndarray, reference: np.ndarray, selector: np.ndarray) -> float | None:
    if output.shape != reference.shape or output.shape[:2] != selector.shape:
        raise ValueError(f"Image/mask shape mismatch: output={output.shape}, reference={reference.shape}, mask={selector.shape}")
    if not selector.any():
        return None
    return float(np.abs(output - reference)[selector].mean())


def _mean(values: Sequence[float | bool | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return float(sum(selected) / len(selected)) if selected else None


def _metric_row(source: Mapping[str, Any], target: Mapping[str, Any]) -> Dict[str, float | bool | None]:
    return {
        "source_state_accuracy": bool(source["state_correct"]),
        "target_state_accuracy": bool(target["state_correct"]),
        "same_seed_cot_switch_accuracy": bool(source["state_correct"] and target["state_correct"]),
        "source_roi_reference_mae": source["roi_reference_mae"],
        "target_roi_reference_mae": target["roi_reference_mae"],
        "source_outside_mask_preservation_mae": source["outside_mask_preservation_mae"],
        "target_outside_mask_preservation_mae": target["outside_mask_preservation_mae"],
    }


def _bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metric: str,
    samples: int,
    rng: random.Random,
    cluster_by: str,
) -> Dict[str, Any]:
    clusters: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cluster_id = str(row["pair_id"]) if cluster_by == "pair_id" else f"{row['pair_id']}::{row['seed']}"
        clusters[cluster_id].append(row)
    cluster_ids = sorted(clusters)
    observed_values = [row[metric] for row in rows if row.get(metric) is not None]
    result: Dict[str, Any] = {
        "estimate": _mean(observed_values),
        "n_pairs": len({row["pair_id"] for row in rows}),
        "n_pair_seed_observations": len(rows),
        "n_metric_observations": len(observed_values),
        "bootstrap_cluster": cluster_by,
        "n_bootstrap_clusters": len(cluster_ids),
    }
    if not observed_values:
        result.update({"ci95_low": None, "ci95_high": None})
        return result
    draws: List[float] = []
    for _ in range(samples):
        sampled_rows = [row for cluster_id in (rng.choice(cluster_ids) for _ in cluster_ids) for row in clusters[cluster_id]]
        values = [row[metric] for row in sampled_rows if row.get(metric) is not None]
        draws.append(float(sum(float(value) for value in values) / len(values)))
    draws.sort()
    result["ci95_low"] = draws[int(0.025 * (samples - 1))]
    result["ci95_high"] = draws[int(0.975 * (samples - 1))]
    return result


def _summarize(rows: Sequence[Mapping[str, Any]], samples: int, seed: int, cluster_by: str = "pair_id") -> Dict[str, Any]:
    metrics = tuple(_metric_row(rows[0], rows[0]).keys()) if rows else ()
    rng = random.Random(seed)
    return {
        "n_pairs": len({row["pair_id"] for row in rows}),
        "n_pair_seed_observations": len(rows),
        "metrics": {metric: _bootstrap(rows, metric, samples, rng, cluster_by) for metric in metrics},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-manifest", required=True, type=Path, help="JSON/JSONL records: pair_id, seed, state, output_path")
    parser.add_argument("--paired-manifest", required=True, type=Path, help="Reviewed Gold40 paired JSON/JSONL manifest")
    parser.add_argument("--automated-audit-labels", required=True, type=Path, help="JSON/JSONL labels: pair_id, seed, state, state_correct")
    parser.add_argument("--output", required=True, type=Path, help="Output report JSON")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--allow-unreviewed-candidate-masks", action="store_true", help="Diagnostic-only override; report marks mask metrics as candidate based")
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    return args


def main() -> None:
    args = _parse_args()
    generation_records = _read_records(args.generation_manifest)
    label_records = _read_records(args.automated_audit_labels)
    paired_records = _read_records(args.paired_manifest)

    generations: Dict[Tuple[str, int, str], Path] = {}
    for index, record in enumerate(generation_records, start=1):
        context = f"generation {args.generation_manifest}:{index}"
        key = _required_pair_seed_state(record, context)
        if key in generations:
            raise ValueError(f"Duplicate generation result for pair_id/seed/state: {key}")
        generations[key] = _output_path(record, args.generation_manifest.parent, context)

    labels: Dict[Tuple[str, int, str], bool] = {}
    for index, record in enumerate(label_records, start=1):
        context = f"automated label {args.automated_audit_labels}:{index}"
        key = _required_pair_seed_state(record, context)
        if key in labels:
            raise ValueError(f"Duplicate automated audit label for pair_id/seed/state: {key}")
        labels[key] = _label_success(record, context)
    if set(generations) != set(labels):
        missing_labels = sorted(set(generations) - set(labels))[:10]
        labels_without_generation = sorted(set(labels) - set(generations))[:10]
        raise ValueError(f"Generation/label keys must match. missing_labels={missing_labels}, labels_without_generation={labels_without_generation}")

    active_pair_ids = {key[0] for key in generations}
    pairs: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(paired_records, start=1):
        context = f"paired manifest {args.paired_manifest}:{index}"
        pair_id = str(record.get("pair_id") or record.get("macro_pair_id") or "").strip()
        if not pair_id:
            raise ValueError(f"{context}: missing pair_id")
        if pair_id not in active_pair_ids:
            continue
        if pair_id in pairs:
            raise ValueError(f"Duplicate paired-manifest pair_id: {pair_id}")
        mask_path, mask_status = _approved_mask(record, args.paired_manifest.parent, args.allow_unreviewed_candidate_masks, context)
        pairs[pair_id] = {
            "pair_id": pair_id,
            "intervention_type": _intervention_type(record),
            "source_reference_path": _image_path(record, "source", args.paired_manifest.parent, context),
            "target_reference_path": _image_path(record, "target", args.paired_manifest.parent, context),
            "mask_path": mask_path,
            "mask_status": mask_status,
        }

    missing_pairs = sorted({key[0] for key in generations} - set(pairs))
    if missing_pairs:
        raise ValueError(f"Generation manifest has pair_id absent from paired manifest: {missing_pairs[:10]}")

    artifacts: Dict[str, Dict[str, Any]] = {}
    for pair_id, pair in pairs.items():
        if not any(key[0] == pair_id for key in generations):
            continue
        source = _load_rgb(pair["source_reference_path"])
        target = _load_rgb(pair["target_reference_path"])
        mask = _load_mask(pair["mask_path"])
        if source.shape != target.shape or source.shape[:2] != mask.shape:
            raise ValueError(f"{pair_id}: source/target/mask shapes differ")
        artifacts[pair_id] = {"source": source, "target": target, "mask": mask}

    grouped: Dict[Tuple[str, int], Dict[str, Any]] = defaultdict(dict)
    for (pair_id, seed, state), output_path in generations.items():
        pair = pairs[pair_id]
        arrays = artifacts[pair_id]
        output = _load_rgb(output_path)
        reference = arrays[state]
        row = {
            "pair_id": pair_id,
            "seed": seed,
            "state": state,
            "output_path": str(output_path),
            "reference_path": str(pair[f"{state}_reference_path"]),
            "state_correct": labels[(pair_id, seed, state)],
            "roi_reference_mae": _mae(output, reference, arrays["mask"]),
            "outside_mask_preservation_mae": _mae(output, arrays["source"], ~arrays["mask"]),
        }
        grouped[(pair_id, seed)][state] = row

    rows: List[Dict[str, Any]] = []
    for (pair_id, seed), states in sorted(grouped.items()):
        if set(states) != {"source", "target"}:
            raise ValueError(f"pair_id={pair_id}, seed={seed}: require both source and target generations for same-seed switch accuracy")
        pair = pairs[pair_id]
        metrics = _metric_row(states["source"], states["target"])
        rows.append({
            "pair_id": pair_id,
            "seed": seed,
            "intervention_type": pair["intervention_type"],
            "mask_path": str(pair["mask_path"]),
            "mask_status": pair["mask_status"],
            "source": states["source"],
            "target": states["target"],
            **metrics,
        })

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_pair: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[str(row["intervention_type"])].append(row)
        by_pair[str(row["pair_id"])].append(row)
    report = {
        "status": "completed",
        "definition": {
            "same_seed_cot_switch_accuracy": "One pair_id + seed succeeds only when explicit automated labels mark both source_state and target_state correct.",
            "state_accuracy": "Mean of explicit automated-audit state_correct labels; this script never infers semantic correctness from pixels.",
            "roi_reference_mae": "Mean absolute RGB error in the reviewed allowed-to-change mask versus that branch's source/target reference image, scaled to [0, 1].",
            "outside_mask_preservation_mae": "Mean absolute RGB error outside the reviewed allowed-to-change mask versus the source reference image, scaled to [0, 1].",
            "bootstrap": "Overall and intervention-type percentile 95% CIs resample pair_id clusters with all same-seed observations retained; each pair report resamples its pair_id+seed observations.",
        },
        "inputs": {
            "generation_manifest": str(args.generation_manifest.resolve()),
            "paired_manifest": str(args.paired_manifest.resolve()),
            "automated_audit_labels": str(args.automated_audit_labels.resolve()),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "allow_unreviewed_candidate_masks": args.allow_unreviewed_candidate_masks,
        },
        "warnings": (["Mask MAE uses unreviewed candidate masks due to explicit override; do not treat it as reviewed-mask evidence."] if args.allow_unreviewed_candidate_masks else []),
        "overall": _summarize(rows, args.bootstrap_samples, args.bootstrap_seed),
        "by_intervention_type": {kind: _summarize(kind_rows, args.bootstrap_samples, args.bootstrap_seed) for kind, kind_rows in sorted(by_type.items())},
        "by_pair": {pair_id: _summarize(pair_rows, args.bootstrap_samples, args.bootstrap_seed, cluster_by="pair_seed") for pair_id, pair_rows in sorted(by_pair.items())},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
