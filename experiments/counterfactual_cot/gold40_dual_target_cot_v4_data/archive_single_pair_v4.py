#!/usr/bin/env python3
"""Freeze a read-only provenance manifest for the single-pair v4 proof of concept."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = (
    SCRIPT_DIR.parent
    / "outputs/cf_singlepair_fullL0-27_lora_r8_dualTargetCoT1024_fsdpFullShard_overfit_v4"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required proof artifact is missing: {path}")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def summarize_switch(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = metrics.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("switch_metrics.json has no non-empty rows list")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("step"), int):
            raise ValueError("switch_metrics.json row has no integer step")
        grouped.setdefault(row["step"], []).append(row)
    summary: dict[str, Any] = {}
    for step, step_rows in sorted(grouped.items()):
        paired = sum(row.get("paired_switch_success") is True for row in step_rows)
        left = sum(row.get("left_proxy_success") is True for row in step_rows)
        right = sum(row.get("right_proxy_success") is True for row in step_rows)
        summary[str(step)] = {
            "seeds": len(step_rows),
            "paired_switch_success": f"{paired}/{len(step_rows)}",
            "source_state_proxy_success": f"{left}/{len(step_rows)}",
            "target_state_proxy_success": f"{right}/{len(step_rows)}",
        }
    if "0" not in summary or "500" not in summary:
        raise ValueError(f"expected step 0 and step 500 in switch evidence, found {sorted(summary)}")
    return summary


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, help="Default: <experiment-dir>/single_pair_v4_provenance.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    experiment = args.experiment_dir.resolve()
    output = (args.output or experiment / "single_pair_v4_provenance.json").resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {output}; pass --overwrite after reviewing the diff")

    required = {
        "experiment_plan": experiment / "experiment_plan.json",
        "region_spec": experiment / "region_spec.json",
        "trainable_audit": experiment / "trainable_audit.json",
        "final_report": experiment / "final_report.json",
        "train_log": experiment / "train_log.jsonl",
        "switch_metrics": experiment / "evaluation_dual_target_cot_switch/analysis/switch_metrics.json",
        "step_0_evaluation": experiment / "evaluation_dual_target_cot_switch/step_000000/evaluation_report.json",
        "step_500_evaluation": experiment / "evaluation_dual_target_cot_switch/step_000500/evaluation_report.json",
    }
    artifacts = {name: record(path) for name, path in required.items()}
    checkpoints = sorted((experiment / "checkpoints").glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint files found: {experiment / 'checkpoints'}")
    metrics = read_json(required["switch_metrics"])
    report = read_json(required["final_report"])
    payload = {
        "schema_version": "single_pair_v4_provenance_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "read_only_source_dir": str(experiment),
        "artifacts": artifacts,
        "checkpoints": [record(path) for path in checkpoints],
        "switch_evidence": summarize_switch(metrics),
        "final_report_schema_keys": sorted(report.keys()),
        "interpretation_boundary": "This archive freezes a single-pair causal proof-of-concept only; it is not a held-out Gold40 generalization result.",
    }
    atomic_json(output, payload)
    print(json.dumps({"status": "archived", "output": str(output), "switch_evidence": payload["switch_evidence"]}, indent=2))


if __name__ == "__main__":
    main()
