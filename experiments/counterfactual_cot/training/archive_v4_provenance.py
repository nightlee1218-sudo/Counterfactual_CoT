#!/usr/bin/env python3
"""Create a non-overwriting provenance manifest for a Gold40 v4 single-pair run.

The input run directory is treated as read-only.  This program reads and hashes
artifacts, then creates exactly one new JSON manifest via an exclusive create.
It never moves, deletes, renames, or modifies a training artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

MANIFEST_FORMAT = "gold40_v4_single_pair_provenance_v1"
REQUIRED_STEPS = (0, 500)
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
FINAL_REPORT_NAMES = ("final_report.json", "single_pair_overfit_report.json", "training_report.json")
TRAIN_LOG_NAMES = ("train_log.jsonl", "train.log", "training.log", "console.log")
PASS_VALUES = {"passed", "pass", "success", "succeeded", "ok"}


class ProvenanceError(RuntimeError):
    """An input run lacks unambiguous evidence required for provenance."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hash and archive evidence from a Gold40 v4 single-pair output directory."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Completed v4 run output directory (read-only input).")
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=list(REQUIRED_STEPS),
        help="Checkpoint steps to require and hash (default: 0 500).",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="STEP=PATH",
        help="Resolve an ambiguous checkpoint discovery, with PATH relative to --output-dir or absolute.",
    )
    parser.add_argument("--final-report", type=Path, help="Override final report path.")
    parser.add_argument("--train-log", type=Path, help="Override train log path.")
    parser.add_argument(
        "--switch-report",
        type=Path,
        help="Explicit structured 16-seed switch report. It is validated when supplied.",
    )
    parser.add_argument(
        "--require-switch-report",
        action="store_true",
        help="Fail unless a structured switch report is present and proves 16 successful seeds.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        help="New manifest path. Default: <output-dir>/provenance_manifest_v4.json. Existing files are never overwritten.",
    )
    args = parser.parse_args()
    if any(step < 0 for step in args.steps):
        parser.error("--steps values must be non-negative")
    if len(set(args.steps)) != len(args.steps):
        parser.error("--steps must not contain duplicates")
    return args


def _resolved(path: Path, output_dir: Path) -> Path:
    return path.resolve() if path.is_absolute() else (output_dir / path).resolve()


def _require_regular_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ProvenanceError(f"Missing {label}: {path}. Supply the correct path explicitly or produce the artifact before archiving.")
    return path


def _relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(role: str, path: Path, output_dir: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "role": role,
        "path": str(path),
        "relative_path": _relative_to_output(path, output_dir),
        "bytes": stat.st_size,
        "sha256": _sha256(path),
    }


def _choose_one(candidates: Iterable[Path], label: str, override: Path | None, output_dir: Path, guidance: str) -> Path:
    if override is not None:
        return _require_regular_file(_resolved(override, output_dir), label)
    unique = sorted({candidate.resolve() for candidate in candidates if candidate.is_file()})
    if not unique:
        raise ProvenanceError(f"Missing {label}. {guidance}")
    if len(unique) > 1:
        choices = ", ".join(str(path) for path in unique)
        raise ProvenanceError(f"Ambiguous {label}: found {choices}. {guidance}")
    return unique[0]


def _named_candidates(output_dir: Path, names: Sequence[str]) -> list[Path]:
    return [path for path in output_dir.rglob("*") if path.is_file() and path.name in names]


def _parse_checkpoint_overrides(values: Sequence[str], output_dir: Path) -> dict[int, Path]:
    resolved: dict[int, Path] = {}
    for value in values:
        step_text, separator, path_text = value.partition("=")
        if not separator or not step_text.isdigit() or not path_text:
            raise ProvenanceError(f"Invalid --checkpoint {value!r}; expected STEP=PATH, for example 500=checkpoints/step_000500.pt.")
        step = int(step_text)
        if step in resolved:
            raise ProvenanceError(f"Duplicate --checkpoint override for step {step}.")
        resolved[step] = _require_regular_file(_resolved(Path(path_text), output_dir), f"checkpoint for step {step}")
    return resolved


def _checkpoint_candidates(output_dir: Path, step: int) -> list[Path]:
    # Boundaries prevent step 50 from matching step 500.  Names intentionally
    # support historical `step_000500.pt` and `step_500_trainable.pt` variants.
    pattern = re.compile(rf"(?:^|[_-])(?:step|checkpoint)[_-]?0*{step}(?:$|[-_.])", re.IGNORECASE)
    return [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in CHECKPOINT_SUFFIXES and pattern.search(path.stem)
    ]


def _find_checkpoints(output_dir: Path, steps: Sequence[int], overrides: Mapping[int, Path]) -> list[tuple[int, Path]]:
    unknown = sorted(set(overrides) - set(steps))
    if unknown:
        raise ProvenanceError(f"--checkpoint supplied for steps not requested by --steps: {unknown}.")
    found: list[tuple[int, Path]] = []
    for step in steps:
        if step in overrides:
            found.append((step, overrides[step]))
            continue
        candidates = _checkpoint_candidates(output_dir, step)
        if not candidates:
            raise ProvenanceError(
                f"Missing checkpoint for step {step} under {output_dir}. "
                f"Expected a file such as checkpoints/step_{step:06d}.pt, or pass --checkpoint {step}=PATH."
            )
        if len(candidates) > 1:
            choices = ", ".join(str(path) for path in candidates)
            raise ProvenanceError(
                f"Ambiguous checkpoint for step {step}: found {choices}. "
                f"Pass --checkpoint {step}=PATH to select the evidence file."
            )
        found.append((step, candidates[0]))
    return found


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProvenanceError(f"Switch report is not valid JSON: {path}: {exc}") from exc


def _status_is_pass(value: Any) -> bool:
    return isinstance(value, str) and value.strip().casefold() in PASS_VALUES


def _seed_result_status(row: Mapping[str, Any]) -> bool | None:
    for key in ("success", "passed", "ok"):
        if key in row:
            return row[key] is True
    for key in ("status", "outcome", "result"):
        if key in row:
            return _status_is_pass(row[key])
    return None


def _seed_rows(payload: Any) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    for key in ("seed_results", "switch_results", "results", "runs", "records", "per_seed"):
        value = payload.get(key)
        if isinstance(value, list) and all(isinstance(row, Mapping) for row in value):
            seed_rows = [row for row in value if "seed" in row]
            if seed_rows:
                return seed_rows
    return []


def _is_switch_report(path: Path, payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    if _seed_rows(payload):
        return True
    names = {str(key).casefold() for key in payload}
    return "switch" in path.as_posix().casefold() and bool(names & {"status", "success", "passed", "seed_count", "successful_seed_count"})


def _validate_switch_report(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    rows = _seed_rows(payload)
    if not rows:
        raise ProvenanceError(
            f"Switch report {path} has no per-seed structured results. Expected `seed_results` (or `switch_results`) "
            "with 16 objects containing `seed` and `success: true` (or `status: passed`)."
        )
    seeds: list[int] = []
    failed: list[Any] = []
    for row in rows:
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ProvenanceError(f"Switch report {path} has non-integer seed in row: {row!r}")
        seeds.append(seed)
        if _seed_result_status(row) is not True:
            failed.append(seed)
    if len(seeds) != 16 or len(set(seeds)) != 16:
        raise ProvenanceError(
            f"Switch report {path} must contain exactly 16 unique seed rows; found {len(seeds)} rows and {len(set(seeds))} unique seeds."
        )
    if failed:
        raise ProvenanceError(f"Switch report {path} does not prove success for seeds: {failed}.")
    if isinstance(payload, Mapping) and "status" in payload and not _status_is_pass(payload["status"]):
        raise ProvenanceError(f"Switch report {path} has non-passing top-level status {payload['status']!r}.")
    return {"report_path": str(path), "seed_count": 16, "seeds": sorted(seeds), "all_successful": True}


def _discover_switch_reports(output_dir: Path) -> list[Path]:
    reports: list[Path] = []
    for path in output_dir.rglob("*.json"):
        if "switch" not in path.as_posix().casefold():
            continue
        name = path.name.casefold()
        if "report" in name:
            payload = _read_json(path)
        else:
            try:
                payload = _read_json(path)
            except ProvenanceError:
                continue
        if _is_switch_report(path, payload):
            reports.append(path.resolve())
    return sorted(set(reports))


def _switch_output_files(output_dir: Path) -> list[Path]:
    return sorted(
        {
            path.resolve()
            for path in output_dir.rglob("*")
            if path.is_file() and "switch" in path.relative_to(output_dir).as_posix().casefold()
        }
    )


def _write_new_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise ProvenanceError(
            f"Refusing to overwrite existing manifest: {path}. Choose a new --manifest-path to preserve the prior record."
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise ProvenanceError(f"--output-dir does not exist or is not a directory: {output_dir}")

    checkpoint_overrides = _parse_checkpoint_overrides(args.checkpoint, output_dir)
    checkpoints = _find_checkpoints(output_dir, args.steps, checkpoint_overrides)
    final_report = _choose_one(
        _named_candidates(output_dir, FINAL_REPORT_NAMES),
        "final report",
        args.final_report,
        output_dir,
        "Produce one of final_report.json, single_pair_overfit_report.json, or pass --final-report PATH.",
    )
    train_log = _choose_one(
        _named_candidates(output_dir, TRAIN_LOG_NAMES),
        "training log",
        args.train_log,
        output_dir,
        "Produce one of train_log.jsonl, train.log, training.log, console.log, or pass --train-log PATH.",
    )

    if args.switch_report is not None:
        switch_reports = [_require_regular_file(_resolved(args.switch_report, output_dir), "structured switch report")]
    else:
        switch_reports = _discover_switch_reports(output_dir)
    if args.require_switch_report and not switch_reports:
        raise ProvenanceError(
            "No structured switch report found. Write switch_evaluation_report.json with exactly 16 `seed_results` rows, "
            "each containing an integer `seed` and `success: true`, then rerun."
        )

    switch_validation = [_validate_switch_report(path) for path in switch_reports]
    artifacts = [_artifact(f"checkpoint_step_{step}", path, output_dir) for step, path in checkpoints]
    artifacts.extend((_artifact("final_report", final_report, output_dir), _artifact("train_log", train_log, output_dir)))
    artifact_paths = {entry["path"] for entry in artifacts}
    for path in _switch_output_files(output_dir):
        if str(path) not in artifact_paths:
            artifacts.append(_artifact("switch_output", path, output_dir))
            artifact_paths.add(str(path))
    for path in switch_reports:
        if str(path) not in artifact_paths:
            artifacts.append(_artifact("switch_report", path, output_dir))
            artifact_paths.add(str(path))

    manifest_path = _resolved(args.manifest_path, output_dir) if args.manifest_path else output_dir / "provenance_manifest_v4.json"
    payload = {
        "format": MANIFEST_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_output_dir": str(output_dir),
        "read_only_input_contract": "Input artifacts were only read and hashed; no input artifact was moved, deleted, renamed, or modified.",
        "requested_checkpoint_steps": args.steps,
        "artifacts": artifacts,
        "switch_evaluation": {
            "structured_reports_present": bool(switch_reports),
            "validated_reports": switch_validation,
            "switch_output_files_hashed": len(_switch_output_files(output_dir)),
        },
    }
    _write_new_manifest(manifest_path, payload)
    print(json.dumps({"status": "created", "manifest": str(manifest_path), "artifacts": len(artifacts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProvenanceError as exc:
        print(f"provenance validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
