#!/usr/bin/env python3
"""Create pair-aligned source/target/mask aliases without duplicating image bytes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def link_or_replace(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--target-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    linked = 0
    missing: list[tuple[str, str, str]] = []
    for record in records:
        pair_id = str(record["pair_id"])
        paths = {
            "source": resolve(args.source_root, str(record.get("source", {}).get("image", ""))),
            "target": resolve(args.target_root, str(record.get("target", {}).get("image", ""))),
            "mask": resolve(args.target_root, str(record.get("target", {}).get("mask", ""))),
        }
        for role, source in paths.items():
            if not source.exists():
                missing.append((pair_id, role, str(source)))
                continue
            link_or_replace(source, args.output_dir / f"{pair_id}__{role}.png")
            linked += 1

    report: dict[str, Any] = {
        "pairs": len(records),
        "linked_files": linked,
        "missing_files": [
            {"pair_id": pair_id, "role": role, "path": path}
            for pair_id, role, path in missing
        ],
        "output_dir": str(args.output_dir),
        "link_type": "relative_symlink",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pair_review_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
