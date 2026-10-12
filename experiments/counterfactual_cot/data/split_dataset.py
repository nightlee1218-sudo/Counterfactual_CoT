from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable

from .schema import load_jsonl, write_jsonl


def assign_split(scene_id: str, template_family: str, train_ratio: float, val_ratio: float) -> str:
    if not scene_id:
        raise ValueError("scene_id is required for scene-level split assignment")
    key = f"{template_family}:{scene_id}"
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 10_000 / 10_000.0
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def split_records(records: Iterable[Dict], train_ratio: float, val_ratio: float):
    for pair in records:
        pair = dict(pair)
        scene_id = str(pair.get("scene_id", "")).strip()
        pair["split"] = assign_split(scene_id, str(pair.get("template_family", "unknown")), train_ratio, val_ratio)
        yield pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign deterministic scene-level train/val/test splits.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()
    records = list(split_records(load_jsonl(args.input), args.train_ratio, args.val_ratio))
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} split pairs to {args.output}")


if __name__ == "__main__":
    main()
