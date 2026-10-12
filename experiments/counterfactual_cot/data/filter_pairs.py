from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable

from .schema import load_jsonl, write_jsonl

DEFAULT_THRESHOLDS = {
    "target_success": 1.0,
    "preservation_score": 0.8,
    "binding_score": 0.8,
    "locality_score": 0.7,
    "image_quality_score": 0.6,
}


def is_accepted(pair: Dict[str, Any], thresholds: Dict[str, float]) -> bool:
    quality = pair.get("quality", {})
    if not quality.get("accepted", False):
        return False
    for key, threshold in thresholds.items():
        value = float(quality.get(key, 0.0))
        if value < threshold:
            return False
    return True


def filter_pairs(records: Iterable[Dict[str, Any]], thresholds: Dict[str, float]) -> Iterable[Dict[str, Any]]:
    for pair in records:
        if is_accepted(pair, thresholds):
            yield pair


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter accepted counterfactual pairs by quality thresholds.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    for key, default in DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=float, default=default)
    args = parser.parse_args()
    thresholds = {key: getattr(args, key) for key in DEFAULT_THRESHOLDS}
    records = list(filter_pairs(load_jsonl(args.input), thresholds))
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} filtered pairs to {args.output}")


if __name__ == "__main__":
    main()
