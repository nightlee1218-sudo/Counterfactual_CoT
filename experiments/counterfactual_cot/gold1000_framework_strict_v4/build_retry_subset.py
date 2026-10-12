"""Build a retry manifest from pairs that failed local image checks.

Reads a generated manifest and keeps only the pairs whose local checks flagged
a problem (e.g. no_visible_change), so they can be regenerated alone, optionally
with a stronger model, without touching the pairs that already passed.

Usage:
    python3 build_retry_subset.py \
        --manifest smoke_30_pairs.generated.jsonl \
        --output smoke_30_pairs.retry.jsonl \
        --include-flag no_visible_change \
        --include-flag global_rerender_suspected
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-flag", action="append", default=[],
                        help="Local check flag to retry; repeatable. Default: any failure.")
    parser.add_argument("--review-status", default=None,
                        help="Only retry pairs with this review_status (default: auto_failed).")
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    status = args.review_status or "auto_failed"
    selected: List[Dict[str, Any]] = []
    for row in rows:
        quality = row.get("quality", {})
        if quality.get("review_status") != status:
            continue
        failures = quality.get("local_checks", {}).get("failures", [])
        if args.include_flag and not (set(args.include_flag) & set(failures)):
            continue
        selected.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "input_pairs": len(rows),
        "retry_pairs": len(selected),
        "pair_ids": [row["pair_id"] for row in selected],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
