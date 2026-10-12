from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

from .schema import load_jsonl, write_jsonl


TRUE_VALUES = {"1", "true", "yes", "y", "accepted"}


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in TRUE_VALUES


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply human source-image review CSV decisions to a source registry.")
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--output-registry", required=True, type=Path)
    args = parser.parse_args()

    decisions: Dict[str, Dict[str, str]] = {}
    with args.review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            scene_id = str(row.get("scene_id", "")).strip()
            if scene_id:
                decisions[scene_id] = row

    records = load_jsonl(args.source_registry)
    accepted_count = 0
    for record in records:
        scene_id = str(record.get("scene_id", ""))
        if scene_id not in decisions:
            continue
        row = decisions[scene_id]
        accepted = _as_bool(row.get("accepted", "false"))
        record["source_quality"] = {
            "accepted": accepted,
            "review_status": "accepted" if accepted else "rejected",
            "review_notes": str(row.get("review_notes", "")),
        }
        accepted_count += int(accepted)
    write_jsonl(args.output_registry, records)
    print(f"Applied {len(decisions)} source decisions; {accepted_count} accepted. Output: {args.output_registry}")


if __name__ == "__main__":
    main()
