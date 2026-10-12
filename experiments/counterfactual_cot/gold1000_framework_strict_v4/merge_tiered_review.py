#!/usr/bin/env python3
"""Merge filled decisions from the compact tiered review CSV into the full CSV.

Only non-empty decisions are copied, so a half-finished tiered review can be
merged repeatedly without losing earlier work in the full table.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

SCORE_FIELDS = (
    "target_success",
    "preservation_score",
    "binding_score",
    "locality_score",
    "image_quality_score",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiered-csv", required=True, type=Path)
    parser.add_argument("--full-csv", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None, help="Defaults to overwriting --full-csv")
    args = parser.parse_args()
    output = args.output_csv or args.full_csv

    with args.tiered_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        tiered = list(csv.DictReader(handle))
    with args.full_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        full = list(reader)

    decisions = {}
    for row in tiered:
        pair_id = str(row.get("pair_id", "")).strip()
        if not pair_id:
            continue
        has_score = any(str(row.get(field, "")).strip() for field in SCORE_FIELDS)
        has_verdict = str(row.get("accepted", "")).strip().lower() in ("true", "false", "1", "0", "yes", "no")
        has_notes = bool(str(row.get("review_notes", "")).strip())
        if has_score or has_verdict or has_notes:
            decisions[pair_id] = row

    merged_count = 0
    for row in full:
        source = decisions.get(str(row.get("pair_id", "")).strip())
        if not source:
            continue
        for field in (*SCORE_FIELDS, "accepted", "review_notes"):
            if field in fieldnames and str(source.get(field, "")).strip():
                row[field] = source[field]
        merged_count += 1

    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(full)

    filled = sum(
        1 for row in full
        if str(row.get("review_notes", "")).strip()
        or any(str(row.get(field, "")).strip() for field in SCORE_FIELDS)
    )
    print(f"merged {merged_count} tiered decisions; full table now has {filled}/{len(full)} reviewed rows -> {output}")


if __name__ == "__main__":
    main()
