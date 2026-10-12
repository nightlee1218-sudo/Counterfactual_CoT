from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

REQUIRED_COLUMNS = ["pair_id", "method", "intervention_type", "ts", "ps", "ba", "ls", "iq"]


def load_scores(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing score columns: {missing}")
        return list(reader)


def aggregate(rows: List[Dict[str, str]], ps_threshold: float, ls_threshold: float) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        key = f"{row['method']}::{row['intervention_type']}"
        numeric = {metric: float(row[metric]) for metric in ["ts", "ps", "ba", "ls", "iq"]}
        numeric["grounded_success"] = float(
            numeric["ts"] >= 1.0 and numeric["ps"] >= ps_threshold and numeric["ba"] >= 1.0 and numeric["ls"] >= ls_threshold
        )
        buckets.setdefault(key, []).append(numeric)
    summary: Dict[str, Dict[str, float]] = {}
    for key, values in buckets.items():
        summary[key] = {metric: sum(item[metric] for item in values) / len(values) for metric in values[0]}
        summary[key]["n"] = float(len(values))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate TS/PS/BA/LS/IQ grounding scores.")
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ps-threshold", type=float, default=0.8)
    parser.add_argument("--ls-threshold", type=float, default=0.7)
    args = parser.parse_args()
    summary = aggregate(load_scores(args.scores), args.ps_threshold, args.ls_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
