from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate faithfulness diagnostic scores by condition.")
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    buckets = defaultdict(list)
    with args.scores.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            buckets[row["condition"]].append(float(row.get("score", 0.0)))
    summary = {condition: {"mean": sum(values) / len(values), "n": len(values)} for condition, values in buckets.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
