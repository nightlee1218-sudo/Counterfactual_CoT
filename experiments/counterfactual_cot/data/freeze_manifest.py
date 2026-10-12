from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from .schema import canonical_json, load_jsonl, write_jsonl


def record_hash(record: Dict) -> str:
    return hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()


def freeze(records: List[Dict], output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=lambda item: str(item.get("pair_id", "")))
    by_split: Dict[str, List[Dict]] = {}
    for record in records:
        split = str(record.get("split", "train"))
        by_split.setdefault(split, []).append(record)
    for split, split_records in by_split.items():
        write_jsonl(output_dir / f"manifest_{split}.jsonl", split_records)
    combined = "\n".join(canonical_json(record) for record in records) + "\n"
    manifest_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    (output_dir / "manifest.sha256").write_text(manifest_hash + "\n", encoding="utf-8")
    report = {
        "manifest_sha256": manifest_hash,
        "num_records": len(records),
        "splits": {split: len(items) for split, items in sorted(by_split.items())},
        "record_hashes": {str(record.get("pair_id")): record_hash(record) for record in records},
    }
    (output_dir / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze accepted counterfactual manifest into split files and hash.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    report = freeze(load_jsonl(args.input), args.output_dir)
    print(json.dumps({"manifest_sha256": report["manifest_sha256"], "splits": report["splits"]}, indent=2))


if __name__ == "__main__":
    main()
