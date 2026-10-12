"""Build a deduplicated flat source-scene registry from gold1000_framework_strict_v4.jsonl.

Input:  gold1000_framework_strict_v4.jsonl (1000 pair records, nested source/target)
Output: source_registry_v4.jsonl (one row per unique source_scene_id), with the flat
        field names expected by
        reruns/nanobanana_v3/scripts/generate_source_images_strict_v3.py
        (scene_id, template_family, prompt, cot, cot_text).

Usage:
    python3 build_source_registry_v4.py \
        --input gold1000_framework_strict_v4.jsonl \
        --output source_registry_v4.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    pairs = load_jsonl(args.input)
    seen: Dict[str, Dict[str, Any]] = {}
    consistency_errors: List[str] = []

    for pair in pairs:
        scene_id = str(pair["source_scene_id"])
        source = pair["source"]
        if scene_id not in seen:
            seen[scene_id] = {
                "schema_version": "1.0",
                "scene_id": scene_id,
                "template_family": pair.get("template_family", "unknown"),
                "prompt": source["prompt"],
                "cot": source["cot"],
                "cot_text": source["cot_text"],
            }
        else:
            expected = seen[scene_id]
            actual = {
                "template_family": pair.get("template_family", "unknown"),
                "prompt": source["prompt"],
                "cot": source["cot"],
                "cot_text": source["cot_text"],
            }
            if any(expected[key] != value for key, value in actual.items()):
                consistency_errors.append(scene_id)

    if consistency_errors:
        raise SystemExit(
            f"FATAL: {len(consistency_errors)} source_scene_id groups have inconsistent "
            f"source.cot_text across pairs, e.g. {consistency_errors[:5]}"
        )

    registry = sorted(seen.values(), key=lambda item: item["scene_id"])
    write_jsonl(args.output, registry)
    print(f"Wrote {len(registry)} unique source scenes (from {len(pairs)} pairs) to {args.output}")


if __name__ == "__main__":
    main()
