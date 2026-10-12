from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List

from .schema import SceneGraph, canonical_json, load_jsonl, render_cot_text, write_jsonl


def normalize_record(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    scene = SceneGraph.from_dict(record["cot"])
    scene_json = scene.to_dict()
    scene_hash = hashlib.sha256(canonical_json(scene_json).encode("utf-8")).hexdigest()
    return {
        "schema_version": "1.0",
        "scene_id": str(record.get("scene_id") or record.get("sample_id") or f"scene_{index:06d}"),
        "template_family": str(record.get("template_family", "unknown")),
        "prompt": str(record["prompt"]),
        "cot": scene_json,
        "cot_text": str(record.get("cot_text") or render_cot_text(scene)),
        "image": str(record.get("image", "")),
        "scene_hash": scene_hash,
        "source_metadata": record.get("metadata", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a normalized source scene registry.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    normalized: List[Dict[str, Any]] = []
    seen_hashes = set()
    for index, record in enumerate(load_jsonl(args.input)):
        item = normalize_record(record, index)
        if item["scene_hash"] in seen_hashes:
            continue
        seen_hashes.add(item["scene_hash"])
        normalized.append(item)
    write_jsonl(args.output, normalized)
    print(f"Wrote {len(normalized)} source scenes to {args.output}")


if __name__ == "__main__":
    main()
