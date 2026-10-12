"""Build a runnable subset from a partially generated source registry.

After a fail-fast generation run, some scenes are missing an image. This keeps
only the pairs whose source scene actually exists, so target/mask generation
can start immediately instead of waiting for the stragglers.

Usage:
    python3 build_ready_subset.py \
        --pairs smoke_30_pairs.jsonl \
        --source-registry smoke_source_registry.generated.jsonl \
        --image-root source_images \
        --output-pairs smoke_30_pairs.ready.jsonl \
        --output-registry smoke_source_registry.ready.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-pairs", required=True, type=Path)
    parser.add_argument("--output-registry", required=True, type=Path)
    args = parser.parse_args()

    pairs = load_jsonl(args.pairs)
    registry = load_jsonl(args.source_registry)
    ready: List[Dict[str, Any]] = []
    for item in registry:
        image = item.get("image")
        if not image:
            continue
        path = Path(image)
        if not path.is_absolute():
            path = args.image_root / path
        if path.exists() and path.stat().st_size > 0:
            ready.append(item)

    ready_ids = {str(item["scene_id"]) for item in ready}
    kept = [pair for pair in pairs if str(pair["source_scene_id"]) in ready_ids]
    missing = sorted({str(pair["source_scene_id"]) for pair in pairs} - ready_ids)

    write_jsonl(args.output_pairs, kept)
    write_jsonl(args.output_registry, ready)
    print(
        json.dumps(
            {
                "ready_scenes": len(ready_ids),
                "ready_pairs": len(kept),
                "missing_scenes": missing,
                "output_pairs": str(args.output_pairs),
                "output_registry": str(args.output_registry),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
