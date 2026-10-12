#!/usr/bin/env python3
"""Convert GenEval generations into the layout the official evaluator expects.

``eval/gen/geneval/evaluation/evaluate_images.py`` walks::

    <imagedir>/<subfolder>/metadata.jsonl      # a SINGLE json object, despite the name
    <imagedir>/<subfolder>/samples/<digits>.png

and skips any subfolder whose name is not ``isdigit()``. This script produces
exactly that layout from a generation manifest produced by
``mini_eval_generate.py``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

BAGEL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA = BAGEL_ROOT / "eval/gen/geneval/prompts/evaluation_metadata.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="generation manifest jsonl")
    parser.add_argument("--out", type=Path, required=True, help="imagedir for the official evaluator")
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=DEFAULT_METADATA,
        help="used to recover the authoritative include[] when the manifest lacks it",
    )
    args = parser.parse_args()

    official = {}
    if args.metadata_file.exists():
        for line in args.metadata_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                official[row["prompt"]] = row

    rows: list[dict[str, Any]] = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    index_map: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        meta = row.get("meta", {}) or {}
        source = official.get(row["prompt"], {})
        record = {
            "tag": meta.get("tag") or source.get("tag"),
            "include": meta.get("include") or source.get("include", []),
            "prompt": row["prompt"],
        }
        if record["tag"] is None:
            raise RuntimeError(f"missing tag for {row['id']}")
        folder = args.out / f"{index:05d}"
        samples = folder / "samples"
        samples.mkdir(parents=True, exist_ok=True)
        (folder / "metadata.jsonl").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        shutil.copyfile(row["output_path"], samples / "00000.png")
        index_map.append(
            {
                "subfolder": folder.name,
                "item_id": row["id"],
                "variant": row.get("variant"),
                "seed": row.get("seed"),
                "tag": record["tag"],
                "prompt": row["prompt"],
            }
        )

    (args.out / "index.json").write_text(
        json.dumps(index_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", "imagedir": str(args.out.resolve()), "n": len(index_map)}))


if __name__ == "__main__":
    main()
