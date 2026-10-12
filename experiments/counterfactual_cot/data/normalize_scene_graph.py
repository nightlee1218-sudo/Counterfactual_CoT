from __future__ import annotations

import argparse
from pathlib import Path

from .schema import SceneGraph, load_jsonl, render_cot_text, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize scene graph fields and canonical CoT text.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    records = []
    for record in load_jsonl(args.input):
        scene = SceneGraph.from_dict(record["cot"])
        record = dict(record)
        record["cot"] = scene.to_dict()
        record["cot_text"] = render_cot_text(scene)
        records.append(record)
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} normalized scene graphs to {args.output}")


if __name__ == "__main__":
    main()
