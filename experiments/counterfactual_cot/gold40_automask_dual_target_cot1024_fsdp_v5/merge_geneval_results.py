#!/usr/bin/env python3
"""Merge the official GenEval evaluator output back into per-item records.

``evaluate_images.py`` emits one row per image with a ``filename`` and a boolean
``correct``. This joins that with ``index.json`` (written by
``build_geneval_eval_input.py``) so downstream analysis sees the same unified
schema as the GEdit / WISE scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="results.jsonl from evaluate_images.py")
    parser.add_argument("--index", type=Path, required=True, help="index.json next to the imagedir")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    index_rows = json.loads(args.index.read_text(encoding="utf-8"))
    by_subfolder = {r["subfolder"]: r for r in index_rows}

    results = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    merged: list[dict[str, Any]] = []
    for row in results:
        subfolder = Path(row["filename"]).parent.parent.name
        meta = by_subfolder.get(subfolder)
        if meta is None:
            raise RuntimeError(f"no index entry for subfolder {subfolder}")
        merged.append(
            {
                "id": meta["item_id"],
                "bench": "geneval",
                "variant": meta["variant"],
                "seed": meta["seed"],
                "tag": meta["tag"],
                "correct": bool(row["correct"]),
                "reason": str(row.get("reason", "")),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    n = len(merged)
    summary = {
        "bench": "geneval",
        "evaluator": "official_mask2former_clip",
        "variant": merged[0]["variant"] if merged else None,
        "num_items": n,
        "overall": round(sum(r["correct"] for r in merged) / n, 4) if n else None,
        "by_tag": {},
    }
    tags = sorted({r["tag"] for r in merged})
    for tag in tags:
        rows = [r for r in merged if r["tag"] == tag]
        summary["by_tag"][tag] = round(sum(r["correct"] for r in rows) / len(rows), 4)
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
