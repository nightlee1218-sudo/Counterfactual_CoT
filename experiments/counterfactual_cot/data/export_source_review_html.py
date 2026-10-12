#!/usr/bin/env python3
"""Build a scrollable HTML viewer for the strict-v3 source review CSV."""
from __future__ import annotations

import argparse
import csv
import html
import json
import posixpath
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True, type=Path, help="source_review_strict_v3.csv")
    parser.add_argument("--image-root", required=True, type=Path, help="Directory the CSV image paths are relative to")
    parser.add_argument("--output", required=True, type=Path, help="Output HTML path")
    args = parser.parse_args()

    with args.review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"no rows in {args.review_csv}")

    groups: dict = defaultdict(list)
    missing: list = []
    for row in rows:
        rel = str(row.get("image", "")).strip()
        if not rel or not (args.image_root / rel).is_file():
            missing.append(rel or str(row.get("scene_id", "?")))
        groups[str(row.get("template_family", ""))].append(row)

    sections: list = []
    for family in sorted(groups):
        group = sorted(groups[family], key=lambda r: str(r.get("scene_id", "")))
        sections.append(f"<h2>Template family: {html.escape(family)} ({len(group)} scenes)</h2>")
        for row in group:
            rel = str(row.get("image", "")).strip()
            try:
                uri = posixpath.relpath((args.image_root / rel).resolve(), args.output.resolve().parent)
            except ValueError:
                uri = (args.image_root / rel).resolve().as_uri()
            sections.append(
                "<section><img src='{uri}'><div class='meta'><h3>{scene}</h3>"
                "<pre>{cot}</pre>"
                "<details><summary>Source Prompt</summary><pre>{prompt}</pre></details>"
                "<p class='decision'>accepted = <b>{accepted}</b> | notes: {notes}</p></div></section>".format(
                    uri=html.escape(uri),
                    scene=html.escape(str(row.get("scene_id", ""))),
                    cot=html.escape(str(row.get("cot_text", ""))),
                    prompt=html.escape(str(row.get("prompt", ""))),
                    accepted=html.escape(str(row.get("accepted", ""))),
                    notes=html.escape(str(row.get("review_notes", ""))),
                )
            )
    document = (
        "<!doctype html><meta charset='utf-8'><title>Source Review strict-v3</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px;background:#f5f6f8}"
        "h2{margin-top:44px}section{background:white;padding:14px;margin:0 0 14px;border-radius:10px;display:flex;gap:14px}"
        "img{width:400px;height:400px;object-fit:contain;background:#eee;flex:none}"
        ".meta{max-width:620px;overflow:auto}h3{margin:0 0 8px;font-size:14px;word-break:break-all}"
        "pre{white-space:pre-wrap;font-size:11px}.decision{color:#b00;font-size:12px}</style>"
        + "".join(sections)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(json.dumps({
        "scenes": len(rows),
        "families": len(groups),
        "missing_images": len(missing),
        "missing_examples": missing[:10],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
