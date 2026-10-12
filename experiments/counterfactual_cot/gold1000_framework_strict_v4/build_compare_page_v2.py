#!/usr/bin/env python3
"""Three-way comparison pages: ORIGINAL vs previous-pro vs NEW.

Used after each regeneration round so the reviewer can see, side by side, what
the original (mixed/flash) target looked like, what the previous pro round
produced, and what the latest round produced.  Paginated with thumbnails.
"""
from __future__ import annotations

import argparse
import html
import json
import posixpath
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from experiments.counterfactual_cot.data.schema import load_jsonl

THUMB = 384


def _thumb(src: Path, dst: Path) -> str:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img.thumbnail((THUMB, THUMB))
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.save(dst, "JPEG", quality=82)
        return dst.name
    except Exception:  # noqa: BLE001
        return ""


def _cell(abs_path: Path, caption: str, out_dir: Path, tag: str, pid: str) -> str:
    if not abs_path.is_file():
        return f"<figure><div class='missing'>&#9888; missing</div><figcaption>{caption}</figcaption></figure>"
    thumb_name = _thumb(abs_path, out_dir / "thumbs" / f"{tag}_{pid}.jpg")
    thumb_href = f"thumbs/{thumb_name}" if thumb_name else str(abs_path)
    full_href = posixpath.relpath(abs_path.resolve(), out_dir.resolve())
    return (
        f"<figure><a href='{html.escape(full_href)}' target='_blank'>"
        f"<img src='{html.escape(thumb_href)}' loading='lazy'></a>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--original-manifest", required=True, type=Path)
    parser.add_argument("--previous-sandbox", required=True, type=Path, help="Earlier pro round, may be absent")
    parser.add_argument("--new-sandbox", required=True, type=Path)
    parser.add_argument("--pair-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--page-size", type=int, default=10)
    args = parser.parse_args()

    origin: Path = args.original_root
    previous: Path = args.previous_sandbox
    new: Path = args.new_sandbox

    def source_for(pid: str, record: Dict[str, Any]) -> Path:
        """When the pair's source scene was regenerated, show the NEW source:
        the target was edited from it, so it is the correct reference."""
        candidate = new / "source" / f"{pid}__source.png"
        if candidate.is_file():
            return candidate
        return origin / record["source"]["image"]
    original = {r["pair_id"]: r for r in load_jsonl(args.original_manifest)}
    pair_ids = [l.strip() for l in args.pair_list.read_text(encoding="utf-8").splitlines() if l.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def render(chunk: List[str], number: int, total: int) -> str:
        cards = []
        for pid in chunk:
            rec = original.get(pid)
            if not rec:
                continue
            figures = [_cell(source_for(pid, rec), "SOURCE (edit input)", args.output_dir, "src", pid)]
            figures.append(_cell(origin / "target" / f"{pid}__target.png", "ORIGINAL (flash/mixed)", args.output_dir, "orig", pid))
            figures.append(_cell(previous / "target" / f"{pid}__target.png", "ROUND-1 pro", args.output_dir, "prev", pid))
            figures.append(_cell(new / "target" / f"{pid}__target.png", "ROUND-2 NEW", args.output_dir, "new", pid))
            figures.append(_cell(new / "masks" / f"{pid}__mask.png", "mask NEW", args.output_dir, "mnew", pid))
            meta = {
                "subtype": rec.get("subtype"),
                "intervention_field": rec.get("intervention", {}).get("field"),
                "old_value": rec.get("intervention", {}).get("old_value"),
                "new_value": rec.get("intervention", {}).get("new_value"),
            }
            cards.append(
                f"<section><h2>{html.escape(pid)}</h2>"
                f"<div class='images'>{''.join(figures)}</div>"
                f"<pre>{html.escape(json.dumps(meta, ensure_ascii=False))}</pre></section>"
            )
        nav = []
        if number > 1:
            nav.append(f"<a class='nav' href='page_{number-1:02d}.html'>&larr; prev</a>")
        nav.append(f"<span class='nav current'>page {number} / {total}</span>")
        if number < total:
            nav.append(f"<a class='nav' href='page_{number+1:02d}.html'>next &rarr;</a>")
        return (
            "<!doctype html><meta charset='utf-8'><title>Compare page {}</title>"
            "<style>body{{font-family:Arial,sans-serif;margin:20px;background:#f5f6f8}}"
            "section{{background:white;padding:14px;margin:0 0 12px;border-radius:10px}}"
            "h2{{font-size:13px;margin:0 0 8px;word-break:break-all}}"
            ".images{{display:flex;gap:8px;flex-wrap:wrap}}figure{{margin:0}}"
            "img{{width:170px;height:170px;object-fit:contain;background:#eee;border:1px solid #ddd}}"
            "pre{{white-space:pre-wrap;font-size:11px;margin:8px 0 0}}"
            ".missing{{width:170px;height:170px;display:flex;align-items:center;justify-content:center;"
            "background:#fdd;color:#900;font-size:13px}}"
            ".nav{{display:inline-block;margin:0 8px 16px 0;padding:6px 14px;background:#fff;"
            "border:1px solid #ccc;border-radius:6px;text-decoration:none;color:#0366d6}}"
            ".current{{background:#eee;color:#333}}</style>"
            "<h1>ORIGINAL vs ROUND-1 pro vs ROUND-2 NEW</h1>".format(number)
            + "".join(nav) + "".join(cards) + "".join(nav)
        )

    pages = [pair_ids[i:i + args.page_size] for i in range(0, len(pair_ids), args.page_size)] or [[]]
    for number, chunk in enumerate(pages, start=1):
        (args.output_dir / f"page_{number:02d}.html").write_text(render(chunk, number, len(pages)), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "pairs": len(pair_ids), "pages": len(pages)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
