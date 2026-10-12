#!/usr/bin/env python3
"""Side-by-side OLD vs NEW comparison pages for the sandbox regeneration.

Shows, per pair, the original target next to the freshly generated pro target
(and both source versions when the shared scene was regenerated), plus the new
automatic mask.  Paginated with thumbnails so the browser never has to load
dozens of full-size 1024px PNGs at once.
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
    parser.add_argument("--sandbox", required=True, type=Path)
    parser.add_argument("--original-root", required=True, type=Path)
    parser.add_argument("--original-manifest", required=True, type=Path)
    parser.add_argument("--pair-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--page-size", type=int, default=10)
    args = parser.parse_args()

    sandbox: Path = args.sandbox
    origin: Path = args.original_root
    original = {r["pair_id"]: r for r in load_jsonl(args.original_manifest)}
    pair_ids = [l.strip() for l in args.pair_list.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Include pairs dragged in by a regenerated scene.
    sandbox_manifest = {r["pair_id"]: r for r in load_jsonl(sandbox / "manifest.jsonl")}
    items: List[Dict[str, Any]] = []
    for pid in pair_ids:
        record = original.get(pid)
        if not record:
            continue
        scene = record["source_scene_id"]
        src_new = sandbox / "source" / "_by_scene" / f"cf_source_{scene}__source.png"
        src_old = origin / "source" / "_by_scene" / f"cf_source_{scene}__source.png"
        source_changed = src_new.is_file() and src_old.is_file() and src_new.stat().st_ino != src_old.stat().st_ino
        if source_changed:
            for member, rec in original.items():
                if rec["source_scene_id"] == scene and member not in pair_ids:
                    items.append({"pair_id": member, "record": rec, "source_changed": True, "spillover": True})
        items.append({"pair_id": pid, "record": record, "source_changed": source_changed, "spillover": False})

    seen = set()
    ordered = [i for i in items if not (i["pair_id"] in seen or seen.add(i["pair_id"]))]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def render(chunk: List[Dict[str, Any]], number: int, total: int) -> str:
        cards = []
        for item in chunk:
            pid = item["pair_id"]
            rec = item["record"]
            scene = rec["source_scene_id"]
            figures = []
            if item["source_changed"]:
                figures.append(_cell(origin / "source" / "_by_scene" / f"cf_source_{scene}__source.png",
                                     "source OLD", args.output_dir, "srcold", pid))
                figures.append(_cell(sandbox / "source" / "_by_scene" / f"cf_source_{scene}__source.png",
                                     "source NEW", args.output_dir, "srcnew", pid))
            else:
                figures.append(_cell(origin / rec["source"]["image"], "source", args.output_dir, "src", pid))
            figures.append(_cell(origin / "target" / f"{pid}__target.png", "target OLD (flash/mixed)", args.output_dir, "told", pid))
            figures.append(_cell(sandbox / "target" / f"{pid}__target.png", "target NEW (pro)", args.output_dir, "tnew", pid))
            figures.append(_cell(sandbox / "masks" / f"{pid}__mask.png", "mask NEW", args.output_dir, "mnew", pid))
            meta = {
                "subtype": rec.get("subtype"),
                "intervention": rec.get("intervention", {}).get("intervention_type"),
                "field": rec.get("intervention", {}).get("field"),
                "old_value": rec.get("intervention", {}).get("old_value"),
                "new_value": rec.get("intervention", {}).get("new_value"),
                "spillover_from_source_regen": item.get("spillover", False),
                "old_review_status": rec.get("quality", {}).get("review_status"),
            }
            badge = "<span class='badge spill'>source 连坐</span>" if item.get("spillover") else ""
            cards.append(
                f"<section><h2>{html.escape(pid)}{badge}</h2>"
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
            "img{{width:180px;height:180px;object-fit:contain;background:#eee;border:1px solid #ddd}}"
            "pre{{white-space:pre-wrap;font-size:11px;margin:8px 0 0}}"
            ".missing{{width:180px;height:180px;display:flex;align-items:center;justify-content:center;"
            "background:#fdd;color:#900;font-size:13px}}"
            ".badge{{background:#d97706;color:#fff;font-size:11px;padding:2px 8px;border-radius:9px;margin-left:8px}}"
            ".nav{{display:inline-block;margin:0 8px 16px 0;padding:6px 14px;background:#fff;"
            "border:1px solid #ccc;border-radius:6px;text-decoration:none;color:#0366d6}}"
            ".current{{background:#eee;color:#333}}</style>"
            "<h1>OLD vs NEW (pro regeneration)</h1>".format(number)
            + "".join(nav) + "".join(cards) + "".join(nav)
        )

    pages = [ordered[i:i + args.page_size] for i in range(0, len(ordered), args.page_size)] or [[]]
    for number, chunk in enumerate(pages, start=1):
        (args.output_dir / f"page_{number:02d}.html").write_text(
            render(chunk, number, len(pages)), encoding="utf-8")
    (args.output_dir / "compare_index.json").write_text(
        json.dumps({"pairs": len(ordered), "pages": len(pages),
                    "pair_ids": [i["pair_id"] for i in ordered]}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "pairs": len(ordered), "pages": len(pages)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
