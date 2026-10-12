#!/usr/bin/env python3
"""Build a paginated, thumbnail-based human review site.

A single HTML page holding hundreds of full-size 1024px PNGs (about 1 MB each)
overloads the browser: concurrent image requests time out and most figures end
up as broken icons.  This builder avoids that by

  * rendering 384px JPEG thumbnails (about 30 KB each) and linking each one to
    the full-size asset, and
  * splitting the records into pages of ``--page-size`` cards with navigation.

The CSV stays in one piece so reviewers fill a single table.
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
from experiments.counterfactual_cot.data.review_export import export_review

try:
    from .build_tiered_review import build_tiers
except ImportError:  # executed as a plain script
    from build_tiered_review import build_tiers

THUMB_SIZE = 384


def _rel(absolute: Path, base: Path) -> str:
    try:
        return posixpath.relpath(absolute.resolve(), base.resolve())
    except ValueError:
        return absolute.resolve().as_uri()


def _thumb_path(image_root: Path, output_dir: Path, rel_asset: str) -> Path:
    stem = Path(rel_asset).with_suffix(".jpg")
    return output_dir / "thumbs" / stem


def _build_thumb(source_abs: Path, thumb_abs: Path) -> bool:
    try:
        thumb_abs.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source_abs) as image:
            image.thumbnail((THUMB_SIZE, THUMB_SIZE))
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            image.save(thumb_abs, "JPEG", quality=82)
        return True
    except Exception:  # noqa: BLE001 - fall back to the full-size asset
        return False


def _figure(row: Dict[str, Any], key: str, caption: str, image_root: Path, output_dir: Path) -> str:
    value = str(row.get(key, "")).strip()
    if not value:
        return (
            f"<figure><div class='missing'>&#9888; missing</div>"
            f"<figcaption>{caption}</figcaption></figure>"
        )
    absolute = image_root / value
    if not absolute.is_file():
        return (
            f"<figure><div class='missing'>&#9888; missing</div>"
            f"<figcaption>{caption}</figcaption></figure>"
        )
    full_href = _rel(absolute, output_dir)
    thumb_abs = _thumb_path(image_root, output_dir, value)
    if _build_thumb(absolute, thumb_abs):
        thumb_href = _rel(thumb_abs, output_dir)
    else:
        thumb_href = full_href
    return (
        f"<figure><a href='{html.escape(full_href)}' target='_blank'>"
        f"<img src='{html.escape(thumb_href)}' loading='lazy'></a>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def _page(records: List[Dict[str, Any]], rows_by_id: Dict[str, Dict[str, Any]], index: int, total: int,
          image_root: Path, output_dir: Path, page_name: str) -> str:
    cards: List[str] = []
    for record in records:
        pair_id = str(record.get("pair_id", ""))
        row = rows_by_id.get(pair_id, {})
        tier = record.get("review_tier", "")
        checks = record.get("quality", {}).get("local_checks") or {}
        badge = (
            f"<span class='badge'>{html.escape(tier)}</span>"
            if tier else ""
        )
        meta = json.dumps({
            "mask_area_ratio": checks.get("mask_area_ratio"),
            "changed_pixel_ratio": checks.get("changed_pixel_ratio"),
            "review_status": record.get("quality", {}).get("review_status"),
            "failures": checks.get("failures", []),
            "intervention_type": record.get("intervention", {}).get("intervention_type"),
            "subtype": record.get("subtype"),
        }, ensure_ascii=False)
        cards.append(
            "<section><h2>{pair}{badge}</h2><div class='images'>"
            "{source}{target}{mask}</div><pre>{meta}</pre></section>".format(
                pair=html.escape(pair_id),
                badge=badge,
                source=_figure(row, "source_image", "source", image_root, output_dir),
                target=_figure(row, "target_image", "target", image_root, output_dir),
                mask=_figure(row, "candidate_mask", "candidate mask", image_root, output_dir),
                meta=html.escape(meta),
            )
        )
    nav: List[str] = []
    if index > 1:
        nav.append(f"<a class='nav' href='page_{index - 1:02d}.html'>&larr; prev</a>")
    nav.append(f"<span class='nav current'>page {index} / {total}</span>")
    if index < total:
        nav.append(f"<a class='nav' href='page_{index + 1:02d}.html'>next &rarr;</a>")
    document = (
        "<!doctype html><meta charset='utf-8'><title>Review page {index}</title>"
        "<style>body{{font-family:Arial,sans-serif;margin:20px;background:#f5f6f8}}"
        "section{{background:white;padding:14px;margin:0 0 12px;border-radius:10px}}"
        "h2{{font-size:14px;margin:0 0 8px;word-break:break-all}}"
        ".images{{display:flex;gap:10px;flex-wrap:wrap}}figure{{margin:0}}"
        "img{{width:200px;height:200px;object-fit:contain;background:#eee;border:1px solid #ddd}}"
        "pre{{white-space:pre-wrap;font-size:11px;margin:8px 0 0}}"
        ".missing{{width:200px;height:200px;display:flex;align-items:center;justify-content:center;"
        "background:#fdd;color:#900;font-size:14px}}"
        ".badge{{background:#0366d6;color:#fff;font-size:11px;padding:2px 8px;border-radius:9px;margin-left:8px}}"
        ".nav{{display:inline-block;margin:0 8px 16px 0;padding:6px 14px;background:#fff;"
        "border:1px solid #ccc;border-radius:6px;text-decoration:none;color:#0366d6}}"
        ".current{{background:#eee;color:#333}}</style>"
        "<h1>Counterfactual CoT review</h1>"
        + "".join(nav)
        + "".join(cards)
        + "".join(nav)
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--tiered-only", action="store_true",
                        help="Paginate only the tiered review subset instead of the whole manifest.")
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    records = load_jsonl(args.manifest)
    if args.tiered_only:
        records, tiers = build_tiers(records, args.seed)
        n1, n2 = tiers["tier1"], tiers["tier1"] + tiers["tier2"]
        for index, record in enumerate(records):
            record["review_tier"] = (
                "tier1_must_review" if index < n1
                else ("tier2_should_review" if index < n2 else "tier3_sampled")
            )
    export_review(records, args.output_dir, args.image_root)  # writes review.csv + a single review.html
    (args.output_dir / "review.html").unlink(missing_ok=True)  # replaced by the paginated pages

    import csv

    with (args.output_dir / "review.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows_by_id = {str(row["pair_id"]): row for row in csv.DictReader(handle)}

    pages = [records[i:i + args.page_size] for i in range(0, len(records), args.page_size)] or [[]]
    for number, chunk in enumerate(pages, start=1):
        page_name = f"page_{number:02d}.html"
        (args.output_dir / page_name).write_text(
            _page(chunk, rows_by_id, number, len(pages), args.image_root, args.output_dir, page_name),
            encoding="utf-8",
        )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "records": len(records),
        "pages": len(pages),
        "page_size": args.page_size,
        "entry": str(args.output_dir / "page_01.html"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
