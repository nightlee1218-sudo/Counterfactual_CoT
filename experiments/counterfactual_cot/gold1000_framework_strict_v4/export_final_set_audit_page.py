#!/usr/bin/env python3
"""Export paginated audit pages for an arbitrary subset of the final training manifest.

The final manifest carries absolute paths, so this script copies local thumbnails
into the output directory and references them relatively. Read-only: it never
modifies, moves or deletes any dataset asset.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from PIL import Image

FIELDS = [
    "pair_id", "subtype", "asset_batch", "reason", "scene_id",
    "intervention", "old_value", "new_value",
    "source_abs", "target_abs", "mask_abs",
    "audit_decision", "audit_notes",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def thumb(src: Path, dst: Path, size: int = 460) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        with Image.open(src) as image:
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            image.thumbnail((size, size))
            image.save(dst, "JPEG", quality=84)
    return dst.name


def esc(value: object) -> str:
    return html.escape(str(value if value is not None else ""))


def build(manifest_rows: list[dict], out_dir: Path, page_size: int, title: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    thumbs = out_dir / "thumbs"
    cards: list[str] = []
    csv_rows: list[dict] = []

    for index, record in enumerate(manifest_rows, start=1):
        assets = record.get("training_assets", {})
        pair_id = record["pair_id"]
        source = Path(assets.get("source_abs", ""))
        target = Path(assets.get("target_abs", ""))
        mask = Path(assets.get("mask_abs", ""))
        ts = thumb(target, thumbs / f"{pair_id}__target.jpg")
        ss = thumb(source, thumbs / f"{pair_id}__source.jpg")
        ms = thumb(mask, thumbs / f"{pair_id}__mask.jpg") if mask.is_file() else ""
        intervention = record.get("intervention", {})
        reason = record.get("audit_reason", "")
        cards.append(f"""
<section class="card" id="p{index}">
  <div class="head">
    <span class="idx">#{index}</span>
    <span class="pid">{esc(pair_id)}</span>
    <span class="tag">{esc(record.get('asset_batch',''))}</span>
    <span class="sub">{esc(record.get('subtype',''))}</span>
  </div>
  <div class="reason">{esc(reason)}</div>
  <div class="grid">
    <figure><figcaption>SOURCE</figcaption><img src="thumbs/{ss}" loading="lazy"></figure>
    <figure class="hl"><figcaption>TARGET（审核对象）</figcaption><img src="thumbs/{ts}" loading="lazy"></figure>
    <figure><figcaption>MASK</figcaption><img src="thumbs/{ms}" loading="lazy"></figure>
  </div>
  <table>
    <tr><th>scene_id</th><td>{esc(record.get('source_scene_id',''))}</td></tr>
    <tr><th>intervention</th><td>{esc(json.dumps(intervention, ensure_ascii=False))}</td></tr>
    <tr><th>old → new</th><td>{esc(intervention.get('old_value'))} → {esc(intervention.get('new_value'))}</td></tr>
    <tr><th>source_abs</th><td class="path">{esc(source)}</td></tr>
    <tr><th>target_abs</th><td class="path">{esc(target)}</td></tr>
    <tr><th>mask_abs</th><td class="path">{esc(mask)}</td></tr>
  </table>
</section>""")
        csv_rows.append({
            "pair_id": pair_id,
            "subtype": record.get("subtype", ""),
            "asset_batch": record.get("asset_batch", record.get("training_assets", {}).get("asset_batch", "")),
            "reason": reason,
            "scene_id": record.get("source_scene_id", ""),
            "intervention": json.dumps(intervention, ensure_ascii=False),
            "old_value": intervention.get("old_value", ""),
            "new_value": intervention.get("new_value", ""),
            "source_abs": str(source),
            "target_abs": str(target),
            "mask_abs": str(mask),
            "audit_decision": "",
            "audit_notes": "",
        })

    pages = (len(cards) + page_size - 1) // page_size or 1
    nav = lambda current: "".join(
        f'<a class="{"on" if i == current else ""}" href="page_{i:03d}.html">{i}</a>'
        for i in range(1, pages + 1))
    for page in range(1, pages + 1):
        chunk = cards[(page - 1) * page_size:page * page_size]
        body = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>{esc(title)} - page {page}</title><style>
body{{font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif;background:#f5f6f8;margin:0;padding:18px;color:#1c2530}}
h1{{font-size:18px;margin:0 0 10px}}.nav{{margin-bottom:14px}}.nav a{{display:inline-block;padding:4px 10px;margin-right:6px;border-radius:6px;background:#fff;border:1px solid #d7dbe2;color:#33507a;text-decoration:none;font-size:13px}}
.nav a.on{{background:#305496;color:#fff;border-color:#305496}}
.card{{background:#fff;border:1px solid #e3e7ee;border-radius:12px;padding:14px;margin-bottom:16px}}
.head{{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}}
.idx{{background:#eef2f8;color:#33507a;border-radius:6px;padding:2px 8px;font-size:13px}}
.pid{{font-weight:600;font-size:14px;word-break:break-all}}
.tag{{background:#e8f3ec;color:#217a45;border-radius:6px;padding:2px 8px;font-size:12px}}
.sub{{background:#fdf2e3;color:#8a5a12;border-radius:6px;padding:2px 8px;font-size:12px}}
.reason{{background:#fff6f6;border-left:4px solid #d9534f;color:#8a2b2b;padding:8px 10px;font-size:13px;border-radius:6px;margin-bottom:10px}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
figure{{margin:0;background:#fafbfc;border:1px solid #eceff3;border-radius:8px;padding:8px}}
figure.hl{{border-color:#f0a6a6;background:#fffafa}}
figcaption{{font-size:12px;color:#6b7684;margin-bottom:6px;font-weight:600}}
img{{width:100%;display:block;border-radius:6px}}
table{{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px}}
th{{text-align:left;width:110px;color:#6b7684;font-weight:600;padding:3px 6px;vertical-align:top}}
td{{padding:3px 6px;word-break:break-all}}td.path{{color:#7b8794;font-size:11px}}
</style></head><body><h1>{esc(title)}　共 {len(cards)} 个 pair / {pages} 页</h1>
<div class="nav">{nav(page)}</div>
{''.join(chunk)}
<div class="nav">{nav(page)}</div></body></html>"""
        (out_dir / f"page_{page:03d}.html").write_text(body, encoding="utf-8")

    csv_path = out_dir / "audit.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)

    return {"pairs": len(cards), "pages": pages, "csv": str(csv_path),
            "entry": str(out_dir / "page_001.html")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--pair-ids", required=True, type=Path, help="jsonl with pair_id, optional reason")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--title", default="审计")
    ap.add_argument("--page-size", type=int, default=12)
    args = ap.parse_args()

    wanted: list[tuple[str, str]] = []
    for line in args.pair_ids.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            wanted.append((str(row["pair_id"]), str(row.get("reason", ""))))
    manifest = {str(r["pair_id"]): r for r in load_jsonl(args.manifest)}

    rows = []
    for pid, reason in wanted:
        record = manifest.get(pid)
        if record is None:
            print(f"skip missing: {pid}")
            continue
        record = json.loads(json.dumps(record))
        record["audit_reason"] = reason
        rows.append(record)
    rows.sort(key=lambda r: r["pair_id"])
    print(json.dumps(build(rows, args.output_dir, args.page_size, args.title), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
