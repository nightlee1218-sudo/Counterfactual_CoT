#!/usr/bin/env python3
"""Create a paginated Qwen-VL triage review site without changing audit results."""
from __future__ import annotations

import argparse
import csv
import html
import json
import posixpath
from pathlib import Path
from typing import Any

from PIL import Image


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def asset(root: Path, value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def thumbnail(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.thumbnail((384, 384))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(dst, "JPEG", quality=82)
    return posixpath.relpath(dst.resolve(), dst.parent.parent.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--audit-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--page-size", type=int, default=20)
    args = parser.parse_args()

    manifest = {}
    for line in args.manifest.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            manifest[row["pair_id"]] = row
    audits = {}
    for line in args.audit_jsonl.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            audits[row["pair_id"]] = row

    rows = []
    for pid, audit in audits.items():
        if pid not in manifest:
            continue
        pair = manifest[pid]
        source = asset(args.image_root, pair["source"]["image"])
        target = asset(args.image_root, pair["target"]["image"])
        mask = asset(args.image_root, pair["target"]["mask"])
        rows.append((pid, pair, audit, source, target, mask))
    rows.sort(key=lambda item: (item[2].get("auto_candidate_pass", False), item[0]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "qwen_triage.csv"
    fields = [
        "pair_id", "auto_candidate_pass", "needs_human_review", "confidence",
        "source_correct", "target_success", "preservation_score", "binding_score",
        "locality_score", "image_quality_score", "issue_codes", "reason", "human_decision", "human_notes",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for pid, _, audit, _, _, _ in rows:
            writer.writerow({
                **{key: audit.get(key, "") for key in fields if key not in {"issue_codes", "human_decision", "human_notes"}},
                "issue_codes": ";".join(audit.get("issue_codes", [])),
                "human_decision": "",
                "human_notes": "",
            })

    def fig(path: Path, label: str, number: int) -> str:
        thumb = args.output_dir / "thumbs" / f"{number:04d}_{path.stem}.jpg"
        thumbnail(path, thumb)
        href = posixpath.relpath(path.resolve(), args.output_dir.resolve())
        thumb_href = posixpath.relpath(thumb.resolve(), args.output_dir.resolve())
        return f"<figure><a href='{html.escape(href)}' target='_blank'><img src='{html.escape(thumb_href)}' loading='lazy'></a><figcaption>{label}</figcaption></figure>"

    pages = [rows[i:i + args.page_size] for i in range(0, len(rows), args.page_size)] or [[]]
    for page_no, chunk in enumerate(pages, 1):
        cards = []
        for number, (pid, pair, audit, source, target, mask) in enumerate(chunk, 1):
            global_no = (page_no - 1) * args.page_size + number
            passed = bool(audit.get("auto_candidate_pass"))
            badges = "<b class='pass'>Qwen candidate pass</b>" if passed else "<b class='review'>HUMAN REVIEW</b>"
            meta = {
                "confidence": audit.get("confidence"),
                "source_correct": audit.get("source_correct"),
                "target_success": audit.get("target_success"),
                "preservation_score": audit.get("preservation_score"),
                "binding_score": audit.get("binding_score"),
                "locality_score": audit.get("locality_score"),
                "image_quality_score": audit.get("image_quality_score"),
                "issue_codes": audit.get("issue_codes", []),
                "reason": audit.get("reason"),
                "intervention": pair.get("intervention"),
                "subtype": pair.get("subtype"),
            }
            cards.append(f"<section><h2>{html.escape(pid)} {badges}</h2><div class='images'>{fig(source,'SOURCE',global_no*3,)}{fig(target,'TARGET',global_no*3+1)}{fig(mask,'MASK',global_no*3+2)}</div><pre>{html.escape(json.dumps(meta,ensure_ascii=False,indent=2))}</pre></section>")
        nav=[]
        if page_no>1: nav.append(f"<a href='page_{page_no-1:02d}.html'>prev</a>")
        nav.append(f"<span>page {page_no}/{len(pages)}</span>")
        if page_no<len(pages): nav.append(f"<a href='page_{page_no+1:02d}.html'>next</a>")
        doc="""<!doctype html><meta charset='utf-8'><title>Qwen-VL triage</title><style>body{font-family:Arial;margin:20px;background:#f4f6f8}section{background:#fff;padding:14px;margin:0 0 14px;border-radius:9px}.images{display:flex;gap:10px;flex-wrap:wrap}figure{margin:0}img{width:220px;height:220px;object-fit:contain;background:#eee}.pass{color:#16803c}.review{color:#b42318}pre{white-space:pre-wrap;font-size:11px}a{display:inline-block;margin:0 10px 15px 0;padding:7px 12px;background:#fff;border:1px solid #ccc;text-decoration:none}</style><h1>Qwen-VL triage review</h1>"""+" ".join(nav)+"".join(cards)+" "+" ".join(nav)
        (args.output_dir/f"page_{page_no:02d}.html").write_text(doc,encoding="utf-8")
    (args.output_dir/"audit_source.json").write_text(json.dumps({"audit_jsonl":str(args.audit_jsonl),"audit_sha256":sha256(args.audit_jsonl),"pairs":len(rows),"pages":len(pages)},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"pairs":len(rows),"pages":len(pages),"csv":str(csv_path),"entry":str(args.output_dir/'page_01.html')},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
