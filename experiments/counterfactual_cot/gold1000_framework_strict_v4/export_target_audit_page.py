#!/usr/bin/env python3
"""Export target-focused audit pages from the FINAL merged manifest.

The page resolves every SOURCE/TARGET/MASK from the supplied final manifest.
With --audit-jsonl it shows only Qwen-flagged pairs; without it, it shows all
pairs and marks the target as the primary review object.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import posixpath
from pathlib import Path

from PIL import Image


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(root: Path, value: str) -> Path:
    p = (Path(value) if Path(value).is_absolute() else root / value).resolve()
    p.relative_to(root.resolve())
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def thumbnail(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.thumbnail((420, 420))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(dst, "JPEG", quality=82)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--audit-jsonl", type=Path, default=None)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--page-size", type=int, default=15)
    ap.add_argument("--include-all", action="store_true")
    args = ap.parse_args()

    manifest = {}
    for line in args.manifest.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            manifest[row["pair_id"]] = row
    audits = {}
    if args.audit_jsonl:
        for line in args.audit_jsonl.open(encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                audits[row["pair_id"]] = row

    selected = []
    for pid, pair in manifest.items():
        audit = audits.get(pid, {})
        flagged = not args.audit_jsonl or audit.get("needs_human_review", True) or bool(audit.get("error")) or not audit.get("auto_candidate_pass", False)
        if args.include_all or flagged:
            source = resolve(args.image_root, pair["source"]["image"])
            target = resolve(args.image_root, pair["target"]["image"])
            mask = resolve(args.image_root, pair["target"]["mask"])
            selected.append((pid, pair, audit, source, target, mask))
    selected.sort(key=lambda x: x[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["pair_id", "target_path", "target_sha256", "source_path", "mask_path", "subtype", "intervention", "old_value", "new_value", "qwen_status", "qwen_confidence", "qwen_issue_codes", "qwen_reason", "human_decision", "human_notes"]
    with (args.output_dir / "target_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pid, pair, audit, source, target, mask in selected:
            iv = pair.get("intervention", {})
            writer.writerow({"pair_id": pid, "target_path": str(target), "target_sha256": sha(target), "source_path": str(source), "mask_path": str(mask), "subtype": pair.get("subtype", ""), "intervention": iv.get("field", ""), "old_value": iv.get("old_value", ""), "new_value": iv.get("new_value", ""), "qwen_status": "HUMAN_REVIEW" if audit.get("needs_human_review", True) else "CANDIDATE_PASS", "qwen_confidence": audit.get("confidence", ""), "qwen_issue_codes": ";".join(audit.get("issue_codes", [])), "qwen_reason": audit.get("reason", ""), "human_decision": "", "human_notes": ""})

    pages = [selected[i:i + args.page_size] for i in range(0, len(selected), args.page_size)] or [[]]
    for page_no, page in enumerate(pages, 1):
        cards = []
        for pid, pair, audit, source, target, mask in page:
            iv = pair.get("intervention", {})
            status = "HUMAN REVIEW" if audit.get("needs_human_review", True) else "QWEN CANDIDATE PASS"
            cls = "review" if audit.get("needs_human_review", True) else "pass"
            figures = []
            for label, path, key in (("SOURCE", source, "source"), ("TARGET TO AUDIT", target, "target"), ("CANDIDATE MASK", mask, "mask")):
                t = args.output_dir / "thumbs" / f"{page_no:03d}_{pid}_{key}.jpg"
                thumbnail(path, t)
                href = posixpath.relpath(path.resolve(), args.output_dir.resolve())
                thref = posixpath.relpath(t.resolve(), args.output_dir.resolve())
                figures.append(f"<figure class='{key}'><a href='{html.escape(href)}' target='_blank'><img src='{html.escape(thref)}' loading='lazy'></a><figcaption>{label}</figcaption></figure>")
            meta = {"pair_id": pid, "subtype": pair.get("subtype"), "intervention": iv, "source_cot": pair.get("source", {}).get("cot_text"), "target_cot": pair.get("target", {}).get("cot_text"), "qwen_status": status, "qwen_confidence": audit.get("confidence"), "qwen_issue_codes": audit.get("issue_codes", []), "qwen_reason": audit.get("reason"), "target_sha256": sha(target)}
            cards.append(f"<section><h2>{html.escape(pid)} <span class='{cls}'>{status}</span></h2><div class='images'>{''.join(figures)}</div><h3>TARGET TO AUDIT</h3><pre>{html.escape(json.dumps(meta, ensure_ascii=False, indent=2))}</pre></section>")
        nav=[]
        if page_no > 1: nav.append(f"<a href='page_{page_no-1:03d}.html'>上一页</a>")
        nav.append(f"<span>第 {page_no}/{len(pages)} 页，共 {len(selected)} 个 target</span>")
        if page_no < len(pages): nav.append(f"<a href='page_{page_no+1:03d}.html'>下一页</a>")
        doc = ("<!doctype html><meta charset='utf-8'><title>Target audit</title><style>body{font-family:Arial;margin:20px;background:#f4f6f8}section{background:#fff;padding:16px;margin:0 0 16px;border-radius:10px}h2{font-size:14px;word-break:break-all}.images{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;padding:5px}figure.target{border:4px solid #d92d20;border-radius:8px;background:#fff5f5}img{width:240px;height:240px;object-fit:contain;background:#eee}figcaption{text-align:center;font-weight:bold;margin-top:4px}.review{color:#b42318}.pass{color:#16803c}h3{color:#b42318}pre{white-space:pre-wrap;font-size:11px;max-height:500px;overflow:auto}a{display:inline-block;margin:0 10px 15px 0;padding:8px 13px;background:#fff;border:1px solid #ccc;text-decoration:none}</style><h1>需要人工确认的 TARGET 审计页</h1>" + " ".join(nav) + "".join(cards) + " ".join(nav))
        (args.output_dir / f"page_{page_no:03d}.html").write_text(doc, encoding="utf-8")
    (args.output_dir / "page_source.json").write_text(json.dumps({"manifest": str(args.manifest), "manifest_sha256": sha(args.manifest), "audit_jsonl": str(args.audit_jsonl) if args.audit_jsonl else None, "pairs": len(selected), "pages": len(pages), "include_all": args.include_all}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": len(selected), "pages": len(pages), "csv": str(args.output_dir / 'target_audit.csv'), "entry": str(args.output_dir / 'page_001.html')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
