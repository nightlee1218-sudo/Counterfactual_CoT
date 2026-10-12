#!/usr/bin/env python3
"""Inventory and compare every sandbox target that differs from the final dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import posixpath
from collections import defaultdict
from pathlib import Path
from PIL import Image

KEEP_OLD = {
    "g1000_bathroom_counter__rinsecup_soap_towel__functional_state_005",
    "g1000_flower_shop__tulip_pot_ribbon__pattern_006",
    "g1000_flower_shop__tulip_pot_ribbon__pattern_007",
    "g1000_flower_shop__tulip_pot_ribbon__pattern_008",
}

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def thumb(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.thumbnail((360, 360))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        im.save(dst, "JPEG", quality=82)

def rel(path: Path, out: Path) -> str:
    return posixpath.relpath(path.resolve(), out.resolve())

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--sandbox", action="append", nargs=2, metavar=("LABEL", "DIR"), required=True)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--page-size", type=int, default=8)
    args = ap.parse_args()

    final = {}
    for line in args.manifest.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            final[r["pair_id"]] = r

    candidates = defaultdict(list)
    for label, directory in args.sandbox:
        root = Path(directory).resolve()
        manifest_path = root / "manifest.jsonl"
        if not manifest_path.is_file():
            continue
        for line in manifest_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            pid = record.get("pair_id")
            if pid not in final:
                continue
            target = root / "target" / f"{pid}__target.png"
            source = root / "source" / f"{pid}__source.png"
            mask = root / "masks" / f"{pid}__mask.png"
            final_target = args.image_root / "target" / f"{pid}__target.png"
            if not target.is_file() or not final_target.is_file() or digest(target) == digest(final_target):
                continue
            candidates[pid].append({
                "label": label,
                "root": str(root),
                "target": str(target),
                "source": str(source) if source.is_file() else str(args.image_root / final[pid]["source"]["image"]),
                "mask": str(mask) if mask.is_file() else str(args.image_root / final[pid]["target"]["mask"]),
                "target_sha256": digest(target),
                "model": record.get("generation", {}).get("model"),
                "sandbox_merged": record.get("generation", {}).get("sandbox_merged", False),
            })

    rows = []
    for pid in sorted(candidates):
        candidates[pid].sort(key=lambda x: x["label"])
        rows.append((pid, final[pid], candidates[pid]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sandbox_candidates.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["pair_id", "intervention", "subtype", "candidate_label", "candidate_target", "candidate_source", "candidate_mask", "target_sha256", "model", "already_merged", "keep_old_requested"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for pid, pair, versions in rows:
            iv = pair.get("intervention", {})
            for v in versions:
                writer.writerow({
                    "pair_id": pid, "intervention": iv.get("field", ""), "subtype": pair.get("subtype", ""),
                    "candidate_label": v["label"], "candidate_target": v["target"], "candidate_source": v["source"],
                    "candidate_mask": v["mask"], "target_sha256": v["target_sha256"], "model": v["model"],
                    "already_merged": v["sandbox_merged"], "keep_old_requested": pid in KEEP_OLD,
                })

    pages = [rows[i:i + args.page_size] for i in range(0, len(rows), args.page_size)] or [[]]
    for page_no, page in enumerate(pages, 1):
        cards = []
        for pid, pair, versions in page:
            final_target = args.image_root / "target" / f"{pid}__target.png"
            figures = []
            for label, path in [("FINAL CURRENT TARGET", final_target)] + [
                (f"SANDBOX {v['label']} TARGET", Path(v["target"])) for v in versions
            ]:
                t = args.output_dir / "thumbs" / f"{page_no:03d}_{pid}_{len(figures):02d}.jpg"
                thumb(path, t)
                css_class = "candidate" if label.startswith("SANDBOX") else "final"
                figures.append(f"<figure class='{css_class}'><a target='_blank' href='{html.escape(rel(path,args.output_dir))}'><img src='{html.escape(rel(t,args.output_dir))}' loading='lazy'></a><figcaption>{html.escape(label)}</figcaption></figure>")
            iv = pair.get("intervention", {})
            source_labels = ", ".join(v["label"] for v in versions)
            meta = {"pair_id": pid, "subtype": pair.get("subtype"), "intervention": iv, "candidate_versions": source_labels, "keep_old_requested": pid in KEEP_OLD, "candidate_paths": [{"label":v["label"],"target":v["target"],"source":v["source"],"mask":v["mask"],"sha256":v["target_sha256"]} for v in versions]}
            cards.append(f"<section><h2>{html.escape(pid)}</h2><div class='images'>{''.join(figures)}</div><pre>{html.escape(json.dumps(meta,ensure_ascii=False,indent=2))}</pre></section>")
        nav=[]
        if page_no>1: nav.append(f"<a href='page_{page_no-1:03d}.html'>上一页</a>")
        nav.append(f"<span>第 {page_no}/{len(pages)} 页，共 {len(rows)} 个 pair</span>")
        if page_no<len(pages): nav.append(f"<a href='page_{page_no+1:03d}.html'>下一页</a>")
        doc=("<!doctype html><meta charset='utf-8'><title>Sandbox candidates</title><style>body{font-family:Arial;margin:20px;background:#f4f6f8}section{background:#fff;padding:14px;margin-bottom:14px;border-radius:10px}h2{font-size:13px;word-break:break-all}.images{display:flex;gap:10px;flex-wrap:wrap}figure{margin:0;padding:5px}figure.candidate{border:4px solid #d92d20;border-radius:8px;background:#fff5f5}figure.final{border:2px solid #999}img{width:220px;height:220px;object-fit:contain;background:#eee}figcaption{text-align:center;font-weight:bold;font-size:11px}pre{white-space:pre-wrap;font-size:10px;max-height:430px;overflow:auto}a{display:inline-block;margin:0 10px 15px 0;padding:8px 13px;background:#fff;border:1px solid #ccc;text-decoration:none}</style><h1>最终目录 vs 隔离沙箱候选 target</h1>"+" ".join(nav)+"".join(cards)+" ".join(nav))
        (args.output_dir/f"page_{page_no:03d}.html").write_text(doc,encoding='utf-8')
    (args.output_dir/'inventory.json').write_text(json.dumps({'pairs':len(rows),'candidate_versions':sum(len(v) for _,_,v in rows),'keep_old':sorted(KEEP_OLD)},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pairs':len(rows),'candidate_versions':sum(len(v) for _,_,v in rows),'pages':len(pages),'entry':str(args.output_dir/'page_001.html'),'csv':str(args.output_dir/'sandbox_candidates.csv')},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
