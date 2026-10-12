#!/usr/bin/env python3
"""Build an isolated comparison page for one pair across final and sandbox assets."""
from __future__ import annotations
import argparse, html, json, posixpath
from pathlib import Path
from PIL import Image

def thumb(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.thumbnail((480,480))
        if im.mode not in ('RGB','L'): im=im.convert('RGB')
        im.save(dst,'JPEG',quality=84)

def cell(path: Path, label: str, out: Path, tag: str) -> str:
    if not path.is_file():
        return f"<figure><div class='missing'>missing</div><figcaption>{label}</figcaption></figure>"
    t=out/'thumbs'/f'{tag}.jpg'; thumb(path,t)
    return f"<figure class='{tag}'><a target='_blank' href='{html.escape(posixpath.relpath(path.resolve(),out.resolve()))}'><img src='{html.escape(posixpath.relpath(t.resolve(),out.resolve()))}'></a><figcaption>{label}</figcaption></figure>"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pair-id',required=True)
    ap.add_argument('--manifest',required=True,type=Path)
    ap.add_argument('--image-root',required=True,type=Path)
    ap.add_argument('--sandbox',required=True,type=Path)
    ap.add_argument('--output-dir',required=True,type=Path)
    args=ap.parse_args(); rec=None
    for line in args.manifest.open():
        r=json.loads(line)
        if r['pair_id']==args.pair_id: rec=r; break
    if rec is None: raise SystemExit('pair not found')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pid=args.pair_id; root=args.image_root; sb=args.sandbox
    old_src=root/rec['source']['image']; old_tgt=root/rec['target']['image']; old_mask=root/rec['target']['mask']
    new_src=sb/'source'/f'{pid}__source.png'; new_tgt=sb/'target'/f'{pid}__target.png'; new_mask=sb/'masks'/f'{pid}__mask.png'
    figures=[cell(old_src,'FINAL DATASET SOURCE',args.output_dir,'old_src'),cell(new_src,'SANDBOX SOURCE',args.output_dir,'new_src'),cell(old_tgt,'FINAL DATASET TARGET',args.output_dir,'old_tgt'),cell(new_tgt,'SANDBOX TARGET TO AUDIT',args.output_dir,'new_tgt'),cell(new_mask,'SANDBOX MASK',args.output_dir,'new_mask')]
    meta={'pair_id':pid,'subtype':rec.get('subtype'),'intervention':rec.get('intervention'),'source_cot':rec['source'].get('cot_text'),'target_cot':rec['target'].get('cot_text'),'target_prompt':rec['target'].get('prompt')}
    doc="<!doctype html><meta charset='utf-8'><title>Pair comparison</title><style>body{font-family:Arial;margin:24px;background:#f4f6f8}section{background:#fff;padding:18px;border-radius:10px}.images{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;padding:6px}.new_tgt{border:4px solid #d92d20;background:#fff5f5;border-radius:8px}img{width:250px;height:250px;object-fit:contain;background:#eee}.missing{width:250px;height:250px;background:#fdd;color:#900;display:flex;align-items:center;justify-content:center}pre{white-space:pre-wrap;font-size:12px}</style><h1>TARGET TO AUDIT</h1><section><h2>"+html.escape(pid)+"</h2><div class='images'>"+''.join(figures)+"</div><pre>"+html.escape(json.dumps(meta,ensure_ascii=False,indent=2))+"</pre></section>"
    (args.output_dir/'page_001.html').write_text(doc,encoding='utf-8')
    print(args.output_dir/'page_001.html')
if __name__=='__main__': main()
