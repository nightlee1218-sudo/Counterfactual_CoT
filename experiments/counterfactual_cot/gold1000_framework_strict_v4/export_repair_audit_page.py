#!/usr/bin/env python3
"""Export a paginated audit site for every generated repair target.

The red TARGET TO AUDIT image is always the target from the isolated repair
sandbox manifest. Source-regenerated pairs additionally show original and new
source. Every selected/spillover pair in the repair batch is included; failures
are shown as explicit missing/failed records when present in repair_report.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import posixpath
from pathlib import Path
from typing import Any

from PIL import Image


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def thumb(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as image:
        image.thumbnail((400, 400))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(dst, "JPEG", quality=82)


def href(path: Path, output: Path) -> str:
    return posixpath.relpath(path.resolve(), output.resolve())


def figure(path: Path | None, label: str, key: str, number: int, output: Path, red: bool = False) -> str:
    if path is None or not path.is_file():
        return f"<figure class='{key} missing'><div>missing</div><figcaption>{html.escape(label)}</figcaption></figure>"
    t = output / "thumbs" / f"{number:05d}_{key}.jpg"
    thumb(path, t)
    return (
        f"<figure class='{key}{' target-highlight' if red else ''}'>"
        f"<a href='{html.escape(href(path, output))}' target='_blank'>"
        f"<img src='{html.escape(href(t, output))}' loading='lazy'></a>"
        f"<figcaption>{html.escape(label)}</figcaption></figure>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-manifest", required=True, type=Path)
    ap.add_argument("--original-manifest", required=True, type=Path)
    ap.add_argument("--original-image-root", required=True, type=Path)
    ap.add_argument("--batch-root", required=True, type=Path)
    ap.add_argument("--repair-plan", type=Path, default=None, help="Approved repair_plan_20260923.jsonl; includes failed/skipped pair IDs")
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--page-size", type=int, default=12)
    args = ap.parse_args()

    batch = [json.loads(line) for line in args.batch_manifest.open(encoding="utf-8") if line.strip()]
    batch_by_id = {r["pair_id"]: r for r in batch}
    original = {r["pair_id"]: r for r in (json.loads(line) for line in args.original_manifest.open(encoding="utf-8") if line.strip())}
    plan_by_id = {}
    if args.repair_plan:
        plan_by_id = {r["pair_id"]: r for r in (json.loads(line) for line in args.repair_plan.open(encoding="utf-8") if line.strip())}
    batch_plan_path = args.batch_root / "repair_batch_plan.json"
    selected_ids = list(json.loads(batch_plan_path.read_text(encoding="utf-8")).get("pair_ids", [])) if batch_plan_path.is_file() else list(batch_by_id)
    report_path = args.batch_root / "repair_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
    events = {(e.get("stage"), e.get("pair_id"), e.get("scene_id")): e for e in report.get("items", [])}

    records = []
    for pid in selected_ids:
        r = batch_by_id.get(pid)
        if r is None:
            r = json.loads(json.dumps(original[pid]))
            plan = plan_by_id.get(pid, {})
            r["repair"] = {
                "repair_scope": plan.get("repair_scope", ""),
                "repair_role": plan.get("repair_role", "source_scene_spillover"),
                "source_regenerated": plan.get("repair_scope") == "source+target",
                "target_regenerated": False,
                "user_issue": plan.get("user_issue", ""),
                "source_measure": plan.get("source_measure", ""),
                "target_measure": plan.get("target_measure", ""),
                "target_prompt_used": "NOT GENERATED: see repair_report.json",
            }
            r["source"]["image"] = str(Path("source") / f"{pid}__source.png")
            r["target"]["image"] = str(Path("target") / f"{pid}__target.png")
            r["target"]["mask"] = str(Path("masks") / f"{pid}__mask.png")
        repair = r.get("repair", {})
        source_new = args.batch_root / r["source"]["image"]
        target_new = args.batch_root / r["target"]["image"]
        mask_new = args.batch_root / r["target"]["mask"]
        old = original.get(pid, {})
        source_old = args.original_image_root / old.get("source", {}).get("image", "") if old else None
        target_old = args.original_image_root / old.get("target", {}).get("image", "") if old else None
        records.append({"pair": r, "repair": repair, "source_new": source_new, "target_new": target_new, "mask_new": mask_new, "source_old": source_old, "target_old": target_old})
    records.sort(key=lambda x: x["pair"]["pair_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "pair_id", "repair_scope", "repair_role", "source_regenerated", "target_regenerated",
        "source_new", "target_new_to_audit", "mask_new", "source_original", "target_original",
        "subtype", "intervention_field", "old_value", "new_value", "user_issue",
        "source_measure", "target_measure", "audit_decision", "audit_notes",
    ]
    with (args.output_dir / "repair_audit.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in records:
            r = item["pair"]; repair = item["repair"]; iv = r.get("intervention", {})
            writer.writerow({
                "pair_id": r["pair_id"], "repair_scope": repair.get("repair_scope"), "repair_role": repair.get("repair_role"),
                "source_regenerated": repair.get("source_regenerated"), "target_regenerated": repair.get("target_regenerated"),
                "source_new": str(item["source_new"]), "target_new_to_audit": str(item["target_new"]), "mask_new": str(item["mask_new"]),
                "source_original": str(item["source_old"]), "target_original": str(item["target_old"]),
                "subtype": r.get("subtype"), "intervention_field": iv.get("field"), "old_value": iv.get("old_value"), "new_value": iv.get("new_value"),
                "user_issue": repair.get("user_issue", ""), "source_measure": repair.get("source_measure", ""),
                "target_measure": repair.get("target_measure", ""), "audit_decision": "", "audit_notes": "",
            })

    pages = [records[i:i + args.page_size] for i in range(0, len(records), args.page_size)] or [[]]
    for page_no, page in enumerate(pages, 1):
        cards = []
        for local, item in enumerate(page):
            r = item["pair"]; repair = item["repair"]; iv = r.get("intervention", {})
            number = (page_no - 1) * args.page_size + local
            scope = repair.get("repair_scope", "")
            role = repair.get("repair_role", "reported_problem")
            badge = "SOURCE + TARGET" if scope == "source+target" else "TARGET ONLY"
            if role == "source_scene_spillover": badge += " / SOURCE SCENE SPILLOVER"
            if repair.get("target_regenerated") is False: badge += " / GENERATION FAILED OR SKIPPED"
            images = []
            if repair.get("source_regenerated"):
                images.append(figure(item["source_old"], "SOURCE ORIGINAL (reference)", "source-old", number * 10, args.output_dir))
                images.append(figure(item["source_new"], "SOURCE NEW (used for target)", "source-new", number * 10 + 1, args.output_dir))
            else:
                images.append(figure(item["source_new"], "SOURCE USED", "source", number * 10, args.output_dir))
            images.append(figure(item["target_old"], "TARGET ORIGINAL (reference)", "target-old", number * 10 + 2, args.output_dir))
            images.append(figure(item["target_new"], "TARGET TO AUDIT (NEW REPAIR)", "target-new", number * 10 + 3, args.output_dir, red=True))
            images.append(figure(item["mask_new"], "MASK NEW", "mask", number * 10 + 4, args.output_dir))
            meta = {
                "pair_id": r["pair_id"], "repair_scope": scope, "repair_role": role,
                "subtype": r.get("subtype"), "intervention": iv,
                "source_regenerated": repair.get("source_regenerated"), "target_regenerated": repair.get("target_regenerated"),
                "user_issue": repair.get("user_issue"), "source_measure": repair.get("source_measure"),
                "target_measure": repair.get("target_measure"), "target_prompt_used": repair.get("target_prompt_used", r.get("target", {}).get("prompt")),
                "target_sha256": sha(item["target_new"]) if item["target_new"].is_file() else None,
            }
            cards.append(
                f"<section><h2>{html.escape(r['pair_id'])} <span class='badge'>{html.escape(badge)}</span></h2>"
                f"<div class='images'>{''.join(images)}</div><h3>重点审计：红框 TARGET TO AUDIT (NEW REPAIR)</h3>"
                f"<pre>{html.escape(json.dumps(meta, ensure_ascii=False, indent=2))}</pre></section>"
            )
        nav=[]
        if page_no > 1: nav.append(f"<a href='page_{page_no-1:03d}.html'>上一页</a>")
        nav.append(f"<span>第 {page_no}/{len(pages)} 页，共 {len(records)} 个新target</span>")
        if page_no < len(pages): nav.append(f"<a href='page_{page_no+1:03d}.html'>下一页</a>")
        doc=("<!doctype html><meta charset='utf-8'><title>Repair audit</title><style>body{font-family:Arial;margin:20px;background:#f4f6f8}section{background:#fff;padding:16px;margin:0 0 16px;border-radius:10px}h2{font-size:13px;word-break:break-all}.badge{font-size:11px;color:#fff;background:#0369a1;padding:3px 8px;border-radius:8px}.images{display:flex;gap:10px;flex-wrap:wrap}figure{margin:0;padding:5px}figure.target-highlight{border:4px solid #d92d20;border-radius:8px;background:#fff5f5}img{width:220px;height:220px;object-fit:contain;background:#eee}figcaption{text-align:center;font-weight:bold;font-size:11px}h3{color:#b42318}pre{white-space:pre-wrap;font-size:10px;max-height:520px;overflow:auto}a{display:inline-block;margin:0 10px 15px 0;padding:8px 13px;background:#fff;border:1px solid #ccc;text-decoration:none}.missing div{width:220px;height:220px;background:#fdd;color:#900;display:flex;align-items:center;justify-content:center}</style><h1>修补候选 TARGET 审计</h1>"+" ".join(nav)+"".join(cards)+" ".join(nav))
        (args.output_dir / f"page_{page_no:03d}.html").write_text(doc, encoding="utf-8")
    (args.output_dir / "page_source.json").write_text(json.dumps({"batch_manifest":str(args.batch_manifest),"batch_manifest_sha256":sha(args.batch_manifest),"original_manifest":str(args.original_manifest),"pair_count":len(records),"page_count":len(pages)},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"pairs":len(records),"pages":len(pages),"csv":str(args.output_dir/'repair_audit.csv'),"entry":str(args.output_dir/'page_001.html')},ensure_ascii=False,indent=2))


if __name__ == "__main__": main()
