from __future__ import annotations

import csv
import html
import json
import posixpath
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .render_prompts import render_intervention_text


FIELDS = [
    "pair_id",
    "scene_id",
    "intervention_type",
    "instruction",
    "source_prompt",
    "target_prompt",
    "source_image",
    "target_image",
    "candidate_mask",
    "target_success",
    "preservation_score",
    "binding_score",
    "locality_score",
    "image_quality_score",
    "review_status",
    "accepted",
    "review_notes",
]


def _row(pair: Dict[str, Any]) -> Dict[str, Any]:
    quality = pair.get("quality", {})
    intervention = pair.get("intervention", {})
    return {
        "pair_id": pair.get("pair_id", ""),
        "scene_id": pair.get("source_scene_id", pair.get("scene_id", "")),
        "intervention_type": intervention.get("intervention_type", ""),
        "instruction": render_intervention_text(intervention),
        "source_prompt": pair.get("source", {}).get("prompt", ""),
        "target_prompt": pair.get("target", {}).get("prompt", ""),
        "source_image": pair.get("source", {}).get("image", ""),
        "target_image": pair.get("target", {}).get("image", ""),
        "candidate_mask": pair.get("target", {}).get("mask", ""),
        "target_success": quality.get("target_success", 0.0),
        "preservation_score": quality.get("preservation_score", 0.0),
        "binding_score": quality.get("binding_score", 0.0),
        "locality_score": quality.get("locality_score", 0.0),
        "image_quality_score": quality.get("image_quality_score", 0.0),
        "review_status": quality.get("review_status", "pending"),
        "accepted": quality.get("accepted", False),
        "review_notes": quality.get("review_notes", ""),
    }


def export_review(records: Iterable[Dict[str, Any]], review_dir: Path, image_root: Path) -> None:
    rows = [_row(pair) for pair in records]
    review_dir.mkdir(parents=True, exist_ok=True)
    csv_path = review_dir / "review.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    def uri(value: str) -> str:
        path = Path(value)
        absolute = path if path.is_absolute() else image_root / path
        try:
            return posixpath.relpath(absolute.resolve(), review_dir.resolve())
        except ValueError:
            return absolute.resolve().as_uri()

    def figure(value: str, caption: str) -> str:
        if not str(value).strip() or not (image_root / str(value)).is_file():
            return (
                f"<figure><div class='missing'>&#9888; missing</div>"
                f"<figcaption>{caption}</figcaption></figure>"
            )
        return (
            f"<figure><img src='{html.escape(uri(str(value)))}'>"
            f"<figcaption>{caption}</figcaption></figure>"
        )

    cards: List[str] = []
    for row in rows:
        cards.append(
            "<section><h2>{pair}</h2><div class='images'>"
            "{source}{target}{mask}"
            "</div><pre>{details}</pre></section>".format(
                pair=html.escape(str(row["pair_id"])),
                source=figure(str(row["source_image"]), "source"),
                target=figure(str(row["target_image"]), "target"),
                mask=figure(str(row["candidate_mask"]), "candidate mask"),
                details=html.escape(json.dumps(row, ensure_ascii=False, indent=2)),
            )
        )
    document = """<!doctype html><meta charset='utf-8'><title>Counterfactual CoT Review</title>
<style>body{font-family:Arial,sans-serif;margin:24px;background:#f5f6f8}section{background:white;padding:18px;margin:0 0 20px;border-radius:10px}.images{display:flex;gap:14px;flex-wrap:wrap}figure{margin:0}img{width:280px;height:280px;object-fit:contain;background:#eee}pre{white-space:pre-wrap}.missing{width:280px;height:280px;display:flex;align-items:center;justify-content:center;background:#fdd;color:#900;font-size:20px}</style>
<h1>Counterfactual CoT Human Review</h1>""" + "".join(cards)
    (review_dir / "review.html").write_text(document, encoding="utf-8")
