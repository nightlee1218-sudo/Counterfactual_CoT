"""Shared utilities for the data_v2 compositing pipeline.

Self-contained by design: stdlib + numpy/PIL only, no imports from the legacy
``data/`` package so that data_v2 stays portable and its selftests never touch
legacy manifests.  Record shapes stay schema-compatible with
gold1000_framework_strict_v4 (SceneGraph / Intervention / preservation spec).
"""
from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


# ---------------------------------------------------------------------------
# Scene-graph helpers (subset of data/schema.py, v4-compatible, dict-based)
# ---------------------------------------------------------------------------

def scene_atom_paths(scene: Dict[str, Any]) -> List[str]:
    """All addressable atoms of a scene graph (v4 semantics)."""
    paths: List[str] = []
    for entity in scene.get("entities", []):
        eid = entity["id"]
        paths.append(f"{eid}.category")
        if entity.get("count") is not None:
            paths.append(f"{eid}.count")
        if entity.get("location") is not None:
            paths.append(f"{eid}.location")
        for key in sorted(entity.get("attributes") or {}):
            paths.append(f"{eid}.attributes.{key}")
        if entity.get("relations"):
            paths.append(f"{eid}.relations")
    for key in sorted(scene.get("background") or {}):
        paths.append(f"background.{key}")
    return paths


def preservation_for(scene: Dict[str, Any], intervention: Dict[str, Any]) -> Dict[str, List[str]]:
    """v4 single-atom preservation spec: everything but the allowed path."""
    allowed = f"{intervention['entity_id']}.{intervention['field']}"
    must_preserve = [p for p in scene_atom_paths(scene) if p != allowed]
    must_revalidate = [f"{intervention['entity_id']}.attributes"] \
        if intervention["field"] == "category" else []
    return {
        "must_preserve": must_preserve,
        "allowed_to_change": [allowed],
        "must_revalidate": must_revalidate,
    }


def apply_intervention(scene: Dict[str, Any], intervention: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-copy the scene and set the single atom named by the intervention."""
    import copy as _copy
    target = _copy.deepcopy(scene)
    eid = intervention["entity_id"]
    field = intervention["field"]
    new_value = intervention["new_value"]
    for entity in target.get("entities", []):
        if entity["id"] != eid:
            continue
        if field == "relations":
            entity["relations"] = new_value if isinstance(new_value, list) else [new_value]
            if not entity["relations"]:
                entity.pop("relations", None)
        elif field.startswith("attributes."):
            entity.setdefault("attributes", {})[field.split(".", 1)[1]] = new_value
        else:
            entity[field] = new_value
        return target
    raise ValueError(f"entity {eid!r} not found in scene")


def render_cot_text(scene: Dict[str, Any]) -> str:
    """Compact deterministic COT text renderer (v4 style)."""
    parts: List[str] = []
    for entity in scene.get("entities", []):
        desc = entity["category"]
        attrs = entity.get("attributes") or {}
        for key in sorted(attrs):
            desc += f", {attrs[key]} {key}" if key in {"color", "material", "pattern"} \
                else f", {key}={attrs[key]}"
        if entity.get("count") is not None:
            desc += f" (count={entity['count']})"
        if entity.get("location"):
            desc += f" at {entity['location']}"
        for rel in entity.get("relations") or []:
            desc += f", {rel.get('type')} {rel.get('target')}"
        parts.append(f"- {entity['id']}: {desc}")
    bg = scene.get("background") or {}
    if bg:
        parts.append("- background: " + ", ".join(f"{k}={v}" for k, v in sorted(bg.items())))
    return "Scene COT:\n" + "\n".join(parts)


def scene_to_prompt(scene: Dict[str, Any]) -> str:
    """Natural-language rendering of a scene graph for image generation."""
    clauses: List[str] = []
    for entity in scene.get("entities", []):
        desc = str(entity["category"])
        attrs = entity.get("attributes") or {}
        for key in ("color", "material", "size", "pattern"):
            if key in attrs:
                desc = f"{attrs[key]} {desc}"
        if entity.get("count") is not None and entity["count"] > 1:
            desc = f"{entity['count']} {desc}s"
        if entity.get("location"):
            desc += f" in the {entity['location']}"
        for rel in entity.get("relations") or []:
            desc += f" {rel.get('type')} the {rel.get('target')}"
        clauses.append(desc)
    bg = scene.get("background") or {}
    if bg.get("description"):
        clauses.append(f"The background is {bg['description']}.")
    return "A photograph of " + "; ".join(clauses) + "."


# ---------------------------------------------------------------------------
# bbox helpers
# ---------------------------------------------------------------------------

def normalize_bbox(bbox: Any) -> Optional[Tuple[int, int, int, int]]:
    """Accept [x1,y1,x2,y2] / {x1..y2} / {x,y,w,h}; return int (x1,y1,x2,y2)."""
    if bbox is None:
        return None
    if isinstance(bbox, dict):
        if {"x", "y", "w", "h"} <= set(bbox):
            x, y = int(bbox["x"]), int(bbox["y"])
            return (x, y, x + int(bbox["w"]), y + int(bbox["h"]))
        return (int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"]))
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = (int(v) for v in bbox)
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    return None


def bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    return (max(0, bbox[0]), max(0, bbox[1]), min(width, bbox[2]), min(height, bbox[3]))


def context_crop_bbox(bbox: Tuple[int, int, int, int], width: int, height: int,
                      context_frac: float = 0.10) -> Tuple[int, int, int, int]:
    """Crop box = object bbox expanded by ~context_frac margin, clamped."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    mx = max(2, int(round(max(w, h) * context_frac)))
    my = max(2, int(round(max(w, h) * context_frac)))
    return clamp_bbox((x1 - mx, y1 - my, x2 + mx, y2 + my), width, height)


def dilate_mask(mask, ksize: int = 9):
    """Dilate a binary mask with an elliptical kernel (cv2, lazy import)."""
    import cv2
    import numpy as np
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1) > 0


def make_rng(seed: Any) -> random.Random:
    return random.Random(seed)
