"""data_v2 intervention planner.

Reuses the gold1000_framework_strict_v4 generation semantics (single atomic
change + must_preserve spec) with a rebalanced type mix:

  * attribute_intervention (color / material / size / pattern)  ~30%
  * relative_position (NEW: "A left of B" / "A on top of B",
    both object bboxes attached)                                ~30%
  * category_replace (down-weighted)                            ~20%
  * count_change (down-weighted)                                ~10%
  * location_change (spatial_relation_change, occlusion-style)  ~10%

Input : source manifest JSONL, one record per source scene:
    {"scene_id": ..., "prompt": ..., "image": "<path>",
     "cot": {"schema_version": "1.0", "entities": [...], "background": {...}},
     "bboxes": {"<entity_id>": [x1, y1, x2, y2], ...}}        # optional inline
    (or pass --bbox-file with {"scene_id": {"entity_id": [x1,y1,x2,y2]}} rows)
Output: intervention plan JSONL consumed by composite_target.py.  Each plan:
    pair_id / scene_id / subtype / intervention / preservation /
    source{...} / target{...} / objects{entity_id: bbox} /
    compositing{mode, crop_bbox, edit_prompt}

Pure-planning stage: no images are opened and no API is called.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    apply_intervention, bbox_area, clamp_bbox, context_crop_bbox, load_jsonl,
    make_rng, normalize_bbox, preservation_for, render_cot_text, scene_to_prompt,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Type mix and candidate pools (v4 style, rebalanced for data_v2)
# ---------------------------------------------------------------------------

SUBTYPE_WEIGHTS: Dict[str, float] = {
    "attribute_intervention": 0.30,   # color / material / size / pattern
    "relative_position": 0.30,        # A left of B / A on top of B (new)
    "category_replace": 0.20,         # down-weighted vs v4
    "count_change": 0.10,             # down-weighted vs v4
    "location_change": 0.10,          # absolute location swap (v4 relation style)
}

ATTRIBUTE_POOLS: Dict[str, List[str]] = {
    "color": ["red", "blue", "green", "yellow", "purple", "orange", "white", "black"],
    "material": ["wooden", "metal", "glass", "plastic", "ceramic", "fabric"],
    "size": ["small", "large", "tiny", "big"],
    "pattern": ["striped", "dotted", "plain", "checkered", "floral"],
}

CATEGORY_POOLS: Dict[str, List[str]] = {
    "banana": ["apple", "orange", "pear"],
    "apple": ["banana", "orange", "lemon"],
    "cup": ["bowl", "bottle", "mug"],
    "ball": ["box", "cube", "toy car"],
    "car": ["van", "taxi", "pickup truck"],
    "dog": ["cat", "rabbit", "fox"],
    "chair": ["stool", "armchair", "bench"],
    "flower": ["sunflower", "tulip", "rose"],
    "book": ["notebook", "magazine", "newspaper"],
    "bird": ["duck", "seagull", "parrot"],
    "bicycle": ["scooter", "motorcycle", "tricycle"],
}

LOCATIONS = ["upper-left", "upper-right", "center", "lower-left", "lower-right"]

RELATIVE_POSITION_KINDS = ("left_of", "on_top_of")


# ---------------------------------------------------------------------------
# Candidate generation (one scene -> typed candidate list)
# ---------------------------------------------------------------------------

def _entity_bbox(record: Dict[str, Any], scene_id: str, entity_id: str
                 ) -> Optional[Tuple[int, int, int, int]]:
    bboxes = record.get("bboxes") or {}
    return normalize_bbox(bboxes.get(entity_id))


def candidates_for_scene(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """All single-atom intervention candidates for one source record."""
    scene = record["cot"]
    scene_id = str(record["scene_id"])
    out: List[Dict[str, Any]] = []

    entities = scene.get("entities", [])
    for entity in entities:
        eid = entity["id"]
        attrs = entity.get("attributes") or {}

        # -- attribute_intervention: color / material / size / pattern --------
        for key in ("color", "material", "size", "pattern"):
            current = attrs.get(key)
            if current is None:
                continue
            for new_value in ATTRIBUTE_POOLS.get(key, []):
                if new_value != current:
                    out.append({
                        "subtype": "attribute_intervention",
                        "intervention": {
                            "operation": "replace",
                            "intervention_type": "attribute_replace",
                            "entity_id": eid,
                            "field": f"attributes.{key}",
                            "old_value": current,
                            "new_value": new_value,
                        },
                        "requires_bbox": [eid],
                    })

        # -- category_replace (down-weighted) --------------------------------
        for new_category in CATEGORY_POOLS.get(entity["category"], []):
            out.append({
                "subtype": "category_replace",
                "intervention": {
                    "operation": "replace",
                    "intervention_type": "object_replace",
                    "entity_id": eid,
                    "field": "category",
                    "old_value": entity["category"],
                    "new_value": new_category,
                },
                "requires_bbox": [eid],
            })

        # -- count_change (down-weighted) -------------------------------------
        if entity.get("count") is not None:
            for new_count in sorted({max(1, entity["count"] - 1), entity["count"] + 1}):
                if new_count != entity["count"]:
                    out.append({
                        "subtype": "count_change",
                        "intervention": {
                            "operation": "replace",
                            "intervention_type": "count_change",
                            "entity_id": eid,
                            "field": "count",
                            "old_value": entity["count"],
                            "new_value": new_count,
                        },
                        "requires_bbox": [eid],
                    })

        # -- location_change (v4 spatial_relation_change, occlusion style) ----
        if entity.get("location"):
            for new_location in LOCATIONS:
                if new_location != entity["location"]:
                    out.append({
                        "subtype": "location_change",
                        "intervention": {
                            "operation": "replace",
                            "intervention_type": "spatial_relation_change",
                            "entity_id": eid,
                            "field": "location",
                            "old_value": entity["location"],
                            "new_value": new_location,
                        },
                        "requires_bbox": [eid],
                    })

    # -- relative_position (NEW): A left of B / A on top of B -----------------
    for i, entity_a in enumerate(entities):
        for j, entity_b in enumerate(entities):
            if i == j:
                continue
            for kind in RELATIVE_POSITION_KINDS:
                out.append({
                    "subtype": "relative_position",
                    "intervention": {
                        "operation": "replace",
                        "intervention_type": "spatial_relation_change",
                        "entity_id": entity_a["id"],
                        "field": "relations",
                        "old_value": list(entity_a.get("relations") or []),
                        "new_value": [{"type": kind, "target": entity_b["id"]}],
                    },
                    "requires_bbox": [entity_a["id"], entity_b["id"]],
                })
    return out


def _bbox_available(record: Dict[str, Any], cand: Dict[str, Any]) -> bool:
    return all(_entity_bbox(record, "", eid) is not None for eid in cand["requires_bbox"])


# ---------------------------------------------------------------------------
# Plan assembly
# ---------------------------------------------------------------------------

def _edit_prompt_for(record: Dict[str, Any], cand: Dict[str, Any]) -> str:
    """Crop-level edit instruction for the image-editing API."""
    iv = cand["intervention"]
    eid = iv["entity_id"]
    scene = record["cot"]
    entity = next(e for e in scene["entities"] if e["id"] == eid)
    subject = f"the {entity['category']}"
    attrs = entity.get("attributes") or {}
    if attrs.get("color"):
        subject = f"the {attrs['color']} {entity['category']}"
    if cand["subtype"] == "attribute_intervention":
        key = iv["field"].split(".", 1)[1]
        instruction = (f"Change {subject} so its {key} becomes {iv['new_value']} "
                       f"(currently {iv['old_value']}).")
    elif cand["subtype"] == "category_replace":
        instruction = f"Replace {subject} with a {iv['new_value']}, same pose and position."
    elif cand["subtype"] == "count_change":
        instruction = (f"Change the number of {entity['category']}s from {iv['old_value']} "
                       f"to {iv['new_value']}, keeping their arrangement natural.")
    else:  # relative_position / location_change handled by MOVE mode
        instruction = ""
    if not instruction:
        return ""
    return (
        f"Edit this image crop. {instruction}\n"
        "- Modify ONLY the named object; keep every other pixel of the crop unchanged.\n"
        "- Preserve the original lighting, shadows, perspective, camera viewpoint and color temperature.\n"
        "- Do not restyle, re-render or filter the background of the crop.\n"
        "- Keep the object centered at the same position and scale unless the edit requires otherwise."
    )


def _move_prompt_for(cand: Dict[str, Any], scene: Dict[str, Any]) -> str:
    iv = cand["intervention"]
    rel = iv["new_value"][0]
    kind, target_id = rel["type"], rel["target"]
    entity_a = next(e for e in scene["entities"] if e["id"] == iv["entity_id"])
    entity_b = next(e for e in scene["entities"] if e["id"] == target_id)
    phrasing = {
        "left_of": f"immediately to the left of the {entity_b['category']}",
        "on_top_of": f"directly on top of the {entity_b['category']}",
    }.get(kind, f"{kind} the {entity_b['category']}")
    return (f"Move the {entity_a['category']} so it sits {phrasing}. "
            f"Keep its appearance, lighting and shadows consistent.")


def build_plan(record: Dict[str, Any], cand: Dict[str, Any], local_id: int
               ) -> Optional[Dict[str, Any]]:
    scene = record["cot"]
    scene_id = str(record["scene_id"])
    iv = cand["intervention"]

    objects: Dict[str, Tuple[int, int, int, int]] = {}
    for eid in cand["requires_bbox"]:
        bbox = _entity_bbox(record, scene_id, eid)
        if bbox is None:
            return None  # bbox missing -> candidate not realizable in v2
        x1, y1, x2, y2 = bbox
        objects[eid] = [x1, y1, x2, y2]

    target_scene = apply_intervention(scene, iv)
    pair_id = f"cfv2_{scene_id}_{local_id:04d}"

    if cand["subtype"] == "relative_position":
        mode = "move"
        edit_prompt = _move_prompt_for(cand, scene)
        crop_bbox = None
    else:
        mode = "edit"
        edit_prompt = _edit_prompt_for(record, cand)
        x1, y1, x2, y2 = objects[iv["entity_id"]]
        crop_bbox = [x1, y1, x2, y2]

    plan = {
        "schema_version": "counterfactual_data_v2_plan/1.0",
        "pair_id": pair_id,
        "scene_id": scene_id,
        "subtype": cand["subtype"],
        "intervention": iv,
        "preservation": preservation_for(scene, iv),
        "source": {
            "prompt": record.get("prompt") or scene_to_prompt(scene),
            "cot": scene,
            "cot_text": record.get("cot_text") or render_cot_text(scene),
            "image": record.get("image", ""),
        },
        "target": {
            "prompt": scene_to_prompt(target_scene),
            "cot": target_scene,
            "cot_text": render_cot_text(target_scene),
        },
        "objects": objects,
        "compositing": {
            "mode": mode,            # edit: crop->API->matte->paste; move: cut->inpaint->paste
            "crop_bbox": crop_bbox,  # object bbox; composite_target adds context margin
            "context_frac": 0.10,
            "edit_prompt": edit_prompt,
        },
    }
    return plan


# ---------------------------------------------------------------------------
# Weighted sampling across scenes
# ---------------------------------------------------------------------------

def sample_plans(records: List[Dict[str, Any]], n_plans: int, seed: int,
                 max_per_scene_subtype: int = 2
                 ) -> List[Dict[str, Any]]:
    """Sample n_plans plans following SUBTYPE_WEIGHTS as closely as possible.

    Bbox availability gates candidate realization; if a subtype runs dry the
    remaining quota is redistributed proportionally over the survivors.
    """
    rng = make_rng(seed)
    per_subtype: Dict[str, List[Dict[str, Any]]] = {k: [] for k in SUBTYPE_WEIGHTS}
    for record in records:
        for cand in candidates_for_scene(record):
            per_subtype.setdefault(cand["subtype"], []).append((record, cand))

    # target counts per subtype
    quotas = {k: int(w * n_plans) for k, w in SUBTYPE_WEIGHTS.items()}
    remainder = n_plans - sum(quotas.values())
    if remainder:
        # give leftovers to the largest-weight subtypes, deterministically
        for k in sorted(SUBTYPE_WEIGHTS, key=lambda k: -SUBTYPE_WEIGHTS[k]):
            if remainder <= 0:
                break
            quotas[k] += 1
            remainder -= 1

    # drop subtypes with no realizable candidates, redistribute
    available = {k: v for k, v in per_subtype.items() if v}
    for k in list(quotas):
        if k not in available and quotas[k] > 0:
            spare = quotas.pop(k)
            total_w = sum(SUBTYPE_WEIGHTS[x] for x in quotas)
            for k2 in quotas:
                quotas[k2] += int(spare * SUBTYPE_WEIGHTS[k2] / total_w)
            # exact remainder -> biggest weight
            diff = n_plans - sum(quotas.values())
            if diff > 0:
                quotas[max(quotas, key=lambda x: SUBTYPE_WEIGHTS[x])] += diff

    plans: List[Dict[str, Any]] = []
    used_per_scene_subtype: Counter = Counter()
    for subtype in sorted(quotas):
        pool = list(available.get(subtype, []))
        rng.shuffle(pool)
        taken = 0
        for record, cand in pool:
            if taken >= quotas[subtype]:
                break
            key = (str(record["scene_id"]), subtype)
            if used_per_scene_subtype[key] >= max_per_scene_subtype:
                continue
            plan = build_plan(record, cand, len(plans))
            if plan is None:
                continue
            plans.append(plan)
            used_per_scene_subtype[key] += 1
            taken += 1
    return plans


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-manifest", required=True, type=Path)
    ap.add_argument("--bbox-file", type=Path, default=None,
                    help="optional JSONL rows {scene_id, bboxes:{entity_id:[x1,y1,x2,y2]}}")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-plans", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    records = load_jsonl(args.source_manifest)
    if args.bbox_file and Path(args.bbox_file).exists():
        for row in load_jsonl(args.bbox_file):
            for rec in records:
                if str(rec["scene_id"]) == str(row["scene_id"]):
                    merged = dict(rec.get("bboxes") or {})
                    merged.update(row.get("bboxes") or {})
                    rec["bboxes"] = merged
    for rec in records:  # basic validation
        if "scene_id" not in rec or "cot" not in rec:
            raise SystemExit(f"source record missing scene_id/cot: {rec}")
        if not str(rec.get("image", "")).strip():
            raise SystemExit(f"source scene {rec['scene_id']} has no image path")

    plans = sample_plans(records, args.n_plans, args.seed)
    write_jsonl(args.out, plans)
    dist = Counter(p["subtype"] for p in plans)
    print({"n_plans": len(plans), "distribution": dict(sorted(dist.items())),
           "out": str(args.out)})


if __name__ == "__main__":
    main()
