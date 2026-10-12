#!/usr/bin/env python3
"""Make wet/dry edits visible by darkening the fabric / soap base colour.

A wet white or ivory towel or bar of soap looks almost identical to a dry one:
both stay light, and no prompt can conjure a visible difference.  Damp深色 fabric
and damp dark soap, by contrast, darken noticeably and pick up specular
highlights, so the functional-state change becomes obvious at a glance.

Only the base COLOUR attribute is changed.  Colour is not the intervened field
(the intervention is functional_state, wet <-> dry), and both the source and the
target CoT receive the same new colour, so the edit stays single-atom.

Rewrites, per affected pair:
  * source.cot / target.cot entity colour
  * source.cot_text / target.cot_text colour token
and installs a prompt that spells out the wet vs dry appearance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# entity category -> replacement colour chosen for maximum wet/dry contrast
COLOUR_BY_CATEGORY = {
    "towel": "navy",
    "bar of soap": "charcoal",
    "soap": "charcoal",
}

WET_DRY_CUE = {
    "wet": (
        "WET means the surface is visibly soaked: clearly darker than a dry one, with a glossy specular sheen, "
        "clinging damp folds or a wet film, and visible moisture"
    ),
    "dry": (
        "DRY means the surface is uniformly matte and lighter: no sheen, no damp patches, crisp and fluffy or chalky"
    ),
}

TAIL = (
    "The edit must be obvious when the two images are compared side by side. "
    "Keep a native square, full-bleed scene filling the canvas edge to edge. "
    "Do not introduce an inset image, photo frame, border, blank margin, white wedge, collage, card, poster, page, or mockup. "
    "No readable text, labels, letters, numbers, logos, watermarks, signs, or UI."
)
FIXED_INVENTORY = (
    "CRITICAL CONSTRAINT: the scene must contain EXACTLY the same objects as the source image. "
    "Do not add any new object, do not duplicate any object, and do not delete any object. "
)


def _stem(word: str) -> str:
    return word[:-2] if word.endswith("en") and len(word) > 4 else word


def natural(value: object) -> str:
    return str(value).replace("_", " ")


def medium_phrase(entity: dict) -> str:
    attrs = entity.get("attributes", {}) or {}
    used = {_stem(w) for w in natural(entity["category"]).lower().split()}
    parts: list[str] = []
    for key in ("material",):
        value = natural(attrs.get(key, "")).strip()
        if value and all(_stem(t) not in used for t in value.lower().split()):
            parts.append(value)
    parts.append(natural(attrs.get("color", "")))
    parts.append(natural(entity["category"]))
    return " ".join(p for p in parts if p.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    args = parser.parse_args()

    records = [json.loads(l) for l in args.manifest.open(encoding="utf-8") if l.strip()]
    affected = []
    for record in records:
        iv = record.get("intervention", {})
        if not iv.get("field", "").endswith("functional_state"):
            continue
        if {str(iv.get("old_value")), str(iv.get("new_value"))} != {"wet", "dry"}:
            continue
        affected.append(record)

    for record in affected:
        iv = record["intervention"]
        eid = iv["entity_id"]
        src = next(e for e in record["source"]["cot"]["entities"] if e["id"] == eid)
        category = str(src["category"]).lower()
        new_colour = next((v for k, v in COLOUR_BY_CATEGORY.items() if k in category), "navy")
        old_colour = str(src["attributes"].get("color", ""))

        for side in ("source", "target"):
            entity = next(e for e in record[side]["cot"]["entities"] if e["id"] == eid)
            entity["attributes"]["color"] = new_colour
            text = record[side]["cot_text"]
            token = f"{eid}.attribute.color={old_colour}"
            if token not in text:
                raise SystemExit(f"{record['pair_id']}: token {token!r} missing from {side} cot_text")
            record[side]["cot_text"] = text.replace(token, f"{eid}.attribute.color={new_colour}")

        desc = medium_phrase(src)
        wet_v, dry_v = str(iv["old_value"]), str(iv["new_value"])
        cue = WET_DRY_CUE[dry_v]
        record["target"]["prompt"] = (
            f"Change only the functional state of the {desc} from {natural(wet_v)} to {natural(dry_v)}. "
            f"{cue}. "
            "The difference must be unmistakable at a glance: a clear change in darkness and surface sheen on that object. "
            "A subtle or ambiguous change counts as a failure. "
            f"{FIXED_INVENTORY}Keep the object's silhouette, size, position, and count unchanged, and do not alter any other object. {TAIL}"
        )
        record["target"]["wetdry_colour_repaired"] = True
        record["target"]["colour_change"] = f"{old_colour} -> {new_colour}"

    with args.output_manifest.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({
        "affected": len(affected),
        "changes": {r["pair_id"]: r["target"]["colour_change"] for r in affected},
        "output_manifest": str(args.output_manifest),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
