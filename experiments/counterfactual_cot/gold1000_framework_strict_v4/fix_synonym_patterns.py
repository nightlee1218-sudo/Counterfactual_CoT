#!/usr/bin/env python3
"""Replace the semantically synonymous pattern pair `solid -> plain`.

`solid` and `plain` mean the same thing visually, so no prompt can render a
visible difference: the editor returns the source image and the sample teaches
nothing.  This script rewrites the TARGET value to a pattern that is unmistakable
at a glance, keeping the edit single-atom (only `attributes.pattern` changes).

For every affected pair it updates all three places that must agree:
  * ``intervention.new_value``
  * ``target.cot.entities[<id>].attributes.pattern``
  * ``target.cot_text`` (``<id>.attribute.pattern=<value>``)
and installs a prompt that spells out the target motif in concrete visual terms.

The source CoT / image are never touched.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict

# Rotated so the regenerated targets are not all identical in style.
REPLACEMENTS = ["striped", "polka_dot", "checkered"]

PATTERN_CUES = {
    "striped": "clear bold stripes in a strongly contrasting colour running across the whole surface",
    "polka_dot": "regular round polka dots in a strongly contrasting colour, evenly spaced over the whole surface",
    "checkered": "a regular two-colour checkerboard grid covering the whole surface",
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


def medium_phrase(entity: Dict) -> str:
    attrs = entity.get("attributes", {}) or {}
    used = {_stem(w) for w in natural(entity["category"]).lower().split()}
    parts: list[str] = []
    for key in ("color", "material"):
        value = natural(attrs.get(key, "")).strip()
        tokens = value.lower().split() if value else []
        if tokens and all(_stem(t) not in used for t in tokens):
            parts.append(value)
            used.update(_stem(t) for t in tokens)
    parts.append(natural(entity["category"]))
    return " ".join(parts)


def build_prompt(record: Dict, new_value: str) -> str:
    entity = next(e for e in record["source"]["cot"]["entities"] if e["id"] == record["intervention"]["entity_id"])
    desc = medium_phrase(entity)
    cue = PATTERN_CUES[new_value]
    return (
        f"Change only the pattern of the {desc} from solid to {natural(new_value)}. "
        f"{natural(new_value)} means {cue}. "
        "The difference must be unmistakable at a glance: the surface must clearly show the motif in a contrasting colour, "
        "not a faint or partial hint of it. A subtle or ambiguous change counts as a failure. "
        f"{FIXED_INVENTORY}Keep the object's silhouette, size, position, and count unchanged. {TAIL}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--old-value", default="solid")
    parser.add_argument("--bad-value", default="plain")
    args = parser.parse_args()

    records = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    affected: list[Dict] = []
    for record in records:
        iv = record.get("intervention", {})
        if iv.get("field") != "attributes.pattern":
            continue
        if str(iv.get("old_value")) != args.old_value or str(iv.get("new_value")) != args.bad_value:
            continue
        affected.append(record)

    for index, record in enumerate(affected):
        new_value = REPLACEMENTS[index % len(REPLACEMENTS)]
        iv = record["intervention"]
        entity_id = iv["entity_id"]
        iv["new_value"] = new_value

        for entity in record["target"]["cot"]["entities"]:
            if entity["id"] == entity_id:
                entity.setdefault("attributes", {})["pattern"] = new_value

        cot_text = record["target"]["cot_text"]
        token = f"{entity_id}.attribute.pattern={args.bad_value}"
        if token not in cot_text:
            raise SystemExit(f"{record['pair_id']}: expected token {token!r} in target cot_text")
        record["target"]["cot_text"] = cot_text.replace(token, f"{entity_id}.attribute.pattern={new_value}")

        # Sanity: source must still say `old_value`, and only pattern differs.
        src_entity = next(e for e in record["source"]["cot"]["entities"] if e["id"] == entity_id)
        if str(src_entity.get("attributes", {}).get("pattern")) != args.old_value:
            raise SystemExit(f"{record['pair_id']}: source pattern is not {args.old_value}")
        tgt_entity = next(e for e in record["target"]["cot"]["entities"] if e["id"] == entity_id)
        if tgt_entity["attributes"].get("pattern") != new_value:
            raise SystemExit(f"{record['pair_id']}: target pattern not updated")

        record["target"]["prompt"] = build_prompt(record, new_value)
        record["target"]["pattern_value_repaired"] = True

    with args.output_manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({
        "affected": len(affected),
        "replacements": {r["pair_id"]: (r["intervention"]["old_value"], r["intervention"]["new_value"]) for r in affected},
        "output_manifest": str(args.output_manifest),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
