#!/usr/bin/env python3
"""v4 prompt redesign driven by the 2026-09-22 human review.

Hard constraints added in response to the observed failure modes:

1. NO NEW OBJECTS -- especially for occlusion edits.  The editor was inventing a
   second cutting board to create the occlusion instead of moving the existing
   one.  Every prompt now states the object inventory is fixed.
2. NO DUPLICATION -- for count edits the editor copied the existing instances
   instead of adding/removing exactly one, producing 2 -> 6 instead of 2 -> 3.
3. COUNT IS PRESERVED for location edits (the editor changed 3 blocks into 2
   while moving them) and for containment edits.
4. HIGH-CONTRAST MATERIAL/PATTERN -- the change must be obvious at a glance or
   the sample cannot train anything.
5. OTHER OBJECTS STAY PUT -- moving one object must not shuffle the rest.

The planning manifests are never modified: patched copies are written as
``chunk_XX.retry_v4.jsonl``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

SCRIPT_DIR = Path(__file__).resolve().parent
CHUNK_DIR = SCRIPT_DIR / "full_1000_20260918" / "chunks"

COUNT_ADD = {"1_to_2", "1_to_3", "2_to_3"}
COUNT_REMOVE = {"2_to_1", "3_to_1", "3_to_2"}

# material pairs whose visual difference the editor must exaggerate
MATERIAL_CUES = {
    ("wire", "plastic_mesh"): (
        "wire means thin metal wires with open gaps you can see through; "
        "plastic mesh means a moulded plastic grid with solid matte strands and a uniform surface"
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


def natural(value: Any) -> str:
    return str(value).replace("_", " ")


def medium_phrase(entity: Dict[str, Any]) -> str:
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


def kind_of(record: Dict[str, Any]) -> str:
    subtype = str(record["subtype"])
    if subtype in COUNT_ADD:
        return "count_add"
    if subtype in COUNT_REMOVE:
        return "count_remove"
    if subtype == "occlusion_state":
        return "occlusion"
    if subtype in ("absolute_horizontal", "absolute_vertical"):
        return "spatial"
    if subtype == "containment":
        return "containment"
    if subtype in ("material", "pattern"):
        return "surface"
    return "attribute"


def new_prompt(record: Dict[str, Any]) -> str:
    kind = kind_of(record)
    iv = record["intervention"]
    entity = next(e for e in record["source"]["cot"]["entities"] if e["id"] == iv["entity_id"])
    desc = medium_phrase(entity)
    old_v, new_v = iv["old_value"], iv["new_value"]

    if kind == "count_add":
        delta = int(new_v) - int(old_v)
        # NOTE: FIXED_INVENTORY is deliberately NOT used here -- it would
        # contradict "add exactly N" and the editor froze or mis-placed the new
        # instance.  Count edits get their own scoped inventory rule.
        return (
            f"Add exactly {delta} new {desc} to the scene. "
            f"PLACE IT DIRECTLY BESIDE the existing {desc}: immediately adjacent to them, on the SAME shelf plank or "
            f"surface, in the SAME row and the SAME section, at touching distance. Do NOT put it on another shelf level, "
            f"in another compartment, or in a different area of the scene. "
            f"CRITICAL: do NOT copy or duplicate the existing {desc} into extra positions, and do NOT add more than {delta}. "
            f"After the edit the scene contains exactly {new_v} {desc} in total (the original {old_v} plus {delta} new), clearly countable at one glance. "
            f"The {old_v} original instances stay exactly where they are, same size and same appearance. "
            f"OBJECT INVENTORY: apart from the {delta} added instance(s), no other object is added, duplicated, removed "
            f"or moved; every other object stays in its exact original position. {TAIL}"
        )

    if kind == "count_remove":
        delta = int(old_v) - int(new_v)
        return (
            f"Remove exactly {delta} of the {desc}, for example the rightmost one. "
            f"CRITICAL: do NOT remove more than {delta}, and do NOT move or restack the survivors. "
            f"Erase the removed one completely and let the bare surface show through where it stood: no outline, no shadow, no leftover. "
            f"After the edit exactly {new_v} {desc} remain, not {int(new_v) - 1} and not {int(new_v) + 1}. "
            f"The surviving {desc} keep their exact original positions, size, and every detail. "
            f"OBJECT INVENTORY: apart from the {delta} removed instance(s), no other object is added, duplicated, "
            f"removed or moved; every other object stays in its exact original position. {TAIL}"
        )

    if kind == "occlusion":
        ref = next(r for r in entity.get("relations", []) if r["predicate"] == "occlusion_reference")
        occluder = next(e for e in record["source"]["cot"]["entities"] if e["id"] == ref["target"])
        occ = medium_phrase(occluder)
        if str(new_v) == "unoccluded":
            geometry = (
                f"Move the EXISTING {occ} away so it no longer overlaps the {desc}: pick it up and place it at a clearly "
                f"different spot on the same surface, at least one object width away, with the complete outline of the {desc} visible."
            )
        else:
            geometry = (
                f"Move the EXISTING {occ} in front of the {desc} so it hides about one third of the {desc}. "
                f"Do NOT create a second {occ}: there is only one {occ} in the scene, and it is the same one that was already there, just repositioned."
            )
        return (
            f"Change only the visibility of the {desc}. {geometry} "
            f"After the edit both objects remain visible and both are the SAME objects as in the source image; only their relative arrangement differs. "
            f"{FIXED_INVENTORY}"
            "Preserve every object identity, count, non-occlusion attribute, background, lighting, and camera exactly. "
            f"{TAIL}"
        )

    if kind == "spatial":
        count = entity.get("count", 1)
        return (
            f"Move the {desc} from {natural(old_v)} to {natural(new_v)}. "
            f"CRITICAL: keep exactly {count} {desc} in the scene; do not add one, do not remove one, and do not duplicate it. "
            "Only the position changes: size, orientation, colour, and material stay identical, and nothing is left at the old position. "
            f"The object's centre must end up at least one third of the frame away from where it started. "
            f"{FIXED_INVENTORY}Do not move, resize, or alter any other object in the scene. {TAIL}"
        )

    if kind == "containment":
        inside_now = str(old_v).startswith("inside")
        container_id = str(old_v).split(":", 1)[1] if ":" in str(old_v) else ""
        container = next((e for e in record["source"]["cot"]["entities"] if e["id"] == container_id), None)
        cont = medium_phrase(container) if container else "container"
        if inside_now:
            action = (
                f"Take the {desc} out of the {cont} and set it down beside the {cont} on the same surface, clearly outside it. "
                f"The {cont} stays exactly where it is and stays open and visible."
            )
        else:
            action = (
                f"Place the {desc} inside the {cont}, fully contained by it. "
                f"The {cont} stays exactly where it is."
            )
        return (
            f"Change only the spatial relation of the {desc}. {action} "
            f"{FIXED_INVENTORY}"
            "Both the contained object and the container remain visible. Do not change the count, colour, material, or identity of anything. "
            f"{TAIL}"
        )

    if kind == "surface":
        cue = MATERIAL_CUES.get((str(old_v), str(new_v)))
        contrast = (
            f" {cue}. " if cue else
            f" The {natural(new_v)} surface must look clearly different from the {natural(old_v)} surface. "
        )
        return (
            f"Change only the {record['subtype']} of the {desc} from {natural(old_v)} to {natural(new_v)}.{contrast}"
            "The difference must be unmistakable at a glance on the object's own surface: distinct texture, distinct sheen, distinct colour response. "
            "A subtle or ambiguous change counts as a failure. "
            f"{FIXED_INVENTORY}Keep the object's silhouette, size, position, and count unchanged. {TAIL}"
        )

    return str(record["target"]["prompt"]).rstrip() + " " + FIXED_INVENTORY + TAIL


def check_prompt(pair_id: str, prompt: str) -> None:
    if re.search(r"\b[a-z]+_\d+\b", prompt) or "_region" in prompt:
        raise SystemExit(f"{pair_id}: internal token leaked into v4 prompt")
    if re.search(r"\b(\w+)\s+\1\b", prompt.lower()):
        raise SystemExit(f"{pair_id}: repeated-word bigram in v4 prompt")


def main() -> None:
    import re as _re

    pair_file = SCRIPT_DIR / "flagged_20260922_v4.txt"
    targets = {l.strip() for l in pair_file.read_text(encoding="utf-8").splitlines() if l.strip()}
    chunk_paths = sorted(p for p in CHUNK_DIR.glob("chunk_*.jsonl") if _re.fullmatch(r"chunk_\d{2}\.jsonl", p.name))

    patched_ids, per_chunk = set(), {}
    for path in chunk_paths:
        records, hits = [], []
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("pair_id") in targets:
                record["target"]["prompt"] = new_prompt(record)
                record["target"]["prompt_retry_v4"] = True
                check_prompt(record["pair_id"], record["target"]["prompt"])
                hits.append(record["pair_id"])
            records.append(record)
        if not hits:
            continue
        out = path.with_name(path.stem + ".retry_v4.jsonl")
        with out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        per_chunk[path.name] = len(hits)
        patched_ids.update(hits)

    missing = sorted(targets - patched_ids)
    print(json.dumps({"patched": len(patched_ids), "expected": len(targets), "chunks": per_chunk, "missing": missing}, indent=2))
    if missing:
        raise SystemExit("not every flagged pair was patched")


if __name__ == "__main__":
    main()
