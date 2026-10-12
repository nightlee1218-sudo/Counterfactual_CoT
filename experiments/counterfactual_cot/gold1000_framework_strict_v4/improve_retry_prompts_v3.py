#!/usr/bin/env python3
"""Outcome-first retry prompts (v3) for the five stubborn failed pairs.

The four "no visible change" pairs and the one empty-mask pair proved immune
to the step-by-step ("Step 1: erase ... Step 2: redraw") instruction style, so
v3 switches strategy entirely: describe what the OUTPUT IMAGE must look like,
in short sentences, with no constraint pile-up.  Every sentence leads with the
required visual state.

Also fixes an operational gap: previously an empty-mask failure (target
identical to source) aborted before the retry logic could run; the trainer now
supports ``--no-change-retries`` so these pairs get several attempts.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

SCRIPT_DIR = Path(__file__).resolve().parent
CHUNK_DIR = SCRIPT_DIR / "full_1000_20260918" / "chunks"

RETRY_V3_PROMPTS: Dict[str, str] = {
    # unoccluded -> partially_occluded_by_wooden_cutting_board (empty-mask pair)
    # v3.1: the board must stay lying flat (the first v3 output stood it
    # upright, which is physically wrong and diverges from the source pose).
    # Natural occlusion at the slightly elevated camera: slide the dish to the
    # far side of the flat board so its near edge covers part of the dish.
    "g1000_kitchen_counter__toaster_dish_jar__occlusion_state_006": (
        "The output image keeps both objects lying flat on the countertop. "
        "The wooden cutting board stays in its exact original position and orientation, completely flat on the counter. "
        "Only the white ceramic butter dish moves slightly toward the far side of the board, so the near edge of the "
        "flat board now covers about one third of the dish from the camera's view. "
        "The hidden part of the dish is clearly not visible behind the board. "
        "Nothing is stood upright: the board never tilts, never stands on its edge, and never changes its flat pose. "
        "Both objects keep their exact material, color, and shape, and both rest on the same countertop. "
        "Everything else in the scene stays exactly the same as the source image."
    ),
    # partially_occluded_by_bedside_lamp -> unoccluded (edit happened but invisible)
    "g1000_bedroom_nightstand__glasses_charger_glass__occlusion_state_003": (
        "The output image shows the clear glass completely visible: nothing stands in front of it, nothing overlaps it. "
        "The cream ceramic bedside lamp has been picked up and placed at a clearly different spot on the same surface, "
        "far away from the glass, with a wide empty gap between them. Both objects are fully visible at the same time. "
        "The closed book, the nightstand, the background, the lighting, and the camera stay exactly the same as the "
        "source image."
    ),
    # location right_region -> upper_right (no visible change)
    "g1000_bookshelf_study__mugcollection_folder__absolute_vertical_001": (
        "The output image shows the gray cardboard magazine folder standing on the UPPER shelf level of the bookshelf, "
        "one full level higher than in the source image. It has been physically lifted and placed upright on that "
        "upper level, same size and orientation, not floating and not tilted. The lower spot where it used to stand "
        "is now visibly empty. The hardcover book, the shelf, the background, the lighting, and the camera stay "
        "exactly the same as the source image."
    ),
    # count 3 -> 2 (no visible change)
    "g1000_classroom__notebook_fill__3_to_2_001": (
        "The output image shows exactly TWO green paper notebooks on the desk where the source image shows three. "
        "One notebook has been completely removed, and the bare desk surface shows through where it used to lie. "
        "The two remaining notebooks are untouched: same positions, same size, same details as the source image. "
        "The pencil cup, the eraser, the background, the lighting, and the camera stay exactly the same as the "
        "source image."
    ),
    # count 2 -> 3 (no visible change)
    "g1000_warehouse_shelf__wrench_crate_boxes__2_to_3_001": (
        "The output image shows exactly THREE brown cardboard boxes on the shelf where the source image shows two. "
        "One additional identical box stands right beside the other two, clearly visible and easy to count, the same "
        "size as the existing boxes. Everything else in the scene, including the shelf, the background, the lighting, "
        "and the camera, stays exactly the same as the source image."
    ),
}


def check_prompt(pair_id: str, prompt: str) -> None:
    if re.search(r"\b[a-z]+_\d+\b", prompt) or "_region" in prompt:
        raise SystemExit(f"{pair_id}: internal token leaked into v3 prompt")
    if re.search(r"\b(\w+)\s+\1\b", prompt.lower()):
        raise SystemExit(f"{pair_id}: repeated-word bigram in v3 prompt")


def main() -> None:
    patched: Dict[str, int] = {}
    patched_ids: set = set()
    import re as _re

    chunk_paths = sorted(
        path for path in CHUNK_DIR.glob("chunk_*.jsonl")
        if _re.fullmatch(r"chunk_\d{2}\.jsonl", path.name)
    )
    for chunk_path in chunk_paths:
        hit_ids = [pid for pid in RETRY_V3_PROMPTS if str(pid).rsplit("__", 1)[0] in chunk_path.read_text(encoding="utf-8")]
        records = []
        hits = []
        with chunk_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("pair_id") in RETRY_V3_PROMPTS:
                    record["target"]["prompt"] = RETRY_V3_PROMPTS[record["pair_id"]]
                    record["target"]["prompt_retry_v3"] = True
                    check_prompt(record["pair_id"], record["target"]["prompt"])
                    hits.append(record["pair_id"])
                records.append(record)
        if not hits:
            continue
        output_path = chunk_path.with_name(chunk_path.stem + ".retry_v3.jsonl")
        with output_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        patched[chunk_path.name] = len(hits)
        patched_ids.update(hits)
        for pair_id in hits:
            print(f"patched {pair_id} -> {output_path.name}")

    missing = sorted(set(RETRY_V3_PROMPTS) - patched_ids)
    print(json.dumps({"patched_total": len(patched_ids), "chunks": patched, "expected": len(RETRY_V3_PROMPTS), "missing": missing}, indent=2))
    if missing:
        raise SystemExit("not every stubborn pair was patched")


if __name__ == "__main__":
    main()
