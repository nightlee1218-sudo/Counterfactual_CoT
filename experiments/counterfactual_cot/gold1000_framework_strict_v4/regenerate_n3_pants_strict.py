#!/usr/bin/env python3
"""Regenerate the shared n3_pants scene and its three targets in isolation.

The original dataset is read-only.  This script deliberately does not use the
previous `LAYOUT_RULES` or `retry_v4` prompts: those prompts described multiple
"stacks" and therefore changed the meaning of the count intervention.

Here, count means individual folded garments inside ONE pile:
- source: one pile with 2 folded trousers and one pile with 3 folded shirts;
- 2_to_1: the trousers pile has 1 folded trouser;
- 2_to_3: the trousers pile has 3 folded trousers;
- 3_to_2: the shirts pile has 2 folded shirts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from experiments.counterfactual_cot.data.image_providers import build_provider
from experiments.counterfactual_cot.data.mask_utils import create_candidate_mask
from experiments.counterfactual_cot.data.schema import load_jsonl, write_jsonl

PAIR_IDS = (
    "g1000_closet_wardrobe__n3_pants__2_to_1_001",
    "g1000_closet_wardrobe__n3_pants__2_to_3_001",
    "g1000_closet_wardrobe__n3_pants__3_to_2_001",
)
SCENE_ID = "closet_wardrobe__n3_pants__v0330"

SOURCE_OBJECT_SENTENCE = (
    "Show one single compact pile containing exactly 2 individual khaki cotton folded trousers "
    "with surface appearance smooth fine weave and soft rounded folds at the left side; "
    "show one single compact pile containing exactly 3 individual white cotton folded shirts "
    "with surface appearance smooth fine weave and soft rounded folds in the center. "
    "Each number counts individual folded garments inside its one pile, not separate piles. "
    "Do not create multiple trouser piles, multiple shirt piles, duplicate garments, or garments on another level."
)

TARGET_PROMPTS = {
    "...__2_to_1_001": (
        "Change only the number of individual folded trousers inside the existing single khaki trouser pile from 2 to 1. "
        "Remove exactly one folded trouser layer from that same pile. The remaining one folded trouser stays in the same "
        "pile position, on the same shelf, with the same size and appearance. Do not create a second pile, do not move "
        "the pile to another shelf, and do not alter the white shirt pile. After the edit, exactly one individual folded "
        "trouser is visible in the single pile and exactly three individual folded shirts remain in their single pile. "
        "Do not add, remove, duplicate, or move any other object."
    ),
    "...__2_to_3_001": (
        "Change only the number of individual folded trousers inside the existing single khaki trouser pile from 2 to 3. "
        "Add exactly one folded trouser layer directly on top of or directly underneath the existing two layers, forming "
        "one compact pile of three folded trousers in the same position on the same shelf. Do not create a second pile, "
        "do not place the added garment in another area or on another shelf, and do not move the original pile. Do not "
        "alter the white shirt pile. After the edit, exactly three individual folded trousers are visible in the one pile "
        "and exactly three individual folded shirts remain in their one pile. Do not add, remove, duplicate, or move any other object."
    ),
    "...__3_to_2_001": (
        "Change only the number of individual folded shirts inside the existing single white shirt pile from 3 to 2. "
        "Remove exactly one folded shirt layer from that same pile. The remaining two folded shirts stay in the same pile "
        "position, on the same shelf, with the same size and appearance. Do not create a second pile, do not move the pile "
        "to another shelf, and do not alter the khaki trouser pile. After the edit, exactly two individual folded shirts are "
        "visible in the single pile and exactly two individual folded trousers remain in their single pile. Do not add, remove, duplicate, or move any other object."
    ),
}
PROMPT_TAIL = (
    " The edit must be obvious when the two images are compared side by side. "
    "Keep a natural 1024x1024 square image, full-bleed scene filling all four canvas edges. "
    "Never place the scene inside a photo, frame, card, poster, page, border, blank margin, white wedge, collage, or mockup. "
    "Use the same camera, lighting, wardrobe, shelf, background, and composition as the source image. "
    "No readable text, labels, letters, numbers, logos, watermarks, signs, or UI."
)


def _source_prompt(original: str) -> str:
    marker = "Show exactly 2 khaki cotton folded trousers with surface appearance smooth fine weave and soft rounded folds at the left side; exactly 3 white cotton folded shirt with surface appearance smooth fine weave and soft rounded folds in the center."
    if marker not in original:
        raise RuntimeError("original n3_pants source prompt format changed; refusing to patch by guess")
    return original.replace(marker, SOURCE_OBJECT_SENTENCE, 1)


def _provider(args: argparse.Namespace):
    return build_provider(
        "nano_banana_pro", model=args.model, api_key_env=args.api_key_env,
        endpoint=args.endpoint, timeout_s=args.timeout_s,
        max_retries=args.max_retries, retry_backoff_s=3.0,
        rate_limit_qps=args.rate_limit_qps, max_output_tokens=args.max_output_tokens,
    )


def _check_square(path: Path) -> None:
    with Image.open(path) as image:
        if image.size != (1024, 1024):
            raise RuntimeError(f"{path.name}: model returned {image.size}, expected (1024, 1024)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sandbox", required=True, type=Path)
    parser.add_argument("--model", default="gemini-3-pro-image")
    parser.add_argument("--api-key-env", default="CFCOT_IMAGE_API_KEY")
    parser.add_argument("--endpoint", default="http://v2.open.venus.oa.com/llmproxy/chat/completions")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--rate-limit-qps", type=float, default=0.2)
    parser.add_argument("--pair-id", choices=PAIR_IDS, default=None,
                        help="Regenerate only this pair; the original dataset is never changed.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_pairs = [args.pair_id] if args.pair_id else list(PAIR_IDS)
    records = {r["pair_id"]: r for r in load_jsonl(args.manifest)}
    missing = [pid for pid in PAIR_IDS if pid not in records]
    if missing:
        raise RuntimeError(f"missing pair IDs: {missing}")
    if any(records[pid]["source_scene_id"] != SCENE_ID for pid in PAIR_IDS):
        raise RuntimeError("n3_pants pair/source_scene_id mismatch")

    source_prompt = _source_prompt(records[PAIR_IDS[0]]["source"]["prompt"])
    target_prompts = {
        pid: TARGET_PROMPTS[pid.replace("g1000_closet_wardrobe__n3_pants", "...")]
        + PROMPT_TAIL
        for pid in selected_pairs
    }
    if args.dry_run:
        print(json.dumps({"source_prompt": source_prompt, "target_prompts": target_prompts}, ensure_ascii=False, indent=2))
        return

    sandbox = args.sandbox.resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    scene_path = sandbox / "source" / "_by_scene" / f"cf_source_{SCENE_ID}__source.png"
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    provider = _provider(args)
    ledger = sandbox / "regen_ledger.jsonl"

    def log(item: Dict[str, Any]) -> None:
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    tmp_scene = scene_path.with_suffix(".tmp.png")
    provider.generate_source(source_prompt, tmp_scene)
    _check_square(tmp_scene)
    os.replace(tmp_scene, scene_path)
    log({"stage": "source", "status": "success", "scene_id": SCENE_ID,
         "model": args.model, "prompt_sha256": hashlib.sha256(source_prompt.encode()).hexdigest()})

    out_records: List[Dict[str, Any]] = []
    for pid in selected_pairs:
        source_path = sandbox / "source" / f"{pid}__source.png"
        if source_path.exists():
            source_path.unlink()
        source_path.hardlink_to(scene_path)
        target_rel = Path("target") / f"{pid}__target.png"
        mask_rel = Path("masks") / f"{pid}__mask.png"
        target_path = sandbox / target_rel
        mask_path = sandbox / mask_rel
        target_tmp = target_path.with_suffix(".tmp.png")
        mask_tmp = mask_path.with_suffix(".tmp.png")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        provider.edit_target(source_path, target_prompts[pid], target_tmp)
        _check_square(target_tmp)
        create_candidate_mask(source_path, target_tmp, mask_tmp, threshold=24)
        os.replace(target_tmp, target_path)
        os.replace(mask_tmp, mask_path)
        record = json.loads(json.dumps(records[pid]))
        record["source"]["image"] = str(Path("source") / f"{pid}__source.png")
        record["target"]["image"] = str(target_rel)
        record["target"]["mask"] = str(mask_rel)
        record["target"]["prompt"] = target_prompts[pid]
        record["generation"] = {"provider": "nano_banana_pro", "model": args.model,
                                 "regenerated_with": "n3_pants_strict_single_pile"}
        out_records.append(record)
        log({"stage": "target", "status": "success", "pair_id": pid, "model": args.model,
             "prompt_sha256": hashlib.sha256(target_prompts[pid].encode()).hexdigest()})

    write_jsonl(sandbox / "manifest.jsonl", out_records)
    print(json.dumps({"sandbox": str(sandbox), "source": 1, "targets": len(out_records),
                      "semantics": "individual folded garments inside one pile"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
