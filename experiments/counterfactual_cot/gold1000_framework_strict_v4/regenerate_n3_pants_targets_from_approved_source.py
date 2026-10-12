#!/usr/bin/env python3
"""Generate the three n3_pants targets from the user-approved source only.

The approved source is not regenerated here.  All output is isolated in a new
sandbox; the original 1000-pair directory is read-only.

Count semantics are explicit: the count is the number of individual folded
clothing layers inside one compact pile, never the number of piles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict

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

TAIL = (
    "The edit must be obvious when the two images are compared side by side. "
    "Keep a natural 1024x1024 square, full-bleed scene filling the canvas edge to edge. "
    "Use the same camera, lighting, wardrobe, shelf, background, and composition as the source image. "
    "Do not introduce an inset image, frame, card, poster, page, border, blank margin, white wedge, collage, or mockup. "
    "Do not alter anything beyond this required atomic visual change. "
    "No readable text, labels, letters, numbers, logos, watermarks, signs, or UI."
)

PROMPTS = {
    PAIR_IDS[0]: (
        "Change only the number of individual folded trousers inside the existing single khaki trouser pile from 2 to 1. "
        "Remove exactly one folded trouser layer from that same pile. The remaining one folded trouser layer stays in "
        "the same pile, at the same position on the same shelf, with the same size and appearance. "
        "Do not create a second pile. Do not move the pile to another shelf or another area. "
        "Do not alter the single white-shirt pile. After the edit, exactly 1 individual folded trouser layer is visible "
        "inside the one khaki pile, and exactly 3 individual folded shirt layers remain inside the one white-shirt pile. "
        "Each number counts individual folded garment layers inside one pile, not separate piles. "
        "Do not add, remove, duplicate, or move any other object. "
        + TAIL
    ),
    PAIR_IDS[1]: (
        "Change only the number of individual folded trousers inside the existing single khaki trouser pile from 2 to 3. "
        "Add exactly one folded trouser layer directly on top of or directly underneath the existing two layers, so they "
        "form one compact pile of three folded trouser layers in the same position on the same shelf. "
        "The added layer must touch the existing pile and must not be placed beside the pile, on another shelf, in another "
        "compartment, or in a new area. Do not create a second pile. Do not move the original pile. "
        "Do not alter the single white-shirt pile. After the edit, exactly 3 individual folded trouser layers are visible "
        "inside the one khaki pile, and exactly 3 individual folded shirt layers remain inside the one white-shirt pile. "
        "Each number counts individual folded garment layers inside one pile, not separate piles. "
        "Do not add, remove, duplicate, or move any other object. "
        + TAIL
    ),
    PAIR_IDS[2]: (
        "Change only the number of individual folded shirts inside the existing single white-shirt pile from 3 to 2. "
        "Remove exactly one folded shirt layer from that same pile. The remaining two folded shirt layers stay in the same "
        "pile, at the same position on the same shelf, with the same size and appearance. "
        "Do not create a second pile. Do not move the pile to another shelf or another area. "
        "Do not alter the single khaki-trouser pile. After the edit, exactly 2 individual folded shirt layers are visible "
        "inside the one white-shirt pile, and exactly 2 individual folded trouser layers remain inside the one khaki pile. "
        "Each number counts individual folded garment layers inside one pile, not separate piles. "
        "Do not add, remove, duplicate, or move any other object. "
        + TAIL
    ),
}


def check_square(path: Path) -> None:
    with Image.open(path) as image:
        if image.size != (1024, 1024):
            raise RuntimeError(f"{path.name}: returned {image.size}, expected (1024, 1024)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Original manifest; read-only")
    parser.add_argument("--approved-source", required=True, type=Path, help="User-approved 1024x1024 source")
    parser.add_argument("--sandbox", required=True, type=Path)
    parser.add_argument("--model", default="gemini-3-pro-image")
    parser.add_argument("--api-key-env", default="CFCOT_IMAGE_API_KEY")
    parser.add_argument("--endpoint", default="http://v2.open.venus.oa.com/llmproxy/chat/completions")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--rate-limit-qps", type=float, default=0.2)
    args = parser.parse_args()

    records = {record["pair_id"]: record for record in load_jsonl(args.manifest)}
    if any(pair_id not in records for pair_id in PAIR_IDS):
        raise RuntimeError("manifest does not contain all three n3_pants pairs")
    if any(records[pair_id]["source_scene_id"] != SCENE_ID for pair_id in PAIR_IDS):
        raise RuntimeError("n3_pants source scene mismatch")
    if not args.approved_source.is_file():
        raise FileNotFoundError(args.approved_source)
    check_square(args.approved_source)

    sandbox = args.sandbox.resolve()
    original_root = args.manifest.resolve().parent
    if original_root in sandbox.parents or sandbox == original_root:
        raise RuntimeError("sandbox must be outside the original manifest directory")
    sandbox.mkdir(parents=True, exist_ok=True)

    source_scene = sandbox / "source" / "_by_scene" / f"cf_source_{SCENE_ID}__source.png"
    source_scene.parent.mkdir(parents=True, exist_ok=True)
    source_scene.write_bytes(args.approved_source.read_bytes())

    provider = build_provider(
        "nano_banana_pro", model=args.model, api_key_env=args.api_key_env,
        endpoint=args.endpoint, timeout_s=args.timeout_s, max_retries=args.max_retries,
        retry_backoff_s=3.0, rate_limit_qps=args.rate_limit_qps,
        max_output_tokens=args.max_output_tokens,
    )
    ledger = sandbox / "regen_ledger.jsonl"

    def log(entry: Dict[str, Any]) -> None:
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

    output_records = []
    for pair_id in PAIR_IDS:
        source_path = sandbox / "source" / f"{pair_id}__source.png"
        source_path.hardlink_to(source_scene)
        target_path = sandbox / "target" / f"{pair_id}__target.png"
        mask_path = sandbox / "masks" / f"{pair_id}__mask.png"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        target_tmp = target_path.with_suffix(".tmp.png")
        mask_tmp = mask_path.with_suffix(".tmp.png")
        started = time.monotonic()
        provider.edit_target(source_path, PROMPTS[pair_id], target_tmp)
        check_square(target_tmp)
        create_candidate_mask(source_path, target_tmp, mask_tmp, threshold=24)
        os.replace(target_tmp, target_path)
        os.replace(mask_tmp, mask_path)
        record = json.loads(json.dumps(records[pair_id]))
        record["source"]["image"] = str(Path("source") / f"{pair_id}__source.png")
        record["target"]["image"] = str(Path("target") / f"{pair_id}__target.png")
        record["target"]["mask"] = str(Path("masks") / f"{pair_id}__mask.png")
        record["target"]["prompt"] = PROMPTS[pair_id]
        record["generation"] = {
            "provider": "nano_banana_pro", "model": args.model,
            "regenerated_with": "approved_source_single_pile_count",
        }
        output_records.append(record)
        log({"pair_id": pair_id, "status": "success", "model": args.model,
             "latency_s": round(time.monotonic() - started, 1),
             "prompt_sha256": hashlib.sha256(PROMPTS[pair_id].encode()).hexdigest()})

    write_jsonl(sandbox / "manifest.jsonl", output_records)
    (sandbox / "prompts.json").write_text(json.dumps(PROMPTS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sandbox": str(sandbox), "source": "approved_source_copy", "targets": 3}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
