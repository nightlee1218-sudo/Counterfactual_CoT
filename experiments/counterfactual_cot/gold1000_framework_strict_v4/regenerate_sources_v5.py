#!/usr/bin/env python3
"""Regenerate specific source scenes with extra spatial-layout constraints.

Several scenes produced sources whose physical layout made the intended edit
impossible: duplicated furniture rows, objects stacked on top of each other so
no occlusion could be created, or two entities crammed into the same region with
no free space to add a third instance.

For each requested scene this appends a LAYOUT block to the original conditioning
text that pins down where every entity must sit and how much free space must
remain, then regenerates the shared scene image into an isolated sandbox.

The dataset is never modified: output goes to a sandbox, original stays read-only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

from experiments.counterfactual_cot.data.image_providers import build_provider
from experiments.counterfactual_cot.data.generate_source_images import source_conditioning_text
from experiments.counterfactual_cot.data.schema import load_jsonl

PROVIDER = "nano_banana_pro"

# scene_id -> extra layout requirements appended to the conditioning text
LAYOUT_RULES = {
    # wet/dry edits were invisible on light fabric or soap
    "bathroom_counter__rinsecup_soap_towel__v0044": (
        "LAYOUT REQUIREMENT: the bar of soap on the counter is DARK CHARCOAL GREY, not white and not ivory. "
        "It sits on the bathroom counter beside the rinse cup, clearly visible and unobstructed, with even lighting. "
        "A dark soap bar shows wetness clearly, so keep its surface clean and evenly lit."
    ),
    "bathroom_counter__rinsecup_soap_towel__v0045": (
        "LAYOUT REQUIREMENT: the bar of soap on the counter is DARK CHARCOAL GREY, not white and not ivory. "
        "It sits on the bathroom counter beside the rinse cup, clearly visible and unobstructed, with even lighting. "
        "A dark soap bar shows wetness clearly, so keep its surface clean and evenly lit."
    ),
    "bathroom_counter__rinsecup_soap_towel__v0046": (
        "LAYOUT REQUIREMENT: the folded bath towel is DEEP NAVY BLUE, not white and not ivory. "
        "It lies folded on the bathroom counter, clearly visible and unobstructed, with even lighting. "
        "A dark towel shows dampness clearly, so keep its surface clean and evenly lit."
    ),
    "bathroom_counter__rinsecup_soap_towel__v0047": (
        "LAYOUT REQUIREMENT: the folded bath towel is DEEP NAVY BLUE, not white and not ivory. "
        "It lies folded on the bathroom counter, clearly visible and unobstructed, with even lighting. "
        "A dark towel shows dampness clearly, so keep its surface clean and evenly lit."
    ),
    # duplicated rows: one row only, counts must match the CoT
    "closet_wardrobe__n3_pants__v0330": (
        "LAYOUT REQUIREMENT: the wardrobe must show TWO clearly visible rigid horizontal shelf levels. "
        "LOWER LEVEL holds all the folded laundry in ONE single row: exactly TWO folded khaki trouser stacks side by side "
        "on the LEFT half of this level, and exactly THREE folded white shirt stacks side by side on the RIGHT half of "
        "the same level - FIVE stacks in total on this level, never four and never six, clearly separated and easy to "
        "count at a glance. Leave a clear empty gap on this level immediately to the RIGHT of the two trouser stacks, "
        "wide enough for one more trouser stack to be placed directly beside them. "
        "UPPER LEVEL stays empty, or holds only a small clearly separated spare item, leaving obvious free space. "
        "Every stack rests fully and flatly on its shelf plank: NOTHING floats, NOTHING hovers, NOTHING leans, "
        "and no stack sits on top of another stack. "
        "The trousers are only on the lower level: never split the trousers across two levels."
    ),
    # stacked objects: the mill must stand ON the counter, beside the board
    "kitchen_counter__bowl_block_mill__v0594": (
        "LAYOUT REQUIREMENT: the wooden cutting board lies completely flat on the counter surface in the right part of "
        "the scene. The black pepper mill stands upright ON THE COUNTER, NOT on top of the board, placed to the left of "
        "the board with a clear gap between them. The two objects do not touch and do not overlap, and there is empty "
        "counter space behind the board so the mill could later be moved behind it."
    ),
    # knife block and board stacked in the same spot
    "kitchen_counter__bowl_block_mill__v0590": (
        "LAYOUT REQUIREMENT: the brown knife block stands upright in the centre of the counter. The wooden cutting board "
        "lies completely flat on the counter to the RIGHT of the knife block, with a clear empty gap of at least one "
        "object width between them. The two objects do not touch, do not overlap, and are not stacked."
    ),
    # tiny glasses right next to a bulky lamp: no occlusion room
    "bedroom_nightstand__glasses_charger_glass__v0069": (
        "LAYOUT REQUIREMENT: the ceramic mug stands upright on the nightstand in the FRONT-CENTRE area, clearly visible "
        "with its full body showing, and it is tall enough to be partly hidden later by the lamp. "
        "The cream ceramic bedside lamp stands at the BACK of the nightstand, clearly separated from the mug by a wide "
        "empty gap, so it could later be moved forward in front of the mug. "
        "The blue closed book lies flat in the CENTRE of the nightstand. "
        "Nothing overlaps: the mug, the lamp and the book are all fully and separately visible."
    ),
    # shelf too cramped to add a third bin without moving the drums
    "warehouse_shelf__bin_drum_strapping__v0800": (
        "LAYOUT REQUIREMENT: the shelf has three clearly separated zones. LEFT ZONE: exactly TWO yellow plastic storage "
        "bins standing side by side, with a wide empty gap of at least one bin width immediately to their right, free of "
        "any object. CENTRE-RIGHT ZONE: exactly TWO green metal drums, standing upright and well separated from the bins. "
        "BOTTOM ZONE: exactly THREE black strapping bundles. The gap beside the yellow bins must be empty so a third bin "
        "can be added there later without moving the drums or the bundles."
    ),
    # container state kept stable when its contents are taken out
    "warehouse_shelf__fasteners_bin_sacks__v0778": (
        "LAYOUT REQUIREMENT: the red open plastic small bin sits on the left side of the shelf with its lid open exactly "
        "as a storage bin with a hinged lid that is raised. The gray metal fastener box sits INSIDE the red bin, clearly "
        "contained and partly below the rim. Leave clear empty shelf space immediately to the right of the red bin so the "
        "gray box can later be placed there. Keep the red bin's lid position and shape stable."
    ),
}




# Appended to every layout rule: the regeneration must stay a native square,
# otherwise the mask step fails with a source/target size mismatch.
SQUARE_CONSTRAINT = (
    "FORMAT: the output must be a native SQUARE image, exactly square in shape, 1:1 aspect ratio. "
    "Do not produce a wide or panoramic image. "
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--source-image-root", required=True, type=Path)
    parser.add_argument("--sandbox", required=True, type=Path)
    parser.add_argument("--scenes", required=True, help="Comma-separated scene ids")
    parser.add_argument("--model", default="gemini-3-pro-image")
    parser.add_argument("--api-key-env", default="CFCOT_IMAGE_API_KEY")
    parser.add_argument("--endpoint", default="http://v2.open.venus.oa.com/llmproxy/chat/completions")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--rate-limit-qps", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    args = parser.parse_args()

    sandbox: Path = args.sandbox
    if args.source_image_root.resolve() in sandbox.resolve().parents or sandbox.resolve() == args.source_image_root.resolve():
        raise SystemExit("sandbox must not live inside the original image root")
    sandbox.mkdir(parents=True, exist_ok=True)

    registry = {r["scene_id"]: r for r in load_jsonl(args.source_registry)}
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    provider = build_provider(PROVIDER, model=args.model, api_key_env=args.api_key_env, endpoint=args.endpoint,
                              timeout_s=args.timeout_s, max_retries=args.max_retries, retry_backoff_s=3.0,
                              rate_limit_qps=args.rate_limit_qps, max_output_tokens=args.max_output_tokens)

    ledger = sandbox / "source_regen_ledger.jsonl"
    for scene_id in scenes:
        if scene_id not in registry:
            print(f"SKIP unknown scene {scene_id}")
            continue
        if scene_id not in LAYOUT_RULES:
            print(f"SKIP scene without layout rule: {scene_id}")
            continue
        item = registry[scene_id]
        prompt = source_conditioning_text(item) + "\n\n" + SQUARE_CONSTRAINT + LAYOUT_RULES[scene_id]
        rel = Path("source") / "_by_scene" / f"cf_source_{scene_id}__source.png"
        out = sandbox / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        started = time.monotonic()
        try:
            provider.generate_source(prompt, tmp)
        except Exception as exc:  # noqa: BLE001
            if tmp.exists():
                tmp.unlink()
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"scene_id": scene_id, "status": "failed", "error": str(exc)}, ensure_ascii=False) + "\n")
            print(f"FAILED {scene_id}: {exc}")
            continue
        from PIL import Image

        with Image.open(tmp) as check:
            if check.size != (1024, 1024):
                check.convert("RGB").resize((1024, 1024)).save(tmp)
        os.replace(tmp, out)
        with ledger.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"scene_id": scene_id, "status": "success", "model": args.model,
                                 "latency_s": round(time.monotonic() - started, 1),
                                 "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest()[:16]},
                                ensure_ascii=False) + "\n")
        print(f"OK {scene_id}")

    ok = sum(1 for l in ledger.read_text(encoding="utf-8").splitlines() if json.loads(l).get("status") == "success") if ledger.exists() else 0
    print(json.dumps({"sandbox": str(sandbox), "requested": len(scenes), "succeeded": ok}, indent=2))


if __name__ == "__main__":
    main()
