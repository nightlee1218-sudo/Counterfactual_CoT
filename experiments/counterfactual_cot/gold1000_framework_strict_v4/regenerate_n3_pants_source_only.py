#!/usr/bin/env python3
"""Generate only the n3_pants source image in an isolated sandbox.

This intentionally does not generate targets.  The source must be visually
approved first because all three n3_pants pairs share the same source scene.
The original manifest and original image root are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from PIL import Image

from experiments.counterfactual_cot.data.generate_source_images import source_conditioning_text
from experiments.counterfactual_cot.data.image_providers import build_provider
from experiments.counterfactual_cot.data.schema import load_jsonl

SCENE_ID = "closet_wardrobe__n3_pants__v0330"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--sandbox", required=True, type=Path)
    parser.add_argument("--model", default="gemini-3-pro-image")
    parser.add_argument("--api-key-env", default="CFCOT_IMAGE_API_KEY")
    parser.add_argument("--endpoint", default="http://v2.open.venus.oa.com/llmproxy/chat/completions")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=16384)
    parser.add_argument("--rate-limit-qps", type=float, default=0.2)
    args = parser.parse_args()

    registry = {r["scene_id"]: r for r in load_jsonl(args.source_registry)}
    if SCENE_ID not in registry:
        raise SystemExit(f"missing source scene: {SCENE_ID}")
    item = registry[SCENE_ID]

    # Keep the repository's normal SOURCE_ORACLE_COT format and original
    # source prompt.  Only disambiguate what the count means: individual folded
    # garments inside one compact pile, not multiple piles or rows.
    prompt = source_conditioning_text(item) + (
        "\nInterpret the garment counts literally as individual folded garments "
        "inside one compact pile: make one single khaki-trouser pile containing "
        "exactly 2 individual folded trousers, and one single white-shirt pile "
        "containing exactly 3 individual folded shirts. Do not create multiple "
        "trouser piles, multiple shirt piles, duplicate rows, or any extra clothing. "
        "All five individual garments must rest fully on the same visible shelf "
        "level and must be clearly countable from their separated folded edges."
    )

    sandbox = args.sandbox.resolve()
    source_root = args.source_registry.resolve().parents[1] / "source_images"
    if source_root in sandbox.parents or sandbox == source_root:
        raise SystemExit("sandbox must be outside the original source image root")
    out = sandbox / "source" / "_by_scene" / f"cf_source_{SCENE_ID}__source.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")

    provider = build_provider(
        "nano_banana_pro", model=args.model, api_key_env=args.api_key_env,
        endpoint=args.endpoint, timeout_s=args.timeout_s,
        max_retries=args.max_retries, retry_backoff_s=3.0,
        rate_limit_qps=args.rate_limit_qps, max_output_tokens=args.max_output_tokens,
    )
    started = time.monotonic()
    provider.generate_source(prompt, tmp)
    with Image.open(tmp) as image:
        if image.size != (1024, 1024):
            raise RuntimeError(f"source model returned {image.size}; expected (1024, 1024)")
    os.replace(tmp, out)

    report = {
        "scene_id": SCENE_ID,
        "status": "success",
        "model": args.model,
        "path": str(out),
        "size": [1024, 1024],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "latency_s": round(time.monotonic() - started, 1),
        "targets_generated": False,
        "original_manifest_read_only": True,
    }
    (sandbox / "source_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (sandbox / "source_prompt.txt").write_text(prompt, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
