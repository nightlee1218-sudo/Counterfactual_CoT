#!/usr/bin/env python3
"""Score GEdit generations with the repo's official VIE-Score evaluator (SC only).

Why SC only
-----------
``VIEScore.evaluate`` issues two calls per item: SC (semantic: edit success +
over-editing) and PQ (perceptual quality). PQ measures image quality, which is
irrelevant to "did Gold40 improve editing", so we issue the SC call alone. That
halves the API cost and keeps the metric interpretable.

The prompt text, image ordering and parsing are taken verbatim from
``eval/gen/gedit/viescore`` so the numbers are comparable with BAGEL's official
GEdit run.

Notes
-----
* ``viescore/__init__.py`` does ``sys.path.insert(0, 'viescore')``, a *relative*
  path, so this script must chdir into ``eval/gen/gedit`` before importing.
* The upstream wrapper builds an ``AzureOpenAI`` client; we replace it with a
  plain OpenAI-compatible client pointed at the internal Venus endpoint.
* ``mllm_output_to_dict(..., give_up_parsing=True)`` invents a random score. We
  never use that path: a parse failure is retried, then recorded as null.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

BAGEL_ROOT = Path(__file__).resolve().parents[3]
VIESCORE_CWD = BAGEL_ROOT / "eval/gen/gedit"

DEFAULT_ENDPOINT = "http://v2.open.venus.oa.com/llmproxy"
# VIE-Score's upstream default backbone. NOTE: the fully versioned name
# "gpt-4.1-2025-04-14" is NOT permitted on this app group; the alias is.
DEFAULT_MODEL = "gpt-4.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="generation manifest jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-env", default="VENUS_API_KEY")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="do not resize the edited image to the source size (upstream does resize)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing API secret in {args.token_env}")

    os.chdir(VIESCORE_CWD)
    sys.path.insert(0, str(VIESCORE_CWD))
    sys.path.insert(0, str(VIESCORE_CWD / "viescore"))

    from viescore import VIEScore  # noqa: E402
    from utils import mllm_output_to_dict  # noqa: E402

    import openai  # noqa: E402

    vs = VIEScore(backbone="gpt4o", task="tie", key_path=token, azure_endpoint="")
    vs.model.client = openai.OpenAI(base_url=args.endpoint, api_key=token, timeout=args.timeout)
    vs.model.model_name = args.model
    # VIEScore.evaluate() sets this from the input type; we call prepare_prompt
    # directly, so set it explicitly to get base64 data URLs (official behaviour
    # for local PIL inputs).
    vs.model.use_encode = True

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def score_one(row: dict[str, Any]) -> dict[str, Any]:
        source = Image.open(row["image_path"]).convert("RGB")
        edited = Image.open(row["output_path"]).convert("RGB")
        if not args.no_resize:
            edited = edited.resize((source.size[0], source.size[1]))
        text_prompt = vs.SC_prompt.replace("<instruction>", row["prompt"])
        content = vs.model.prepare_prompt([source, edited], text_prompt)

        last_raw = ""
        for attempt in range(1, args.retries + 1):
            try:
                raw = vs.model.get_parsed_output(content)
                last_raw = raw
                parsed = mllm_output_to_dict(raw, give_up_parsing=False)
                if parsed and isinstance(parsed, dict) and parsed.get("score"):
                    scores = [float(v) for v in parsed["score"]]
                    return {
                        "id": row["id"],
                        "variant": row["variant"],
                        "bench": "gedit",
                        "task_type": row.get("meta", {}).get("task_type"),
                        "seed": row.get("seed"),
                        "sc_edit_success": scores[0] if len(scores) > 0 else None,
                        "sc_overedit_penalty": scores[1] if len(scores) > 1 else None,
                        "sc": min(scores),
                        "reasoning": str(parsed.get("reasoning", ""))[:600],
                        "attempts": attempt,
                    }
            except Exception as exc:  # noqa: BLE001 - transient API/parse errors
                if attempt >= args.retries:
                    return {
                        "id": row["id"],
                        "variant": row["variant"],
                        "bench": "gedit",
                        "task_type": row.get("meta", {}).get("task_type"),
                        "seed": row.get("seed"),
                        "sc_edit_success": None,
                        "sc_overedit_penalty": None,
                        "sc": None,
                        "reasoning": f"FAILED: {type(exc).__name__}: {exc}"[:600],
                        "attempts": attempt,
                    }
                time.sleep(2.0 * attempt)
        return {
            "id": row["id"],
            "variant": row["variant"],
            "bench": "gedit",
            "task_type": row.get("meta", {}).get("task_type"),
            "seed": row.get("seed"),
            "sc_edit_success": None,
            "sc_overedit_penalty": None,
            "sc": None,
            "reasoning": f"UNPARSEABLE: {last_raw[:400]}",
            "attempts": args.retries,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    from concurrent.futures import ThreadPoolExecutor

    scored: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        for result in pool.map(score_one, rows):
            scored.append(result)
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            print(json.dumps({"event": "scored", "id": result["id"], "sc": result["sc"]}), flush=True)

    valid = [r["sc"] for r in scored if r["sc"] is not None]
    summary = {
        "bench": "gedit",
        "evaluator": "viescore_sc_only",
        "model": args.model,
        "variant": scored[0]["variant"] if scored else None,
        "num_items": len(scored),
        "num_failed": len(scored) - len(valid),
        "sc_mean": round(sum(valid) / len(valid), 4) if valid else None,
        "sc_edit_success_mean": round(
            sum(r["sc_edit_success"] for r in scored if r["sc_edit_success"] is not None)
            / max(1, len(valid)),
            4,
        ) if valid else None,
        "sc_overedit_mean": round(
            sum(r["sc_overedit_penalty"] for r in scored if r["sc_overedit_penalty"] is not None)
            / max(1, len(valid)),
            4,
        ) if valid else None,
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
