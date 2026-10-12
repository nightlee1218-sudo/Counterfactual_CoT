#!/usr/bin/env python3
"""Score mini-benchmark generations (GEdit / WISE / GenEval) with a VLM judge.

The official scorers cannot run here: GenEval needs mmdet + open_clip (not
installed) and WISE/GEdit need an OpenAI judge. We use the same Qwen3-VL
endpoint already used by this repo (gold40_automask_v5_baselines), and score
BASE and Gold40 with an identical prompt/rubric so the deltas are comparable.

Outputs one jsonl with per-item sub-scores plus a summary json.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

DEFAULT_ENDPOINT = "http://v2.open.venus.oa.com/llmproxy/chat/completions"
# qwen3-vl-235b-a22b-thinking is not enabled for this app group; gpt-4o is
# reachable and accepts image_url input, so it is the judge for all benches.
DEFAULT_MODEL = "gpt-4o"
TOKEN_ENV = "VENUS_API_KEY"
TOKEN_SUFFIX = ""

COMMON_RULES = """You are a strict visual evaluator. Judge semantics only, not pixel similarity or
aesthetic quality. Never give partial credit: a sub-score is 1.0 only if the
requirement is fully met, 0.5 if it is partially/ambiguously met, 0.0 otherwise.
Return JSON only (no Markdown), with exactly these keys and no extras."""

GEdit_SCHEMA = """{
  "edit_success": <0|0.5|1>,
  "target_grounding": <0|0.5|1>,
  "preservation": <0|0.5|1>,
  "locality": <0|0.5|1>,
  "reason": "<=40 words, concrete visual evidence>"
}
Scoring:
- edit_success: the requested edit was actually performed.
- target_grounding: the edit was applied to the correct object/region.
- preservation: all unrelated objects and the overall identity of the scene are kept.
- locality: no changes outside the intended region (no global restyle, no layout drift)."""

WISE_SCHEMA = """{
  "prompt_following": <0|0.5|1>,
  "factual_correctness": <0|0.5|1>,
  "multi_element": <0|0.5|1>,
  "reason": "<=40 words, concrete visual evidence>"
}
Scoring:
- prompt_following: every entity/relation named in the prompt is present.
- factual_correctness: world knowledge conveyed by the prompt is respected.
- multi_element: when the prompt names several elements, all are correct and
  correctly related (score 1 if only one element is required and it is correct)."""

GENEVAL_SCHEMA = """{
  "object_correctness": <0|0.5|1>,
  "count_correctness": <0|0.5|1>,
  "spatial_correctness": <0|0.5|1>,
  "attribute_correctness": <0|0.5|1>,
  "reason": "<=40 words, concrete visual evidence>"
}
Scoring:
- object_correctness: all requested object classes appear.
- count_correctness: the number of instances of every class matches exactly.
- spatial_correctness: every requested positional relation holds.
- attribute_correctness: every requested colour/material attribute is on the right object.
- Score a non-applicable dimension 1 (e.g. no colour requested -> attribute 1)."""

BENCH = {
    "gedit": GEdit_SCHEMA,
    "wise": WISE_SCHEMA,
    "geneval": GENEVAL_SCHEMA,
}


def image_data_url(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def parse_json(text: str, required: list[str]) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"judge response is not JSON: {text[:400]}")
        value = json.loads(match.group(0))
    out: dict[str, Any] = {}
    for key in required:
        raw = value.get(key)
        raw = 0.0 if raw is None else float(raw)
        out[key] = max(0.0, min(1.0, raw))
    out["reason"] = str(value.get("reason", ""))[:400]
    return out


def call_judge(
    session: requests.Session, endpoint: str, token: str, model: str,
    text: str, images: list[Path], timeout: float, retries: int, wait: float,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for path in images:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": COMMON_RULES},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": 600,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    for attempt in range(1, retries + 1):
        try:
            response = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 429:
                time.sleep(wait)
                continue
            if response.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:400]} "
                    f"(model={model}, images={len(images)}, token_len={len(token)})"
                )
            body = response.json()
            choices = body.get("choices") or []
            message = choices[0].get("message", {}) if choices else {}
            content_field = message.get("content")
            if isinstance(content_field, str):
                return content_field
            if isinstance(content_field, list):
                return "".join(p.get("text", "") for p in content_field)
            raise ValueError(f"unexpected response: {str(body)[:400]}")
        except Exception as exc:  # noqa: BLE001 - transient API errors
            if attempt >= retries:
                raise RuntimeError(f"judge call failed after {retries} attempts: {exc}") from exc
            time.sleep(wait)
    raise RuntimeError("unreachable")


def build_prompt(bench: str, row: dict[str, Any]) -> str:
    meta = row.get("meta", {}) or {}
    if bench == "gedit":
        return (
            "IMAGE 1 is the SOURCE image. IMAGE 2 is the EDITED result.\n"
            f"Edit instruction: {row['prompt']}\n"
            f"Edit type: {meta.get('task_type', '')}\n"
            "Judge how well the instruction was executed while everything else is preserved.\n"
            + BENCH[bench]
        )
    if bench == "wise":
        hint = meta.get("hint") or ""
        return (
            "IMAGE 1 is the generated image for the prompt below.\n"
            f"Prompt: {row['prompt']}\n"
            f"Category: {meta.get('category', '')} / {meta.get('subcategory', '')}\n"
            f"Expected answer reference: {hint}\n"
            "Judge whether the image depicts the requested content correctly.\n"
            + BENCH[bench]
        )
    return (
        "IMAGE 1 is the generated image for the prompt below.\n"
        f"Prompt: {row['prompt']}\n"
        f"Requirements: {json.dumps(meta.get('include', []), ensure_ascii=False)}\n"
        "Judge whether every requirement is satisfied exactly.\n"
        + BENCH[bench]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", choices=("gedit", "wise", "geneval"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-env", default=TOKEN_ENV)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--wait", type=float, default=8.0)
    args = parser.parse_args()

    secret = os.environ.get(args.token_env)
    if not secret:
        raise RuntimeError(f"missing API secret in {args.token_env}")
    token = secret + TOKEN_SUFFIX

    rows = [json.loads(l) for l in args.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    keys = [k for k in ("edit_success", "target_grounding", "preservation", "locality",
                        "prompt_following", "factual_correctness", "multi_element",
                        "object_correctness", "count_correctness", "spatial_correctness",
                        "attribute_correctness") if k in BENCH[args.bench]]

    session = requests.Session()
    scored: list[dict[str, Any]] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        images = [Path(row["output_path"])]
        if args.bench == "gedit":
            images.insert(0, Path(row["image_path"]))
        text = build_prompt(args.bench, row)
        raw = call_judge(session, args.endpoint, token, args.model, text, images, args.timeout, args.retries, args.wait)
        result = parse_json(raw, keys)
        result.update({"id": row["id"], "variant": row["variant"], "bench": args.bench, "meta": row.get("meta", {})})
        scored.append(result)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"event": "scored", "index": index, "id": row["id"], **{k: result[k] for k in keys}}), flush=True)

    summary = {
        "bench": args.bench,
        "variant": rows[0]["variant"] if rows else None,
        "num_items": len(scored),
        "mean": {k: round(sum(r[k] for r in scored) / max(1, len(scored)), 4) for k in keys},
        "overall": round(sum(sum(r[k] for k in keys) / len(keys) for r in scored) / max(1, len(scored)), 4),
    }
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **summary}), flush=True)


if __name__ == "__main__":
    main()
