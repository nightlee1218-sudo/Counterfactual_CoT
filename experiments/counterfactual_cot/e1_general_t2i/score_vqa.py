#!/usr/bin/env python3
"""E1 optional VQA scoring: binary prompt-image faithfulness via Qwen3-VL (venus llmproxy).

Public-model pool is rate-limited (~1/min), so this runs on a capped, resumable
subset by default (--limit-per-model 60 => 120 calls). CLIP-T remains the
primary metric; VQA rows are merged into stats only when available.

Output:
  scores/vqa_{model}.jsonl   per-item rows {item_id, vqa_match, reason, ...}

Requires env ENV_VENUS_OPENAPI_SECRET_ID (same as gold40 v5 labeling).
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

import requests

HERE = Path(__file__).resolve().parent
DEFAULT_ENDPOINT = "http://v2.open.venus.oa.com/llmproxy/chat/completions"
DEFAULT_MODEL = "qwen3-vl-235b-a22b-thinking"
DEFAULT_TOKEN_ENV = "ENV_VENUS_OPENAPI_SECRET_ID"
TOKEN_SUFFIX = "@2198"

SYSTEM_PROMPT = """You are a strict text-to-image evaluator. The user gives you an image and the exact prompt it was generated from. Judge whether the image faithfully satisfies the prompt, focusing on the prompt's binding requirements (attributes, colors, shapes, textures), spatial relations, and object counts.
Return JSON only, no Markdown, no extra keys:
{
  "match": true,
  "confidence": "high",
  "reason": "brief evidence"
}
"match" is true only if every requirement in the prompt is satisfied (objects present with the right attributes/counts/relations, no missing or wrong-bound elements). Minor style differences do not count as failures."""


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def extract_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError(f"no choices: {response}")
    content = choices[0].get("message", {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [x.get("text", "") for x in content if isinstance(x, dict)]
        return "".join(parts)
    raise ValueError(f"no content: {response}")


def parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"not JSON: {text[:300]}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict) or not isinstance(value.get("match"), bool):
        raise ValueError(f"bad fields: {value}")
    return value


def call_vqa(session: requests.Session, endpoint: str, token: str, model: str,
             prompt: str, image: Path, timeout: float = 180.0, retries: int = 4,
             rate_limit_wait: float = 65.0) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"PROMPT: {prompt}"},
                {"type": "image_url", "image_url": {"url": image_data_url(image)}},
            ]},
        ],
        "temperature": 0,
        "max_tokens": 400,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 429:
                if attempt >= retries:
                    raise RuntimeError(f"HTTP 429 after {retries} attempts: {response.text[:300]}")
                print(f"429 rate limit; sleeping {rate_limit_wait:.0f}s ({attempt}/{retries - 1})", file=sys.stderr, flush=True)
                time.sleep(rate_limit_wait)
                continue
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            return parse_json_object(extract_text(response.json()))
        except (requests.RequestException, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(60.0, 2.0 ** (attempt - 1)))
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gen-root", type=Path, default=HERE / "../outputs/e1_general_t2i/gen")
    parser.add_argument("--out-dir", type=Path, default=HERE / "../outputs/e1_general_t2i/scores")
    parser.add_argument("--models", nargs="*", default=["base", "final991"])
    parser.add_argument("--limit-per-model", type=int, default=60, help="0 = all items (not recommended on public pool)")
    parser.add_argument("--stride", type=int, default=4, help="sample every Nth item to spread coverage across subsets")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV)
    args = parser.parse_args()

    token = os.environ.get(args.token_env, "").strip()
    if not token:
        print(json.dumps({"event": "skip_no_token", "env": args.token_env}), flush=True)
        return 2
    token = token if token.endswith(TOKEN_SUFFIX) else token + TOKEN_SUFFIX

    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for model_name in args.models:
        manifest = args.gen_root / model_name / "generation_manifest.jsonl"
        if not manifest.exists():
            continue
        records = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
        records = records[:: max(args.stride, 1)]
        if args.limit_per_model:
            records = records[: args.limit_per_model]
        out_path = args.out_dir / f"vqa_{model_name}.jsonl"
        done: set[str] = set()
        if out_path.exists():
            for l in out_path.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    done.add(json.loads(l)["item_id"])
        rows_written = 0
        for rec in records:
            if rec["item_id"] in done:
                continue
            image = Path(rec["output_path"])
            if not image.exists():
                continue
            try:
                answer = call_vqa(session, args.endpoint, token, args.model_name,
                                  rec["prompt"], image)
            except Exception as exc:  # noqa: BLE001
                print(json.dumps({"event": "vqa_error", "item_id": rec["item_id"], "error": str(exc)[:200]}), flush=True)
                continue
            row = {"item_id": rec["item_id"], "subset": rec["subset"], "source_category": rec["source_category"],
                   "model": model_name, "vqa_match": answer["match"],
                   "confidence": answer.get("confidence"), "reason": answer.get("reason")}
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            rows_written += 1
            print(json.dumps({"event": "vqa", "model": model_name, "item_id": rec["item_id"],
                              "match": row["vqa_match"]}), flush=True)
        print(json.dumps({"event": "vqa_model_done", "model": model_name, "rows": rows_written}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
