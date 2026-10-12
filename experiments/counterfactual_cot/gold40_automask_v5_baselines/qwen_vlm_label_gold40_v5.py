#!/usr/bin/env python3
"""Create explicit Qwen3-VL semantic labels for same-seed Gold40 generations."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import requests

DEFAULT_ENDPOINT = "http://v2.open.venus.oa.com/llmproxy/chat/completions"
DEFAULT_MODEL = "qwen3-vl-235b-a22b-thinking"
DEFAULT_TOKEN_ENV = "ENV_VENUS_OPENAPI_SECRET_ID"
TOKEN_SUFFIX = "@2198"
STATES = ("source_state", "target_state")

SYSTEM_PROMPT = """You are a strict visual evaluator for a counterfactual image-generation experiment.
You must compare each generated image with the corresponding reference image and the
provided semantic specification. Judge semantic correctness, not pixel similarity.
Return JSON only, with no Markdown and no extra keys:
{
  \"source_state_correct\": true,
  \"target_state_correct\": false,
  \"confidence\": \"high\",
  \"reason\": \"brief evidence for both judgments\"
}
A state is correct only if all required objects, counts, attributes, locations,
relations, and background facts are satisfied, and no unrelated object or major
unintended change is introduced. The source-state generation must match the source
reference state; the target-state generation must match the target reference state.
"""


def read_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: record must be an object")
        rows.append(value)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def endpoint_path(value: Any, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: missing path")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: file does not exist: {path}")
    return path


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def state_name(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"source", "source_state"}:
        return "source_state"
    if normalized in {"target", "target_state"}:
        return "target_state"
    raise ValueError(f"invalid state: {value!r}")


def pair_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    pair_id = str(row.get("pair_id") or "").strip()
    if not pair_id or row.get("seed") is None:
        raise ValueError("generation row requires pair_id and seed")
    return pair_id, int(row["seed"]), state_name(row.get("state"))


def extract_text(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"API response has no choices: {response}")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "".join(parts)
    raise ValueError(f"API response has no textual message content: {response}")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Qwen response is not JSON: {text[:500]}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Qwen response JSON must be an object")
    for key in ("source_state_correct", "target_state_correct"):
        if not isinstance(value.get(key), bool):
            raise ValueError(f"Qwen response field {key} must be boolean: {value}")
    if value.get("confidence") not in {"high", "medium", "low"}:
        raise ValueError(f"Qwen response confidence must be high/medium/low: {value}")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError(f"Qwen response reason must be non-empty text: {value}")
    return value


def call_qwen(
    session: requests.Session,
    endpoint: str,
    token: str,
    model: str,
    text: str,
    images: list[Path],
    timeout: float,
    retries: int,
    rate_limit_wait: float,
) -> tuple[dict[str, Any], str | None]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for path in images:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(path)}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": 800,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                try:
                    wait_seconds = max(rate_limit_wait, float(retry_after)) if retry_after else rate_limit_wait
                except ValueError:
                    wait_seconds = rate_limit_wait
                if attempt >= retries:
                    raise RuntimeError(f"HTTP 429 after {retries} attempts: {response.text[:1000]}")
                print(
                    f"HTTP 429 public-model rate limit; waiting {wait_seconds:.1f}s before retry "
                    f"({attempt}/{retries - 1})",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            body = response.json()
            answer = parse_json_object(extract_text(body))
            response_id = body.get("id") if isinstance(body.get("id"), str) else None
            return answer, response_id
        except (requests.RequestException, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(60.0, 2.0 ** (attempt - 1)))
    assert last_error is not None
    raise last_error


def build_spec(pair: Mapping[str, Any]) -> str:
    source = pair.get("source")
    target = pair.get("target")
    intervention = pair.get("intervention")
    preservation = pair.get("preservation")
    return "\n".join((
        "Evaluate the four images in this exact order:",
        "1. SOURCE_REFERENCE: the intended source state.",
        "2. TARGET_REFERENCE: the intended target state.",
        "3. GENERATED_SOURCE_STATE: model output conditioned on source Target CoT.",
        "4. GENERATED_TARGET_STATE: model output conditioned on target Target CoT.",
        "\nSemantic pair specification:",
        json.dumps({
            "pair_id": pair.get("pair_id"),
            "source_scene_id": pair.get("source_scene_id"),
            "intervention": intervention,
            "preservation": preservation,
            "source_cot": source.get("cot_text") if isinstance(source, Mapping) else None,
            "target_cot": target.get("cot_text") if isinstance(target, Mapping) else None,
        }, ensure_ascii=False, indent=2),
        "\nJudge each generated state independently against its matching reference and semantic specification.",
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-auto-manifest", required=True, type=Path)
    parser.add_argument("--generation-manifest", required=True, type=Path)
    parser.add_argument("--output-label-manifest", required=True, type=Path)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--rate-limit-wait", type=float, default=65.0, help="Minimum seconds to wait after HTTP 429")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries must be positive")

    secret_id = os.environ.get(args.token_env)
    if not secret_id:
        raise RuntimeError(f"missing API secret in environment variable {args.token_env}; do not hardcode credentials")
    token = secret_id + TOKEN_SUFFIX

    pairs = {str(row.get("pair_id")): row for row in read_records(args.v5_auto_manifest)}
    generations: dict[tuple[str, int, str], Path] = {}
    for row in read_records(args.generation_manifest):
        key = pair_key(row)
        if key in generations:
            raise ValueError(f"duplicate generation key: {key}")
        generations[key] = endpoint_path(row.get("output_path"), args.generation_manifest.parent, f"generation {key}")
    grouped: dict[tuple[str, int], dict[str, Path]] = {}
    for (pair_id, seed, state), path in generations.items():
        grouped.setdefault((pair_id, seed), {})[state] = path
    if not grouped:
        raise ValueError("generation manifest has no rows")
    for key, states in grouped.items():
        if set(states) != set(STATES):
            raise ValueError(f"{key} must contain both source_state and target_state")
        if key[0] not in pairs:
            raise ValueError(f"pair missing from v5 manifest: {key[0]}")

    existing: dict[tuple[str, int, str], dict[str, Any]] = {}
    if args.output_label_manifest.exists():
        for row in read_records(args.output_label_manifest):
            key = pair_key(row)
            if key in existing:
                raise ValueError(f"duplicate existing label key: {key}")
            existing[key] = row

    jobs = sorted(grouped.items())
    if args.limit is not None:
        jobs = jobs[:args.limit]
    args.output_label_manifest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for index, ((pair_id, seed), states) in enumerate(jobs, 1):
        keys = [(pair_id, seed, state) for state in STATES]
        if all(key in existing for key in keys):
            print(json.dumps({"event": "skip_existing", "pair_id": pair_id, "seed": seed}, ensure_ascii=False), flush=True)
            continue
        pair = pairs[pair_id]
        source = pair.get("source")
        target = pair.get("target")
        if not isinstance(source, Mapping) or not isinstance(target, Mapping):
            raise ValueError(f"{pair_id}: source/target objects are required")
        reference_images = [
            endpoint_path(source.get("image"), args.v5_auto_manifest.parent, f"{pair_id}.source.reference"),
            endpoint_path(target.get("image"), args.v5_auto_manifest.parent, f"{pair_id}.target.reference"),
        ]
        answer, response_id = call_qwen(
            session, args.endpoint, token, args.model, build_spec(pair),
            reference_images + [states["source_state"], states["target_state"]],
            args.timeout, args.retries, args.rate_limit_wait,
        )
        rows = [
            {
                "schema_version": "gold40_automask_v5_qwen3_vl_labels_v1",
                "pair_id": pair_id,
                "seed": seed,
                "state": "source_state",
                "automated_semantic_correct": answer["source_state_correct"],
                "judge_model": args.model,
                "judge_endpoint": args.endpoint,
                "confidence": answer["confidence"],
                "reason": answer["reason"],
                "response_id": response_id,
            },
            {
                "schema_version": "gold40_automask_v5_qwen3_vl_labels_v1",
                "pair_id": pair_id,
                "seed": seed,
                "state": "target_state",
                "automated_semantic_correct": answer["target_state_correct"],
                "judge_model": args.model,
                "judge_endpoint": args.endpoint,
                "confidence": answer["confidence"],
                "reason": answer["reason"],
                "response_id": response_id,
            },
        ]
        with args.output_label_manifest.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                existing[(row["pair_id"], row["seed"], row["state"])] = row
        print(json.dumps({"event": "judged", "index": index, "total": len(jobs), "pair_id": pair_id, "seed": seed, "source_state_correct": answer["source_state_correct"], "target_state_correct": answer["target_state_correct"], "confidence": answer["confidence"]}, ensure_ascii=False), flush=True)
    print(json.dumps({"status": "completed", "label_rows": len(existing), "expected_label_rows": len(generations)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
