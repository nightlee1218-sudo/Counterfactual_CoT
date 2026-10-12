#!/usr/bin/env python3
"""Qwen3-VL triage audit for the final counterfactual pair manifest.

This is a conservative pre-audit only. It never edits the dataset manifest or
human review CSV and never marks a pair as training-accepted. It writes one
JSON object per pair to an isolated output JSONL. Any API/parse/visual
uncertainty is routed to ``needs_human_review=true``.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import requests
from PIL import Image

DEFAULT_ENDPOINT = "http://v2.open.venus.oa.com/llmproxy/chat/completions"
DEFAULT_MODEL = "qwen3-vl-235b-a22b-instruct"
DEFAULT_TOKEN_ENV = "VENUS_OPENAPI_TOKEN"
DEFAULT_TOKEN_SUFFIX = "@2198"
SCORES = (
    "preservation_score",
    "binding_score",
    "locality_score",
    "image_quality_score",
)
THRESHOLDS = {
    "target_success": 1.0,
    "preservation_score": 0.8,
    "binding_score": 0.8,
    "locality_score": 0.7,
    "image_quality_score": 0.6,
}

SYSTEM_PROMPT = """You are a conservative first-pass judge for a counterfactual image-editing dataset. Your output only triages pairs for human review; it never authorizes training acceptance.
You receive exactly three images in this order: image 1 SOURCE, image 2 TARGET, image 3 CANDIDATE_MASK. Judge visible pixels against the structured specification. Never assume the target is correct because the prompt says it should be correct.

MANDATORY DATASET STANDARD:
1. SOURCE must match source_cot: every named entity, individual count, category, colour, material, location and relation must be visibly correct. If the source itself is wrong, source_correct=false.
2. TARGET must perform exactly ONE atomic intervention: only intervention.field for intervention.entity_id may change. Do not accept a visually plausible but different edit.
3. Count semantics are INDIVIDUAL OBJECTS. For folded garments, count individual garments/layers inside ONE pile; do NOT count piles, stacks, rows, or containers. A 2-to-3 trousers edit means one existing pile containing two individual trousers becomes the same one pile containing three trousers. A second pile, a new shelf/area, duplication of the whole pile, or 6 total trousers is wrong.
4. For count edits, existing instances and their distinctive appearance must be preserved. Removing a candle must not replace a wax-dripped candle with a new clean candle. Adding one must add exactly one instance, not copy the whole group.
5. For spatial edits, only the named entity may move. Its count, identity, size, orientation and distinctive details must remain stable. Unrelated objects must not move, rotate, resize, shorten, disappear, or change material. A necessary local occlusion/lighting consequence is allowed only if it is physically caused by the specified edit.
6. For occlusion/containment edits, use the EXISTING named objects. No new cutting board, lamp, box, or duplicate object may be invented. No existing unrelated object may disappear or change state. The requested relation must be visually unmistakable, not merely a tiny ambiguous overlap.
7. For material/pattern/functional-state edits, the change must be clearly visible at a glance on the correct object while its position, count, silhouette and identity stay stable. Subtle/ambiguous wet-dry or material changes are failures.
8. Preservation is strict: every must_preserve entity, background, camera, composition, lighting and unrelated object remains stable. If an object is moved merely to make room, treat it as a preservation failure unless that object is the intervention target.
9. The mask is supporting evidence only. A non-empty or localized mask does not prove semantic correctness; a large/global mask is a warning.

Only set target_success=false when a visible, specific violation is present in TARGET. Do not call a pair failed merely because the edit is subtle, difficult to verify, or because the specification is ambiguous; in those cases set target_success=true, confidence=low or medium, and add an uncertainty issue code so it is routed to human review. When a clear violation is visible, fail closed and name the target object and violation. Return JSON only, no Markdown, no pair_id, and exactly these keys: source_correct, target_success, preservation_score, binding_score, locality_score, image_quality_score, confidence, issue_codes, reason. All five scores are numbers from 0 to 1.
"""

OUTPUT_SCHEMA = {
    "source_correct": True,
    "target_success": True,
    "preservation_score": 0.0,
    "binding_score": 0.0,
    "locality_score": 0.0,
    "image_quality_score": 0.0,
    "confidence": "high",
    "issue_codes": [],
    "reason": "brief evidence grounded in visible image details",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: record must be an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def resolve_asset(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: missing path")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label}: asset escapes image root: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    with Image.open(path) as image:
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"{label}: invalid image size {image.size}")
    return path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def extract_text(body: Mapping[str, Any]) -> str:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response contains no choices")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = "".join(
            item.get("text", "") for item in content
            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
        )
        if text:
            return text
    raise ValueError("API response contains no textual message content")


def _bool_value(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "pass", "passed", "correct", "accepted", "合格", "正确"}:
            return True
        if text in {"false", "no", "fail", "failed", "incorrect", "rejected", "不合格", "错误"}:
            return False
    raise ValueError(f"{field} cannot be normalized to boolean: {value!r}")


def _list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value)]


def _text_value(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if value is not None and not isinstance(value, (dict, list)):
        return str(value)
    return default


def _judgment(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key in ("is_correct", "correct", "pass", "passed", "accepted", "success"):
            if key in value:
                return _bool_value(value[key], f"judgment.{key}")
        for key in ("judgment", "result", "label", "status"):
            if key in value:
                return _judgment(value[key])
    return _bool_value(value, "judgment")


def parse_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise ValueError(f"response is not JSON: {text[:500]}")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")

    # Qwen has returned several valid semantic schemas despite the requested
    # schema: judgment/audit_result/is_correct plus issues/notes/explanation.
    # Normalize them instead of routing all such responses to audit_error.
    if "source_correct" in value:
        source_correct = _bool_value(value["source_correct"], "source_correct")
    else:
        source_correct = True
    if "target_success" in value:
        target_success = _bool_value(value["target_success"], "target_success")
    else:
        raw_judgment = value.get("judgment", value.get("audit_result", value.get("is_correct")))
        if raw_judgment is None:
            raise ValueError(f"response has no judgment field: {sorted(value)}")
        target_success = _judgment(raw_judgment)

    issue_codes: list[str] = []
    if "source_correct" not in value:
        issue_codes.append("missing_source_correct")
    for key in ("issue_codes", "issues", "issue_code"):
        issue_codes.extend(_list_value(value.get(key)))
    reason = _text_value(
        value.get("reason", value.get("explanation", value.get("notes", value.get("reasoning")))),
        "Qwen returned a judgment without detailed evidence.",
    )
    confidence = _text_value(value.get("confidence"), "medium").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    # If the model did not emit numeric scores, fail closed for human review.
    # We still retain the normalized judgment so the triage list is useful.
    normalized: dict[str, Any] = {
        "source_correct": source_correct,
        "target_success": target_success,
        "confidence": confidence,
        "issue_codes": issue_codes,
        "reason": reason,
    }
    for field in SCORES:
        raw = value.get(field)
        try:
            score = float(raw)
            if not 0.0 <= score <= 1.0:
                raise ValueError
        except (TypeError, ValueError):
            # The observed alternate Qwen schema reports a binary judgment but
            # omits numeric scores. Preserve the judgment for triage and infer
            # threshold-level provisional scores; these are never human scores.
            score = THRESHOLDS[field] if target_success else 0.0
            normalized.setdefault("issue_codes", []).append("scores_inferred_from_judgment")
        normalized[field] = score
    normalized["issue_codes"] = sorted(set(normalized["issue_codes"]))
    return normalized


def auth_token(secret: str, suffix: str) -> str:
    secret = secret.strip()
    if not suffix or secret.endswith(suffix):
        return secret
    return secret + suffix


def audit_prompt(pair: Mapping[str, Any]) -> str:
    source = pair.get("source", {})
    target = pair.get("target", {})
    spec = {
        "pair_id": pair.get("pair_id"),
        "source_scene_id": pair.get("source_scene_id"),
        "subtype": pair.get("subtype"),
        "intervention": pair.get("intervention"),
        "preservation": pair.get("preservation"),
        "source_cot": source.get("cot_text"),
        "target_cot": target.get("cot_text"),
    }
    return (
        "IMAGE ORDER: image 1=SOURCE, image 2=TARGET, image 3=CANDIDATE_MASK.\n"
        "Audit this one pair independently. Treat the JSON below as the specification, not as evidence that the edit succeeded.\n"
        "Audit the TARGET as the primary object of review. Pay special attention to exact individual counts inside one pile, preservation of distinctive survivor details such as wax drips, "
        "no invented/deleted objects, no displaced or resized unrelated objects, no changed orientation for a move, and whether a material/functional/pattern change is clearly visible.\n"
        "Only a clear visible violation is a failure. Mere uncertainty is not a failure: keep target_success=true but use confidence=low or medium and add an uncertainty issue code.\n"
        "For every clear failure, name the target object and state the exact visible violation.\n"
        "Return exactly the required JSON schema and do not include pair_id in the response.\n\nSPECIFICATION:\n"
        + json.dumps(spec, ensure_ascii=False, indent=2)
    )


def call_model(session: requests.Session, endpoint: str, token: str, model: str,
               prompt: str, images: list[Path], timeout: float, retries: int,
               retry_wait: float, max_tokens: int) -> tuple[dict[str, Any], str | None]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for label, image in zip(("SOURCE", "TARGET", "CANDIDATE_MASK"), images):
        content.append({"type": "text", "text": f"[{label}]"})
        content.append({"type": "image_url", "image_url": {"url": data_url(image)}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.post(endpoint, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", retry_wait))
                if attempt == retries:
                    raise RuntimeError(f"HTTP 429 after {retries} attempts")
                time.sleep(max(retry_wait, wait))
                continue
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")
            body = response.json()
            answer = parse_json(extract_text(body))
            return answer, body.get("id") if isinstance(body.get("id"), str) else None
        except (requests.RequestException, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(min(60.0, 2.0 ** (attempt - 1)))
    assert last is not None
    raise last


def decision(answer: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    if not answer["source_correct"]:
        failures.append("source_state_incorrect")
    if not answer["target_success"]:
        failures.append("target_intervention_incorrect")
    for field, threshold in THRESHOLDS.items():
        if field == "target_success":
            continue
        if answer[field] < threshold:
            failures.append(f"{field}_below_threshold")
    if answer["confidence"] != "high":
        failures.append("confidence_not_high")
    return (not failures), failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV)
    parser.add_argument("--token-suffix", default=DEFAULT_TOKEN_SUFFIX)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--retry-wait", type=float, default=65)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--pair-list", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.retries < 1 or args.max_tokens < 256:
        parser.error("--retries must be positive and --max-tokens must be >=256")

    secret = os.environ.get(args.token_env)
    if not secret:
        raise RuntimeError(f"missing token environment variable {args.token_env}; credentials are never read from source code")
    token = auth_token(secret, args.token_suffix)
    records = {str(r["pair_id"]): r for r in read_jsonl(args.manifest)}
    selected = set(records)
    if args.pair_list:
        selected = {x.strip() for x in args.pair_list.read_text(encoding="utf-8").splitlines() if x.strip()}
        unknown = selected - set(records)
        if unknown:
            raise ValueError(f"pair list contains unknown IDs: {sorted(unknown)[:5]}")
    jobs = [(pid, records[pid]) for pid in sorted(selected)]
    if args.limit is not None:
        jobs = jobs[:args.limit]

    existing: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        for row in read_jsonl(args.output):
            existing[str(row["pair_id"])] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    with args.output.open("a", encoding="utf-8") as handle:
        for index, (pid, pair) in enumerate(jobs, 1):
            try:
                source = resolve_asset(args.image_root, pair.get("source", {}).get("image"), f"{pid}:source")
                target = resolve_asset(args.image_root, pair.get("target", {}).get("image"), f"{pid}:target")
                mask = resolve_asset(args.image_root, pair.get("target", {}).get("mask"), f"{pid}:mask")
                fingerprint = hashlib.sha256(
                    (pid + "\n" + file_sha256(source) + "\n" + file_sha256(target) + "\n" + file_sha256(mask)).encode()
                ).hexdigest()
                if pid in existing and existing[pid].get("input_fingerprint") == fingerprint:
                    continue
                answer, response_id = call_model(
                    session, args.endpoint, token, args.model, audit_prompt(pair),
                    [source, target, mask], args.timeout, args.retries, args.retry_wait, args.max_tokens,
                )
                passed, failures = decision(answer)
                result = {
                    "schema_version": "counterfactual_qwen_vl_triage_v1",
                    "pair_id": pid,
                    "source_scene_id": pair.get("source_scene_id"),
                    "judge_model": args.model,
                    "judge_endpoint": args.endpoint,
                    "input_fingerprint": fingerprint,
                    "source_correct": answer["source_correct"],
                    "target_success": answer["target_success"],
                    **{field: answer[field] for field in SCORES},
                    "confidence": answer["confidence"],
                    "issue_codes": sorted(set(answer["issue_codes"] + failures)),
                    "reason": answer["reason"],
                    "auto_candidate_pass": passed,
                    "needs_human_review": not passed,
                    "response_id": response_id,
                    "error": None,
                }
            except Exception as exc:  # fail closed: every uncertain pair is reviewed by a human
                result = {
                    "schema_version": "counterfactual_qwen_vl_triage_v1",
                    "pair_id": pid,
                    "source_scene_id": pair.get("source_scene_id"),
                    "judge_model": args.model,
                    "judge_endpoint": args.endpoint,
                    "input_fingerprint": None,
                    "auto_candidate_pass": False,
                    "needs_human_review": True,
                    "issue_codes": ["audit_error"],
                    "reason": str(exc),
                    "response_id": None,
                    "error": str(exc),
                }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(f"[{index}/{len(jobs)}] {pid}: {'PASS_CANDIDATE' if result['auto_candidate_pass'] else 'HUMAN_REVIEW'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
