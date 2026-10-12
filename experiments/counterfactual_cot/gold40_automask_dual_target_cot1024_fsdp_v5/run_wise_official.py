#!/usr/bin/env python3
"""Score WISE generations with the repo's official WISE evaluator rubric.

The prompt, the 0-2 scale and the parsing regex are copied verbatim from
``eval/gen/wise/gpt_eval_mp.py`` so the numbers match BAGEL's official WISE run.
Only the transport changes: the upstream script uses an AzureOpenAI client with a
placeholder endpoint, we use the internal Venus endpoint.

Scoring (per the repo's ``cal_score.py``):
    WiScore = 0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality
and the aggregate divides by 2 to normalise 0-2 scales back to 0-1. That
normalisation is applied in ``analyze_v2.py``, not here.

Model note: the upstream script requests ``gpt-4o-2024-05-13``, but that fully
versioned name is rejected by the Venus gateway; the ``gpt-4o`` alias resolves to
exactly that snapshot, so it is the faithful equivalent.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

BAGEL_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_ENDPOINT = "http://v2.open.venus.oa.com/llmproxy"
DEFAULT_MODEL = "gpt-4o"

SYSTEM_PROMPT = (
    "You are a professional Vincennes image quality audit expert, please evaluate "
    "the image quality strictly according to the protocol."
)


def build_user_prompt(prompt: str, explanation: str) -> str:
    """Verbatim copy of eval/gen/wise/gpt_eval_mp.py::build_evaluation_messages."""
    return f"""Please evaluate strictly and return ONLY the three scores as requested.

# Text-to-Image Quality Evaluation Protocol

## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. Only images meeting the HIGHEST standards should receive top scores.

**Input Parameters**
- PROMPT: [User's original prompt to]
- EXPLANATION: [Further explanation of the original prompt]
---

## Scoring Criteria

**Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
* **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

**Realism (0-2):**  How realistically the image is rendered.
* **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
* **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

**Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
* **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
* **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

---

## Output Format

**Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

**Example Output:**
Consistency: 2
Realism: 1
Aesthetic Quality: 0

---

**IMPORTANT Enforcement:**

Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

---

Here are the Prompt and EXPLANATION for this evaluation:
PROMPT: "{prompt}"
EXPLANATION: "{explanation}"
Please strictly adhere to the scoring criteria and follow the template format when providing your results."""


SCORE_PATTERN = r"\*{0,2}(Consistency|Realism|Aesthetic Quality)\*{0,2}\s*[:：]?\s*(\d)"


def extract_scores(text: str) -> dict[str, float]:
    scores = {"consistency": 9.9, "realism": 9.9, "aesthetic_quality": 9.9}
    for key, value in re.findall(SCORE_PATTERN, text, re.IGNORECASE):
        key = key.lower().replace(" ", "_")
        if key in scores:
            scores[key] = float(value)
    return scores


def call_judge(session, endpoint: str, token: str, model: str, text: str, image_path: Path,
               timeout: float, retries: int) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,"
                            f"{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
                        },
                    },
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
        "n": 1,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    for attempt in range(1, retries + 1):
        try:
            response = session.post(f"{endpoint}/chat/completions", headers=headers,
                                    json=payload, timeout=timeout)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
            choices = response.json().get("choices") or []
            return choices[0]["message"]["content"].strip()
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                raise RuntimeError(f"judge call failed after {retries} attempts: {exc}") from exc
            time.sleep(2.0 * attempt)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="generation manifest jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-env", default="VENUS_API_KEY")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"missing API secret in {args.token_env}")

    import requests

    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()

    session = requests.Session()
    scored: list[dict[str, Any]] = []
    for row in rows:
        meta = row.get("meta", {}) or {}
        text = build_user_prompt(row["prompt"], meta.get("explanation") or meta.get("hint") or "")
        raw = call_judge(session, args.endpoint, token, args.model, text,
                         Path(row["output_path"]), args.timeout, args.retries)
        scores = extract_scores(raw)
        result = {
            "id": row["id"],
            "variant": row["variant"],
            "bench": "wise",
            "group": meta.get("group"),
            "category": meta.get("category"),
            "subcategory": meta.get("subcategory"),
            "seed": row.get("seed"),
            **scores,
            "wiscore": round(
                0.7 * scores["consistency"]
                + 0.2 * scores["realism"]
                + 0.1 * scores["aesthetic_quality"],
                4,
            ),
            "raw": raw[:400],
        }
        scored.append(result)
        with args.out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"event": "scored", "id": result["id"], **scores}), flush=True)

    def mean(key: str) -> float:
        return round(sum(r[key] for r in scored) / max(1, len(scored)), 4)

    failed = sum(1 for r in scored if 9.9 in (r["consistency"], r["realism"], r["aesthetic_quality"]))
    summary = {
        "bench": "wise",
        "evaluator": "wise_official_rubric",
        "model": args.model,
        "variant": scored[0]["variant"] if scored else None,
        "num_items": len(scored),
        "num_failed": failed,
        "consistency_mean": mean("consistency"),
        "realism_mean": mean("realism"),
        "aesthetic_quality_mean": mean("aesthetic_quality"),
        "wiscore_mean": mean("wiscore"),
    }
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "done", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
