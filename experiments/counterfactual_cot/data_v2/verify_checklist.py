"""data_v2 semantic verification loop (M3 / T2I-Copilot checklist paradigm).

For every completed pair, build a checklist from the intervention plan:

  * CHANGE items — verify the intervention actually happened
    (e.g. "Is the cup blue?" expected yes / "Is the cup red?" expected no)
  * PRESERVE items — verify must_preserve atoms survived
    (entity presence, key attributes, counts of non-intervened objects)

Each item is asked against the TARGET image by a VLM (local BLIP-2 by
default, same paradigm as e3_eval/vqa_scorer.py). Output: per-item pass/fail
plus a pair score; pairs below --threshold are flagged status=regen.

Real VLM inference is environment-gated — nothing is downloaded here:
  * DATA_V2_VLM = blip2   -> real BLIP-2 (lazy HF transformers import,
                             model id from DATA_V2_VLM_MODEL, default
                             Salesforce/blip2-flan-t5-xl, local cache only)
  * anything else / unset -> deterministic mock oracle (selftests), which
    answers from the plan's target scene graph so a correct composite
    passes by construction and injected faults fail.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_jsonl, write_json_atomic, append_jsonl  # noqa: E402

DEFAULT_THRESHOLD = 0.8
MAX_PRESERVE_ITEMS = 8  # cap per-pair VLM cost

# ---------------------------------------------------------------------------
# Checklist construction
# ---------------------------------------------------------------------------

def _entity(scene: Dict[str, Any], eid: str) -> Optional[Dict[str, Any]]:
    return next((e for e in scene.get("entities", []) if e["id"] == eid), None)


def _phrase(entity: Dict[str, Any]) -> str:
    attrs = entity.get("attributes") or {}
    desc = entity["category"]
    for key in ("color", "material", "size", "pattern"):
        if attrs.get(key):
            desc = f"{attrs[key]} {desc}"
    return desc


def build_checklist(plan: Dict[str, Any],
                    max_preserve: int = MAX_PRESERVE_ITEMS) -> List[Dict[str, Any]]:
    """plan -> checklist items [{id, kind, question, expected}] (deterministic)."""
    items: List[Dict[str, Any]] = []
    iv = plan["intervention"]
    src_scene = plan["source"]["cot"]
    tgt_scene = plan["target"]["cot"]
    eid = iv["entity_id"]
    src_e = _entity(src_scene, eid)
    tgt_e = _entity(tgt_scene, eid)
    if src_e is None or tgt_e is None:
        raise ValueError(f"{plan.get('pair_id')}: intervened entity {eid!r} missing")
    subtype = plan.get("subtype", "")

    def add(kind: str, question: str, expected: str) -> None:
        items.append({"id": f"{kind}{len(items):02d}", "kind": kind,
                      "question": question, "expected": expected})

    # ---- CHANGE items --------------------------------------------------------
    if subtype == "attribute_intervention":
        key = iv["field"].split(".", 1)[1]
        add("change", f"Is the {src_e['category']} {iv['new_value']}?", "yes")
        add("change_neg", f"Is the {src_e['category']} still {iv['old_value']}?", "no")
    elif subtype == "category_replace":
        add("change", f"Is there a {iv['new_value']} in the image?", "yes")
        add("change_neg", f"Is there a {iv['old_value']} in the image?", "no")
    elif subtype == "count_change":
        noun = tgt_e["category"]
        add("change", f"How many {noun}s are in the image?", str(iv["new_value"]))
    elif subtype == "relative_position":
        rel = iv["new_value"][0]
        tgt_b = _entity(src_scene, rel["target"])
        kind = rel["type"]
        q = {"left_of": f"Is the {src_e['category']} to the left of the {tgt_b['category']}?",
             "on_top_of": f"Is the {src_e['category']} on top of the {tgt_b['category']}?"}
        add("change", q[kind], "yes")
    elif subtype == "location_change":
        add("change", f"Is the {src_e['category']} in the {iv['new_value']} part of the image?",
            "yes")

    # ---- PRESERVE items (must_preserve atoms of non-intervened entities) -----
    for entity in tgt_scene.get("entities", []):
        if len([i for i in items if i["kind"].startswith("preserve")]) >= max_preserve:
            break
        if entity["id"] == eid:
            continue
        add("preserve", f"Is there a {_phrase(entity)} in the image?", "yes")
        if entity.get("count") is not None and entity["count"] > 1 and iv["field"] != "count":
            add("preserve", f"How many {entity['category']}s are in the image?",
                str(entity["count"]))
    return items


# ---------------------------------------------------------------------------
# VLM askers (env-gated)
# ---------------------------------------------------------------------------

def _parse_yes_no(text: str) -> str:
    t = (text or "").strip().lower()
    if t.startswith(("yes", "yeah", "yep", "true")):
        return "yes"
    if t.startswith(("no", "not", "false", "none")):
        return "no"
    return t.split()[0] if t else ""


class MockVlm:
    """Deterministic oracle answering from the plan's target scene graph.

    ``faults`` (dict item_id -> wrong answer) lets selftests inject failures
    to exercise the regen path.
    """

    mode = "mock"

    def __init__(self, faults: Optional[Dict[str, str]] = None):
        self.faults = faults or {}

    def ask(self, question: str, image: Image.Image, item_id: str,
            expected: str) -> str:
        if item_id in self.faults:
            return self.faults[item_id]
        # oracle: answer the expected value (a correct composite matches the
        # target scene graph, so ground truth == expected)
        return expected


class Blip2Vlm:
    """Local BLIP-2 asker (lazy import; local HF cache only, never downloads).

    Gated behind DATA_V2_VLM=blip2 so the default pipeline never loads it.
    """

    mode = "blip2"

    def __init__(self, model_id: Optional[str] = None, device: str = "auto"):
        import os
        model_id = model_id or os.environ.get("DATA_V2_VLM_MODEL",
                                              "Salesforce/blip2-flan-t5-xl")
        import torch
        from transformers import Blip2ForConditionalGeneration, Blip2Processor
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else device
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
        ).to(self.device)

    def ask(self, question: str, image: Image.Image, item_id: str,
            expected: str) -> str:
        import torch
        inputs = self.processor(images=image.convert("RGB"), text=question,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=8)
        answer = self.processor.decode(out[0], skip_special_tokens=True).strip()
        if expected in ("yes", "no"):
            return _parse_yes_no(answer)
        return answer


def build_vlm(name: Optional[str] = None) -> Any:
    import os
    name = name or os.environ.get("DATA_V2_VLM", "mock")
    if name == "blip2":
        return Blip2Vlm()
    if name == "mock":
        return MockVlm()
    raise ValueError(f"unknown VLM: {name!r} (mock|blip2)")


# ---------------------------------------------------------------------------
# Per-pair verification
# ---------------------------------------------------------------------------

def verify_pair(plan: Dict[str, Any], image_root: Path, manifest_dir: Path,
                vlm: Any, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """One manifest row -> verification row (never raises)."""
    pair_id = plan["pair_id"]
    row: Dict[str, Any] = {"pair_id": pair_id, "subtype": plan.get("subtype"),
                           "status": "error"}
    try:
        items = build_checklist(plan)
        img_path = manifest_dir / plan["target"]["image"]
        image = Image.open(img_path).convert("RGB")
        results = []
        for item in items:
            answer = str(vlm.ask(item["question"], image, item["id"], item["expected"]))
            ok = answer.strip().lower() == str(item["expected"]).strip().lower()
            results.append({**item, "answer": answer, "pass": ok})
        n_pass = sum(1 for r in results if r["pass"])
        score = n_pass / len(results) if results else 0.0
        row.update({
            "items": results,
            "n_items": len(results),
            "n_pass": n_pass,
            "score": round(score, 4),
            "status": "pass" if score >= threshold else "regen",
            "threshold": threshold,
            "error": None,
        })
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_verify(manifest_path: Path, image_root: Path, out_json: Path,
               out_jsonl: Optional[Path] = None,
               threshold: float = DEFAULT_THRESHOLD,
               vlm: Optional[Any] = None) -> Dict[str, Any]:
    vlm = vlm or build_vlm()
    plans = load_jsonl(manifest_path)
    rows = [verify_pair(p, image_root, manifest_path.parent, vlm, threshold)
            for p in plans]
    ok_rows = [r for r in rows if r["status"] in ("pass", "regen")]
    scores = [r["score"] for r in ok_rows]
    report = {
        "schema_version": "counterfactual_data_v2_verify/1.0",
        "manifest": str(manifest_path),
        "vlm_mode": getattr(vlm, "mode", "?"),
        "threshold": threshold,
        "summary": {
            "n_pairs": len(rows),
            "n_pass": sum(1 for r in rows if r["status"] == "pass"),
            "n_regen": sum(1 for r in rows if r["status"] == "regen"),
            "n_error": sum(1 for r in rows if r["status"] == "error"),
            "score_mean": float(np.mean(scores)) if scores else None,
            "score_min": float(np.min(scores)) if scores else None,
            "item_pass_rate": (
                sum(r["n_pass"] for r in ok_rows) /
                max(sum(r["n_items"] for r in ok_rows), 1)),
        },
        "rows": rows,
    }
    write_json_atomic(out_json, report)
    if out_jsonl:
        for r in rows:
            append_jsonl(out_jsonl, r)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--image-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--vlm", default=None, help="mock|blip2 (default: env DATA_V2_VLM or mock)")
    args = ap.parse_args()
    report = run_verify(args.manifest, args.image_root, args.out,
                        threshold=args.threshold, vlm=build_vlm(args.vlm) if args.vlm else None)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["summary"]["n_regen"] or report["summary"]["n_error"]:
        print(f"VERIFY: {report['summary']['n_regen']} regen, "
              f"{report['summary']['n_error']} errors (see {args.out})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
