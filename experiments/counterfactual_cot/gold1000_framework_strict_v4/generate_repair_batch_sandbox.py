#!/usr/bin/env python3
"""Generate all approved repair candidates into an isolated sandbox.

The final 1000-pair dataset is read-only. This command does not delete or
overwrite any original image or manifest. It expands source+target repairs to
all pairs sharing the regenerated source_scene_id, so those target images are
never left paired with an incompatible source.

COT/intervention/preservation fields are copied unchanged. Only the API image
prompts are augmented with the approved repair measures from repair_plan.jsonl.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Make direct execution robust even when the caller did not export PYTHONPATH.
# This resolves to /group/40016/niightli/BAGEL.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from PIL import Image

from experiments.counterfactual_cot.data.generate_source_images import source_conditioning_text
from experiments.counterfactual_cot.data.image_providers import build_provider
from experiments.counterfactual_cot.data.mask_utils import create_candidate_mask
from experiments.counterfactual_cot.data.schema import load_jsonl, write_jsonl


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".repair_tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def ensure_square(path: Path) -> None:
    with Image.open(path) as image:
        if image.size != (1024, 1024):
            raise RuntimeError(f"{path}: expected 1024x1024, got {image.size}")


def load_plan(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            result[str(row["pair_id"])] = row
    return result


def source_prompt(base: str, measures: list[str]) -> str:
    unique = list(dict.fromkeys(x.strip() for x in measures if x and x.strip()))
    return (
        base.rstrip()
        + "\n\nAPPROVED SOURCE REPAIR CONSTRAINTS (preserve the original COT semantics):\n"
        + "\n".join(f"- {x}" for x in unique)
        + "\n- Keep every source COT entity, count, category, attribute and relation semantically correct."
        + "\n- Use a native 1024x1024 square full-bleed scene with no frame, border, crop artifact or inset."
        + "\n- Do not add any object not named by the source COT and do not delete any named object."
    )


def target_prompt(base: str, measure: str, spillover: bool) -> str:
    if spillover:
        extra = (
            "\n\nSOURCE-SCENE SYNC CONSTRAINT: the source scene was regenerated. Perform the original target intervention "
            "exactly as written above, using this source, and preserve all unrelated entities and the single-atom "
            "intervention semantics. Do not introduce any additional edit."
        )
    else:
        extra = (
            "\n\nAPPROVED TARGET REPAIR CONSTRAINTS: the following constraints repair the reported failure while preserving "
            "the original target COT and intervention.\n- " + (measure or "Make only the original intervention correct and visibly verifiable.")
        )
    return (
        base.rstrip() + extra
        + "\n- Only intervention.entity_id and intervention.field may change."
        + "\n- Do not add, delete, duplicate, move, resize, rotate or alter unrelated objects."
        + "\n- Preserve distinctive survivor details and all must_preserve entities."
        + "\n- Keep a native 1024x1024 square full-bleed scene with no frame, border, crop artifact, inset, text, logo, watermark or UI."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path, help="Final 1000-pair manifest, read-only")
    ap.add_argument("--source-registry", required=True, type=Path, help="Final source registry, read-only")
    ap.add_argument("--repair-plan", required=True, type=Path, help="Approved repair_plan_20260923.jsonl")
    ap.add_argument("--image-root", required=True, type=Path, help="Final source_images, read-only")
    ap.add_argument("--sandbox", required=True, type=Path)
    ap.add_argument("--model", default="gemini-3-pro-image")
    ap.add_argument("--api-key-env", default="CFCOT_IMAGE_API_KEY")
    ap.add_argument("--endpoint", default="http://v2.open.venus.oa.com/llmproxy/chat/completions")
    ap.add_argument("--timeout-s", type=int, default=180)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--max-output-tokens", type=int, default=16384)
    ap.add_argument("--rate-limit-qps", type=float, default=0.2)
    ap.add_argument("--allow-existing-sandbox", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sandbox_resolved = args.sandbox.resolve()
    image_root_resolved = args.image_root.resolve()
    if sandbox_resolved == image_root_resolved or image_root_resolved in sandbox_resolved.parents:
        raise RuntimeError("sandbox must be outside the original image root")
    if not args.dry_run and sandbox_resolved.exists() and any(sandbox_resolved.iterdir()) and not args.allow_existing_sandbox:
        raise RuntimeError(f"sandbox is non-empty; use a new path or --allow-existing-sandbox: {sandbox_resolved}")

    final_records = {r["pair_id"]: r for r in load_jsonl(args.manifest)}
    plans = load_plan(args.repair_plan)
    unknown = sorted(set(plans) - set(final_records))
    if unknown:
        raise RuntimeError(f"repair plan has unknown pair IDs: {unknown[:10]}")

    scene_members: dict[str, list[str]] = defaultdict(list)
    for pair_id, record in final_records.items():
        scene_members[str(record["source_scene_id"])].append(pair_id)
    source_scene_ids = {
        str(final_records[pid]["source_scene_id"])
        for pid, plan in plans.items()
        if plan.get("repair_scope") == "source+target"
    }
    selected: dict[str, dict[str, Any]] = {}
    for pid, plan in plans.items():
        selected[pid] = plan
    for scene_id in source_scene_ids:
        for pid in scene_members[scene_id]:
            if pid not in selected:
                selected[pid] = {
                    "pair_id": pid,
                    "source_scene_id": scene_id,
                    "repair_scope": "source+target",
                    "repair_role": "source_scene_spillover",
                    "user_issue": "同一 source_scene_id 的连带 target；source 改变后必须重新生成",
                    "target_measure": "",
                    "source_measure": "",
                    "approval_status": "用户已批准 source 修补的连带 target",
                }
            else:
                selected[pid]["repair_role"] = selected[pid].get("repair_role", "reported_problem")
    for pid in selected:
        selected[pid].setdefault("repair_role", "reported_problem")

    source_measures: dict[str, list[str]] = defaultdict(list)
    for pid, plan in plans.items():
        if plan.get("repair_scope") == "source+target":
            source_measures[str(final_records[pid]["source_scene_id"])].append(str(plan.get("source_measure", "")))
    plan_summary = {
        "schema_version": "counterfactual_repair_batch_v1",
        "original_manifest": str(args.manifest.resolve()),
        "original_image_root": str(args.image_root.resolve()),
        "model": args.model,
        "source_scene_count": len(source_scene_ids),
        "reported_problem_count": sum(1 for plan in plans.values() if plan.get("repair_role") == "reported_problem"),
        "selected_target_count": len(selected),
        "source_scene_ids": sorted(source_scene_ids),
        "spillover_pair_count": sum(1 for p in selected.values() if p.get("repair_role") == "source_scene_spillover"),
        "pair_ids": sorted(selected),
    }
    args.sandbox.mkdir(parents=True, exist_ok=True)
    (args.sandbox / "repair_batch_plan.json").write_text(json.dumps(plan_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan_summary, ensure_ascii=False, indent=2))
        return

    registry = {r["scene_id"]: r for r in load_jsonl(args.source_registry)}
    provider = build_provider(
        "nano_banana_pro", model=args.model, api_key_env=args.api_key_env,
        endpoint=args.endpoint, timeout_s=args.timeout_s, max_retries=args.max_retries,
        retry_backoff_s=3.0, rate_limit_qps=args.rate_limit_qps,
        max_output_tokens=args.max_output_tokens,
    )
    ledger = args.sandbox / "repair_ledger.jsonl"
    report = {"plan": plan_summary, "items": []}

    def record_event(event: dict[str, Any]) -> None:
        report["items"].append(event)
        with ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    sandbox_scene_paths: dict[str, Path] = {}
    for scene_id in sorted(source_scene_ids):
        if scene_id not in registry:
            raise RuntimeError(f"source registry missing scene_id: {scene_id}")
        out = args.sandbox / "source" / "_by_scene" / f"cf_source_{scene_id}__source.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        prompt = source_prompt(source_conditioning_text(registry[scene_id]), source_measures[scene_id])
        started = time.monotonic()
        try:
            provider.generate_source(prompt, tmp)
            ensure_square(tmp)
            os.replace(tmp, out)
            sandbox_scene_paths[scene_id] = out
            record_event({"stage": "source", "scene_id": scene_id, "status": "success", "model": args.model,
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                          "latency_s": round(time.monotonic() - started, 1)})
        except Exception as exc:  # fail closed; do not generate targets from an unapproved source
            if tmp.exists():
                tmp.unlink()
            record_event({"stage": "source", "scene_id": scene_id, "status": "failed", "error": str(exc)})

    output_records = []
    for pid in sorted(selected):
        original = final_records[pid]
        plan = selected[pid]
        scene_id = str(original["source_scene_id"])
        source_rel = Path("source") / f"{pid}__source.png"
        source_path = args.sandbox / source_rel
        source_was_regenerated = scene_id in source_scene_ids
        if source_was_regenerated and scene_id not in sandbox_scene_paths:
            record_event({"stage": "target", "pair_id": pid, "status": "skipped_source_failed"})
            continue
        if source_was_regenerated:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.exists():
                source_path.unlink()
            source_path.hardlink_to(sandbox_scene_paths[scene_id])
        else:
            original_source = args.image_root / original["source"]["image"]
            copy_atomic(original_source, source_path)
        target_rel = Path("target") / f"{pid}__target.png"
        mask_rel = Path("masks") / f"{pid}__mask.png"
        target_path = args.sandbox / target_rel
        mask_path = args.sandbox / mask_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        prompt = target_prompt(str(original["target"]["prompt"]), str(plan.get("target_measure", "")), plan.get("repair_role") == "source_scene_spillover")
        tmp_target = target_path.with_name(target_path.name + ".tmp")
        tmp_mask = mask_path.with_name(mask_path.name + ".tmp")
        started = time.monotonic()
        try:
            provider.edit_target(source_path, prompt, tmp_target)
            ensure_square(tmp_target)
            create_candidate_mask(source_path, tmp_target, tmp_mask, threshold=24)
            os.replace(tmp_target, target_path)
            os.replace(tmp_mask, mask_path)
            status = "success"
            error = None
        except Exception as exc:  # keep no invalid partial asset
            for path in (tmp_target, tmp_mask):
                if path.exists():
                    path.unlink()
            status = "failed"
            error = str(exc)
        record_event({"stage": "target", "pair_id": pid, "status": status, "model": args.model,
                      "source_regenerated": source_was_regenerated, "repair_role": plan.get("repair_role"),
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                      "latency_s": round(time.monotonic() - started, 1), "error": error})
        if status != "success":
            continue
        output = copy.deepcopy(original)
        output["source"]["image"] = str(source_rel)
        output["target"]["image"] = str(target_rel)
        output["target"]["mask"] = str(mask_rel)
        output["target"]["prompt"] = prompt
        output["repair"] = {
            "repair_scope": "source+target" if source_was_regenerated else "target-only",
            "repair_role": plan.get("repair_role", "reported_problem"),
            "source_regenerated": source_was_regenerated,
            "target_regenerated": True,
            "user_issue": plan.get("user_issue", ""),
            "source_measure": plan.get("source_measure", ""),
            "target_measure": plan.get("target_measure", ""),
            "original_target_prompt": original["target"]["prompt"],
            "source_prompt_used": "see repair_ledger source event" if source_was_regenerated else "original source image",
            "target_prompt_used": prompt,
        }
        output["generation"] = {**output.get("generation", {}), "repair_sandbox": True, "repair_model": args.model}
        output_records.append(output)

    write_jsonl(args.sandbox / "manifest.jsonl", output_records)
    report["summary"] = {
        "selected_targets": len(selected),
        "generated_records": len(output_records),
        "source_scenes": len(source_scene_ids),
        "source_success": sum(1 for e in report["items"] if e.get("stage") == "source" and e.get("status") == "success"),
        "target_success": sum(1 for e in report["items"] if e.get("stage") == "target" and e.get("status") == "success"),
        "failures": sum(1 for e in report["items"] if e.get("status") in {"failed", "skipped_source_failed"}),
    }
    (args.sandbox / "repair_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
