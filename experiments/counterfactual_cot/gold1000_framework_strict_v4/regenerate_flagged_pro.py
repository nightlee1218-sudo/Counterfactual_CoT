#!/usr/bin/env python3
"""Regenerate human-flagged pairs with nano-banana-pro into an isolated sandbox.

SAFETY CONTRACT
---------------
The original 1000-pair dataset is opened READ-ONLY.  Every regenerated asset
goes into a separate sandbox directory, so nothing in the source dataset can be
deleted, overwritten, or mixed with unvalidated output.

Sandbox layout::

    <sandbox>/
      source/_by_scene/<scene>.png      regenerated source scenes (or symlink to original)
      source/<pair_id>__source.png      per-pair view (hard link to the scene file)
      target/<pair_id>__target.png      regenerated target
      masks/<pair_id>__mask.png         recomputed automatic difference mask
      manifest.jsonl                    full 1000-record manifest pointing into the sandbox
      regen_report.json                 per-item outcome + provenance

Stage A regenerates only the shared source scenes explicitly requested, and
rebuilds the per-pair view for every pair that shares that scene (pair-level
source paths are hard links to the scene file, so a regenerated scene must be
re-linked for all its pairs to stay consistent).
Stage B regenerates the target of every flagged pair plus any pair dragged in
by a regenerated scene, writing to the sandbox and recomputing the mask.

Every failure keeps the ORIGINAL asset: generation always happens in a
``.tmp`` file and only replaces the sandbox copy on success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

from experiments.counterfactual_cot.data.image_providers import build_provider
from experiments.counterfactual_cot.data.mask_utils import create_candidate_mask
from experiments.counterfactual_cot.data.generate_pair_images import target_edit_instruction
from experiments.counterfactual_cot.data.generate_source_images import source_conditioning_text
from experiments.counterfactual_cot.data.schema import canonical_json, load_jsonl, write_jsonl

DEFAULT_MODEL = "gemini-3-pro-image"
PROVIDER = "nano_banana_pro"


class RegenerationError(RuntimeError):
    """The regeneration plan cannot be executed safely."""


def _scene_rel(scene_id: str) -> Path:
    return Path("source") / "_by_scene" / f"cf_source_{scene_id}__source.png"


def _pair_source_rel(pair_id: str) -> Path:
    return Path("source") / f"{pair_id}__source.png"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _plan(records: List[Dict[str, Any]], flagged: List[str], regen_scenes: List[str]) -> Dict[str, Any]:
    by_id = {r["pair_id"]: r for r in records}
    unknown = [p for p in flagged if p not in by_id]
    if unknown:
        raise RegenerationError(f"unknown pair ids: {unknown}")

    scene_to_pairs: Dict[str, List[str]] = {}
    for record in records:
        scene_to_pairs.setdefault(record["source_scene_id"], []).append(record["pair_id"])

    target_pairs = set(flagged)
    spillover: Dict[str, List[str]] = {}
    for scene_id in regen_scenes:
        members = sorted(scene_to_pairs.get(scene_id, []))
        extra = [p for p in members if p not in target_pairs]
        if extra:
            spillover[scene_id] = extra
        target_pairs.update(members)

    return {
        "flagged": sorted(flagged),
        "regen_source_scenes": sorted(regen_scenes),
        "regenerate_target": sorted(target_pairs),
        "spillover": spillover,
        "target_count": len(target_pairs),
        "source_count": len(regen_scenes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Original manifest; opened READ-ONLY")
    parser.add_argument("--source-registry", required=True, type=Path, help="Source registry; READ-ONLY")
    parser.add_argument("--source-image-root", required=True, type=Path, help="Original image root; READ-ONLY")
    parser.add_argument("--sandbox", required=True, type=Path, help="Isolated output directory")
    parser.add_argument("--pair-list", required=True, type=Path)
    parser.add_argument("--regen-source-scenes", default="", help="Comma-separated source_scene_ids to regenerate")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-model", default=None, help="Disabled unless given explicitly")
    parser.add_argument("--api-key-env", default="CFCOT_IMAGE_API_KEY")
    parser.add_argument("--endpoint", default="http://v2.open.venus.oa.com/llmproxy/chat/completions")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--rate-limit-qps", type=float, default=0.2)
    parser.add_argument("--mask-threshold", type=int, default=24)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=16384,
                        help="Output budget; reasoning image models need far more than 1000 to emit the image.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sandbox: Path = args.sandbox
    if args.dry_run:
        records = load_jsonl(args.manifest)
        flagged = [l.strip() for l in args.pair_list.read_text(encoding="utf-8").splitlines() if l.strip()]
        scenes = [s.strip() for s in args.regen_source_scenes.split(",") if s.strip()]
        plan = _plan(records, flagged, scenes)
        print(json.dumps({
            "dry_run": True, "sandbox": str(sandbox), "model": args.model, "fallback": args.fallback_model,
            "source_read_only": str(args.source_image_root),
            "regenerate_source_scenes": plan["regen_source_scenes"],
            "regenerate_target_count": plan["target_count"],
            "spillover": plan["spillover"],
            "target_pairs": plan["regenerate_target"],
        }, ensure_ascii=False, indent=2))
        return

    # Refuse to write anywhere inside the pristine dataset.
    if sandbox.resolve() == args.source_image_root.resolve() or args.source_image_root.resolve() in sandbox.resolve().parents:
        raise RegenerationError("sandbox must not live inside the original image root")
    sandbox.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.manifest)
    registry = {r["scene_id"]: r for r in load_jsonl(args.source_registry)}
    flagged = [l.strip() for l in args.pair_list.read_text(encoding="utf-8").splitlines() if l.strip()]
    scenes = [s.strip() for s in args.regen_source_scenes.split(",") if s.strip()]
    plan = _plan(records, flagged, scenes)

    provider_kwargs = dict(
        api_key_env=args.api_key_env, endpoint=args.endpoint, timeout_s=args.timeout_s,
        max_retries=args.max_retries, retry_backoff_s=3.0, rate_limit_qps=args.rate_limit_qps,
        max_output_tokens=args.max_output_tokens,
    )
    pro = build_provider(PROVIDER, model=args.model, **provider_kwargs)
    fallback = build_provider(PROVIDER, model=args.fallback_model, **provider_kwargs) if args.fallback_model else None

    ledger_path = sandbox / "regen_ledger.jsonl"
    report: Dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sandbox": str(sandbox),
        "source_image_root_read_only": str(args.source_image_root),
        "model": args.model,
        "fallback": args.fallback_model,
        "plan": {k: plan[k] for k in ("regen_source_scenes", "target_count", "source_count", "spillover")},
        "items": [],
    }

    def note(entry: Dict[str, Any]) -> None:
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        report["items"].append(entry)

    scene_to_pairs: Dict[str, List[str]] = {}
    for record in records:
        scene_to_pairs.setdefault(record["source_scene_id"], []).append(record["pair_id"])

    # ---------------- Stage A: regenerate requested source scenes ----------
    for scene_id in plan["regen_source_scenes"]:
        if scene_id not in registry:
            raise RegenerationError(f"scene not in source registry: {scene_id}")
        item = registry[scene_id]
        sandbox_scene = sandbox / _scene_rel(scene_id)
        if not sandbox_scene.exists():
            _link_or_copy(args.source_image_root / _scene_rel(scene_id), sandbox_scene)
        tmp = sandbox_scene.with_name(sandbox_scene.name + ".tmp")
        started = time.monotonic()
        prompt = source_conditioning_text(item)
        used = None
        try:
            pro.generate_source(prompt, tmp)
            used = args.model
        except Exception as exc:  # noqa: BLE001
            note({"scene_id": scene_id, "stage": "source", "status": "pro_failed", "error": str(exc)})
            if fallback is not None:
                try:
                    fallback.generate_source(prompt, tmp)
                    used = args.fallback_model
                except Exception as exc2:  # noqa: BLE001
                    note({"scene_id": scene_id, "stage": "source", "status": "failed", "error": str(exc2)})
        if used is None:
            if tmp.exists():
                tmp.unlink()
            print(f"source FAILED (sandbox keeps original): {scene_id}")
            continue
        os.replace(tmp, sandbox_scene)
        linked = 0
        for pair_id in sorted(scene_to_pairs.get(scene_id, [])):
            _link_or_copy(sandbox_scene, sandbox / _pair_source_rel(pair_id))
            linked += 1
        note({"scene_id": scene_id, "stage": "source", "status": "success", "model": used,
              "latency_s": round(time.monotonic() - started, 1), "pairs_linked": linked})
        print(f"source regenerated: {scene_id} ({linked} pairs linked)")

    # ---------------- Stage B: regenerate targets --------------------------
    by_id = {r["pair_id"]: r for r in records}
    for pair_id in plan["regenerate_target"]:
        record = by_id[pair_id]
        original_source = args.source_image_root / record["source"]["image"]
        if not original_source.is_file():
            original_source = args.source_image_root / _pair_source_rel(pair_id)
        # The pair's source view inside the sandbox is authoritive when the
        # scene was regenerated; otherwise mirror the original.
        sandbox_scene_path = sandbox / _scene_rel(record["source_scene_id"])
        pair_source = sandbox / _pair_source_rel(pair_id)
        if not pair_source.exists():
            _link_or_copy(sandbox_scene_path if sandbox_scene_path.exists() else original_source, pair_source)
        source_for_edit = pair_source

        target_rel = Path("target") / f"{pair_id}__target.png"
        mask_rel = Path("masks") / f"{pair_id}__mask.png"
        sandbox_target = sandbox / target_rel
        sandbox_mask = sandbox / mask_rel
        if not sandbox_target.exists():
            _link_or_copy(args.source_image_root / target_rel, sandbox_target)
        if not sandbox_mask.exists():
            _link_or_copy(args.source_image_root / mask_rel, sandbox_mask)

        instruction = target_edit_instruction(record)
        started = time.monotonic()
        tmp_target = sandbox_target.with_name(sandbox_target.name + ".tmp")
        tmp_mask = sandbox_mask.with_name(sandbox_mask.name + ".tmp")
        used = None
        try:
            pro.edit_target(source_for_edit, instruction, tmp_target)
            used = args.model
        except Exception as exc:  # noqa: BLE001
            note({"pair_id": pair_id, "stage": "target", "status": "pro_failed", "error": str(exc)})
            if fallback is not None:
                try:
                    fallback.edit_target(source_for_edit, instruction, tmp_target)
                    used = args.fallback_model
                except Exception as exc2:  # noqa: BLE001
                    note({"pair_id": pair_id, "stage": "target", "status": "failed", "error": str(exc2)})
        if used is None:
            for leftover in (tmp_target, tmp_mask):
                if leftover.exists():
                    leftover.unlink()
            print(f"target FAILED (sandbox keeps original): {pair_id}")
            continue
        create_candidate_mask(source_for_edit, tmp_target, tmp_mask, threshold=args.mask_threshold)
        os.replace(tmp_target, sandbox_target)
        os.replace(tmp_mask, sandbox_mask)
        note({"pair_id": pair_id, "stage": "target", "status": "success", "model": used,
              "latency_s": round(time.monotonic() - started, 1)})
        print(f"target regenerated: {pair_id} ({used})")

    # ---------------- Emit a standalone sandbox manifest -------------------
    out_records = []
    for record in records:
        item = json.loads(json.dumps(record))
        pid = item["pair_id"]
        item["source"]["image"] = str(_pair_source_rel(pid))
        item["target"]["image"] = str(Path("target") / f"{pid}__target.png")
        item["target"]["mask"] = str(Path("masks") / f"{pid}__mask.png")
        out_records.append(item)
    write_jsonl(sandbox / "manifest.jsonl", out_records)

    successes = sum(1 for i in report["items"] if i.get("status") == "success")
    failures = sum(1 for i in report["items"] if i.get("status") in ("failed", "pro_failed"))
    report["summary"] = {"planned_sources": plan["source_count"], "planned_targets": plan["target_count"],
                         "succeeded": successes, "failed_or_kept_original": failures}
    (sandbox / "regen_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
