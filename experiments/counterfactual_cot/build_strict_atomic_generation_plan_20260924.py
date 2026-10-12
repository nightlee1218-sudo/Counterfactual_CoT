#!/usr/bin/env python3
"""Create a generator-compatible plan from the validated round-two COT package."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

BAGEL_ROOT = Path("/group/40016/niightli/BAGEL")
ROOT = BAGEL_ROOT / "experiments/counterfactual_cot/gold1000_framework_strict_v4"
PACKAGE = Path(os.environ.get(
    "STRICT_ATOMIC_COT_PACKAGE",
    str(ROOT / "repair_round2_strict_atomic_20260924_v3"),
))
MANIFEST = PACKAGE / "revised_cot_manifest.jsonl"
REGISTRY = PACKAGE / "revised_source_registry.jsonl"
PLAN = PACKAGE / "strict_atomic_generation_plan.jsonl"
SUMMARY = PACKAGE / "strict_atomic_generation_plan_summary.json"

if str(BAGEL_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.build_second_round_repair_plan_20260924 import SPECS


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    manifest = load_jsonl(MANIFEST)
    registry = load_jsonl(REGISTRY)
    if len(manifest) != 49:
        raise RuntimeError(f"expected 49 revised records, got {len(manifest)}")
    if len(registry) != 30:
        raise RuntimeError(f"expected 30 revised source variants, got {len(registry)}")

    by_scene: dict[str, list[str]] = defaultdict(list)
    for row in manifest:
        by_scene[str(row["source_scene_id"])].append(str(row["pair_id"]))
    registry_ids = {str(row["scene_id"]) for row in registry}
    source_scene_ids = {
        str(row["source_scene_id"])
        for row in manifest
        if row["repair_round2"]["source_regenerated"]
    }
    if source_scene_ids != registry_ids:
        raise RuntimeError(
            "revised source scene mismatch: "
            f"manifest-only={sorted(source_scene_ids - registry_ids)}, "
            f"registry-only={sorted(registry_ids - source_scene_ids)}"
        )

    plans: list[dict[str, Any]] = []
    for row in sorted(manifest, key=lambda item: item["pair_id"]):
        pair_id = str(row["pair_id"])
        repair_round2 = row["repair_round2"]
        source_regenerated = bool(repair_round2["source_regenerated"])
        repair_scope = "source+target" if source_regenerated else "target-only"
        repair_role = "reported_problem" if pair_id in SPECS else "source_scene_spillover"
        spec = SPECS.get(pair_id, {})
        scene_id = str(row["source_scene_id"])
        members = sorted(by_scene[scene_id])
        direct_members = sorted(member for member in members if member in SPECS)
        source_measure = (
            "Use the revised source registry prompt and revised fixed scene plan exactly. "
            + str(spec.get("source_extra", ""))
            if source_regenerated and pair_id in SPECS
            else ""
        )
        target_measure = (
            str(spec.get("target_extra", ""))
            if pair_id in SPECS
            else ""
        )
        plans.append(
            {
                "schema_version": "counterfactual_repair_batch_strict_atomic_v1",
                "audit_round": 2,
                "pair_id": pair_id,
                "source_scene_id": scene_id,
                "source_variant_key": scene_id,
                "source_variant_member_pair_ids": members,
                "source_variant_direct_problem_pair_ids": direct_members,
                "repair_scope": repair_scope,
                "repair_role": repair_role,
                "user_issue": (
                    str(spec.get("issue", ""))
                    if pair_id in SPECS
                    else "未列出，默认合理；因共享 revised source variant 必须同步生成 target。"
                ),
                "approval_status": "strict_atomic_cot_validated",
                "generation_gate": "ready",
                "source_measure": source_measure,
                "target_measure": target_measure,
                "revised_manifest": str(MANIFEST),
                "revised_source_registry": str(REGISTRY),
                "source_cot_sha256": repair_round2["source_cot_sha256"],
                "target_cot_sha256": repair_round2["target_cot_sha256"],
                "atomic_change": repair_round2["atomic_change"],
                "generation_status": "pending_generation",
                "replacement_status": "blocked_until_round2_audit",
                "fusion_status": "not_ready",
            }
        )

    if len(plans) != 49 or len({row["pair_id"] for row in plans}) != 49:
        raise RuntimeError("generation plan pair IDs are not unique")
    if any(row["atomic_change"]["changed_atom_count"] != 1 for row in plans):
        raise RuntimeError("generation plan contains a non-atomic COT")
    write_jsonl(PLAN, plans)
    summary = {
        "schema_version": "counterfactual_repair_batch_strict_atomic_summary_v1",
        "audit_round": 2,
        "manifest": str(MANIFEST),
        "source_registry": str(REGISTRY),
        "generation_plan": str(PLAN),
        "record_count": len(plans),
        "source_variant_count": len(source_scene_ids),
        "source_plus_target_count": sum(row["repair_scope"] == "source+target" for row in plans),
        "target_only_count": sum(row["repair_scope"] == "target-only" for row in plans),
        "reported_problem_count": sum(row["repair_role"] == "reported_problem" for row in plans),
        "spillover_count": sum(row["repair_role"] == "source_scene_spillover" for row in plans),
        "all_cot_atomic": all(row["atomic_change"]["changed_atom_count"] == 1 for row in plans),
        "generation_gate": "ready",
        "replacement_status": "blocked_until_round2_audit",
        "fusion_status": "not_ready",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
