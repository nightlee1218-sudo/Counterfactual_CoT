from pathlib import Path
import hashlib
import json

ROOT = Path("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/gold1000_framework_strict_v4")
DATA = ROOT / "full_1000_20260918"
V3_ROOT = ROOT / "repair_batch_20260923_v3"
CURRENT_ROOT = ROOT / "repair_batch_20240924_v1"
OUT_PATH = ROOT / "repair_replacement_tracking_20260924.jsonl"
SUMMARY_PATH = ROOT / "repair_replacement_tracking_20260924_summary.json"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def path_for(base: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    return str(path if path.is_absolute() else base / path)


def digest(path: str | Path | None) -> str | None:
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            checksum.update(block)
    return checksum.hexdigest()


def intervention_summary(record: dict) -> dict:
    intervention = record.get("intervention") or {}
    keys = (
        "entity_id",
        "field",
        "intervention_subtype",
        "intervention_type",
        "old_value",
        "new_value",
        "operation",
    )
    return {key: intervention.get(key) for key in keys}


def main() -> None:
    if OUT_PATH.exists() or SUMMARY_PATH.exists():
        raise RuntimeError("tracking output already exists; refusing to overwrite")

    original = load(DATA / "gold1000.generated.jsonl")
    original_by_id = {str(record["pair_id"]): record for record in original}
    original_line_by_id = {
        str(record["pair_id"]): index for index, record in enumerate(original, start=1)
    }
    v3 = load(V3_ROOT / "manifest.jsonl")
    v3_by_id = {str(record["pair_id"]): record for record in v3}
    detailed = load(ROOT / "detailed_repair_plan_20260924_v2.jsonl")
    detailed_by_id = {str(record["pair_id"]): record for record in detailed}
    current_plan = json.loads(
        (CURRENT_ROOT / "repair_batch_plan.json").read_text(encoding="utf-8")
    )
    current_ids = [str(pair_id) for pair_id in current_plan["pair_ids"]]
    current_ledger = load(CURRENT_ROOT / "repair_ledger.jsonl")
    source_success_scenes = {
        str(event["scene_id"])
        for event in current_ledger
        if event.get("stage") == "source" and event.get("status") == "success"
    }
    overlap = set(v3_by_id) & set(current_ids)
    original_manifest_path = DATA / "gold1000.generated.jsonl"
    original_manifest_hash = digest(original_manifest_path)
    v3_manifest_hash = digest(V3_ROOT / "manifest.jsonl")
    records = []

    for pair_id in sorted(v3_by_id):
        candidate = v3_by_id[pair_id]
        original_record = original_by_id[pair_id]
        repair = candidate.get("repair") or {}
        source_regenerated = bool(repair.get("source_regenerated"))
        source_path = path_for(V3_ROOT, (candidate.get("source") or {}).get("image"))
        target_path = path_for(V3_ROOT, (candidate.get("target") or {}).get("image"))
        mask_path = path_for(V3_ROOT, (candidate.get("target") or {}).get("mask"))
        scene_id = str(original_record["source_scene_id"])
        scene_path = V3_ROOT / "source" / "_by_scene" / (
            "cf_source_" + scene_id + "__source.png"
        )
        records.append(
            {
                "schema_version": "repair_replacement_tracking_v1",
                "tracking_created_date": "2026-09-24",
                "pair_id": pair_id,
                "original_manifest_line": original_line_by_id[pair_id],
                "source_scene_id": scene_id,
                "batch_id": "repair_batch_20260923_v3",
                "batch_path": str(V3_ROOT),
                "record_type": "no_problem_candidate",
                "repair_role": repair.get("repair_role", "reported_problem"),
                "repair_scope": repair.get(
                    "repair_scope", "source+target" if source_regenerated else "target-only"
                ),
                "subtype": original_record.get("subtype"),
                "intervention": intervention_summary(original_record),
                "user_issue": repair.get("user_issue", ""),
                "review_status": "no_problem",
                "review_basis": "user_instruction",
                "repair_status": "completed_candidate",
                "generation_status": "complete",
                "source_status": "regenerated" if source_regenerated else "original_preserved",
                "target_status": "generated",
                "mask_status": "generated",
                "source_regenerated": source_regenerated,
                "target_regenerated": True,
                "source_path": source_path,
                "target_path": target_path,
                "mask_path": mask_path,
                "source_scene_asset_path": str(scene_path) if scene_path.is_file() else None,
                "original_source_path": path_for(
                    DATA / "source_images", (original_record.get("source") or {}).get("image")
                ),
                "original_target_path": path_for(
                    DATA / "source_images", (original_record.get("target") or {}).get("image")
                ),
                "original_mask_path": path_for(
                    DATA / "source_images", (original_record.get("target") or {}).get("mask")
                ),
                "candidate_manifest_path": str(V3_ROOT / "manifest.jsonl"),
                "candidate_manifest_sha256": v3_manifest_hash,
                "source_sha256": digest(source_path),
                "target_sha256": digest(target_path),
                "mask_sha256": digest(mask_path),
                "automatic_quality_passed": bool(
                    ((candidate.get("quality") or {}).get("local_checks") or {}).get("passed")
                ),
                "automatic_quality_review_status": (candidate.get("quality") or {}).get(
                    "review_status"
                ),
                "replacement_status": (
                    "hold_current_batch_overlap" if pair_id in overlap else "pending"
                ),
                "fusion_status": "not_fused",
                "candidate_for_replacement": pair_id not in overlap,
                "superseded_by_batch": (
                    "repair_batch_20240924_v1" if pair_id in overlap else None
                ),
                "notes": (
                    "按用户指令记录为 v3 本轮没有问题；尚未替换进1000-pair。"
                    "与当前修补批次重叠的pair暂缓使用v3版本。"
                    if pair_id in overlap
                    else "按用户指令记录为 v3 本轮没有问题；作为后续替换候选，尚未融合。"
                ),
            }
        )

    for pair_id in sorted(current_ids):
        original_record = original_by_id[pair_id]
        detail = detailed_by_id.get(pair_id)
        scene_id = str(original_record["source_scene_id"])
        role = "reported_problem" if detail else "source_scene_spillover"
        scope = detail.get("repair_scope") if detail else "source+target"
        source_regenerated = scope == "source+target"
        scene_path = CURRENT_ROOT / "source" / "_by_scene" / (
            "cf_source_" + scene_id + "__source.png"
        )
        pair_source_path = CURRENT_ROOT / "source" / (pair_id + "__source.png")
        target_path = CURRENT_ROOT / "target" / (pair_id + "__target.png")
        mask_path = CURRENT_ROOT / "masks" / (pair_id + "__mask.png")
        source_status = (
            "generated_unreviewed"
            if source_regenerated and scene_id in source_success_scenes
            else "pending_generation"
            if source_regenerated
            else "original_preserved"
        )
        target_status = "generated_unreviewed" if target_path.is_file() else "pending_generation"
        mask_status = "generated_unreviewed" if mask_path.is_file() else "pending_generation"
        source_input_path = (
            str(scene_path)
            if source_regenerated
            else path_for(DATA / "source_images", (original_record.get("source") or {}).get("image"))
        )
        generation_status = (
            "complete_unreviewed"
            if target_path.is_file() and mask_path.is_file()
            else "in_progress"
        )
        records.append(
            {
                "schema_version": "repair_replacement_tracking_v1",
                "tracking_created_date": "2026-09-24",
                "pair_id": pair_id,
                "original_manifest_line": original_line_by_id[pair_id],
                "source_scene_id": scene_id,
                "batch_id": "repair_batch_20240924_v1",
                "batch_path": str(CURRENT_ROOT),
                "record_type": "problem_under_repair",
                "repair_role": role,
                "repair_scope": scope,
                "subtype": original_record.get("subtype") if not detail else detail.get("subtype"),
                "intervention": intervention_summary(original_record),
                "user_issue": (
                    detail.get("user_issue")
                    if detail
                    else "同一source_scene_id的连带target，source改变后同步修改。"
                ),
                "review_status": "problem_under_repair",
                "review_basis": "user_instruction",
                "repair_status": "in_progress",
                "generation_status": generation_status,
                "source_status": source_status,
                "target_status": target_status,
                "mask_status": mask_status,
                "source_regenerated": source_regenerated,
                "target_regenerated": target_path.is_file(),
                "source_path": str(pair_source_path),
                "target_path": str(target_path),
                "mask_path": str(mask_path),
                "source_scene_asset_path": str(scene_path),
                "source_scene_asset_exists": scene_path.is_file(),
                "source_input_path": source_input_path,
                "original_source_path": path_for(
                    DATA / "source_images", (original_record.get("source") or {}).get("image")
                ),
                "original_target_path": path_for(
                    DATA / "source_images", (original_record.get("target") or {}).get("image")
                ),
                "original_mask_path": path_for(
                    DATA / "source_images", (original_record.get("target") or {}).get("mask")
                ),
                "candidate_manifest_path": str(CURRENT_ROOT / "manifest.jsonl"),
                "candidate_manifest_sha256": digest(CURRENT_ROOT / "manifest.jsonl"),
                "source_sha256": digest(scene_path if source_regenerated else source_input_path),
                "target_sha256": digest(target_path),
                "mask_sha256": digest(mask_path),
                "automatic_quality_passed": None,
                "automatic_quality_review_status": None,
                "replacement_status": "blocked_until_repair_complete",
                "fusion_status": "not_ready",
                "candidate_for_replacement": False,
                "supersedes_batch": "repair_batch_20260923_v3" if pair_id in overlap else None,
                "notes": (
                    "当前批次记录为有问题正在修改；当前快照source成功场景数为"
                    + str(len(source_success_scenes))
                    + "/"
                    + str(len(current_plan.get("source_scene_ids", [])))
                    + "，target/mask完成后需重新审计，未完成前不得替换或融合。"
                ),
            }
        )

    OUT_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
        + "\n",
        encoding="utf-8",
    )
    v3_records = [
        record for record in records if record["batch_id"] == "repair_batch_20260923_v3"
    ]
    current_records = [
        record for record in records if record["batch_id"] == "repair_batch_20240924_v1"
    ]
    summary = {
        "schema_version": "repair_replacement_tracking_v1",
        "tracking_created_date": "2026-09-24",
        "tracking_file": str(OUT_PATH),
        "original_manifest": str(original_manifest_path),
        "original_manifest_sha256": original_manifest_hash,
        "record_count": len(records),
        "distinct_pair_count": len({record["pair_id"] for record in records}),
        "overlap_pair_count": len(overlap),
        "v3_nonoverlap_candidate_count": sum(
            record["candidate_for_replacement"] for record in v3_records
        ),
        "batches": {
            "repair_batch_20260923_v3": {
                "record_count": len(v3_records),
                "review_status": "no_problem",
                "replacement_candidate_count": sum(
                    record["candidate_for_replacement"] for record in v3_records
                ),
                "held_by_current_batch_overlap_count": sum(
                    record["replacement_status"] == "hold_current_batch_overlap"
                    for record in v3_records
                ),
                "target_generated_count": sum(
                    record["target_status"] == "generated" for record in v3_records
                ),
                "mask_generated_count": sum(
                    record["mask_status"] == "generated" for record in v3_records
                ),
            },
            "repair_batch_20240924_v1": {
                "record_count": len(current_records),
                "review_status": "problem_under_repair",
                "source_scene_count": len(current_plan.get("source_scene_ids", [])),
                "source_success_count_at_snapshot": len(source_success_scenes),
                "source_pending_count_at_snapshot": len(
                    set(current_plan.get("source_scene_ids", [])) - source_success_scenes
                ),
                "reported_problem_count": sum(
                    record["repair_role"] == "reported_problem" for record in current_records
                ),
                "spillover_count": sum(
                    record["repair_role"] == "source_scene_spillover"
                    for record in current_records
                ),
                "target_generated_count_at_snapshot": sum(
                    record["target_status"] == "generated_unreviewed"
                    for record in current_records
                ),
                "target_pending_count_at_snapshot": sum(
                    record["target_status"] == "pending_generation"
                    for record in current_records
                ),
            },
        },
        "replacement_rule": (
            "同一pair若同时存在v3与当前v1记录，当前v1优先；"
            "v1完成并人工通过前，不得使用v3替换。"
        ),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
