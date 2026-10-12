#!/usr/bin/env python3
"""Record v3 / v1 batches as human-approved and build the full replacement package.

Decision source
  - v3  batch: 261 records marked no_problem -> 187 candidates pass, 74 held by v1 overlap
  - v1  batch: 81 records -> 49 handed to round2, 32 pass by user default rule
  - round2   : 49 audited -> 42 pass, 7 deleted (repair_status_20260928_final.jsonl)
  - baseline : 732 untouched

Outputs
  - human_approval_20260928.jsonl          268 decisions
  - final_replacement_20260928_full.jsonl  261 records ready for fusion
  - final_replacement_20260928_full_summary.json
  - audit_and_replacement_20260928_final.xlsx
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/gold1000_framework_strict_v4")
LEDGER = ROOT / "repair_cumulative_status_20260924_post_generation.jsonl"
ROUND2_STATUS = ROOT / "repair_status_20260928_final.jsonl"
ORIGINAL_MANIFEST = ROOT / "full_1000_20260918" / "gold1000.generated.jsonl"
V3_ROOT = ROOT / "repair_batch_20260923_v3"
V1_ROOT = ROOT / "repair_batch_20240924_v1"
ROUND2_ROOT = ROOT / "repair_batch_20260924_round2_strict_atomic_v4"
ROUND3_ROOT = ROOT / "repair_batch_20260928_round3"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(batch_root: Path, value: str) -> str:
    path = Path(str(value))
    return str(path if path.is_absolute() else batch_root / path)


def build() -> dict:
    ledger = {str(r["pair_id"]): r for r in load_jsonl(LEDGER)}
    round2_status = {str(r["pair_id"]): r for r in load_jsonl(ROUND2_STATUS)}
    original = {str(r["pair_id"]): r for r in load_jsonl(ORIGINAL_MANIFEST)}
    v3_manifest = {str(r["pair_id"]): r for r in load_jsonl(V3_ROOT / "manifest.jsonl")}
    v1_manifest = {str(r["pair_id"]): r for r in load_jsonl(V1_ROOT / "manifest.jsonl")}
    round2_manifest = {str(r["pair_id"]): r for r in load_jsonl(ROUND2_ROOT / "manifest.jsonl")}
    round3_manifest = {str(r["pair_id"]): r for r in load_jsonl(ROUND3_ROOT / "manifest.jsonl")}

    decisions: list[dict] = []
    replacement: list[dict] = []
    missing: list[tuple[str, str]] = []

    for pid in sorted(ledger):
        row = ledger[pid]
        state = str(row.get("current_state"))

        if state == "original_baseline_default_reasonable":
            decisions.append({
                "pair_id": pid, "batch": "baseline", "decision": "keep_original",
                "review_basis": "各轮均未列出，用户规则默认合理",
            })
            continue

        if state == "v3_candidate_pending_fusion":
            record = v3_manifest.get(pid)
            if record is None:
                decisions.append({"pair_id": pid, "batch": "v3", "decision": "missing_v3_record", "review_basis": ""})
                continue
            asset_root = V3_ROOT
            batch = "v3"
            decision = "pass_human_approved"
            basis = "v3 已记录 no_problem，用户确认为人审通过"
        elif state == "v1_default_reasonable_pending_fusion":
            record = v1_manifest.get(pid)
            if record is None:
                decisions.append({"pair_id": pid, "batch": "v1", "decision": "missing_v1_record", "review_basis": ""})
                continue
            asset_root = V1_ROOT
            batch = "v1"
            decision = "pass_human_approved"
            basis = "v1 已生成资产，用户规则默认合理并确认为人审通过"
        elif state == "round2_generated_pending_audit":
            status = round2_status.get(pid, {})
            final = str(status.get("final_decision", ""))
            if final.startswith("deleted"):
                decisions.append({
                    "pair_id": pid, "batch": "round2", "decision": final,
                    "review_basis": "人工审计不通过，删除",
                })
                continue
            if pid in round3_manifest:
                record, asset_root, batch = round3_manifest[pid], ROUND3_ROOT, "round3"
            else:
                record, asset_root, batch = round2_manifest[pid], ROUND2_ROOT, "round2"
            decision = "pass_human_approved"
            basis = "round2/round3 分页人工审计通过"
        else:
            decisions.append({"pair_id": pid, "batch": state, "decision": "unhandled", "review_basis": ""})
            continue

        source = asset_root / str(record["source"]["image"])
        target = asset_root / str(record["target"]["image"])
        mask_value = str(record["target"].get("mask", ""))
        mask = asset_root / mask_value if mask_value else None
        for label, path in (("source", source), ("target", target), ("mask", mask)):
            if path is not None and not path.is_file():
                missing.append((pid, f"{label}:{path}"))

        out = json.loads(json.dumps(record))
        out["source"]["image"] = str(source)
        out["target"]["image"] = str(target)
        if mask is not None:
            out["target"]["mask"] = str(mask)
        out["replacement"] = {
            "replaced_at": "2026-09-28",
            "asset_batch": batch,
            "human_approved": True,
            "review_basis": basis,
            "source_scene_id": str(row.get("source_scene_id_original", "")),
            "source_sha256": sha256_file(source) if source.is_file() else "",
            "target_sha256": sha256_file(target) if target.is_file() else "",
            "mask_sha256": sha256_file(mask) if mask is not None and mask.is_file() else "",
        }
        replacement.append(out)
        decisions.append({"pair_id": pid, "batch": batch, "decision": decision, "review_basis": basis})

    decisions_path = ROOT / "human_approval_20260928.jsonl"
    decisions_path.write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in decisions), encoding="utf-8")

    replacement_path = ROOT / "final_replacement_20260928_full.jsonl"
    with replacement_path.open("w", encoding="utf-8") as handle:
        for record in replacement:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    kept_baseline = len(ledger) - sum(1 for d in decisions if d["batch"] not in {"baseline", ""})
    kept_baseline = sum(1 for d in decisions if d["batch"] == "baseline")
    deleted = [d["pair_id"] for d in decisions if d["decision"].startswith("deleted")]

    summary = {
        "created_at": "2026-09-28",
        "ledger_pairs": len(ledger),
        "decision_counts": dict(Counter(d["decision"] for d in decisions)),
        "asset_batch_counts": dict(Counter(r["replacement"]["asset_batch"] for r in replacement)),
        "keep_original_baseline": kept_baseline,
        "to_replace": len(replacement),
        "deleted": len(deleted),
        "deleted_pair_ids": deleted,
        "final_dataset_size_if_applied": kept_baseline + len(replacement),
        "missing_assets": missing[:20],
        "missing_asset_count": len(missing),
        "replacement_jsonl": str(replacement_path),
        "decisions_jsonl": str(decisions_path),
    }
    (ROOT / "final_replacement_20260928_full_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    xlsx_path = ROOT / "audit_and_replacement_20260928_final.xlsx"
    build_workbook(decisions, replacement, summary).save(xlsx_path)
    summary["xlsx"] = str(xlsx_path)
    return summary


def build_workbook(decisions, replacement, summary) -> Workbook:
    head_fill = PatternFill("solid", fgColor="305496")
    head_font = Font(color="FFFFFF", bold=True)
    workbook = Workbook()

    def add_sheet(title: str, headers: list[str], data: list[list], widths: list[int]):
        sheet = workbook.create_sheet(title)
        sheet.append(headers)
        for index in range(1, len(headers) + 1):
            cell = sheet.cell(row=1, column=index)
            cell.fill = head_fill
            cell.font = head_font
            sheet.column_dimensions[get_column_letter(index)].width = widths[index - 1]
        for row in data:
            sheet.append(row)
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        sheet.freeze_panes = "A2"

    overview = workbook.active
    overview.title = "总览"
    overview.append(["项目", "数量"])
    for cell in overview[1]:
        cell.fill = head_fill
        cell.font = head_font
    overview.append(["台账 pair 总数", summary["ledger_pairs"]])
    overview.append(["保持原始基线", summary["keep_original_baseline"]])
    overview.append(["人审通过待替换", summary["to_replace"]])
    overview.append(["删除", summary["deleted"]])
    overview.append(["缺失资产", summary["missing_asset_count"]])
    overview.append(["执行融合后数据集规模", summary["final_dataset_size_if_applied"]])
    for batch, count in sorted(summary["asset_batch_counts"].items()):
        overview.append([f"资产来源 {batch}", count])
    overview.column_dimensions["A"].width = 32
    overview.column_dimensions["B"].width = 22

    add_sheet(
        "替换清单",
        ["pair_id", "subtype", "asset_batch", "human_approved", "source_sha256", "target_sha256", "mask_sha256", "source", "target", "mask"],
        [[r["pair_id"], r.get("subtype", ""), r["replacement"]["asset_batch"], "是",
          r["replacement"]["source_sha256"][:16], r["replacement"]["target_sha256"][:16],
          r["replacement"]["mask_sha256"][:16], r["source"]["image"], r["target"]["image"],
          r["target"].get("mask", "")] for r in replacement],
        [46, 18, 14, 14, 20, 20, 20, 56, 56, 56],
    )
    add_sheet(
        "删除清单",
        ["pair_id", "decision"],
        [[d["pair_id"], d["decision"]] for d in decisions if d["decision"].startswith("deleted")],
        [50, 24],
    )
    add_sheet(
        "全部决策",
        ["pair_id", "batch", "decision", "review_basis"],
        [[d["pair_id"], d["batch"], d["decision"], d.get("review_basis", "")] for d in decisions],
        [46, 14, 24, 60],
    )
    return workbook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(build(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
