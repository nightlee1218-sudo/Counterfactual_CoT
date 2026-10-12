#!/usr/bin/env python3
"""Apply the 2026-09-28 human audit of round2 (49 pairs) and build the round3 package.

Outputs
  1. round3_audit_decisions_20260928.jsonl   one decision row per audited pair
  2. round3_audit_decisions_20260928.xlsx    statistics + issue/measure sheets
  3. repair_round3_20260928/revised_cot_manifest.jsonl
  4. repair_round3_20260928/revised_source_registry.jsonl
  5. repair_round3_20260928/generation_plan.jsonl
  6. repair_round3_20260928/round3_plan_summary.json
  7. audit_page/repair_audit_filled_20260928.csv  audit CSV with decisions filled in

Nothing here modifies the original 1000-pair data.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/gold1000_framework_strict_v4")
ROUND2_PLAN = ROOT / "repair_round2_strict_atomic_20260924_v4" / "strict_atomic_generation_plan.jsonl"
ROUND2_REGISTRY = ROOT / "repair_round2_strict_atomic_20260924_v4" / "revised_source_registry.jsonl"
ROUND2_SANDBOX_MANIFEST = ROOT / "repair_batch_20260924_round2_strict_atomic_v4" / "manifest.jsonl"
AUDIT_CSV = ROOT / "repair_batch_20260924_round2_strict_atomic_v4" / "audit_page" / "repair_audit.csv"

# ---------------------------------------------------------------------------
# Human audit result 2026-09-28 (unlisted pairs default to "accept")
# ---------------------------------------------------------------------------
AUDIT: dict[str, dict[str, str]] = {
    "g1000_classroom__n3_chair__3_to_2_002": {
        "decision": "regen_source",
        "issue": "source 中的椅子数量/可数性不成立，3->2 的实例数量变更失败",
        "source_measure": "source 必须严格显示 3 把独立、可逐一计数的 student chair：横向排成一行、彼此完全分离、不重叠不堆叠、全部完整轮廓可见；不要出现部分出画或被遮挡的椅子。",
        "target_measure": "严格把 student chair 从 3 把变为 2 把：只移除其中一把，其余椅子的位置、颜色、材质、朝向完全不变，不得新增任何物体或留下残影。",
    },
    "g1000_closet_wardrobe__beanie_hatbox_scarf__1_to_2_002": {
        "decision": "regen_source",
        "issue": "source 里的 beanie 视觉上看起来是两只，1->2 的初始数量不成立",
        "source_measure": "beanie 在 source 中必须严格只有 1 个：单顶、独立放置、轮廓完整、与 hatbox/scarf 明显分离，不得出现两顶帽子或成对摆放造成的数量歧义。",
        "target_measure": "严格把 knitted beanie 从 1 个变为 2 个：只新增一个同款同色同材质的 beanie，其余物体的数量、位置、属性完全不变。",
    },
    "g1000_closet_wardrobe__socks_basket_pajamas_hat__occlusion_state_001": {
        "decision": "regen_source",
        "issue": "source 的遮盖关系不明确，hat 落在 sweater 正上方，unoccluded 初始状态不成立",
        "source_measure": "hat 必须严格不被遮盖、也不遮盖 sweater：把 hat 完全放在 sweater 旁边，两者之间有明显可见间隙，hat 与 sweater 的完整轮廓都可见；source 中不得存在任何接触或重叠。",
        "target_measure": "只让 hat 部分遮盖 sweater：hat 移到与 sweater 相邻并压住其边缘一小部分，hat 与 sweater 都保持可见；不得改变其他物体的位置、数量或属性。",
    },
    "g1000_dining_table__plate_quota__2_to_1_001": {
        "decision": "regen_source",
        "issue": "source 中 plate / glass 数量可数性不成立，数量变更失败",
        "source_measure": "source 中 dinner plate 严格 2 个、water glass 严格 3 个，不多不少；所有器物排成一行、彼此分离、完整可见、可逐一计数，不堆叠、不重叠、不部分出画。",
        "target_measure": "严格把 dinner plate 从 2 个变为 1 个：只移除其中一个 plate，water glass 仍为 3 个且位置不变，不得新增任何物体。",
    },
    "g1000_dining_table__plate_quota__2_to_3_001": {
        "decision": "regen_source",
        "issue": "source 中 plate / glass 数量可数性不成立，数量变更失败",
        "source_measure": "source 中 dinner plate 严格 2 个、water glass 严格 3 个，不多不少；所有器物排成一行、彼此分离、完整可见、可逐一计数，不堆叠、不重叠、不部分出画。",
        "target_measure": "严格把 dinner plate 从 2 个变为 3 个：只新增一个同款 plate，water glass 仍为 3 个且位置不变，不得改变其他物体。",
    },
    "g1000_dining_table__plate_quota__3_to_2_001": {
        "decision": "regen_source",
        "issue": "source 中 plate / glass 数量可数性不成立，数量变更失败（审计文字与 CoT 不一致，以 CoT：plate=2、glass=3 为准）",
        "source_measure": "source 中 dinner plate 严格 2 个、water glass 严格 3 个，不多不少；所有器物排成一行、彼此分离、完整可见、可逐一计数，不堆叠、不重叠、不部分出画。",
        "target_measure": "严格把 water glass 从 3 个变为 2 个：只移除其中一个 glass，dinner plate 仍为 2 个且位置不变，不得新增任何物体。",
    },
    "g1000_office_desk__stand_keyboard_pad__occlusion_state_002": {
        "decision": "regen_target",
        "issue": "target 中本子压在小桌子上，遮盖对象错误",
        "source_measure": "",
        "target_measure": "把 notebook 完整移到小桌子（stand）之外，不要遮盖住小桌子；stand 的完整轮廓必须可见，notebook 与 stand 明显分离放置；除 notebook 外不得改变任何物体。",
    },
    "g1000_pantry_shelf_closeup__cans_open_crate_out__containment_001": {
        "decision": "regen_source",
        "issue": "移入容器后看不到 object，容器内外数量不一致，与 prompt 不一致",
        "source_measure": "4 个 soup can 在 source 中必须全部位于 open crate 之外：排成一行、彼此分离、完整可见、可逐一计数；crate 为空且完整可见，开口朝向画面。",
        "target_measure": "把全部 4 个 soup can 一起移入 crate 内部：容器内外的 can 数量必须一致（外 4 -> 内 4），移入后每个 can 仍然可见可数，crate 本身位置与外观不变，不得新增或删除任何 can。",
    },
    "g1000_warehouse_shelf__box_sack_pallet__2_to_1_001": {
        "decision": "regen_source",
        "issue": "货架下层两个 pallet 叠在一起，且出现两行相同物体，2->1 无法执行",
        "source_measure": "只显示一行物体，不要两行；wooden pallet 严格 2 个且并排分离、不叠放、不紧贴，各自完整可见可单独移除；cardboard box 3 个、paper sack 2 个同样分区分离、可逐一计数。",
        "target_measure": "严格把 wooden pallet 从 2 个变为 1 个：只移除其中一个 pallet，其余物体数量、位置、属性完全不变。",
    },
    "g1000_warehouse_shelf__box_sack_pallet__2_to_3_001": {
        "decision": "regen_source",
        "issue": "货架下层两个 pallet 叠在一起，且出现两行相同物体，2->3 无法执行",
        "source_measure": "只显示一行物体，不要两行；wooden pallet 严格 2 个且并排分离、不叠放、不紧贴，各自完整可见可单独计数；cardboard box 3 个、paper sack 2 个同样分区分离、可逐一计数。",
        "target_measure": "严格把 wooden pallet 从 2 个变为 3 个：只新增一个同款 pallet，其余物体数量、位置、属性完全不变。",
    },
    "g1000_warehouse_shelf__box_sack_pallet__3_to_2_001": {
        "decision": "regen_source",
        "issue": "货架下层两个 pallet 叠在一起，且出现两行相同物体，3->2 无法执行",
        "source_measure": "只显示一行物体，不要两行；wooden pallet 严格 2 个且并排分离、不叠放、不紧贴，各自完整可见可单独移除；cardboard box 3 个、paper sack 2 个同样分区分离、可逐一计数。",
        "target_measure": "严格把 wooden pallet 从 3 个变为 2 个：只移除其中一个 pallet，其余物体数量、位置、属性完全不变。",
    },
    "g1000_warehouse_shelf__box_tote_drum__3_to_2_001": {
        "decision": "regen_source",
        "issue": "source 图里没有 3 个 tote，tote 数量变更失败（若重生成仍失败则删除该 pair）",
        "source_measure": "plastic storage tote 必须严格 3 个：横向并排、彼此分离、完整可见、可逐一计数；cardboard box 2 个、metal drum 2 个同样分区分离可数；不堆叠、不互相遮挡。",
        "target_measure": "严格把 plastic storage tote 从 3 个变为 2 个：只移除其中一个 tote，其余物体数量、位置、属性完全不变。",
    },
    "g1000_warehouse_shelf__drum_pallet_tote__2_to_3_001": {
        "decision": "delete",
        "issue": "数量变更失败且改善空间有限，审计决定删除该 pair",
        "source_measure": "",
        "target_measure": "",
    },
    "g1000_warehouse_shelf__sack_bundle_bin__3_to_2_001": {
        "decision": "regen_target",
        "issue": "3->2 数量变更失败，需要重新生成 target（source 已可接受）",
        "source_measure": "",
        "target_measure": "严格把 strapping bundle 从 3 个变为 2 个：只移除其中一个 bundle，paper sack 2 个、storage bin 2 个的数量与位置完全不变；移除后不得新增物体或留下残影。",
    },
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def generic_target_measure(record: dict) -> str:
    inter = record.get("intervention", {})
    return (
        f"严格只执行唯一干预：{inter.get('entity_id')} 的 {inter.get('field')} "
        f"从 {inter.get('old_value')} 变为 {inter.get('new_value')}；"
        "其余所有实体的数量、位置、属性、关系完全不变，不得新增或删除其他物体。"
    )


def build() -> dict:
    plans = {str(r["pair_id"]): r for r in load_jsonl(ROUND2_PLAN)}
    registry = {str(r["scene_id"]): r for r in load_jsonl(ROUND2_REGISTRY)}
    sandbox_manifest = {str(r["pair_id"]): r for r in load_jsonl(ROUND2_SANDBOX_MANIFEST)}

    audited = sorted(sandbox_manifest)
    scene_of = {pid: str(sandbox_manifest[pid]["source_scene_id"]) for pid in audited}
    members: dict[str, list[str]] = defaultdict(list)
    for pid in audited:
        members[scene_of[pid]].append(pid)

    # explicit decisions
    decisions: dict[str, dict] = {}
    for pid in audited:
        record = sandbox_manifest[pid]
        plan = plans.get(pid, {})
        item = AUDIT.get(pid)
        if item is None:
            decisions[pid] = {
                "pair_id": pid,
                "source_scene_id": scene_of[pid],
                "subtype": record.get("subtype", ""),
                "intervention": json.dumps(record.get("intervention", {}), ensure_ascii=False),
                "decision": "accept",
                "spillover": False,
                "user_issue": str(plan.get("user_issue", "")),
                "audit_issue": "审计未提出问题，默认通过",
                "source_measure": "",
                "target_measure": "",
                "repair_scope": "none",
            }
            continue
        decisions[pid] = {
            "pair_id": pid,
            "source_scene_id": scene_of[pid],
            "subtype": record.get("subtype", ""),
            "intervention": json.dumps(record.get("intervention", {}), ensure_ascii=False),
            "decision": item["decision"],
            "spillover": False,
            "user_issue": str(plan.get("user_issue", "")),
            "audit_issue": item["issue"],
            "source_measure": item["source_measure"],
            "target_measure": item["target_measure"],
            "repair_scope": "source+target" if item["decision"] == "regen_source"
            else ("target-only" if item["decision"] == "regen_target" else "none"),
        }

    # source regeneration pulls every pair sharing that source scene
    regen_scenes = {scene_of[p] for p, d in decisions.items() if d["decision"] == "regen_source"}
    for scene_id in sorted(regen_scenes):
        for pid in members[scene_id]:
            if decisions[pid]["decision"] == "accept":
                decisions[pid].update({
                    "decision": "regen_source_spillover",
                    "spillover": True,
                    "audit_issue": "同一 source scene 的连带 pair；source 重新生成后 target 必须同步重生成",
                    "source_measure": decisions[pid]["source_measure"] or "",
                    "target_measure": generic_target_measure(sandbox_manifest[pid]),
                    "repair_scope": "source+target",
                })

    to_generate = [p for p in audited if decisions[p]["decision"] in
                   {"regen_source", "regen_source_spillover", "regen_target"}]
    deleted = [p for p in audited if decisions[p]["decision"] == "delete"]
    accepted = [p for p in audited if decisions[p]["decision"] == "accept"]

    # ---- round3 package -------------------------------------------------
    out_dir = ROOT / "repair_round3_20260928"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    plan_rows = []
    for pid in sorted(to_generate):
        record = json.loads(json.dumps(sandbox_manifest[pid]))
        record["target"]["prompt"] = record.get("repair", {}).get(
            "original_target_prompt", record["target"]["prompt"])
        decision = decisions[pid]
        if not decision["target_measure"]:
            decision["target_measure"] = generic_target_measure(record)
        manifest_rows.append(record)
        plan_rows.append({
            "pair_id": pid,
            "source_scene_id": decision["source_scene_id"],
            "repair_scope": decision["repair_scope"],
            "repair_role": "reported_problem" if not decision["spillover"] else "source_scene_spillover",
            "user_issue": decision["audit_issue"],
            "source_measure": decision["source_measure"],
            "target_measure": decision["target_measure"],
            "approval_status": "round2_human_audit_20260928",
            "audit_decision": decision["decision"],
        })

    scene_ids = sorted({decisions[p]["source_scene_id"] for p in to_generate
                        if decisions[p]["repair_scope"] == "source+target"})
    registry_rows = [registry[s] for s in scene_ids]

    (out_dir / "revised_cot_manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest_rows), encoding="utf-8")
    (out_dir / "revised_source_registry.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in registry_rows), encoding="utf-8")
    (out_dir / "generation_plan.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in plan_rows), encoding="utf-8")

    summary = {
        "audit_date": "2026-09-28",
        "audited_pairs": len(audited),
        "accept": len(accepted),
        "regen_source": sum(1 for p in audited if decisions[p]["decision"] == "regen_source"),
        "regen_source_spillover": sum(1 for p in audited if decisions[p]["decision"] == "regen_source_spillover"),
        "regen_target": sum(1 for p in audited if decisions[p]["decision"] == "regen_target"),
        "delete": len(deleted),
        "targets_to_generate": len(to_generate),
        "source_scenes_to_regenerate": scene_ids,
        "deleted_pair_ids": deleted,
        "round3_manifest": str(out_dir / "revised_cot_manifest.jsonl"),
        "round3_registry": str(out_dir / "revised_source_registry.jsonl"),
        "round3_plan": str(out_dir / "generation_plan.jsonl"),
    }
    (out_dir / "round3_plan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    decisions_path = ROOT / "round3_audit_decisions_20260928.jsonl"
    decisions_path.write_text(
        "".join(json.dumps(decisions[p], ensure_ascii=False) + "\n" for p in audited), encoding="utf-8")

    # ---- filled audit csv ------------------------------------------------
    filled = ROOT / "repair_batch_20260924_round2_strict_atomic_v4" / "audit_page" / "repair_audit_filled_20260928.csv"
    with AUDIT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        pid = str(row.get("pair_id", "")).strip()
        decision = decisions.get(pid)
        if not decision:
            continue
        row["audit_decision"] = {"accept": "accept", "delete": "delete"}.get(
            decision["decision"], "regen")
        notes = decision["audit_issue"]
        if decision["repair_scope"] == "source+target":
            notes += f"｜措施[source]: {decision['source_measure']}"
        if decision["target_measure"]:
            notes += f"｜措施[target]: {decision['target_measure']}"
        row["audit_notes"] = notes
    with filled.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---- xlsx ------------------------------------------------------------
    workbook = build_workbook(decisions, audited, summary, to_generate, deleted, accepted)
    xlsx_path = ROOT / "round3_audit_decisions_20260928.xlsx"
    workbook.save(xlsx_path)

    summary["files"] = {
        "decisions_jsonl": str(decisions_path),
        "xlsx": str(xlsx_path),
        "filled_csv": str(filled),
    }
    return summary


def build_workbook(decisions, audited, summary, to_generate, deleted, accepted) -> Workbook:
    head_fill = PatternFill("solid", fgColor="305496")
    head_font = Font(color="FFFFFF", bold=True)
    workbook = Workbook()

    def add_sheet(title: str, headers: list[str], rows: list[list], widths: list[int]):
        sheet = workbook.create_sheet(title)
        sheet.append(headers)
        for index, _ in enumerate(headers, start=1):
            cell = sheet.cell(row=1, column=index)
            cell.fill = head_fill
            cell.font = head_font
            sheet.column_dimensions[get_column_letter(index)].width = widths[index - 1]
        for row in rows:
            sheet.append(row)
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        sheet.freeze_panes = "A2"
        return sheet

    summary_sheet = workbook.active
    summary_sheet.title = "汇总统计"
    summary_sheet.append(["项目", "数量"])
    for cell in summary_sheet[1]:
        cell.fill = head_fill
        cell.font = head_font
    rows = [
        ["本轮审计 pair 总数", summary["audited_pairs"]],
        ["通过（可直接替换进 1000）", summary["accept"]],
        ["需重新生成 source", summary["regen_source"]],
        ["source 连带重生成 target", summary["regen_source_spillover"]],
        ["仅需重新生成 target", summary["regen_target"]],
        ["删除", summary["delete"]],
        ["本轮预计生成 target 数", summary["targets_to_generate"]],
        ["需重新生成的 source 场景数", len(summary["source_scenes_to_regenerate"])],
        ["审计日期", summary["audit_date"]],
    ]
    for row in rows:
        summary_sheet.append(row)
    summary_sheet.column_dimensions["A"].width = 34
    summary_sheet.column_dimensions["B"].width = 60

    add_sheet(
        "决策明细",
        ["pair_id", "source_scene_id", "subtype", "intervention", "decision", "连带", "round2 原始问题", "复核问题", "改善方法"],
        [[decisions[p]["pair_id"], decisions[p]["source_scene_id"], decisions[p]["subtype"],
          decisions[p]["intervention"], decisions[p]["decision"],
          "是" if decisions[p]["spillover"] else "否",
          decisions[p]["user_issue"], decisions[p]["audit_issue"],
          " ".join(x for x in (("source: " + decisions[p]["source_measure"]) if decisions[p]["source_measure"] else "",
                               ("target: " + decisions[p]["target_measure"]) if decisions[p]["target_measure"] else "") if x)]
         for p in audited],
        [46, 40, 16, 46, 22, 8, 40, 44, 70],
    )
    add_sheet(
        "重生成清单",
        ["pair_id", "repair_scope", "是否连带", "source_measure", "target_measure"],
        [[decisions[p]["pair_id"], decisions[p]["repair_scope"],
          "是" if decisions[p]["spillover"] else "否",
          decisions[p]["source_measure"], decisions[p]["target_measure"]]
         for p in sorted(to_generate)],
        [46, 16, 10, 70, 70],
    )
    add_sheet(
        "通过清单",
        ["pair_id", "source_scene_id", "subtype", "备注"],
        [[p, decisions[p]["source_scene_id"], decisions[p]["subtype"], "直接替换进 1000 pairs"]
         for p in accepted],
        [46, 40, 16, 30],
    )
    add_sheet(
        "删除清单",
        ["pair_id", "source_scene_id", "subtype", "原因"],
        [[p, decisions[p]["source_scene_id"], decisions[p]["subtype"], decisions[p]["audit_issue"]]
         for p in deleted],
        [46, 40, 16, 60],
    )
    add_sheet(
        "问题与改善方法",
        ["pair_id", "问题分类", "问题原因", "source 改善方法", "target 改善方法"],
        [[p, decisions[p]["decision"], decisions[p]["audit_issue"],
          decisions[p]["source_measure"], decisions[p]["target_measure"]]
         for p in audited if p in AUDIT],
        [46, 16, 60, 70, 70],
    )
    return workbook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    summary = build()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
