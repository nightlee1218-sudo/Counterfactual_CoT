#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/gold1000_framework_strict_v4")
DATA = ROOT / "full_1000_20260918"
V1_ROOT = ROOT / "repair_batch_20240924_v1"
BASE_TRACKING = ROOT / "repair_replacement_tracking_20260924.jsonl"
ROUND2_PLAN = ROOT / "detailed_repair_plan_20260924_round2.jsonl"
ROUND2_GENERATION_PLAN = ROOT / "repair_round2_generation_plan_20260924.jsonl"
ROUND2_SCENE_PLAN = ROOT / "scene_repair_plan_20260924_round2.json"
ROUND2_TRACKING = ROOT / "repair_round2_tracking_20260924.jsonl"
CUMULATIVE_TRACKING = ROOT / "repair_replacement_tracking_20260924_v2.jsonl"
ROUND2_SUMMARY = ROOT / "repair_round2_summary_20260924.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cot_signature(record: dict[str, Any]) -> str:
    cot = record.get("source", {}).get("cot", {})
    canonical = json.dumps(cot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha_text(canonical)


def entity(record: dict[str, Any], entity_id: str) -> dict[str, Any]:
    for item in record.get("source", {}).get("cot", {}).get("entities", []):
        if str(item.get("id")) == str(entity_id):
            return item
    return {}


def entity_label(record: dict[str, Any]) -> str:
    intervention = record.get("intervention", {})
    item = entity(record, str(intervention.get("entity_id", "")))
    category = item.get("category", "target entity")
    return f"{intervention.get('entity_id')} ({category}, source count={item.get('count')}, source location={item.get('location')})"


def generic_source(record: dict[str, Any]) -> str:
    return (
        "严格以原始 source COT 为语义真值；保留全部 entity_id、category、count、location、"
        "attributes、relations、背景、镜头和材质，不添加、删除、复制或合并任何未列出的实体。"
        "每个实体必须按 individual instances 清晰可数；不得把一摞、一行、一个容器误当成数量。"
        "生成 native 1024x1024 square full-bleed scene，所有对象完整入框，禁止裁切、边框、留白、"
        "拼贴、文字、logo、水印和 UI；为本轮 target 操作预留物理净空。"
    )


def generic_target(record: dict[str, Any]) -> str:
    iv = record.get("intervention", {})
    return (
        f"只允许一个原子变化：entity_id={iv.get('entity_id')}，field={iv.get('field')}，"
        f"{iv.get('old_value')} -> {iv.get('new_value')}。保持所有 must_preserve 实体、数量、"
        "位置、尺寸、方向、材质、背景和镜头不变；不得新增、删除、复制、移动或改变无关对象。"
        "目标变化必须肉眼明显且可逐项验收，mask 应主要覆盖 intervention 对象/关系区域。"
    )


# source_required is intentionally conservative for failed count/placement cases: when the
# destination capacity or the old state is ambiguous, regenerate a clean source instead of
# asking the target editor to repair an impossible layout.
SPECS: dict[str, dict[str, Any]] = {
    "g1000_bedroom_nightstand__glasses_charger_glass__occlusion_state_008": {
        "source_required": True,
        "issue": "需修改source；当前source存在部分遮盖，初始unoccluded状态不成立。",
        "source_extra": "初始 source 中 cream ceramic bedside lamp 与 clear water glass 必须完全分离，有明显间隙，两个轮廓完整可见；不要让灯、桌沿或其他结构形成隐性遮挡。",
        "target_extra": "将同一个 bedside lamp 放到 water glass 前方，恰好遮挡 glass 约四分之一至三分之一轮廓；两者都必须保留且可辨，不得变成左右换位、灯遮住全部玻璃或新增第三物体。",
        "validation": "source=完全无重叠；target=lamp 前景遮挡 glass 约25%-33%，两对象各1个，灯位置/尺寸/属性保持。",
    },
    "g1000_bedroom_nightstand__lamp_glass_tray__absolute_vertical_001": {
        "source_required": True,
        "issue": "table lamp 被截断只剩底座；需改成完整、较迷你的灯并为下移留出空间。",
        "source_extra": "使用完整入框的 compact mini cream ceramic table lamp：底座、灯杆和灯罩全部可见，灯不贴边、不被桌面裁断；glass 和 leather tray 保持完整，lower_left 目标区域留出至少一个灯体宽度净空。",
        "target_extra": "只把同一个完整 mini table lamp 从 upper_left 移到 lower_left；清除旧位置残影，灯必须完整落在支撑面上；glass、tray、桌面和画面边界完全不动。",
        "validation": "source=灯完整入框且有可辨灯杆/灯罩；target=灯仅改变位置，完整位于lower_left，无截断、悬空、穿透或无关物体位移。",
    },
    "g1000_bedroom_nightstand__lamp_glass_tray__absolute_vertical_004": {
        "source_required": True,
        "issue": "lamp 太大，无法从 top_region 移到 center_region；需改成 mini lamp。",
        "source_extra": "使用完整入框的 compact mini cream ceramic table lamp，保留灯的类别、颜色、陶瓷材质和单个数量；center_region 必须有完整灯体高度的空位，灯不贴边、不切桌面。",
        "target_extra": "只将同一个 mini table lamp 从 top_region 移到 center_region，保持尺寸、方向和细节；旧位置恢复为自然背景，glass/tray及桌面结构不动。",
        "validation": "source=灯完整且目标center_region有净空；target=灯完整落在center_region，未改尺寸、未裁切、未移动glass/tray。",
    },
    "g1000_bookshelf_study__mugcollection_folder__absolute_vertical_001": {
        "source_required": True,
        "issue": "folder_2 只发生旋转，没有 vertical 位移；source COT/布局没有为上下移动提供足够空间。",
        "source_extra": "source 必须有两个清晰、可区分且高度足够的 shelf levels；同一个 magazine folder（entity_id=folder_2）处于明确 upper level，lower destination 完全留空，至少容纳一个 folder 高度；folder 不得被 hardcover book 或货架结构压住。",
        "target_extra": "只把同一个 folder_2 从 upper level vertical position 移到 lower level 对应 vertical position；禁止仅旋转、倾斜、缩放或在原位重画；旧位置必须为空，hardcover book 和其他对象完全不动。",
        "validation": "同一个 folder_2 跨越两个明确层级并发生明显 vertical 位移；数量、朝向、尺寸和材质不变，无第二个folder、无仅旋转。",
    },
    "g1000_bookshelf_study__penjar_folder_trophy__absolute_vertical_001": {
        "source_required": True,
        "issue": "right_region -> upper_right 不足以产生有效变化，原图中 trophy_1 已接近 upper_right；需重设旧状态并强化 prompt。",
        "source_extra": "将 intervention entity trophy_1 放在 right_region 的中部或右下部，明确远离 upper_right；upper_right 留出完整实体净空，确保 old right_region 与 new upper_right 可视觉区分。",
        "target_extra": "只把 trophy_1 从明确的 right-middle/right-lower 位置移到 upper_right；必须产生明显 vertical 位移，不得原地旋转、微移或把 hardcover book/bookend 移动。",
        "validation": "source old位置不在upper_right；target trophy_1 完整位于upper_right，旧位置清空，无旋转代替位移。",
    },
    "g1000_classroom__chair_backpack_notebook__1_to_2_002": {
        "source_required": True,
        "issue": "Jacket 变成两个时遮盖 notebook；应在同一摞原实例正上方增加一个。",
        "source_extra": "source 只放 exactly 1 folded jacket（entity_id=jacket_1），置于单一垂直堆叠区域；jacket 上方预留至少一个 jacket 高度，notebook 必须在旁边且完整可见，不位于堆叠扩展路径下。",
        "target_extra": "在原 jacket_1 正上方、同一摞内部增加恰好1个 individual jacket；不得生成第二摞/新区域，不得遮盖、移动、缩放或重画 notebook；原 jacket 保持原位和细节。",
        "validation": "source=1个清晰jacket；target=同一摞上下两层、恰好2个jacket，notebook完整可见且位置不变。",
    },
    "g1000_classroom__n3_chair__3_to_2_002": {
        "source_required": True,
        "issue": "3to2 失败，需要重新生成清晰 source。",
        "source_extra": "source 必须展示 exactly 3 个可独立辨认、互不混叠的 individual student chair，统一在同一明确区域/行内，保留删除一个后的净空；禁止重复背景椅、重叠成不可数轮廓。",
        "target_extra": "从同一组 exactly 3 把 chair 中移除恰好1把，得到2把；幸存两把位置、尺寸、方向和细节不变，不移动或复制其他对象。",
        "validation": "source=3个可数chair；target=同一组2个chair，非3、非1、无位置漂移。",
    },
    "g1000_closet_wardrobe__beanie_hatbox_scarf__1_to_2_002": {
        "source_required": True,
        "issue": "当前 manifest 的实际 intervention 是 scarf_1（folded scarf），而 pair 名称含 beanie；按 COT canonical entity 处理为 scarf_1，先避免把错误类别当成目标对象。",
        "source_extra": "source 只展示 exactly 1 个清晰 folded scarf（entity_id=scarf_1）；在同一垂直堆叠位置上方预留一个 scarf 高度，context sweater 不占用该净空。若人工确认目标应为 beanie，必须先改写 source/target COT 的 category，再进入生图。",
        "target_extra": "在原 scarf_1 正上方、同一摞内部增加恰好1个同类 folded scarf；不得创建第二个区域、第二排或移动 context sweater。",
        "validation": "source=1个清晰scarf；target=同一堆叠区域2个上下相邻scarf，无第三个、无新区域；不得把 scarf 错生成为 beanie。",
        "semantic_note": "pair_id 的 beanie 命名与 manifest 的 folded scarf entity 不一致；本计划不静默改类别。",
    },
    "g1000_closet_wardrobe__n2_sweater__2_to_3_002": {
        "source_required": True,
        "issue": "一沓 2 件应变成同一沓 3 件；需重新生成 source 以提供空间。",
        "source_extra": "source 必须是同一摞内 exactly 2 个 individual folded sweater，层边清晰可数；摞上方预留一件 sweater 的高度，禁止第二摞、第二行或额外衣物。",
        "target_extra": "只在原两件 sweater 的同一摞最上方增加恰好1件，最终同一摞恰好3件；保留原两件层边、位置、材质和纹理，不改其他衣物。",
        "validation": "source=同一摞2层；target=同一摞3层，未生成第二摞/新行，其他物体不动。",
    },
    "g1000_closet_wardrobe__socks_basket_pajamas_hat__occlusion_state_001": {
        "source_required": True,
        "issue": "需修改source；原遮盖关系不明确，hat 不要落在 sweater 正上方。",
        "source_extra": "按 COT 建立清晰、可判断的指定遮挡关系；hat 必须横向偏移/位于 sweater 侧前方或侧后方，不能正好落在 sweater 垂直投影正上方，也不能与无关衣物混叠；两对象都完整可辨。",
        "target_extra": "只改变 intervention 指定的 occlusion_state，使用已有 hat/sweater，不新增或移动无关衣物；目标关系需由轮廓遮挡清楚表达，不能变成简单上下叠放或对象消失。",
        "validation": "source=指定初始遮挡清晰且hat不在sweater正上方；target=关系按COT改变，两者数量和其他位置不变。",
    },
    "g1000_dining_table__fruit_bowl_napkins__containment_001": {
        "source_required": False,
        "issue": "必须把碗内全部 3 个苹果拿出放在桌上，不能只拿出1个并删除另外2个。",
        "source_extra": "",
        "target_extra": "把 bowl 内原有的全部3个 individual apple 一次性移到同一张桌面上，三者均须保留、可独立数出且与 bowl 分离；bowl 的位置、形状、数量和其他 napkin 完全不变，禁止删除或复制苹果。",
        "validation": "source=3个apple在bowl内；target=同一3个apple全部在桌面外置，数量仍3，bowl仍在原位。",
    },
    "g1000_dining_table__napkin_fill__3_to_2_001": {
        "source_required": True,
        "issue": "source 的 3 个 napkin 不明显，需要重新生成 source。",
        "source_extra": "source 必须展示 exactly 3 个可独立辨认的 individual napkin，三个折叠层/实例清晰可数，不被盘、碗或彼此完全遮住。",
        "target_extra": "从同一组3个 napkin 中移除恰好1个，剩余2个保持原位置/折叠细节；不得移动餐桌上其他对象。",
        "validation": "source=清晰3个；target=清晰2个，幸存者未移动，未出现额外napkin。",
    },
    "g1000_dining_table__plate_quota__2_to_3_001": {
        "source_required": True,
        "issue": "增加 plate 时不得改变 water glass（entity_id=glass_2）的位置。",
        "source_extra": "source 使用 old count=2 的 dinner plate 状态，两个 plate 清晰可数并在同一目标区域留出第三个 plate 的空位；center_region 的 3 个 water glass 远离该扩展路径且完整可见。",
        "target_extra": "在原 plate_1 组内增加恰好1个 plate，最终3个；water glass（glass_2，count=3）的像素位置、方向、大小、材质和遮挡关系完全不变，禁止为了放第三个而移动或重画 water glass。",
        "validation": "source=2 plates；target=3 plates；3 个 water glass 的位置和外观保持，未被挪动/重画。",
    },
    "g1000_grocery_display__biscuit_tea_honey__2_to_1_001": {
        "source_required": False,
        "issue": "2to1 同时不得修改任何其他 object；实际 intervention 是 tea_2（tea caddy），不是 biscuit。",
        "source_extra": "",
        "target_extra": "只从同一 tea_2 组移除恰好1个 tea caddy，剩余1个保持原位、尺寸、材质和细节；biscuit tin、honey jar、货架和背景完全不动，不重排其他 object。",
        "validation": "target 恰好1个 tea caddy；biscuit tin/honey jar及其位置无变化，mask不覆盖无关对象。",
    },
    "g1000_grocery_display__biscuit_tea_honey__3_to_2_001": {
        "source_required": True,
        "issue": "3to2 失败，需要重新生成符合 honey_3 old count=3 的 source；实际 intervention 是 honey jar，不是 biscuit。",
        "source_extra": "source variant 必须展示 exactly 3 个可数 honey jar（entity_id=honey_3），属于同一明确组且无重复行/隐藏实例；为移除1个保留局部净空，biscuit tin/tea caddy 不变。",
        "target_extra": "从同一组三个 honey jar 中移除恰好1个，最终2个；保留两个幸存者的位置与细节，biscuit tin 和 tea caddy 不动。",
        "validation": "source honey jar=3；target honey jar=2；没有把 tea 2to1 或其他 count 状态混入同一 source variant。",
    },
    "g1000_grocery_display__flowers_vase_plants__absolute_horizontal_001": {
        "source_required": True,
        "gate": "blocked_cot_revision",
        "issue": "用户明确要求只移动一个 plant，但原始 COT 的 plants_1.count=2；必须先把 source/target COT 同步为 count=1，避免 prompt 同时要求2个和1个。",
        "source_extra": "COT 修订为 plants_1（potted plant）count=1 后，source 只保留 exactly 1 个 intervention plant；在 left_region 与 right_region 之间留出完整净空，wooden display crate 不得挡住目标区域。",
        "target_extra": "COT 修订为 count=1 后，只移动这1个 plant：旧位置完全清除并恢复背景，新位置完整入框且不贴边；不得复制第二株、越出画面、移动 wooden display crate 或 woven produce basket。",
        "validation": "source/target 的 plants_1 均为1；同一株仅出现1次，旧位置为空，新位置完整入框，context objects 不动。",
        "cot_revision": {"entity_id": "plants_1", "field": "count", "from": 2, "to": 1, "reason": "用户要求只进行一个 plant 的移动；否则 source/target COT 与视觉验收冲突"},
    },
    "g1000_grocery_display__gd04b_flowers_vase_separate__containment_001": {
        "source_required": True,
        "gate": "blocked_cot_revision",
        "issue": "用户明确要求只要一朵花，但原始 COT 的 flowers_1.count=5；必须先同步 count。",
        "source_extra": "COT 修订为 flowers_1（tulip flower）count=1 后，source 必须只有 exactly 1 朵 intervention flower 和1个指定 vase，花朵完整可见、与其他花/叶不混叠；为目标位置留出空间。",
        "target_extra": "COT 修订为 count=1 后，只把这1朵 flower 放入同一个 vase，完成 separate_from -> inside；vase 原位不动，花数量仍1，不新增第二朵、不删除花瓣主体。",
        "validation": "source/target 均恰好1朵 tulip flower；vase位置不变；containment 变化清晰且花完整可见。",
        "cot_revision": {"entity_id": "flowers_1", "field": "count", "from": 5, "to": 1, "reason": "用户要求只保留一朵花并只移动这一朵"},
    },
    "g1000_grocery_display__gd05b_milk_crate_separate__containment_001": {
        "source_required": True,
        "gate": "blocked_cot_revision",
        "issue": "用户明确要求只保留一个 milk，但原始 COT 的 milk_1.count=3；必须先同步 count。",
        "source_extra": "COT 修订为 milk_1（milk carton）count=1 后，source 必须只有 exactly 1 个 intervention milk object，crate 轮廓和开口完整、目标路径无遮挡；禁止第二个 milk 或相似包装干扰。",
        "target_extra": "COT 修订为 count=1 后，只将这1个 milk 按 COT 从 separate_from 移入同一个 crate；crate 保持原位、尺寸和结构，milk 数量仍1且完全可验证。",
        "validation": "source/target 均1个 milk carton；crate未移动；containment 改变明确，无第二个milk/删除milk。",
        "cot_revision": {"entity_id": "milk_1", "field": "count", "from": 3, "to": 1, "reason": "用户要求只保留一个 milk 并移动该单个实例"},
    },
    "g1000_living_room_play__ball_quota__2_to_1_001": {
        "source_required": False,
        "issue": "2to1 时不得改变其他 object 的位置。",
        "source_extra": "",
        "target_extra": "从同一 ball 组移除恰好1个；幸存 ball 和所有无关玩具保持原像素语义位置，禁止重排、缩放或背景重绘。",
        "validation": "target恰好1个ball；其他玩具位置不变，幸存ball不移动。",
    },
    "g1000_living_room_play__n1_rattle__2_to_1_001": {
        "source_required": True,
        "issue": "2to1 时不得改变其他 object 位置，且 source 中 rattle 不得混叠。",
        "source_extra": "source variant old count=2，两个 baby rattle 完全分离、轮廓和手柄清晰可数，放在同一明确目标区域，其他玩具远离该区域。",
        "target_extra": "只移除恰好1个 rattle_1，幸存 baby rattle 原位原尺寸；所有其他玩具位置不变，不允许把移除操作变成重排。",
        "validation": "source=2个清晰baby rattle；target=1个，其他玩具无位置变化。",
    },
    "g1000_living_room_play__n1_rattle__2_to_3_001": {
        "source_required": True,
        "issue": "2to3 失败，重新生成 source，避免 rattle 混叠。",
        "source_extra": "source variant old count=2，两个 baby rattle 清晰分离，并在同一目标组内预留第三个实例的空位，不与其他玩具重叠。",
        "target_extra": "在同一 rattle_1 组内增加恰好1个 individual baby rattle，最终3个；不得新建第二组、改变其他玩具或把三个rattle画成混叠团。",
        "validation": "source=2个；target=同一组3个清晰baby rattle，无重复行/新区域。",
    },
    "g1000_living_room_play__n1_rattle__3_to_2_001": {
        "source_required": True,
        "gate": "blocked_cot_revision",
        "issue": "pair 名称含 n1_rattle，但 manifest 的实际 intervention 是 cube_3（foam cube）3to2；用户又指出 rattle 不得混叠。为避免把错误实体当作 intervention，先确认/同步 COT：默认保留 rattle_1=2 作为 must_preserve，3to2 作用于 cube_3。",
        "source_extra": "在 COT 确认后，source 必须同时展示 rattle_1 的2个 baby rattle（彼此不混叠、作为 must_preserve）以及 exactly 3 个可独立辨认的 foam cube（entity_id=cube_3）；两组分离，不能把 rattle 当成 cube 数量。",
        "target_extra": "按 canonical COT 只从 cube_3 的3个 foam cube 中移除恰好1个，得到2个；rattle_1 的2个 baby rattle 位置、数量、大小、材质和清晰度完全不变。若用户意图确实是 rattle 3to2，必须先改写 intervention entity 和 target COT，禁止直接生成。",
        "validation": "COT确认后 source=cube_3 3 + rattle_1 2；target=cube_3 2 + rattle_1 2；不得把 rattle 误删或混叠。",
        "cot_revision": {"entity_id": "cube_3", "field": "semantic_target", "current": "foam cube 3to2", "user_label": "n1_rattle 3to2", "reason": "pair name/user issue 与 manifest intervention entity 不一致，需在生图前确认 canonical target"},
    },
    "g1000_living_room_play__puzzle_fill__3_to_2_001": {
        "source_required": True,
        "issue": "source 必须严格生成三块拼图，再进行数量变换。",
        "source_extra": "source 必须有 exactly 3 块可独立数出的 puzzle pieces，分离摆放、形状边缘清楚，不把一块拼图或一堆碎片误算成3块。",
        "target_extra": "只移除恰好1块已有 puzzle piece，最终保留2块；幸存两块位置和形状细节不变，不改变其他玩具。",
        "validation": "source=3块独立拼图；target=2块独立拼图，非整张puzzle/非3块残影。",
    },
    "g1000_office_desk__stand_keyboard_pad__occlusion_state_002": {
        "source_required": True,
        "issue": "需修改 source，使遮盖关系明确。",
        "source_extra": "source 必须按 COT 显示指定 target 与 reference 的清晰、可判断部分遮挡，约四分之一至三分之一轮廓被遮住；两者都完整可辨，不允许仅接触、左右并排或被第三物体遮挡。",
        "target_extra": "只改变指定 occlusion relation，使用已有 stand/keyboard/pad；目标关系要明显反转或达到 new_value，所有非 intervention 对象位置、大小和细节不变。",
        "validation": "source关系清晰；target关系按COT清晰改变；无对象消失、复制或换成左右位置变化。",
    },
    "g1000_pantry_shelf_closeup__cans_open_crate__containment_001": {
        "source_required": False,
        "issue": "将容器内全部 cans 移出，数量必须与 source 一致。",
        "source_extra": "",
        "target_extra": "把 crate 内全部 COT-listed cans 移到 crate 外同一支撑面，逐个保留且数量与 source 完全一致；crate 原位不变，不能只移出一个并删除其余。",
        "validation": "target中全部cans在crate外、数量等于source；crate位置/结构不变，所有罐体可数。",
    },
    "g1000_pantry_shelf_closeup__cans_open_crate_out__containment_001": {
        "source_required": True,
        "issue": "需修改 source；移入容器后不能看不到 object，容器内外数量必须一致。",
        "source_extra": "source 中全部 COT-listed cans 在 crate 外且数量清晰可数；crate 开口足够大、前后无遮挡，为将全部 cans 放入后仍露出可验证轮廓预留空间。",
        "target_extra": "将全部 source cans 移入同一个 crate，逐个保留、数量完全一致；每个 can 至少部分露出 crate rim 以便审计，crate 位置/形状/数量不变。",
        "validation": "source/target cans数量一致；target全部进入同一crate且仍可数，无罐体消失/复制。",
    },
    "g1000_pantry_shelf_closeup__jars_open_box_out__containment_001": {
        "source_required": True,
        "issue": "需修改 source；移入容器后不能看不到 object，容器内外数量必须一致。",
        "source_extra": "source 中全部 COT-listed jars 在 box 外且数量清晰；open box 开口和深度足以容纳全部 jars，同时保留可审计的露出轮廓。",
        "target_extra": "将全部 jars 移入同一个 open box，数量与 source 完全一致，每个 jar 保留可见上半部/轮廓；box 固定不动，不删除或新增 jar。",
        "validation": "source/target jars数量相同；target全部在同一box内且每个仍可辨，无隐藏额外实例。",
    },
    "g1000_toolshed_garage__hammer_oilcan_basket__functional_state_002": {
        "source_required": True,
        "gate": "blocked_cot_revision",
        "issue": "oil can 的 open/closed 状态不符合本轮生活实际；需要更换为明确具有开合状态的 object 并重设计 prompt。",
        "source_extra": "先修订 source/target COT：将 oilcan_1 统一改为 compact hinged metal toolbox/tool case（或同等具有明确 open/closed 状态的 garage object），保持 count=1、location=center_region；修订后 source 显示 lid open，target 只改为 lid closed。未完成 COT 修订前禁止生图。",
        "target_extra": "COT 修订通过后，target 只改变新 object 的 functional_state open->closed；category/material/location/count 与其他 hammer/basket/background 保持，禁止继续用 oil can prompt 冒充 toolbox。",
        "validation": "COT/source/target 三处 object 定义一致；open/closed 由同一铰链盖清晰表达；未修订前不进入生成队列。",
        "cot_revision": {"entity_id": "oilcan_1", "from_category": "oil can", "proposed_category": "compact hinged metal toolbox/tool case", "reason": "open/closed state must be visually and functionally unambiguous"},
    },
    "g1000_warehouse_shelf__bag_quota__3_to_2_001": {
        "source_required": True,
        "issue": "source 原数量不是3个，需要重新生成 source；manifest 的实际 intervention 是 can_2（paint can），pair 名称中的 bag 是命名别名。",
        "source_extra": "按 manifest canonical entity can_2 生成 exactly 3 个 individual paint can，彼此可数、白色金属材质一致但轮廓不混叠；bag_1 作为 must_preserve 的2个 paper bag 保持，不得把 paint can 误生成成 bag。",
        "target_extra": "只从同一 can_2 组移除恰好1个 paint can，最终2个；两个幸存 paint can 位置、尺寸和细节保持，不移动货架上其他对象或 bag_1。",
        "validation": "source=3个清晰paint can；target=2个paint can；数量变化来自单个实例删除而非重画整个区域；pair 命名与实体类别差异已记录。",
        "semantic_note": "pair_id 含 bag_quota，但 intervention.entity_id=can_2、category=paint can；本计划以 manifest COT 为 canonical，不静默改成 bag。",
    },
    "g1000_warehouse_shelf__bottles_shippingcrate__containment_001": {
        "source_required": False,
        "issue": "把 bottles 全部移出来，但 shipping crate 的位置不能改变。",
        "source_extra": "",
        "target_extra": "将 crate 内全部 COT-listed bottles 移到 crate 外，数量和每个 bottle 的身份/材质保持；shipping crate 原位、大小、朝向和结构不变，不能只移出一个或移动 crate。",
        "validation": "target全部bottles在crate外且数量一致；crate像素语义位置不变，瓶子无丢失/复制。",
    },
    "g1000_warehouse_shelf__box_sack_pallet__2_to_1_001": {
        "source_required": True,
        "issue": "货架下层两个 pallet 叠在一起导致模型无法执行2to1，需要更容易分辨的 source。",
        "source_extra": "source old count=2 的 pallet 必须分处两个清晰、非重叠的支撑位置/层位；不得上下完全重合或被 box/sack遮挡；保留一个 pallet 删除后的空间。",
        "target_extra": "只移除恰好1个 pallet，另一个 pallet 原位保留；box、sack、货架和其他层位不动，不把2to1做成缩小或整体重排。",
        "validation": "source=2个可分辨pallet；target=1个幸存pallet，其他对象位置不变。",
    },
    "g1000_warehouse_shelf__box_tote_drum__2_to_1_001": {
        "source_required": True,
        "issue": "source 有两行一样的东西；需同一 object 组内放在一起并留出空间。",
        "source_extra": "source 只展示一个紧凑、连续的 target group，exactly 2 个 individual target objects 同组相邻、不是两行重复陈列；留出移除一个后的空位，box/tote/drum 不形成同类干扰。",
        "target_extra": "从同一连续组移除恰好1个 target object，剩余者原位保留；禁止移动到另一行、生成新组、改变无关 box/tote/drum。",
        "validation": "source=单组2个且无重复行；target=同组1个，其他对象不动。",
    },
    "g1000_warehouse_shelf__drum_pallet_tote__2_to_3_001": {
        "source_required": True,
        "issue": "2to3 时 object 被改变大小；应在原 pallet 上方增加一个 pallet。",
        "source_extra": "source old count=2 的 target object 在同一 pallet 区域清晰可数，并在原 pallet 上方预留一个同尺寸实例的垂直空间；禁止通过缩放现有对象制造空间。",
        "target_extra": "只在原 pallet/target group 的正上方增加恰好1个同尺寸 individual pallet/object，最终3个；两个原实例尺寸、位置、纹理不变，其他 tote/drum不动。",
        "validation": "target=3个同尺寸实例，新增者在原pallet上方；无缩放、无第二区域、无无关对象位移。",
    },
    "g1000_warehouse_shelf__helmets_gearbin_handtruck__2_to_3_001": {
        "source_required": True,
        "issue": "需要让 source 的箱/gear bin 更大，为第三个 helmet 留出空间。",
        "source_extra": "在不改变 COT 类别和数量的前提下使用较大的 open plastic gear bin/箱体作为背景容器，source 中 exactly 2 个 helmet 清晰可辨，并为第三个 helmet 预留完整空间；handtruck 不遮挡目标区域。",
        "target_extra": "在同一 helmet 组/gear-bin 邻近区域增加恰好1个 helmet，最终3个；不得缩放原 helmet、移动 handtruck 或改变 gear bin 以外的对象。",
        "validation": "source=2个清晰helmet且箱体有空间；target=3个同尺寸helmet，handtruck/gearbin位置与属性保持。",
    },
    "g1000_warehouse_shelf__helmets_gearbin_handtruck__containment_001": {
        "source_required": True,
        "issue": "source 中两个帽子，变换后外面只有一个；需要全部移出来且外面仍有两个。",
        "source_extra": "source 必须展示 exactly 2 个 helmet 在 gearbin 内，两个都至少露出足够轮廓可数；gearbin 开口足够且目标外部区域留空。",
        "target_extra": "把 source 中两个 helmet 全部移出同一个 gearbin，两个都在外部且数量仍为2；gearbin 原位不动，不可只移出一个、删除另一个或复制第三个。",
        "validation": "source/target均2个helmet；target两个都在bin外且可数，gearbin位置不变。",
    },
    "g1000_warehouse_shelf__sack_bundle_bin__2_to_1_001": {
        "source_required": False,
        "issue": "修改数量的同时不得移动位置。",
        "source_extra": "",
        "target_extra": "只从同一 sack/bundle/bin 场景中移除恰好1个 intervention instance；幸存者和所有无关对象位置、尺寸、方向完全不变。",
        "validation": "target数量符合new_value；幸存者和bundle/bin位置不变，无重排。",
    },
    "g1000_warehouse_shelf__sack_bundle_bin__2_to_3_001": {
        "source_required": True,
        "issue": "原位置不够放第三个 sack，需要重新设计 source。",
        "source_extra": "source old count=2 的 sack 必须在同一连续目标区域清晰可数，并在原区域旁/上方留出一个同尺寸 sack 的净空；不得出现第二行、第二堆或挤压 bundle/bin。",
        "target_extra": "只在原 sack group 内增加恰好1个同尺寸 sack，最终3个；两个原 sack、bundle、bin均保持原位和细节，不把第三个放到新区域。",
        "validation": "source=同组2个且有净空；target=同组3个同尺寸sack，无第二组/缩放/无关移动。",
    },
}


def main() -> None:
    original = load_jsonl(DATA / "gold1000.generated.jsonl")
    v1 = load_jsonl(V1_ROOT / "manifest.jsonl")
    original_by_id = {str(row["pair_id"]): row for row in original}
    v1_by_id = {str(row["pair_id"]): row for row in v1}
    original_line = {str(row["pair_id"]): i for i, row in enumerate(original, 1)}
    if set(SPECS) - set(v1_by_id):
        raise RuntimeError(f"second-round IDs absent from v1: {sorted(set(SPECS) - set(v1_by_id))}")

    scene_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in original:
        scene_members[str(row["source_scene_id"])].append(row)

    signatures: dict[str, str] = {pid: cot_signature(row) for pid, row in original_by_id.items()}
    explicit_source_keys = {
        (str(original_by_id[pid]["source_scene_id"]), signatures[pid])
        for pid, spec in SPECS.items()
        if spec.get("source_required")
    }

    direct: dict[str, dict[str, Any]] = {}
    for pid, spec in SPECS.items():
        row = original_by_id[pid]
        scene_id = str(row["source_scene_id"])
        sig = signatures[pid]
        promoted = (scene_id, sig) in explicit_source_keys
        effective_source = bool(spec.get("source_required") or promoted)
        reason = "用户明确要求 source 重生成" if spec.get("source_required") else "round-2 target-only"
        if promoted and not spec.get("source_required"):
            reason = "同一 source COT variant 被其他问题要求重生成，必须同步 source+target"
        direct[pid] = {
            "pair_id": pid,
            "source_scene_id": scene_id,
            "source_cot_sha256": sig,
            "spec": spec,
            "effective_source_required": effective_source,
            "scope_reason": reason,
        }

    variants: dict[tuple[str, str], dict[str, Any]] = {}
    for pid, item in direct.items():
        if not item["effective_source_required"]:
            continue
        key = (item["source_scene_id"], item["source_cot_sha256"])
        variant = variants.setdefault(
            key,
            {
                "source_scene_id": key[0],
                "source_cot_sha256": key[1],
                "source_variant_key": f"{key[0]}__state_{key[1][:12]}",
                "direct_pair_ids": [],
                "source_measures": [],
                "blocked_gates": [],
            },
        )
        variant["direct_pair_ids"].append(pid)
        source_measure = generic_source(original_by_id[pid]) + str(item["spec"].get("source_extra", ""))
        variant["source_measures"].append(source_measure)
        if item["spec"].get("gate"):
            variant["blocked_gates"].append(item["spec"]["gate"])

    for variant in variants.values():
        variant["source_measure"] = "\n\n".join(dict.fromkeys(variant["source_measures"]))
        variant["generation_gate"] = (
            "blocked_cot_revision" if variant["blocked_gates"] else "ready"
        )
        variant["variant_member_pair_ids"] = [
            str(row["pair_id"])
            for row in scene_members[variant["source_scene_id"]]
            if signatures[str(row["pair_id"])] == variant["source_cot_sha256"]
        ]

    selected: dict[str, dict[str, Any]] = {}
    for pid, item in direct.items():
        if item["effective_source_required"]:
            key = (item["source_scene_id"], item["source_cot_sha256"])
            variant = variants[key]
            for member_pid in variant["variant_member_pair_ids"]:
                selected.setdefault(
                    member_pid,
                    {
                        "pair_id": member_pid,
                        "source_variant_key": variant["source_variant_key"],
                        "source_cot_sha256": variant["source_cot_sha256"],
                        "source_scene_id": variant["source_scene_id"],
                        "spillover_from": list(variant["direct_pair_ids"]),
                        "variant_gate": variant["generation_gate"],
                    },
                )
        else:
            selected.setdefault(
                pid,
                {
                    "pair_id": pid,
                    "source_variant_key": None,
                    "source_cot_sha256": signatures[pid],
                    "source_scene_id": str(original_by_id[pid]["source_scene_id"]),
                    "spillover_from": [],
                    "variant_gate": "ready",
                },
            )

    selected_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    for pid in sorted(selected):
        row = original_by_id[pid]
        direct_item = direct.get(pid)
        selected_item = selected[pid]
        is_direct = direct_item is not None
        if is_direct:
            spec = direct_item["spec"]
            effective_source = direct_item["effective_source_required"]
            source_measure = generic_source(row) + str(spec.get("source_extra", "")) if effective_source else "本轮不重生成 source；使用当前 repair_batch_20240924_v1 的已生成 source 作为 target edit 输入，不回退到原始 source。"
            target_measure = generic_target(row) + str(spec.get("target_extra", ""))
            issue = spec["issue"]
            variant_gate = (
                variants[(direct_item["source_scene_id"], direct_item["source_cot_sha256"])]
                ["generation_gate"]
                if effective_source
                else "ready"
            )
            gate = variant_gate if variant_gate != "ready" else spec.get("gate", "ready")
            role = "reported_problem"
            scope = "source+target" if effective_source else "target-only"
            scope_reason = direct_item["scope_reason"]
            cot_revision = spec.get("cot_revision")
        else:
            variant = variants[(selected_item["source_scene_id"], selected_item["source_cot_sha256"])]
            source_measure = variant["source_measure"]
            target_measure = generic_target(row) + "这是同一 source COT variant 的连带 target：使用本 variant 新 source，严格执行该 pair 原始 intervention，不引入额外 edit。"
            issue = "用户未列出，默认当前结果合理；因共享 source COT variant 重生成，必须同步重做 target。"
            gate = selected_item["variant_gate"]
            role = "source_scene_spillover"
            scope = "source+target"
            scope_reason = "共享 source COT variant 同步"
            cot_revision = None
            effective_source = True
        old_source = V1_ROOT / "source" / f"{pid}__source.png"
        plan_row = {
            "plan_schema_version": "counterfactual_repair_round2_v1",
            "audit_round": 2,
            "pair_id": pid,
            "source_scene_id": str(row["source_scene_id"]),
            "source_cot_sha256": signatures[pid],
            "source_variant_key": selected_item["source_variant_key"],
            "source_variant_gate": gate,
            "source_variant_member_pair_ids": (
                variants[(selected_item["source_scene_id"], selected_item["source_cot_sha256"])]["variant_member_pair_ids"]
                if selected_item["source_variant_key"]
                else [pid]
            ),
            "source_variant_direct_problem_pair_ids": (
                variants[(selected_item["source_scene_id"], selected_item["source_cot_sha256"])]["direct_pair_ids"]
                if selected_item["source_variant_key"]
                else [pid]
            ),
            "repair_scope": scope,
            "repair_role": role,
            "scope_reason": scope_reason,
            "generation_gate": gate,
            "approval_status": "用户第二轮审计已记录",
            "generation_status": "blocked_cot_revision" if gate == "blocked_cot_revision" else "待生成",
            "merge_status": "未融合",
            "review_status_round2": "problem_under_repair" if is_direct else "default_reasonable_but_sync_required",
            "user_issue": issue,
            "intervention": row.get("intervention"),
            "subtype": row.get("subtype"),
            "source_entities": row.get("source", {}).get("cot", {}).get("entities", []),
            "target_entities": row.get("target", {}).get("cot", {}).get("entities", []),
            "source_cot_text": row.get("source", {}).get("cot_text"),
            "target_cot_text": row.get("target", {}).get("cot_text"),
            "original_target_prompt": row.get("target", {}).get("prompt"),
            "source_measure": source_measure,
            "target_measure": target_measure,
            "validation_measure": spec.get("validation", "共享 source variant 与 target 均须重新人工核对数量、位置、关系和 must_preserve。") if is_direct else "source变更后重新核对该pair原始target intervention、所有must_preserve实体和mask局部性。",
            "preservation_measure": "除 intervention 指定的单一 field/relation/count 外，不改变任何实体、数量、位置、尺寸、方向、材质、背景、镜头或容器；count 按 individual instances 验收。",
            "cot_revision": cot_revision,
            "base_source_policy": "round2_source_variant_regeneration" if effective_source else "prior_round_v1_source",
            "base_source_path": str(old_source) if old_source.is_file() else None,
            "source_regeneration_required": effective_source,
            "target_regeneration_required": True,
            "candidate_for_replacement": False,
            "replacement_status": "blocked_cot_revision" if gate == "blocked_cot_revision" else "blocked_until_round2_audit",
            "fusion_status": "not_ready",
        }
        selected_rows.append(plan_row)
        if gate != "blocked_cot_revision":
            generation_rows.append(plan_row)

    direct_rows = []
    for pid in sorted(SPECS):
        direct_rows.append(next(row for row in selected_rows if row["pair_id"] == pid))
    write_jsonl(ROUND2_PLAN, direct_rows)
    write_jsonl(ROUND2_GENERATION_PLAN, generation_rows)

    scene_plan_rows = []
    for key, variant in sorted(variants.items()):
        scene_plan_rows.append(
            {
                "source_scene_id": variant["source_scene_id"],
                "source_variant_key": variant["source_variant_key"],
                "source_cot_sha256": variant["source_cot_sha256"],
                "generation_gate": variant["generation_gate"],
                "direct_problem_pair_ids": sorted(variant["direct_pair_ids"]),
                "variant_member_pair_ids": sorted(variant["variant_member_pair_ids"]),
                "variant_member_count": len(variant["variant_member_pair_ids"]),
                "source_measure": variant["source_measure"],
            }
        )
    scene_doc = {
        "schema_version": "counterfactual_scene_repair_round2_v1",
        "round": 2,
        "original_manifest": str(DATA / "gold1000.generated.jsonl"),
        "previous_batch": str(V1_ROOT),
        "reported_pair_count": len(SPECS),
        "reported_source_required_pair_count": sum(bool(spec.get("source_required")) for spec in SPECS.values()),
        "source_variant_count": len(scene_plan_rows),
        "source_scene_count": len({row["source_scene_id"] for row in scene_plan_rows}),
        "selected_target_count_including_variant_spillover": len(selected_rows),
        "generation_target_count_excluding_cot_blocked": len(generation_rows),
        "spillover_count": sum(row["repair_role"] == "source_scene_spillover" for row in selected_rows),
        "cot_revision_blocked_pair_count": sum(row["generation_gate"] == "blocked_cot_revision" for row in selected_rows),
        "count_conflict_policy": "按 source COT canonical signature 拆分 source variant；同一 source_scene_id 但 old count/COT 不同的 pair 不共用一张 source。",
        "source_variant_rows": scene_plan_rows,
    }
    ROUND2_SCENE_PLAN.write_text(json.dumps(scene_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Record the complete second-round decision for every pair in the previous 81-pair batch.
    current_ids = set(v1_by_id)
    selected_ids = set(selected)
    tracking_rows: list[dict[str, Any]] = []
    for pid in sorted(current_ids | selected_ids):
        row = original_by_id[pid]
        direct_item = direct.get(pid)
        selected_item = selected.get(pid)
        if direct_item:
            plan = next(x for x in selected_rows if x["pair_id"] == pid)
            classification = "reported_problem_round2"
            review_status = "problem_under_repair_round2"
            repair_status = "planned_source_target_repair" if plan["repair_scope"] == "source+target" else "planned_target_repair"
            replacement_status = plan["replacement_status"]
            issue = direct_item["spec"]["issue"]
        elif selected_item:
            plan = next(x for x in selected_rows if x["pair_id"] == pid)
            classification = "source_scene_spillover_round2"
            review_status = "default_reasonable_but_sync_required"
            repair_status = "planned_source_scene_sync"
            replacement_status = "blocked_until_round2_audit"
            issue = "未列出，默认当前结果合理；因共享 source COT variant 被重生成，需同步重做。"
        else:
            plan = None
            classification = "default_reasonable_round2"
            review_status = "no_problem"
            repair_status = "not_required_round2"
            replacement_status = "pending_fusion_integrity_check"
            issue = "用户未列出，按用户指令默认合理；保留现有 v1 资产，未融合。"
        tracking_rows.append(
            {
                "schema_version": "repair_round2_tracking_v1",
                "tracking_created_date": "2026-09-24",
                "audit_round": 2,
                "pair_id": pid,
                "source_scene_id": str(row["source_scene_id"]),
                "in_previous_batch_v1": pid in current_ids,
                "in_round2_selected_plan": pid in selected_ids,
                "classification": classification,
                "review_status": review_status,
                "repair_status": repair_status,
                "user_issue": issue,
                "repair_scope": plan["repair_scope"] if plan else "not_required",
                "repair_role": plan["repair_role"] if plan else "not_selected",
                "source_variant_key": plan["source_variant_key"] if plan else None,
                "source_variant_gate": plan["source_variant_gate"] if plan else None,
                "source_cot_sha256": signatures[pid],
                "base_source_policy": plan["base_source_policy"] if plan else "retain_prior_round_v1_source",
                "base_source_path": str(V1_ROOT / "source" / f"{pid}__source.png") if pid in v1_by_id else None,
                "source_regeneration_required": bool(plan and plan["source_regeneration_required"]),
                "target_regeneration_required": bool(plan),
                "generation_status": plan["generation_status"] if plan else "not_required_round2",
                "replacement_status": replacement_status,
                "fusion_status": "not_ready",
                "candidate_for_replacement": False,
                "notes": (
                    "未列出，默认合理；但共享 source variant 改变后必须完成同步 target 和人工审计。"
                    if classification == "source_scene_spillover_round2"
                    else "未列出，默认合理；保留当前 v1 结果，后续仍需执行融合完整性检查。"
                    if classification == "default_reasonable_round2"
                    else "列出问题，按第二轮 source/target 措施重新优化；完成前不得替换1000-pair。"
                ),
            }
        )
    write_jsonl(ROUND2_TRACKING, tracking_rows)

    # Preserve the old cumulative tracking file and add round-2 fields to its v1 rows.
    if BASE_TRACKING.is_file():
        base_rows = load_jsonl(BASE_TRACKING)
        round2_by_id = {row["pair_id"]: row for row in tracking_rows}
        cumulative = []
        for row in base_rows:
            new_row = dict(row)
            if row.get("batch_id") == "repair_batch_20240924_v1" and row.get("pair_id") in round2_by_id:
                r2 = round2_by_id[row["pair_id"]]
                new_row["audit_round_2_classification"] = r2["classification"]
                new_row["audit_round_2_review_status"] = r2["review_status"]
                new_row["audit_round_2_repair_status"] = r2["repair_status"]
                new_row["audit_round_2_source_variant_key"] = r2["source_variant_key"]
                new_row["audit_round_2_generation_status"] = r2["generation_status"]
                new_row["audit_round_2_replacement_status"] = r2["replacement_status"]
                new_row["audit_round_2_fusion_status"] = r2["fusion_status"]
                new_row["audit_round_2_issue"] = r2["user_issue"]
            cumulative.append(new_row)
        write_jsonl(CUMULATIVE_TRACKING, cumulative)

    current_unlisted = sorted(set(v1_by_id) - set(SPECS))
    summary = {
        "schema_version": "repair_round2_summary_v1",
        "tracking_created_date": "2026-09-24",
        "previous_batch": str(V1_ROOT),
        "original_manifest": str(DATA / "gold1000.generated.jsonl"),
        "round2_direct_problem_count": len(SPECS),
        "round2_direct_source_required_count": sum(bool(spec.get("source_required")) for spec in SPECS.values()),
        "round2_direct_target_only_count": sum(not bool(spec.get("source_required")) for spec in SPECS.values()),
        "round2_direct_cot_revision_blocked_count": sum(spec.get("gate") == "blocked_cot_revision" for spec in SPECS.values()),
        "round2_direct_ready_count": sum(spec.get("gate") != "blocked_cot_revision" for spec in SPECS.values()),
        "round2_effective_cot_revision_blocked_selected_count": sum(
            row["generation_gate"] == "blocked_cot_revision" for row in selected_rows
        ),
        "previous_v1_pair_count": len(v1_by_id),
        "previous_v1_unlisted_default_reasonable_count": len(current_unlisted),
        "source_scene_count": len({row["source_scene_id"] for row in scene_plan_rows}),
        "source_variant_count": len(scene_plan_rows),
        "selected_target_count_including_variant_spillover": len(selected_rows),
        "generation_target_count_excluding_cot_blocked": len(generation_rows),
        "source_scene_spillover_count": sum(row["repair_role"] == "source_scene_spillover" for row in selected_rows),
        "source_variant_count_conflict_policy": "同一 source_scene_id 下 source COT canonical signature 不同（尤其 old count=2 与 old count=3）时拆成不同 source_variant，不共用 source。",
        "historical_record_as_user_provided": {
            "repair_batch_20260923_v3": {
                "record_count": 261,
                "review_status": "no_problem",
                "target_generated_count": 261,
                "mask_generated_count": 261,
                "replacement_candidate_count": 187,
                "replacement_status": "pending",
                "held_by_current_batch_overlap_count": 74,
            },
            "repair_batch_20240924_v1": {
                "record_count": 81,
                "review_status": "problem_under_repair",
                "reported_problem_count": 65,
                "spillover_count": 16,
                "generation_snapshot": {
                    "source_success": 21,
                    "source_total": 47,
                    "target_pending": 81,
                },
                "replacement_status": "blocked_until_repair_complete",
            },
        },
        "current_verified_v1_status": {
            "source_success": 47,
            "source_total": 47,
            "target_success": 81,
            "failures": 0,
            "replacement_status": "blocked_until_manual_audit",
        },
        "files": {
            "direct_plan": str(ROUND2_PLAN),
            "generation_plan": str(ROUND2_GENERATION_PLAN),
            "scene_plan": str(ROUND2_SCENE_PLAN),
            "round2_tracking": str(ROUND2_TRACKING),
            "cumulative_tracking": str(CUMULATIVE_TRACKING) if CUMULATIVE_TRACKING.exists() else None,
        },
        "status_policy": {
            "listed": "problem_under_repair_round2; blocked_until_round2_audit",
            "unlisted": "no_problem or default_reasonable_but_sync_required; never auto-fused",
            "cot_revision": "blocked_cot_revision; do not send to image generation before COT/source/target definitions are synchronized",
        },
    }
    ROUND2_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
