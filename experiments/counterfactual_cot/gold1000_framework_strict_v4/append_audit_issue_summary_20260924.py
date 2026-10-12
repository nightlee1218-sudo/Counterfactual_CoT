#!/usr/bin/env python3
"""Append the 2026-09-24 repair-batch audit report to a new XLSX.

This is a reporting-only step. It does not call APIs, generate images, alter
any manifest, delete pairs, or alter the previous XLSX. It keeps every raw
report line, resolves exact/fuzzy manifest IDs conservatively, and marks rows
with explicit source wording for later repair planning.
"""
from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

RAW_REPORT = r"""
g1000_bathroom_counter__rinsecup_soap_towel__occlusion_state_004	需重生成source，source没有遮盖关系
g1000_bedroom_nightstand__book_remote_vase__absolute_vertical_001	需修改source，现在的变化和没优化之前一样，下面的遥控器会妨碍书的下移
g1000_bedroom_nightstand__glasses_charger_glass__occlusion_state_008	需修改source，原source有些遮盖
g1000_bedroom_nightstand__glasses_charger_glass__occlusion_state_010	需修改source，原source没有遮盖
g1000_bedroom_nightstand__lamp_clock_book__functional_state_006	需修改source，原source书本没有打开
g1000_bedroom_nightstand__lamp_glass_tray__absolute_vertical_001	需修改source，桌篼空间不足放置台灯
g1000_bedroom_nightstand__lamp_glass_tray__absolute_vertical_004	需修改source，桌篼空间不足放置台灯
g1000_bookshelf_study__mugcollection_folder__absolute_vertical_001	文件夹还是只进行了旋转，没有发生vertical变换，source应留出足够空间
g1000_bookshelf_study__penjar_folder_trophy__absolute_vertical_001	从right region变成upper right不合适；原本就在upper right，需修改prompt
g1000_classroom__chair_backpack_notebook__1_to_2_002	Jacket变成两个的同时把书遮盖住；需要在同一摞上增加一个即可
g1000_classroom__n3_chair__3_to_2_002	3to2变化失败，重新generate
g1000_classroom__n4_ruler__2_to_3_001	Source有两个交叉在一起的尺子，生成后变成了double两个交叉尺子，重新生成source
g1000_closet_wardrobe__beanie_hatbox_scarf__1_to_2_002	Source有歧义：一处一摞两个围巾，另一处一摞一个，无法确定增加哪个，重新生成source
g1000_closet_wardrobe__jeans_belt__occlusion_state_001	需修改source，原source有些遮盖
g1000_closet_wardrobe__jeans_belt__occlusion_state_002	需修改source，原source遮盖关系不明确
g1000_closet_wardrobe__jeans_belt__occlusion_state_004	需修改source，原source遮盖关系不明确
g1000_closet_wardrobe__n2_sweater__2_to_3_002	一沓两个变成两沓四个而非一沓三个，重新生成
g1000_closet_wardrobe__socks_basket_pajamas_hat__occlusion_state_001	需修改source，原source遮盖关系不明确
g1000_closet_wardrobe__tshirts_jewelrybox__3_to_2_001	需修改source，不要生成两沓
g1000_dining_table__fruit_bowl_napkins__containment_001	Source容器里有三个苹果，target容器里还有三个，外面有一个，重新生成
g1000_dining_table__napkin_fill__3_to_2_001	变成一个，可能source的三个不明显，重新生成source
g1000_dining_table__plate_quota__2_to_3_001	变成四个，重新生成source
g1000_dining_table__plate_quota__3_to_2_001	source有四个杯子，重新生成source
g1000_flower_shop__peony_crate_pot__1_to_2_004	Source仍有一个无关花瓶；生成时crate多了两个并遮盖花瓶，需重新设计source/prompt
g1000_flower_shop__tulip_pot_ribbon__containment_001	需修改source，容器关系不正确
g1000_grocery_display__biscuit_tea_honey__2_to_3_001	Source瓶罐太多，模型搞混目标；两个变成四个且挤动其他物体，需重新设计source并留空间
g1000_grocery_display__biscuit_tea_honey__3_to_2_001	变换失败；source太复杂、空间不足，重新设计source
g1000_grocery_display__cans_bin_jars__containment_001	Source里面有三个，target里面仍三个外面多一个，重新生成source
g1000_grocery_display__flowers_vase_plants__absolute_horizontal_001	左边两个植物移到右边超出范围只剩一个，左边两个也没消除，需留移动空间
g1000_grocery_display__gd04b_flowers_vase_separate__containment_001	Source外面花太多，重新生成source，只要一朵
g1000_grocery_display__gd05b_milk_crate_separate__containment_001	容器内多生成一个，外面仍三个；修改source只保留一个
g1000_grocery_display__honey_biscuit_water__2_to_3_00	两个water变三个且source数量不对，蓝色管子四个，重新生成source优化
g1000_grocery_display__honey_biscuit_water__3_to_2_001	source不对，重新生成source优化
g1000_kitchen_counter__bowl_block_mill__occlusion_state_005	Source遮盖关系不正确，重新生成source
g1000_kitchen_counter__toaster_dish_jar__occlusion_state_004	Source遮盖关系不正确，重新生成source
g1000_living_room_play__ball_quota__2_to_3_001	两个变成四个，需重新生成
g1000_living_room_play__n1_rattle__2_to_3_001	Source有一模一样的两行物品；3to2可能按总数4个处理，重新设计source，变换失败
g1000_living_room_play__puzzle_fill__3_to_2_001	需修改source，source数量不正确
g1000_office_desk__stand_keyboard_pad__occlusion_state_002	需修改source，遮盖关系不明确
g1000_office_desk__stand_keyboard_pad__occlusion_state_001	需修改source，遮盖关系不明确
g1000_office_desk__tray_calendar_dispenser__1_to_2_002	变换同时改变物体大小，可能桌面空间不足，重新生成source
g1000_pantry_shelf_closeup__cans_open_crate__containment_001	容器内四个，变换后只移出一个；需确认can_1是否代表第一个can，四个can是否为can_1至can_4
g1000_pantry_shelf_closeup__cans_open_crate_out__containment_001	Source容器外四个，变换后外面仍四个、容器内一个，变化不正确
g1000_pantry_shelf_closeup__jars_open_box_out__containment_001	玻璃瓶放入箱子会看不见，重新设计pair
g1000_parking_lot__suv_trolley_pole__shape_divergent_replace_003	原车子未被消除，重新生成
g1000_parking_lot__van_barrier_sign__shape_divergent_replace_002	原车子未被消除，重新生成
g1000_playground__carousel_tunnel_bin__absolute_horizontal_003	移动carousel同时把凳子往右移；其他object必须原地不变
g1000_toolshed_garage__hammer_oilcan_basket__functional_state_002	开关变化不明显，目标物体似乎不适合开关状态，重新设计prompt
g1000_warehouse_shelf__bag_quota__3_to_2_001	数量变化同时带来位置变化，其余物体必须原地不动
g1000_warehouse_shelf__bottles_shippingcrate__containment_001	容器样子发生变化，容器必须保持不变
g1000_warehouse_shelf__box_sack_pallet__2_to_1_001	Source物体太多太杂、没有空间，重新简化source
g1000_warehouse_shelf__box_sack_pallet__2_to_3_001	Source物体太多太杂、没有空间，重新简化source
g1000_warehouse_shelf__box_sack_pallet__3_to_2_001	Source物体太多太杂、没有空间，重新简化source
g1000_warehouse_shelf__box_tote_drum__2_to_1_001	仍有两行一样的东西，重新设计source
g1000_warehouse_shelf__box_tote_drum__2_to_3_001	仍有两行一样的东西，重新设计source
g1000_warehouse_shelf__box_tote_drum__3_to_2_001	仍有两行一样的东西，重新设计source
g1000_warehouse_shelf__drum_pallet_tote__2_to_3_001	Source物体太多太杂、空间不足，重新简化source
g1000_warehouse_shelf__helmets_gearbin_handtruck__2_to_3_001	source箱子变大，重新设计source
g1000_warehouse_shelf__helmets_gearbin_handtruck__containment_001	Source里面两个帽子，变换后外面只有一个
g1000_warehouse_shelf__paint_tool_pallet__2_to_3_001	数量失败、空间不足、物体太杂，重新简化source
g1000_warehouse_shelf__paint_tool_pallet__3_to_2_001	数量失败、空间不足、物体太杂，重新简化source
g1000_warehouse_shelf__sack_bundle_bin__2_to_1_001	数量失败、空间不足、物体太杂，重新简化source
g1000_warehouse_shelf__sack_bundle_bin__2_to_3_001	数量失败、空间不足、物体太杂，重新简化source
g1000_warehouse_shelf__sack_bundle_bin__3_to_2_001	数量失败、空间不足、物体太杂，重新简化source
g1000_warehouse_shelf__tool_bin_paint__2_to_3_001	数量变化失败，重新生成
g1000_warehouse_shelf__box_sack_pallet__2_to_1_001	Source物体太多太杂、没有空间，重新简化source
g1000_warehouse_shelf__box_sack_pallet__2_to_3_001	Source物体太多太杂、没有空间，重新简化source
g1000_warehouse_shelf__box_sack_pallet__3_to_2_001	Source物体太多太杂、没有空间，重新简化source
"""

SOURCE_MARKERS = ("source", "原source", "重新设计source", "重新生成source", "重新设计 source", "重新生成 source")

def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    return {r["pair_id"]: r for r in (json.loads(l) for l in path.open(encoding="utf-8") if l.strip())}

def parse_rows() -> list[tuple[str, str]]:
    result=[]
    for n,line in enumerate(RAW_REPORT.splitlines(),1):
        line=line.strip()
        if not line: continue
        pid,issue=line.split("\t",1)
        result.append((pid.strip(),issue.strip()))
    return result

def resolve(pid: str, manifest: dict[str, dict[str, Any]]) -> tuple[str,str]:
    if pid in manifest: return pid,"exact"
    m=__import__('difflib').get_close_matches(pid,list(manifest),n=1,cutoff=.72)
    return (m[0],"fuzzy_candidate_do_not_auto_correct") if m else ("","not_found")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True,type=Path); ap.add_argument('--output',required=True,type=Path); args=ap.parse_args()
    manifest=load_manifest(args.manifest); raw=parse_rows(); rows=[]
    for i,(raw_id,issue) in enumerate(raw,1):
        pid,status=resolve(raw_id,manifest); r=manifest.get(pid,{})
        iv=r.get('intervention',{}); explicit=any(x.lower() in issue.lower() for x in SOURCE_MARKERS)
        rows.append([i,raw_id,pid,status,r.get('source_scene_id',''),r.get('subtype',''),iv.get('field',''),iv.get('old_value',''),iv.get('new_value',''),issue,'是' if explicit else '否','','待分析','待生成','未融合'])
    wb=Workbook(); ws=wb.active; ws.title='新增问题汇总'
    headers=['序号','pair_id_raw','pair_id_manifest','ID匹配状态','source_scene_id','subtype','intervention_field','old_value','new_value','本次审计问题','用户明确source重生成','建议修补范围','修改措施（第二步）','生成状态','融合状态']; ws.append(headers)
    for row in rows: ws.append(row)
    ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
    for c in ws[1]: c.font=Font(bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='1F4E78'); c.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
    for rr in ws.iter_rows(min_row=2):
        for c in rr: c.alignment=Alignment(vertical='top',wrap_text=True)
    widths=[8,58,58,28,42,24,30,16,16,80,18,18,80,16,16]
    for i,w in enumerate(widths,1): ws.column_dimensions[get_column_letter(i)].width=w
    tab=Table(displayName='NewAuditIssues',ref=f'A1:O{ws.max_row}'); tab.tableStyleInfo=TableStyleInfo(name='TableStyleMedium2',showRowStripes=True,showColumnStripes=False); ws.add_table(tab)
    st=wb.create_sheet('统计'); st.append(['统计项','数量']); st.append(['原始条目',len(raw)]); st.append(['唯一原始ID',len({x[0] for x in raw})]); st.append(['精确匹配',sum(x[3]=='exact' for x in rows)]); st.append(['模糊候选',sum(x[3]=='fuzzy_candidate_do_not_auto_correct' for x in rows)]); st.append(['未找到',sum(x[3]=='not_found' for x in rows)]); st.append(['用户明确source',sum(x[10]=='是' for x in rows)])
    for c in st[1]: c.font=Font(bold=True,color='FFFFFF'); c.fill=PatternFill('solid',fgColor='1F4E78')
    st.column_dimensions['A'].width=30; st.column_dimensions['B'].width=15
    args.output.parent.mkdir(parents=True,exist_ok=True); wb.save(args.output); print(json.dumps({'output':str(args.output),'raw_entries':len(raw),'unique_raw_pair_ids':len({x[0] for x in raw}),'exact':sum(x[3]=='exact' for x in rows),'fuzzy':sum(x[3]=='fuzzy_candidate_do_not_auto_correct' for x in rows),'not_found':sum(x[3]=='not_found' for x in rows)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
