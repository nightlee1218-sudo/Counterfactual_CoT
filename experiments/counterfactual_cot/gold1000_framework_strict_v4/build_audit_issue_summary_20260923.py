#!/usr/bin/env python3
"""Create the first-step XLSX summary of the 2026-09-23 audit report.

This step only records the user's reported pair IDs and observed problems. It
intentionally does not propose prompts, regenerate images, delete pairs, or
modify the dataset/review CSV. Source flags are copied conservatively from the
user's explicit source-related wording; other rows remain target-only for now.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

RAW_REPORT = r"""
g1000_dining_table__vase_flowers_candles__2_to_1_001	蜡烛2to1更改了原candle；原candle是流蜡的，更改后是崭新的，没有保留原有蜡痕细节。
g1000_living_room_play__stuffed_animals_blanket_drum__2_to_3_001	2to3时把无关的drum往右挤；目标数量变化正确，但无关物体位置发生变化。
g1000_living_room_play__blocks_cushion_ball__2_to_3_001	block 2to3时为了留空间把无关的cushion变短。
g1000_warehouse_shelf__bag_quota__2_to_3_001	增加数量时产生了其他object的位置移动。
g1000_closet_wardrobe__jeans_belt__2_to_3_001	原来一沓两个裤子，变成了三沓六个裤子；没有在原有一沓内部增加一条。
g1000_parking_lot__pickup_meter_cart__absolute_horizontal_001	车辆移动时同时调转了方向。
g1000_playground__carousel_tunnel_bin__absolute_horizontal_002	两个物体的位置发生了互换；是否符合目标移动关系需要复核。
g1000_living_room_play__lego_pillow_train__1_to_2_001	数量变成了三个枕头而不是两个。
g1000_warehouse_shelf__bag_quota__3_to_2_00	数量变化失败；白色桶还是四个，只是位置布局不同。
g1000_warehouse_shelf__tool_bin_paint__2_to_1_001	数量变更同时带来了位置变化。
g1000_warehouse_shelf__drum_pallet_tote__2_to_3_001	木板生成错误；多生成了三个，而不是在原两个之上增加一个。
g1000_warehouse_shelf__sack_bundle_bin__3_to_2_001	source图中看起来没有三个目标物体，无法可靠完成3to2。
g1000_warehouse_shelf__paint_tool_pallet__2_to_3_001	多生成了三个黑箱子，而不是增加一个。
g1000_warehouse_shelf__paint_tool_pallet__3_to_2_001	白罐子仍然是五个，只是改变了位置；source画面复杂拥挤，考虑重新生成source。
g1000_warehouse_shelf__box_tote_drum__2_to_3_001	画面拥挤；2to3导致其他物体减少，考虑重新生成source。
g1000_warehouse_shelf__box_tote_drum__3_to_2_001	需要重新生成source。
g1000_picnic_blanket__apples_sandwich_oranges__3_to_2_001	数量变化错误，三个苹果变成了一个。
g1000_living_room_play__n3_doll__3_to_2_001	三块拼图变化后仍是三块，只是位置变了。
g1000_living_room_play__n2_train__2_to_3_001	目标不清楚；套圈桶变成了两个，相关物体总数变成四个。
g1000_living_room_play__n2_train__3_to_2_001	目标不清楚，原图看起来没有三个目标物体。
g1000_living_room_play__n1_rattle__2_to_1_001	数量变化同时带来了位置变化。
g1000_living_room_play__n1_rattle__2_to_3_001	数量变化错误，变成了四个。
g1000_bathroom_counter__rinsecup_soap_towel__occlusion_state_004	遮挡关系不明显，更像是左右位置互换。
g1000_bathroom_counter__skincare_toothpaste_pad__occlusion_state_001	牙膏与source不一致，需要重新生成source。
g1000_bedroom_nightstand__book_remote_vase__absolute_vertical_001	改变目标位置的同时改变了无关遥控器的位置。
g1000_bedroom_nightstand__glasses_charger_glass__occlusion_state_008	遮挡关系不明显，变成杯子遮挡台灯。
g1000_bedroom_nightstand__glasses_charger_glass__occlusion_state_010	遮挡关系不明显。
g1000_bedroom_nightstand__lamp_clock_book__functional_state_006	书没有打开，只是换了位置。
g1000_bedroom_nightstand__lamp_clock_book__functional_state_011	闹钟打开了，但显示时间与source不一致。
g1000_bedroom_nightstand__lamp_clock_book__functional_state_012	source书状态不对，target变成了两本书。
1000_bedroom_nightstand__lamp_glass_tray__absolute_vertical_001	台灯移下去时把桌子切割开了，桌子大小不合适。
g1000_bedroom_nightstand__lamp_glass_tray__absolute_vertical_004	台灯移下去时把桌子切割开了，桌子大小不合适。
g1000_bedroom_nightstand__tissuebox_coaster_readinglight__2_to_1_002	两个杯托都消失了。
g1000_bookshelf_study__book_bookend_plant__affordance_divergent_replace_002	source有两本红色书；target折纸位置改变，原有物体关系不一致。
g1000_bookshelf_study__book_globe_notebook__affordance_divergent_replace_004	source有两本蓝色书；target折纸位置改变，原书也没有被移除。
g1000_bookshelf_study__book_vase_candle__affordance_divergent_replace_004	target折纸位置改变，不在原物体位置。
g1000_bookshelf_study__bookframe_box_clock__affordance_divergent_replace_004	target折纸位置改变，不在原物体位置。
g1000_bookshelf_study__bookshelf_affordance_base__affordance_divergent_replace_006	target折纸位置改变，原来的书也没有被移除。
g1000_bookshelf_study__bottle_holder__containment_001	没有完成separate_from containment。
g1000_bookshelf_study__bottle_holder__occlusion_state_001	遮挡关系不正确，需要重新调整source图。
g1000_bookshelf_study__bottle_holder__occlusion_state_003	source并非unoccluded，遮挡关系不正确。
g1000_bookshelf_study__bs01b_book_crate__containment_002	移出containment时修改了containment的位置。
g1000_bookshelf_study__cardstack_figurine__2_to_1_001	source有两块区域都是两本书，target变成一块区域两本书，source/target结构不一致。
g1000_bookshelf_study__cardstack_figurine__occlusion_state_001	source并非unoccluded，遮挡关系不正确。
g1000_bookshelf_study__clock_radio_toycar__absolute_vertical_001	未改变成功。
g1000_bookshelf_study__magazine_lamp_globe__affordance_divergent_replace_001	lamp没有被replace。
g1000_bookshelf_study__magazine_lamp_globe__affordance_divergent_replace_002	lamp没有被replace。
g1000_bookshelf_study__mugcollection_folder__3_to_2_002	Source图不正确，一个杯子有两个握把，且target object发生位置变动；重新生成source。
g1000_bookshelf_study__mugcollection_folder__absolute_vertical_001	文件夹只旋转，没有发生vertical变换，且目标位置空间不足。
g1000_bookshelf_study__mugcollection_folder__affordance_divergent_replace_001	old value没有被replace。
g1000_bookshelf_study__penjar_folder_trophy__absolute_vertical_001	Trophy消失而不是移动。
g1000_bookshelf_study__picturebook_bin__containment_001	对象消失而不是移出containment。
g1000_bookshelf_study__reading_shelf__absolute_vertical_001	移下来的同时带来其他object位置变化，应修改source以不影响target任务。
g1000_bookshelf_study__reading_shelf__absolute_vertical_002	没有正向上移动。
g1000_bookshelf_study__reference_desk__absolute_vertical_001	移动下来时把书放倒了。
g1000_bookshelf_study__stationery_box__containment_001	对象消失而不是移出containment。
g1000_bookshelf_study__stationery_tier__absolute_vertical_003	移动下来时带来其他object位置变化；应修改source，或明确允许放在下方object之上。
g1000_bookshelf_study__tin_plant__containment_001	plant消失而不是移入containment。
g1000_city_street__busstop_planterbox_bikerack__shape_divergent_replace_004	未按照target要求修改。
g1000_city_street__busstop_planterbox_bikerack__shape_divergent_replace_007	Source图有黑框，重新生成source。
g1000_city_street__newsstand_barrier_lamp__shape_divergent_replace_006	Source图有白框，重新生成source。
g1000_classroom__book_quota__2_to_3_001	数量变成了四个。
g1000_classroom__book_quota__3_to_2_001	数量变成了零个。
g1000_classroom__chair_backpack_notebook__1_to_2_002	未改变成功。
g1000_classroom__classroom_same_base__same_affordance_replace_008	Source桌子遮挡了凳子，重新生成source。
g1000_classroom__desk_books_pencilcase__2_to_3_002	数量变成了四个。
g1000_classroom__desk_ruler_eraser__same_affordance_replace_007	old value没有被replace。
g1000_classroom__n1_chalk__2_to_3_001	Source图是抠图而不是完整场景，重新生成source。
g1000_classroom__n1_chalk__3_to_2_001	数量变成了一个。
g1000_classroom__n2_globe__3_to_2_001	数量变成了一个。
g1000_classroom__n3_chair__3_to_2_001	source原本就只有两个，无法完成3to2，target变成了零个。
g1000_classroom__n3_chair__3_to_2_002	Source图不对，需要重新生成source。
g1000_classroom__n4_ruler__2_to_3_001	数量变成了四个。
g1000_closet_wardrobe__beanie_hatbox_scarf__1_to_2_002	未改变成功。
g1000_closet_wardrobe__beanie_hatbox_scarf__material_001	看不出material变化，需要修改prompt。
g1000_closet_wardrobe__belt_fill__3_to_2_001	source没有三个belt，需要重新修改source/prompt。
g1000_closet_wardrobe__cw06b_socks_basket_separate__containment_001	containment不明显；袜子放进筐里会看不到，袜子数量/选择也不清楚，重新生成source。
g1000_closet_wardrobe__cw06b_socks_basket_separate__containment_002	containment不明显；袜子放进筐里会看不到，袜子数量/选择也不清楚，重新生成source。
g1000_closet_wardrobe__jeans_belt__2_to_3_001	一沓两个变成三沓六个；应在原有一沓内部增加一条。
g1000_closet_wardrobe__jeans_belt__occlusion_state_001	遮挡关系不明确，需要修正。
g1000_closet_wardrobe__jeans_belt__occlusion_state_002	遮挡关系不明确，需要修正。
g1000_closet_wardrobe__jeans_belt__occlusion_state_003	遮挡关系不明确，需要修正。
g1000_closet_wardrobe__jeans_belt__occlusion_state_004	遮挡关系不明确，需要修正。
g1000_closet_wardrobe__n1_tie__3_to_2_001	source没有三个tie。
g1000_closet_wardrobe__n2_sweater__2_to_3_002	source有一沓两个毛衣，下层还有一个无关毛衣，最好移除无关干扰。
g1000_closet_wardrobe__n2_sweater__3_to_2_001	数量变成了一个。
g1000_closet_wardrobe__shirt_quota__2_to_3_001	一沓两个变成两沓四个。
g1000_closet_wardrobe__shirt_quota__3_to_2_001	变更失败。
g1000_closet_wardrobe__shirt_sweater_shoebox__containment_001	移出containment失败。
g1000_closet_wardrobe__shirt_sweater_shoebox__material_001	看不出材质变化，需要修改prompt使其肉眼可见。
g1000_closet_wardrobe__socks_basket_pajamas_hat__1_to_2_001	Source有一沓两个，变成两沓四个。
g1000_closet_wardrobe__socks_basket_pajamas_hat__1_to_2_004	变化失败，只把帽子变大了。
g1000_closet_wardrobe__socks_basket_pajamas_hat__containment_001	containment里袜子太多，移出来只看到一个，重新生成source。
g1000_closet_wardrobe__socks_basket_pajamas_hat__containment_002	containment里袜子太多，移出来只看到一个，重新生成source。
g1000_closet_wardrobe__socks_basket_pajamas_hat__material_001	看不出材质变化，需要修改prompt使其肉眼可见。
g1000_closet_wardrobe__socks_basket_pajamas_hat__material_003	看不出材质变化，需要修改prompt使其肉眼可见。
g1000_closet_wardrobe__socks_basket_pajamas_hat__occlusion_state_001	遮挡关系不明确，变化失败。
g1000_closet_wardrobe__socks_basket_pajamas_hat__occlusion_state_002	遮挡关系不明确，变化失败。
g1000_closet_wardrobe__sunglasses_giftbox_tie__1_to_2_002	数量变化失败。
g1000_closet_wardrobe__sunglasses_giftbox_tie__containment_002	移出后对象消失。
g1000_closet_wardrobe__sunglasses_giftbox_tie__material_001	看不出材质变化，需要修改prompt使其肉眼可见。
g1000_closet_wardrobe__tshirts_jewelrybox__3_to_2_001	Source有两沓，其中一沓四个；target仍是一沓四个，需要修改source图。
g1000_closet_wardrobe__tshirts_jewelrybox__occlusion_state_001	遮挡关系不明确，变化失败。
g1000_closet_wardrobe__tshirts_jewelrybox__occlusion_state_002	遮挡关系不明确，变化失败。
g1000_dining_table__basket_bread_jar__containment_001	Source有两个，移出来后只剩外面一个，里面对象消失。
g1000_dining_table__basket_bread_jar__containment_002	两个对象都移出来了，是否符合要求需要复核。
g1000_dining_table__fruit_bowl_napkins__1_to_2_003	数量变化失败。
g1000_dining_table__fruit_bowl_napkins__containment_001	Source里面有三个，移出来一个后其他两个消失，是否符合要求需要复核。
g1000_dining_table__fruit_bowl_napkins_out__containment_001	外面有三个，target变成里面两个、外面两个。
g1000_dining_table__fruit_bowl_napkins_out__containment_002	外面有三个，target变成里面三四个、外面一个。
g1000_dining_table__n1_napkin__2_to_1_001	Source有两个摞在一起的napkin，target一个也没了。
g1000_dining_table__n1_napkin__2_to_3_001	Source有两个摞在一起的napkin，target变成两个摞在一起的两个napkin，总数四个。
g1000_dining_table__n2_cup__2_to_1_001	数量变化失败，source图重新生成。
g1000_dining_table__n2_cup__2_to_3_001	数量变化失败，source图重新生成。
g1000_dining_table__n2_cup__3_to_2_001	数量变化失败，source图重新生成。
g1000_dining_table__n4_jar__2_to_3_001	Source有无关项影响，重新生成source。
g1000_dining_table__napkin_fill__3_to_2_001	数量变化失败。
g1000_dining_table__plate_quota__2_to_1_001	数量变化失败。
g1000_dining_table__plate_quota__2_to_3_001	数量变化失败。
g1000_dining_table__plate_quota__3_to_2_00	原来有四个glass，变成了零个，无法完成3to2，数量变化失败。
g1000_flower_shop__lily_vase_ribbon__containment_002	移出失败。
g1000_flower_shop__orchid_stand_gift__containment_002	移入失败。
g1000_flower_shop__orchid_stand_giftb__containment_002	移出失败。
g1000_flower_shop__peony_crate_pot__1_to_2_004	Source有一个无关vase，1to2时把瓶子去掉了；需修改source或让目标生成在不影响其他object的位置。
g1000_flower_shop__peony_crate_pot__containment_002	移入后对象不可见。
g1000_flower_shop__peony_crate_pot__pattern_002	看不出pattern/材质变化，需要修改prompt使其肉眼可见。
g1000_flower_shop__peony_crate_pot__pattern_007	看不出pattern/材质变化，需要修改prompt使其肉眼可见。
g1000_flower_shop__peony_crate_potb__containment_002	移出失败。
g1000_flower_shop__rose_vase_wrap__1_to_2_003	数量变成了三个。
g1000_flower_shop__rose_vase_wrapb__containment_002	source有两个vase，重新生成source。
g1000_flower_shop__rose_vase_wrapc__containment_001	old value没有被替换。
g1000_flower_shop__rose_vase_wrapc__containment_002	移出失败。
g1000_flower_shop__tulip_pot_ribbon__containment_001	移出失败。
g1000_flower_shop__tulip_pot_ribbonb__containment_001	移出失败。
g1000_flower_shop__tulip_pot_ribbonb__containment_002	移出后对象不可见。
g1000_grocery_display__apple_basket_oranges__3_to_2_001	数量变化失败。
g1000_grocery_display__apple_basket_oranges__same_affordance_replace_002	变化后source里的水果不可见。
g1000_grocery_display__apple_basket_oranges__same_affordance_replace_003	变化后source里的水果不可见。
g1000_grocery_display__biscuit_tea_honey__2_to_1_001	Source前后有三个tea，变化后仍有两个；source太复杂，重新生成source。
g1000_grocery_display__biscuit_tea_honey__2_to_3_001	Source太复杂，重新生成source；修改后影响其他object。
g1000_grocery_display__biscuit_tea_honey__3_to_2_001	Source太复杂，重新生成source。
g1000_grocery_display__bread_basket_croissants__containment_001	容器里面有两个，变化后外面一个、里面没有。
g1000_grocery_display__cans_bin_jars__3_to_2_001	Source前后关系不是三个，重新生成source。
g1000_grocery_display__cans_bin_jars__containment_001	Source有三个can，变化后外面有四个can。
g1000_grocery_display__cereal_quota__2_to_3_001	数量变化失败。
g1000_grocery_display__coffee_milk_cereal__2_to_1_001	数量从2变成0，目标对象消失。
g1000_grocery_display__flowers_vase_plants__absolute_horizontal_001	左边两个花移到右边时超出范围，只看到一个花。
g1000_grocery_display__gd04b_flowers_vase_separate__containment_001	Source里外都有花，重新生成source。
g1000_grocery_display__gd05b_milk_crate_separate__containment_001	容器内多生成一个，外面仍是三个。
g1000_grocery_display__gd07b_potatoes_bag_separate__containment_001	Source的meshbag里外都有土豆，重新生成source。
g1000_grocery_display__honey_biscuit_water__2_to_1_001	变化了其他object，重新生成source降低复杂度。
g1000_grocery_display__honey_biscuit_water__2_to_3_001	Water变成四个而不是三个。
g1000_grocery_display__honey_biscuit_water__3_to_2_001	三个biscuit全部消失。
g1000_grocery_display__jam_pasta_tea__2_to_1_001	两个tea全部消失。
g1000_grocery_display__jam_pasta_tea__2_to_3_001	数量增加影响其他object，重新生成source降低复杂度。
g1000_grocery_display__jam_pasta_tea__3_to_2_001	Source没有三个目标，重新生成source降低复杂度。
g1000_grocery_display__potatoes_bag_tomatoes__absolute_horizontal_002	移动后左边多了一个，右边仍是三个。
g1000_grocery_display__potatoes_bag_tomatoes__containment_001	移出后数量变化；重新生成source让袋子里只有一个土豆，降低复杂度。
g1000_grocery_display__potatoes_bag_tomatoes__material_001	材质变化同时导致数量变化。
g1000_grocery_display__potatoes_bag_tomatoes__same_affordance_replace_002	变化后容器内东西不可见。
g1000_grocery_display__potatoes_bag_tomatoes__same_affordance_replace_003	生成了两个box分装，应保持一个。
g1000_kitchen_counter__bowl_block_mill__occlusion_state_005	旧的木板没有被删除。
g1000_kitchen_counter__bowl_holder_bottle__functional_state_006	bottle消失。
g1000_kitchen_counter__kettle_tin_lemon__1_to_2_002	变化影响其他object。
g1000_kitchen_counter__toaster_dish_jar__occlusion_state_004	遮挡关系不正确。
g1000_kitchen_counter__toaster_dish_jar__occlusion_state_005	遮挡关系不正确。
g1000_living_room_play__ball_quota__2_to_3_001	两个变成了四个。
g1000_living_room_play__ball_quota__3_to_2_001	Block数量变化同时改变了位置。
g1000_living_room_play__blocks_cushion_ball__2_to_3_001	数量增加同时把cushion变短。
g1000_living_room_play__lego_pillow_train__1_to_2_001	变成了三个而不是两个。
g1000_living_room_play__puzzle_book_nook__absolute_vertical_002	移动时把书立起来了而不是保持平放。
g1000_living_room_play__puzzle_book_nook__absolute_vertical_001	移动时把书平放了而不是保持原本立着。
g1000_living_room_play__puzzle_fill__3_to_2_001	原来有四块而不是三块。
g1000_living_room_play__rings_shelf_car__absolute_vertical_001	上移时上方有物体，发生混叠且物体被复制，需重新设计source。
g1000_living_room_play__rings_shelf_car__absolute_vertical_002	下移时下方有物体，只能移动到右边。
g1000_living_room_play__robot_shelf_ball__absolute_vertical_003	移动失败。
g1000_living_room_play__stuffed_animal_corner__absolute_vertical_002	旧位置没有消除。
g1000_living_room_play__stuffed_animal_corner__absolute_vertical_003	旧位置没有消除。
g1000_living_room_play__stuffed_animals_blanket_drum__absolute_horizontal_003	变化失败，效果不明显。
g1000_living_room_play__stuffed_shelf_drum__absolute_vertical_001	变化失败。
g1000_living_room_play__stuffed_shelf_drum__absolute_vertical_003	变化失败，大小发生改变且有其他物体阻挡。
g1000_living_room_play__toy_shelf_setup__absolute_vertical_002	上方空间不足，无法移动。
g1000_office_desk__lamp_holder_notebook__functional_state_002	old value不够open，重新生成source。
g1000_office_desk__mug_stapler_pad__material_002	材质变化不明显，需要重新设计使其肉眼可见。
g1000_office_desk__mug_stapler_pad__material_005	材质变化不明显，需要重新设计使其肉眼可见。
g1000_office_desk__stand_keyboard_pad__occlusion_state_001	看起来只是位置移动，遮挡关系不明显，需要重新设计。
g1000_office_desk__stand_keyboard_pad__occlusion_state_002	看起来只是位置移动，遮挡关系不明显，需要重新设计。
g1000_office_desk__stand_keyboard_pad__occlusion_state_005	old value未被消除。
g1000_office_desk__tray_calendar_dispenser__1_to_2_002	空间不足影响其他object，需在calendar前边的空白位置生成且不影响其他object。
g1000_office_desk__tray_calendar_dispenser__material_002	材质变化不明显，需要重新设计使其肉眼可见。
g1000_pantry_shelf_closeup__cans_open_crate__containment_001	Source容器内很多object，移出后只剩一个；重新设计source让容器内只有一个物体。
g1000_pantry_shelf_closeup__cans_open_crate_out__containment_001	Source容器外很多object，移入后只有一个；重新设计source让容器内外各只有一个物体。
g1000_pantry_shelf_closeup__jars_open_box__containment_001	容器内三个移出后变成四个。
g1000_pantry_shelf_closeup__jars_open_box_out__containment_001	容器太大，移入后物体不可见，需重新设计source。
g1000_parking_lot__car_cart_cone__shape_divergent_replace_004	old value未被消除，而且与shape_divergent_replace_003很相似。
g1000_parking_lot__pickup_meter_cart__absolute_horizontal_001	移动的同时车辆进行了旋转。
g1000_parking_lot__pickup_meter_cart__absolute_horizontal_003	位置移动失败，source本来就在中间，重新生成source。
g1000_parking_lot__pickup_meter_cart__shape_divergent_replace_003	old value未被消除替换。
g1000_parking_lot__suv_trolley_pole__absolute_horizontal_003	old value未被消除替换。
g1000_parking_lot__suv_trolley_pole__shape_divergent_replace_002	old value未被消除替换。
g1000_parking_lot__suv_trolley_pole__shape_divergent_replace_003	old value未被消除替换。
g1000_parking_lot__van_barrier_sign__shape_divergent_replace_002	old value未被消除替换。
g1000_picnic_blanket__apples_sandwich_oranges__3_to_2_001	数量从三变成一个，而不是两个。
g1000_playground__carousel_tunnel_bin__absolute_horizontal_003	变化失败。
g1000_playground__slide_swing_bench__absolute_horizontal_003	变化失败。
g1000_toolshed_garage__drill_toolbox_bin__containment_001	变化失败。
g1000_toolshed_garage__drill_toolbox_bin__material_001	材质变化不明显，需要重新设计使其肉眼可见。
g1000_toolshed_garage__hammer_oilcan_basket__containment_001	变化失败。
g1000_toolshed_garage__hammer_oilcan_basket__functional_state_002	状态变化不明显，需要重新设计使其肉眼可见。
g1000_toolshed_garage__hammer_oilcan_basket__material_001	材质变化不明显，需要重新设计使其肉眼可见。
g1000_toolshed_garage__hammer_oilcan_basketb__containment_001	变化失败。
g1000_toolshed_garage__saw_gasoline_crate__containment_001	old value未被消除替换。
g1000_toolshed_garage__saw_gasoline_crate__functional_state_002	状态变化不明显，需要重新设计使其肉眼可见。
g1000_toolshed_garage__saw_gasoline_crateb__containment_001	old value未被消除替换。
g1000_toolshed_garage__screwdriver_bolt_bucket__containment_001	多生成了一个bucket。
g1000_toolshed_garage__screwdriver_bolt_bucket__functional_state_002	变化失败。
g1000_warehouse_shelf__bag_quota__2_to_3_001	增加数量时移动了其他object，需重新设计source留出空间降低复杂度。
g1000_warehouse_shelf__bag_quota__3_to_2_001	数量变化失败，需重新设计source留出空间降低复杂度。
g1000_warehouse_shelf__bottles_shippingcrate__containment_001	old value未被消除替换。
g1000_warehouse_shelf__bottles_shippingcrate__material_001	材质变化同时改变了木箱方向。
g1000_warehouse_shelf__box_sack_pallet__2_to_1_001	看起来不是2to1，重新设计source留出空间降低复杂度。
g1000_warehouse_shelf__box_sack_pallet__2_to_3_001	重新设计source留出空间降低复杂度。
g1000_warehouse_shelf__box_sack_pallet__3_to_2_001	重新设计source留出空间降低复杂度。
g1000_warehouse_shelf__box_tote_drum__2_to_3_001	增加数量时移动其他object，重新设计source留出空间降低复杂度。
g1000_warehouse_shelf__box_tote_drum__3_to_2_001	数量变化时移动其他object，重新设计source留出空间降低复杂度。
g1000_warehouse_shelf__bucket_fill__3_to_2_001	Source有四个bucket且数量变化带来位置变化，重新生成source。
g1000_warehouse_shelf__drum_pallet_tote__2_to_3_001	Source是一沓两个pallet，target多了三个；应在原有的同一沓上增加一个。
g1000_warehouse_shelf__drums_palletcage__2_to_1_001	数量变化错误。
g1000_warehouse_shelf__drums_palletcage__containment_001	old value未被消除替换。
g1000_warehouse_shelf__fasteners_bin_sacks__1_to_2_001	old value未被消除替换。
g1000_warehouse_shelf__fasteners_bin_sacks__1_to_2_002	数量变化错误。
g1000_warehouse_shelf__fasteners_bin_sacks__2_to_1_001	重新生成source；原source的sacks后方还有sacks。
g1000_warehouse_shelf__helmets_gearbin_handtruck__2_to_3_001	新增物体不在原有物体附近，而在很远的区域。
g1000_warehouse_shelf__helmets_gearbin_handtruck__containment_001	Source里面有两个，移出后只移出一个。
g1000_warehouse_shelf__helmets_gearbin_handtruck__same_affordance_replace_003	替换带来了数量变化。
g1000_warehouse_shelf__inventory_rack__absolute_vertical_001	移动同时带来其他object位置变化，需重新生成source。
g1000_warehouse_shelf__pail_fill__2_to_3_001	Source有两层且每层四个；修改source，只保留一层的两个。
g1000_warehouse_shelf__paint_tool_pallet__2_to_3_001	增加数量时移动其他object，重新设计source留出空间降低复杂度。
g1000_warehouse_shelf__paint_tool_pallet__3_to_2_001	数量变化时移动其他object，重新设计source留出空间降低复杂度。
g1000_warehouse_shelf__parts_metalbasket_tires__3_to_2_001	Source不是三个，重新生成source。
g1000_warehouse_shelf__parts_metalbasket_tires__containment_001	old value未被消除替换。
g1000_warehouse_shelf__storage_bay__absolute_vertical_001	Helmet向下移动似乎进入箱子而不可见。
g1000_warehouse_shelf__storage_bay__absolute_vertical_002	上方没有移动空间。
g1000_warehouse_shelf__tool_bin_paint__2_to_1_001	数量变化同时带来位置变化。
g1000_warehouse_shelf__tote_paint_box__2_to_3_001	生成后总数变成四个。
g1000_warehouse_shelf__wh07b_fasteners_bin_separate__containment_001	Source有两层且old value未被消除，修改source只保留一层。
g1000_warehouse_shelf__wh07b_fasteners_bin_separate__containment_002	containment比object小很多，不符合物理常识，需修改source。
g1000_warehouse_shelf__wrench_crate_boxes__containment_001	old value未被消除替换。
"""

SOURCE_MARKERS = (
    "重新生成source", "重新生成 source", "重新设计source", "重新设计 source",
    "修改source", "修改 source", "source图", "Source图", "Source有", "source有",
    "Source并非", "source并非", "Source不是", "source不是", "Source没有", "source没有",
    "Source太复杂", "source太复杂", "Source的", "source的", "Source本来", "source本来",
    "Source里面", "source里面", "Source前后", "source前后", "Source容器", "source容器",
    "source重新", "修改source/prompt", "修改source或",
)


def parse_raw() -> list[tuple[str, str]]:
    rows = []
    for line_no, line in enumerate(RAW_REPORT.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if "\t" not in line:
            raise ValueError(f"raw report line {line_no} has no tab delimiter: {line}")
        pair_id, issue = line.split("\t", 1)
        rows.append((pair_id.strip(), issue.strip()))
    return rows


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for line in path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            result[str(row["pair_id"])] = row
    return result


def match_id(raw_id: str, manifest: dict[str, dict[str, Any]]) -> tuple[str, str]:
    if raw_id in manifest:
        return raw_id, "exact"
    matches = difflib.get_close_matches(raw_id, list(manifest), n=1, cutoff=0.72)
    if matches:
        return matches[0], "fuzzy_candidate_do_not_auto_correct"
    return "", "not_found"


def source_marked(issue: str) -> bool:
    return any(marker.lower() in issue.lower() for marker in SOURCE_MARKERS)


def add_table(ws, ref: str, name: str) -> None:
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False)
    ws.add_table(table)


def style_sheet(ws) -> None:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
    ws.row_dimensions[1].height = 30


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    raw_rows = parse_raw()
    manifest = load_manifest(args.manifest)
    workbook = Workbook()
    summary = workbook.active
    summary.title = "问题汇总"
    headers = [
        "序号", "pair_id_raw", "pair_id_manifest", "ID匹配状态", "manifest_subtype",
        "intervention_field", "old_value", "new_value", "用户审计问题",
        "用户明确标记source重生成", "本阶段处理范围", "后续修改措施（第三步填写）",
        "最终处理状态（待填写）",
    ]
    summary.append(headers)
    source_rows = []
    subtype_counts = Counter()
    match_counts = Counter()
    source_count = 0
    for index, (raw_id, issue) in enumerate(raw_rows, 1):
        matched, match_status = match_id(raw_id, manifest)
        record = manifest.get(matched, {})
        intervention = record.get("intervention", {})
        marked = source_marked(issue)
        source_count += int(marked)
        subtype_counts[record.get("subtype", "未匹配")] += 1
        match_counts[match_status] += 1
        scope = "source + target（用户明确标记）" if marked else "target-only（用户未标记source）"
        row = [
            index, raw_id, matched, match_status, record.get("subtype", ""),
            intervention.get("field", ""), intervention.get("old_value", ""),
            intervention.get("new_value", ""), issue, "是" if marked else "否", scope, "", "",
        ]
        summary.append(row)
        if marked:
            source_rows.append(row)

    source_ws = workbook.create_sheet("Source重生成标记")
    source_ws.append(headers)
    for row in source_rows:
        source_ws.append(row)

    stats = workbook.create_sheet("统计")
    stats.append(["统计项", "数值"])
    stats_rows = [
        ("用户报告原始条目数", len(raw_rows)),
        ("原始pair_id去重数", len({x[0] for x in raw_rows})),
        ("manifest精确匹配数", match_counts["exact"]),
        ("manifest疑似模糊匹配数（需人工确认）", match_counts["fuzzy_candidate_do_not_auto_correct"]),
        ("manifest未找到数", match_counts["not_found"]),
        ("用户明确标记source重生成数", source_count),
        ("用户未标记source、暂按target-only数", len(raw_rows) - source_count),
    ]
    for item, value in stats_rows:
        stats.append([item, value])
    stats.append([])
    stats.append(["manifest subtype", "条目数"])
    for subtype, count in sorted(subtype_counts.items(), key=lambda x: str(x[0])):
        stats.append([subtype, count])

    raw_ws = workbook.create_sheet("原始问题文本")
    raw_ws.append(["序号", "pair_id_raw", "用户原始问题"])
    for index, (raw_id, issue) in enumerate(raw_rows, 1):
        raw_ws.append([index, raw_id, issue])

    for ws in (summary, source_ws, stats, raw_ws):
        style_sheet(ws)
    widths = {
        "问题汇总": [8, 58, 58, 28, 24, 30, 16, 16, 70, 18, 28, 50, 20],
        "Source重生成标记": [8, 58, 58, 28, 24, 30, 16, 16, 70, 18, 28, 50, 20],
        "统计": [42, 20],
        "原始问题文本": [8, 58, 100],
    }
    for ws in (summary, source_ws, stats, raw_ws):
        for column_index, width in enumerate(widths[ws.title], 1):
            ws.column_dimensions[get_column_letter(column_index)].width = width
    add_table(summary, f"A1:M{summary.max_row}", "AuditIssueSummary")
    if source_ws.max_row > 1:
        add_table(source_ws, f"A1:M{source_ws.max_row}", "SourceRegenerationFlags")
    add_table(raw_ws, f"A1:C{raw_ws.max_row}", "RawAuditReport")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.output)
    print(json.dumps({
        "output": str(args.output),
        "raw_entries": len(raw_rows),
        "unique_raw_pair_ids": len({x[0] for x in raw_rows}),
        "source_marked": source_count,
        "exact_manifest_matches": match_counts["exact"],
        "fuzzy_candidates": match_counts["fuzzy_candidate_do_not_auto_correct"],
        "not_found": match_counts["not_found"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
