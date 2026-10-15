# final991_automask_v5 data audit report

Pairs audited: **991** (style rows: 991, integrity rows: 991; row errors: 0). Clean: **219**; with problems: **772**.

## Problem counts by flag

| flag | pairs |
|---|---|
| bg_style_drift | 717 |
| global_style_drift | 479 |
| recorded_local_check_failed | 14 |
| sat_shift | 4 |
| brightness_shift | 1 |
| mask_too_large | 1 |

## Problems by split

| split | problem pairs |
|---|---|
| test | 34 |
| train | 698 |
| val | 40 |

## Metric percentiles (all audited pairs)

### style

| metric | p50 | p90 | p99 | max |
|---|---|---|---|---|
| hist_l1 | 0.0979 | 0.1938 | 0.4370 | 0.8199 |
| bg_hist_l1 | 0.0674 | 0.1276 | 0.2241 | 0.8275 |
| bg_sat_delta | 0.0054 | 0.0144 | 0.0245 | 0.0331 |
| bg_val_delta | -0.0035 | 0.0004 | 0.0043 | 0.0172 |
| bg_contrast_delta | 0.0001 | 0.0029 | 0.0054 | 0.0107 |

### integrity

| metric | p50 | p90 | p99 | max |
|---|---|---|---|---|
| integrity.mask_coverage | 0.0962 | 0.2638 | 0.5441 | 0.8296 |
| integrity.changed_in_frac | 0.6186 | 0.8355 | 0.9216 | 0.9423 |
| integrity.changed_in_frac_sensitive | 0.8490 | 0.9532 | 0.9820 | 0.9939 |
| integrity.changed_out_frac | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| integrity.precision_in | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| integrity.iou_change_mask | 0.6186 | 0.8355 | 0.9216 | 0.9423 |

## Worst offenders

**Top 10 by background style drift (bg_hist_l1):**

- g1000_classroom__n1_chalk__2_to_3_002: 0.8275
- g1000_office_desk__lamp_holder_notebook__functional_state_001: 0.4996
- g1000_classroom__desk_ruler_eraser__same_affordance_replace_005: 0.2830
- g1000_office_desk__lamp_holder_notebook__functional_state_004: 0.2804
- g1000_classroom__desk_ruler_eraser__same_affordance_replace_007: 0.2592
- g1000_living_room_play__toy_shelf_setup__absolute_vertical_002: 0.2578
- g1000_flower_shop__rose_vase_wrap__1_to_2_002: 0.2547
- g1000_dining_table__fruit_bowl_napkins__absolute_horizontal_002: 0.2493
- g1000_flower_shop__rose_vase_wrap__pattern_001: 0.2290
- g1000_classroom__desk_ruler_eraser__same_affordance_replace_001: 0.2241

**Top 10 by background churn (changed_out_frac):**

- g1000_warehouse_shelf__wrench_toolbox_drum__containment_001: 0.0000
- g1000_warehouse_shelf__wrench_crate_boxes__same_affordance_replace_003: 0.0000
- g1000_warehouse_shelf__wrench_crate_boxes__same_affordance_replace_002: 0.0000
- g1000_warehouse_shelf__wrench_crate_boxes__same_affordance_replace_001: 0.0000
- g1000_warehouse_shelf__wrench_crate_boxes__containment_001: 0.0000
- g1000_warehouse_shelf__wrench_crate_boxes__color_001: 0.0000
- g1000_warehouse_shelf__wrench_crate_boxes__absolute_vertical_001: 0.0000
- g1000_warehouse_shelf__wrench_crate_boxes__2_to_3_001: 0.0000
- g1000_warehouse_shelf__wh07b_fasteners_bin_separate__containment_002: 0.0000
- g1000_warehouse_shelf__wh07b_fasteners_bin_separate__containment_001: 0.0000

**Top 10 by weakest edits (changed_in_frac@8, ascending):**

- g1000_toolshed_garage__drill_toolbox_bin__material_001: 0.4286
- g1000_flower_shop__orchid_stand_gift__pattern_003: 0.4521
- g1000_beach_scene__kayak_cooler_sandal__material_002: 0.4752
- g1000_dining_table__n1_napkin__3_to_2_001: 0.4795
- g1000_bedroom_nightstand__lamp_clock_book__functional_state_008: 0.4908
- g1000_toolshed_garage__screwdriver_bolt_bucket__functional_state_001: 0.4959
- g1000_living_room_play__n1_rattle__2_to_3_001: 0.5089
- g1000_closet_wardrobe__beanie_hatbox_scarf__pattern_002: 0.5127
- g1000_closet_wardrobe__belt_fill__3_to_2_001: 0.5178
- g1000_toolshed_garage__wrench_paintcan_crate__functional_state_001: 0.5202

## v8 filter guidance

Thresholds used by this report (regenerate with different values
via the CLI knobs; the raw audit rows stay unchanged):

| flag | metric | threshold |
|---|---|---|
| bg_style_drift | bg_hist_l1 | > 0.05 |
| global_style_drift | hist_l1 | > 0.1 |
| sat_shift | bg_sat_delta | |x| > 0.03 |
| contrast_shift | bg_contrast_delta | |x| > 0.03 |
| brightness_shift | bg_val_delta | |x| > 0.03 |

Suggested v8 filter (initial; calibrate on the training-loss
response): drop pairs with any hard integrity flag
(`bg_churn`, `no_change_in_mask`, `mask_too_small`,
`mask_too_large`, `mask_nonzero_mismatch`, `*_sha_mismatch`,
`missing_*`, `low_change_concentration`, `new_value_unreferenced`,
`recorded_local_check_failed`, `audit_error`); treat the style
flags (bg_style_drift / sat_shift / ...) as a soft tier —
inspect the worst-offender lists before dropping.

Machine-readable inputs for the v8 build:
`problem_pairs.jsonl` (per-pair flags + metrics) and
`audit_summary.json` (aggregates).
