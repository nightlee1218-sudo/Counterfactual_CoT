# data_audit: final991_automask_v5 data quality audit (CPU only)

Audit of the 991 counterfactual pairs (train 896 / val 51 / test 44,
`final991_automask_v5_data/artifacts/splits/*.jsonl`) ahead of a possible
v8 data-filtered training run. Everything is pure PIL + numpy (no torch,
no GPU); images/masks are read from the absolute /group paths recorded in
the manifests.

## Files

| file | role |
|---|---|
| `style_audit.py` | target-vs-source style drift (global + **outside-mask**): saturation / brightness / contrast deltas and 16-bin RGB histogram L1. Rows -> `style_audit.jsonl` |
| `integrity_audit.py` | annotation-vs-pixels checks: file existence, sizes, sha256, mask binary/nonzero/coverage, recomputed change-mask consistency (changed_in/out, precision, IoU, MAD), prompt atomicity, recorded local-check cross-ref. Rows -> `integrity_audit.jsonl` |
| `report.py` | merges both row files, applies style thresholds, writes `problem_pairs.jsonl` + `audit_report.md` + `audit_summary.json` |
| `selftest_data_audit.py` | 36-check CPU suite on synthetic 96x96 pairs (every flag fires exactly once where it should; pure-black-pixel saturation regression) |

## Usage

```bash
python3 selftest_data_audit.py                      # CPU suite
# real audit (login node or GPU machine; ~991 pairs, minutes with workers)
python3 style_audit.py --splits-dir /group/40016/niightli/BAGEL/experiments/counterfactual_cot/final991_automask_v5_data/artifacts/splits \
    --out-dir run_final991_v5_20261015 --workers 16
python3 integrity_audit.py --splits-dir <same> --out-dir run_final991_v5_20261015 --workers 16
python3 report.py --audit-dir run_final991_v5_20261015
```

Default `--splits-dir` is the sibling `final991_automask_v5_data/artifacts/splits`
(works on the GPU-instance tree; on the login-node checkout pass the absolute
BAGEL-side path). Default `--out-dir` is `outputs/data_audit_v1` (gitignored
scratch); committed runs use an in-repo `run_*` directory.

## Flags and defaults

Integrity (computed at audit time; CLI knobs on `integrity_audit.py`):
`missing_*` / `*_size_mismatch` / `*_sha_mismatch` / `mask_not_binary` /
`mask_nonzero_mismatch` (exact nonzero-pixel count vs record) /
`mask_too_small` (<0.001) / `mask_too_large` (>0.8) /
`no_change_in_mask` (changed_in_frac@thr8 < 0.02) /
`bg_churn` (changed_out_frac@thr24 > 0.02 — background changed with the
edit) / `low_change_concentration` (precision_in@thr24 < 0.5) /
`identical_prompts` / `new_value_unreferenced` (token-overlap rule, see
below) / `recorded_local_check_failed` / `audit_error`.

Style (evaluated by `report.py` so thresholds stay report-time decisions):
`bg_style_drift` (bg_hist_l1 > 0.05) / `global_style_drift` (hist_l1 >
0.10) / `sat_shift` (|bg_sat_delta| > 0.03) / `contrast_shift` /
`brightness_shift` — all CLI-tunable; re-run `report.py` with new knobs to
re-flag without re-auditing.

Prompt atomicity rule (`integrity_audit.new_value_ref_ratio`): the
intervention `new_value` must be referenced by `target.prompt` — >= 50% of
its significant tokens (function words dropped, `:entity_N` index digits
dropped, `_region` suffix stripped) appear in the prompt, with field-aware
synonyms (`top_region`->upper, `bottom_region`->lower,
`unoccluded`->fully visible/nothing overlaps, `separate_from`->beside/
outside/next to, count digits->spelled-out words). Calibrated on the real
991 splits: all pairs pass (min ratio 0.5); the raw literal-substring
check would false-flag 287 pairs, hence the token rule.

## Notes

- Key findings of the committed run (`run_final991_v5_20261015`, all 991 pairs,
  0 row errors): integrity is clean (no sha/size/nonzero mismatches, no
  missing files, `changed_out_frac@24 == 0` everywhere, `precision_in == 1.0`
  at p50) — the recorded masks are exact. The style story is the opposite:
  717/991 pairs exceed `bg_hist_l1 > 0.05` (p50 = 0.067) while background
  pixels stay below the change threshold — a sub-threshold but systematic
  global re-render of the target (tone/color stylization), not local churn.
  14 pairs carry `recorded_local_check_failed`, 1 `mask_too_large`. For v8:
  drop the hard integrity failures, then treat `bg_hist_l1` as the main
  style-filter knob (p90 = 0.128 as a stricter cut leaves ~90% in).
- `quality.accepted` is `False` for all 991 pairs by design (the v5
  pipeline marks them auto-checked, human review pending) — reported as a
  global observation, not a per-pair flag.
- The recorded `quality.local_checks.changed_pixel_ratio` uses a build-time
  diff measure that is NOT reproducible from the shipped manifests; the
  audit recomputes its own change metrics (documented thresholds) and only
  echoes the recorded values, it does not cross-flag them. `mask
  nonzero_pixels` IS exact and is cross-checked (mask PNGs are lossless).
- v8 filter input: `problem_pairs.jsonl` (per-pair flags + key metrics) and
  `audit_summary.json` (aggregates + `problem_pair_ids`).
