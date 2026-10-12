#!/usr/bin/env python3
"""C3: pixel-level equivalence between merged final991 checkpoint and LoRA-loaded reference.

Compares gen/final991/robustness/merge_equiv (merged weights BAGEL-7B-MoT-final991-automask-v5)
against gen/final991/val (A-module LoRA-loaded references) for 4 val pairs x 2 states x seed 1234.

Verdict per image:
  fully_equivalent   max_abs_diff == 0 (bit-identical)
  within_tolerance   max_abs_diff <= 8 (8-bit units) AND mse <= 1e-3
  not_equivalent     otherwise

Note: GPU nondeterminism (atomics/reduction order) can yield small diffs even for
mathematically identical weights; bf16 merge rounding may add quantization-level noise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/final991_eval_v1")
MERGED_IMG = ROOT / "gen/final991/robustness/merge_equiv/images"
LORA_IMG = ROOT / "gen/final991/val/images"
SEED = 1234
STATES = ("source_state", "target_state")
TOL_MAX_DIFF = 8   # 8-bit units (~3% quantization noise)
TOL_MSE = 1e-3     # normalized [0,1]


def load(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)


def main() -> None:
    if not MERGED_IMG.is_dir():
        sys.exit(f"missing merged output dir: {MERGED_IMG}")
    pair_dirs = sorted(p for p in MERGED_IMG.iterdir() if p.is_dir())
    rows = []
    for pair_dir in pair_dirs:
        pair_id = pair_dir.name
        for state in STATES:
            m = pair_dir / f"seed_{SEED:010d}__{state}.png"
            l = LORA_IMG / pair_id / f"seed_{SEED:010d}__{state}.png"
            row = {"pair_id": pair_id, "state": state, "seed": SEED}
            if not m.is_file() or not l.is_file():
                row.update(verdict="not_equivalent", error="missing_file",
                           merged_path=str(m), lora_path=str(l))
                rows.append(row)
                continue
            a, b = load(m), load(l)
            if a.shape != b.shape:
                row.update(verdict="not_equivalent", error="shape_mismatch",
                           merged_shape=list(a.shape), lora_shape=list(b.shape))
                rows.append(row)
                continue
            diff = a - b
            abs_diff = np.abs(diff)
            max_d = float(abs_diff.max())
            mse = float((diff ** 2).mean())
            identical_ratio = float((abs_diff == 0).mean())
            if max_d == 0.0:
                verdict = "fully_equivalent"
            elif max_d <= TOL_MAX_DIFF and mse <= TOL_MSE:
                verdict = "within_tolerance"
            else:
                verdict = "not_equivalent"
            row.update(verdict=verdict, max_abs_diff=max_d, mse=mse,
                       identical_pixel_ratio=identical_ratio)
            rows.append(row)

    n = len(rows)
    counts = {v: sum(1 for r in rows if r["verdict"] == v)
              for v in ("fully_equivalent", "within_tolerance", "not_equivalent")}
    overall = ("fully_equivalent" if counts["fully_equivalent"] == n
               else "within_tolerance" if counts["not_equivalent"] == 0
               else "not_equivalent")
    report = {
        "comparison": "merged_final991_vs_lora_loaded",
        "merged_model": "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT-final991-automask-v5",
        "lora_checkpoint": "/dockerdata/niightli/BAGEL/models/final991_automask_v5_latest.pt",
        "seed": SEED, "n_images": n, "tolerance": {"max_abs_diff": TOL_MAX_DIFF, "mse": TOL_MSE},
        "counts": counts, "overall_verdict": overall,
        "rows": rows,
    }
    out = ROOT / "scores" / "merge_equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"[C3] {n} images compared -> {out}")
    print(f"[C3] counts: {counts}")
    print(f"[C3] OVERALL: {overall}")
    for r in rows:
        print(f"  {r['pair_id'][:52]:52s} {r['state']:13s} "
              f"{r['verdict']:17s} maxd={r.get('max_abs_diff', -1):6.1f} "
              f"mse={r.get('mse', -1):.2e} ident={r.get('identical_pixel_ratio', 0):.4f}")


if __name__ == "__main__":
    main()
