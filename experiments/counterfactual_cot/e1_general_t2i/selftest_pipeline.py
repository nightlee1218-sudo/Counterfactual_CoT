#!/usr/bin/env python3
"""Offline pipeline self-test for E1 downstream stages (stats + contact sheet).

Creates synthetic generation manifests, fake PNGs and fake CLIP-T/VQA score
rows under /tmp, then runs stats_paired.py and contact_sheet.py against them.
Run:  python3 selftest_pipeline.py
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
TMP = Path("/tmp/e1_selftest")
PY = sys.executable
VENVPY = Path("/group/40016/niightli/BAGEL/venvs/geneval_eval/bin/python")
PY = str(VENVPY) if VENVPY.exists() else PY

subsets = {"attribute_binding": 12, "spatial_relations": 10, "counting": 8}
rng = random.Random(0)

if TMP.exists():
    shutil.rmtree(TMP)
for model, offset in (("base", 0.0), ("final991", 0.01)):
    gen_dir = TMP / "gen" / model
    manifest = gen_dir / "generation_manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w") as mf:
        for subset, n in subsets.items():
            for i in range(n):
                item_id = f"{subset}__x__{i:04d}"
                png = gen_dir / "images" / subset / f"{item_id}__seed1234.png"
                png.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (64, 64), tuple(rng.randrange(256) for _ in range(3))).save(png)
                mf.write(json.dumps({
                    "item_id": item_id, "subset": subset, "source_category": "x",
                    "prompt": f"a test prompt {i} for {subset}", "model": model, "seed": 1234,
                    "output_path": str(png)}, sort_keys=True) + "\n")

scores = TMP / "scores"
scores.mkdir(parents=True)
for model, shift in (("base", -0.005), ("final991", 0.005)):
    with (scores / f"clipT_{model}.jsonl").open("w") as f:
        for subset, n in subsets.items():
            for i in range(n):
                item_id = f"{subset}__x__{i:04d}"
                cos = 0.30 + 0.02 * rng.random() + shift
                f.write(json.dumps({"item_id": item_id, "subset": subset, "source_category": "x",
                                    "model": model, "clip_cos": cos, "clipscore": 2.5 * cos}, sort_keys=True) + "\n")
for model, acc in (("base", 0.4), ("final991", 0.6)):
    with (scores / f"vqa_{model}.jsonl").open("w") as f:
        for subset, n in subsets.items():
            for i in range(n):
                item_id = f"{subset}__x__{i:04d}"
                match = (rng.random() < acc)
                f.write(json.dumps({"item_id": item_id, "subset": subset, "source_category": "x",
                                    "model": model, "vqa_match": match, "confidence": "high",
                                    "reason": "synthetic"}, sort_keys=True) + "\n")

r1 = subprocess.run([PY, str(HERE / "stats_paired.py"), "--scores-dir", str(scores)], capture_output=True, text=True)
print("--- stats stdout ---"); print(r1.stdout[-2000:]); print(r1.stderr[-500:])
r2 = subprocess.run([PY, str(HERE / "contact_sheet.py"), "--gen-root", str(TMP / "gen"),
                     "--out-dir", str(TMP / "contact_sheets"), "--rows", "5"], capture_output=True, text=True)
print("--- sheet stdout ---"); print(r2.stdout[-1000:]); print(r2.stderr[-500:])

ok = (scores / "stats_summary.json").exists() and (scores / "comparison_table.md").exists() \
     and (TMP / "contact_sheets" / "overview.png").exists()
print("SELFTEST", "PASS" if ok and r1.returncode == 0 and r2.returncode == 0 else "FAIL")
