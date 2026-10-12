#!/usr/bin/env python3
"""Build a local HTML gallery to visually review Sentinel-6 matched inference outputs.

Reads generation_manifest.json for each profile and renders a comparison grid:
  columns = source ref | golden ref | step_000 (seed x3) | step_050 (seed x3)
  rows    = one per pair_id

This script only reads existing outputs; it does not run inference or modify
any generation results.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = REPO_ROOT / "experiments/counterfactual_cot/outputs"

PROFILES = ("delta_only_v2", "cot_core_v1")
CHECKPOINTS = ("step_000", "step_050")
SEEDS = (1234, 2234, 3234)


def profile_dir(profile: str) -> Path:
    return OUTPUTS_DIR / (
        f"sentinel6_gold40_final_v2__profile={profile}__checkpoints=000_050"
        "__seeds=1234_2234_3234__sampling=nt25_cfgT4_cfgI2_1024__inference"
    )


def rel(base: Path, target: Path) -> str:
    return os.path.relpath(str(target.resolve()), str(base.resolve()))


def build_profile_section(profile: str, out_html: Path) -> str:
    out = profile_dir(profile)
    manifest = json.loads((out / "generation_manifest.json").read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = manifest["records"]

    by_pair: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    for record in records:
        pair_bucket = by_pair.setdefault(record["pair_id"], {})
        ckpt_bucket = pair_bucket.setdefault(record["checkpoint"], {})
        ckpt_bucket[record["seed"]] = record

    rows_html = []
    for pair_id in sorted(by_pair):
        source_img = out / "references" / f"{pair_id}__source.png"
        golden_img = out / "references" / f"{pair_id}__golden.png"
        cells = [
            f'<td class="ref"><div class="lbl">source</div>'
            f'<img src="{rel(out_html.parent, source_img)}"></td>',
            f'<td class="ref"><div class="lbl">golden (target)</div>'
            f'<img src="{rel(out_html.parent, golden_img)}"></td>',
        ]
        for ckpt in CHECKPOINTS:
            seed_cells = []
            for seed in SEEDS:
                record = by_pair[pair_id].get(ckpt, {}).get(seed)
                if not record:
                    seed_cells.append('<div class="seed missing">missing</div>')
                    continue
                img_path = Path(record["image"])
                seed_cells.append(
                    f'<div class="seed"><div class="lbl">seed {seed}</div>'
                    f'<img src="{rel(out_html.parent, img_path)}"></div>'
                )
            cells.append(f'<td class="ckpt"><div class="lbl big">{ckpt}</div>{"".join(seed_cells)}</td>')
        rows_html.append(
            f'<tr><th class="pair">{html.escape(pair_id)}</th>{"".join(cells)}</tr>'
        )

    return f"""
    <h2>{html.escape(profile)}</h2>
    <p class="meta">manifest status: {html.escape(manifest.get("status",""))}
      | images: {manifest.get("generated_image_count")}
      | output_dir: <code>{html.escape(str(out))}</code></p>
    <table>
      <tr><th></th><th>source</th><th>golden</th><th>step_000</th><th>step_050</th></tr>
      {''.join(rows_html)}
    </table>
    """


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "sentinel6_review" / "review.html")
    args = parser.parse_args()
    out_html = args.out
    out_html.parent.mkdir(parents=True, exist_ok=True)

    sections = "".join(build_profile_section(profile, out_html) for profile in PROFILES)
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sentinel-6 Review</title>
<style>
body {{ font-family: sans-serif; margin: 20px; background:#111; color:#eee; }}
h1 {{ font-size: 20px; }}
h2 {{ margin-top: 40px; border-bottom: 1px solid #444; padding-bottom: 6px; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
th, td {{ border: 1px solid #333; padding: 6px; vertical-align: top; }}
th.pair {{ writing-mode: vertical-rl; text-orientation: mixed; font-size: 11px; max-width: 24px; }}
td.ref img {{ width: 160px; }}
td.ckpt {{ display: flex; gap: 6px; }}
.seed {{ text-align: center; }}
.seed img {{ width: 160px; }}
.lbl {{ font-size: 11px; color: #aaa; }}
.lbl.big {{ font-size: 13px; color: #fff; font-weight: bold; }}
.missing {{ color: #f55; padding: 40px 0; }}
.meta {{ color: #999; font-size: 12px; }}
code {{ color: #6cf; }}
</style></head>
<body>
<h1>Sentinel-6 Matched Inference Review (Step 0 vs Step 50, seeds 1234/2234/3234)</h1>
{sections}
</body></html>"""
    out_html.write_text(page, encoding="utf-8")
    print(str(out_html))


if __name__ == "__main__":
    main()
