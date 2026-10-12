#!/usr/bin/env python3
"""E1 contact sheets: side-by-side base vs final991 per subset (PIL only).

Layout: each row = [base | final991] for one item, 256px thumbs, prompt caption.
One sheet per subset (first --rows items) + one overview sheet mixing subsets.
Output: outputs/e1_general_t2i/contact_sheets/{subset}.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent


def try_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/TTF/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-root", type=Path, default=HERE / "../outputs/e1_general_t2i/gen")
    parser.add_argument("--out-dir", type=Path, default=HERE / "../outputs/e1_general_t2i/contact_sheets")
    parser.add_argument("--rows", type=int, default=20, help="items per subset sheet")
    parser.add_argument("--thumb", type=int, default=256)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    font = try_font(13)
    font_head = try_font(16)
    models = ["base", "final991"]
    manifests = {}
    for m in models:
        mf = args.gen_root / m / "generation_manifest.jsonl"
        manifests[m] = {} if not mf.exists() else {
            json.loads(l)["item_id"]: json.loads(l)
            for l in mf.read_text(encoding="utf-8").splitlines() if l.strip()}

    common_ids = sorted(set(manifests["base"]) & set(manifests["final991"]))
    by_subset: dict[str, list[str]] = {}
    for i in common_ids:
        by_subset.setdefault(manifests["base"][i]["subset"], []).append(i)

    label_w, caption_h, pad = 18, 40, 6
    cell_w, cell_h = args.thumb, args.thumb + caption_h
    for subset, ids in sorted(by_subset.items()):
        ids = ids[: args.rows]
        if not ids:
            continue
        w = label_w + 2 * (cell_w + pad) + pad
        h = 34 + len(ids) * (cell_h + pad) + pad
        sheet = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((pad, 8), f"{subset}  |  left: base   right: final991  |  seed 1234, 25 steps, CFG 4.0/2.0",
                  fill="black", font=font_head)
        y = 34
        for item_id in ids:
            x = label_w
            for m in models:
                rec = manifests[m][item_id]
                try:
                    im = Image.open(rec["output_path"]).convert("RGB").resize((args.thumb, args.thumb), Image.BICUBIC)
                except Exception:
                    im = Image.new("RGB", (args.thumb, args.thumb), "gray")
                sheet.paste(im, (x, y))
                prompt = rec["prompt"]
                if len(prompt) > 60:
                    prompt = prompt[:57] + "..."
                draw.text((x, y + args.thumb + 2), f"{m[:1].upper()}: {prompt}", fill="black", font=font)
                x += cell_w + pad
            y += cell_h + pad
        out = args.out_dir / f"{subset}.png"
        sheet.save(str(out))
        print(json.dumps({"event": "sheet", "subset": subset, "rows": len(ids), "out": str(out)}))

    # overview: first 5 of each subset
    overview_ids = [i for ids in by_subset.values() for i in ids[:5]]
    if overview_ids:
        w = label_w + 2 * (cell_w + pad) + pad
        h = 34 + len(overview_ids) * (cell_h + pad) + pad
        sheet = Image.new("RGB", (w, h), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((pad, 8), "E1 overview: left=base right=final991 (5 per subset)", fill="black", font=font_head)
        y = 34
        for item_id in overview_ids:
            x = label_w
            for m in models:
                rec = manifests[m][item_id]
                try:
                    im = Image.open(rec["output_path"]).convert("RGB").resize((args.thumb, args.thumb), Image.BICUBIC)
                except Exception:
                    im = Image.new("RGB", (args.thumb, args.thumb), "gray")
                sheet.paste(im, (x, y))
                prompt = rec["prompt"]
                if len(prompt) > 60:
                    prompt = prompt[:57] + "..."
                draw.text((x, y + args.thumb + 2), f"{m[:1].upper()}: {prompt}", fill="black", font=font)
                x += cell_w + pad
            y += cell_h + pad
        sheet.save(str(args.out_dir / "overview.png"))
        print(json.dumps({"event": "sheet", "subset": "overview", "rows": len(overview_ids)}))


if __name__ == "__main__":
    main()
