#!/usr/bin/env python3
"""E2 visualization: attention heatmaps overlaid on the image with the edit mask.

Produces (per pair, per model):
  figures/{model}/{pair_id}_grid.png     rows = selected layers,
                                        cols = [source + mask contour,
                                                flow map (latent->changed token),
                                                prefill map (changed token->source VAE)]
  figures/overview_{model}.png           first N pairs at the best layer (from stats)

Flow maps: head-mean attention of each latent position on changed-token keys
(64x64 -> upsampled 1024, jet overlay). Prefill maps: changed-token query
attention over source-VAE keys. Mask contour in white.

Usage:
  python3 visualize.py --probe-root ../outputs/e2_token_probe \
      --layers 8 16 24 --pairs 6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
CFCOT = HERE.parent


def try_font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def load_mask(mask_path: Path) -> np.ndarray:
    m = np.asarray(Image.open(mask_path).convert("L").resize((1024, 1024), Image.NEAREST))
    return (m > 127).astype(np.uint8)


def mask_contour(mask: np.ndarray) -> Image.Image:
    """White contour of the mask on transparent background (1024)."""
    from PIL import ImageFilter
    inner = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))
    edge = (Image.fromarray((mask * 255).astype(np.uint8)).point(lambda x: 255 if x else 0)
            .convert("L"))
    inner_l = inner.point(lambda x: 255 if x else 0).convert("L")
    band = np.asarray(edge, dtype=np.int16) - np.asarray(inner_l, dtype=np.int16)
    band = np.clip(band, 0, 255).astype(np.uint8)
    out = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    white = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
    out.paste(white, (0, 0), Image.fromarray(band))
    return out


def upsample_map(flat: np.ndarray, side: int, out_size: int = 1024) -> Image.Image:
    grid = flat.reshape(side, side)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    rgb = np.stack([
        np.clip(1.5 - np.abs(4 * grid - 3), 0, 1),
        np.clip(1.5 - np.abs(4 * grid - 2), 0, 1),
        np.clip(1.5 - np.abs(4 * grid - 1), 0, 1)], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((out_size, out_size), Image.BICUBIC)


def overlay(base: Image.Image, heat: Image.Image, contour: Image.Image, alpha: float = 0.55) -> Image.Image:
    out = base.convert("RGB").copy()
    out = Image.blend(out, heat.convert("RGB"), alpha)
    out.paste(contour, (0, 0), contour)
    return out


def render_pair_grid(npz: dict, src_img: Image.Image, mask: np.ndarray,
                     layers: list[int], side: int, font, step_pref: str = "") -> Image.Image:
    contour = mask_contour(mask)
    src_thumb = src_img.convert("RGB").resize((1024, 1024))
    src_thumb.paste(contour, (0, 0), contour)

    rows = []
    for layer in layers:
        cells = [src_thumb]
        flow_key = next((k for k in npz if k.startswith(f"flow_map_L{layer}_")), None)
        if flow_key and step_pref in flow_key:
            m = npz[flow_key].astype(np.float32).mean(axis=0)
            cells.append(overlay(src_thumb, upsample_map(m, side), contour))
        else:
            cells.append(Image.new("RGB", (1024, 1024), "black"))
        pre_key = f"prefill_map_L{layer}"
        if pre_key in npz:
            cells.append(overlay(src_thumb, upsample_map(npz[pre_key].astype(np.float32), side),
                                 contour))
        else:
            cells.append(Image.new("RGB", (1024, 1024), "black"))
        rows.append(cells)

    thumb = 341
    w, h = 3 * thumb + 40, len(layers) * (thumb + 28) + 30
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), "left: source + mask contour | mid: flow attention (latent->changed tokens)"
                      " | right: prefill attention (changed tokens->source VAE)", fill="black", font=font)
    y = 30
    for cells, layer in zip(rows, layers):
        x = 8
        for c in cells:
            sheet.paste(c.resize((thumb, thumb), Image.BICUBIC), (x, y))
            x += thumb + 12
        draw.text((8, y + thumb + 2), f"layer {layer}", fill="black", font=font)
        y += thumb + 28
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path,
                        default=HERE / "../outputs/e2_token_probe")
    parser.add_argument("--out-dir", type=Path,
                        default=HERE / "../outputs/e2_token_probe/figures")
    parser.add_argument("--models", nargs="*", default=["base", "final991"])
    parser.add_argument("--layers", type=int, nargs="*", default=[8, 16, 24])
    parser.add_argument("--pairs", type=int, default=8, help="pairs to render")
    parser.add_argument("--step-pref", default="", help="filter flow maps by step suffix e.g. _S11")
    args = parser.parse_args()

    font = try_font(15)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_path = CFCOT / "final991_automask_v5_data/artifacts/splits/val.jsonl"
    records = {}
    for line in val_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            records[rec["pair_id"]] = rec

    for model in args.models:
        mdir = args.probe_root / f"probe_{model}"
        maps_dir = mdir / "maps"
        manifest_path = mdir / "probe_manifest.jsonl"
        if not manifest_path.exists():
            print(f"skip {model}: no manifest")
            continue
        ok_pairs = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    ok_pairs.append(rec["pair_id"])
        (args.out_dir / model).mkdir(exist_ok=True)
        for pid in ok_pairs[: args.pairs]:
            npz_path = maps_dir / f"{pid}.npz"
            if not npz_path.exists():
                continue
            rec = records.get(pid)
            if rec is None:
                continue
            with np.load(npz_path) as z:
                npz = {k: z[k] for k in z.files}
            src = Image.open(rec["source"]["image"]).convert("RGB")
            mask = load_mask(Path(rec["automatic_difference_mask"]["path"]))
            side = int(round(np.sqrt(next(
                (v.shape[-1] for k, v in npz.items() if k.startswith("flow_map")), 4096))))
            sheet = render_pair_grid(npz, src, mask, args.layers, side, font, args.step_pref)
            out = args.out_dir / model / f"{pid}_grid.png"
            sheet.save(str(out))
            print(f"saved {out}")


if __name__ == "__main__":
    main()
