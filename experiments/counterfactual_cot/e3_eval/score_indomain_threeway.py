#!/usr/bin/env python3
"""Module-A scoring for the E3 three-way eval: per-item paired metrics.

Adapted copy of final991_eval_v1/score_indomain.py with an auditable CLI
patch — the METRIC CODE IS LINE-BY-LINE IDENTICAL (load_img / load_mask /
psnr / ssim / clip / dino / lpips / mask decomposition / directionality);
only the driver changed:

  * --models base final991 v6 (subset; default v6 only — base/final991 rows
    from final991_eval_v1 are copied into the e3_eval scores dir instead of
    being rescored, so the replication pair stays byte-identical);
  * per-model gen dirs (--gen-base-dir/--gen-final991-dir/--gen-v6-dir),
    defaulting to the final991_eval_v1 gen layout and the e3_eval v6 layout;
  * --splits/--splits-dir/--seeds; --rescore to overwrite; skip-if-exists
    otherwise (resumable);
  * heavy imports (torch/open_clip/dinov2/lpips) are lazy so the CPU
    selftest can import this module;
  * the paired-stats section moved to e3_eval/indomain_stats.py (pure).

Rows: {pair_id, field, op, intervention_type, model, split,
       src_mse, tgt_mse, tgt_psnr, tgt_ssim, tgt_lpips, clip_src, clip_tgt,
       clip_tgt_vs_src, dino_tgt, in_mse, out_mse, strength, bg_drift,
       directional}   (3-seed means, exactly like final991_eval_v1)

Run with the open_clip-capable python:
  /group/40016/niightli/BAGEL/venvs/geneval_eval/bin/python score_indomain_threeway.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFCOT = HERE.parent
BAGEL = Path("/group/40016/niightli/BAGEL")
DEFAULT_SPLITS_DIR = CFCOT / "final991_automask_v5_data/artifacts/splits"
DEFAULT_OUT_DIR = CFCOT / "outputs/e3_eval/scores_indomain"
DEFAULT_GEN_DIRS = {
    "base": CFCOT / "outputs/final991_eval_v1/gen/base",
    "final991": CFCOT / "outputs/final991_eval_v1/gen/final991",
    "v6": CFCOT / "outputs/e3_eval/gen_indomain/v6",
    "v7": CFCOT / "outputs/e3_eval/gen_indomain/v7",
}
VENV_SITE = BAGEL / "venvs/geneval_eval/lib/python3.10/site-packages"
SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["v6"],
                        help="subset of: base final991 v6 (default: v6 only)")
    parser.add_argument("--splits", nargs="*", default=["val", "test"])
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--gen-base-dir", type=Path, default=DEFAULT_GEN_DIRS["base"])
    parser.add_argument("--gen-final991-dir", type=Path, default=DEFAULT_GEN_DIRS["final991"])
    parser.add_argument("--gen-v6-dir", type=Path, default=DEFAULT_GEN_DIRS["v6"])
    parser.add_argument("--gen-v7-dir", type=Path, default=DEFAULT_GEN_DIRS["v7"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seeds", nargs="*", default=["1234", "2234", "3234"])
    parser.add_argument("--rescore", action="store_true", help="overwrite existing row files")
    args = parser.parse_args()
    bad = [m for m in args.models if m not in DEFAULT_GEN_DIRS]
    if bad:
        parser.error(f"unknown models: {bad} (base|final991|v6)")
    return args


def gen_dir_for(args: argparse.Namespace, model: str) -> Path:
    return {"base": args.gen_base_dir, "final991": args.gen_final991_dir,
            "v6": args.gen_v6_dir, "v7": args.gen_v7_dir}[model]


def score_split(model: str, split: str, pairs: list[dict], gen_dir: Path,
                seeds: list[str], ctx: dict) -> list[dict]:
    """ctx carries the lazily-built feature models + metric closures."""
    gen_manifest: dict[tuple, str] = {}
    man = gen_dir / split / "generation_manifest.jsonl"
    for line in man.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            gen_manifest[(r["pair_id"], r["state"], str(r["seed"]))] = r["output_path"]
    out_rows = []
    missing = []
    for p in pairs:
        pid = p["pair_id"]
        gold_src, gold_tgt = p["source"]["image"], p["target"]["image"]
        mask = ctx["load_mask"](p["automatic_difference_mask"]["path"])
        m_in, m_out = mask, ~mask
        row = {"pair_id": pid, "field": p["intervention"]["field"], "op": p["intervention"]["operation"],
               "intervention_type": p["intervention_type"], "model": model, "split": split}
        acc = {k: [] for k in ("src_mse", "tgt_mse", "tgt_psnr", "tgt_ssim", "tgt_lpips",
                               "clip_src", "clip_tgt", "clip_tgt_vs_src", "dino_tgt",
                               "in_mse", "out_mse", "strength", "bg_drift")}
        for seed in seeds:
            key_src = (pid, "source_state", seed)
            key_tgt = (pid, "target_state", seed)
            if key_src not in gen_manifest or key_tgt not in gen_manifest:
                missing.append(f"{pid}/{seed}")
                continue
            gs, gt = gen_manifest[key_src], gen_manifest[key_tgt]
            a_src, a_tgt = ctx["load_img"](gs), ctx["load_img"](gt)
            g_src, g_tgt = ctx["load_img"](gold_src), ctx["load_img"](gold_tgt)
            # pixel
            acc["src_mse"].append(float(((a_src - g_src) ** 2).mean()))
            tgt_mse = float(((a_tgt - g_tgt) ** 2).mean())
            acc["tgt_mse"].append(tgt_mse)
            acc["tgt_psnr"].append(ctx["psnr"](tgt_mse))
            acc["tgt_ssim"].append(ctx["ssim"](a_tgt, g_tgt))
            if ctx["lp"] is not None:
                acc["tgt_lpips"].append(float(ctx["lp"](ctx["to_tensor"](a_tgt), ctx["to_tensor"](g_tgt))))
            # semantic
            acc["clip_src"].append(ctx["cos"](ctx["clip_emb"](gs), ctx["clip_emb"](gold_src)))
            acc["clip_tgt"].append(ctx["cos"](ctx["clip_emb"](gt), ctx["clip_emb"](gold_tgt)))
            acc["clip_tgt_vs_src"].append(ctx["cos"](ctx["clip_emb"](gt), ctx["clip_emb"](gold_src)))
            if ctx["dino_emb"] is not None:
                acc["dino_tgt"].append(ctx["cos"](ctx["dino_emb"](gt), ctx["dino_emb"](gold_tgt)))
            # mask decomposition
            sq = ((a_tgt - g_tgt) ** 2).mean(-1)
            acc["in_mse"].append(float(sq[m_in].mean()) if m_in.any() else 0.0)
            acc["out_mse"].append(float(sq[m_out].mean()) if m_out.any() else 0.0)
            gold_change = float(((g_tgt - g_src) ** 2).mean(-1)[m_in].mean()) if m_in.any() else 1e-8
            gen_change = float(((a_tgt - a_src) ** 2).mean(-1)[m_in].mean()) if m_in.any() else 0.0
            acc["strength"].append(gen_change / (gold_change + 1e-8))
            acc["bg_drift"].append(float(((a_tgt - a_src) ** 2).mean(-1)[m_out].mean()) if m_out.any() else 0.0)
        for k, v in acc.items():
            row[k] = float(statistics.mean(v)) if v else None
        row["directional"] = row["clip_tgt"] - row["clip_tgt_vs_src"] \
            if row["clip_tgt"] is not None and row["clip_tgt_vs_src"] is not None else None
        out_rows.append(row)
    if missing:
        raise RuntimeError(
            f"gen manifest {man} is missing {len(missing)} (pair/seed) combos, "
            f"e.g. {missing[:3]} — generation incomplete; rerun the indomain-gen stage")
    return out_rows


def build_ctx() -> dict:
    """Lazy heavy imports + feature models (verbatim from score_indomain.py)."""
    import numpy as np  # noqa: F401
    import torch
    import torch.nn.functional as F
    from PIL import Image

    if str(VENV_SITE) not in sys.path:
        sys.path.insert(0, str(VENV_SITE))
    import open_clip  # noqa: E402

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def load_img(path: str, size: int = SIZE):
        return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC),
                          dtype=np.float32) / 255.0

    def load_mask(path: str, size: int = SIZE):
        m = np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST),
                       dtype=np.float32) / 255.0
        return m > 0.5

    def psnr(mse: float):
        return float("inf") if mse <= 0 else -10.0 * np.log10(mse)

    def to_tensor(img):
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(dev)

    def ssim(a, b):
        """8x8-window SSIM, single channel averaged over RGB."""
        try:
            from skimage.metrics import structural_similarity
            return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))
        except Exception:
            C1, C2 = 0.01 ** 2, 0.03 ** 2
            a_t, b_t = to_tensor(a), to_tensor(b)
            mu_a, mu_b = F.avg_pool2d(a_t, 8, 8), F.avg_pool2d(b_t, 8, 8)
            mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
            var_a = F.avg_pool2d(a_t ** 2, 8, 8) - mu_a2
            var_b = F.avg_pool2d(b_t ** 2, 8, 8) - mu_b2
            cov = F.avg_pool2d(a_t * b_t, 8, 8) - mu_ab
            ssim_map = ((2 * mu_ab + C1) * (2 * cov + C2)) / (
                (mu_a2 + mu_b2 + C1) * (var_a + var_b + C2))
            return float(ssim_map.mean())

    clip_model, _, clip_pre = open_clip.create_model_and_transforms(
        "ViT-g-14", pretrained="laion2b_s34b_b88k")
    clip_model = clip_model.eval().to(dev)

    dino_emb = None
    try:
        dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(dev)

        def dino_emb(path: str):
            im = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
            x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            x = (x - 0.485) / 0.229  # approx imagenet norm
            return F.normalize(dino(x.to(dev)))
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] dinov2 unavailable ({exc!r}); dino_tgt rows will be null", flush=True)

    try:
        import lpips as lpips_lib
        lp = lpips_lib.LPIPS(net="alex").to(dev)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] lpips unavailable ({exc!r}); tgt_lpips rows will be null", flush=True)
        lp = None

    @torch.no_grad()
    def clip_emb(path: str):
        return F.normalize(clip_model.encode_image(
            clip_pre(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)))

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a @ b.T).item())

    return {"load_img": load_img, "load_mask": load_mask, "psnr": psnr,
            "to_tensor": to_tensor, "ssim": ssim, "clip_emb": clip_emb,
            "dino_emb": dino_emb, "lp": lp, "cos": cos}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ctx = build_ctx()
    for model in args.models:
        for split in args.splits:
            out = args.out_dir / f"indomain_{model}_{split}.jsonl"
            if out.exists() and not args.rescore:
                print(f"[skip] {out} exists (use --rescore to overwrite)", flush=True)
                continue
            pairs = [json.loads(l) for l in
                     (args.splits_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
            rows = score_split(model, split, pairs, gen_dir_for(args, model), args.seeds, ctx)
            with out.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"[scored] {model}/{split}: {len(rows)} items -> {out}", flush=True)


if __name__ == "__main__":
    main()
