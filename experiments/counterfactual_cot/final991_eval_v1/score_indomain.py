#!/usr/bin/env python3
"""Module A scoring: per-item paired metrics for base vs final991.

Metrics: MSE / PSNR / SSIM, LPIPS, CLIP-I (ViT-g-14), DINOv2, plus task-specific
mask decomposition (in/out MSE, edit-strength ratio, background drift) and
directionality (gen_target closer to golden_target than to golden_source).

Outputs:
  scores/indomain_{model}_{split}.jsonl   per-item rows
  scores/summary.json                     aggregate + paired bootstrap CI
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BAGEL = Path("/group/40016/niightli/BAGEL")
ROOT = BAGEL / "experiments/counterfactual_cot/outputs/final991_eval_v1"
SPLITS = BAGEL / "experiments/counterfactual_cot/final991_automask_v5_data/artifacts/splits"
VENV_SITE = BAGEL / "venvs/geneval_eval/lib/python3.10/site-packages"

sys.path.insert(0, str(VENV_SITE))
import open_clip  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SIZE = 512


def load_img(path: str, size: int = SIZE) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0


def load_mask(path: str, size: int = SIZE) -> np.ndarray:
    m = np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST), dtype=np.float32) / 255.0
    return m > 0.5


def psnr(mse: float) -> float:
    return float("inf") if mse <= 0 else -10.0 * np.log10(mse)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEV)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
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
        ssim_map = ((2 * mu_ab + C1) * (2 * cov + C2)) / ((mu_a2 + mu_b2 + C1) * (var_a + var_b + C2))
        return float(ssim_map.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--seeds", nargs="*", default=["1234", "2234", "3234"])
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    root = Path(args.root)
    scores_dir = root / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # feature models
    clip_model, _, clip_pre = open_clip.create_model_and_transforms("ViT-g-14", pretrained="laion2b_s34b_b88k")
    clip_model = clip_model.eval().to(DEV)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(DEV)
    dino_pre = lambda im: F.interpolate(  # noqa: E731
        torch.from_numpy(np.asarray(im)).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
        scale_factor=14 / 8, mode="bilinear", align_corners=False)
    try:
        import lpips as lpips_lib
        lp = lpips_lib.LPIPS(net="alex").to(DEV)
    except Exception:
        lp = None

    @torch.no_grad()
    def clip_emb(path: str) -> torch.Tensor:
        return F.normalize(clip_model.encode_image(clip_pre(Image.open(path).convert("RGB")).unsqueeze(0).to(DEV)))

    @torch.no_grad()
    def dino_emb(path: str) -> torch.Tensor:
        im = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = (x - 0.485) / 0.229  # approx imagenet norm
        return F.normalize(dino(x.to(DEV)))

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a @ b.T).item())

    for split in ("val", "test"):
        pairs = [json.loads(l) for l in open(SPLITS / f"{split}.jsonl")]
        for model in ("base", "final991"):
            gen_manifest = {}
            for l in open(root / f"gen/{model}/{split}/generation_manifest.jsonl"):
                r = json.loads(l)
                gen_manifest[(r["pair_id"], r["state"], str(r["seed"]))] = r["output_path"]
            out_rows = []
            for p in pairs:
                pid = p["pair_id"]
                gold_src, gold_tgt = p["source"]["image"], p["target"]["image"]
                mask = load_mask(p["automatic_difference_mask"]["path"])
                m_in, m_out = mask, ~mask
                row = {"pair_id": pid, "field": p["intervention"]["field"], "op": p["intervention"]["operation"],
                       "intervention_type": p["intervention_type"], "model": model, "split": split}
                acc = {k: [] for k in ("src_mse", "tgt_mse", "tgt_psnr", "tgt_ssim", "tgt_lpips",
                                        "clip_src", "clip_tgt", "clip_tgt_vs_src", "dino_tgt",
                                        "in_mse", "out_mse", "strength", "bg_drift")}
                for seed in args.seeds:
                    gs = gen_manifest[(pid, "source_state", seed)]
                    gt = gen_manifest[(pid, "target_state", seed)]
                    a_src, a_tgt = load_img(gs), load_img(gt)
                    g_src, g_tgt = load_img(gold_src), load_img(gold_tgt)
                    # pixel
                    acc["src_mse"].append(float(((a_src - g_src) ** 2).mean()))
                    tgt_mse = float(((a_tgt - g_tgt) ** 2).mean())
                    acc["tgt_mse"].append(tgt_mse)
                    acc["tgt_psnr"].append(psnr(tgt_mse))
                    acc["tgt_ssim"].append(ssim(a_tgt, g_tgt))
                    if lp is not None:
                        acc["tgt_lpips"].append(float(lp(to_tensor(a_tgt), to_tensor(g_tgt))))
                    # semantic
                    acc["clip_src"].append(cos(clip_emb(gs), clip_emb(gold_src)))
                    acc["clip_tgt"].append(cos(clip_emb(gt), clip_emb(gold_tgt)))
                    acc["clip_tgt_vs_src"].append(cos(clip_emb(gt), clip_emb(gold_src)))
                    acc["dino_tgt"].append(cos(dino_emb(gt), dino_emb(gold_tgt)))
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
                row["directional"] = row["clip_tgt"] - row["clip_tgt_vs_src"]
                out_rows.append(row)
            out = scores_dir / f"indomain_{model}_{split}.jsonl"
            with out.open("w") as f:
                for r in out_rows:
                    f.write(json.dumps(r) + "\n")
            print(f"[scored] {model}/{split}: {len(out_rows)} items -> {out}", flush=True)

    # paired stats
    METRICS = ["src_mse", "tgt_mse", "tgt_psnr", "tgt_ssim", "tgt_lpips", "clip_src", "clip_tgt",
               "dino_tgt", "in_mse", "out_mse", "strength", "bg_drift", "directional"]
    summary = {}
    for split in ("val", "test", "pooled"):
        base_rows = {}
        for s in ("val", "test"):
            if split != "pooled" and s != split:
                continue
            for l in open(scores_dir / f"indomain_base_{s}.jsonl"):
                r = json.loads(l)
                base_rows[r["pair_id"]] = r
        f_rows = {}
        for s in ("val", "test"):
            if split != "pooled" and s != split:
                continue
            for l in open(scores_dir / f"indomain_final991_{s}.jsonl"):
                r = json.loads(l)
                f_rows[r["pair_id"]] = r
        ids = sorted(set(base_rows) & set(f_rows))
        split_summary = {"n": len(ids)}
        for metric in METRICS:
            d = [f_rows[i][metric] - base_rows[i][metric] for i in ids]
            random.seed(0)
            boots = []
            for _ in range(args.bootstrap):
                smp = [d[random.randrange(len(d))] for _ in range(len(d))]
                boots.append(statistics.mean(smp))
            boots.sort()
            split_summary[metric] = {
                "base_mean": statistics.mean(base_rows[i][metric] for i in ids),
                "final991_mean": statistics.mean(f_rows[i][metric] for i in ids),
                "delta": statistics.mean(d),
                "ci_lo": boots[int(0.025 * len(boots))],
                "ci_hi": boots[int(0.975 * len(boots))],
                "win": sum(1 for x in d if x > 0), "tie": sum(1 for x in d if x == 0),
                "loss": sum(1 for x in d if x < 0),
            }
        summary[split] = split_summary
    (scores_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("[done] summary.json written")


if __name__ == "__main__":
    main()
