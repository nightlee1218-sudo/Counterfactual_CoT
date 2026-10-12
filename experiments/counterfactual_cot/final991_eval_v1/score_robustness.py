#!/usr/bin/env python3
"""Module C scoring: sampling-sensitivity robustness, base vs final991.

For configs main (A-module val, seed 1234), alt (50step/CFG 8.0/1.5), soft (25step/CFG 2.0/4.0)
on the same 16 val pairs, compute A-module metric family per pair and paired bootstrap stats.

Outputs:
  scores/robustness_rows.jsonl       per (config, model, pair) rows
  scores/robustness_summary.json     per-config paired stats + per-model config curve
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

CONFIG_DIRS = {
    "main": lambda root, m: root / f"gen/{m}/val",
    "alt": lambda root, m: root / f"gen/{m}/robustness/alt_sampling",
    "soft": lambda root, m: root / f"gen/{m}/robustness/soft_sampling",
}


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
        return float((((2 * mu_ab + C1) * (2 * cov + C2)) / ((mu_a2 + mu_b2 + C1) * (var_a + var_b + C2))).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    root = Path(args.root)
    scores_dir = root / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    clip_model, _, clip_pre = open_clip.create_model_and_transforms("ViT-g-14", pretrained="laion2b_s34b_b88k")
    clip_model = clip_model.eval().to(DEV)
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(DEV)
    try:
        import lpips as lpips_lib
        lp = lpips_lib.LPIPS(net="alex").to(DEV)
    except Exception:
        lp = None

    @torch.no_grad()
    def clip_emb(p: str) -> torch.Tensor:
        return F.normalize(clip_model.encode_image(clip_pre(Image.open(p).convert("RGB")).unsqueeze(0).to(DEV)))

    @torch.no_grad()
    def dino_emb(im_arr: np.ndarray) -> torch.Tensor:
        im = Image.fromarray(np.uint8(im_arr * 255.0)).convert("RGB").resize((224, 224), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return F.normalize(dino(((x - 0.485) / 0.229).to(DEV)))

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a @ b.T).item())

    pairs = [json.loads(l) for l in open(root.parent.parent / "final991_automask_v5_data/artifacts/splits/val.jsonl")][:16]
    pair_ids = {p["pair_id"] for p in pairs}

    rows = []
    all_rows = []
    for config, dirfn in CONFIG_DIRS.items():
        for model in ("base", "final991"):
            gen_dir = dirfn(root, model)
            gen_manifest = {}
            for l in open(gen_dir / "generation_manifest.jsonl"):
                r = json.loads(l)
                if str(r["seed"]) == "1234":
                    gen_manifest[(r["pair_id"], r["state"])] = r["output_path"]
            for p in pairs:
                pid = p["pair_id"]
                if (pid, "target_state") not in gen_manifest or (pid, "source_state") not in gen_manifest:
                    continue
                gt = gen_manifest[(pid, "target_state")]
                gs = gen_manifest[(pid, "source_state")]
                g_src, g_tgt = load_img(p["source"]["image"]), load_img(p["target"]["image"])
                a_src, a_tgt = load_img(gs), load_img(gt)
                mask = load_mask(p["automatic_difference_mask"]["path"])
                m_in, m_out = mask, ~mask
                row = {"config": config, "model": model, "pair_id": pid,
                       "intervention_type": p["intervention_type"]}
                row["src_mse"] = float(((a_src - g_src) ** 2).mean())
                tgt_mse = float(((a_tgt - g_tgt) ** 2).mean())
                row["tgt_mse"] = tgt_mse
                row["tgt_psnr"] = psnr(tgt_mse)
                row["tgt_ssim"] = ssim(a_tgt, g_tgt)
                if lp is not None:
                    row["tgt_lpips"] = float(lp(to_tensor(a_tgt), to_tensor(g_tgt)))
                row["clip_src"] = cos(clip_emb(gs), clip_emb(p["source"]["image"]))
                row["clip_tgt"] = cos(clip_emb(gt), clip_emb(p["target"]["image"]))
                row["clip_tgt_vs_src"] = cos(clip_emb(gt), clip_emb(p["source"]["image"]))
                row["dino_tgt"] = cos(dino_emb(a_tgt), dino_emb(g_tgt))
                sq = ((a_tgt - g_tgt) ** 2).mean(-1)
                row["in_mse"] = float(sq[m_in].mean()) if m_in.any() else 0.0
                row["out_mse"] = float(sq[m_out].mean()) if m_out.any() else 0.0
                row["bg_drift"] = float(((a_tgt - a_src) ** 2).mean(-1)[m_out].mean()) if m_out.any() else 0.0
                row["directional"] = row["clip_tgt"] - row["clip_tgt_vs_src"]
                rows.append(row)
                all_rows.append(row)
                print(f"[scored] {config} {model} {pid[:44]}", flush=True)

    with (scores_dir / "robustness_rows.jsonl").open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    METRICS = ["src_mse", "tgt_mse", "tgt_psnr", "tgt_ssim", "tgt_lpips", "clip_src", "clip_tgt",
               "dino_tgt", "in_mse", "out_mse", "bg_drift", "directional"]
    summary = {}
    for config in CONFIG_DIRS:
        base_rows = {r["pair_id"]: r for r in rows if r["config"] == config and r["model"] == "base"}
        f_rows = {r["pair_id"]: r for r in rows if r["config"] == config and r["model"] == "final991"}
        ids = sorted(set(base_rows) & set(f_rows))
        cs = {"n": len(ids)}
        for metric in METRICS:
            d = [f_rows[i][metric] - base_rows[i][metric] for i in ids]
            random.seed(0)
            boots = sorted(statistics.mean([d[random.randrange(len(d))] for _ in range(len(d))]) for _ in range(args.bootstrap))
            cs[metric] = {
                "base_mean": statistics.mean(base_rows[i][metric] for i in ids),
                "final991_mean": statistics.mean(f_rows[i][metric] for i in ids),
                "delta": statistics.mean(d),
                "ci_lo": boots[int(0.025 * len(boots))], "ci_hi": boots[int(0.975 * len(boots))],
                "win": sum(1 for x in d if x > 0), "tie": sum(1 for x in d if x == 0),
                "loss": sum(1 for x in d if x < 0),
            }
        summary[config] = cs
    # per-model config curve (sampling sensitivity of each model)
    curve = {}
    for model in ("base", "final991"):
        curve[model] = {}
        for config in CONFIG_DIRS:
            mrows = [r for r in rows if r["config"] == config and r["model"] == model]
            curve[model][config] = {metric: statistics.mean(r[metric] for r in mrows) for metric in
                                    ("tgt_ssim", "tgt_lpips", "clip_tgt", "dino_tgt", "bg_drift", "directional")}
    summary["config_curve"] = curve
    (scores_dir / "robustness_summary.json").write_text(json.dumps(summary, indent=1))
    print("[done] robustness_summary.json written", flush=True)


if __name__ == "__main__":
    main()
