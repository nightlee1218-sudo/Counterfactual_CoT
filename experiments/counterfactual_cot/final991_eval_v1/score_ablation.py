#!/usr/bin/env python3
"""Module B scoring: per-item paired metrics for base vs final991 prompt-profile ablation.

Same metric family as Module A (score_indomain.py), applied to the ablation outputs:
for every (pair_id, profile_id) we compare the generated edit against golden target
(and golden source for directionality), with mask decomposition from the pair's
automatic_difference_mask.

Outputs:
  scores/ablation_{model}.jsonl        per-item rows (51 pairs x 7 profiles)
  scores/ablation_summary.json         per-profile paired bootstrap CI, base vs final991
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

PROFILES = ["cot_core_v1", "full_without_target_cot_v1", "target_prompt_only_v1",
            "delta_only_v2", "full_paired_v2", "legacy_paired_v1", "wrong_cot_v1"]


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
        ssim_map = ((2 * mu_ab + C1) * (2 * cov + C2)) / ((mu_a2 + mu_b2 + C1) * (var_a + var_b + C2))
        return float(ssim_map.mean())


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
    def clip_emb(path: str) -> torch.Tensor:
        return F.normalize(clip_model.encode_image(clip_pre(Image.open(path).convert("RGB")).unsqueeze(0).to(DEV)))

    @torch.no_grad()
    def dino_emb(path: str) -> torch.Tensor:
        im = Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = (x - 0.485) / 0.229
        return F.normalize(dino(x.to(DEV)))

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a @ b.T).item())

    pairs = [json.loads(l) for l in open(SPLITS / "val.jsonl")]
    out_rows = {}
    for model in ("base", "final991"):
        gen_manifest = {}
        for l in open(root / f"gen/{model}/ablation/generation_manifest.jsonl"):
            r = json.loads(l)
            gen_manifest[(r["pair_id"], r["profile_id"])] = r["output_path"]
        if len(gen_manifest) < len(pairs) * len(PROFILES):
            print(f"[warn] {model}: manifest has {len(gen_manifest)} entries, expected {len(pairs) * len(PROFILES)}", flush=True)
        rows = []
        for p in pairs:
            pid = p["pair_id"]
            gold_src, gold_tgt = p["source"]["image"], p["target"]["image"]
            mask = load_mask(p["automatic_difference_mask"]["path"])
            m_in, m_out = mask, ~mask
            for profile_id in PROFILES:
                key = (pid, profile_id)
                if key not in gen_manifest:
                    continue
                gp = gen_manifest[key]
                a_tgt = load_img(gp)
                g_src, g_tgt = load_img(gold_src), load_img(gold_tgt)
                row = {"pair_id": pid, "profile_id": profile_id, "model": model,
                       "intervention_type": p["intervention_type"]}
                tgt_mse = float(((a_tgt - g_tgt) ** 2).mean())
                row["tgt_mse"] = tgt_mse
                row["tgt_psnr"] = psnr(tgt_mse)
                row["tgt_ssim"] = ssim(a_tgt, g_tgt)
                if lp is not None:
                    row["tgt_lpips"] = float(lp(to_tensor(a_tgt), to_tensor(g_tgt)))
                row["clip_tgt"] = cos(clip_emb(gp), clip_emb(gold_tgt))
                row["clip_tgt_vs_src"] = cos(clip_emb(gp), clip_emb(gold_src))
                row["dino_tgt"] = cos(dino_emb(gp), dino_emb(gold_tgt))
                sq = ((a_tgt - g_tgt) ** 2).mean(-1)
                row["in_mse"] = float(sq[m_in].mean()) if m_in.any() else 0.0
                row["out_mse"] = float(sq[m_out].mean()) if m_out.any() else 0.0
                gold_change = float(((g_tgt - g_src) ** 2).mean(-1)[m_in].mean()) if m_in.any() else 1e-8
                row["bg_drift"] = float(((a_tgt - g_src) ** 2).mean(-1)[m_out].mean()) if m_out.any() else 0.0
                row["directional"] = row["clip_tgt"] - row["clip_tgt_vs_src"]
                rows.append(row)
                print(f"[scored] {model} {pid[:48]} {profile_id}", flush=True)
        out = scores_dir / f"ablation_{model}.jsonl"
        with out.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[done] {model}: {len(rows)} items -> {out}", flush=True)
        out_rows[model] = {(r["pair_id"], r["profile_id"]): r for r in rows}

    # paired stats per profile
    METRICS = ["tgt_mse", "tgt_psnr", "tgt_ssim", "tgt_lpips", "clip_tgt",
               "dino_tgt", "in_mse", "out_mse", "bg_drift", "directional"]
    summary = {}
    base_rows, f_rows = out_rows["base"], out_rows["final991"]
    ids = sorted(set(base_rows) & set(f_rows))
    summary["n_total"] = len(ids)
    for profile_id in PROFILES:
        pid_ids = [k for k in ids if k[1] == profile_id]
        prof_summary = {"n": len(pid_ids)}
        for metric in METRICS:
            d = [f_rows[i][metric] - base_rows[i][metric] for i in pid_ids]
            if not d:
                continue
            random.seed(0)
            boots = []
            for _ in range(args.bootstrap):
                smp = [d[random.randrange(len(d))] for _ in range(len(d))]
                boots.append(statistics.mean(smp))
            boots.sort()
            prof_summary[metric] = {
                "base_mean": statistics.mean(base_rows[i][metric] for i in pid_ids),
                "final991_mean": statistics.mean(f_rows[i][metric] for i in pid_ids),
                "delta": statistics.mean(d),
                "ci_lo": boots[int(0.025 * len(boots))],
                "ci_hi": boots[int(0.975 * len(boots))],
                "win": sum(1 for x in d if x > 0), "tie": sum(1 for x in d if x == 0),
                "loss": sum(1 for x in d if x < 0),
            }
        summary[profile_id] = prof_summary
    (scores_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=1))
    print("[done] ablation_summary.json written", flush=True)


if __name__ == "__main__":
    main()
