#!/usr/bin/env python3
"""E1 CLIP-T scoring: text-image alignment (CLIPScore) for base vs final991.

Model: OpenAI CLIP ViT-L/14 loaded from local weights
(/group/40016/niightli/univfd/clip_cache/ViT-L-14.pt) via open_clip — no download.
CLIPScore = 2.5 * max(cos(prompt_emb, image_emb), 0)  (Hessel et al., 2021).
Raw cosine is also reported.

Output:
  scores/clipT_{model}.jsonl        per-item rows
  scores/clipT_summary.json         quick aggregates (final stats in stats_paired.py)

Run with the open_clip-capable python, e.g.
  /group/40016/niightli/BAGEL/venvs/geneval_eval/bin/python score_clip.py --gen-root ../outputs/e1_general_t2i/gen
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = Path("/group/40016/niightli/univfd/clip_cache/ViT-L-14.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-root", type=Path, default=HERE / "../outputs/e1_general_t2i/gen")
    parser.add_argument("--out-dir", type=Path, default=HERE / "../outputs/e1_general_t2i/scores")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--models", nargs="*", default=["base", "final991"])
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import open_clip

    print(json.dumps({"event": "clip_load_start", "weights": str(args.weights), "device": dev}), flush=True)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=str(args.weights))
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model = model.eval().to(dev)
    print(json.dumps({"event": "clip_load_done"}), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"model": "CLIP ViT-L/14 (OpenAI weights)", "clipscore_w": 2.5}

    for model_name in args.models:
        manifest = args.gen_root / model_name / "generation_manifest.jsonl"
        if not manifest.exists():
            print(json.dumps({"event": "skip_missing_manifest", "model": model_name, "path": str(manifest)}), flush=True)
            continue
        rows = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            img_path = Path(rec["output_path"])
            if not img_path.exists():
                print(json.dumps({"event": "missing_image", "item_id": rec["item_id"]}), flush=True)
                continue
            with torch.no_grad():
                image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(dev)
                text = tokenizer([rec["prompt"]]).to(dev)
                img_emb = F.normalize(model.encode_image(image), dim=-1)
                txt_emb = F.normalize(model.encode_text(text), dim=-1)
                cos = float((img_emb @ txt_emb.T).item())
            rows.append({
                "item_id": rec["item_id"],
                "subset": rec["subset"],
                "source_category": rec["source_category"],
                "model": model_name,
                "clip_cos": cos,
                "clipscore": 2.5 * max(cos, 0.0),
            })
            if len(rows) % 50 == 0:
                print(json.dumps({"event": "progress", "model": model_name, "done": len(rows)}), flush=True)

        out = args.out_dir / f"clipT_{model_name}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        by_subset: dict[str, list[float]] = {}
        for r in rows:
            by_subset.setdefault(r["subset"], []).append(r["clip_cos"])
        summary[model_name] = {
            "n": len(rows),
            "clip_cos_mean": statistics.mean([r["clip_cos"] for r in rows]) if rows else None,
            "clipscore_mean": statistics.mean([r["clipscore"] for r in rows]) if rows else None,
            "by_subset": {k: statistics.mean(v) for k, v in sorted(by_subset.items())},
        }
        print(json.dumps({"event": "scored", "model": model_name, "n": len(rows),
                          "clip_cos_mean": summary[model_name]["clip_cos_mean"]}), flush=True)

    (args.out_dir / "clipT_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "all_done"}), flush=True)


if __name__ == "__main__":
    main()
