#!/usr/bin/env python3
"""Convert the OpenAI CLIP ViT-L/14 TorchScript archive to a plain state_dict.

Why: torch>=2.6 defaults torch.load(weights_only=True), which cannot load
TorchScript archives, so open_clip.create_model_and_transforms("ViT-L-14",
pretrained=<jit path>) crashes at model build (RuntimeError about
weights_only=True + TorchScript). open_clip's load_state_dict DOES understand
ScriptModule state dicts (it handles torch.jit.ScriptModule explicitly), so
extracting the state_dict once and saving it as a plain tensor dict is
sufficient. The original JIT archive is NOT modified.

The converted file is numerically verified against the original JIT model on
the same random input: open_clip is built with force_quick_gelu=True (OpenAI
CLIP was trained with QuickGELU) and image/text logits must match the JIT
model's closely (fp16 JIT vs fp32 open_clip allows small tolerance).

Usage:
  /group/40016/niightli/BAGEL/venvs/geneval_eval/bin/python convert_clip_jit_to_statedict.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SRC = Path("/group/40016/niightli/univfd/clip_cache/ViT-L-14.pt")
DST = Path("/group/40016/niightli/univfd/clip_cache/ViT-L-14_state_dict.pt")
META_KEYS = ("input_resolution", "context_length", "vocab_size")


def main() -> None:
    assert SRC.exists(), f"missing {SRC}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) extract state_dict from the TorchScript archive (original untouched)
    jit = torch.jit.load(str(SRC), map_location=dev).eval()
    sd = jit.state_dict()
    meta = {}
    for key in META_KEYS:
        if key in sd:
            t = sd.pop(key)
            meta[key] = t.item() if t.numel() == 1 else t.tolist()
    torch.save(sd, DST)

    # 2) verify the new file reloads under torch>=2.6 default weights_only=True
    reloaded = torch.load(DST, map_location="cpu", weights_only=True)
    assert isinstance(reloaded, dict) and set(reloaded) == set(sd), "reload key mismatch"
    print(json.dumps({"event": "converted", "src": str(SRC), "dst": str(DST),
                      "num_tensors": len(sd), "meta": meta,
                      "size_bytes": DST.stat().st_size}), flush=True)

    # 3) end-to-end numeric verification vs the original JIT model
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=str(DST), force_quick_gelu=True)
    model = model.eval().to(dev)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    torch.manual_seed(0)
    img = torch.randn(2, 3, 224, 224, device=dev)
    txt = tokenizer(["a photo of a cat", "a red bus on a street"]).to(dev)
    with torch.no_grad():
        logits_jit, _ = jit(img, txt)
        i_emb = F.normalize(model.encode_image(img), dim=-1)
        t_emb = F.normalize(model.encode_text(txt), dim=-1)
        logits_oc = i_emb @ t_emb.T * model.logit_scale.exp()
    max_abs = float((logits_jit.float() - logits_oc).abs().max())
    cos = float(F.cosine_similarity(logits_jit.float().flatten(),
                                    logits_oc.flatten(), dim=0))
    passed = cos > 0.999 and max_abs < 0.5
    print(json.dumps({"event": "verify", "max_abs_logit_diff": max_abs,
                      "logit_cosine": cos, "pass": passed}), flush=True)
    if not passed:
        print("VERIFICATION FAILED: converted weights do not reproduce the JIT model", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({"event": "all_done", "dst": str(DST)}), flush=True)


if __name__ == "__main__":
    main()
