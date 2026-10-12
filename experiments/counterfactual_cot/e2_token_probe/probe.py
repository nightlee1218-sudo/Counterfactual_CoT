#!/usr/bin/env python3
"""E2 token-localization probe: does the model bind "changed tokens" to the edit region?

Hook-point analysis (verified against BAGEL source, see README):

1. Generation for [source_image, prompt] runs as:
   - source-image prefill: VAE latents (gen expert, is_causal=False) then ViT
     patches (und expert, is_causal=False) enter the KV cache;
   - prompt prefill: text tokens are QUERIES (und expert, is_causal=True) and
     attend to all cached source-image keys -> the *changed token -> source
     image* direction EXISTS here (P1 metrics);
   - denoising flow steps: the query sequence is only [SOI, N latents, EOI]
     (gen expert, is_causal=False); the prompt lives in the KV cache, so the
     *target latent -> changed token* direction exists (attend-and-segment
     direction, P2 metrics); changed tokens are never queries during
     denoising (architectural fact).
2. All inference attention goes through flash_attn_varlen_func which does not
   return weights. The probe monkeypatches PackedAttentionMoT.forward_inference:
   for capture-targeted calls it re-executes the identical projection/norm/RoPE
   code and computes attention manually (chunked softmax, numerically
   equivalent), extracting only the needed rows/columns; ALL other calls
   (non-probe steps, CFG passes, image prefills) delegate to the original
   flash path untouched. Capture targeting uses KV-cache object identity, so
   the conditional pass is distinguished from cfg_text/cfg_img passes exactly.

Metrics per layer (0..L-1), per pair, per model:
  P1 prefill attention mass (head-mean over changed-token queries):
     mass on source-VAE in-mask keys / out-mask keys, ViT in/out,
     normalized density ratio (in/out), mass vs uniform baseline.
  P2 flow attention (per probe step, head-mean):
     per-latent-position attention on changed-token keys -> log2 in/out
     density contrast, in-mask mass fraction, AUROC vs mask labels.
  P3 hidden-state contrast margin:
     cos(h_tok, mu_in) - cos(h_tok, mu_out) for source-VAE / ViT / flow latent.

Usage (GPU machine, one model per GPU):
  CUDA_VISIBLE_DEVICES=0 python3 probe.py --model base     --out-dir ../outputs/e2_token_probe/probe_base
  CUDA_VISIBLE_DEVICES=1 python3 probe.py --model final991 --out-dir ../outputs/e2_token_probe/probe_final991
  smoke: --limit 1 --probe-steps 11
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

BAGEL_ROOT = Path("/group/40016/niightli/BAGEL")
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.gold40_automask_dual_target_cot1024_fsdp_v5.generate_same_seed_states import (  # noqa: E402
    build_inference_model,
)
from experiments.counterfactual_cot.gold40_automask_dual_target_cot1024_fsdp_v5.paired_data import (  # noqa: E402
    load_automask_manifest,
    render_state_prompt,
)

HERE = Path(__file__).resolve().parent
CFCOT = HERE.parent
DEFAULT_MODEL_DIR = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_CKPT = CFCOT / "outputs/final991_automask_v5_train2973/checkpoints/latest.pt"
DEFAULT_VAL = CFCOT / "final991_automask_v5_data/artifacts/splits/val.jsonl"

MODELS = {
    "base": {"trainable_checkpoint": None, "skip_trainable_reload": True},
    "final991": {"trainable_checkpoint": DEFAULT_CKPT, "skip_trainable_reload": False},
}

GEN_KWARGS = dict(num_timesteps=25, cfg_text=4.0, cfg_img=2.0, shift=3.0,
                  cfg_interval=(0.0, 1.0), image_size=1024, seed=1234)
VIT_RESIZE = 980   # vit_transform = ImageTransform(980, 224, 14)
VIT_PATCH = 14

# ---------------------------------------------------------------------------
# capture state (module-global; single-threaded probing)
# ---------------------------------------------------------------------------
CAPTURE: dict[str, Any] = {
    "phase": None,           # img_prefill_vae | img_prefill_vit | text_prefill | flow
    "main_cache": None,      # identity of the KV cache whose passes we capture
    "step_idx": -1,          # conditional-pass counter within generate_image
    "probe_steps": set(),
    "text": {}, "flow": {}, "img": {},
    "store": None,
    "_text_pass": False,     # marker: current text-prefill is the capture target
    "_flow_capture": False,  # marker: current flow pass is a capture target
}


def new_store() -> dict[str, Any]:
    return {
        "prefill_rows": {},   # layer -> {local_row -> {head -> probs(np)}}
        "h_tok": {},          # layer -> (n_ch, D) fp32
        "mu": {},             # ("vae"|"vit", layer) or ("lat", layer, step) -> (mu_in, mu_out)
        "flow_maps": {},      # (layer, step) -> (H, n_lat) attention on changed tokens
    }


# ---------------------------------------------------------------------------
# token span locating
# ---------------------------------------------------------------------------
def find_changed_occurrences(prompt: str, new_value: Any) -> list[tuple[int, int]]:
    """Occurrences of the intervention new value in the rendered prompt.

    Preference: occurrences right after '=' or ':' (Target CoT assignments,
    e.g. ``vase_1.object=pen-and-note caddy``); fallback: all word-boundary
    occurrences; last resort: plain substring.
    """
    value = str(new_value).strip()
    if not value:
        return []
    pattern = r"(?<![0-9A-Za-z])" + re.escape(value) + r"(?![0-9A-Za-z])"
    matches = [(m.start(), m.end()) for m in re.finditer(pattern, prompt)]
    if not matches:
        idx = prompt.find(value)
        return [(idx, idx + len(value))] if idx >= 0 else []
    anchored = [(s, e) for (s, e) in matches if s > 0 and prompt[s - 1] in "=:."]
    return anchored or matches


def span_to_token_range(tokenizer, prompt: str, start: int, end: int,
                        cache: dict[int, int]) -> tuple[int, int]:
    """Char span -> [first_tok, last_tok) within tokenizer.encode(prompt).

    Prefix re-encoding keeps this correct for slow (non-offset) tokenizers.
    """
    for cut in (start, end):
        if cut not in cache:
            cache[cut] = len(tokenizer.encode(prompt[:cut]))
    return cache[start], cache[end]


# ---------------------------------------------------------------------------
# mask -> grid mapping (BAGEL grids are row-major: token k <-> (k // w, k % w))
# ---------------------------------------------------------------------------
def mask_grid_fractions(mask_path: Path, grid_hw: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    if mask.shape[:2] != (1024, 1024):
        mask = np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8)).resize((1024, 1024), Image.NEAREST),
            dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
    h, w = grid_hw
    t = torch.from_numpy(mask)[None, None]
    return F.interpolate(t, size=(h, w), mode="area")[0, 0].numpy()


def in_mask_flags(frac: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (frac >= threshold).reshape(-1)


# ---------------------------------------------------------------------------
# manual attention (numerically equivalent to flash varlen, single sample)
# ---------------------------------------------------------------------------
def manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     scale: float, causal: bool, kv_len: int,
                     keep_rows: list[int], keep_cols: list[int],
                     block: int = 1024):
    """Chunked softmax attention for ONE varlen sample (GQA-aware).

    q: (n_q, H, d) bf16; k/v: (n_k, H_kv, d) bf16.
    causal=True replicates flash semantics: query i (0-based within the query
    block) attends keys j in [0, kv_len + i + 1).
    Returns (output (n_q, H, d) bf16,
             row_probs {local_row -> {head -> np.ndarray(n_k)}},
             col_probs per head list of np.ndarray (n_q, len(keep_cols))).
    """
    n_q, H, d = q.shape
    n_k = k.shape[0]
    ng = H // k.shape[1]
    out = torch.empty((n_q, H, d), dtype=v.dtype, device=q.device)
    row_probs: dict[int, dict[int, np.ndarray]] = {}
    col_probs: list[np.ndarray] = []
    row_set = set(int(r) for r in keep_rows)
    col_idx = torch.tensor(sorted(set(int(c) for c in keep_cols)), device=q.device,
                            dtype=torch.long) if keep_cols else None
    key_range = torch.arange(n_k, device=q.device)
    for h in range(H):
        qh = q[:, h, :].float()
        kh = k[:, h // ng, :].float()
        vh = v[:, h // ng, :]
        col_parts = []
        for s in range(0, n_q, block):
            e = min(s + block, n_q)
            scores = (qh[s:e] @ kh.T).float() * scale
            if causal:
                qpos = torch.arange(s, e, device=q.device)
                allowed = key_range[None, :] <= (kv_len + qpos)[:, None]
                scores = scores.masked_fill(~allowed, float("-inf"))
            row_max = scores.max(dim=-1, keepdim=True).values
            exp_s = torch.exp(scores - row_max)
            denom = exp_s.sum(dim=-1, keepdim=True)
            probs = (exp_s / denom).float()
            if row_set:
                for li in range(s, e):
                    if li in row_set:
                        row_probs.setdefault(li, {})[h] = probs[li - s].detach().float().cpu().numpy()
            if col_idx is not None:
                col_parts.append(probs[:, col_idx].detach().float().cpu().numpy())
            attn_v = (probs.to(v.dtype) @ vh)
            out[s:e, h, :] = attn_v
        if col_idx is not None:
            col_probs.append(np.concatenate(col_parts, axis=0))
    return out, row_probs, col_probs


# ---------------------------------------------------------------------------
# attention monkeypatch
# ---------------------------------------------------------------------------
_ORIG_ATTN_FORWARD = None


def _forward_manual(self, packed_query_sequence, query_lens,
                    packed_query_position_embeddings, packed_query_indexes,
                    past_key_values=None, key_values_lens=None,
                    packed_key_value_indexes=None, update_past_key_values=True,
                    is_causal=True, mode="und", packed_vae_token_indexes=None,
                    packed_text_indexes=None):
    """Replica of PackedAttentionMoT.forward_inference with manual attention.

    Projection/norm/RoPE/cache-merge mirror qwen2_navit.py line-by-line; the
    flash call is replaced by ``manual_attention``. ``_flow_capture`` /
    ``_text_pass`` markers decide whether extracts are stashed.
    """
    from modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    if mode == "und":
        packed_query_states = self.q_proj(packed_query_sequence).view(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_query_sequence).view(-1, self.num_key_value_heads, self.head_dim)
        packed_query_states = self.q_norm(packed_query_states)
        packed_key_states = self.k_norm(packed_key_states)
    elif mode == "gen":
        packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
        packed_query_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_heads * self.head_dim))
        packed_key_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))
        packed_value_states = packed_query_sequence.new_zeros((packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim))
        text_q = packed_query_sequence[packed_text_indexes]
        vae_q = packed_query_sequence[packed_vae_token_indexes]
        packed_query_states[packed_text_indexes] = self.q_proj(text_q)
        packed_query_states[packed_vae_token_indexes] = self.q_proj_moe_gen(vae_q)
        packed_key_states[packed_text_indexes] = self.k_proj(text_q)
        packed_key_states[packed_vae_token_indexes] = self.k_proj_moe_gen(vae_q)
        packed_value_states[packed_text_indexes] = self.v_proj(text_q)
        packed_value_states[packed_vae_token_indexes] = self.v_proj_moe_gen(vae_q)
        packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim).to(torch.float32)
        packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim).to(torch.float32)
        packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
        packed_query_states[packed_text_indexes] = self.q_norm(packed_query_states[packed_text_indexes])
        packed_query_states[packed_vae_token_indexes] = self.q_norm_moe_gen(packed_query_states[packed_vae_token_indexes])
        packed_key_states[packed_text_indexes] = self.k_norm(packed_key_states[packed_text_indexes])
        packed_key_states[packed_vae_token_indexes] = self.k_norm_moe_gen(packed_key_states[packed_vae_token_indexes])
    else:
        raise ValueError(mode)

    packed_cos, packed_sin = packed_query_position_embeddings
    packed_query_states, packed_key_states = apply_rotary_pos_emb(
        packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1)
    packed_query_states = packed_query_states.to(torch.bfloat16)
    packed_key_states = packed_key_states.to(torch.bfloat16)
    packed_value_states = packed_value_states.to(torch.bfloat16)

    has_past = (past_key_values is not None
                and past_key_values.key_cache[self.layer_idx] is not None)
    kv_len = int(sum(key_values_lens).item()) if has_past else 0

    if has_past:
        past_key_states = past_key_values.key_cache[self.layer_idx]
        past_value_states = past_key_values.value_cache[self.layer_idx]
        seqlens = sum(query_lens) + sum(key_values_lens)
        merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
        merged_value_states = past_value_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
        merged_key_states[packed_query_indexes] = packed_key_states
        merged_key_states[packed_key_value_indexes] = past_key_states
        merged_value_states[packed_query_indexes] = packed_value_states
        merged_value_states[packed_key_value_indexes] = past_value_states
    else:
        merged_key_states = packed_key_states
        merged_value_states = packed_value_states

    capture_text = CAPTURE["_text_pass"] and CAPTURE["phase"] == "text_prefill"
    capture_flow = CAPTURE["_flow_capture"] and CAPTURE["phase"] == "flow"
    keep_rows = list(CAPTURE["text"]["changed_rows_local"]) if capture_text else []
    keep_cols = list(CAPTURE["flow"]["changed_key_cols"]) if capture_flow else []

    packed_attn_output, row_probs, col_probs = manual_attention(
        packed_query_states, merged_key_states, merged_value_states,
        float(self.head_dim ** -0.5), bool(is_causal), kv_len,
        keep_rows, keep_cols)

    packed_attn_output = packed_attn_output.reshape(-1, self.num_heads * self.head_dim)
    if mode == "und":
        packed_attn_output = self.o_proj(packed_attn_output)
    else:
        packed_attn_output = packed_attn_output.clone()
        packed_attn_output[packed_text_indexes] = self.o_proj(packed_attn_output[packed_text_indexes])
        packed_attn_output[packed_vae_token_indexes] = self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes])

    if update_past_key_values:
        past_key_values.key_cache[self.layer_idx] = merged_key_states
        past_key_values.value_cache[self.layer_idx] = merged_value_states

    store = CAPTURE["store"]
    if store is not None:
        if capture_text and row_probs:
            store["prefill_rows"][self.layer_idx] = row_probs
        if capture_flow and col_probs:
            step = CAPTURE["step_idx"]
            maps = np.stack(col_probs, axis=0)         # (H, n_q, n_ch)
            store["flow_maps"][(self.layer_idx, step)] = maps[:, 1:-1, :].sum(-1)  # drop SOI/EOI rows
    return packed_attn_output, past_key_values


def _patched_attn_forward(self, *args, **kwargs):
    """Routing wrapper: capture-targeted calls -> manual; everything -> flash."""
    global _ORIG_ATTN_FORWARD
    phase = CAPTURE["phase"]
    if phase is None:
        return _ORIG_ATTN_FORWARD(self, *args, **kwargs)
    pv = kwargs.get("past_key_values")
    main = CAPTURE["main_cache"]
    if phase in ("img_prefill_vae", "img_prefill_vit"):
        # hidden-state hooks need no attention capture -> original path
        return _ORIG_ATTN_FORWARD(self, *args, **kwargs)
    if phase == "text_prefill":
        CAPTURE["_text_pass"] = (pv is main)
        if CAPTURE["_text_pass"]:
            return _forward_manual(self, *args, **kwargs)
        return _ORIG_ATTN_FORWARD(self, *args, **kwargs)
    if phase == "flow":
        if pv is main:
            CAPTURE["step_idx"] += 1
            CAPTURE["_flow_capture"] = CAPTURE["step_idx"] in CAPTURE["probe_steps"]
            if CAPTURE["_flow_capture"]:
                CAPTURE["_flow_layer_marker"] = self.layer_idx
                return _forward_manual(self, *args, **kwargs)
        else:
            CAPTURE["_flow_capture"] = False
        return _ORIG_ATTN_FORWARD(self, *args, **kwargs)
    return _ORIG_ATTN_FORWARD(self, *args, **kwargs)


def install_attention_patch() -> None:
    global _ORIG_ATTN_FORWARD
    from modeling.bagel.qwen2_navit import PackedAttentionMoT
    if getattr(PackedAttentionMoT, "_e2_patched", False):
        return
    _ORIG_ATTN_FORWARD = PackedAttentionMoT.forward_inference
    PackedAttentionMoT.forward_inference = _patched_attn_forward
    PackedAttentionMoT._e2_patched = True
    PackedAttentionMoT._e2_orig_forward_inference = _ORIG_ATTN_FORWARD


# ---------------------------------------------------------------------------
# hidden-state hooks (layer outputs; query rows only)
# ---------------------------------------------------------------------------
_HIDDEN_HOOKS: list[Any] = []


def _make_hidden_hook(layer_idx: int):
    def hook(module, args, output):
        phase = CAPTURE["phase"]
        store = CAPTURE["store"]
        if phase is None or store is None:
            return
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden is None:
            return
        if phase == "text_prefill" and CAPTURE.get("_text_pass"):
            rows = CAPTURE["text"]["changed_rows_local"]
            if rows:
                store["h_tok"][layer_idx] = hidden[rows].float().cpu().numpy()
        elif phase == "img_prefill_vae":
            rows, flags = CAPTURE["img"]["vae_rows"], CAPTURE["img"]["vae_in"]
            if rows:
                h = hidden[rows].float()
                f = torch.from_numpy(flags.astype(np.float32)).to(h.device)
                store["mu"][("vae", layer_idx)] = (
                    h[f > 0.5].mean(0).cpu().numpy(), h[f <= 0.5].mean(0).cpu().numpy())
        elif phase == "img_prefill_vit":
            rows, flags = CAPTURE["img"]["vit_rows"], CAPTURE["img"]["vit_in"]
            if rows:
                h = hidden[rows].float()
                f = torch.from_numpy(flags.astype(np.float32)).to(h.device)
                store["mu"][("vit", layer_idx)] = (
                    h[f > 0.5].mean(0).cpu().numpy(), h[f <= 0.5].mean(0).cpu().numpy())
        elif phase == "flow":
            if CAPTURE.get("_flow_layer_marker") == layer_idx and CAPTURE.get("_flow_capture"):
                rows, flags = CAPTURE["flow"]["latent_rows"], CAPTURE["flow"]["latent_in"]
                step = CAPTURE["step_idx"]
                if rows:
                    h = hidden[rows].float()
                    f = torch.from_numpy(flags.astype(np.float32)).to(h.device)
                    store["mu"][("lat", layer_idx, step)] = (
                        h[f > 0.5].mean(0).cpu().numpy(), h[f <= 0.5].mean(0).cpu().numpy())
                CAPTURE["_flow_layer_marker"] = None
    return hook


def install_hidden_hooks(model) -> None:
    for i, layer in enumerate(model.language_model.model.layers):
        _HIDDEN_HOOKS.append(layer.register_forward_hook(_make_hidden_hook(i)))


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels.astype(bool)
    neg = ~pos
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize_prefill(store: dict, geom: dict) -> dict[int, dict]:
    """P1: changed-token query rows -> source image key masses, per layer."""
    out: dict[int, dict] = {}
    n_vae, n_vit = geom["n_vae_keys"], geom["n_vit_keys"]
    vae_in, vit_in = geom["vae_key_in"], geom["vit_key_in"]
    for layer, rows in store["prefill_rows"].items():
        agg = {k: [] for k in ("vae_in", "vae_out", "vit_in", "vit_out", "text", "visual")}
        for _row, head_rows in rows.items():
            for _head, probs in head_rows.items():
                v, t, rest = probs[:n_vae], probs[n_vae:n_vae + n_vit], probs[n_vae + n_vit:]
                agg["vae_in"].append(v[vae_in].sum()); agg["vae_out"].append(v[~vae_in].sum())
                agg["vit_in"].append(t[vit_in].sum()); agg["vit_out"].append(t[~vit_in].sum())
                agg["text"].append(rest.sum())
                agg["visual"].append(v.sum() + t.sum())
        m = {f"mass_{k}": float(np.mean(v)) for k, v in agg.items() if v}
        n_vi, n_vo = int(vae_in.sum()), int((~vae_in).sum())
        n_ti, n_to = int(vit_in.sum()), int((~vit_in).sum())
        m.update(n_vae_in=n_vi, n_vae_out=n_vo, n_vit_in=n_ti, n_vit_out=n_to)
        if n_vi and n_vo:
            m["norm_ratio_vae"] = (m["mass_vae_in"] / n_vi) / (m["mass_vae_out"] / n_vo)
            m["vs_uniform_vae"] = m["mass_vae_in"] / (n_vi / n_vae)
        if n_ti and n_to:
            m["norm_ratio_vit"] = (m["mass_vit_in"] / n_ti) / (m["mass_vit_out"] / n_to)
            m["vs_uniform_vit"] = m["mass_vit_in"] / (n_ti / n_vit)
        vis_i, vis_o = n_vi + n_ti, n_vo + n_to
        if vis_i and vis_o:
            m["norm_ratio_visual"] = ((m["mass_vae_in"] + m["mass_vit_in"]) / vis_i) / (
                (m["mass_vae_out"] + m["mass_vit_out"]) / vis_o)
        out[int(layer)] = m
    return out


def summarize_flow(store: dict, lat_in: np.ndarray) -> dict[int, dict]:
    """P2: latent-position attention on changed tokens, per layer (mean steps)."""
    n_lat = len(lat_in)
    n_in, n_out = int(lat_in.sum()), int((~lat_in).sum())
    by_layer: dict[int, list[dict]] = {}
    for (layer, _step), maps in store["flow_maps"].items():
        head_mean = maps.mean(axis=0)      # (n_lat,)
        rec: dict[str, Any] = {"map": head_mean}
        if n_in and n_out:
            d_in, d_out = head_mean[lat_in].mean(), head_mean[~lat_in].mean()
            rec["log2_contrast"] = float(np.log2((d_in + 1e-12) / (d_out + 1e-12)))
        total = head_mean.sum()
        rec["mass_in_frac"] = float(head_mean[lat_in].sum() / total) if total > 0 else 0.0
        rec["auroc"] = auroc(head_mean, lat_in.astype(np.int8))
        by_layer.setdefault(int(layer), []).append(rec)
    out = {}
    for layer, recs in by_layer.items():
        keys = ("log2_contrast", "mass_in_frac", "auroc")
        out[layer] = {k: float(np.nanmean([r[k] for r in recs])) for k in keys}
        out[layer]["n_steps"] = len(recs)
    return out


def hidden_margins(store: dict) -> dict[str, dict[int, float]]:
    """P3: cos(h_tok_mean, mu_in) - cos(h_tok_mean, mu_out)."""
    margins: dict[str, dict[int, float]] = {"vae": {}, "vit": {}, "lat": {}}
    h_mean = {L: h.mean(0) for L, h in store["h_tok"].items()}

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    for key, (mu_in, mu_out) in store["mu"].items():
        if len(key) == 3:
            continue  # ("lat", layer, step) handled below
        kind, layer = key
        if layer not in h_mean:
            continue
        margins[kind][layer] = cos(h_mean[layer], mu_in) - cos(h_mean[layer], mu_out)
    lat_by_layer: dict[int, list] = {}
    for key, (mu_in, mu_out) in store["mu"].items():
        if len(key) == 3 and key[0] == "lat":
            lat_by_layer.setdefault(key[1], []).append((mu_in, mu_out))
    for layer, pairs in lat_by_layer.items():
        if layer in h_mean:
            margins["lat"][layer] = float(np.mean(
                [cos(h_mean[layer], a) - cos(h_mean[layer], b) for a, b in pairs]))
    return margins


# ---------------------------------------------------------------------------
# per-pair driver
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def probe_pair(inferencer, tokenizer, record: dict, args, num_layers: int) -> dict:
    import copy as _copy
    from data.data_utils import pil_img2rgb

    pair_id = record["pair_id"]
    prompt = render_state_prompt(record, "target_state")
    new_value = record["intervention"].get("new_value")
    occurrences = find_changed_occurrences(prompt, new_value)
    if not occurrences:
        return {"pair_id": pair_id, "status": "span_not_found"}

    tok_cache: dict[int, int] = {}
    spans = [span_to_token_range(tokenizer, prompt, s, e, tok_cache) for (s, e) in occurrences]
    uniq: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if not uniq or a >= uniq[-1][1]:
            uniq.append((a, b))
    if not uniq:
        return {"pair_id": pair_id, "status": "span_empty"}

    mask_path = Path(record["automatic_difference_mask"]["path"])
    src_path = Path(record["source"]["image"])
    if not mask_path.exists() or not src_path.exists():
        return {"pair_id": pair_id, "status": "files_missing"}

    model = inferencer.model
    lat_side = args.image_size // model.latent_downsample          # 1024/16 = 64
    vit_side = VIT_RESIZE // VIT_PATCH                              # 980/14  = 70
    n_lat = lat_side * lat_side
    n_vit = vit_side * vit_side

    lat_in = in_mask_flags(mask_grid_fractions(mask_path, (lat_side, lat_side)))
    vae_in = in_mask_flags(mask_grid_fractions(mask_path, (lat_side, lat_side)))
    vit_in = in_mask_flags(mask_grid_fractions(mask_path, (vit_side, vit_side)))

    store = new_store()
    CAPTURE.update(store=store, phase=None, main_cache=None, step_idx=-1,
                   _text_pass=False, _flow_capture=False,
                   probe_steps=set(args.probe_steps),
                   text={}, flow={}, img={})
    device = next(model.parameters()).device
    set_seed(args.seed)

    text_ids = ([inferencer.new_token_ids["bos_token_id"]]
                + tokenizer.encode(prompt)
                + [inferencer.new_token_ids["eos_token_id"]])
    changed_rows_local = sorted({r for (a, b) in uniq for r in range(1 + a, 1 + b)})

    image = inferencer.vae_transform.resize_transform(
        pil_img2rgb(Image.open(src_path).convert("RGB")))

    gen_context = inferencer.init_gen_context()
    cfg_text_context = _copy.deepcopy(gen_context)
    cfg_img_context = _copy.deepcopy(gen_context)

    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda"),
                        dtype=torch.bfloat16):
        # -- source image prefill (hidden-state capture only) ---------------
        CAPTURE["phase"] = "img_prefill_vae"
        CAPTURE["img"]["vae_rows"] = list(range(1, 1 + n_lat))
        CAPTURE["img"]["vae_in"] = vae_in
        gen_context = inferencer.update_context_image(image, gen_context, vae=True, vit=False)
        CAPTURE["phase"] = "img_prefill_vit"
        CAPTURE["img"]["vit_rows"] = list(range(1, 1 + n_vit))
        CAPTURE["img"]["vit_in"] = vit_in
        gen_context = inferencer.update_context_image(image, gen_context, vae=False, vit=True)
        cfg_text_context = _copy.deepcopy(gen_context)

        # -- prompt prefill: changed tokens are queries ---------------------
        CAPTURE["phase"] = "text_prefill"
        CAPTURE["main_cache"] = gen_context["past_key_values"]
        CAPTURE["text"]["changed_rows_local"] = changed_rows_local
        gen_context = inferencer.update_context_text(prompt, gen_context)
        CAPTURE["main_cache"] = None
        CAPTURE["_text_pass"] = False
        cfg_img_context = inferencer.update_context_text(prompt, cfg_img_context)
        CAPTURE["phase"] = None

        kv_after_text = gen_context["kv_lens"][0]
        n_vae_keys = n_lat + 2                       # SOI + latents + EOI
        n_vit_keys = n_vit + 2
        text_cache_start = kv_after_text - len(text_ids)
        assert text_cache_start == n_vae_keys + n_vit_keys, (
            f"cache layout mismatch: {text_cache_start} != {n_vae_keys}+{n_vit_keys}")
        changed_key_cols = sorted(text_cache_start + r for r in changed_rows_local)

        geom = {
            "n_vae_keys": n_vae_keys, "n_vit_keys": n_vit_keys,
            "vae_key_in": np.concatenate([[False], vae_in, [False]]),
            "vit_key_in": np.concatenate([[False], vit_in, [False]]),
        }
        store["geom"] = geom

        # -- denoising flow: latent queries -> changed-token keys -----------
        timesteps = torch.linspace(1, 0, args.num_timesteps)
        timesteps = args.timestep_shift * timesteps / (1 + (args.timestep_shift - 1) * timesteps)
        step_ts = {i: float(timesteps[i]) for i in args.probe_steps if i < len(timesteps) - 1}

        CAPTURE["phase"] = "flow"
        CAPTURE["main_cache"] = gen_context["past_key_values"]
        CAPTURE["step_idx"] = -1
        CAPTURE["flow"] = {
            "changed_key_cols": changed_key_cols,
            "latent_rows": list(range(1, 1 + n_lat)),
            "latent_in": lat_in,
        }
        _ = inferencer.gen_image(
            (args.image_size, args.image_size), gen_context,
            cfg_text_precontext=cfg_text_context,
            cfg_img_precontext=cfg_img_context,
            cfg_text_scale=GEN_KWARGS["cfg_text"],
            cfg_img_scale=GEN_KWARGS["cfg_img"],
            cfg_interval=list(GEN_KWARGS["cfg_interval"]),
            cfg_renorm_min=0.0,
            cfg_renorm_type="text_channel",
            num_timesteps=args.num_timesteps,
            timestep_shift=args.timestep_shift,
        )
        CAPTURE.update(phase=None, main_cache=None, _flow_capture=False)

    prefill_metrics = summarize_prefill(store, geom)
    flow_metrics = summarize_flow(store, lat_in)
    margins = hidden_margins(store)

    per_layer = {}
    for layer in range(num_layers):
        row: dict[str, Any] = {"layer": layer}
        if layer in prefill_metrics:
            row.update(prefill_metrics[layer])
        if layer in flow_metrics:
            row.update({f"flow_{k}": v for k, v in flow_metrics[layer].items()})
        for kind in ("vae", "vit", "lat"):
            if layer in margins[kind]:
                row[f"margin_{kind}"] = margins[kind][layer]
        per_layer[layer] = row

    return {
        "pair_id": pair_id,
        "status": "ok",
        "n_changed_tokens": len(changed_rows_local),
        "changed_token_text": " | ".join(prompt[a:b] for a, b in occurrences[:4]),
        "n_vae_in": int(geom["vae_key_in"].sum()),
        "n_vit_in": int(geom["vit_key_in"].sum()),
        "n_latent": n_lat,
        "n_latent_in": int(lat_in.sum()),
        "probe_steps": {str(k): round(v, 4) for k, v in step_ts.items()},
        "per_layer": per_layer,
        "_store": store,
        "_lat_side": lat_side,
        "_vit_side": vit_side,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--seed", type=int, default=GEN_KWARGS["seed"])
    parser.add_argument("--num-timesteps", type=int, default=GEN_KWARGS["num_timesteps"])
    parser.add_argument("--timestep-shift", type=float, default=GEN_KWARGS["shift"])
    parser.add_argument("--image-size", type=int, default=GEN_KWARGS["image_size"])
    parser.add_argument("--probe-steps", type=int, nargs="*", default=[0, 11, 23],
                        help="denoising step indices to capture (24 steps for 25-step CFG)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep-vectors", action="store_true",
                        help="save h_tok/mu vectors + per-head maps in npz (heavier)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("E2 probe requires CUDA (run on the GPU instance)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    maps_dir = args.out_dir / "maps"
    maps_dir.mkdir(exist_ok=True)

    m = MODELS[args.model]
    ns = argparse.Namespace(
        llm_path=args.llm_path,
        vae_path=args.llm_path / "ae.safetensors",
        base_checkpoint_path=args.llm_path / "ema.safetensors",
        trainable_checkpoint=m["trainable_checkpoint"],
        skip_trainable_reload=m["skip_trainable_reload"],
        num_timesteps=args.num_timesteps,
        cfg_text_scale=GEN_KWARGS["cfg_text"],
        cfg_img_scale=GEN_KWARGS["cfg_img"],
        cfg_interval_start=GEN_KWARGS["cfg_interval"][0],
        cfg_interval_end=GEN_KWARGS["cfg_interval"][1],
        timestep_shift=args.timestep_shift,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
    )
    print(json.dumps({"event": "model_load_start", "model": args.model}), flush=True)
    model, vae, inferencer, infer_args, audit = build_inference_model(ns)
    num_layers = len(model.language_model.model.layers)
    tokenizer = inferencer.tokenizer
    print(json.dumps({"event": "model_load_done", "model": args.model,
                      "layers": num_layers}), flush=True)

    install_attention_patch()
    install_hidden_hooks(model)

    records, _ = load_automask_manifest(args.val, require_files=True)
    if args.limit:
        records = records[: args.limit]

    config = {
        "model": args.model,
        "trainable_checkpoint": str(m["trainable_checkpoint"]) if m["trainable_checkpoint"] else None,
        "skip_trainable_reload": m["skip_trainable_reload"],
        "lora_wrappers": audit["lora_wrappers"],
        "num_layers": num_layers,
        "gen_kwargs": {**GEN_KWARGS, "num_timesteps": args.num_timesteps,
                       "shift": args.timestep_shift},
        "probe_steps": list(args.probe_steps),
        "seed": args.seed,
        "val": str(args.val),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "probe_config.json").write_text(json.dumps(config, indent=2))

    manifest_path = args.out_dir / "probe_manifest.jsonl"
    done_ids = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    done_ids.add(rec["pair_id"])

    t0 = time.time()
    for idx, record in enumerate(records):
        pair_id = record["pair_id"]
        if pair_id in done_ids and (maps_dir / f"{pair_id}.npz").exists():
            continue
        t_pair = time.time()
        try:
            result = probe_pair(inferencer, tokenizer, record, args, num_layers)
        except Exception as exc:  # noqa: BLE001
            result = {"pair_id": pair_id, "status": f"error: {exc!r}"[:500]}
        status = result.get("status", "error")
        store = result.pop("_store", None)
        result.pop("_lat_side", None)
        result.pop("_vit_side", None)
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        if status == "ok" and store is not None:
            payload = {}
            for (layer, step), maps in store["flow_maps"].items():
                payload[f"flow_map_L{layer}_S{step}"] = maps.astype(np.float16)
            # prefill spatial maps: changed-token rows over source-VAE keys,
            # head/token-mean, shape (n_vae,) without SOI/EOI
            geom = store.get("geom")
            if geom:
                n_v = geom["n_vae_keys"] - 2
                for layer, rows in store["prefill_rows"].items():
                    per_tok = []
                    for _row, head_rows in rows.items():
                        per_tok.append(np.stack(
                            [p[1:1 + n_v] for _h, p in sorted(head_rows.items())]))
                    payload[f"prefill_map_L{layer}"] = np.mean(
                        np.stack(per_tok), axis=(0, 1)).astype(np.float16)
            if args.keep_vectors:
                for L, h in store["h_tok"].items():
                    payload[f"h_tok_L{L}"] = h.astype(np.float16)
                for key, (mu_in, mu_out) in store["mu"].items():
                    name = "mu_" + "_".join(map(str, key))
                    payload[name + "_in"] = mu_in.astype(np.float16)
                    payload[name + "_out"] = mu_out.astype(np.float16)
            np.savez_compressed(maps_dir / f"{pair_id}.npz", **payload)
        print(json.dumps({
            "event": "pair_done", "model": args.model, "i": idx, "pair_id": pair_id,
            "status": status, "sec": round(time.time() - t_pair, 1),
            "eta_min": round((len(records) - idx - 1) * (time.time() - t0) / (idx + 1) / 60, 1)}),
            flush=True)

    print(json.dumps({"event": "all_done", "model": args.model,
                      "pairs": len(records)}), flush=True)


if __name__ == "__main__":
    main()
