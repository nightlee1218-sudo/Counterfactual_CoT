#!/usr/bin/env python3
"""E2 offline self-test (CPU, no GPU / no real BAGEL weights needed).

Covers:
  1. manual_attention vs a clean-room reference varlen attention
     (full + causal, GQA), routed through the real PackedAttentionMoT module:
       a. und mode (text-prefill shape) output parity with the flash path
          (flash replaced by a naive reference on CPU) + row extract sanity
       b. gen mode (flow shape) output parity + column extract sanity
  2. token span locating (occurrences + prefix-tokenization mapping)
  3. mask -> grid mapping + in/out flags
  4. auroc on a known case
  5. summarize_prefill / summarize_flow / hidden_margins on synthetic captures
  6. analyze.py + visualize.py end-to-end on synthetic probe manifests/npz

Run:  python3 selftest_e2.py
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BAGEL = Path("/group/40016/niightli/BAGEL")
sys.path.insert(0, str(BAGEL))

PASS, FAIL = 0, 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name} {extra}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {extra}")


# ---------------------------------------------------------------------------
# reference varlen attention (clean-room; replaces flash on CPU)
# ---------------------------------------------------------------------------
def ref_flash_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None,
                     max_seqlen_k=None, causal=False, **kw):
    outs = []
    for s in range(len(cu_seqlens_q) - 1):
        qs, qe = int(cu_seqlens_q[s]), int(cu_seqlens_q[s + 1])
        ks, ke = int(cu_seqlens_k[s]), int(cu_seqlens_k[s + 1])
        q_s, k_s, v_s = q[qs:qe], k[ks:ke], v[ks:ke]
        nq, H, d = q_s.shape
        Hkv = k_s.shape[1]
        ng = H // Hkv
        out = torch.empty_like(q_s)
        for h in range(H):
            qh = q_s[:, h].float()
            kh = k_s[:, h // ng].float()
            vh = v_s[:, h // ng]
            scores = (qh @ kh.T) / math.sqrt(d)
            if causal:
                # flash_attn varlen causal semantics: queries are the LAST nq
                # positions of the kv sequence -> key j allowed for query i
                # iff j <= (kv_total - nq) + i
                kv_total = ke - ks
                pos = torch.arange(nq)
                allowed = torch.arange(ke - ks)[None, :] <= (kv_total - nq + pos)[:, None]
                scores = scores.masked_fill(~allowed, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            out[:, h] = (probs.to(vh.dtype) @ vh)
        outs.append(out)
    return torch.cat(outs, dim=0)


def build_tiny_attn():
    from modeling.bagel.qwen2_navit import PackedAttentionMoT, Qwen2Config, NaiveCache
    cfg = Qwen2Config(vocab_size=32, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, qk_norm=True, rms_norm_eps=1e-6)
    torch.manual_seed(0)
    attn = PackedAttentionMoT(cfg, layer_idx=0).eval()
    for p in attn.parameters():
        p.data = p.data.float()
    return attn, cfg, NaiveCache


def rope_tables(config, positions: torch.Tensor):
    from modeling.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    rot = Qwen2RotaryEmbedding(config=config)
    # HF style: cos/sin over full head_dim, duplicated halves
    inv_freq, attn_factor = rot.inv_freq, None
    freqs = positions[:, None].float() @ inv_freq[None, :].float()
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def test_attention_paths():
    print("[1] attention manual vs reference (und + gen, full + causal)")
    import modeling.bagel.qwen2_navit as navit
    attn, cfg, NaiveCache = build_tiny_attn()
    orig_flash = navit.flash_attn_varlen_func
    navit.flash_attn_varlen_func = ref_flash_varlen

    sys.path.insert(0, str(HERE))
    import probe as probe_mod
    probe_mod.install_attention_patch()

    H, Hkv, d, hidden = 4, 2, 16, 64
    n_kv, n_q = 12, 8

    # ---- und mode (text prefill: queries attend cache + causal self) ----
    qseq = torch.randn(n_q, hidden)
    pos_ids = torch.arange(n_kv, n_kv + n_q)
    cos, sin = rope_tables(cfg, pos_ids)
    cache = NaiveCache(1)
    cache.key_cache[0] = torch.randn(n_kv, Hkv, d, dtype=torch.bfloat16)
    cache.value_cache[0] = torch.randn(n_kv, Hkv, d, dtype=torch.bfloat16)

    kwargs = dict(
        packed_query_sequence=qseq,
        query_lens=torch.tensor([n_q]),
        packed_query_position_embeddings=(cos, sin),
        packed_query_indexes=torch.arange(n_kv, n_kv + n_q),
        past_key_values=cache,
        key_values_lens=torch.tensor([n_kv]),
        packed_key_value_indexes=torch.arange(n_kv),
        update_past_key_values=False,
        is_causal=True,
        mode="und",
    )
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_ref, _ = navit.PackedAttentionMoT.forward_inference(attn, **kwargs)

    probe_mod.CAPTURE.update(phase="text_prefill", main_cache=cache, store=probe_mod.new_store(),
                             _text_pass=True, text={"changed_rows_local": [2, 3]})
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_man, _ = navit.PackedAttentionMoT.forward_inference(attn, **kwargs)
    diff = (out_ref.float() - out_man.float()).abs().max().item()
    check("und output parity", diff < 0.05, f"maxdiff={diff:.4f}")

    rows = probe_mod.CAPTURE["store"]["prefill_rows"][0]
    check("row extract shape", 2 in rows and 3 in rows and len(rows[2]) == H,
          f"heads={len(rows[2])}")
    prob = rows[2][0]
    check("row softmax sums to 1", abs(prob.sum() - 1.0) < 1e-3, f"sum={prob.sum():.5f}")
    # causal: row 2 (cache offset 12) must not attend key index > 12+2
    check("causal masking in rows", float(prob[15:].sum()) == 0.0 and float(prob[13]) >= 0.0,
          f"tail_sum={float(prob[15:].sum()):.2e}")

    # ---- gen mode (flow: [soi, latents..., eoi] queries, full attention) --
    n_lat = 6
    n_q_gen = n_lat + 2
    qseq_gen = torch.randn(n_q_gen, hidden)
    text_idx = torch.tensor([0, n_q_gen - 1])
    vae_idx = torch.arange(1, n_q_gen - 1)
    pos_ids_gen = torch.arange(n_kv, n_kv + n_q_gen)
    cos_g, sin_g = rope_tables(cfg, pos_ids_gen)
    kwargs_gen = dict(
        packed_query_sequence=qseq_gen,
        query_lens=torch.tensor([n_q_gen]),
        packed_query_position_embeddings=(cos_g, sin_g),
        packed_query_indexes=torch.arange(n_kv, n_kv + n_q_gen),
        past_key_values=cache,
        key_values_lens=torch.tensor([n_kv]),
        packed_key_value_indexes=torch.arange(n_kv),
        update_past_key_values=False,
        is_causal=False,
        mode="gen",
        packed_vae_token_indexes=vae_idx,
        packed_text_indexes=text_idx,
    )
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_ref_g, _ = navit.PackedAttentionMoT.forward_inference(attn, **kwargs_gen)
    store = probe_mod.new_store()
    probe_mod.CAPTURE.update(phase="flow", main_cache=cache, store=store, step_idx=-1,
                             probe_steps={0}, _flow_capture=False,
                             text={},
                             flow={"changed_key_cols": [n_kv + 2, n_kv + 4], "latent_rows": [],
                                   "latent_in": np.zeros(n_lat, dtype=bool)})
    # the routing wrapper bumps step_idx itself when it sees the main cache
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_man_g, _ = navit.PackedAttentionMoT.forward_inference(attn, **kwargs_gen)
    diff_g = (out_ref_g.float() - out_man_g.float()).abs().max().item()
    check("gen output parity", diff_g < 0.05, f"maxdiff={diff_g:.4f}")

    maps = store["flow_maps"].get((0, 0))
    check("flow map shape (heads, n_lat)", maps is not None and maps.shape == (H, n_lat),
          f"{None if maps is None else maps.shape}")
    check("flow map nonneg", bool((maps >= 0).all()))
    # manual cross-check of one map value: latent row 0, head 0, cols 14 & 16
    # (merged key indices: cache 0..11 + query keys 12..19; 14=query row 2 (vae),
    #  16=query row 4 (vae) -> both gen-expert projections)
    with torch.no_grad():
        q_row = qseq_gen[1]
        ln = attn.q_proj_moe_gen(q_row)
        qh = attn.q_norm_moe_gen(ln.view(H, d))[0]
        from modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        q2 = qh[None, None]
        c2, s2 = cos_g[1:2], sin_g[1:2]
        q2r, _ = apply_rotary_pos_emb(q2, q2, c2, s2, unsqueeze_dim=1)
        k_new = torch.zeros(n_q_gen, Hkv, d)
        for i in range(n_q_gen):
            is_text = i in (0, n_q_gen - 1)
            proj = attn.k_proj if is_text else attn.k_proj_moe_gen
            norm = attn.k_norm if is_text else attn.k_norm_moe_gen
            k_new[i] = norm(proj(qseq_gen[i]).view(Hkv, d))
        # rope the new keys with the same per-row tables as the module does
        from modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb as _rope
        _, k_new_roped = _rope(torch.zeros(1), k_new, cos_g, sin_g, unsqueeze_dim=1)
        k_new = k_new_roped.float()
        k_all = torch.cat([cache.key_cache[0].float(), k_new.float()], 0)
        kh = k_all[:, 0]
        scores = (q2r[0, 0].float() @ kh.T) / math.sqrt(d)
        p = torch.softmax(scores, dim=-1)
        expected = float(p[n_kv + 2] + p[n_kv + 4])
    got = float(maps[0, 0])
    check("flow map value vs hand-recompute", abs(expected - got) < 5e-3,
          f"exp={expected:.4f} got={got:.4f}")

    # routing: non-captured call must go to flash path (reference) unchanged
    probe_mod.CAPTURE.update(phase="flow", main_cache=cache, store=store, step_idx=0,
                             probe_steps={5}, _flow_capture=False)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out_flash, _ = navit.PackedAttentionMoT.forward_inference(attn, **kwargs_gen)
    check("non-capture routes to flash path",
          (out_flash.float() - out_ref_g.float()).abs().max().item() < 1e-5)

    navit.flash_attn_varlen_func = orig_flash


# ---------------------------------------------------------------------------
# span locating
# ---------------------------------------------------------------------------
class MockTok:
    """Char-level tokenizer: encode returns per-character ids."""
    def encode(self, text: str):
        return list(range(len(text)))


def test_span():
    print("[2] token span locating")
    import probe as probe_mod
    prompt = ("MODE\nSource Prompt: a ceramic vase on a table\n"
              "Target CoT: vase_1.object=pen-and-note caddy; vase_1.count=1\n"
              "Generate the target image.")
    occ = probe_mod.find_changed_occurrences(prompt, "pen-and-note caddy")
    check("anchored occurrence found", len(occ) == 1 and prompt[occ[0][0] - 1] == "=",
          f"{occ}")
    cache = {}
    a, b = probe_mod.span_to_token_range(MockTok(), prompt, occ[0][0], occ[0][1], cache)
    check("span maps to token range", a == occ[0][0] and b == occ[0][1], f"[{a},{b})")

    occ2 = probe_mod.find_changed_occurrences(prompt, "ceramic")
    check("unanchored fallback (prompt body)", len(occ2) == 1, f"{occ2}")
    occ3 = probe_mod.find_changed_occurrences("two 3 cats and 13 dogs", 3)
    check("numeric word boundary", all(prompt2 := "two 3 cats and 13 dogs"[s:e] == "3"
                                       for s, e in occ3) and len(occ3) == 1, f"{occ3}")
    occ4 = probe_mod.find_changed_occurrences("nothing here", "missing")
    check("missing value -> empty", occ4 == [])


# ---------------------------------------------------------------------------
# mask grid + auroc
# ---------------------------------------------------------------------------
def test_mask_and_auroc():
    print("[3] mask grid mapping + auroc")
    import probe as probe_mod
    from PIL import Image
    tmp = Path(tempfile.mkdtemp())
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[0:256, 0:512] = 255  # top-left quadrant-ish (1/8 of image)
    mpath = tmp / "mask.png"
    Image.fromarray(mask).save(mpath)
    frac = probe_mod.mask_grid_fractions(mpath, (64, 64))
    flags = probe_mod.in_mask_flags(frac)
    check("grid fraction shape", frac.shape == (64, 64))
    # top-left 16x32 cells fully covered; rest empty
    check("in/out counts", int(flags.sum()) == 16 * 32 and len(flags) == 64 * 64,
          f"in={int(flags.sum())}")
    check("fraction values sane", abs(frac[0, 0] - 1.0) < 1e-4 and frac[63, 63] == 0.0)

    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.75])
    labels = np.array([0, 1, 0, 1, 0, 1])
    a = probe_mod.auroc(scores, labels)
    # positives: ranks 2,4,6 -> mean 4; negatives mean 2.5 -> auroc 0.75? compute:
    # ranks: 0.1->1, 0.2->2, 0.35->3, 0.4->4, 0.75->5, 0.8->6
    # pos ranks {4,5,6} sum 15; (15 - 3*4/2)/(3*3) = 9/9 = 1.0
    check("auroc perfect", abs(a - 1.0) < 1e-9, f"auroc={a}")
    scores2 = np.array([0.5, 0.5, 0.5])
    labels2 = np.array([1, 0, 0])
    check("auroc ties", abs(probe_mod.auroc(scores2, labels2) - 0.5) < 1e-9)
    shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# metric summarizers on synthetic captures
# ---------------------------------------------------------------------------
def test_summarizers():
    print("[4] summarize_prefill / summarize_flow / hidden_margins")
    import probe as probe_mod
    rng = np.random.default_rng(0)
    n_vae, n_vit, n_text = 66, 72, 20
    vae_in = np.zeros(n_vae, dtype=bool)
    vae_in[:16] = True
    vit_in = np.zeros(n_vit, dtype=bool)
    vit_in[:20] = True
    store = probe_mod.new_store()
    geom = {"n_vae_keys": n_vae, "n_vit_keys": n_vit,
            "vae_key_in": vae_in, "vit_key_in": vit_in}
    # synthetic rows: bias in-mask vae keys upward
    for layer in (0, 1):
        rows = {}
        for r in (2, 3):
            heads = {}
            for h in range(4):
                p = rng.random(n_vae + n_vit + n_text) + 0.001
                p[:16] += 6.0  # in-mask boost
                p = p / p.sum()
                heads[h] = p
            rows[r] = heads
        store["prefill_rows"][layer] = rows
    out = probe_mod.summarize_prefill(store, geom)
    check("prefill metrics computed", 0 in out and "norm_ratio_vae" in out[0])
    check("prefill in-mask ratio > 1", out[0]["norm_ratio_vae"] > 2,
          f"ratio={out[0]['norm_ratio_vae']:.2f}")
    check("prefill masses sum <= 1", out[0]["mass_visual"] + out[0]["mass_text"] <= 1.0 + 1e-6)

    # flow maps
    n_lat = 64
    lat_in = np.zeros(n_lat, dtype=bool)
    lat_in[:16] = True
    for layer in (0, 1):
        m = rng.random((4, n_lat)) + 0.001
        m[:, :16] += 5.0
        store["flow_maps"][(layer, 0)] = m
    fl = probe_mod.summarize_flow(store, lat_in)
    check("flow metrics computed", 0 in fl and "log2_contrast" in fl[0])
    check("flow contrast positive", fl[0]["log2_contrast"] > 1,
          f"log2={fl[0]['log2_contrast']:.2f}")
    check("flow auroc high", fl[0]["auroc"] > 0.9, f"auroc={fl[0]['auroc']:.3f}")

    # hidden margins
    D = 32
    h = rng.normal(size=(2, D))
    store["h_tok"] = {0: h, 1: h}
    mu_in = h.mean(0) + 0.5
    mu_out = rng.normal(size=D)
    store["mu"][("vae", 0)] = (mu_in, mu_out)
    store["mu"][("vit", 0)] = (mu_out, mu_in)   # inverted on purpose
    store["mu"][("lat", 0, 0)] = (mu_in, mu_out)
    mg = probe_mod.hidden_margins(store)
    check("margin vae positive", mg["vae"][0] > 0.3, f"{mg['vae'][0]:.3f}")
    check("margin vit negative", mg["vit"][0] < -0.3, f"{mg['vit'][0]:.3f}")
    check("margin lat averaged", mg["lat"][0] > 0.3, f"{mg['lat'][0]:.3f}")


# ---------------------------------------------------------------------------
# end-to-end analyze + visualize on synthetic probe outputs
# ---------------------------------------------------------------------------
def test_analyze_viz():
    print("[5] analyze.py + visualize.py end-to-end (synthetic)")
    import probe as probe_mod
    from PIL import Image
    tmp = Path(tempfile.mkdtemp(prefix="e2_self_"))
    probe_root = tmp / "outputs/e2_token_probe"
    rng = np.random.default_rng(1)
    n_layers, D = 28, 64
    for model, boost in (("base", 0.0), ("final991", 0.35)):
        mdir = probe_root / f"probe_{model}"
        (mdir / "maps").mkdir(parents=True)
        (mdir / "probe_config.json").write_text("{}")
        with (mdir / "probe_manifest.jsonl").open("w") as f:
            for pid in range(12):
                per_layer = {}
                for L in range(n_layers):
                    base_ratio = 1.2 + 0.02 * L
                    per_layer[str(L)] = {
                        "layer": L,
                        "norm_ratio_vae": base_ratio + boost + rng.normal(0, 0.15),
                        "norm_ratio_vit": base_ratio + boost + rng.normal(0, 0.15),
                        "norm_ratio_visual": base_ratio + boost + rng.normal(0, 0.15),
                        "mass_visual": 0.3 + rng.normal(0, 0.02),
                        "flow_log2_contrast": 0.5 + boost + rng.normal(0, 0.3),
                        "flow_mass_in_frac": 0.4 + rng.normal(0, 0.05),
                        "flow_auroc": 0.65 + 0.1 * boost + rng.normal(0, 0.05),
                        "margin_vae": 0.02 * boost + rng.normal(0, 0.05),
                        "margin_vit": rng.normal(0, 0.05),
                        "margin_lat": rng.normal(0, 0.05),
                    }
                rec = {"pair_id": f"p{pid:03d}", "status": "ok", "per_layer": per_layer,
                       "n_changed_tokens": 3, "changed_token_text": "x",
                       "n_vae_in": 100, "n_vit_in": 100, "n_latent": 4096,
                       "n_latent_in": 512, "probe_steps": {"11": 0.5}}
                f.write(json.dumps(rec) + "\n")
                payload = {}
                for L in (8, 16, 24):
                    m = rng.random((4, 4096))
                    m[:, :512] += 4.0
                    payload[f"flow_map_L{L}_S11"] = m.astype(np.float16)
                    payload[f"prefill_map_L{L}"] = rng.random(4096).astype(np.float16)
                np.savez_compressed(mdir / "maps" / f"p{pid:03d}.npz", **payload)

    # fake val records for visualize (source image + mask)
    val_dir = probe_root.parent / "valdata"
    (val_dir / "masks").mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1024, 1024), (120, 140, 160))
    img.save(val_dir / "src.png")
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    mask[200:500, 200:500] = 255
    Image.fromarray(mask).save(val_dir / "masks" / "m.png")

    # analyze (against synthetic manifests only)
    r = subprocess.run([sys.executable, str(HERE / "analyze.py"),
                        "--probe-root", str(probe_root), "--out-dir",
                        str(probe_root / "scores"), "--bootstrap", "2000"],
                       capture_output=True, text=True)
    check("analyze exit 0", r.returncode == 0, r.stderr[-200:] if r.returncode else "")
    table = probe_root / "scores/e2_layer_table.md"
    summary = probe_root / "scores/e2_stats_summary.json"
    check("analyze table + summary written", table.exists() and summary.exists())
    txt = table.read_text()
    check("table has SIG rows for boosted metrics", "SIG" in txt)
    j = json.loads(summary.read_text())
    check("summary has best_layer", "best_layer" in j["metrics"]["flow_log2_contrast"])

    # visualize with a stub val split (monkeypatch path by temporary copy)
    # -> visualize reads CFCOT/final991_automask_v5_data/...; point it via env-free
    #    trick: run with --probe-root and temporarily create the val file path?
    # Instead call visualize functions directly on synthetic npz.
    sys.path.insert(0, str(HERE))
    import visualize as viz
    with np.load(probe_root / "probe_base/maps/p000.npz") as z:
        npz = {k: z[k] for k in z.files}
    sheet = viz.render_pair_grid(npz, img, (mask > 127).astype(np.uint8),
                                 [8, 16, 24], 64, viz.try_font(15))
    out_png = probe_root / "grid_test.png"
    sheet.save(str(out_png))
    check("visualize grid rendered", out_png.exists() and out_png.stat().st_size > 10000,
          f"{out_png.stat().st_size} bytes")
    shutil.rmtree(tmp)


if __name__ == "__main__":
    test_attention_paths()
    test_span()
    test_mask_and_auroc()
    test_summarizers()
    test_analyze_viz()
    print(f"\nSELFTEST {'PASS' if FAIL == 0 else 'FAIL'}: {PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
