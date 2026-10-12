# E2: Token-Localization Probe (base vs final991)

**Question**: did the counterfactual-CoT LoRA (final991, train2973) learn a
mechanism that binds *changed tokens* in the prompt to the *image edit region*?

No training involved — measurement only, on the 51-pair val split used by the
10.8 in-domain evaluation (Module A), with `intervention.new_value` defining
the changed tokens and `automatic_difference_mask` defining the edit region.

## Hook-point feasibility analysis (read before running)

Verified against BAGEL source (`modeling/bagel/bagel.py`,
`modeling/bagel/qwen2_navit.py`, `inferencer.py`):

1. **How the prompt enters the backbone (MoT dual-expert)**. Generation for
   `[source_image, prompt]` runs: (a) source-image prefill — VAE latents via the
   *gen expert* (`q_proj_moe_gen` etc., `is_causal=False`), ViT patches via the
   *und expert*; (b) prompt prefill — text tokens are QUERIES (und expert,
   `is_causal=True`) attending to all cached source-image keys; (c) denoising
   flow steps — the query sequence is only `[SOI, 4096 latents, EOI]`
   (`is_causal=False`, gen expert), the prompt lives in the KV cache.
2. **Attention directions that actually exist**:
   - *changed token -> source image*: exists during prompt prefill (P1);
   - *target latent -> changed token*: exists during flow steps — the
     attend-and-segment direction (P2);
   - *changed token -> target latent*: does NOT exist — changed tokens are
     never queries during denoising (architectural fact, both in inference and
     training). The probe therefore measures the two existing directions and
     states this limitation explicitly.
3. **Attention weights**: all inference attention uses
   `flash_attn_varlen_func`, which does not return matrices. The probe
   monkeypatches `PackedAttentionMoT.forward_inference`: for capture-targeted
   calls it re-executes the identical projection/norm/RoPE/cache-merge code
   and replaces only the flash kernel with a chunked manual softmax
   (verified numerically equivalent, selftest maxdiff=0.0); all other calls
   (non-probe steps, CFG passes, image prefills) use the original flash path.
   Capture targeting uses KV-cache object identity, so the conditional pass is
   distinguished from cfg_text/cfg_img passes exactly. No degraded
   grad-CAM/rollout approximation was needed.
4. **Token geometry**: latent and ViT grids are row-major
   (`token k <-> (k // w, k % w)`); 1024px -> 64x64 = 4096 latent tokens
   (`latent_downsample = 16`), ViT 980/14 -> 70x70 = 4900 patches. KV cache
   layout: `[SOI + vae_src + EOI][SOI + vit_src + EOI][bos + prompt + eos]`,
   then flow query keys appended.

## Metrics (per layer 0..27, per pair, per model)

- **P1 prefill attention mass** (head-mean over changed-token queries):
  `norm_ratio_vae/vit/visual` = (in-mask mass density) / (out-mask density)
  on source-image keys (attend-and-segment style normalization), plus raw
  masses and mass-vs-uniform-baseline.
- **P2 flow attention** (per probe step, head-mean): per-latent-position
  attention on changed-token keys -> `flow_log2_contrast` (log2 in/out
  density), `flow_mass_in_frac` (in-mask share of changed-token attention),
  `flow_auroc` (threshold-free segmentation quality of the map vs the mask).
- **P3 hidden-state contrast margin**: `cos(h_tok, mu_in) - cos(h_tok, mu_out)`
  for source-VAE / source-ViT / flow-latent token groups. Caveat: h_tok comes
  from the prompt prefill (which sees the images) while visual mu come from
  the image prefills (which do not see the prompt) — inherent to the
  architecture, documented as an interpretation limit.

All layers are scanned (risk noted in the task: last layer may be
output-semantic; mid layers can be more informative). Probe steps default to
`0 11 23` (early/mid/late of the 24 denoising steps).

## Usage (GPU instance, ~27GB VRAM per model)

```bash
bash /group/40016/niightli/BAGEL/experiments/counterfactual_cot/e2_token_probe/run_e2.sh          # probe (GPU0=base, GPU1=final991) -> analyze -> viz
bash /group/40016/niightli/BAGEL/experiments/counterfactual_cot/e2_token_probe/run_e2.sh smoke    # 1 pair, 1 probe step
```

Stages: `probe | analyze | viz | all` (all resumable; watchdog rewrites
`../wb_e2_progress.txt` every 30s). Sampling matches training/eval:
seed 1234, 25 steps, CFG text 4.0 / img 2.0 (text_channel renorm), shift 3.0,
1024px; the probe re-runs the *real* generation trajectory and only intercepts
attention at the probe steps.

Outputs (`outputs/e2_token_probe/`):
- `probe_{model}/probe_manifest.jsonl` — per-pair per-layer metrics
- `probe_{model}/maps/{pair_id}.npz` — flow attention maps (per head),
  prefill spatial maps; `--keep-vectors` adds h_tok/mu vectors
- `scores/e2_layer_table.md`, `scores/e2_stats_summary.json`,
  `scores/e2_layer_curves.csv` — per-layer tables, paired diff
  (final991 - base) with 20000-resample bootstrap 95% CI, best layer per
  metric (same statistical method as final991_eval_v1 Module A)
- `figures/{model}/{pair_id}_grid.png` — heatmaps over the source image with
  mask contour: flow map (latent->changed tokens) and prefill map
  (changed tokens->source VAE); paper-figure candidates

## Files

| file | role |
|---|---|
| `probe.py` | capture machinery + per-pair probe driver (uses v5 `build_inference_model`, LoRA direct-load) |
| `analyze.py` | per-layer stats, paired bootstrap CI, markdown/json/csv reports |
| `visualize.py` | attention heatmap grids overlaid with the edit mask |
| `run_e2.sh` | single-command launcher with watchdog + smoke mode |
| `selftest_e2.py` | CPU self-test: attention parity vs clean-room reference, span/mask/auroc/summarizer units, synthetic end-to-end (33/33 PASS, 2026-10-12) |

## Known limitations

- ViT grid mapping uses area-downsample of the 1024 mask to 70x70 (the ViT
  branch resizes through 980px; uniform scaling makes this equivalent up to
  half-pixel edges).
- P2 maps are head-means; per-head maps are kept in the npz for finer analyses.
- The changed-token span is `intervention.new_value` occurrences in the
  rendered Target CoT prompt (anchored to `=`/`:` assignments when present).
  Pairs where the value is absent from the prompt are skipped and reported
  (`status: span_not_found`).
