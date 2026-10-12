# E1: General Text-to-Image Benchmark (base vs final991)

**Question**: does the counterfactual-CoT LoRA (final991, trained on 991 paired
counterfactual edits, +6.3dB PSNR on in-domain editing) degrade or improve
general text-to-image generation relative to the base model?

## Design

- **Models**: `base` = BAGEL-7B-MoT with zero-init LoRA path (exact identity,
  same loading path as the 10.8 evaluation); `final991` = LoRA **direct load**
  from `outputs/final991_automask_v5_train2973/checkpoints/latest.pt`
  (strict trainable reload, no merged weights).
- **Prompts**: official T2I-CompBench (Karine-Huang/T2I-CompBench,
  `examples/dataset/*.txt`), sampled with seed 1234:
  | subset | source categories | n |
  |---|---|---|
  | attribute_binding | color 50 + shape 50 + texture 50 | 150 |
  | spatial_relations | spatial | 150 |
  | counting | numeracy | 150 |
  Spatial and attribute binding are BAGEL's reported weak dimensions;
  numeracy covers counting.
  GenAI-Bench and DPG official prompt data could not be retrieved in this
  environment (HF parquet scan limit; no prompt-only mirror reachable via
  proxy) — noted as a limitation.
- **Sampling** (matches training): seed 1234, 25 timesteps, CFG text 4.0 /
  img 2.0, timestep shift 3.0, 1024x1024, one image per prompt per model
  (900 images total; GPU0=base, GPU1=final991 in parallel).
- **Scoring**: primary CLIP-T (CLIP ViT-L/14 OpenAI weights, local file;
  CLIPScore = 2.5*max(cos,0) plus raw cosine). Optional VQA faithfulness via
  Qwen3-VL (venus llmproxy) on a capped resumable subset (public pool is rate
  limited ~1/min, so VQA covers ≤60 items/model by default).
- **Statistics**: per-item paired diff (final991 - base), 20000-resample
  bootstrap 95% CI (same method as final991_eval_v1 Module A), win/loss/tie,
  Wilcoxon signed-rank when scipy is available. Reported per subset + overall.

## Usage

```bash
# on the 2xA100 GPU machine (mounts /group):
bash /group/40016/niightli/BAGEL/experiments/counterfactual_cot/e1_general_t2i/run_e1.sh          # full run
bash /group/40016/niightli/BAGEL/experiments/counterfactual_cot/e1_general_t2i/run_e1.sh smoke    # 2-item pipeline check
```

Stages are resumable: `gen | score | vqa | stats | sheet | all`.
Watchdog rewrites `../wb_e1_progress.txt` every 30s during generation.

Outputs land in `outputs/e1_general_t2i/`:
`gen/{model}/images/`, `scores/clipT_*.jsonl`, `scores/stats_summary.json`,
`scores/comparison_table.md`, `contact_sheets/*.png`.

## Files

| file | role |
|---|---|
| `build_prompts.py` | sample official prompts into `prompts/*.jsonl` (seed 1234) |
| `generate_t2i.py` | T2I generation, reuses v5 `build_inference_model` (LoRA direct load) |
| `score_clip.py` | CLIP-T / CLIPScore (open_clip, local ViT-L/14 weights) |
| `score_vqa.py` | optional Qwen3-VL binary faithfulness (rate-limit tolerant) |
| `stats_paired.py` | paired bootstrap stats + markdown comparison table |
| `contact_sheet.py` | side-by-side base vs final991 contact sheets |
| `run_e1.sh` | single-command launcher with watchdog |
| `selftest_pipeline.py` | offline self-test of stats + sheet stages (PASS 2026-10-12) |
| `bench_raw/` | official T2I-CompBench prompt files (provenance) |
