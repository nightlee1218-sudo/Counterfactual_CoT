# Counterfactual CoT: Full-Backbone LoRA Experiments

This repository contains code for Counterfactual CoT image-editing overfit experiments built on BAGEL.

## Included experiments

- `full_backbone_lora_overfit_v1`: full L0-L27 LoRA baseline with global flow-MSE.
- `full_backbone_lora_region1024_v2`: 1024 VAE / 980 ViT, CoT-core conditioning and old/new/preserve region loss.
- `full_backbone_lora_region1024_fsdp_v3`: two-A100-40GB FSDP FULL_SHARD implementation of the 1024 region-loss experiment.

## Main training condition

`Source Image + Source Prompt + Target CoT`

## Region objective

`0.25 * old_region_mse + 0.25 * new_region_mse + 0.50 * preserve_region_mse`

## Upstream dependency

The experiment code is built on BAGEL. The exact upstream commit is recorded in `BAGEL_UPSTREAM_COMMIT.txt`; required local BAGEL modifications are in `patches/bagel_local_changes.patch`.

## Gold40 v4 provenance archive

After a v4 single-pair proof-of-concept has produced its output directory, archive immutable-style evidence without changing the run artifacts:

```bash
cd /group/40016/niightli/Counterfactual_CoT
python experiments/counterfactual_cot/training/archive_v4_provenance.py \
  --output-dir /absolute/path/to/v4-output-dir \
  --require-switch-report
```

The default evidence set is checkpoints at steps `0` and `500`, one final report, and one training log. The script computes SHA256 for each required artifact plus any `switch` outputs, validates every discovered structured switch report as exactly 16 unique successful seeds, and exclusively creates `provenance_manifest_v4.json`. It never overwrites an existing manifest or changes input artifacts. For alternate milestones, use `--steps 0 250 500`; use `--checkpoint STEP=PATH`, `--final-report PATH`, or `--train-log PATH` to resolve nonstandard names.

## Excluded artifacts

Model weights, source/target images, masks, manifests, checkpoints, logs and generated outputs are intentionally excluded.
