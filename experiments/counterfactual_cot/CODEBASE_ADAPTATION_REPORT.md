## BAGEL codebase adaptation report

### Audited training call chain

- Entry point: `train/pretrain_unified_navit.py`.
- Argument parsing: `ModelArguments`, `DataArguments`, and `TrainingArguments` via `HfArgumentParser`.
- Model construction:
  - `Qwen2ForCausalLM` from `modeling/qwen2`.
  - `SiglipVisionModel` when `visual_und=True`.
  - VAE from `modeling/autoencoder.load_ae` when `visual_gen=True`.
  - Unified model: `modeling.bagel.Bagel`.
- Tokenizer and special tokens:
  - `Qwen2Tokenizer`.
  - `data.data_utils.add_special_tokens`.
- Distributed wrapping and checkpoint:
  - `train.fsdp_utils.FSDPConfig`.
  - `FSDPCheckpoint.try_load_ckpt`.
  - `FSDPCheckpoint.try_load_train_state`.
  - `FSDPCheckpoint.fsdp_save_ckpt`.
- Dataset loading:
  - YAML path is `DataArguments.dataset_config_file`.
  - YAML group names are resolved through `data.dataset_info.DATASET_INFO` and `DATASET_REGISTRY`.
  - `data.dataset_base.PackedDataset` packs each raw sample into token/image sequences.
- Batch format:
  - Samples must provide `sequence_plan`, `text_ids_list`, `image_tensor_list`, `num_tokens`, and `data_indexes`.
  - Image target supervision uses `sequence_plan` items with `type=vae_image` and `loss=1`.
  - Text CE supervision uses `type=text` and `loss=1`; this MVP keeps CoT in conditioning text and does not train CoT generation CE unless a future dataset variant adds it.
- Losses:
  - BAGEL returns `loss_dict["ce"]` and `loss_dict["mse"]`.
  - Image generation loss is native VAE latent MSE over `mse_loss_indexes`.
  - The audited path has no public hook for per-token spatial mask weighting or direct source-target latent delta contrastive loss.

### Audited inference call chain

- General wrapper: `inferencer.InterleaveInferencer`.
- Context construction:
  - `update_context_text` calls `model.prepare_prompts` and `forward_cache_update_text`.
  - `update_context_image` calls `prepare_vae_images` / `forward_cache_update_vae` and `prepare_vit_images` / `forward_cache_update_vit`.
- Image generation:
  - `gen_image` calls `prepare_vae_latent`, CFG latent preparation, and `model.generate_image`.
  - `decode_image` decodes VAE latents into PIL images.
- Existing editing examples:
  - `image_edit.py` runs source image + instruction -> edited image through `InterleaveInferencer`.
  - `eval/gen/gen_images_mp_imgedit.py` contains distributed image-edit generation utilities.

### Reused BAGEL files

- `train/pretrain_unified_navit.py` for model loading, FSDP, optimizer, scheduler, checkpoint, and native losses.
- `data/dataset_base.py` for sequence packing and collator.
- `data/interleave_datasets/interleave_t2i_dataset.py` and `data/interleave_datasets/edit_dataset.py` as the design reference for image-edit sequence layouts.
- `inferencer.py` and `image_edit.py` for inference prompt/image ordering.
- `eval/gen/*` as external benchmark execution references.

### New experiment files

- Data/schema and generation:
  - `experiments/counterfactual_cot/data/schema.py`
  - `build_source_registry.py`
  - `generate_interventions.py`
  - `render_prompts.py`
  - `api_client_text.py`
  - `api_client_image.py`
  - `filter_pairs.py`
  - `split_dataset.py`
  - `freeze_manifest.py`
  - `validate_manifest.py`
- BAGEL-compatible dataset:
  - `experiments/counterfactual_cot/datasets/bagel_manifest_dataset.py`
- Training wrapper:
  - `experiments/counterfactual_cot/training/train.py`
  - `objectives.py`
  - `paired_losses.py`
- Inference and evaluation:
  - `experiments/counterfactual_cot/inference/*`
  - `experiments/counterfactual_cot/evaluation/*`
- Configs, scripts, and tests under the same experiment directory.

### Core file modifications

No BAGEL core source file was modified. Dataset registration is done at runtime in `experiments/counterfactual_cot/training/train.py` by adding `counterfactual_cot` to `data.dataset_info.DATASET_REGISTRY` and `DATASET_INFO` before delegating to `train.pretrain_unified_navit.main`.

### Paired MVP implementation path

- `CounterfactualCoTDataset` reads frozen JSONL manifests.
- `standard_cot` mode serializes `Prompt + Target/Source CoT -> Image`.
- `independent_cf` alternates source and target as independent samples from the same frozen pair set.
- `paired_cf_mvp` serializes source CoT, source image, intervention, target CoT, and preservation spec, then supervises only the target image with BAGEL native MSE.
- Existing BAGEL edit capability is used by putting source image tokens before the paired text instruction, following the audited edit dataset and inference examples.

### Mask and paired-loss feasibility

- Feasible now:
  - Pair-aware conditioning.
  - Native target image MSE supervision.
  - External mask validation and reporting.
  - Grounding/faithfulness metrics computed from local outputs and judge CSVs.
- Not implemented as a real loss yet:
  - Mask-aware preservation loss.
  - Delta alignment loss.
- Reason:
  - Audited `PackedDataset` exposes `mse_loss_indexes` but not a mask-weight tensor per VAE token.
  - Audited training loop reduces `loss_dict["mse"]` uniformly over target VAE tokens.
  - No stable public hook was found for spatial visual-token features aligned to target masks.
- Guardrail:
  - `training/paired_losses.py` raises `PairedLossUnavailable` if full paired losses are requested.
  - `paired_cf_full.yaml` is a guarded placeholder and must not be used to claim explicit preservation/delta loss.

### Confirmed facts

- BAGEL supports source image + text instruction -> generated edited image through interleaved VAE/ViT conditioning.
- BAGEL training can consume new JSONL datasets when a registry entry provides `manifest_path`.
- The existing checkpoint, distributed training, and native image-generation loss can be reused unchanged.
- Independent and Paired variants can be made fair by reading the same frozen manifest and controlling seen image/token budgets.

### Remaining risks

- Gold overfit requires real accepted source/target images and masks; code is ready but no gold data is bundled.
- Explicit preservation/delta losses require a small BAGEL core extension or exposed model features.
- Public R2I-Bench and GenEval execution remains delegated to existing eval scripts; this experiment currently aggregates their outputs rather than vendoring those benchmarks.
