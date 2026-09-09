from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image
from torch.utils.data import DataLoader

from data.data_utils import pil_img2rgb
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.dataset_info import DATASET_INFO, DATASET_REGISTRY
from data.transforms import ImageTransform
from experiments.counterfactual_cot.data.render_prompts import paired_prompt, render_intervention_text, render_preservation_text
from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.datasets import CounterfactualCoTDataset
from experiments.counterfactual_cot.training.smoke_train import (
    DEFAULT_MODEL_PATH,
    _assert_float_input_dtype,
    _assert_paired_vit_inputs,
    _autocast_for_model,
    _batch_report,
    _build_model,
    _cuda_memory_report,
    _device,
    _dtype_for_precision,
    _grad_report,
    _inspect_raw_samples,
    _load_llm_config,
    _load_tokenizer,
    _load_vit_config,
    _log_event,
    _loss_from_dict,
    _move_batch_to_device,
    _path_size_gib,
    _set_seed,
    _set_trainable_parameters,
    _transform_args,
    _vae_config,
)
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import Bagel, BagelConfig, Qwen2ForCausalLM, SiglipVisionModel


FIXED_GOLD40_MANIFEST = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
    "reruns/nanobanana_v4_absolute_region/manifests/"
    "intervention_pairs_gold40_human_verified_final_normalized.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/gold40_overfit_p1"
)
DEFAULT_TRAINABLE_REGEX = r"^(time_embedder|latent_pos_embed|vit_pos_embed|vae2llm|llm2vae|connector)\."
DEFAULT_EVAL_STEPS = (0, 25, 50, 100, 200)
PRIMARY_DIAGNOSTIC_TYPES = (
    "attribute_replace",
    "object_replace",
    "count_change",
    "spatial_position_change",
)
OPTIONAL_DIAGNOSTIC_TYPES = ("spatial_relation_change",)


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _build_model_for_gold40(args: argparse.Namespace, device: torch.device, tokenizer_size: int, num_new_tokens: int) -> Tuple[Bagel, Any, Dict[str, Any]]:
    if args.model_device_map == "single":
        return _build_model(
            args.llm_path,
            args.vae_path,
            args.checkpoint_path,
            device,
            tokenizer_size=tokenizer_size,
            num_new_tokens=num_new_tokens,
            random_init=False,
            precision=args.precision,
        )

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("--model-device-map auto requires at least two visible CUDA devices.")

    dtype = _dtype_for_precision(args.precision)
    llm_config = _load_llm_config(args.llm_path)
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False

    vit_config = _load_vit_config(args.llm_path)
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae_config = _vae_config()
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
        timestep_shift=1.0,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        if num_new_tokens > 0:
            language_model.resize_token_embeddings(tokenizer_size)
            language_model.config.vocab_size = tokenizer_size
            llm_config.vocab_size = tokenizer_size
        vit_model = SiglipVisionModel(vit_config)
        vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        model = Bagel(language_model=language_model, vit_model=vit_model, config=bagel_config)

    gpu_count = torch.cuda.device_count()
    if args.model_device_map == "manual_2gpu":
        split_layer = max(1, llm_config.num_hidden_layers // 2)
        device_map = {
            "language_model.model.embed_tokens": 0,
            "language_model.model.rotary_emb": 0,
            "vit_model": 0,
            "connector": 0,
            "vit_pos_embed": 0,
            "time_embedder": 0,
            "latent_pos_embed": 0,
            "vae2llm": 0,
            "llm2vae": 1,
            "language_model.model.norm": 1,
            "language_model.model.norm_moe_gen": 1,
            "language_model.lm_head": 1,
        }
        for layer_index in range(llm_config.num_hidden_layers):
            device_map[f"language_model.model.layers.{layer_index}"] = 0 if layer_index < split_layer else 1
    else:
        max_memory = {i: args.device_map_max_memory for i in range(gpu_count)}
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["Qwen2MoTDecoderLayer"],
            dtype=dtype,
        )
        offloaded = {name: target for name, target in device_map.items() if str(target) in {"cpu", "disk"}}
        if offloaded:
            raise RuntimeError(
                "Refusing to train with CPU/disk offload. Increase --device-map-max-memory or use --model-device-map manual_2gpu. "
                f"Offloaded modules sample: {dict(list(offloaded.items())[:20])}"
            )
    _log_event(
        "bagel_model_dispatch_start",
        checkpoint_path=str(args.checkpoint_path),
        dtype=str(dtype),
        model_device_map=args.model_device_map,
        device_map_max_memory=args.device_map_max_memory,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        torch_cuda_device_count=gpu_count,
        device_map_summary={str(k): str(v) for k, v in list(device_map.items())[:120]},
    )
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(args.checkpoint_path),
        device_map=device_map,
        offload_buffers=False,
        dtype=dtype,
        force_hooks=True,
    )
    _log_event(
        "bagel_model_dispatch_done",
        first_param_device=str(_module_device(model)),
        cuda_memory_gpu0=_cuda_memory_report(torch.device("cuda:0")),
        cuda_memory_gpu1=_cuda_memory_report(torch.device("cuda:1")) if torch.cuda.device_count() > 1 else None,
    )

    _log_event("vae_load_start", vae_path=str(args.vae_path), vae_size_gib=_path_size_gib(args.vae_path))
    vae_model, _ = load_ae(args.vae_path)
    _log_event("vae_load_done", vae_path=str(args.vae_path))
    vae_model.to(device=torch.device("cuda:0"), dtype=dtype).eval()
    checkpoint_report = {
        "checkpoint_path": str(args.checkpoint_path),
        "status": "loaded_with_accelerate_device_map",
        "model_device_map": args.model_device_map,
        "device_map_max_memory": args.device_map_max_memory,
    }
    return model, vae_model, checkpoint_report


def _validate_fixed_manifest(manifest: Path) -> None:
    if manifest.resolve() != FIXED_GOLD40_MANIFEST.resolve():
        raise ValueError(
            "P1 Gold40 overfit is intentionally locked to the normalized human-verified manifest. "
            f"Expected {FIXED_GOLD40_MANIFEST}, got {manifest}."
        )
    if not manifest.exists():
        raise FileNotFoundError(f"Fixed Gold40 manifest does not exist: {manifest}")


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)


def _register_gold40_dataset(manifest_path: Path, variant: str, seed: int, max_records: int) -> None:
    DATASET_REGISTRY["counterfactual_cot_gold40"] = CounterfactualCoTDataset
    DATASET_INFO["counterfactual_cot_gold40"] = {
        "gold40_train": {
            "manifest_path": str(manifest_path),
            "variant": variant,
            "training_seed": seed,
            "num_total_samples": max_records,
            "max_records": max_records,
        }
    }


def _build_gold40_loader(
    manifest_path: Path,
    tokenizer: Any,
    special_tokens: Dict[str, int],
    args: argparse.Namespace,
    *,
    seed_offset: int = 0,
) -> DataLoader:
    _register_gold40_dataset(manifest_path, args.variant, args.seed + seed_offset, args.num_samples)
    dataset_meta = {
        "counterfactual_cot_gold40": {
            "dataset_names": ["gold40_train"],
            "weight": 1.0,
            "is_mandatory": True,
            "image_transform_args": _transform_args(args.image_size, args.image_size, 16),
            "vit_image_transform_args": _transform_args(args.vit_image_size, args.vit_min_image_size, 14),
            "num_used_data": [args.num_samples],
            "max_records": args.num_samples,
        }
    }
    data_config = DataConfig(
        grouped_datasets=copy.deepcopy(dataset_meta),
        text_cond_dropout_prob=0.0,
        vit_cond_dropout_prob=0.0,
        vae_cond_dropout_prob=0.0,
        vae_image_downsample=16,
        max_latent_size=32,
    )
    packed_dataset = PackedDataset(
        data_config=data_config,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        local_rank=0,
        world_size=1,
        num_workers=0,
        expected_num_tokens=args.expected_num_tokens,
        max_num_tokens_per_sample=args.max_num_tokens_per_sample,
        max_num_tokens=args.max_num_tokens,
        prefer_buffer_before=0,
        max_buffer_size=4,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    )
    packed_dataset.set_epoch(args.seed + seed_offset)
    return DataLoader(
        packed_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_wrapper(),
        drop_last=True,
    )


def _resolve_image_path(pair: Dict[str, Any], path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    image_root = pair.get("final_image_root")
    if image_root:
        rooted = Path(image_root) / path
        if rooted.exists():
            return rooted
    return FIXED_GOLD40_MANIFEST.parent / path


def _load_pair_image(pair: Dict[str, Any], side: str) -> Image.Image:
    return pil_img2rgb(Image.open(_resolve_image_path(pair, pair[side]["image"])))


def _save_reference_image(pair: Dict[str, Any], side: str, output_path: Path, image_size: int) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = _load_pair_image(pair, side).resize((image_size, image_size), Image.Resampling.LANCZOS)
    image.save(output_path)
    return str(output_path)


def _select_visualization_pairs(records: Sequence[Dict[str, Any]], per_type: int, total: int) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for pair in records:
        intervention_type = str(pair.get("intervention", {}).get("intervention_type", "unknown"))
        by_type.setdefault(intervention_type, []).append(pair)

    missing_primary = [name for name in PRIMARY_DIAGNOSTIC_TYPES if not by_type.get(name)]
    if missing_primary:
        raise RuntimeError(f"Gold40 diagnostic set is missing required intervention types: {missing_primary}")

    selected_by_type: Dict[str, List[Dict[str, Any]]] = {}
    selected_ids = set()
    for intervention_type in PRIMARY_DIAGNOSTIC_TYPES:
        chosen = by_type[intervention_type][:per_type]
        selected_by_type[intervention_type] = chosen
        selected_ids.update(pair["pair_id"] for pair in chosen)

    selected: List[Dict[str, Any]] = []
    for slot in range(per_type):
        for intervention_type in PRIMARY_DIAGNOSTIC_TYPES:
            chosen = selected_by_type.get(intervention_type, [])
            if slot < len(chosen) and len(selected) < total:
                selected.append(chosen[slot])

    for intervention_type in OPTIONAL_DIAGNOSTIC_TYPES:
        if len(selected) >= total:
            break
        for pair in by_type.get(intervention_type, [])[:1]:
            if pair["pair_id"] not in selected_ids:
                selected.append(pair)
                selected_ids.add(pair["pair_id"])
                break

    if len(selected) < total:
        for pair in records:
            if pair["pair_id"] not in selected_ids:
                selected.append(pair)
                selected_ids.add(pair["pair_id"])
            if len(selected) >= total:
                break

    selected = selected[:total]
    counts: Dict[str, int] = {}
    for pair in selected:
        intervention_type = str(pair.get("intervention", {}).get("intervention_type", "unknown"))
        counts[intervention_type] = counts.get(intervention_type, 0) + 1
    required_per_primary = min(per_type, max(1, total // len(PRIMARY_DIAGNOSTIC_TYPES)))
    undercovered = {
        name: counts.get(name, 0)
        for name in PRIMARY_DIAGNOSTIC_TYPES
        if counts.get(name, 0) < min(required_per_primary, len(by_type.get(name, [])))
    }
    if undercovered:
        raise RuntimeError(f"Visualization set must cover primary intervention types; got {counts}, undercovered={undercovered}")
    return selected


def _make_inferencer(model: torch.nn.Module, vae_model: torch.nn.Module, tokenizer: Any, special_tokens: Dict[str, int], args: argparse.Namespace) -> InterleaveInferencer:
    vae_transform = ImageTransform(args.visual_image_size, args.visual_vae_min_image_size, 16)
    vit_transform = ImageTransform(args.visual_vit_image_size, args.visual_vit_min_image_size, 14)
    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=special_tokens,
    )


def _full_prompt(pair: Dict[str, Any]) -> str:
    return paired_prompt(pair, include_source_image=True)


def _field_token(intervention: Dict[str, Any]) -> str:
    field = str(intervention.get("field", ""))
    entity_id = str(intervention.get("entity_id", "entity"))
    if field.startswith("attributes."):
        return f"{entity_id}.attribute.{field.split('.', 1)[1]}"
    return f"{entity_id}.{field}"


def _choose_conflict_value(intervention: Dict[str, Any]) -> Any:
    old_value = intervention.get("old_value")
    new_value = intervention.get("new_value")
    field = str(intervention.get("field", ""))
    intervention_type = str(intervention.get("intervention_type", ""))
    if isinstance(new_value, int):
        return new_value + 1 if new_value != old_value + 1 else new_value + 2
    if field.endswith("color") or "attribute" in intervention_type:
        colors = ["blue", "green", "red", "yellow", "purple", "orange", "white", "black"]
        for color in colors:
            if color not in {str(old_value), str(new_value)}:
                return color
    if "position" in intervention_type or "location" in field:
        positions = ["upper-left", "upper-right", "lower-left", "lower-right", "center", "left", "right"]
        for position in positions:
            if position not in {str(old_value), str(new_value)}:
                return position
    if "category" in intervention_type or field in {"category", "object"}:
        categories = ["chair", "table", "book", "lamp", "car", "basket", "plant", "box"]
        for category in categories:
            if category not in {str(old_value), str(new_value)}:
                return category
    return f"conflict_{new_value}"


def _conflict_prompt(pair: Dict[str, Any]) -> Tuple[str, Any]:
    intervention = dict(pair["intervention"])
    conflict_value = _choose_conflict_value(intervention)
    original_new_value = str(intervention.get("new_value"))
    intervention["new_value"] = conflict_value
    target_cot = str(pair["target"]["cot_text"])
    field_token = _field_token(intervention)
    target_cot = target_cot.replace(f"{field_token}={original_new_value}", f"{field_token}={conflict_value}")
    if target_cot == pair["target"]["cot_text"]:
        target_cot = f"{pair['source']['cot_text']}; conflict_override.{field_token}={conflict_value}"
    prompt = (
        "[MODE=COUNTERFACTUAL_EDIT_CONFLICT_COT]\n"
        f"Source Prompt: {pair['source']['prompt']}\n"
        f"Source CoT: {pair['source']['cot_text']}\n"
        f"Intervention: {render_intervention_text(intervention)}\n"
        f"Target CoT: {target_cot}\n"
        f"{render_preservation_text(pair.get('preservation', {}))}\n"
        "Generate the target image following the Structured CoT, even if it conflicts with the usual target."
    )
    return prompt, conflict_value


def _no_cot_prompt(pair: Dict[str, Any]) -> str:
    return (
        "[MODE=COUNTERFACTUAL_EDIT_NO_COT]\n"
        "Task: Edit the provided source image.\n"
        "Do not use a structured atomic change description.\n"
        "Preserve the source image background, viewpoint, lighting, layout, object identities, counts, and non-target details as much as possible.\n"
        "Generate one plausible edited image."
    )


def _delta_only_prompt(pair: Dict[str, Any]) -> str:
    intervention = pair["intervention"]
    preservation = pair.get("preservation", {})
    must_preserve = ", ".join(preservation.get("must_preserve", [])) or "all other objects, counts, positions, background, viewpoint and lighting"
    return (
        "[MODE=COUNTERFACTUAL_EDIT_DELTA_ONLY]\n"
        f"Change {intervention['entity_id']}.{intervention['field']} from {intervention['old_value']} to {intervention['new_value']}.\n"
        f"Preserve: {must_preserve}.\n"
        "Do not redraw the whole scene from a final scene description. Use the source image as the visual anchor."
    )


def _generate_one(
    inferencer: InterleaveInferencer,
    pair: Dict[str, Any],
    output_path: Path,
    args: argparse.Namespace,
    *,
    seed: int,
    prompt_pair: Dict[str, Any] | None = None,
    source_pair: Dict[str, Any] | None = None,
    prompt: str | None = None,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    source = _load_pair_image(source_pair or pair, "source")
    prompt = prompt or paired_prompt(prompt_pair or pair, include_source_image=True)
    result = inferencer.interleave_inference(
        input_lists=[source, prompt],
        think=False,
        understanding_output=False,
        cfg_text_scale=args.cfg_text_scale,
        cfg_img_scale=args.cfg_img_scale,
        cfg_interval=[args.cfg_interval_start, args.cfg_interval_end],
        timestep_shift=args.timestep_shift,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=args.cfg_renorm_min,
        cfg_renorm_type=args.cfg_renorm_type,
        image_shapes=(args.visual_image_size, args.visual_image_size),
    )
    images = [item for item in result if isinstance(item, Image.Image)]
    if not images:
        raise RuntimeError(f"No image generated for pair {pair.get('pair_id')}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[-1].save(str(output_path))
    return str(output_path)


def _generate_visualizations(
    model: torch.nn.Module,
    vae_model: torch.nn.Module,
    tokenizer: Any,
    special_tokens: Dict[str, int],
    records: Sequence[Dict[str, Any]],
    selected_pairs: Sequence[Dict[str, Any]],
    step: int,
    args: argparse.Namespace,
    *,
    full_diagnostics: bool = False,
) -> Dict[str, Any]:
    model.eval()
    vae_model.eval()
    inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, args)
    try:
        step_dir = args.output_dir / "visualizations" / f"step_{step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, pair in enumerate(selected_pairs):
            pair_id = pair["pair_id"]
            pair_dir = step_dir / pair_id
            source_path = _save_reference_image(pair, "source", pair_dir / "source.png", args.visual_image_size)
            golden_path = _save_reference_image(pair, "target", pair_dir / "golden_target.png", args.visual_image_size)
            output_path = _generate_one(
                inferencer,
                pair,
                pair_dir / f"model_step_{step:03d}.png",
                args,
                seed=args.generation_seed,
            )
            row = {
                "pair_id": pair_id,
                "intervention_type": pair.get("intervention", {}).get("intervention_type"),
                "step": step,
                "source": source_path,
                "model_output": output_path,
                "golden_target": golden_path,
                "prompt": paired_prompt(pair, include_source_image=True),
                "seed": args.generation_seed,
                "generation_parameters": _generation_parameters(args),
            }
            if full_diagnostics:
                swapped_source_pair = selected_pairs[(index + 1) % len(selected_pairs)]
                conflict_prompt, conflict_value = _conflict_prompt(pair)
                no_cot_prompt = _no_cot_prompt(pair)
                delta_only_prompt = _delta_only_prompt(pair)
                row["diagnostic_source_swap"] = _generate_one(
                    inferencer,
                    pair,
                    pair_dir / "diagnostic_source_swap.png",
                    args,
                    seed=args.generation_seed,
                    source_pair=swapped_source_pair,
                    prompt=_full_prompt(pair),
                )
                row["diagnostic_conflict_cot"] = _generate_one(
                    inferencer,
                    pair,
                    pair_dir / "diagnostic_conflict_cot.png",
                    args,
                    seed=args.generation_seed,
                    prompt=conflict_prompt,
                )
                row["diagnostic_no_cot"] = _generate_one(
                    inferencer,
                    pair,
                    pair_dir / "diagnostic_no_cot.png",
                    args,
                    seed=args.generation_seed,
                    prompt=no_cot_prompt,
                )
                row["diagnostic_delta_only"] = _generate_one(
                    inferencer,
                    pair,
                    pair_dir / "diagnostic_delta_only.png",
                    args,
                    seed=args.generation_seed,
                    prompt=delta_only_prompt,
                )
                row["diagnostic_conflict_value"] = conflict_value
                row["diagnostic_conflict_prompt"] = conflict_prompt
                row["diagnostic_no_cot_prompt"] = no_cot_prompt
                row["diagnostic_delta_only_prompt"] = delta_only_prompt
            rows.append(row)
            _log_event("visualization_generated", step=step, index=index, pair_id=pair_id, output=output_path)
        manifest_path = step_dir / "visualization_manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return {"step": step, "num_visualizations": len(rows), "full_diagnostics": full_diagnostics, "manifest": str(manifest_path)}
    finally:
        del inferencer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model.train()


def _rel(path: str | Path, root: Path) -> str:
    return Path(os.path.relpath(str(path), str(root))).as_posix()


def _write_image_table_html(path: Path, title: str, sections: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    style = """
    body { font-family: Arial, sans-serif; margin: 18px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 36px; }
    th, td { border: 1px solid #ccc; padding: 6px; vertical-align: top; }
    img { width: 180px; height: 180px; object-fit: contain; background: #f7f7f7; }
    .meta { font-size: 12px; line-height: 1.35; max-width: 1500px; white-space: pre-wrap; }
    """
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>", f"<style>{style}</style></head><body>"]
    parts.append(f"<h1>{title}</h1>")
    for section in sections:
        parts.append(f"<h2>{section['title']}</h2>")
        if section.get("meta"):
            parts.append(f"<pre class='meta'>{section['meta']}</pre>")
        parts.append("<table><thead><tr>")
        for column in section["columns"]:
            parts.append(f"<th>{column['label']}</th>")
        parts.append("</tr></thead><tbody><tr>")
        for column in section["columns"]:
            image_path = column.get("path")
            if image_path:
                rel = _rel(image_path, output_dir)
                parts.append(f"<td><a href='{rel}'><img src='{rel}'></a><br><code>{rel}</code></td>")
            else:
                parts.append("<td>missing</td>")
        parts.append("</tr></tbody></table>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_training_review_pages(output_dir: Path, visual_reports: Sequence[Dict[str, Any]], final_step: int) -> Dict[str, str]:
    by_step: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for report in visual_reports:
        manifest = Path(report["manifest"])
        if not manifest.exists():
            continue
        step = int(report["step"])
        by_step[step] = {}
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                by_step[step][row["pair_id"]] = row

    available_steps = sorted(by_step)
    intermediate_steps = [step for step in available_steps if step not in {0, final_step}]
    representative_intermediate = intermediate_steps[-1] if intermediate_steps else None
    pair_ids = list(by_step.get(0, by_step.get(final_step, {})).keys())

    main_sections = []
    for pair_id in pair_ids:
        base_row = by_step.get(0, {}).get(pair_id, {})
        final_row = by_step.get(final_step, {}).get(pair_id, {})
        intermediate_row = by_step.get(representative_intermediate, {}).get(pair_id, {}) if representative_intermediate is not None else {}
        main_sections.append(
            {
                "title": pair_id,
                "meta": json.dumps(
                    {
                        "intervention_type": base_row.get("intervention_type") or final_row.get("intervention_type"),
                        "intermediate_step": representative_intermediate,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "columns": [
                    {"label": "Source", "path": base_row.get("source") or final_row.get("source")},
                    {"label": "Base step 0", "path": base_row.get("model_output")},
                    {"label": "Intermediate", "path": intermediate_row.get("model_output")},
                    {"label": "Final", "path": final_row.get("model_output")},
                    {"label": "Golden target", "path": base_row.get("golden_target") or final_row.get("golden_target")},
                ],
            }
        )

    conditioning_sections = []
    final_rows = by_step.get(final_step, {})
    for pair_id in pair_ids:
        row = final_rows.get(pair_id) or by_step.get(0, {}).get(pair_id, {})
        conditioning_sections.append(
            {
                "title": pair_id,
                "meta": json.dumps(
                    {
                        "intervention_type": row.get("intervention_type"),
                        "step": row.get("step"),
                        "conflict_value": row.get("diagnostic_conflict_value"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "columns": [
                    {"label": "Correct CoT", "path": row.get("model_output")},
                    {"label": "Conflict CoT", "path": row.get("diagnostic_conflict_cot")},
                    {"label": "No CoT", "path": row.get("diagnostic_no_cot")},
                    {"label": "Swapped source", "path": row.get("diagnostic_source_swap")},
                    {"label": "Delta-only", "path": row.get("diagnostic_delta_only")},
                ],
            }
        )

    main_path = output_dir / "visualizations" / "review_main.html"
    conditioning_path = output_dir / "visualizations" / "review_conditioning.html"
    _write_image_table_html(main_path, "Gold40 Overfit Main Trend", main_sections, output_dir)
    _write_image_table_html(conditioning_path, "Gold40 Overfit Conditioning Diagnostics", conditioning_sections, output_dir)
    return {"main_review_html": str(main_path), "conditioning_review_html": str(conditioning_path)}


def _generation_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "train_image_size": args.image_size,
        "train_vit_image_size": args.vit_image_size,
        "visual_image_size": args.visual_image_size,
        "visual_vae_min_image_size": args.visual_vae_min_image_size,
        "visual_vit_image_size": args.visual_vit_image_size,
        "visual_vit_min_image_size": args.visual_vit_min_image_size,
        "seed": args.generation_seed,
        "cfg_text_scale": args.cfg_text_scale,
        "cfg_img_scale": args.cfg_img_scale,
        "cfg_interval": [args.cfg_interval_start, args.cfg_interval_end],
        "timestep_shift": args.timestep_shift,
        "num_timesteps": args.num_timesteps,
        "cfg_renorm_min": args.cfg_renorm_min,
        "cfg_renorm_type": args.cfg_renorm_type,
    }


def _tensor_tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tensor_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tensor_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tensor_tree_to_cpu(item) for item in value)
    return value


def _random_state_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_random_state(payload: Dict[str, Any] | None) -> None:
    if not payload:
        return
    python_state = payload.get("python_random_state")
    if python_state is not None:
        random.setstate(python_state)
    torch_state = payload.get("torch_rng_state")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_states = payload.get("torch_cuda_rng_state_all")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def _trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}


def _save_trainable_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    path: Path,
    step: int,
    args: argparse.Namespace,
    trainable_report: Dict[str, Any],
    loss_history: Sequence[Dict[str, Any]],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "counterfactual_cot_trainable_only_v2",
        "step": step,
        "global_step": step,
        "trainable_state_dict": _trainable_state_dict(model),
        "optimizer_state_dict": _tensor_tree_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": _tensor_tree_to_cpu(scheduler.state_dict()) if scheduler is not None else None,
        "random_state": _random_state_payload(),
        "config": _serializable_args(args),
        "trainable_report": trainable_report,
        "loss_history": list(loss_history),
    }
    torch.save(payload, path)
    return str(path)


def _load_trainable_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    strict: bool = True,
) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint has no trainable_state_dict: {checkpoint_path}")
    named_params = dict(model.named_parameters())
    missing = []
    for name, tensor in state.items():
        if name not in named_params:
            missing.append(name)
            continue
        named_params[name].data.copy_(tensor.to(device=named_params[name].device, dtype=named_params[name].dtype))
    if strict and missing:
        raise RuntimeError(f"Trainable checkpoint contains unknown parameters: {missing[:20]}")
    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    _restore_random_state(payload.get("random_state"))
    return {
        "checkpoint_path": str(checkpoint_path),
        "step": payload.get("step", payload.get("global_step")),
        "global_step": payload.get("global_step", payload.get("step")),
        "num_loaded_tensors": len(state),
        "unknown_tensors": missing[:20],
        "optimizer_loaded": optimizer is not None and payload.get("optimizer_state_dict") is not None,
        "scheduler_loaded": scheduler is not None and payload.get("scheduler_state_dict") is not None,
        "random_state_loaded": payload.get("random_state") is not None,
    }


def _serializable_args(args: argparse.Namespace) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def _prepare_batch(batch: Any, vae_model: torch.nn.Module, device: torch.device, model_dtype: torch.dtype) -> Dict[str, Any]:
    data = _move_batch_to_device(batch, device, model_dtype)
    with torch.no_grad():
        vae_dtype = next(vae_model.parameters()).dtype
        data["padded_latent"] = vae_model.encode(data.pop("padded_images").to(dtype=vae_dtype))
    _assert_float_input_dtype(data, model_dtype)
    return data


def _forward_loss(model: torch.nn.Module, data: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
    with _autocast_for_model(model):
        loss_dict = model(**data)
    tensor_losses = [loss for loss in loss_dict.values() if loss is not None]
    if not tensor_losses:
        raise RuntimeError(f"Model returned no trainable losses: {loss_dict}")
    total = torch.zeros((), device=tensor_losses[0].device)
    scalars: Dict[str, float] = {}
    ce = loss_dict.get("ce")
    if ce is not None:
        ce_loss = ce.mean()
        total = total + ce_loss.to(total.device)
        scalars["ce"] = float(ce_loss.detach().cpu())
    else:
        scalars["ce"] = 0.0
    mse = loss_dict.get("mse")
    if mse is not None:
        mse_loss = mse.mean()
        total = total + mse_loss.to(total.device)
        scalars["mse"] = float(mse_loss.detach().cpu())
    else:
        scalars["mse"] = 0.0
    scalars["total"] = float(total.detach().cpu())
    return total, scalars


def _mean_train_mse(
    model: torch.nn.Module,
    vae_model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_dtype: torch.dtype,
    max_eval_batches: int,
) -> Dict[str, Any]:
    model.train()
    losses: List[float] = []
    seen_pair_ids = set()
    iterator = iter(loader)
    with torch.no_grad():
        for _ in range(max_eval_batches):
            batch = next(iterator)
            data = _prepare_batch(batch, vae_model, device, model_dtype)
            _, scalars = _forward_loss(model, data)
            losses.append(float(scalars["mse"]))
            for row in batch.batch_data_indexes:
                if row.get("pair_id"):
                    seen_pair_ids.add(row["pair_id"])
            del data, scalars
            if len(seen_pair_ids) >= 40:
                break
    del iterator
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mean_mse = sum(losses) / max(1, len(losses))
    return {"mean_mse": mean_mse, "num_batches": len(losses), "num_seen_pairs": len(seen_pair_ids)}


def _cleanup_cuda_memory(device: torch.device, event: str) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _log_event(event, cuda_memory=_cuda_memory_report(device))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _git_metadata() -> Dict[str, Any]:
    def run_git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[3], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "diff_status": run_git(
            "status",
            "--short",
            "--",
            "modeling/bagel/qwen2_navit.py",
            "experiments/counterfactual_cot/training/gold40_overfit.py",
            "experiments/counterfactual_cot/training/smoke_train.py",
        ),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    _validate_fixed_manifest(args.manifest)
    _prepare_output_dir(args.output_dir)
    _set_seed(args.seed)
    device = _device(args.device)
    records = load_jsonl(args.manifest)
    if len(records) != args.num_samples:
        raise ValueError(f"P1 expects exactly {args.num_samples} Gold40 records, found {len(records)} in {args.manifest}")
    accepted = [bool(row.get("human_review", {}).get("accepted")) for row in records]
    if not all(accepted):
        raise ValueError("All Gold40 records must be human_review.accepted=true before P1 overfit")

    selected_pairs = _select_visualization_pairs(records, args.visual_per_type, args.visual_total)
    selected_path = args.output_dir / "visualizations" / "fixed_visualization_set.jsonl"
    with selected_path.open("w", encoding="utf-8") as handle:
        for pair in selected_pairs:
            handle.write(json.dumps({"pair_id": pair["pair_id"], "intervention_type": pair.get("intervention", {}).get("intervention_type")}, ensure_ascii=False, sort_keys=True) + "\n")

    _log_event(
        "gold40_overfit_start",
        manifest=str(args.manifest),
        output_dir=str(args.output_dir),
        steps=args.steps,
        eval_steps=args.eval_steps,
        precision=args.precision,
        llm_path=args.llm_path,
        checkpoint_path=args.checkpoint_path,
        checkpoint_size_gib=_path_size_gib(args.checkpoint_path),
        vae_path=args.vae_path,
        vae_size_gib=_path_size_gib(args.vae_path),
        cuda_memory=_cuda_memory_report(device),
    )

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)
    raw_sequence_report = _inspect_raw_samples(
        args.manifest,
        tokenizer,
        args.variant,
        args.image_size,
        args.vit_image_size,
        args.vit_min_image_size,
        min(4, args.num_samples),
    )
    train_loader = _build_gold40_loader(args.manifest, tokenizer, special_tokens, args, seed_offset=0)
    eval_loader = _build_gold40_loader(args.manifest, tokenizer, special_tokens, args, seed_offset=100000)

    model, vae_model, checkpoint_load_report = _build_model_for_gold40(
        args,
        device,
        tokenizer_size=len(tokenizer),
        num_new_tokens=num_new_tokens,
    )
    trainable_report = _set_trainable_parameters(model, args.trainable_name_regex)
    model.train()
    vae_model.eval()
    model_dtype = next(model.parameters()).dtype
    _log_event(
        "gold40_runtime_context",
        checkpoint_path=str(args.checkpoint_path),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        model_main_device=str(next(model.parameters()).device),
        trainable_parameters=trainable_report["trainable_parameters"],
        manifest=str(args.manifest),
        model_device_map=args.model_device_map,
        git=_git_metadata(),
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    start_step = 0
    if args.resume_trainable_checkpoint:
        resume_report = _load_trainable_checkpoint(model, args.resume_trainable_checkpoint, optimizer=optimizer, scheduler=scheduler)
        start_step = int(resume_report.get("global_step") or resume_report.get("step") or 0)
        _log_event("trainable_checkpoint_loaded", **resume_report)

    visual_reports: List[Dict[str, Any]] = []
    eval_reports: List[Dict[str, Any]] = []
    loss_history: List[Dict[str, Any]] = []
    checkpoint_paths: List[str] = []

    step0_eval = _mean_train_mse(model, vae_model, eval_loader, device, model_dtype, args.eval_max_batches)
    eval_reports.append({"step": 0, **step0_eval})
    if not args.no_visualizations:
        visual_reports.append(_generate_visualizations(model, vae_model, tokenizer, special_tokens, records, selected_pairs, 0, args, full_diagnostics=True))
    ckpt0 = _save_trainable_checkpoint(
        model,
        optimizer,
        scheduler,
        args.output_dir / "checkpoints" / "step_000_trainable.pt",
        0,
        args,
        trainable_report,
        loss_history,
    )
    checkpoint_paths.append(ckpt0)
    _log_event("eval_step_done", step=0, **step0_eval, trainable_checkpoint=ckpt0)
    _cleanup_cuda_memory(device, "post_step0_cleanup_done")

    train_iterator = iter(train_loader)
    first_batch_report: Dict[str, Any] | None = None
    first_grad_report: Dict[str, Any] | None = None
    eval_step_set = set(args.eval_steps)

    for step in range(1, args.steps + 1):
        if step <= start_step:
            continue
        batch = next(train_iterator)
        data = _prepare_batch(batch, vae_model, device, model_dtype)
        if first_batch_report is None:
            first_batch_report = _batch_report(batch, data, device)
            _assert_paired_vit_inputs(args, raw_sequence_report, first_batch_report, data)
        optimizer.zero_grad(set_to_none=True)
        loss, scalars = _forward_loss(model, data)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {scalars}")
        loss.backward()
        if first_grad_report is None:
            first_grad_report = _grad_report(model)
            if not first_grad_report["nonzero_finite_grads"]:
                raise RuntimeError("No expected trainable parameter has non-zero finite gradient")
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm).detach().cpu())
        optimizer.step()
        scheduler.step()

        row = {"step": step, **scalars, "grad_norm": grad_norm}
        loss_history.append(row)
        del data, loss, scalars
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if step % args.log_every == 0 or step in eval_step_set:
            _log_event("train_step", **row, cuda_memory=_cuda_memory_report(device))

        if step in eval_step_set:
            eval_report = _mean_train_mse(model, vae_model, eval_loader, device, model_dtype, args.eval_max_batches)
            eval_reports.append({"step": step, **eval_report})
            if not args.no_visualizations:
                visual_reports.append(
                    _generate_visualizations(
                        model,
                        vae_model,
                        tokenizer,
                        special_tokens,
                        records,
                        selected_pairs,
                        step,
                        args,
                        full_diagnostics=(step == args.steps),
                    )
                )
            ckpt = _save_trainable_checkpoint(
                model,
                optimizer,
                scheduler,
                args.output_dir / "checkpoints" / f"step_{step:03d}_trainable.pt",
                step,
                args,
                trainable_report,
                loss_history,
            )
            checkpoint_paths.append(ckpt)
            _log_event("eval_step_done", step=step, **eval_report, trainable_checkpoint=ckpt)
            _cleanup_cuda_memory(device, "post_eval_cleanup_done")

    final_checkpoint_reload_report = None
    if checkpoint_paths:
        final_checkpoint_reload_report = _load_trainable_checkpoint(model, Path(checkpoint_paths[-1]))
        if not args.no_visualizations:
            reloaded_eval = _mean_train_mse(model, vae_model, eval_loader, device, model_dtype, args.eval_max_batches)
            final_checkpoint_reload_report["reloaded_mean_mse"] = reloaded_eval["mean_mse"]
            final_checkpoint_reload_report["expected_mean_mse"] = eval_reports[-1]["mean_mse"] if eval_reports else None
            if final_checkpoint_reload_report["expected_mean_mse"] is not None:
                final_checkpoint_reload_report["abs_diff"] = abs(
                    float(final_checkpoint_reload_report["reloaded_mean_mse"]) - float(final_checkpoint_reload_report["expected_mean_mse"])
                )

    review_pages = {}
    if not args.no_visualizations:
        review_pages = _write_training_review_pages(args.output_dir, visual_reports, args.steps)

    report = {
        "status": "passed",
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "variant": args.variant,
        "steps": args.steps,
        "eval_steps": args.eval_steps,
        "generation_parameters": _generation_parameters(args),
        "review_pages": review_pages,
        "checkpoint_load_report": checkpoint_load_report,
        "trainable_report": trainable_report,
        "raw_sequence_report": raw_sequence_report,
        "first_batch_report": first_batch_report,
        "first_grad_report": first_grad_report,
        "eval_reports": eval_reports,
        "visual_reports": visual_reports,
        "trainable_checkpoints": checkpoint_paths,
        "final_checkpoint_reload_report": final_checkpoint_reload_report,
        "loss_history_tail": loss_history[-20:],
        "fixed_visualization_set": str(selected_path),
    }
    if eval_reports:
        base_mse = eval_reports[0]["mean_mse"]
        final_mse = eval_reports[-1]["mean_mse"]
        report["mse_improvement"] = {"step_000_mean_mse": base_mse, "final_mean_mse": final_mse, "relative_drop": (base_mse - final_mse) / base_mse if base_mse else 0.0}
    _write_json(args.output_dir / "gold40_overfit_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P1 Gold40 overfit for Counterfactual CoT paired_cf_mvp.")
    parser.add_argument("--manifest", type=Path, default=FIXED_GOLD40_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--llm-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vae-path", default=str(Path(DEFAULT_MODEL_PATH) / "ae.safetensors"))
    parser.add_argument("--checkpoint-path", default=str(Path(DEFAULT_MODEL_PATH) / "ema.safetensors"))
    parser.add_argument("--resume-trainable-checkpoint", type=Path, default=None)
    parser.add_argument("--variant", default="paired_cf_mvp", choices=["paired_cf_mvp"])
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=list(DEFAULT_EVAL_STEPS))
    parser.add_argument("--eval-max-batches", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--vit-min-image-size", type=int, default=224)
    parser.add_argument("--visual-image-size", type=int, default=1024)
    parser.add_argument("--visual-vae-min-image-size", type=int, default=512)
    parser.add_argument("--visual-vit-image-size", type=int, default=980)
    parser.add_argument("--visual-vit-min-image-size", type=int, default=224)
    parser.add_argument("--expected-num-tokens", type=int, default=2048)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=4096)
    parser.add_argument("--max-num-tokens", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--model-device-map", default="single", choices=("single", "auto", "manual_2gpu"))
    parser.add_argument("--device-map-max-memory", default="34GiB")
    parser.add_argument("--trainable-name-regex", default=DEFAULT_TRAINABLE_REGEX)
    parser.add_argument("--visual-total", type=int, default=12)
    parser.add_argument("--visual-per-type", type=int, default=3)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--no-require-paired-vit", dest="require_paired_vit", action="store_false")
    parser.set_defaults(require_paired_vit=True)
    args = parser.parse_args()
    args.eval_steps = sorted({int(step) for step in args.eval_steps if int(step) >= 0})
    if 0 not in args.eval_steps:
        args.eval_steps = [0] + args.eval_steps
    if args.steps not in args.eval_steps:
        args.eval_steps.append(args.steps)
        args.eval_steps = sorted(args.eval_steps)
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
