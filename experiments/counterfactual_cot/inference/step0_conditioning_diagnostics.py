from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import html
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image

from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from experiments.counterfactual_cot.data.render_prompts import render_intervention_text, render_preservation_text
from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.training.gold40_overfit import FIXED_GOLD40_MANIFEST, _resolve_image_path, _select_visualization_pairs
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2 import Qwen2Tokenizer


DEFAULT_MODEL_PATH = "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT"
DEFAULT_OUTPUT_DIR = Path("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/step0_conditioning_diagnostics")
CSV_FIELDS = [
    "pair_id",
    "checkpoint",
    "source_condition_verified",
    "target_leakage_found",
    "target_atom_success",
    "source_preservation_success",
    "source_swap_changes_output",
    "conflict_cot_changes_output",
    "no_cot_same_as_correct",
    "full_prompt_success",
    "delta_only_success",
    "diagnosis",
    "notes",
]
VALID_DIAGNOSES = {
    "uses_source_and_cot",
    "uses_source_but_ignores_cot",
    "uses_text_but_ignores_source",
    "copies_source_without_edit",
    "possible_target_leakage",
    "inconclusive",
}


def _log_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_fresh_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Diagnostic output directory already exists: {output_dir}. "
                "Refusing to reuse previous step0 diagnostics. Pass --overwrite to delete and recreate only this directory."
            )
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=False)


def _copy_resized_image(src: Path, dst: Path, image_size: int) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = pil_img2rgb(Image.open(src)).resize((image_size, image_size), Image.Resampling.LANCZOS)
    image.save(dst)
    return str(dst)


def _load_image(path: Path) -> Image.Image:
    return pil_img2rgb(Image.open(path))


def _assert_base_checkpoint(model_path: str, checkpoint_path: Path) -> None:
    expected = (Path(model_path) / "ema.safetensors").resolve()
    actual = checkpoint_path.resolve()
    if actual != expected:
        raise ValueError(f"Step0 diagnostics must load the original Base BAGEL checkpoint only. Expected {expected}, got {actual}")
    forbidden_parts = {"smoke", "checkpoint", "checkpoints", "trainable", "adapter", "lora"}
    if any(part.lower() in forbidden_parts for part in actual.parts):
        raise ValueError(f"Checkpoint path looks like a training/adaptation checkpoint, refusing to run: {actual}")


def _build_base_bagel(model_path: str, checkpoint_path: Path, precision: str) -> Tuple[Bagel, Any, Qwen2Tokenizer, Dict[str, int], str]:
    _assert_base_checkpoint(model_path, checkpoint_path)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 0:
        raise RuntimeError("Base BAGEL image diagnostics require at least one CUDA device.")
    device_map = infer_auto_device_map(
        model,
        max_memory={i: "80GiB" for i in range(gpu_count)},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for module_name in same_device_modules:
        device_map[module_name] = first_device if module_name in device_map else "cuda:0"
    _log_event(
        "base_bagel_checkpoint_load_start",
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=_sha256_file(checkpoint_path),
        model_path=model_path,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        torch_cuda_device_count=gpu_count,
        device_map_summary={str(k): str(v) for k, v in list(device_map.items())[:40]},
    )
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(checkpoint_path),
        device_map=device_map,
        offload_buffers=True,
        offload_folder="offload_step0_conditioning_diagnostics",
        dtype=dtype,
        force_hooks=True,
    ).eval()
    first_param_device = next(model.parameters()).device
    vae_model.to(device=first_param_device, dtype=dtype).eval()
    _log_event("base_bagel_checkpoint_load_done", checkpoint_path=str(checkpoint_path), first_param_device=str(first_param_device))
    return model, vae_model, tokenizer, new_token_ids, str(first_param_device)


def _full_prompt(pair: Dict[str, Any]) -> str:
    return (
        "[MODE=COUNTERFACTUAL_EDIT]\n"
        f"Source Prompt: {pair['source']['prompt']}\n"
        f"Source CoT: {pair['source']['cot_text']}\n"
        f"Intervention: {render_intervention_text(pair['intervention'])}\n"
        f"Target CoT: {pair['target']['cot_text']}\n"
        f"{render_preservation_text(pair.get('preservation', {}))}\n"
        "Generate the target image."
    )


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


def _generate_image(
    inferencer: InterleaveInferencer,
    source_image: Image.Image,
    prompt: str,
    output_path: Path,
    args: argparse.Namespace,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.generation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.generation_seed)
    result = inferencer.interleave_inference(
        input_lists=[source_image, prompt],
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
        raise RuntimeError(f"No image generated for {output_path}")
    images[-1].save(str(output_path))
    del result, images
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return str(output_path)


def _select_swap_pair(pair: Dict[str, Any], selected: Sequence[Dict[str, Any]], index: int) -> Dict[str, Any]:
    for offset in range(1, len(selected)):
        candidate = selected[(index + offset) % len(selected)]
        if candidate["pair_id"] != pair["pair_id"] and candidate.get("source", {}).get("prompt") != pair.get("source", {}).get("prompt"):
            return candidate
    return selected[(index + 1) % len(selected)]


def _rel(path: str | Path, root: Path) -> str:
    return os.path.relpath(str(path), str(root))


def _write_review_html(path: Path, rows: Sequence[Dict[str, Any]], output_dir: Path) -> None:
    headers = [
        "Source A",
        "Source B",
        "Golden target A",
        "Correct source + correct CoT",
        "Swapped source + same CoT",
        "Correct source + conflict CoT",
        "Correct source + no CoT",
        "Correct source + delta-only",
    ]
    cells = [
        "source_a_copy",
        "source_b_copy",
        "golden_target_a_copy",
        "correct_output",
        "swapped_source_output",
        "conflict_cot_output",
        "no_cot_output",
        "delta_only_output",
    ]
    style = """
    body { font-family: Arial, sans-serif; margin: 18px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 36px; }
    th, td { border: 1px solid #ccc; padding: 6px; vertical-align: top; }
    img { width: 180px; height: 180px; object-fit: contain; background: #f7f7f7; }
    .meta { font-size: 12px; line-height: 1.35; max-width: 1500px; white-space: pre-wrap; }
    .prompt { font-size: 11px; white-space: pre-wrap; max-height: 180px; overflow: auto; background: #f5f5f5; padding: 6px; }
    """
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Step0 Conditioning Diagnostics</title>", f"<style>{style}</style></head><body>"]
    parts.append("<h1>Step0 Conditioning Diagnostics</h1>")
    parts.append("<p>Manual review should judge atomic edit success and source preservation, not whole-image similarity alone.</p>")
    for row in rows:
        parts.append(f"<h2>{html.escape(row['pair_id'])}</h2>")
        atom = row.get("atomic_change", "")
        meta = {
            "pair_id": row["pair_id"],
            "intervention_type": row.get("intervention_type"),
            "atomic_change": atom,
            "source_sha256": row.get("source_sha256"),
            "target_sha256": row.get("target_sha256"),
            "condition_sha256": row.get("condition_sha256"),
            "source_condition_verified": row.get("source_condition_verified"),
            "target_image_leakage_found": row.get("target_leakage_found"),
            "checkpoint": row.get("checkpoint"),
        }
        parts.append(f"<pre class='meta'>{html.escape(json.dumps(meta, ensure_ascii=False, indent=2))}</pre>")
        parts.append("<table><thead><tr>")
        for header in headers:
            parts.append(f"<th>{html.escape(header)}</th>")
        parts.append("</tr></thead><tbody><tr>")
        for key in cells:
            rel = html.escape(_rel(row[key], output_dir))
            parts.append(f"<td><a href='{rel}'><img src='{rel}'></a><br><code>{rel}</code></td>")
        parts.append("</tr></tbody></table>")
        parts.append("<details><summary>Prompts and sequence plans</summary>")
        for key in ["correct_prompt", "conflict_prompt", "no_cot_prompt", "delta_only_prompt", "sequence_plan"]:
            value = row.get(key)
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            parts.append(f"<h3>{html.escape(key)}</h3><pre class='prompt'>{html.escape(value)}</pre>")
        parts.append("</details>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    checkpoint_path = Path(args.checkpoint_path)
    _assert_base_checkpoint(args.model_path, checkpoint_path)
    _prepare_fresh_output_dir(args.output_dir, args.overwrite)
    records = load_jsonl(args.manifest)
    selected = _select_visualization_pairs(records, args.visual_per_type, args.visual_total)
    model, vae_model, tokenizer, new_token_ids, first_device = _build_base_bagel(args.model_path, checkpoint_path, args.precision)
    vae_transform = ImageTransform(args.visual_image_size, args.visual_vae_min_image_size, 16)
    vit_transform = ImageTransform(args.visual_vit_image_size, args.visual_vit_min_image_size, 14)
    inferencer = InterleaveInferencer(model=model, vae_model=vae_model, tokenizer=tokenizer, vae_transform=vae_transform, vit_transform=vit_transform, new_token_ids=new_token_ids)
    manifest_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    checkpoint_sha = _sha256_file(checkpoint_path)
    for index, pair in enumerate(selected):
        pair_id = pair["pair_id"]
        swap_pair = _select_swap_pair(pair, selected, index)
        pair_dir = args.output_dir / "images" / pair_id
        source_path = _resolve_image_path(pair, pair["source"]["image"])
        target_path = _resolve_image_path(pair, pair["target"]["image"])
        source_b_path = _resolve_image_path(swap_pair, swap_pair["source"]["image"])
        source_sha = _sha256_file(source_path)
        target_sha = _sha256_file(target_path)
        condition_sha = _sha256_file(source_path)
        source_condition_verified = condition_sha == source_sha and condition_sha != target_sha
        if not source_condition_verified:
            raise RuntimeError(f"Condition image SHA check failed for {pair_id}: condition={condition_sha}, source={source_sha}, target={target_sha}")
        correct_prompt = _full_prompt(pair)
        conflict_prompt, conflict_value = _conflict_prompt(pair)
        no_cot_prompt = _no_cot_prompt(pair)
        delta_only_prompt = _delta_only_prompt(pair)
        sequence_plan = {
            "correct": [{"type": "image", "role": "condition_source", "path": str(source_path), "sha256": source_sha}, {"type": "text", "role": "prompt", "variant": "full_correct"}],
            "swapped_source": [{"type": "image", "role": "condition_source_b", "path": str(source_b_path), "sha256": _sha256_file(source_b_path)}, {"type": "text", "role": "prompt", "variant": "same_full_correct_from_source_a"}],
            "conflict_cot": [{"type": "image", "role": "condition_source", "path": str(source_path), "sha256": source_sha}, {"type": "text", "role": "prompt", "variant": "conflict_cot"}],
            "no_cot": [{"type": "image", "role": "condition_source", "path": str(source_path), "sha256": source_sha}, {"type": "text", "role": "prompt", "variant": "no_cot"}],
            "delta_only": [{"type": "image", "role": "condition_source", "path": str(source_path), "sha256": source_sha}, {"type": "text", "role": "prompt", "variant": "delta_only"}],
            "target_image": {"role": "reference_only_not_in_input_sequence", "path": str(target_path), "sha256": target_sha},
        }
        target_image_leakage_found = any(
            step.get("sha256") == target_sha
            for plan in [sequence_plan["correct"], sequence_plan["swapped_source"], sequence_plan["conflict_cot"], sequence_plan["no_cot"], sequence_plan["delta_only"]]
            for step in plan
            if isinstance(step, dict)
        )
        if target_image_leakage_found:
            raise RuntimeError(f"Target image leakage detected in input sequence for {pair_id}")
        source_a_copy = _copy_resized_image(source_path, pair_dir / "source_a.png", args.visual_image_size)
        source_b_copy = _copy_resized_image(source_b_path, pair_dir / "source_b.png", args.visual_image_size)
        golden_copy = _copy_resized_image(target_path, pair_dir / "golden_target_a.png", args.visual_image_size)
        source_a_image = _load_image(source_path)
        source_b_image = _load_image(source_b_path)
        _log_event(
            "diagnostic_sample_start",
            index=index,
            pair_id=pair_id,
            source_image=str(source_path),
            target_image=str(target_path),
            source_sha256=source_sha,
            target_sha256=target_sha,
            condition_sha256=condition_sha,
            source_condition_verified=source_condition_verified,
            output_dir=str(pair_dir),
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha,
            intervention=pair.get("intervention"),
            swap_source_pair_id=swap_pair["pair_id"],
        )
        correct_output = _generate_image(inferencer, source_a_image, correct_prompt, pair_dir / "correct_source_correct_cot.png", args)
        swapped_output = _generate_image(inferencer, source_b_image, correct_prompt, pair_dir / "swapped_source_same_cot.png", args)
        conflict_output = _generate_image(inferencer, source_a_image, conflict_prompt, pair_dir / "correct_source_conflict_cot.png", args)
        no_cot_output = _generate_image(inferencer, source_a_image, no_cot_prompt, pair_dir / "correct_source_no_cot.png", args)
        delta_output = _generate_image(inferencer, source_a_image, delta_only_prompt, pair_dir / "correct_source_delta_only.png", args)
        row = {
            "pair_id": pair_id,
            "intervention_type": pair.get("intervention", {}).get("intervention_type"),
            "atomic_change": render_intervention_text(pair["intervention"]),
            "conflict_value": conflict_value,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_model_path": args.model_path,
            "first_model_device": first_device,
            "source_image": str(source_path),
            "target_image": str(target_path),
            "source_b_image": str(source_b_path),
            "source_sha256": source_sha,
            "target_sha256": target_sha,
            "source_b_sha256": _sha256_file(source_b_path),
            "condition_sha256": condition_sha,
            "source_condition_verified": source_condition_verified,
            "target_leakage_found": target_image_leakage_found,
            "sequence_plan": sequence_plan,
            "correct_prompt": correct_prompt,
            "conflict_prompt": conflict_prompt,
            "no_cot_prompt": no_cot_prompt,
            "delta_only_prompt": delta_only_prompt,
            "source_a_copy": source_a_copy,
            "source_b_copy": source_b_copy,
            "golden_target_a_copy": golden_copy,
            "correct_output": correct_output,
            "swapped_source_output": swapped_output,
            "conflict_cot_output": conflict_output,
            "no_cot_output": no_cot_output,
            "delta_only_output": delta_output,
            "generation_parameters": {
                "seed": args.generation_seed,
                "visual_image_size": args.visual_image_size,
                "visual_vae_min_image_size": args.visual_vae_min_image_size,
                "visual_vit_image_size": args.visual_vit_image_size,
                "visual_vit_min_image_size": args.visual_vit_min_image_size,
                "cfg_text_scale": args.cfg_text_scale,
                "cfg_img_scale": args.cfg_img_scale,
                "cfg_interval": [args.cfg_interval_start, args.cfg_interval_end],
                "timestep_shift": args.timestep_shift,
                "num_timesteps": args.num_timesteps,
                "cfg_renorm_min": args.cfg_renorm_min,
                "cfg_renorm_type": args.cfg_renorm_type,
            },
        }
        manifest_rows.append(row)
        csv_rows.append(
            {
                "pair_id": pair_id,
                "checkpoint": str(checkpoint_path),
                "source_condition_verified": str(source_condition_verified).lower(),
                "target_leakage_found": str(target_image_leakage_found).lower(),
                "target_atom_success": "manual_review_required",
                "source_preservation_success": "manual_review_required",
                "source_swap_changes_output": "manual_review_required",
                "conflict_cot_changes_output": "manual_review_required",
                "no_cot_same_as_correct": "manual_review_required",
                "full_prompt_success": "manual_review_required",
                "delta_only_success": "manual_review_required",
                "diagnosis": "inconclusive",
                "notes": "Manual review required. Judge atomic edit and source preservation; do not rely on whole-image similarity.",
            }
        )
        _log_event("diagnostic_sample_done", index=index, pair_id=pair_id, outputs={k: row[k] for k in ["correct_output", "swapped_source_output", "conflict_cot_output", "no_cot_output", "delta_only_output"]})
    manifest_path = args.output_dir / "diagnostic_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    csv_path = args.output_dir / "diagnostic_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    review_path = args.output_dir / "review.html"
    _write_review_html(review_path, manifest_rows, args.output_dir)
    report = {
        "status": "ready_for_manual_review",
        "output_dir": str(args.output_dir),
        "manifest": str(manifest_path),
        "csv": str(csv_path),
        "review_html": str(review_path),
        "num_pairs": len(manifest_rows),
        "valid_diagnoses": sorted(VALID_DIAGNOSES),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
    }
    _log_event("step0_conditioning_diagnostics_done", **report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent Base BAGEL step-0 conditioning diagnostics for Gold40.")
    parser.add_argument("--manifest", type=Path, default=FIXED_GOLD40_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=str(Path(DEFAULT_MODEL_PATH) / "ema.safetensors"))
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--visual-total", type=int, default=12)
    parser.add_argument("--visual-per-type", type=int, default=3)
    parser.add_argument("--visual-image-size", type=int, default=1024)
    parser.add_argument("--visual-vae-min-image-size", type=int, default=512)
    parser.add_argument("--visual-vit-image-size", type=int, default=980)
    parser.add_argument("--visual-vit-min-image-size", type=int, default=224)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
