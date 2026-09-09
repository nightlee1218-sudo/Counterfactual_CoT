"""Deterministic fixed evaluation for Gold40 counterfactual overfit runs.

This tool is intentionally read-only over existing checkpoints. Each pair is
evaluated in a batch-of-one setting with per-pair deterministic seeds so that:

* The training-side flow-matching ``timestep`` sampled by ``np.random.randn()``
  in ``PackedDataset.pack_sequence`` is fixed for a given ``pair_id``.
* The forward-side ``noise = torch.randn_like(clean_latent)`` inside
  ``Bagel.forward`` is fixed for the same ``pair_id`` because both CPU and CUDA
  RNG states are re-seeded right before ``model(**data)``.

For every checkpoint the model weights are reloaded independently from the same
Base BAGEL snapshot (via ``_build_model_for_gold40`` + ``_load_trainable_checkpoint``)
so that checkpoint updates do not stack.

Outputs:
  * ``fixed_eval_report.json`` with mean/per-type/per-pair MSE per checkpoint,
    an internal replay for consistency, checkpoint metadata (path, SHA256,
    global_step), and per-pair seed/timestep/noise-hash traces.
  * ``review.html`` combining Source / step_XXX / Golden target for four fixed
    diagnostic pairs across the requested checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import html
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from data.data_utils import pil_img2rgb
from data.dataset_base import DataConfig, PackedDataset, SimpleCustomBatch
from experiments.counterfactual_cot.data.render_prompts import paired_prompt
from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.datasets.bagel_manifest_dataset import CounterfactualCoTDataset
from experiments.counterfactual_cot.training.gold40_overfit import (
    FIXED_GOLD40_MANIFEST,
    PRIMARY_DIAGNOSTIC_TYPES,
    _autocast_for_model,
    _assert_float_input_dtype,
    _build_model_for_gold40,
    _forward_loss,
    _generate_one,
    _load_pair_image,
    _load_tokenizer,
    _load_trainable_checkpoint,
    _make_inferencer,
    _move_batch_to_device,
    _prepare_batch,
    _resolve_image_path,
    _save_reference_image,
    _select_visualization_pairs,
    _transform_args,
)


DEFAULT_CHECKPOINT_STEPS = (0, 25, 50, 100, 200, 300, 400)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def _stable_pair_seed(pair_id: str, salt: int) -> int:
    digest = hashlib.md5(f"{salt}:{pair_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_hash(tensor: torch.Tensor) -> str:
    # Cast to float32 and take a stable byte view to hash regardless of dtype.
    arr = tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _log(event: str, **fields: Any) -> None:
    fields["event"] = event
    print(json.dumps(fields, ensure_ascii=False, sort_keys=True), flush=True)


# ---------------------------------------------------------------------------
# Packing a single pair into a deterministic batch of one
# ---------------------------------------------------------------------------


def _build_singleton_packed_dataset(args: argparse.Namespace, tokenizer: Any, special_tokens: Dict[str, int]) -> PackedDataset:
    """Return a ``PackedDataset`` we only use for its packer helpers.

    ``__iter__`` is never invoked; we call ``set_sequence_status`` /
    ``pack_sequence`` / ``to_tensor`` directly for controlled single-sample
    packing.
    """
    dataset_meta = {
        "counterfactual_cot_gold40_fixed_eval": {
            "dataset_names": ["gold40_fixed_eval"],
            "weight": 1.0,
            "is_mandatory": True,
            "image_transform_args": _transform_args(args.image_size, args.image_size, 16),
            "vit_image_transform_args": _transform_args(args.vit_image_size, args.vit_min_image_size, 14),
            "num_used_data": [1],
            "max_records": 1,
        }
    }
    # We register a dummy dataset name to satisfy DATASET_REGISTRY lookup inside
    # PackedDataset; however PackedDataset.__iter__ is never invoked so the
    # actual dataset behaviour is irrelevant.
    from data.dataset_info import DATASET_INFO, DATASET_REGISTRY

    DATASET_REGISTRY["counterfactual_cot_gold40_fixed_eval"] = CounterfactualCoTDataset
    DATASET_INFO["counterfactual_cot_gold40_fixed_eval"] = {
        "gold40_fixed_eval": {
            "manifest_path": str(args.manifest),
            "variant": args.variant,
            "training_seed": args.seed,
            "num_total_samples": 1,
            "max_records": 1,
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
    packed = PackedDataset(
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
        max_buffer_size=1,
        interpolate_pos=False,
        use_flex=False,
        data_status=None,
    )
    return packed


def _build_raw_sample_dataset(args: argparse.Namespace, tokenizer: Any) -> CounterfactualCoTDataset:
    """Build a raw ``CounterfactualCoTDataset`` we only use for ``_paired_sample``."""
    from data.transforms import ImageTransform

    transform_args = _transform_args(args.image_size, args.image_size, 16)
    vit_transform_args = _transform_args(args.vit_image_size, args.vit_min_image_size, 14)
    image_transform = ImageTransform(**transform_args)
    vit_image_transform = ImageTransform(**vit_transform_args)
    dataset = CounterfactualCoTDataset(
        dataset_name="gold40_fixed_eval",
        transform=image_transform,
        tokenizer=tokenizer,
        manifest_path=str(args.manifest),
        variant=args.variant,
        include_source_image=True,
        training_seed=args.seed,
        local_rank=0,
        world_size=1,
        num_workers=0,
        data_status=None,
        vit_transform=vit_image_transform,
        max_records=args.num_samples,
    )
    return dataset


def _pack_single_pair(
    packer: PackedDataset,
    raw_dataset: CounterfactualCoTDataset,
    pair: Dict[str, Any],
    pair_seed: int,
) -> Tuple[SimpleCustomBatch, Dict[str, Any]]:
    """Return a batch containing exactly one deterministic sample for ``pair``."""
    # Ensure image loading + numpy timestep sampling below are both deterministic.
    _set_deterministic_seed(pair_seed)
    sample, side, mode = raw_dataset._paired_sample(pair)
    sample["data_indexes"] = {
        "data_indexes": [0, 0],
        "worker_id": 0,
        "dataset_name": raw_dataset.dataset_name,
        "pair_id": pair.get("pair_id", ""),
        "mode": mode,
        "side": side,
    }
    # Re-seed right before packing so ``np.random.randn()`` inside
    # ``pack_sequence`` for the loss VAE image is pair-deterministic and
    # independent of ``_paired_sample`` internals.
    _set_deterministic_seed(pair_seed)
    status = packer.set_sequence_status()
    status = packer.pack_sequence(sample, status)
    data = packer.to_tensor(status)
    data["batch_data_indexes"] = [sample["data_indexes"]]
    batch = SimpleCustomBatch([data])
    trace = {
        "pair_id": pair.get("pair_id", ""),
        "intervention_type": pair.get("intervention", {}).get("intervention_type"),
        "pair_seed": int(pair_seed),
        "packed_timesteps": [float(v) for v in data.get("packed_timesteps", torch.tensor([])).tolist()],
        "packed_timesteps_hash": _tensor_hash(data["packed_timesteps"]) if "packed_timesteps" in data else None,
        "num_vae_tokens": int(len(data["packed_vae_token_indexes"])) if "packed_vae_token_indexes" in data else 0,
        "num_vit_tokens": int(len(data["packed_vit_token_indexes"])) if "packed_vit_token_indexes" in data else 0,
        "sequence_length": int(data["sequence_length"]),
    }
    return batch, trace


# ---------------------------------------------------------------------------
# Deterministic forward
# ---------------------------------------------------------------------------


def _deterministic_forward_mse(
    model: torch.nn.Module,
    vae_model: torch.nn.Module,
    device: torch.device,
    model_dtype: torch.dtype,
    batch: SimpleCustomBatch,
    pair_seed: int,
) -> Tuple[float, str]:
    """Return (mse, noise_hash) using deterministic forward-time noise sampling."""
    data = _prepare_batch(batch, vae_model, device, model_dtype)
    # Re-seed once more so that ``torch.randn_like`` inside ``Bagel.forward``
    # produces deterministic noise per pair.
    _set_deterministic_seed(pair_seed)
    # Capture the noise that will be drawn inside forward by cloning the current
    # CUDA rng state, sampling a matching tensor for hashing, and restoring the
    # rng state so that the actual forward call re-draws the same values.
    latent_shape = data.get("padded_latent").shape if "padded_latent" in data else None
    noise_hash = None
    if latent_shape is not None:
        # Compute latent -> patchified length exactly like Bagel.forward.
        # We do this only for hashing so we don't need to reproduce the exact
        # layout; we just record a canonical noise sample controlled by the
        # same seed.
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        probe = torch.randn(latent_shape, device=device, dtype=torch.float32)
        noise_hash = _tensor_hash(probe)
        del probe
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    with torch.no_grad():
        _, scalars = _forward_loss(model, data)
    del data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return float(scalars["mse"]), noise_hash


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation
# ---------------------------------------------------------------------------


def _evaluate_checkpoint(
    step_label: str,
    trainable_ckpt: Optional[Path],
    args: argparse.Namespace,
    tokenizer: Any,
    special_tokens: Dict[str, int],
    num_new_tokens: int,
    packer: PackedDataset,
    raw_dataset: CounterfactualCoTDataset,
    device: torch.device,
    ordered_pairs: Sequence[Dict[str, Any]],
    replay_index: int,
) -> Dict[str, Any]:
    """Load Base BAGEL, optionally overlay a trainable checkpoint, then eval all pairs."""
    _log(
        "fixed_eval_checkpoint_load_start",
        step_label=step_label,
        trainable_checkpoint=str(trainable_ckpt) if trainable_ckpt else None,
        replay_index=replay_index,
    )
    tokenizer_size = len(tokenizer)
    model, vae_model, checkpoint_load_report = _build_model_for_gold40(
        args, device, tokenizer_size=tokenizer_size, num_new_tokens=num_new_tokens,
    )
    # forward_train is dispatched by ``self.training``; keep gradients disabled
    # via ``torch.no_grad`` inside ``_deterministic_forward_mse`` but stay in
    # ``model.train()`` so the flow-matching loss path is executed.
    model.train()
    vae_model.eval()
    model_dtype = next(model.parameters()).dtype
    resume_report: Optional[Dict[str, Any]] = None
    sha256_value = None
    global_step_value = None
    if trainable_ckpt is not None:
        resume_report = _load_trainable_checkpoint(model, trainable_ckpt)
        sha256_value = _sha256_file(trainable_ckpt)
        global_step_value = resume_report.get("global_step")
    else:
        # step_base: Base BAGEL only; no trainable overlay.
        sha256_value = None
        global_step_value = 0

    per_pair: List[Dict[str, Any]] = []
    per_type: Dict[str, List[float]] = {}
    all_losses: List[float] = []
    for index, pair in enumerate(ordered_pairs):
        pair_id = pair.get("pair_id", f"pair_{index:03d}")
        pair_seed = _stable_pair_seed(pair_id, args.seed)
        batch, trace = _pack_single_pair(packer, raw_dataset, pair, pair_seed)
        mse_value, noise_hash = _deterministic_forward_mse(
            model, vae_model, device, model_dtype, batch, pair_seed,
        )
        intervention_type = trace["intervention_type"] or "unknown"
        per_type.setdefault(intervention_type, []).append(mse_value)
        all_losses.append(mse_value)
        per_pair.append(
            {
                "pair_id": pair_id,
                "intervention_type": intervention_type,
                "mse": mse_value,
                "pair_seed": trace["pair_seed"],
                "packed_timesteps": trace["packed_timesteps"],
                "packed_timesteps_hash": trace["packed_timesteps_hash"],
                "forward_noise_hash": noise_hash,
                "num_vae_tokens": trace["num_vae_tokens"],
                "num_vit_tokens": trace["num_vit_tokens"],
                "sequence_length": trace["sequence_length"],
            }
        )
        _log(
            "fixed_eval_pair_done",
            step_label=step_label,
            replay_index=replay_index,
            index=index,
            pair_id=pair_id,
            intervention_type=intervention_type,
            mse=mse_value,
        )

    mean_by_type = {name: (sum(values) / len(values)) for name, values in per_type.items()}
    mean_mse = sum(all_losses) / max(1, len(all_losses))
    result = {
        "step_label": step_label,
        "trainable_checkpoint": str(trainable_ckpt) if trainable_ckpt else None,
        "trainable_checkpoint_sha256": sha256_value,
        "global_step": global_step_value,
        "checkpoint_load_report": checkpoint_load_report,
        "trainable_checkpoint_report": resume_report,
        "num_pairs": len(per_pair),
        "mean_mse": mean_mse,
        "mean_mse_by_intervention_type": mean_by_type,
        "per_pair": per_pair,
        "replay_index": replay_index,
    }

    del model, vae_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log(
        "fixed_eval_checkpoint_done",
        step_label=step_label,
        replay_index=replay_index,
        mean_mse=mean_mse,
        mean_mse_by_intervention_type=mean_by_type,
    )
    return result


# ---------------------------------------------------------------------------
# Review HTML generation
# ---------------------------------------------------------------------------


def _reuse_or_generate_step_image(
    step_label: str,
    trainable_ckpt: Optional[Path],
    args: argparse.Namespace,
    tokenizer: Any,
    special_tokens: Dict[str, int],
    num_new_tokens: int,
    device: torch.device,
    review_pairs: Sequence[Dict[str, Any]],
    output_root: Path,
    reuse_map: Dict[str, Dict[str, Path]],
) -> Dict[str, str]:
    """Ensure per-review-pair step images exist. Reuse if provided, else render.

    ``reuse_map[pair_id][step_label]`` may point to an already existing PNG.
    """
    outputs: Dict[str, str] = {}
    missing: List[Dict[str, Any]] = []
    for pair in review_pairs:
        pair_id = pair.get("pair_id", "")
        target_path = output_root / "review_images" / pair_id / f"model_{step_label}.png"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        existing = reuse_map.get(pair_id, {}).get(step_label)
        if existing and Path(existing).exists() and Path(existing).stat().st_size > 0:
            if Path(existing).resolve() != target_path.resolve():
                shutil.copyfile(existing, target_path)
            outputs[pair_id] = str(target_path)
        else:
            missing.append(pair)
    if not missing:
        return outputs

    # We only reach here for step labels we still need to render.
    tokenizer_size = len(tokenizer)
    model, vae_model, _ = _build_model_for_gold40(
        args, device, tokenizer_size=tokenizer_size, num_new_tokens=num_new_tokens,
    )
    model.eval()
    vae_model.eval()
    if trainable_ckpt is not None:
        _load_trainable_checkpoint(model, trainable_ckpt)
    inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, args)
    try:
        for pair in missing:
            pair_id = pair.get("pair_id", "")
            target_path = output_root / "review_images" / pair_id / f"model_{step_label}.png"
            _generate_one(
                inferencer,
                pair,
                target_path,
                args,
                seed=args.generation_seed,
            )
            outputs[pair_id] = str(target_path)
            _log("fixed_eval_review_render_done", step_label=step_label, pair_id=pair_id, output=str(target_path))
    finally:
        del inferencer, model, vae_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs


def _generate_review_html(
    output_root: Path,
    step_labels: Sequence[str],
    review_pairs: Sequence[Dict[str, Any]],
    per_step_images: Dict[str, Dict[str, str]],
    per_step_mse_lookup: Dict[str, Dict[str, float]],
    review_score_fields: Sequence[str],
) -> Path:
    style = (
        "body{font-family:Arial;background:#111;color:#eee;margin:24px}"
        "table{border-collapse:collapse;margin-bottom:32px}"
        "th,td{border:1px solid #555;padding:8px;text-align:center;vertical-align:top}"
        "img{max-width:180px;max-height:180px;object-fit:contain;background:#000}"
        "th{background:#222}"
        ".score-table td{min-width:60px}"
        "h2{margin-top:32px;color:#fff}"
        ".mse{color:#8f8;font-family:monospace}"
        ".meta{color:#aaa;font-size:12px}"
    )
    parts: List[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Fixed deterministic eval review</title><style>{style}</style>",
        "</head><body>",
        "<h1>Fixed deterministic eval review</h1>",
        "<p>Reviewer records TS (target_success), PS (preservation_score), BA (binding_accuracy), LS (locality_score), IQ (image_quality) per pair per checkpoint.</p>",
    ]
    for pair in review_pairs:
        pair_id = pair.get("pair_id", "")
        intervention_type = pair.get("intervention", {}).get("intervention_type", "unknown")
        source_path = output_root / "review_images" / pair_id / "source.png"
        golden_path = output_root / "review_images" / pair_id / "golden_target.png"
        parts.append(f"<h2>{html.escape(pair_id)} <span class='meta'>[{html.escape(intervention_type)}]</span></h2>")
        parts.append("<table><tr><th>Source</th>")
        for step_label in step_labels:
            mse_value = per_step_mse_lookup.get(step_label, {}).get(pair_id)
            mse_html = f"<div class='mse'>MSE={mse_value:.4f}</div>" if mse_value is not None else "<div class='mse'>MSE=?</div>"
            parts.append(f"<th>{html.escape(step_label)}{mse_html}</th>")
        parts.append("<th>Golden</th></tr>")
        parts.append("<tr>")
        parts.append(f"<td><img src='file://{source_path}'></td>")
        for step_label in step_labels:
            img_path = per_step_images.get(step_label, {}).get(pair_id)
            if img_path:
                parts.append(f"<td><img src='file://{img_path}'></td>")
            else:
                parts.append("<td>-</td>")
        parts.append(f"<td><img src='file://{golden_path}'></td>")
        parts.append("</tr></table>")

        # Score table for manual review.
        parts.append("<table class='score-table'><tr><th>Metric</th>")
        for step_label in step_labels:
            parts.append(f"<th>{html.escape(step_label)}</th>")
        parts.append("</tr>")
        for field in review_score_fields:
            parts.append(f"<tr><td>{html.escape(field)}</td>")
            for _ in step_labels:
                parts.append("<td>&nbsp;</td>")
            parts.append("</tr>")
        parts.append("</table>")
    parts.append("</body></html>")
    html_path = output_root / "review.html"
    html_path.write_text("".join(parts), encoding="utf-8")
    return html_path


# ---------------------------------------------------------------------------
# Argument parsing & main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic fixed evaluation for Gold40 counterfactual training.")
    parser.add_argument("--manifest", type=Path, default=FIXED_GOLD40_MANIFEST)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True,
                        help="Directory containing step_XXX_trainable.pt files.")
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=list(DEFAULT_CHECKPOINT_STEPS))
    parser.add_argument("--include-base", action="store_true",
                        help="Additionally evaluate Base BAGEL with no trainable overlay as 'base'.")
    parser.add_argument("--llm-path", default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
    parser.add_argument("--vae-path", default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors")
    parser.add_argument("--checkpoint-path", default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors")
    parser.add_argument("--variant", default="paired_cf_mvp", choices=["paired_cf_mvp"])
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--vit-min-image-size", type=int, default=224)
    parser.add_argument("--expected-num-tokens", type=int, default=1536)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=1536)
    parser.add_argument("--max-num-tokens", type=int, default=1536)
    parser.add_argument("--visual-image-size", type=int, default=1024)
    parser.add_argument("--visual-vae-min-image-size", type=int, default=512)
    parser.add_argument("--visual-vit-image-size", type=int, default=980)
    parser.add_argument("--visual-vit-min-image-size", type=int, default=224)
    parser.add_argument("--visual-total", type=int, default=4)
    parser.add_argument("--visual-per-type", type=int, default=1)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=10)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-device-map", default="single")
    parser.add_argument("--device-map-max-memory", default="34GiB")
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--consistency-replays", type=int, default=2,
                        help="Number of independent re-evaluations to measure determinism.")
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--reuse-images-from", type=Path, nargs="*", default=[],
                        help="Optional acceptance_eval-style step000/step400 folders whose model_step_000.png can be reused as step_000 / step_400 review images.")
    parser.add_argument("--num-timesteps-review", type=int, default=None,
                        help="Override --num-timesteps only for review image generation.")
    return parser.parse_args()


def _resolve_ckpt_paths(args: argparse.Namespace) -> Dict[str, Optional[Path]]:
    resolved: Dict[str, Optional[Path]] = {}
    if args.include_base:
        resolved["base"] = None
    for step in args.checkpoint_steps:
        candidate = args.checkpoint_root / f"step_{step:03d}_trainable.pt"
        if not candidate.exists():
            raise FileNotFoundError(f"Missing checkpoint: {candidate}")
        resolved[f"step_{step:03d}"] = candidate
    return resolved


def _resolve_review_pairs(records: Sequence[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    return _select_visualization_pairs(records, args.visual_per_type, args.visual_total)


def _copy_source_and_golden(review_pairs: Sequence[Dict[str, Any]], args: argparse.Namespace, output_root: Path) -> None:
    for pair in review_pairs:
        pair_id = pair.get("pair_id", "")
        pair_dir = output_root / "review_images" / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        _save_reference_image(pair, "source", pair_dir / "source.png", args.visual_image_size)
        _save_reference_image(pair, "target", pair_dir / "golden_target.png", args.visual_image_size)


def _load_reuse_map(args: argparse.Namespace, review_pairs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Path]]:
    """Parse optional acceptance_eval directories for pre-rendered images."""
    result: Dict[str, Dict[str, Path]] = {pair.get("pair_id", ""): {} for pair in review_pairs}
    for reuse_root in args.reuse_images_from or []:
        step_dir = reuse_root / "visualizations" / "step_000"
        # We treat directory name (step000, step400, ...) as the label mapping source.
        label = reuse_root.name
        if not step_dir.exists():
            continue
        for pair in review_pairs:
            pair_id = pair.get("pair_id", "")
            src = step_dir / pair_id / "model_step_000.png"
            if src.exists() and src.stat().st_size > 0:
                # Map label "step000" -> "step_000".
                if label.startswith("step") and label[4:].isdigit():
                    normalized = f"step_{int(label[4:]):03d}"
                else:
                    normalized = label
                result[pair_id][normalized] = src
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints_placeholder").mkdir(exist_ok=True)

    device = torch.device(args.device)
    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)

    records = load_jsonl(args.manifest)
    if len(records) != args.num_samples:
        raise ValueError(f"Expected {args.num_samples} records, found {len(records)} in {args.manifest}")

    # Ordered pairs: sorted by pair_id for reproducible eval order across replays.
    ordered_pairs = sorted(records, key=lambda p: p.get("pair_id", ""))

    ckpt_paths = _resolve_ckpt_paths(args)
    _log(
        "fixed_eval_start",
        manifest=str(args.manifest),
        output_dir=str(args.output_dir),
        checkpoint_root=str(args.checkpoint_root),
        checkpoint_steps=args.checkpoint_steps,
        include_base=args.include_base,
        num_samples=args.num_samples,
        consistency_replays=args.consistency_replays,
        step_labels=list(ckpt_paths.keys()),
    )

    raw_dataset = _build_raw_sample_dataset(args, tokenizer)
    packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)

    replay_results: List[List[Dict[str, Any]]] = []
    for replay_index in range(args.consistency_replays):
        current_replay: List[Dict[str, Any]] = []
        for step_label, ckpt in ckpt_paths.items():
            result = _evaluate_checkpoint(
                step_label,
                ckpt,
                args,
                tokenizer,
                special_tokens,
                num_new_tokens,
                packer,
                raw_dataset,
                device,
                ordered_pairs,
                replay_index,
            )
            current_replay.append(result)
        replay_results.append(current_replay)

    # Consistency: compare replay 0 vs replay 1 on per-pair MSE.
    consistency: Dict[str, Dict[str, Any]] = {}
    if len(replay_results) >= 2:
        first_by_label = {r["step_label"]: r for r in replay_results[0]}
        second_by_label = {r["step_label"]: r for r in replay_results[1]}
        for label, first in first_by_label.items():
            second = second_by_label.get(label)
            if not second:
                continue
            diffs = []
            per_pair_diffs = {}
            for p1, p2 in zip(first["per_pair"], second["per_pair"]):
                diff = abs(p1["mse"] - p2["mse"])
                diffs.append(diff)
                per_pair_diffs[p1["pair_id"]] = {
                    "mse_r0": p1["mse"],
                    "mse_r1": p2["mse"],
                    "abs_diff": diff,
                    "packed_timesteps_hash_r0": p1["packed_timesteps_hash"],
                    "packed_timesteps_hash_r1": p2["packed_timesteps_hash"],
                    "forward_noise_hash_r0": p1["forward_noise_hash"],
                    "forward_noise_hash_r1": p2["forward_noise_hash"],
                    "timesteps_match": p1["packed_timesteps_hash"] == p2["packed_timesteps_hash"],
                    "noise_match": p1["forward_noise_hash"] == p2["forward_noise_hash"],
                }
            consistency[label] = {
                "max_abs_diff": max(diffs) if diffs else None,
                "mean_abs_diff": (sum(diffs) / len(diffs)) if diffs else None,
                "per_pair": per_pair_diffs,
            }

    # Build the primary summary tables based on replay 0.
    primary = replay_results[0]
    step_labels = [r["step_label"] for r in primary]
    per_step_mse_lookup = {r["step_label"]: {p["pair_id"]: p["mse"] for p in r["per_pair"]} for r in primary}

    # Review HTML for fixed diagnostic pairs.
    per_step_images: Dict[str, Dict[str, str]] = {}
    review_pairs = _resolve_review_pairs(records, args)
    review_html_path: Optional[Path] = None
    if not args.skip_review:
        _copy_source_and_golden(review_pairs, args, args.output_dir)
        reuse_map = _load_reuse_map(args, review_pairs)
        review_num_timesteps = args.num_timesteps_review or args.num_timesteps
        # Temporarily override num-timesteps only for image generation.
        original_num_timesteps = args.num_timesteps
        args.num_timesteps = review_num_timesteps
        try:
            for step_label, ckpt in ckpt_paths.items():
                imgs = _reuse_or_generate_step_image(
                    step_label,
                    ckpt,
                    args,
                    tokenizer,
                    special_tokens,
                    num_new_tokens,
                    device,
                    review_pairs,
                    args.output_dir,
                    reuse_map,
                )
                per_step_images[step_label] = imgs
        finally:
            args.num_timesteps = original_num_timesteps
        review_html_path = _generate_review_html(
            args.output_dir,
            step_labels,
            review_pairs,
            per_step_images,
            per_step_mse_lookup,
            review_score_fields=["TS", "PS", "BA", "LS", "IQ", "notes"],
        )

    fixed_report = {
        "status": "passed",
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "checkpoint_root": str(args.checkpoint_root),
        "step_labels": step_labels,
        "replays": replay_results,
        "consistency": consistency,
        "review_pairs": [
            {
                "pair_id": pair.get("pair_id", ""),
                "intervention_type": pair.get("intervention", {}).get("intervention_type"),
            }
            for pair in review_pairs
        ],
        "review_html": str(review_html_path) if review_html_path else None,
        "generation_parameters": {
            "cfg_text_scale": args.cfg_text_scale,
            "cfg_img_scale": args.cfg_img_scale,
            "cfg_interval": [args.cfg_interval_start, args.cfg_interval_end],
            "cfg_renorm_min": args.cfg_renorm_min,
            "cfg_renorm_type": args.cfg_renorm_type,
            "timestep_shift": args.timestep_shift,
            "num_timesteps": args.num_timesteps,
            "num_timesteps_review": args.num_timesteps_review or args.num_timesteps,
            "seed": args.generation_seed,
            "visual_image_size": args.visual_image_size,
        },
    }
    _write_json(args.output_dir / "fixed_eval_report.json", fixed_report)
    _log(
        "fixed_eval_finished",
        report=str(args.output_dir / "fixed_eval_report.json"),
        review_html=str(review_html_path) if review_html_path else None,
    )


if __name__ == "__main__":
    main()
