"""Single-pair overfit diagnostic for counterfactual CoT gold40 pairs.

The goal of this script is *not* to produce a shipping training run. It is a
diagnostic: repeatedly train a single ``pair_id`` for a bounded number of steps
and record everything needed to answer "does the current 63.45M trainable slice
have enough capacity to move the ball to the target region?"

For each step we:

* pack the same ``pair_id`` deterministically (batch of one, fixed seed);
* forward once, back-prop once, ``optimizer.step`` once;
* record per-trainable-tensor gradient norm and post-step parameter update norm
  (fp32 snapshot diff, taken on CPU so bf16 accumulation is not masked);
* at pre-declared steps, save a trainable-only checkpoint and render an
  inference image against the fixed source with the fixed generation seed.

At startup we also emit a diagnostic dump of:

* tokenizer-decoded conditioning text (verifies source CoT, target CoT and
  intervention atomic change are actually reaching the model);
* the sequence plan for the sample (verifies source is condition-only and only
  the golden target VAE image is loss-supervised);
* source / target image paths with SHA256 (verifies target VAE tokens come from
  the golden target file);
* the boolean mse_loss mask indices and their exact overlap with the target
  VAE token indices (verifies MSE mask covers only target tokens).
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from data.dataset_base import PackedDataset, SimpleCustomBatch
from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.datasets.bagel_manifest_dataset import CounterfactualCoTDataset
from experiments.counterfactual_cot.training.gold40_overfit import (
    FIXED_GOLD40_MANIFEST,
    DEFAULT_TRAINABLE_REGEX,
    _assert_float_input_dtype,
    _autocast_for_model,
    _build_model_for_gold40,
    _forward_loss,
    _generate_one,
    _generation_parameters,
    _load_pair_image,
    _load_tokenizer,
    _load_trainable_checkpoint,
    _make_inferencer,
    _move_batch_to_device,
    _prepare_batch,
    _random_state_payload,
    _resolve_image_path,
    _save_reference_image,
    _save_trainable_checkpoint,
    _set_seed,
    _serializable_args,
)
from experiments.counterfactual_cot.training.smoke_train import _set_trainable_parameters
from experiments.counterfactual_cot.training.fixed_deterministic_eval import (
    _build_raw_sample_dataset,
    _build_singleton_packed_dataset,
    _pack_single_pair,
    _stable_pair_seed,
)


DEFAULT_SAVE_STEPS = (0, 25, 50, 100, 200, 300, 500)
DEFAULT_PAIR_ID = (
    "cf_source_living_room_play_absolute_position_change_"
    "v4_absolute_region_replacement_v4_0000"
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _log(event: str, **fields: Any) -> None:
    fields["event"] = event
    print(json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str), flush=True)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _decode_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(list(token_ids), skip_special_tokens=False)
    except Exception as exc:  # noqa: BLE001
        return f"<decode_error: {exc}>"


def _snapshot_trainable_fp32(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return CPU fp32 copies of every trainable parameter, keyed by name."""
    snap: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            snap[name] = param.detach().to(device="cpu", dtype=torch.float32, copy=True)
    return snap


def _param_update_norms(before: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    """Per-tensor update statistics computed as ``||after - before||``."""
    result: Dict[str, Dict[str, float]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in before:
            continue
        after_cpu = param.detach().to(device="cpu", dtype=torch.float32)
        delta = after_cpu - before[name]
        flat = delta.reshape(-1)
        if flat.numel() == 0:
            continue
        max_abs = float(flat.abs().max().item())
        l2 = float(torch.linalg.vector_norm(flat).item())
        mean_abs = float(flat.abs().mean().item())
        rel_l2 = float(l2 / max(1e-12, float(torch.linalg.vector_norm(before[name].reshape(-1)).item())))
        result[name] = {
            "delta_max_abs": max_abs,
            "delta_l2": l2,
            "delta_mean_abs": mean_abs,
            "delta_l2_over_param_l2": rel_l2,
            "numel": int(flat.numel()),
        }
    return result


def _grad_norms(model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    """Per-tensor gradient statistics captured right after ``loss.backward()``."""
    result: Dict[str, Dict[str, float]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad = param.grad
        if grad is None:
            result[name] = {"grad_present": False, "grad_l2": 0.0, "grad_max_abs": 0.0, "grad_mean_abs": 0.0, "numel": int(param.numel())}
            continue
        g = grad.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        result[name] = {
            "grad_present": True,
            "grad_l2": float(torch.linalg.vector_norm(g).item()),
            "grad_max_abs": float(g.abs().max().item()),
            "grad_mean_abs": float(g.abs().mean().item()),
            "numel": int(g.numel()),
        }
    return result


# ---------------------------------------------------------------------------
# Diagnostics: text, plan, mask, image identity
# ---------------------------------------------------------------------------


def _tokenize_conditioning(tokenizer: Any, pair: Dict[str, Any], variant: str) -> Dict[str, Any]:
    """Return the exact prompt that will be fed to the model plus its token ids."""
    from experiments.counterfactual_cot.data.render_prompts import paired_prompt

    include_source = variant == "paired_cf_mvp"
    prompt = pair.get("training_prompts", {}).get("paired_edit" if include_source else "paired_generation")
    if not prompt:
        prompt = paired_prompt(pair, include_source_image=include_source)
    text_ids = tokenizer.encode(prompt)
    decoded = _decode_text(tokenizer, text_ids)
    return {
        "prompt_raw": prompt,
        "prompt_decoded_roundtrip": decoded,
        "token_ids_len": len(text_ids),
        "token_ids_head": list(text_ids[:32]),
    }


def _describe_pair_files(pair: Dict[str, Any]) -> Dict[str, Any]:
    """SHA256 + path check for source & target images used by the sample."""
    src_path = _resolve_image_path(pair, pair["source"]["image"])
    tgt_path = _resolve_image_path(pair, pair["target"]["image"])
    return {
        "source_image_path": str(src_path),
        "source_image_sha256": _sha256_file(src_path) if src_path.exists() else None,
        "source_image_size_bytes": src_path.stat().st_size if src_path.exists() else None,
        "target_image_path": str(tgt_path),
        "target_image_sha256": _sha256_file(tgt_path) if tgt_path.exists() else None,
        "target_image_size_bytes": tgt_path.stat().st_size if tgt_path.exists() else None,
    }


def _sequence_plan_summary(raw_sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "index": index,
            "type": item["type"],
            "need_loss": bool(item.get("loss", 0)),
            "enable_cfg": bool(item.get("enable_cfg", 0)),
        }
        for index, item in enumerate(raw_sample["sequence_plan"])
    ]


def _loss_mask_summary(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Compare mse_loss_indexes vs packed_vae_token_indexes and analyse the
    timestep vector both before and after the sigmoid transform that BAGEL
    applies inside ``forward`` before ``has_mse = packed_timesteps > 0``.

    Fields:
      - ``raw_positive_timestep_tokens``: count of tokens whose stored
        ``packed_timesteps`` value is > 0 *before* sigmoid. In BAGEL the
        target VAE image draws a *single* ``np.random.randn()`` and repeats it
        for every one of its 256 tokens, so this field can only be ``0`` (when
        that draw was <= 0) or ``256`` (when it was > 0). Source tokens hold
        ``-inf`` here and never count as positive.
      - ``post_sigmoid_positive_timestep_tokens``: count of tokens whose
        ``sigmoid(packed_timesteps)`` is > 0. Only ``-inf`` (source) maps to 0;
        every finite value (all 256 target tokens) maps into (0, 1) and
        therefore contributes to ``has_mse``. Expected value: ``256``.
      - ``effective_mse_tokens``: intersection of ``mse_loss_indexes`` and the
        set of tokens with ``post_sigmoid_timestep > 0`` -- this is the true
        number of tokens that will contribute to the MSE loss. Expected: 256.
    """
    mse_idx = data_dict.get("mse_loss_indexes")
    vae_idx = data_dict.get("packed_vae_token_indexes")
    if mse_idx is None or vae_idx is None:
        return {"has_mse_indexes": False}
    mse_set = set(int(x) for x in mse_idx.tolist())
    vae_set = set(int(x) for x in vae_idx.tolist())
    ts = data_dict.get("packed_timesteps")
    if ts is None:
        raw_pos = 0
        post_sig_pos = 0
        effective_mse_tokens = 0
        target_positions_post_sigmoid: List[int] = []
    else:
        ts_list = [float(v) for v in ts.tolist()]
        raw_pos = sum(1 for v in ts_list if v > 0.0)
        # sigmoid: 1 / (1 + exp(-v)); for v = -inf -> 0, otherwise strictly > 0.
        post_sig = [math.inf if math.isinf(v) and v > 0 else 0.0 if math.isinf(v) else 1.0 / (1.0 + math.exp(-v)) for v in ts_list]
        # Positions in the packed sequence that survive ``has_mse``.
        # ``ts`` is aligned with ``packed_vae_token_indexes`` (a token per vae
        # slot). We map i-th vae-slot -> global sequence index via ``vae_idx``.
        vae_idx_list = [int(x) for x in vae_idx.tolist()]
        post_positive_seq_positions = {
            vae_idx_list[i] for i, v in enumerate(post_sig) if v > 0.0
        }
        target_positions_post_sigmoid = sorted(post_positive_seq_positions)
        post_sig_pos = len(post_positive_seq_positions)
        effective_mse_tokens = len(mse_set & post_positive_seq_positions)
    return {
        "has_mse_indexes": True,
        "num_mse_indexes": len(mse_set),
        "num_vae_indexes": len(vae_set),
        "mse_is_subset_of_vae": mse_set.issubset(vae_set),
        "num_vae_indexes_not_in_mse": len(vae_set - mse_set),
        "num_mse_indexes_not_in_vae": len(mse_set - vae_set),
        "sample_mse_indexes_head": sorted(mse_set)[:8],
        "sample_mse_indexes_tail": sorted(mse_set)[-8:],
        "raw_positive_timestep_tokens": raw_pos,
        "post_sigmoid_positive_timestep_tokens": post_sig_pos,
        "effective_mse_tokens": effective_mse_tokens,
        "post_sigmoid_positive_positions_head": target_positions_post_sigmoid[:8],
        "post_sigmoid_positive_positions_tail": target_positions_post_sigmoid[-8:],
    }


def _snapshot_rng_state() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng_state(payload: Dict[str, Any]) -> None:
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if "torch_cuda_rng_state_all" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])


def _run_one_step_assertions(
    args: argparse.Namespace,
    model: torch.nn.Module,
    vae_model: torch.nn.Module,
    packer: PackedDataset,
    raw_dataset: CounterfactualCoTDataset,
    pair: Dict[str, Any],
    pair_seed: int,
    output_path: Path,
) -> Dict[str, Any]:
    """Run a real single-step forward and hard-assert four invariants:

    1. Every ``sigmoid(packed_timesteps)`` value for target tokens is > 0.
    2. ``effective_mse_tokens == 256`` (all target VAE tokens contribute).
    3. The returned ``mse`` tensor is present, finite, and strictly > 0.
    4. No source VAE token index appears in ``mse_loss_indexes``.

    Nothing is optimized; the model stays exactly as loaded. Random state for
    Python ``random``, NumPy, PyTorch CPU and CUDA is snapshotted before the
    forward and fully restored after so the diagnostic pass does not perturb
    the deterministic training trajectory.
    """
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # Snapshot every RNG we might touch during pack + forward.
    rng_snapshot = _snapshot_rng_state()
    try:
        batch, trace = _pack_single_pair(packer, raw_dataset, pair, pair_seed)
        packed_dict = batch.to_dict()

        # Identify target vs source VAE index sets. Source tokens carry -inf in
        # ``packed_timesteps`` while target tokens carry a finite value drawn
        # from ``np.random.randn()`` (the same draw repeated across all 256
        # target tokens).
        ts = packed_dict["packed_timesteps"]
        vae_idx = [int(x) for x in packed_dict["packed_vae_token_indexes"].tolist()]
        ts_list = [float(v) for v in ts.tolist()]
        source_positions = {vae_idx[i] for i, v in enumerate(ts_list) if math.isinf(v) and v < 0}
        target_positions = {vae_idx[i] for i, v in enumerate(ts_list) if not (math.isinf(v) and v < 0)}
        post_sigmoid_target = [
            1.0 / (1.0 + math.exp(-v)) for i, v in enumerate(ts_list)
            if vae_idx[i] in target_positions
        ]
        assert len(post_sigmoid_target) == len(target_positions), \
            "internal accounting error while extracting target sigmoid timesteps"

        # ---- Actual forward pass (no optimizer step, no autograd) ----
        data = _prepare_batch(batch, vae_model, device, model_dtype)
        was_training = model.training
        model.train()
        with torch.no_grad():
            with _autocast_for_model(model):
                loss_dict = model(**data)

        mse_tensor = loss_dict.get("mse") if isinstance(loss_dict, dict) else None
        ce_tensor = loss_dict.get("ce") if isinstance(loss_dict, dict) else None

        mse_present = mse_tensor is not None
        if mse_present:
            mse_numel = int(mse_tensor.numel())
            mse_all_finite = bool(torch.isfinite(mse_tensor).all().item())
            mse_mean_val = float(mse_tensor.detach().float().mean().item())
        else:
            mse_numel = 0
            mse_all_finite = False
            mse_mean_val = 0.0
        ce_mean_val = float(ce_tensor.detach().float().mean().item()) if ce_tensor is not None else 0.0

        if not was_training:
            model.eval()

        mse_indexes = packed_dict.get("mse_loss_indexes")
        mse_idx_set = set(int(x) for x in mse_indexes.tolist()) if mse_indexes is not None else set()

        checks = {
            "assert_all_target_post_sigmoid_positive": bool(all(v > 0.0 for v in post_sigmoid_target)) and len(post_sigmoid_target) > 0,
            "assert_effective_mse_tokens_equals_256": len(mse_idx_set & target_positions) == 256,
            "assert_mse_numel_positive": mse_numel > 0,
            "assert_mse_all_finite": mse_all_finite,
            "assert_mse_mean_positive": mse_present and mse_mean_val > 0.0,
            "assert_no_source_index_in_mse": len(mse_idx_set & source_positions) == 0,
        }
        detail = {
            "pair_id": pair.get("pair_id"),
            "pair_seed": int(pair_seed),
            "num_source_vae_positions": len(source_positions),
            "num_target_vae_positions": len(target_positions),
            "num_mse_indexes": len(mse_idx_set),
            "effective_mse_tokens": len(mse_idx_set & target_positions),
            "source_indexes_in_mse": sorted(mse_idx_set & source_positions)[:16],
            "post_sigmoid_target_min": min(post_sigmoid_target) if post_sigmoid_target else None,
            "post_sigmoid_target_max": max(post_sigmoid_target) if post_sigmoid_target else None,
            "raw_target_timestep_min": min(v for i, v in enumerate(ts_list) if vae_idx[i] in target_positions) if target_positions else None,
            "raw_target_timestep_max": max(v for i, v in enumerate(ts_list) if vae_idx[i] in target_positions) if target_positions else None,
            "mse_numel": mse_numel,
            "mse_all_finite": mse_all_finite,
            "mse_mean": mse_mean_val,
            "ce_mean": ce_mean_val,
            "canonical_trace": trace,
            "checks": checks,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(detail, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise AssertionError(f"1-step assertions failed: {failed}. Detail written to {output_path}")
        del data, loss_dict
    finally:
        # Always restore every RNG we might have consumed, even on failure.
        _restore_rng_state(rng_snapshot)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return detail


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-pair overfit diagnostic for counterfactual CoT.")
    parser.add_argument("--manifest", type=Path, default=FIXED_GOLD40_MANIFEST)
    parser.add_argument("--pair-id", default=DEFAULT_PAIR_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llm-path", default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
    parser.add_argument("--vae-path", default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors")
    parser.add_argument("--checkpoint-path", default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors")
    parser.add_argument("--variant", default="paired_cf_mvp", choices=["paired_cf_mvp"])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, nargs="+", default=list(DEFAULT_SAVE_STEPS))
    parser.add_argument("--num-samples", type=int, default=40, help="Records loaded from manifest before filter (schema quirk).")
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
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=25)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-device-map", default="single")
    parser.add_argument("--device-map-max-memory", default="34GiB")
    parser.add_argument("--trainable-name-regex", default=DEFAULT_TRAINABLE_REGEX)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Single-pair overfit is aggressive; default lifted well above the 40-pair 1e-6.")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--param-history-every", type=int, default=25,
                        help="Cadence for storing per-tensor grad/update statistics in the report; every step still logs to stdout.")
    parser.add_argument("--diagnostics-only", action="store_true",
                        help="Only dump conditioning_prompt / sequence_plan / image_identity / loss_mask_summary and exit before any model build or training.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _pick_pair(records: Sequence[Dict[str, Any]], pair_id: str) -> Dict[str, Any]:
    for record in records:
        if record.get("pair_id") == pair_id:
            return record
    raise KeyError(f"pair_id {pair_id!r} not found in manifest (loaded {len(records)} records).")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    (args.output_dir / "visualizations").mkdir(exist_ok=True)
    (args.output_dir / "logs").mkdir(exist_ok=True)

    _set_seed(args.seed)
    device = torch.device(args.device)

    _log(
        "single_pair_overfit_start",
        manifest=str(args.manifest),
        pair_id=args.pair_id,
        output_dir=str(args.output_dir),
        steps=args.steps,
        save_steps=args.save_steps,
        lr=args.lr,
        trainable_name_regex=args.trainable_name_regex,
    )

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)

    # Manifest load: we deliberately load *all* records so we can pick the exact
    # pair; only that one record is later fed to the packer.
    records = load_jsonl(args.manifest)
    pair = _pick_pair(records, args.pair_id)

    # ---- Diagnostic dump 1: conditioning text (raw + tokenized roundtrip) ----
    tokenization_dump = _tokenize_conditioning(tokenizer, pair, args.variant)
    tokenization_dump["pair_id"] = args.pair_id
    tokenization_dump["intervention"] = pair.get("intervention")
    tokenization_dump["source_prompt"] = pair.get("source", {}).get("prompt")
    tokenization_dump["source_cot_text"] = pair.get("source", {}).get("cot_text")
    tokenization_dump["target_cot_text"] = pair.get("target", {}).get("cot_text")
    (args.output_dir / "conditioning_prompt.json").write_text(
        json.dumps(tokenization_dump, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    (args.output_dir / "conditioning_prompt.txt").write_text(
        tokenization_dump["prompt_raw"] + "\n\n===DECODED_ROUNDTRIP===\n" + tokenization_dump["prompt_decoded_roundtrip"],
        encoding="utf-8",
    )
    _log("conditioning_prompt_dumped", prompt_len_chars=len(tokenization_dump["prompt_raw"]), token_ids_len=tokenization_dump["token_ids_len"])

    # ---- Diagnostic dump 2: image file identity (SHA256 + size) ----
    image_identity = _describe_pair_files(pair)
    image_identity["pair_id"] = args.pair_id
    (args.output_dir / "image_identity.json").write_text(
        json.dumps(image_identity, ensure_ascii=False, indent=2, default=str), encoding="utf-8",
    )
    _log("image_identity_dumped", **{k: v for k, v in image_identity.items() if k != "pair_id"})

    # ---- Dataset + packer (batch of one, deterministic seed) ----
    raw_dataset = _build_raw_sample_dataset(args, tokenizer)
    packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)

    pair_seed = _stable_pair_seed(args.pair_id, args.seed)
    _log("pair_seed", pair_id=args.pair_id, pair_seed=pair_seed)

    # Build one canonical batch once so we can dump its plan / mask / paths.
    canonical_batch, canonical_trace = _pack_single_pair(packer, raw_dataset, pair, pair_seed)

    # ---- Diagnostic dump 3: sequence plan ----
    raw_sample_probe, side_probe, mode_probe = raw_dataset._paired_sample(pair)
    sequence_plan = _sequence_plan_summary(raw_sample_probe)
    (args.output_dir / "sequence_plan.json").write_text(
        json.dumps({"pair_id": args.pair_id, "side": side_probe, "mode": mode_probe, "sequence_plan": sequence_plan}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _log("sequence_plan_dumped", side=side_probe, mode=mode_probe, plan=sequence_plan)

    # ---- Diagnostic dump 4: loss mask summary (from the packed tensor batch) ----
    packed_data_dict = canonical_batch.to_dict()
    loss_mask_report = _loss_mask_summary(packed_data_dict)
    (args.output_dir / "loss_mask_summary.json").write_text(
        json.dumps({"pair_id": args.pair_id, **loss_mask_report, "canonical_trace": canonical_trace}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _log("loss_mask_dumped", **loss_mask_report)

    if args.diagnostics_only:
        _log(
            "single_pair_overfit_diagnostics_only_done",
            output_dir=str(args.output_dir),
            files=[
                str(args.output_dir / "conditioning_prompt.txt"),
                str(args.output_dir / "conditioning_prompt.json"),
                str(args.output_dir / "sequence_plan.json"),
                str(args.output_dir / "image_identity.json"),
                str(args.output_dir / "loss_mask_summary.json"),
            ],
        )
        return

    # ---- Build model + optimizer ----
    model, vae_model, checkpoint_load_report = _build_model_for_gold40(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
    )
    trainable_report = _set_trainable_parameters(model, args.trainable_name_regex)
    model.train()

    trainable_params: List[torch.nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
    trainable_names: List[str] = [n for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    _log(
        "trainable_report",
        trainable_tensors=trainable_report["trainable_tensors"],
        trainable_parameters=trainable_report["trainable_parameters"],
        matched_names=trainable_names,
    )

    # ---- Pre-training 1-step assertions (no optimizer.step, RNG-preserving) ----
    assertion_detail = _run_one_step_assertions(
        args, model, vae_model, packer, raw_dataset, pair, pair_seed,
        args.output_dir / "pretrain_1step_assertions.json",
    )
    _log(
        "pretrain_1step_assertions_passed",
        **{
            k: assertion_detail[k]
            for k in (
                "num_source_vae_positions",
                "num_target_vae_positions",
                "effective_mse_tokens",
                "post_sigmoid_target_min",
                "post_sigmoid_target_max",
                "mse_numel",
                "mse_all_finite",
                "mse_mean",
            )
        },
    )

    # ---- Save step-0 reference images (source + golden target) ----
    ref_dir = args.output_dir / "visualizations" / "step_ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    src_ref = _save_reference_image(pair, "source", ref_dir / "source.png", args.visual_image_size)
    tgt_ref = _save_reference_image(pair, "target", ref_dir / "golden_target.png", args.visual_image_size)
    _log("reference_images_saved", source=src_ref, golden_target=tgt_ref)

    # ---- Training loop ----
    loss_history: List[Dict[str, Any]] = []
    grad_history: List[Dict[str, Any]] = []
    update_history: List[Dict[str, Any]] = []
    saved_checkpoints: List[Dict[str, Any]] = []
    generated_images: List[Dict[str, Any]] = []

    save_step_set = set(int(s) for s in args.save_steps)

    def _generate_and_save(step: int) -> Dict[str, Any]:
        """Save trainable checkpoint and render one inference PNG for this step.

        All rendered images share a single ``args.generation_seed`` regardless
        of ``step`` so that step_XXX images are visually comparable and only
        differ because of weight changes.
        """
        ckpt_path = args.output_dir / "checkpoints" / f"step_{step:03d}_trainable.pt"
        ckpt_str = _save_trainable_checkpoint(
            model, optimizer, scheduler, ckpt_path, step, args, trainable_report, loss_history,
        )
        ckpt_sha = _sha256_file(Path(ckpt_str))
        ckpt_bytes = Path(ckpt_str).stat().st_size
        # switch to eval for inference to avoid dropout etc.
        was_training = model.training
        model.eval()
        vae_model.eval()
        try:
            inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, args)
            out_path = args.output_dir / "visualizations" / f"step_{step:03d}" / "model_output.png"
            # Fixed eval seed - never a function of step.
            _generate_one(inferencer, pair, out_path, args, seed=args.generation_seed)
        finally:
            if was_training:
                model.train()
        record = {
            "step": step,
            "global_step": step,
            "checkpoint": ckpt_str,
            "checkpoint_sha256": ckpt_sha,
            "checkpoint_bytes": ckpt_bytes,
            "image": str(out_path),
            "eval_seed": int(args.generation_seed),
            "generation_parameters": _generation_parameters(args),
        }
        saved_checkpoints.append(record)
        generated_images.append(record)
        _log("checkpoint_and_image_saved", **{k: v for k, v in record.items() if k != "generation_parameters"})
        return record

    model_dtype = next(model.parameters()).dtype

    # step-0 snapshot before any optimizer.step
    if 0 in save_step_set:
        _generate_and_save(0)

    for step in range(1, args.steps + 1):
        # Re-pack every step; the packer is deterministic under our seed, but we
        # re-seed here so each step picks *its own* timestep/noise, matching a
        # normal training loop.
        step_seed = pair_seed + step
        batch, trace = _pack_single_pair(packer, raw_dataset, pair, step_seed)
        data = _prepare_batch(batch, vae_model, device, model_dtype)

        # deterministic forward-time noise
        torch.manual_seed(step_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(step_seed)

        optimizer.zero_grad(set_to_none=True)
        loss, scalars = _forward_loss(model, data)

        # Take fp32 param snapshot before the update so we can compute
        # per-tensor delta norms.
        before_snapshot = _snapshot_trainable_fp32(model) if (step in save_step_set or step % max(1, args.param_history_every) == 0) else None

        loss.backward()

        grad_report = _grad_norms(model)
        grad_l2_total = math.sqrt(sum(v["grad_l2"] ** 2 for v in grad_report.values() if v.get("grad_present")))
        grad_max_over_tensors = max((v["grad_max_abs"] for v in grad_report.values() if v.get("grad_present")), default=0.0)

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm) if args.max_grad_norm > 0 else None
        optimizer.step()

        update_report: Dict[str, Dict[str, float]] = {}
        if before_snapshot is not None:
            update_report = _param_update_norms(before_snapshot, model)
            del before_snapshot

        row = {
            "step": step,
            "mse": float(scalars["mse"]),
            "ce": float(scalars.get("ce", 0.0)),
            "total": float(scalars["total"]),
            "grad_l2_total": grad_l2_total,
            "grad_max_over_tensors": grad_max_over_tensors,
            "grad_norm_after_clip": float(grad_norm) if grad_norm is not None else None,
            "step_seed": step_seed,
            "timesteps_hash": trace.get("packed_timesteps_hash"),
        }
        loss_history.append(row)
        if step % max(1, args.log_every) == 0:
            _log("train_step", **row)

        # Store per-tensor stats only at save/param-history steps to keep the
        # report a manageable size.
        if step in save_step_set or step % max(1, args.param_history_every) == 0:
            grad_history.append({"step": step, "per_tensor": grad_report})
            update_history.append({"step": step, "per_tensor": update_report})

        del data, loss
        if step in save_step_set:
            _generate_and_save(step)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Final report ----
    final_report = {
        "status": "passed",
        "pair_id": args.pair_id,
        "steps": args.steps,
        "save_steps": args.save_steps,
        "manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "conditioning_prompt": tokenization_dump,
        "image_identity": image_identity,
        "sequence_plan": sequence_plan,
        "loss_mask_summary": loss_mask_report,
        "canonical_batch_trace": canonical_trace,
        "trainable_report": {
            **trainable_report,
            "trainable_parameter_names": trainable_names,
        },
        "checkpoint_load_report": checkpoint_load_report,
        "loss_history": loss_history,
        "loss_history_summary": {
            "step_1_mse": loss_history[0]["mse"] if loss_history else None,
            "step_last_mse": loss_history[-1]["mse"] if loss_history else None,
            "min_mse": min((r["mse"] for r in loss_history), default=None),
            "max_mse": max((r["mse"] for r in loss_history), default=None),
        },
        "grad_history": grad_history,
        "update_history": update_history,
        "saved_checkpoints": saved_checkpoints,
        "generated_images": generated_images,
        "generation_parameters": _generation_parameters(args),
        "args": _serializable_args(args),
    }
    _write_json(args.output_dir / "single_pair_overfit_report.json", final_report)
    _log(
        "single_pair_overfit_finished",
        report=str(args.output_dir / "single_pair_overfit_report.json"),
        step_1_mse=final_report["loss_history_summary"]["step_1_mse"],
        step_last_mse=final_report["loss_history_summary"]["step_last_mse"],
        min_mse=final_report["loss_history_summary"]["min_mse"],
    )


if __name__ == "__main__":
    main()
