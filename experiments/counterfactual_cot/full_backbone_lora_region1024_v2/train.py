#!/usr/bin/env python3
"""Round-2 single-pair overfit: full-backbone LoRA, CoT-core and 1024 region loss."""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import json
import os
import random
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from data.dataset_base import DataConfig, PackedDataset  # noqa: E402
from data.transforms import ImageTransform  # noqa: E402
from experiments.counterfactual_cot.data.prompt_profiles import render_prompt_profile  # noqa: E402
from experiments.counterfactual_cot.data.schema import load_jsonl  # noqa: E402
from experiments.counterfactual_cot.datasets.bagel_manifest_dataset import CounterfactualCoTDataset  # noqa: E402
from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.region_objective import (  # noqa: E402
    EXPECTED_TARGET_TOKENS,
    build_region_objective,
    region_forward_loss,
)
from experiments.counterfactual_cot.training.fixed_deterministic_eval import (  # noqa: E402
    _pack_single_pair,
    _stable_pair_seed,
)
from experiments.counterfactual_cot.training.gold40_overfit import (  # noqa: E402
    DEFAULT_TRAINABLE_REGEX,
    _autocast_for_model,
    _build_model_for_gold40,
    _load_tokenizer,
    _prepare_batch,
    _random_state_payload,
    _restore_random_state,
    _serializable_args,
    _tensor_tree_to_cpu,
    _transform_args,
)
from experiments.counterfactual_cot.training.lora_2step_smoke import (  # noqa: E402
    TARGET_MODULE_SUFFIXES,
    _iter_target_linear_names,
    inject_lora,
    set_trainable,
)
from modeling.bagel.qwen2_navit import Qwen2MoTDecoderLayer  # noqa: E402

EXPERIMENT_NAME = "cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_overfit_v2"
PROMPT_PROFILE = "cot_core_v1"
CHECKPOINT_FORMAT = "full_backbone_lora_region1024_cotcore_resume_v2"
PAIR_ID = (
    "cf_source_living_room_play_absolute_position_change_"
    "v4_absolute_region_replacement_v4_0000"
)
MANIFEST = BAGEL_ROOT / (
    "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/"
    "manifests/intervention_pairs_single_ball_c1.jsonl"
)
EXPECTED_SOURCE = BAGEL_ROOT / (
    "experiments/counterfactual_cot/gold40_final/source/"
    "cf_source_living_room_play_absolute_position_change_"
    "v4_absolute_region_replacement_v4_0000__source.png"
)
EXPECTED_TARGET = BAGEL_ROOT / (
    "experiments/counterfactual_cot/gold40_final/target/"
    "cf_source_living_room_play_absolute_position_change_"
    "v4_absolute_region_replacement_v4_0000__target.png"
)
EXPECTED_MASK = BAGEL_ROOT / (
    "experiments/counterfactual_cot/gold40_final/masks/"
    "cf_source_living_room_play_absolute_position_change_"
    "v4_absolute_region_replacement_v4_0000.png"
)
EXPECTED_SOURCE_SHA256 = "71d756241d84721b7885c68bd4da650bf4873277d48be363fb6990d74ace1539"
EXPECTED_TARGET_SHA256 = "8500c4401b5dd17ca7aa06d78bf9f3e1dbab3ceb67904c0a2903e034477f9dd1"
EXPECTED_INTERVENTION = {
    "entity_id": "ball_1",
    "field": "location",
    "old_value": "left_region",
    "new_value": "right_region",
    "operation": "replace",
}
DEFAULT_MODEL = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_OUTPUT = BAGEL_ROOT / "experiments/counterfactual_cot/outputs" / EXPERIMENT_NAME
DEFAULT_MILESTONES = (0, 25, 50, 100, 200, 300, 400, 500)
_STOP_REQUESTED = False
_LOG_PATH: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable_args(args: argparse.Namespace) -> Dict[str, Any]:
    return _serializable_args(args)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _log(event: str, **fields: Any) -> None:
    row = {"time": _utc_now(), "event": event, **fields}
    line = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    print(line, flush=True)
    if _LOG_PATH is not None:
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(json.dumps({"event": "stop_requested", "signal": signum}), flush=True)


def _resolve_pair_image(pair: Dict[str, Any], side: str) -> Path:
    image = Path(pair[side]["image"])
    if image.is_absolute():
        return image.resolve()
    root = Path(pair.get("final_image_root") or pair.get("image_root") or MANIFEST.parent)
    return (root / image).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pick_and_validate_pair(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    matches = [row for row in records if row.get("pair_id") == PAIR_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one locked pair {PAIR_ID!r}, found {len(matches)}")
    pair = matches[0]
    source = _resolve_pair_image(pair, "source")
    target = _resolve_pair_image(pair, "target")
    if source != EXPECTED_SOURCE.resolve():
        raise RuntimeError(f"Locked source mismatch: expected={EXPECTED_SOURCE.resolve()}, actual={source}")
    if target != EXPECTED_TARGET.resolve():
        raise RuntimeError(f"Locked target mismatch: expected={EXPECTED_TARGET.resolve()}, actual={target}")
    for label, path in (("source", source), ("target", target)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} image: {path}")
    actual_hashes = {"source": _sha256(source), "target": _sha256(target)}
    expected_hashes = {"source": EXPECTED_SOURCE_SHA256, "target": EXPECTED_TARGET_SHA256}
    if actual_hashes != expected_hashes:
        raise RuntimeError(f"Locked image hash mismatch: expected={expected_hashes}, actual={actual_hashes}")
    intervention = pair.get("intervention", {})
    mismatches = {
        key: {"expected": value, "actual": intervention.get(key)}
        for key, value in EXPECTED_INTERVENTION.items()
        if intervention.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Locked intervention mismatch: {json.dumps(mismatches, ensure_ascii=False)}")
    return pair


class CotCoreDataset(CounterfactualCoTDataset):
    """Paired image-edit sample whose only text is Source Prompt + Target CoT."""

    def _paired_sample(self, pair: Dict[str, Any]):
        data = self._init_sample()
        image_root = pair.get("final_image_root")
        source_image = self._load_image(pair["source"]["image"], image_root=image_root)
        data = self._add_vae_image(data, source_image, need_loss=False, enable_cfg=True)
        data = self._add_vit_image(data, source_image, enable_cfg=True)
        prompt = render_prompt_profile(pair, PROMPT_PROFILE)
        required = ("Source Prompt:", "Target CoT:")
        forbidden = ("Source CoT:", "Target Prompt:", "Intervention:", "Preserve:")
        if not all(field in prompt for field in required) or any(field in prompt for field in forbidden):
            raise RuntimeError(f"Invalid {PROMPT_PROFILE} prompt structure: {prompt}")
        data = self._add_text(data, prompt, need_loss=False, enable_cfg=True)
        target_image = self._load_image(pair["target"]["image"], image_root=image_root)
        data = self._add_vae_image(data, target_image, need_loss=True, enable_cfg=False)
        return data, "target", PROMPT_PROFILE


def _build_raw_sample_dataset(args: argparse.Namespace, tokenizer: Any) -> CotCoreDataset:
    return CotCoreDataset(
        dataset_name="counterfactual_cot_region1024_cotcore_raw",
        transform=ImageTransform(**_transform_args(args.image_size, args.image_size, 16)),
        tokenizer=tokenizer,
        manifest_path=str(args.manifest),
        variant=args.variant,
        include_source_image=True,
        training_seed=args.seed,
        local_rank=0,
        world_size=1,
        num_workers=0,
        data_status=None,
        vit_transform=ImageTransform(**_transform_args(args.vit_image_size, args.vit_min_image_size, 14)),
        max_records=1,
    )


def _build_singleton_packed_dataset(
    args: argparse.Namespace,
    tokenizer: Any,
    special_tokens: Dict[str, int],
) -> PackedDataset:
    from data.dataset_info import DATASET_INFO, DATASET_REGISTRY

    registry_name = "counterfactual_cot_region1024_cotcore"
    DATASET_REGISTRY[registry_name] = CotCoreDataset
    DATASET_INFO[registry_name] = {
        "region1024_cotcore": {
            "manifest_path": str(args.manifest),
            "variant": args.variant,
            "training_seed": args.seed,
            "num_total_samples": 1,
            "max_records": 1,
        }
    }
    dataset_meta = {
        registry_name: {
            "dataset_names": ["region1024_cotcore"],
            "weight": 1.0,
            "is_mandatory": True,
            "image_transform_args": _transform_args(args.image_size, args.image_size, 16),
            "vit_image_transform_args": _transform_args(args.vit_image_size, args.vit_min_image_size, 14),
            "num_used_data": [1],
            "max_records": 1,
        }
    }
    data_config = DataConfig(
        grouped_datasets=dataset_meta,
        text_cond_dropout_prob=0.0,
        vit_cond_dropout_prob=0.0,
        vae_cond_dropout_prob=0.0,
        vae_image_downsample=16,
        max_latent_size=64,
        vit_patch_size=14,
        max_num_patch_per_side=70,
    )
    return PackedDataset(
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


def _validate_packed_trace(trace: Dict[str, Any]) -> None:
    if int(trace.get("num_vae_tokens", -1)) != EXPECTED_TARGET_TOKENS * 2:
        raise RuntimeError(f"Expected 8192 source+target VAE tokens, got trace={trace}")
    if int(trace.get("num_vit_tokens", -1)) != 4900:
        raise RuntimeError(f"Expected 4900 source ViT tokens, got trace={trace}")
    sequence_length = int(trace.get("sequence_length", -1))
    if not 13000 <= sequence_length <= 16384:
        raise RuntimeError(f"Unexpected 1024 cot-core sequence length: {sequence_length}")


def _read_num_layers(llm_path: Path) -> int:
    config_path = llm_path / "llm_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    num_layers = int(config["num_hidden_layers"])
    if num_layers <= 0:
        raise RuntimeError(f"Invalid num_hidden_layers={num_layers} in {config_path}")
    return num_layers


def _validate_model_files(args: argparse.Namespace) -> None:
    required = [
        Path(args.llm_path) / "config.json",
        Path(args.llm_path) / "llm_config.json",
        Path(args.llm_path) / "tokenizer.json",
        Path(args.vae_path),
        Path(args.checkpoint_path),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Model migration is incomplete; missing: " + ", ".join(missing))
    if Path(args.checkpoint_path).stat().st_size < 20 * 1024**3:
        raise RuntimeError(f"Base checkpoint looks incomplete: {args.checkpoint_path}")


def _validate_base_load_report(report: Dict[str, Any]) -> None:
    expected_missing = {"latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"}
    actual_missing = set(report.get("missing_keys_sample", []))
    if report.get("num_missing_keys") != len(expected_missing) or actual_missing != expected_missing:
        raise RuntimeError(f"Unexpected missing base-checkpoint keys: {report}")
    if report.get("num_unexpected_keys") != 0:
        raise RuntimeError(f"Unexpected base-checkpoint tensors: {report}")
    if int(report.get("num_loaded_tensors", 0)) < 1200:
        raise RuntimeError(f"Too few base-checkpoint tensors loaded: {report}")


def _fixed_region_eval(
    model: nn.Module,
    vae_model: nn.Module,
    packer: Any,
    raw_dataset: Any,
    pair: Dict[str, Any],
    seed: int,
    device: torch.device,
    region_weights: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    batch, trace = _pack_single_pair(packer, raw_dataset, pair, seed)
    _validate_packed_trace(trace)
    data = _prepare_batch(batch, vae_model, device, next(model.parameters()).dtype)
    was_training = model.training
    model.train()
    with torch.no_grad():
        loss, scalars = region_forward_loss(model, data, region_weights, _autocast_for_model)
    if not was_training:
        model.eval()
    del data, loss
    return scalars, trace


def _canonical_parameter_name(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


def _named_parameters_canonical(model: nn.Module):
    seen = set()
    for name, parameter in model.named_parameters():
        canonical_name = _canonical_parameter_name(name)
        if canonical_name in seen:
            raise RuntimeError(f"Duplicate canonical parameter name: {canonical_name}")
        seen.add(canonical_name)
        yield canonical_name, parameter


def _apply_decoder_activation_checkpointing(model: nn.Module, expected_layers: int) -> int:
    count = sum(isinstance(module, Qwen2MoTDecoderLayer) for module in model.modules())
    if count != expected_layers:
        raise RuntimeError(f"Expected {expected_layers} decoder layers for activation checkpointing, found {count}")
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        ),
        check_fn=lambda module: isinstance(module, Qwen2MoTDecoderLayer),
    )
    wrapped = sum(isinstance(module, CheckpointWrapper) for module in model.modules())
    if wrapped != expected_layers:
        raise RuntimeError(f"Expected {expected_layers} activation-checkpoint wrappers, found {wrapped}")
    remaining = sum(isinstance(module, Qwen2MoTDecoderLayer) for module in model.modules())
    if remaining != expected_layers:
        raise RuntimeError(f"Activation wrappers lost decoder layers: expected={expected_layers}, actual={remaining}")
    return wrapped


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    trainable_report: Dict[str, Any],
    loss_history: Sequence[Dict[str, Any]],
    target_names: Sequence[str],
) -> Dict[str, Any]:
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in _named_parameters_canonical(model)
        if parameter.requires_grad
    }
    random_state = _random_state_payload()
    random_state["numpy_random_state"] = np.random.get_state()
    checkpoint_config = _jsonable_args(args)
    checkpoint_config.update(
        prompt_profile=PROMPT_PROFILE,
        region_objective="0.25_old_0.25_new_0.50_preserve",
        max_latent_size=64,
    )
    return {
        "format": CHECKPOINT_FORMAT,
        "experiment_name": EXPERIMENT_NAME,
        "created_at": _utc_now(),
        "step": int(step),
        "global_step": int(step),
        "trainable_state_dict": trainable_state,
        "optimizer_state_dict": _tensor_tree_to_cpu(optimizer.state_dict()),
        "random_state": random_state,
        "config": checkpoint_config,
        "trainable_report": trainable_report,
        "loss_history": list(loss_history),
        "lora_target_names": list(target_names),
    }


def _atomic_save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    trainable_report: Dict[str, Any],
    loss_history: Sequence[Dict[str, Any]],
    target_names: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = _checkpoint_payload(model, optimizer, step, args, trainable_report, loss_history, target_names)
    torch.save(payload, tmp)
    os.replace(tmp, path)
    del payload


def _replace_latest_with_hardlink(checkpoint: Path, latest: Path) -> None:
    tmp = latest.with_name(latest.name + ".tmp-link")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.link(checkpoint, tmp)
    os.replace(tmp, latest)


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    trainable_report: Dict[str, Any],
    loss_history: Sequence[Dict[str, Any]],
    target_names: Sequence[str],
    milestone: bool,
) -> Path:
    checkpoint_dir = args.output_dir / "checkpoints"
    latest = checkpoint_dir / "latest.pt"
    if milestone:
        path = checkpoint_dir / f"step_{step:06d}.pt"
        _atomic_save_checkpoint(path, model, optimizer, step, args, trainable_report, loss_history, target_names)
        _replace_latest_with_hardlink(path, latest)
    else:
        path = latest
        _atomic_save_checkpoint(path, model, optimizer, step, args, trainable_report, loss_history, target_names)
    _log("checkpoint_saved", step=step, path=str(path), latest=str(latest), bytes=path.stat().st_size)
    gc.collect()
    return path


def _critical_resume_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "pair_id": PAIR_ID,
        "manifest": str(Path(args.manifest)),
        "llm_path": str(Path(args.llm_path)),
        "vae_path": str(Path(args.vae_path)),
        "checkpoint_path": str(Path(args.checkpoint_path)),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "trainable_name_regex": args.trainable_name_regex,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "max_grad_norm": float(args.max_grad_norm),
        "seed": int(args.seed),
        "variant": args.variant,
        "precision": args.precision,
        "model_device_map": args.model_device_map,
        "image_size": int(args.image_size),
        "vit_image_size": int(args.vit_image_size),
        "vit_min_image_size": int(args.vit_min_image_size),
        "expected_num_tokens": int(args.expected_num_tokens),
        "max_num_tokens_per_sample": int(args.max_num_tokens_per_sample),
        "max_num_tokens": int(args.max_num_tokens),
        "max_peak_reserved_gib": float(args.max_peak_reserved_gib) if args.max_peak_reserved_gib is not None else None,
        "activation_checkpointing": bool(args.activation_checkpointing),
        "prompt_profile": PROMPT_PROFILE,
        "region_objective": "0.25_old_0.25_new_0.50_preserve",
        "max_latent_size": 64,
    }


def _load_resume(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    target_names: Sequence[str],
) -> Tuple[int, List[Dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Unsupported resume checkpoint format in {path}: {payload.get('format')}")
    saved_config = payload.get("config", {})
    expected = _critical_resume_config(args)
    mismatches = {
        key: {"checkpoint": saved_config.get(key), "current": value}
        for key, value in expected.items()
        if saved_config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Resume configuration mismatch: {json.dumps(mismatches, default=str)}")
    if list(payload.get("lora_target_names", [])) != list(target_names):
        raise RuntimeError("Resume LoRA target list does not match the current full-backbone target list")
    saved_state = payload.get("trainable_state_dict", {})
    named = dict(_named_parameters_canonical(model))
    expected_keys = {name for name, parameter in named.items() if parameter.requires_grad}
    if set(saved_state) != expected_keys:
        missing = sorted(expected_keys - set(saved_state))
        unexpected = sorted(set(saved_state) - expected_keys)
        raise RuntimeError(f"Resume trainable-key mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    for name, tensor in saved_state.items():
        parameter = named[name]
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise RuntimeError(f"Resume shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(parameter.shape)}")
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    parameter_devices = {
        parameter.device
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    state_devices = {
        value.device
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value) and value.numel() > 1
    }
    if any(device not in parameter_devices for device in state_devices):
        raise RuntimeError(f"Optimizer state device mismatch: params={parameter_devices}, states={state_devices}")
    _restore_random_state(payload.get("random_state"))
    numpy_state = payload.get("random_state", {}).get("numpy_random_state")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    step = int(payload.get("global_step", payload.get("step", 0)))
    history = list(payload.get("loss_history", []))
    del payload, saved_state
    gc.collect()
    _log("resume_loaded", path=str(path), step=step, history_rows=len(history))
    return step, history


def _checkpoint_is_readable(path: Path) -> bool:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        valid = (
            payload.get("format") == CHECKPOINT_FORMAT
            and isinstance(payload.get("trainable_state_dict"), dict)
            and isinstance(payload.get("optimizer_state_dict"), dict)
            and payload.get("global_step", payload.get("step")) is not None
        )
        del payload
        return bool(valid)
    except Exception as exc:  # noqa: BLE001
        _log("checkpoint_rejected", path=str(path), error=repr(exc))
        return False


def _resolve_resume(args: argparse.Namespace) -> Path | None:
    if args.resume == "none":
        return None
    if args.resume == "auto":
        checkpoint_dir = args.output_dir / "checkpoints"
        candidates = [checkpoint_dir / "latest.pt"]
        candidates.extend(sorted(checkpoint_dir.glob("step_*.pt"), reverse=True))
        seen = set()
        for candidate in candidates:
            resolved = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if resolved in seen or not candidate.is_file():
                continue
            seen.add(resolved)
            if _checkpoint_is_readable(candidate):
                return candidate
        return None
    candidate = Path(args.resume)
    if not candidate.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {candidate}")
    if not _checkpoint_is_readable(candidate):
        raise RuntimeError(f"Resume checkpoint is unreadable or incomplete: {candidate}")
    return candidate


def _validate_gpu_capacity(minimum_gib: float) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires one CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    total_gib = properties.total_memory / 1024**3
    if total_gib < minimum_gib:
        raise RuntimeError(
            f"Refusing to run on {properties.name} with {total_gib:.1f} GiB. "
            f"This single-GPU recipe requires at least {minimum_gib:.1f} GiB of physical GPU memory."
        )
    return {"name": properties.name, "total_memory_gib": round(total_gib, 2)}


def _memory_stats() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    gib = 1024**3
    return {
        "allocated_gib": round(torch.cuda.memory_allocated() / gib, 3),
        "reserved_gib": round(torch.cuda.memory_reserved() / gib, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 3),
    }


def _enforce_memory_gate(args: argparse.Namespace, stage: str) -> Dict[str, float]:
    stats = _memory_stats()
    limit = args.max_peak_reserved_gib
    if limit is not None and stats.get("peak_reserved_gib", 0.0) >= limit:
        raise RuntimeError(
            f"Memory gate failed at {stage}: peak_reserved_gib={stats['peak_reserved_gib']} "
            f">= limit={limit}. Do not proceed to formal training on this configuration."
        )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vae-path", type=Path, default=DEFAULT_MODEL / "ae.safetensors")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_MODEL / "ema.safetensors")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--trainable-name-regex", default=DEFAULT_TRAINABLE_REGEX)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-peak-reserved-gib", type=float, default=38.5)
    parser.add_argument("--milestone-steps", type=int, nargs="+", default=list(DEFAULT_MILESTONES))
    parser.add_argument("--eval-steps", type=int, nargs="+", default=list(DEFAULT_MILESTONES))
    parser.add_argument("--resume", default="auto", help="auto, none, or an explicit checkpoint path")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=("bf16",))
    parser.add_argument("--model-device-map", default="single", choices=("single",))
    parser.add_argument("--device-map-max-memory", default="38GiB")
    parser.add_argument("--min-gpu-memory-gib", type=float, default=38.0)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--variant", default="paired_cf_mvp", choices=("paired_cf_mvp",))
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--vit-image-size", type=int, default=980)
    parser.add_argument("--vit-min-image-size", type=int, default=980)
    parser.add_argument("--expected-num-tokens", type=int, default=16384)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=16384)
    parser.add_argument("--max-num-tokens", type=int, default=16384)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.pair_id = PAIR_ID
    if args.steps < 1 or args.checkpoint_every < 1:
        parser.error("--steps and --checkpoint-every must be positive")
    locked = {
        "image_size": (args.image_size, 1024),
        "vit_image_size": (args.vit_image_size, 980),
        "vit_min_image_size": (args.vit_min_image_size, 980),
        "expected_num_tokens": (args.expected_num_tokens, 16384),
        "max_num_tokens_per_sample": (args.max_num_tokens_per_sample, 16384),
        "max_num_tokens": (args.max_num_tokens, 16384),
        "num_samples": (args.num_samples, 1),
    }
    mismatches = {name: {"actual": actual, "required": required} for name, (actual, required) in locked.items() if actual != required}
    if mismatches:
        parser.error(f"Round-2 locked configuration mismatch: {json.dumps(mismatches)}")
    if not args.activation_checkpointing and not args.preflight_only:
        parser.error("Round-2 1024 training requires --activation-checkpointing")
    return args


def main() -> None:
    global _LOG_PATH
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    _LOG_PATH = args.output_dir / "train_log.jsonl"
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    _validate_model_files(args)
    records = load_jsonl(args.manifest)
    pair = _pick_and_validate_pair(records)
    source_path = _resolve_pair_image(pair, "source")
    target_path = _resolve_pair_image(pair, "target")
    region_spec, region_weights = build_region_objective(source_path, target_path, EXPECTED_MASK)
    prompt = render_prompt_profile(pair, PROMPT_PROFILE)
    if "Target Prompt:" in prompt or "Source CoT:" in prompt:
        raise RuntimeError("CoT-core prompt leaked Target Prompt or Source CoT")
    num_layers = _read_num_layers(args.llm_path)
    plan = {
        "experiment_name": EXPERIMENT_NAME,
        "pair_id": PAIR_ID,
        "source": str(source_path),
        "target": str(target_path),
        "mask": str(EXPECTED_MASK),
        "prompt_profile": PROMPT_PROFILE,
        "prompt": prompt,
        "input_structure": ["source_vae_1024", "source_vit_980", "source_prompt_plus_target_cot", "target_vae_1024_loss"],
        "region_objective": region_spec,
        "intervention": pair.get("intervention"),
        "num_backbone_layers": num_layers,
        "lora_layers": list(range(num_layers)),
        "target_module_suffixes": list(TARGET_MODULE_SUFFIXES),
        "expected_lora_wrappers": num_layers * len(TARGET_MODULE_SUFFIXES),
        "activation_checkpointing": bool(args.activation_checkpointing),
        "single_gpu": True,
        "args": _jsonable_args(args),
    }
    _atomic_json(args.output_dir / "experiment_plan.json", plan)
    _atomic_json(args.output_dir / "region_spec.json", region_spec)
    _log("preflight_passed", **{key: plan[key] for key in ("experiment_name", "pair_id", "source", "target", "prompt_profile", "num_backbone_layers", "expected_lora_wrappers")}, region_token_masses=region_spec["token_masses"])
    if args.preflight_only:
        return

    if args.device != "cuda":
        raise RuntimeError("This experiment requires --device cuda")
    gpu_report = _validate_gpu_capacity(args.min_gpu_memory_gib)
    _log("gpu_capacity_passed", **gpu_report)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(str(args.llm_path))
    raw_dataset = _build_raw_sample_dataset(args, tokenizer)
    packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)
    pair_seed = _stable_pair_seed(PAIR_ID, args.seed)
    probe_batch, probe_trace = _pack_single_pair(packer, raw_dataset, pair, pair_seed)
    _validate_packed_trace(probe_trace)
    _log("packed_1024_validated", **probe_trace)
    del probe_batch

    model, vae_model, base_load_report = _build_model_for_gold40(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens
    )
    _validate_base_load_report(base_load_report)
    layer_indexes = list(range(num_layers))
    target_names = _iter_target_linear_names(model, layer_indexes)
    expected_wrappers = num_layers * len(TARGET_MODULE_SUFFIXES)
    if len(target_names) != expected_wrappers:
        raise RuntimeError(f"Expected {expected_wrappers} LoRA targets, found {len(target_names)}")
    wrappers = inject_lora(model, target_names, args.lora_r, args.lora_alpha, args.lora_dropout)
    trainable_groups = set_trainable(model, args.trainable_name_regex, wrappers)
    activation_checkpointed_layers = 0
    if args.activation_checkpointing:
        activation_checkpointed_layers = _apply_decoder_activation_checkpointing(model, num_layers)
    trainable_named = [
        (name, parameter)
        for name, parameter in _named_parameters_canonical(model)
        if parameter.requires_grad
    ]
    trainable_params = [parameter for _, parameter in trainable_named]
    lora_names = set(trainable_groups["lora_trainable_names"])
    lora_numel = sum(parameter.numel() for name, parameter in trainable_named if name in lora_names)
    existing_numel = sum(parameter.numel() for name, parameter in trainable_named if name not in lora_names)
    trainable_report = {
        "num_backbone_layers": num_layers,
        "activation_checkpointing": bool(args.activation_checkpointing),
        "activation_checkpointed_layers": activation_checkpointed_layers,
        "lora_wrappers": len(wrappers),
        "lora_tensors": len(lora_names),
        "lora_parameters": lora_numel,
        "existing_trainable_tensors": len(trainable_groups["existing_trainable_names"]),
        "existing_trainable_parameters": existing_numel,
        "total_trainable_tensors": len(trainable_named),
        "total_trainable_parameters": lora_numel + existing_numel,
        "lora_target_names": target_names,
        **trainable_groups,
    }
    _atomic_json(args.output_dir / "trainable_audit.json", trainable_report)
    _log("model_ready", **{key: trainable_report[key] for key in ("lora_wrappers", "lora_tensors", "lora_parameters", "existing_trainable_parameters", "total_trainable_parameters")}, memory=_memory_stats())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    loss_history: List[Dict[str, Any]] = []
    resume_path = _resolve_resume(args)
    if resume_path is not None:
        start_step, loss_history = _load_resume(resume_path, model, optimizer, args, target_names)
        if start_step >= args.steps:
            _log("already_complete", checkpoint_step=start_step, requested_steps=args.steps)
            return

    eval_steps = {step for step in args.eval_steps if 0 <= step <= args.steps}
    milestones = {step for step in args.milestone_steps if 0 <= step <= args.steps}
    milestones.add(args.steps)
    fixed_eval_seed = pair_seed + 1_000_000

    if start_step == 0 and resume_path is None:
        initial_metrics, initial_trace = _fixed_region_eval(
            model, vae_model, packer, raw_dataset, pair, fixed_eval_seed, device, region_weights
        )
        initial_memory = _enforce_memory_gate(args, "fixed_region_eval_step0")
        _log(
            "fixed_region_eval",
            step=0,
            **initial_metrics,
            timesteps_hash=initial_trace.get("packed_timesteps_hash"),
            memory=initial_memory,
        )
        _save_checkpoint(model, optimizer, 0, args, trainable_report, loss_history, target_names, milestone=0 in milestones)

    model.train()
    for step in range(start_step + 1, args.steps + 1):
        step_seed = pair_seed + step
        batch, trace = _pack_single_pair(packer, raw_dataset, pair, step_seed)
        _validate_packed_trace(trace)
        data = _prepare_batch(batch, vae_model, device, next(model.parameters()).dtype)
        torch.manual_seed(step_seed)
        torch.cuda.manual_seed_all(step_seed)
        optimizer.zero_grad(set_to_none=True)
        loss, loss_metrics = region_forward_loss(model, data, region_weights, _autocast_for_model)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")
        loss.backward()
        if step <= 2:
            missing_grads = [name for name, parameter in trainable_named if parameter.grad is None]
            nonfinite_grads = [name for name, parameter in trainable_named if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            if missing_grads or nonfinite_grads:
                raise RuntimeError(f"Gradient audit failed: missing={missing_grads[:10]}, nonfinite={nonfinite_grads[:10]}")
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}")
        optimizer.step()
        step_memory = _enforce_memory_gate(args, f"train_step_{step}")
        row = {
            "step": step,
            **loss_metrics,
            "grad_norm_pre_clip": float(grad_norm.detach().float().cpu()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "step_seed": step_seed,
            "timesteps_hash": trace.get("packed_timesteps_hash"),
        }
        loss_history.append(row)
        _log("train_step", **row, memory=step_memory)
        del data, loss

        is_milestone = step in milestones
        should_checkpoint = is_milestone or step % args.checkpoint_every == 0 or _STOP_REQUESTED
        if should_checkpoint:
            _save_checkpoint(model, optimizer, step, args, trainable_report, loss_history, target_names, milestone=is_milestone)
        if step in eval_steps:
            fixed_metrics, fixed_trace = _fixed_region_eval(
                model, vae_model, packer, raw_dataset, pair, fixed_eval_seed, device, region_weights
            )
            fixed_memory = _enforce_memory_gate(args, f"fixed_region_eval_step{step}")
            _log(
                "fixed_region_eval",
                step=step,
                **fixed_metrics,
                timesteps_hash=fixed_trace.get("packed_timesteps_hash"),
                memory=fixed_memory,
            )
        if _STOP_REQUESTED:
            if not should_checkpoint:
                _save_checkpoint(model, optimizer, step, args, trainable_report, loss_history, target_names, milestone=False)
            _atomic_json(args.output_dir / "run_state.json", {"status": "interrupted", "step": step, "latest_checkpoint": str(args.output_dir / "checkpoints/latest.pt")})
            _log("stopped_safely", step=step)
            return

    report = {
        "status": "completed",
        "experiment_name": EXPERIMENT_NAME,
        "completed_at": _utc_now(),
        "steps": args.steps,
        "pair_id": PAIR_ID,
        "trainable_report": trainable_report,
        "base_load_report": base_load_report,
        "loss_history": loss_history,
        "region_objective": region_spec,
        "prompt_profile": PROMPT_PROFILE,
        "memory": _memory_stats(),
        "latest_checkpoint": str(args.output_dir / "checkpoints/latest.pt"),
    }
    _atomic_json(args.output_dir / "final_report.json", report)
    _atomic_json(args.output_dir / "run_state.json", {"status": "completed", "step": args.steps, "latest_checkpoint": report["latest_checkpoint"]})
    _log(
        "training_completed",
        steps=args.steps,
        final_region_mse=loss_history[-1]["region_mse"],
        final_native_mse=loss_history[-1]["native_mse"],
        final_old_mse=loss_history[-1]["old_mse"],
        final_new_mse=loss_history[-1]["new_mse"],
        final_preserve_mse=loss_history[-1]["preserve_mse"],
        memory=_memory_stats(),
    )


if __name__ == "__main__":
    main()
