from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader

from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.dataset_info import DATASET_INFO, DATASET_REGISTRY
from data.transforms import ImageTransform
from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.datasets import CounterfactualCoTDataset
from modeling.autoencoder import AutoEncoderParams, load_ae
from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
from modeling.qwen2 import Qwen2Tokenizer


DEFAULT_MODEL_PATH = os.environ.get("BAGEL_MODEL_PATH", "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _dtype_for_precision(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


def _cuda_memory_report(device: torch.device) -> Dict[str, float] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated_mib": round(torch.cuda.memory_allocated(index) / 1024**2, 2),
        "reserved_mib": round(torch.cuda.memory_reserved(index) / 1024**2, 2),
    }


def _log_event(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, sort_keys=True), flush=True)


def _path_size_gib(path: Path | str) -> float | None:
    value = Path(path)
    if not value.exists() or not value.is_file():
        return None
    return round(value.stat().st_size / 1024**3, 3)


def _is_placeholder_path(path: Path | str) -> bool:
    text = str(path)
    return text.startswith("/path/to") or text in {"/path", "path/to"} or "qwen_or_bagel_llm" in text


def _validate_args(args: argparse.Namespace) -> None:
    placeholder_fields = {
        "--manifest": args.manifest,
        "--output-dir": args.output_dir,
        "--llm-path": args.llm_path,
        "--vae-path": args.vae_path,
        "--checkpoint-path": args.checkpoint_path,
    }
    bad_fields = [name for name, value in placeholder_fields.items() if _is_placeholder_path(value)]
    if bad_fields:
        raise ValueError(
            "Placeholder paths are not runnable: "
            + ", ".join(f"{name}={placeholder_fields[name]}" for name in bad_fields)
            + ". Replace /path/to/... with real paths. "
            + f"Default BAGEL model path is {DEFAULT_MODEL_PATH}."
        )
    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Manifest does not exist: {args.manifest}. "
            "Pass a real counterfactual_cot manifest JSONL, for example "
            "experiments/counterfactual_cot/manifests/manifest_train.jsonl after running the data pipeline."
        )
    if not Path(args.llm_path).exists():
        raise FileNotFoundError(f"LLM/BAGEL model path does not exist: {args.llm_path}")
    if not Path(args.vae_path).exists():
        raise FileNotFoundError(f"VAE checkpoint does not exist: {args.vae_path}")
    if not args.allow_random_init and not Path(args.checkpoint_path).exists():
        raise FileNotFoundError(f"BAGEL checkpoint does not exist: {args.checkpoint_path}")


def _prepare_output_dir(output_dir: Path) -> None:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot create output directory {output_dir}. Choose a writable path such as "
            f"/dockerdata/niightli/BAGEL/experiments/counterfactual_cot/smoke_out."
        ) from exc


def _assert_manifest_has_samples(source_manifest: Path, num_samples: int) -> None:
    records = load_jsonl(source_manifest)
    if len(records) < num_samples:
        raise ValueError(f"Need at least {num_samples} manifest records, found {len(records)} in {source_manifest}")


def _register_smoke_dataset(manifest_path: Path, variant: str, seed: int) -> None:
    DATASET_REGISTRY["counterfactual_cot_smoke"] = CounterfactualCoTDataset
    DATASET_INFO["counterfactual_cot_smoke"] = {
        "smoke_train": {
            "manifest_path": str(manifest_path),
            "variant": variant,
            "training_seed": seed,
            "num_total_samples": 0,
        }
    }


def _transform_args(max_image_size: int, min_image_size: int, image_stride: int) -> Dict[str, int]:
    return {
        "max_image_size": max_image_size,
        "min_image_size": min_image_size,
        "image_stride": image_stride,
    }


def _load_tokenizer(llm_path: str) -> Tuple[Qwen2Tokenizer, Dict[str, int], int]:
    if not Path(llm_path).exists():
        raise FileNotFoundError(f"llm path does not exist: {llm_path}")
    tokenizer = Qwen2Tokenizer.from_pretrained(llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    return tokenizer, new_token_ids, num_new_tokens


def _load_bagel_checkpoint(model: Bagel, checkpoint_path: str) -> Dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"BAGEL checkpoint does not exist: {checkpoint}")
    _log_event("bagel_checkpoint_load_start", checkpoint_path=str(checkpoint), checkpoint_size_gib=_path_size_gib(checkpoint))
    state_dict = load_safetensors(str(checkpoint), device="cpu")
    _log_event("bagel_checkpoint_safetensors_loaded_to_cpu", checkpoint_path=str(checkpoint), num_tensors=len(state_dict))
    for fixed_position_key in ("latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"):
        state_dict.pop(fixed_position_key, None)
    incompatible = model.load_state_dict(state_dict, strict=False)
    _log_event(
        "bagel_checkpoint_load_state_dict_done",
        checkpoint_path=str(checkpoint),
        num_loaded_tensors=len(state_dict),
        num_missing_keys=len(incompatible.missing_keys),
        num_unexpected_keys=len(incompatible.unexpected_keys),
    )
    report = {
        "checkpoint_path": str(checkpoint),
        "num_loaded_tensors": len(state_dict),
        "num_missing_keys": len(incompatible.missing_keys),
        "num_unexpected_keys": len(incompatible.unexpected_keys),
        "missing_keys_sample": list(incompatible.missing_keys[:20]),
        "unexpected_keys_sample": list(incompatible.unexpected_keys[:20]),
    }
    if report["num_loaded_tensors"] == 0:
        raise RuntimeError(f"No tensors were loaded from {checkpoint}")
    del state_dict
    return report


def _nested_config(model_path: str, key: str) -> Dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Neither standalone {key}.json nor combined config.json exists under {model_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    value = payload.get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Combined BAGEL config does not contain object field {key!r}: {config_path}")
    return value


def _load_llm_config(model_path: str) -> Qwen2Config:
    standalone = Path(model_path) / "llm_config.json"
    if standalone.exists():
        return Qwen2Config.from_json_file(str(standalone))
    return Qwen2Config(**_nested_config(model_path, "llm_config"))


def _load_vit_config(model_path: str) -> SiglipVisionConfig:
    standalone = Path(model_path) / "vit_config.json"
    if standalone.exists():
        return SiglipVisionConfig.from_json_file(str(standalone))
    return SiglipVisionConfig(**_nested_config(model_path, "vit_config"))


def _set_trainable_parameters(model: torch.nn.Module, trainable_name_regex: str | None) -> Dict[str, Any]:
    if not trainable_name_regex:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"mode": "all_default_trainable", "trainable_tensors": None, "trainable_parameters": int(total)}
    pattern = re.compile(trainable_name_regex)
    matched_names = []
    trainable_parameters = 0
    for name, param in model.named_parameters():
        requires_grad = bool(pattern.search(name))
        param.requires_grad_(requires_grad)
        if requires_grad:
            matched_names.append(name)
            trainable_parameters += param.numel()
    if not matched_names:
        raise RuntimeError(f"--trainable-name-regex matched no parameters: {trainable_name_regex}")
    return {
        "mode": "regex_filtered",
        "regex": trainable_name_regex,
        "trainable_tensors": len(matched_names),
        "trainable_parameters": int(trainable_parameters),
        "matched_names_sample": matched_names[:20],
    }


def _vae_config() -> AutoEncoderParams:
    return AutoEncoderParams(
        resolution=256,
        in_channels=3,
        downsample=8,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    )


def _build_bagel_model(
    llm_path: str,
    vae_path: str,
    checkpoint_path: str,
    device: torch.device,
    tokenizer_size: int,
    num_new_tokens: int,
    random_init: bool,
    precision: str,
) -> Tuple[Bagel, Dict[str, Any]]:
    if not Path(llm_path).exists():
        raise FileNotFoundError(f"llm path does not exist: {llm_path}")
    if not Path(vae_path).exists():
        raise FileNotFoundError(f"vae path does not exist: {vae_path}")

    llm_config = _load_llm_config(llm_path)
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False

    vit_config = _load_vit_config(llm_path)
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    language_model = Qwen2ForCausalLM(llm_config)
    if num_new_tokens > 0:
        language_model.resize_token_embeddings(tokenizer_size)
        language_model.config.vocab_size = tokenizer_size
        llm_config.vocab_size = tokenizer_size
    vit_model = SiglipVisionModel(vit_config)
    vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
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
    model = Bagel(language_model=language_model, vit_model=vit_model, config=bagel_config)
    checkpoint_report: Dict[str, Any]
    if random_init:
        checkpoint_report = {"checkpoint_path": None, "status": "skipped_by_allow_random_init"}
    else:
        checkpoint_report = _load_bagel_checkpoint(model, checkpoint_path)
    dtype = _dtype_for_precision(precision)
    _log_event("bagel_model_to_device_start", device=str(device), dtype=str(dtype), cuda_memory=_cuda_memory_report(device))
    model.to(device=device, dtype=dtype)
    _log_event("bagel_model_to_device_done", device=str(device), dtype=str(dtype), cuda_memory=_cuda_memory_report(device))
    return model, checkpoint_report


def _build_model(
    llm_path: str,
    vae_path: str,
    checkpoint_path: str,
    device: torch.device,
    tokenizer_size: int,
    num_new_tokens: int,
    random_init: bool,
    precision: str,
) -> Tuple[Bagel, Any, Dict[str, Any]]:
    model, checkpoint_report = _build_bagel_model(
        llm_path=llm_path,
        vae_path=vae_path,
        checkpoint_path=checkpoint_path,
        device=device,
        tokenizer_size=tokenizer_size,
        num_new_tokens=num_new_tokens,
        random_init=random_init,
        precision=precision,
    )
    _log_event("vae_load_start", vae_path=str(vae_path), vae_size_gib=_path_size_gib(vae_path))
    vae_model, _ = load_ae(vae_path)
    _log_event("vae_load_done", vae_path=str(vae_path))
    dtype = _dtype_for_precision(precision)
    _log_event("vae_to_device_start", device=str(device), dtype=str(dtype), cuda_memory=_cuda_memory_report(device))
    vae_model.to(device=device, dtype=dtype).eval()
    _log_event("vae_to_device_done", device=str(device), dtype=str(dtype), cuda_memory=_cuda_memory_report(device))
    return model, vae_model, checkpoint_report


def _build_loader(manifest_path: Path, tokenizer: Qwen2Tokenizer, special_tokens: Dict[str, int], args: argparse.Namespace) -> DataLoader:
    _register_smoke_dataset(manifest_path, args.variant, args.seed)
    dataset_meta = {
        "counterfactual_cot_smoke": {
            "dataset_names": ["smoke_train"],
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
    packed_dataset.set_epoch(args.seed)
    return DataLoader(
        packed_dataset,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_wrapper(),
        drop_last=True,
    )


def _inspect_raw_samples(manifest_path: Path, tokenizer: Qwen2Tokenizer, variant: str, image_size: int, vit_image_size: int, vit_min_image_size: int, num_samples: int) -> List[Dict[str, Any]]:
    transform = ImageTransform(max_image_size=image_size, min_image_size=image_size, image_stride=16)
    vit_transform = ImageTransform(max_image_size=vit_image_size, min_image_size=vit_min_image_size, image_stride=14)
    dataset = CounterfactualCoTDataset(
        dataset_name="counterfactual_cot_smoke_raw",
        transform=transform,
        tokenizer=tokenizer,
        manifest_path=str(manifest_path),
        variant=variant,
        include_source_image=True,
        training_seed=0,
        local_rank=0,
        world_size=1,
        num_workers=0,
        data_status=None,
        vit_transform=vit_transform,
    )
    rows = []
    iterator = iter(dataset)
    for _ in range(num_samples):
        sample = next(iterator)
        rows.append(
            {
                "pair_id": sample["data_indexes"].get("pair_id"),
                "mode": sample["data_indexes"].get("mode"),
                "side": sample["data_indexes"].get("side"),
                "num_tokens": sample["num_tokens"],
                "sequence_plan": [
                    {
                        "type": item["type"],
                        "need_loss": bool(item["loss"]),
                        "enable_cfg": bool(item["enable_cfg"]),
                    }
                    for item in sample["sequence_plan"]
                ],
            }
        )
    return rows


def _move_value_to_device(value: Any, device: torch.device, floating_dtype: torch.dtype) -> Any:
    if torch.is_tensor(value):
        value = value.to(device)
        if torch.is_floating_point(value):
            value = value.to(dtype=floating_dtype)
        return value
    if isinstance(value, list):
        return [_move_value_to_device(item, device, floating_dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device, floating_dtype) for item in value)
    if isinstance(value, dict):
        return {key: _move_value_to_device(item, device, floating_dtype) for key, item in value.items()}
    return value


def _move_batch_to_device(batch: Any, device: torch.device, floating_dtype: torch.dtype) -> Dict[str, Any]:
    data = batch.to_dict()
    data.pop("batch_data_indexes", None)
    data.pop("ce_loss_weights", None)
    for key, value in list(data.items()):
        data[key] = _move_value_to_device(value, device, floating_dtype)
    return data


def _collect_float_dtype_mismatches(value: Any, expected_dtype: torch.dtype, path: str, mismatches: List[str]) -> None:
    if torch.is_tensor(value):
        if torch.is_floating_point(value) and value.dtype != expected_dtype:
            mismatches.append(f"{path}: {value.dtype}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _collect_float_dtype_mismatches(item, expected_dtype, f"{path}[{index}]", mismatches)
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _collect_float_dtype_mismatches(item, expected_dtype, f"{path}[{index}]", mismatches)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_float_dtype_mismatches(item, expected_dtype, f"{path}.{key}", mismatches)


def _assert_float_input_dtype(data: Dict[str, Any], expected_dtype: torch.dtype) -> None:
    mismatches: List[str] = []
    for key, value in data.items():
        _collect_float_dtype_mismatches(value, expected_dtype, key, mismatches)
    if mismatches:
        joined = ", ".join(mismatches[:20])
        raise RuntimeError(f"Floating input dtype mismatch before BAGEL forward; expected {expected_dtype}, got {joined}")


def _autocast_for_model(model: torch.nn.Module) -> Any:
    param = next(model.parameters())
    enabled = param.device.type == "cuda" and param.dtype in {torch.bfloat16, torch.float16}
    return torch.amp.autocast(device_type=param.device.type, enabled=enabled, dtype=param.dtype)


def _loss_from_dict(loss_dict: Dict[str, torch.Tensor], data: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = torch.zeros((), device=next(v.device for v in data.values() if torch.is_tensor(v)))
    scalars: Dict[str, float] = {}
    ce = loss_dict.get("ce")
    if ce is not None:
        ce_loss = ce.mean()
        total = total + ce_loss
        scalars["ce"] = float(ce_loss.detach().cpu())
    else:
        scalars["ce"] = 0.0
    mse = loss_dict.get("mse")
    if mse is not None:
        mse_loss = mse.mean()
        total = total + mse_loss
        scalars["mse"] = float(mse_loss.detach().cpu())
    else:
        scalars["mse"] = 0.0
    scalars["total"] = float(total.detach().cpu())
    return total, scalars


def _grad_report(model: torch.nn.Module) -> Dict[str, Any]:
    checked = 0
    nonzero = []
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        checked += 1
        grad = param.grad.detach()
        finite = bool(torch.isfinite(grad).all().item())
        norm = float(grad.float().norm().cpu())
        if finite and norm > 0:
            nonzero.append({"name": name, "norm": norm})
            if len(nonzero) >= 5:
                break
    return {"checked_trainable_grads": checked, "nonzero_finite_grads": nonzero}


def _batch_report(batch: Any, data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "device": str(device),
        "sequence_length": int(data["sequence_length"]),
        "sample_lens": list(data["sample_lens"]),
        "batch_data_indexes": batch.batch_data_indexes,
        "tensors": {},
    }
    for key, value in data.items():
        if torch.is_tensor(value):
            report["tensors"][key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
    if "mse_loss_indexes" in data:
        report["image_loss_mask"] = {
            "num_mse_tokens": int(data["mse_loss_indexes"].numel()),
            "min_index": int(data["mse_loss_indexes"].min().item()),
            "max_index": int(data["mse_loss_indexes"].max().item()),
        }
    if "ce_loss_indexes" in data:
        report["text_loss_mask"] = {"num_ce_tokens": int(data["ce_loss_indexes"].numel())}
    if "packed_vit_tokens" in data:
        report["vit_conditioning"] = {
            "num_vit_tokens": int(data["packed_vit_tokens"].shape[0]),
            "num_vit_token_indexes": int(data["packed_vit_token_indexes"].numel()),
            "vit_token_seqlens": [int(item) for item in data["vit_token_seqlens"].detach().cpu().tolist()],
        }
    return report


def _assert_paired_vit_inputs(args: argparse.Namespace, raw_sequence_report: List[Dict[str, Any]], batch_report: Dict[str, Any], data: Dict[str, Any]) -> None:
    if args.variant != "paired_cf_mvp" or not args.require_paired_vit:
        return
    if not any(any(step["type"] == "vit_image" for step in row["sequence_plan"]) for row in raw_sequence_report):
        raise RuntimeError("paired_cf_mvp smoke did not build any vit_image step for the source image")
    for row in raw_sequence_report:
        vae_steps = [step for step in row["sequence_plan"] if step["type"] == "vae_image"]
        vit_steps = [step for step in row["sequence_plan"] if step["type"] == "vit_image"]
        loss_vae_steps = [step for step in vae_steps if step["need_loss"]]
        loss_vit_steps = [step for step in vit_steps if step["need_loss"]]
        if len(vae_steps) < 2:
            raise RuntimeError(f"paired_cf_mvp smoke expected source and target VAE image steps, got: {row}")
        if not vit_steps:
            raise RuntimeError(f"paired_cf_mvp smoke expected source ViT conditioning step, got: {row}")
        if len(loss_vae_steps) != 1:
            raise RuntimeError(f"paired_cf_mvp smoke expected exactly one VAE loss image, got: {row}")
        if any(step["need_loss"] for step in vae_steps[:-1]):
            raise RuntimeError(f"paired_cf_mvp source VAE image must be conditioning-only, got: {row}")
        if not vae_steps[-1]["need_loss"]:
            raise RuntimeError(f"paired_cf_mvp target VAE image must be the only supervised image, got: {row}")
        if loss_vit_steps:
            raise RuntimeError(f"paired_cf_mvp ViT image must be conditioning-only, got: {row}")
    if "packed_vit_tokens" not in data or data["packed_vit_tokens"].numel() == 0:
        raise RuntimeError("paired_cf_mvp smoke batch has no packed_vit_tokens; source ViT branch is not exercised")
    if "vit_conditioning" not in batch_report or batch_report["vit_conditioning"]["num_vit_tokens"] <= 0:
        raise RuntimeError("paired_cf_mvp smoke report has no positive ViT conditioning token count")


def _attach_vit_forward_counter(model: Bagel) -> Tuple[Dict[str, Any], Any]:
    report: Dict[str, Any] = {"called": False, "num_calls": 0, "input_shapes": []}

    def hook(module, inputs, output):
        report["called"] = True
        report["num_calls"] += 1
        for value in inputs:
            if torch.is_tensor(value):
                report["input_shapes"].append(list(value.shape))
                break

    handle = model.vit_model.register_forward_hook(hook)
    return report, handle


def _detach_outputs_to_cpu(outputs: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in outputs.items()}


def _save_reload_check(
    model: Bagel,
    checkpoint_path: Path,
    sample_data: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    tokenizer_size: int,
    num_new_tokens: int,
) -> Tuple[Dict[str, Any], Bagel]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model.train()
    with torch.no_grad():
        torch.manual_seed(1234)
        with _autocast_for_model(model):
            before = _detach_outputs_to_cpu(model(**sample_data))
    model.to("cpu")
    torch.save({"model": model.state_dict()}, checkpoint_path)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    reloaded, _ = _build_bagel_model(
        llm_path=args.llm_path,
        vae_path=args.vae_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
        tokenizer_size=tokenizer_size,
        num_new_tokens=num_new_tokens,
        random_init=True,
        precision=args.precision,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["model"])
    del payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    reloaded.train()
    with torch.no_grad():
        torch.manual_seed(1234)
        with _autocast_for_model(reloaded):
            after = _detach_outputs_to_cpu(reloaded(**sample_data))
    max_diffs: Dict[str, float] = {}
    for key in ("ce", "mse"):
        left = before.get(key)
        right = after.get(key)
        if left is None and right is None:
            max_diffs[key] = 0.0
        elif left is None or right is None:
            max_diffs[key] = float("inf")
        else:
            max_diffs[key] = float((left - right).abs().max().detach().cpu())
    return {"checkpoint_path": str(checkpoint_path), "reload_max_abs_diff": max_diffs}, reloaded


def _snapshot_trainable_parameters(model: torch.nn.Module, max_tensors: int = 16) -> Dict[str, torch.Tensor]:
    snapshots: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            snapshots[name] = param.detach().cpu().clone()
            if len(snapshots) >= max_tensors:
                break
    if not snapshots:
        raise RuntimeError("No trainable parameters were available to snapshot before optimizer.step")
    return snapshots


def _parameter_update_report(model: torch.nn.Module, before: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    changed = []
    checked = 0
    for name, param in model.named_parameters():
        if name not in before:
            continue
        checked += 1
        after = param.detach().cpu()
        diff = (after - before[name]).abs()
        max_abs = float(diff.max())
        finite = bool(torch.isfinite(after).all().item())
        if finite and max_abs > 0.0:
            changed.append({"name": name, "max_abs_diff": max_abs})
            if len(changed) >= 5:
                break
    return {"checked_trainable_parameters": checked, "changed_trainable_parameters": changed}


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


def _trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}


def _random_state_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "python_random_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def _save_trainable_only_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path, step: int) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "counterfactual_cot_smoke_trainable_only_v1",
        "global_step": step,
        "trainable_state_dict": _trainable_state_dict(model),
        "optimizer_state_dict": _tensor_tree_to_cpu(optimizer.state_dict()),
        "random_state": _random_state_payload(),
    }
    torch.save(payload, path)
    return {"checkpoint_path": str(path), "status": "trainable_only_saved", "global_step": step, "num_trainable_tensors": len(payload["trainable_state_dict"])}


def _load_trainable_only_checkpoint(model: torch.nn.Module, path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("trainable_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint has no trainable_state_dict: {path}")
    named_params = dict(model.named_parameters())
    missing = []
    for name, tensor in state.items():
        if name not in named_params:
            missing.append(name)
            continue
        named_params[name].data.copy_(tensor.to(device=named_params[name].device, dtype=named_params[name].dtype))
    if missing:
        raise RuntimeError(f"Trainable-only checkpoint contains unknown parameters: {missing[:20]}")
    return {"checkpoint_path": str(path), "status": "trainable_only_reloaded", "global_step": payload.get("global_step"), "num_loaded_tensors": len(state)}


def _git_metadata() -> Dict[str, Any]:
    def run_git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=Path(__file__).resolve().parents[3], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    return {
        "commit": run_git("rev-parse", "HEAD"),
        "diff_status": run_git("status", "--short", "--", "modeling/bagel/qwen2_navit.py", "experiments/counterfactual_cot/training/gold40_overfit.py", "experiments/counterfactual_cot/training/smoke_train.py"),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    source_path = Path(__file__).resolve()
    print(
        json.dumps(
            {
                "smoke_train_source": str(source_path),
                "smoke_train_mtime": source_path.stat().st_mtime,
                "precision": args.precision,
                "llm_path": args.llm_path,
                "vae_path": args.vae_path,
                "vae_size_gib": _path_size_gib(args.vae_path),
                "checkpoint_path": args.checkpoint_path,
                "checkpoint_size_gib": _path_size_gib(args.checkpoint_path),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    _set_seed(args.seed)
    device = _device(args.device)
    _validate_args(args)
    output_dir = args.output_dir
    _prepare_output_dir(output_dir)

    _assert_manifest_has_samples(args.manifest, args.num_samples)
    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)
    raw_sequence_report = _inspect_raw_samples(
        args.manifest,
        tokenizer,
        args.variant,
        args.image_size,
        args.vit_image_size,
        args.vit_min_image_size,
        args.num_samples,
    )
    loader = _build_loader(args.manifest, tokenizer, special_tokens, args)
    batch = next(iter(loader))

    model, vae_model, checkpoint_load_report = _build_model(
        args.llm_path,
        args.vae_path,
        args.checkpoint_path,
        device,
        tokenizer_size=len(tokenizer),
        num_new_tokens=num_new_tokens,
        random_init=args.allow_random_init,
        precision=args.precision,
    )
    trainable_report = _set_trainable_parameters(model, args.trainable_name_regex)
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model_dtype = next(model.parameters()).dtype
    _log_event(
        "preflight_runtime_context",
        checkpoint_path=args.checkpoint_path,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        model_main_device=str(next(model.parameters()).device),
        trainable_parameters=trainable_report["trainable_parameters"],
        manifest=str(args.manifest),
        git=_git_metadata(),
    )
    data = _move_batch_to_device(batch, device, model_dtype)
    batch_report = _batch_report(batch, data, device)
    _assert_paired_vit_inputs(args, raw_sequence_report, batch_report, data)
    with torch.no_grad():
        vae_dtype = next(vae_model.parameters()).dtype
        data["padded_latent"] = vae_model.encode(data.pop("padded_images").to(dtype=vae_dtype))
    _assert_float_input_dtype(data, model_dtype)
    optimizer.zero_grad(set_to_none=True)
    vit_forward_report: Dict[str, Any] = {"checked": args.variant == "paired_cf_mvp" and args.require_paired_vit, "called": None}
    vit_hook_handle = None
    if args.variant == "paired_cf_mvp" and args.require_paired_vit:
        vit_forward_report, vit_hook_handle = _attach_vit_forward_counter(model)
    try:
        with _autocast_for_model(model):
            loss_dict = model(**data)
    finally:
        if vit_hook_handle is not None:
            vit_hook_handle.remove()
    if args.variant == "paired_cf_mvp" and args.require_paired_vit and not vit_forward_report.get("called"):
        raise RuntimeError("paired_cf_mvp smoke batch had ViT inputs, but model.vit_model.forward was not called")
    loss, loss_scalars = _loss_from_dict(loss_dict, data)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite total loss: {loss_scalars}")
    loss.backward()
    grad_report = _grad_report(model)
    if not grad_report["nonzero_finite_grads"]:
        raise RuntimeError("No expected trainable parameter has non-zero finite gradient")
    params_before_step = _snapshot_trainable_parameters(model)
    optimizer.step()
    parameter_update_report = _parameter_update_report(model, params_before_step)
    if not parameter_update_report["changed_trainable_parameters"]:
        raise RuntimeError("optimizer.step did not change any sampled trainable parameter")

    if args.checkpoint_mode == "trainable_only":
        checkpoint_report = _save_trainable_only_checkpoint(model, optimizer, output_dir / "smoke_trainable_checkpoint.pt", 1)
        reload_report = _load_trainable_only_checkpoint(model, Path(checkpoint_report["checkpoint_path"]))
        checkpoint_report = {**checkpoint_report, "reload_report": reload_report}
    else:
        reload_input = {key: value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in data.items()}
        del optimizer
        vae_model.to("cpu")
        del vae_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        checkpoint_report, model = _save_reload_check(
            model,
            output_dir / "smoke_checkpoint.pt",
            reload_input,
            args,
            device,
            tokenizer_size=len(tokenizer),
            num_new_tokens=num_new_tokens,
        )
        max_reload_diff = max(checkpoint_report["reload_max_abs_diff"].values())
        if max_reload_diff > args.reload_atol:
            raise RuntimeError(f"Reloaded checkpoint output mismatch: {checkpoint_report}")

    report = {
        "status": "passed",
        "variant": args.variant,
        "num_manifest_samples": args.num_samples,
        "raw_sequence_report": raw_sequence_report,
        "batch_report": batch_report,
        "loss": loss_scalars,
        "grad_report": grad_report,
        "parameter_update_report": parameter_update_report,
        "checkpoint_load_report": checkpoint_load_report,
        "trainable_report": trainable_report,
        "vit_forward_report": vit_forward_report,
        "checkpoint_report": checkpoint_report,
    }
    report_path = output_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P0 single-batch BAGEL smoke train for Counterfactual CoT.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--llm-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vae-path", default=str(Path(DEFAULT_MODEL_PATH) / "ae.safetensors"))
    parser.add_argument("--checkpoint-path", default=str(Path(DEFAULT_MODEL_PATH) / "ema.safetensors"))
    parser.add_argument("--variant", default="paired_cf_generation", choices=sorted(CounterfactualCoTDataset.VALID_VARIANTS))
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--vit-min-image-size", type=int, default=224)
    parser.add_argument("--expected-num-tokens", type=int, default=2048)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=4096)
    parser.add_argument("--max-num-tokens", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--reload-atol", type=float, default=0.0)
    parser.add_argument("--checkpoint-mode", default="full_model", choices=("full_model", "trainable_only"))
    parser.add_argument("--trainable-name-regex", default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--no-require-paired-vit", dest="require_paired_vit", action="store_false")
    parser.set_defaults(require_paired_vit=True)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
