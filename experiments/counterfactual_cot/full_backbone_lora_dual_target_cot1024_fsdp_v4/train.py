#!/usr/bin/env python3
"""Round-4 paired Target-CoT training on one source with left/right targets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import signal
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from unittest import mock
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.data.schema import load_jsonl  # noqa: E402
from experiments.counterfactual_cot.full_backbone_lora_dual_target_cot1024_fsdp_v4.dual_target_data import (  # noqa: E402
    BRANCHES,
    PROMPT_PROFILE,
    build_branch_dataset,
    prompt_manifest,
    render_dual_target_prompt,
)
from experiments.counterfactual_cot.full_backbone_lora_dual_target_cot1024_fsdp_v4.region_objective import (  # noqa: E402
    OBJECTIVE_ID,
    build_paired_region_objective,
    paired_region_forward_loss,
)
from experiments.counterfactual_cot.full_backbone_lora_region1024_fsdp_v3 import train as v3  # noqa: E402
from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.train import (  # noqa: E402
    DEFAULT_TRAINABLE_REGEX,
    EXPECTED_MASK,
    MANIFEST,
    PAIR_ID,
    _build_singleton_packed_dataset,
    _pick_and_validate_pair,
    _read_num_layers,
    _resolve_pair_image,
    _stable_pair_seed,
    _validate_model_files,
    _validate_packed_trace,
)
from experiments.counterfactual_cot.training.fixed_deterministic_eval import _pack_single_pair  # noqa: E402
from experiments.counterfactual_cot.training.gold40_overfit import _load_tokenizer, _prepare_batch, _serializable_args  # noqa: E402
from experiments.counterfactual_cot.training.lora_2step_smoke import (  # noqa: E402
    TARGET_MODULE_SUFFIXES,
    _iter_target_linear_names,
    inject_lora,
    set_trainable,
)
from modeling.autoencoder import load_ae  # noqa: E402

EXPERIMENT_NAME = "cf_singlepair_fullL0-27_lora_r8_dualTargetCoT1024_fsdpFullShard_overfit_v4"
CHECKPOINT_FORMAT = "full_backbone_lora_dual_target_cot1024_fsdp_full_shard_resume_v4"
DEFAULT_MODEL = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_OUTPUT = BAGEL_ROOT / "experiments/counterfactual_cot/outputs" / EXPERIMENT_NAME
DEFAULT_MILESTONES = (0, 25, 50, 100, 200, 300, 400, 500)
DEFAULT_DIAGNOSTIC_STEPS = (0, 100, 250, 500)
TIMESTEP_SHIFT = 3.0
TIMESTEP_MIXTURE = {"native": 0.50, "high": 0.40, "pure_noise": 0.10}
_STOP_REQUESTED = False
_LOG_PATH: Path | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _log(event: str, **fields: Any) -> None:
    if not _is_rank0():
        return
    row = {"time": _utc_now(), "event": event, **fields}
    line = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    print(line, flush=True)
    if _LOG_PATH is not None:
        with _LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _atomic_json(path: Path, payload: Any) -> None:
    if not _is_rank0():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    if _is_rank0():
        print(json.dumps({"event": "stop_requested", "signal": signum}), flush=True)


def _branch_datasets(args: argparse.Namespace, tokenizer: Any) -> Dict[Tuple[str, str], Any]:
    return {
        (target_side, cot_side): build_branch_dataset(args, tokenizer, target_side, cot_side)
        for target_side in BRANCHES
        for cot_side in BRANCHES
    }


def _pack_branch(
    packer: Any,
    dataset: Any,
    pair: Dict[str, Any],
    seed: int,
) -> Tuple[Any, Dict[str, Any]]:
    batch, trace = _pack_single_pair(packer, dataset, pair, seed)
    _validate_packed_trace(trace)
    return batch, trace


def _stateless_rng(seed: int, step: int) -> np.random.Generator:
    digest = hashlib.sha256(f"v4-timestep:{seed}:{step}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def _shifted_timestep(raw_logit: float, shift: float = TIMESTEP_SHIFT) -> float:
    if math.isinf(raw_logit):
        return 1.0
    unshifted = 1.0 / (1.0 + math.exp(-raw_logit))
    return shift * unshifted / (1.0 + (shift - 1.0) * unshifted)


def _raw_logit_for_shifted_timestep(effective_t: float, shift: float = TIMESTEP_SHIFT) -> float:
    if not 0.0 < effective_t < 1.0:
        raise ValueError(f"effective_t must be in (0, 1), got {effective_t}")
    unshifted = effective_t / (shift - (shift - 1.0) * effective_t)
    return math.log(unshifted / (1.0 - unshifted))


def _timestep_schedule(seed: int, step: int) -> Dict[str, Any]:
    rng = _stateless_rng(seed, step)
    selector = float(rng.random())
    if selector < TIMESTEP_MIXTURE["native"]:
        mode = "native"
        raw = float(rng.standard_normal())
        effective = _shifted_timestep(raw)
    elif selector < TIMESTEP_MIXTURE["native"] + TIMESTEP_MIXTURE["high"]:
        mode = "high"
        effective = float(rng.uniform(0.85, 0.999))
        raw = _raw_logit_for_shifted_timestep(effective)
    else:
        mode = "pure_noise"
        effective = 1.0
        raw = float("inf")
    return {
        "mode": mode,
        "selector": selector,
        "raw_logit": raw,
        "effective_t": effective,
        "noise_seed": _stable_pair_seed(f"{PAIR_ID}:matched-noise", seed + step),
        "pack_seed": _stable_pair_seed(f"{PAIR_ID}:matched-pack", seed + step),
    }


def _override_target_timestep(batch: Any, raw_timestep: float) -> int:
    values = batch.packed_timesteps
    target_mask = torch.isfinite(values)
    count = int(target_mask.sum().item())
    if count != 4096:
        raise RuntimeError(f"Expected 4096 finite target timesteps, found {count}")
    values = values.clone()
    values[target_mask] = raw_timestep
    batch.packed_timesteps = values
    return count


def _run_branch(
    model: nn.Module,
    vae_model: nn.Module,
    packer: Any,
    dataset: Any,
    pair: Dict[str, Any],
    pack_seed: int,
    noise_seed: int,
    raw_timestep: float,
    device: torch.device,
    region_weights: Dict[str, torch.Tensor],
    backward_scale: float | None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, Any]]:
    batch, trace = _pack_branch(packer, dataset, pair, pack_seed)
    target_tokens = _override_target_timestep(batch, raw_timestep)
    data = _prepare_batch(batch, vae_model, device, torch.bfloat16)
    if tuple(data["padded_latent"].shape[-2:]) != (128, 128):
        raise RuntimeError(f"Expected 1024 VAE latent spatial size 128x128, got {tuple(data['padded_latent'].shape)}")
    expected_noise_shape = (8192, 64)
    generator = torch.Generator(device=device)
    generator.manual_seed(noise_seed)
    controlled_noise = torch.randn(
        expected_noise_shape,
        generator=generator,
        device=device,
        dtype=data["padded_latent"].dtype,
    )
    noise_hash = hashlib.sha256(controlled_noise.detach().float().cpu().numpy().tobytes()).hexdigest()
    noise_calls = 0

    def controlled_randn_like(input_tensor: torch.Tensor, *unused_args: Any, **unused_kwargs: Any) -> torch.Tensor:
        nonlocal noise_calls
        if tuple(input_tensor.shape) != expected_noise_shape:
            raise RuntimeError(f"Unexpected torch.randn_like shape inside BAGEL forward: {tuple(input_tensor.shape)}")
        if input_tensor.device != device or input_tensor.dtype != controlled_noise.dtype:
            raise RuntimeError(
                f"Unexpected torch.randn_like tensor identity: device={input_tensor.device}, dtype={input_tensor.dtype}"
            )
        noise_calls += 1
        return controlled_noise.clone()

    context = nullcontext() if backward_scale is not None else torch.no_grad()
    with torch.random.fork_rng(devices=[device.index], enabled=True):
        torch.manual_seed(noise_seed)
        torch.cuda.manual_seed(noise_seed)
        with mock.patch.object(torch, "randn_like", side_effect=controlled_randn_like):
            with context:
                loss, metrics = paired_region_forward_loss(model, data, region_weights, v3._autocast)
                v3._assert_loss_agreement(loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite branch loss: {float(loss.detach().cpu())}")
                if backward_scale is not None:
                    (loss * backward_scale).backward()
    if noise_calls != 1:
        raise RuntimeError(f"Expected exactly one controlled BAGEL noise draw, observed {noise_calls}")
    trace.update(
        target_timestep_tokens=target_tokens,
        raw_timestep=raw_timestep,
        effective_t=_shifted_timestep(raw_timestep),
        noise_seed=noise_seed,
        noise_sha256=noise_hash,
        controlled_noise_shape=list(expected_noise_shape),
        controlled_noise_calls=noise_calls,
    )
    detached_loss = loss.detach()
    del data, batch, loss, controlled_noise
    return detached_loss, metrics, trace


def _prefix(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _audit_optimizer_state(optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
    tensor_dtypes: Dict[str, int] = {}
    nonfinite: List[str] = []
    for parameter_index, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            dtype = str(value.dtype)
            tensor_dtypes[dtype] = tensor_dtypes.get(dtype, 0) + 1
            if not torch.isfinite(value).all():
                nonfinite.append(f"state[{parameter_index}].{key}")
            if value.numel() > 1 and value.dtype != torch.float32:
                raise RuntimeError(f"AdamW tensor state must be FP32, got {value.dtype} for {key}")
    if nonfinite:
        raise RuntimeError(f"Non-finite optimizer state: {nonfinite[:10]}")
    return {"tensor_dtypes": tensor_dtypes, "nonfinite_tensors": 0}


BRIDGE_BF16_OUTPUT_MODULES = (
    "connector",
    "vit_pos_embed",
    "time_embedder",
    "vae2llm",
    "latent_pos_embed",
)


def _cast_bridge_output_to_bf16(
    module: nn.Module,
    _inputs: Tuple[Any, ...],
    output: Any,
) -> torch.Tensor:
    if not torch.is_tensor(output):
        raise TypeError(f"Expected Tensor output from {type(module).__name__}, got {type(output)!r}")
    if not torch.is_floating_point(output):
        raise TypeError(f"Expected floating output from {type(module).__name__}, got {output.dtype}")
    return output if output.dtype == torch.bfloat16 else output.to(dtype=torch.bfloat16)


def _install_bridge_output_cast_hooks(model: nn.Module) -> Tuple[List[Any], Dict[str, Any]]:
    handles = []
    report: Dict[str, Any] = {"output_dtype": "torch.bfloat16", "modules": []}
    for module_name in BRIDGE_BF16_OUTPUT_MODULES:
        module = model.get_submodule(module_name)
        handles.append(module.register_forward_hook(_cast_bridge_output_to_bf16))
        report["modules"].append(
            {
                "name": module_name,
                "type": type(module).__name__,
                "trainable_parameter_dtypes": sorted(
                    {str(parameter.dtype) for parameter in module.parameters() if parameter.requires_grad}
                ),
            }
        )
    if len(handles) != len(BRIDGE_BF16_OUTPUT_MODULES):
        raise RuntimeError("Failed to install every FP32-master-to-BF16 bridge output hook")
    return handles, report


def _audit_stochastic_layers(model: nn.Module, lora_dropout: float) -> Dict[str, Any]:
    active_dropout_modules = [
        {"name": name, "type": type(module).__name__, "p": float(module.p)}
        for name, module in model.named_modules()
        if isinstance(module, nn.Dropout) and float(module.p) > 0.0
    ]
    attention_dropout = float(getattr(model.language_model.config, "attention_dropout", 0.0))
    if active_dropout_modules or attention_dropout != 0.0 or lora_dropout != 0.0:
        raise RuntimeError(
            "Matched-noise paired training requires deterministic forward recomputation: "
            f"modules={active_dropout_modules[:10]}, attention_dropout={attention_dropout}, "
            f"lora_dropout={lora_dropout}"
        )
    return {
        "active_dropout_modules": active_dropout_modules,
        "attention_dropout": attention_dropout,
        "lora_dropout": lora_dropout,
        "activation_checkpoint_rng_safe": True,
    }


def _fixed_condition_diagnostic(
    model: nn.Module,
    vae_model: nn.Module,
    packer: Any,
    datasets: Dict[Tuple[str, str], Any],
    pair: Dict[str, Any],
    seed: int,
    device: torch.device,
    region_weights: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    pack_seed = _stable_pair_seed(f"{PAIR_ID}:fixed-condition-pack", seed)
    noise_seed = _stable_pair_seed(f"{PAIR_ID}:fixed-condition-noise", seed)
    results: Dict[str, Dict[str, float]] = {}
    traces: Dict[str, Dict[str, Any]] = {}
    for target_side, cot_side in (("left", "left"), ("left", "right"), ("right", "right"), ("right", "left")):
        key = f"target_{target_side}__cot_{cot_side}"
        _, metrics, trace = _run_branch(
            model,
            vae_model,
            packer,
            datasets[(target_side, cot_side)],
            pair,
            pack_seed,
            noise_seed,
            float("inf"),
            device,
            region_weights,
            backward_scale=None,
        )
        results[key] = metrics
        traces[key] = trace
    left_correct = results["target_left__cot_left"]
    left_swapped = results["target_left__cot_right"]
    right_correct = results["target_right__cot_right"]
    right_swapped = results["target_right__cot_left"]
    gaps = {
        "left_total_swapped_minus_correct": left_swapped["total"] - left_correct["total"],
        "right_total_swapped_minus_correct": right_swapped["total"] - right_correct["total"],
        "left_left_roi_swapped_minus_correct": left_swapped["left_roi_mse"] - left_correct["left_roi_mse"],
        "left_right_roi_swapped_minus_correct": left_swapped["right_roi_mse"] - left_correct["right_roi_mse"],
        "right_left_roi_swapped_minus_correct": right_swapped["left_roi_mse"] - right_correct["left_roi_mse"],
        "right_right_roi_swapped_minus_correct": right_swapped["right_roi_mse"] - right_correct["right_roi_mse"],
    }
    noise_hashes = {trace["noise_sha256"] for trace in traces.values()}
    if (
        len(noise_hashes) != 1
        or any(trace["noise_seed"] != noise_seed or trace["effective_t"] != 1.0 for trace in traces.values())
    ):
        raise RuntimeError("Fixed condition diagnostic did not use identical pure-noise inputs")
    return {
        "fixed_effective_t": 1.0,
        "fixed_noise_seed": noise_seed,
        "fixed_pack_seed": pack_seed,
        "conditions": results,
        "correct_vs_swapped_gaps": gaps,
        "trace_summary": {
            key: {
                "sequence_length": trace["sequence_length"],
                "num_vae_tokens": trace["num_vae_tokens"],
                "num_vit_tokens": trace["num_vit_tokens"],
            }
            for key, trace in traces.items()
        },
    }


def _checkpoint_config(args: argparse.Namespace, prompts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    values = _serializable_args(args)
    values.pop("resume", None)
    values.update(
        prompt_profile=PROMPT_PROFILE,
        prompt_sha256={branch: prompts[branch]["sha256"] for branch in BRANCHES},
        paired_targets={"left": "source_image", "right": "golden_target_image"},
        paired_macro_step=True,
        matched_noise_and_timestep=True,
        branch_loss_scale=0.5,
        timestep_shift=TIMESTEP_SHIFT,
        timestep_mixture=TIMESTEP_MIXTURE,
        region_objective=OBJECTIVE_ID,
        max_latent_size=64,
        fsdp_sharding="FULL_SHARD",
        fsdp_world_size=2,
        fsdp_use_orig_params=True,
        fsdp_trainables_replicated=True,
        trainable_parameter_dtype="float32",
        optimizer_state_dtype="float32",
        lora_dtype="float32",
    )
    return values


def _save_checkpoint(
    step: int,
    args: argparse.Namespace,
    prompts: Dict[str, Dict[str, Any]],
    trainable_named: Sequence[Tuple[str, nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    target_names: Sequence[str],
    trainable_report: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    diagnostics: Sequence[Dict[str, Any]],
    milestone: bool,
) -> None:
    _barrier()
    checkpoint_dir = args.output_dir / "checkpoints"
    path = checkpoint_dir / (f"step_{step:06d}.pt" if milestone else "latest.pt")
    if _is_rank0():
        payload = {
            "format": CHECKPOINT_FORMAT,
            "experiment_name": EXPERIMENT_NAME,
            "created_at": _utc_now(),
            "step": step,
            "global_step": step,
            "config": _checkpoint_config(args, prompts),
            "trainable_state_dict": {name: parameter.detach().cpu() for name, parameter in trainable_named},
            "optimizer_state_dict": v3._cpu_tree(optimizer.state_dict()),
            "lora_target_names": list(target_names),
            "trainable_report": trainable_report,
            "loss_history": list(history),
            "condition_diagnostics": list(diagnostics),
            "sampler_protocol": "stateless_sha256(seed,macro_step); no mutable sampler state",
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }
        v3._atomic_save(path, payload)
        latest = checkpoint_dir / "latest.pt"
        if milestone:
            temporary_link = latest.with_name(latest.name + ".tmp-link")
            if temporary_link.exists():
                temporary_link.unlink()
            os.link(path, temporary_link)
            os.replace(temporary_link, latest)
        _log("checkpoint_saved", step=step, path=str(path), latest=str(latest), bytes=path.stat().st_size)
    _barrier()


def _resolve_resume(args: argparse.Namespace) -> Path | None:
    if args.resume == "none":
        return None
    candidates = [args.output_dir / "checkpoints/latest.pt"] if args.resume == "auto" else [Path(args.resume)]
    if args.resume == "auto":
        candidates.extend(sorted((args.output_dir / "checkpoints").glob("step_*.pt"), reverse=True))
    for candidate in candidates:
        try:
            if candidate.is_file():
                payload = torch.load(candidate, map_location="cpu", weights_only=False)
                valid = payload.get("format") == CHECKPOINT_FORMAT and isinstance(payload.get("trainable_state_dict"), dict)
                del payload
                if valid:
                    return candidate
        except Exception as exc:  # noqa: BLE001
            _log("checkpoint_rejected", path=str(candidate), error=repr(exc))
    if args.resume != "auto":
        raise RuntimeError(f"Invalid v4 resume checkpoint: {args.resume}")
    return None


def _load_resume(
    path: Path,
    args: argparse.Namespace,
    prompts: Dict[str, Dict[str, Any]],
    trainable_named: Sequence[Tuple[str, nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    target_names: Sequence[str],
) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("experiment_name") != EXPERIMENT_NAME:
        raise RuntimeError(f"Unsupported checkpoint identity: {payload.get('experiment_name')} / {payload.get('format')}")
    expected = _checkpoint_config(args, prompts)
    mismatches = {
        key: {"checkpoint": payload.get("config", {}).get(key), "current": value}
        for key, value in expected.items()
        if payload.get("config", {}).get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Resume config mismatch: {json.dumps(mismatches, default=str)}")
    if list(payload.get("lora_target_names", [])) != list(target_names):
        raise RuntimeError("Resume LoRA target list mismatch")
    named = dict(trainable_named)
    saved = payload["trainable_state_dict"]
    if set(saved) != set(named):
        raise RuntimeError("Resume trainable keys mismatch")
    for name, tensor in saved.items():
        if tuple(tensor.shape) != tuple(named[name].shape):
            raise RuntimeError(f"Resume shape mismatch for {name}")
        named[name].data.copy_(tensor.to(device=named[name].device, dtype=named[name].dtype))
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            for value in optimizer.state[parameter].values():
                if torch.is_tensor(value) and value.device != parameter.device:
                    value.data = value.data.to(parameter.device)
    step = int(payload.get("global_step", payload.get("step", 0)))
    history = list(payload.get("loss_history", []))
    diagnostics = list(payload.get("condition_diagnostics", []))
    if len(history) != step or [int(row.get("step", -1)) for row in history] != list(range(1, step + 1)):
        raise RuntimeError(f"Resume history is not contiguous through macro-step {step}")
    if step > 0:
        _audit_optimizer_state(optimizer)
    del payload
    gc.collect()
    _log("resume_loaded", path=str(path), macro_step=step, history_rows=len(history), diagnostic_rows=len(diagnostics))
    return step, history, diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vae-path", type=Path, default=DEFAULT_MODEL / "ae.safetensors")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_MODEL / "ema.safetensors")
    parser.add_argument("--steps", type=int, default=500, help="Optimizer macro-steps; each contains left+right micro-steps")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--trainable-name-regex", default=DEFAULT_TRAINABLE_REGEX)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--milestone-steps", type=int, nargs="+", default=list(DEFAULT_MILESTONES))
    parser.add_argument("--diagnostic-steps", type=int, nargs="+", default=list(DEFAULT_DIAGNOSTIC_STEPS))
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--precision", default="bf16", choices=("bf16",))
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--variant", default="paired_cf_mvp", choices=("paired_cf_mvp",))
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--vit-image-size", type=int, default=980)
    parser.add_argument("--vit-min-image-size", type=int, default=980)
    parser.add_argument("--expected-num-tokens", type=int, default=16384)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=16384)
    parser.add_argument("--max-num-tokens", type=int, default=16384)
    parser.add_argument("--min-gpu-memory-gib", type=float, default=38.0)
    parser.add_argument("--max-peak-reserved-gib", type=float, default=38.5)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    args.pair_id = PAIR_ID
    locked = {
        "image_size": 1024,
        "vit_image_size": 980,
        "vit_min_image_size": 980,
        "expected_num_tokens": 16384,
        "max_num_tokens_per_sample": 16384,
        "max_num_tokens": 16384,
        "num_samples": 1,
        "lora_dropout": 0.0,
    }
    mismatches = {key: {"actual": getattr(args, key), "required": value} for key, value in locked.items() if getattr(args, key) != value}
    if mismatches or args.steps < 1 or args.checkpoint_every < 1:
        parser.error(f"Locked v4 configuration invalid: {json.dumps(mismatches)}")
    if not args.activation_checkpointing and not args.preflight_only:
        parser.error("v4 1024 FSDP training requires --activation-checkpointing")
    return args


def _preflight(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    _validate_model_files(args)
    pair = _pick_and_validate_pair(load_jsonl(args.manifest))
    prompts = prompt_manifest(pair)
    source = _resolve_pair_image(pair, "source")
    target = _resolve_pair_image(pair, "target")
    region_spec, _ = build_paired_region_objective(source, target, EXPECTED_MASK)
    plan = {
        "experiment_name": EXPERIMENT_NAME,
        "pair_id": PAIR_ID,
        "input_structure": [
            "same_source_vae_1024",
            "same_source_vit_980",
            "same_source_prompt",
            "same_source_cot",
            "target_cot_switch_left_or_right",
            "matching_left_or_right_target_vae_1024_loss",
        ],
        "prompts": prompts,
        "branches": {
            "left": {"target_cot": "source.cot_text", "target_image": "source.image"},
            "right": {"target_cot": "target.cot_text", "target_image": "target.image"},
        },
        "macro_step": "0.5*left_micro_loss + 0.5*right_micro_loss before one optimizer update",
        "matched_noise_timestep": True,
        "timestep_shift": TIMESTEP_SHIFT,
        "timestep_mixture": TIMESTEP_MIXTURE,
        "region_objective": region_spec,
        "num_backbone_layers": _read_num_layers(args.llm_path),
        "expected_lora_wrappers": _read_num_layers(args.llm_path) * len(TARGET_MODULE_SUFFIXES),
        "args": _serializable_args(args),
    }
    _atomic_json(args.output_dir / "experiment_plan.json", plan)
    _atomic_json(args.output_dir / "region_spec.json", region_spec)
    return pair, prompts, plan


def main() -> None:
    global _LOG_PATH
    args = _parse_args()
    if args.preflight_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pair, prompts, plan = _preflight(args)
        _log(
            "preflight_passed",
            pair_id=pair["pair_id"],
            prompt_profile=PROMPT_PROFILE,
            prompt_sha256={branch: prompts[branch]["sha256"] for branch in BRANCHES},
            expected_lora_wrappers=plan["expected_lora_wrappers"],
        )
        return

    _, _, device = v3._validate_topology(args)
    try:
        if _is_rank0():
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "checkpoints").mkdir(exist_ok=True)
            _LOG_PATH = args.output_dir / "train_log.jsonl"
        _barrier()
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        pair, prompts, plan = _preflight(args)
        source = _resolve_pair_image(pair, "source")
        target = _resolve_pair_image(pair, "target")
        region_spec, region_weights = build_paired_region_objective(source, target, EXPECTED_MASK)

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        tokenizer, special_tokens, num_new_tokens = _load_tokenizer(str(args.llm_path))
        datasets = _branch_datasets(args, tokenizer)
        packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)
        probe_seed = _stable_pair_seed(f"{PAIR_ID}:v4-probe", args.seed)
        probe_traces = {}
        for branch in BRANCHES:
            batch, trace = _pack_branch(packer, datasets[(branch, branch)], pair, probe_seed)
            probe_traces[branch] = trace
            del batch
        if probe_traces["left"]["num_vae_tokens"] != probe_traces["right"]["num_vae_tokens"]:
            raise RuntimeError("Left/right VAE token counts differ")
        if _is_rank0():
            _log("paired_packing_validated", traces=probe_traces)

        num_layers = plan["num_backbone_layers"]
        model, base_report = v3._build_cpu_model(args, len(tokenizer), num_new_tokens)
        model.timestep_shift = TIMESTEP_SHIFT
        model.config.timestep_shift = TIMESTEP_SHIFT
        if float(model.timestep_shift) != TIMESTEP_SHIFT or float(model.config.timestep_shift) != TIMESTEP_SHIFT:
            raise RuntimeError("Failed to apply v4 timestep shift to BAGEL model")
        target_names = _iter_target_linear_names(model, list(range(num_layers)))
        expected_count = num_layers * len(TARGET_MODULE_SUFFIXES)
        if len(target_names) != expected_count:
            raise RuntimeError(f"Expected {expected_count} LoRA targets, found {len(target_names)}")
        wrappers = inject_lora(model, target_names, args.lora_r, args.lora_alpha, args.lora_dropout)
        trainable_groups = set_trainable(model, args.trainable_name_regex, wrappers)
        stochastic_audit = _audit_stochastic_layers(model, args.lora_dropout)
        trainable_named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        trainable_params = [parameter for _, parameter in trainable_named]
        if not trainable_params:
            raise RuntimeError("v4 requires non-empty replicated trainables")
        with torch.no_grad():
            for parameter in trainable_params:
                parameter.data = parameter.data.to(device=device, dtype=torch.float32, non_blocking=True)
        non_fp32 = [name for name, parameter in trainable_named if parameter.dtype != torch.float32]
        if non_fp32:
            raise RuntimeError(f"v4 requires FP32 optimizer master parameters: {non_fp32[:10]}")
        misplaced = [name for name, parameter in trainable_named if parameter.device != device]
        if misplaced:
            raise RuntimeError(f"Ignored trainables were not moved to {device}: {misplaced[:10]}")
        bridge_cast_handles, bridge_cast_report = _install_bridge_output_cast_hooks(model)
        if len(bridge_cast_handles) != 5:
            raise RuntimeError(f"Expected 5 bridge output hooks, found {len(bridge_cast_handles)}")
        fsdp_model = v3._fsdp_model(model, trainable_params, device)
        activation_layers = v3._apply_activation_checkpointing(fsdp_model, num_layers)
        if activation_layers != num_layers:
            raise RuntimeError("Not every decoder layer is activation checkpointed")
        vae_model, _ = load_ae(str(args.vae_path))
        vae_model.to(device=device, dtype=torch.bfloat16).eval()
        initial_checksum = v3._assert_replicated_trainables(trainable_params, device, "initialization")
        trainable_report = {
            "lora_wrappers": len(wrappers),
            "lora_tensors": len(trainable_groups["lora_trainable_names"]),
            "lora_parameters": sum(
                parameter.numel() for name, parameter in trainable_named
                if name in set(trainable_groups["lora_trainable_names"])
            ),
            "existing_trainable_parameters": sum(
                parameter.numel() for name, parameter in trainable_named
                if name not in set(trainable_groups["lora_trainable_names"])
            ),
            "total_trainable_parameters": sum(parameter.numel() for _, parameter in trainable_named),
            "activation_checkpointed_layers": activation_layers,
            "fsdp_ignored_trainables": len(trainable_params),
            "initial_trainable_checksum": initial_checksum,
            "trainable_dtype": "float32",
            "trainable_dtype_counts": {
                "torch.float32": sum(parameter.numel() for parameter in trainable_params if parameter.dtype == torch.float32)
            },
            "optimizer_state_dtype": "float32",
            "stochastic_audit": stochastic_audit,
            "bridge_output_cast": bridge_cast_report,
            "lora_target_names": target_names,
            **trainable_groups,
        }
        if _is_rank0():
            _atomic_json(args.output_dir / "trainable_audit.json", trainable_report)
            _log(
                "model_ready",
                lora_wrappers=len(wrappers),
                activation_checkpointed_layers=activation_layers,
                total_trainable_parameters=trainable_report["total_trainable_parameters"],
                memory=v3._memory_stats(device),
            )

        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        resume_path = _resolve_resume(args)
        if resume_path is None:
            start_step, history, diagnostics = 0, [], []
        else:
            start_step, history, diagnostics = _load_resume(
                resume_path, args, prompts, trainable_named, optimizer, target_names
            )
        v3._assert_replicated_trainables(trainable_params, device, "resume_or_fresh")
        if start_step >= args.steps:
            _log("already_complete", checkpoint_macro_step=start_step, requested_macro_steps=args.steps)
            return
        milestones = {step for step in args.milestone_steps if 0 <= step <= args.steps}
        milestones.add(args.steps)
        diagnostic_steps = {step for step in args.diagnostic_steps if 0 <= step <= args.steps}
        diagnostic_steps.add(args.steps)
        fixed_seed = _stable_pair_seed(f"{PAIR_ID}:v4-fixed-diagnostic", args.seed)

        if start_step == 0 and resume_path is None:
            diagnostic = _fixed_condition_diagnostic(
                fsdp_model, vae_model, packer, datasets, pair, fixed_seed, device, region_weights
            )
            diagnostic["step"] = 0
            diagnostics.append(diagnostic)
            memory = v3._enforce_memory_gate(args, device, "condition_diagnostic_step0")
            _log("fixed_condition_diagnostic", **diagnostic, memory=memory)
            _save_checkpoint(
                0, args, prompts, trainable_named, optimizer, target_names,
                trainable_report, history, diagnostics, 0 in milestones,
            )

        fsdp_model.train()
        for step in range(start_step + 1, args.steps + 1):
            schedule = _timestep_schedule(args.seed, step)
            optimizer.zero_grad(set_to_none=True)
            branch_results: Dict[str, Dict[str, Any]] = {}
            for branch in BRANCHES:
                _, metrics, trace = _run_branch(
                    fsdp_model,
                    vae_model,
                    packer,
                    datasets[(branch, branch)],
                    pair,
                    schedule["pack_seed"],
                    schedule["noise_seed"],
                    schedule["raw_logit"],
                    device,
                    region_weights,
                    backward_scale=0.5,
                )
                branch_results[branch] = {"metrics": metrics, "trace": trace}
            missing = [name for name, parameter in trainable_named if parameter.grad is None]
            nonfinite = [
                name for name, parameter in trainable_named
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            ]
            if missing or nonfinite:
                raise RuntimeError(f"Gradient audit failed: missing={missing[:10]}, nonfinite={nonfinite[:10]}")
            v3._sync_ignored_trainable_grads(trainable_params)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at macro-step {step}")
            optimizer.step()
            optimizer_state_audit = _audit_optimizer_state(optimizer) if step <= 2 else None
            checksum = v3._assert_replicated_trainables(trainable_params, device, f"optimizer_macro_step_{step}")
            if branch_results["left"]["trace"]["noise_seed"] != branch_results["right"]["trace"]["noise_seed"]:
                raise RuntimeError("Left/right micro-steps did not use matched noise seed")
            if branch_results["left"]["trace"]["noise_sha256"] != branch_results["right"]["trace"]["noise_sha256"]:
                raise RuntimeError("Left/right micro-steps did not use identical noise tensors")
            if branch_results["left"]["trace"]["effective_t"] != branch_results["right"]["trace"]["effective_t"]:
                raise RuntimeError("Left/right micro-steps did not use matched timestep")
            row = {
                "step": step,
                "macro_step": step,
                "micro_steps": 2,
                "macro_region_mse": 0.5 * (
                    branch_results["left"]["metrics"]["region_mse"]
                    + branch_results["right"]["metrics"]["region_mse"]
                ),
                **_prefix(branch_results["left"]["metrics"], "left"),
                **_prefix(branch_results["right"]["metrics"], "right"),
                "timestep_mode": schedule["mode"],
                "effective_t": schedule["effective_t"],
                "raw_timestep": schedule["raw_logit"],
                "matched_noise_seed": schedule["noise_seed"],
                "matched_noise_sha256": branch_results["left"]["trace"]["noise_sha256"],
                "matched_pack_seed": schedule["pack_seed"],
                "grad_norm_pre_clip": float(grad_norm.detach().float().cpu()),
                "trainable_checksum": checksum,
                "optimizer_state_audit": optimizer_state_audit,
            }
            history.append(row)
            memory = v3._enforce_memory_gate(args, device, f"paired_macro_step_{step}")
            _log("paired_train_step", **row, memory=memory)
            milestone = step in milestones
            stop = torch.tensor(int(_STOP_REQUESTED), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            global_stop = bool(stop.item())
            should_save = milestone or step % args.checkpoint_every == 0 or global_stop
            if step in diagnostic_steps:
                diagnostic = _fixed_condition_diagnostic(
                    fsdp_model, vae_model, packer, datasets, pair, fixed_seed, device, region_weights
                )
                diagnostic["step"] = step
                diagnostics.append(diagnostic)
                memory = v3._enforce_memory_gate(args, device, f"condition_diagnostic_step{step}")
                _log("fixed_condition_diagnostic", **diagnostic, memory=memory)
            if should_save:
                _save_checkpoint(
                    step, args, prompts, trainable_named, optimizer, target_names,
                    trainable_report, history, diagnostics, milestone,
                )
            if global_stop:
                _atomic_json(
                    args.output_dir / "run_state.json",
                    {"status": "interrupted", "macro_step": step, "latest_checkpoint": str(args.output_dir / "checkpoints/latest.pt")},
                )
                _log("stopped_safely", macro_step=step)
                return

        final = {
            "status": "completed",
            "experiment_name": EXPERIMENT_NAME,
            "macro_steps": args.steps,
            "micro_steps": args.steps * 2,
            "trainable_report": trainable_report,
            "base_load_report": base_report,
            "region_objective": region_spec,
            "prompt_manifest": prompts,
            "timestep_shift": TIMESTEP_SHIFT,
            "timestep_mixture": TIMESTEP_MIXTURE,
            "loss_history": history,
            "condition_diagnostics": diagnostics,
            "fsdp": {"world_size": 2, "strategy": "FULL_SHARD", "use_orig_params": True},
            "memory": v3._max_memory_stats(device),
        }
        _atomic_json(args.output_dir / "final_report.json", final)
        _atomic_json(
            args.output_dir / "run_state.json",
            {"status": "completed", "macro_step": args.steps, "latest_checkpoint": str(args.output_dir / "checkpoints/latest.pt")},
        )
        _log(
            "training_completed",
            macro_steps=args.steps,
            micro_steps=args.steps * 2,
            final_macro_region_mse=history[-1]["macro_region_mse"],
            memory=v3._max_memory_stats(device),
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
