#!/usr/bin/env python3
"""Round-3: two-rank FSDP FULL_SHARD training for the locked 1024 CoT-core pair."""

from __future__ import annotations

import argparse
import functools
import gc
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
import torch.distributed as dist
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.data.schema import load_jsonl  # noqa: E402
from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.region_objective import (  # noqa: E402
    build_region_objective,
    region_forward_loss,
)
from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.train import (  # noqa: E402
    DEFAULT_TRAINABLE_REGEX,
    EXPECTED_MASK,
    EXPECTED_TARGET,
    MANIFEST,
    PAIR_ID,
    PROMPT_PROFILE,
    _build_raw_sample_dataset,
    _build_singleton_packed_dataset,
    _pick_and_validate_pair,
    _read_num_layers,
    _resolve_pair_image,
    _stable_pair_seed,
    _validate_base_load_report,
    _validate_model_files,
    _validate_packed_trace,
)
from experiments.counterfactual_cot.training.fixed_deterministic_eval import _pack_single_pair  # noqa: E402
from experiments.counterfactual_cot.training.gold40_overfit import (  # noqa: E402
    _load_tokenizer,
    _prepare_batch,
    _serializable_args,
)
from experiments.counterfactual_cot.training.lora_2step_smoke import (  # noqa: E402
    TARGET_MODULE_SUFFIXES,
    _iter_target_linear_names,
    set_trainable,
)
from experiments.counterfactual_cot.training.smoke_train import (  # noqa: E402
    _dtype_for_precision,
    _load_bagel_checkpoint,
    _load_llm_config,
    _load_vit_config,
    _vae_config,
)
from modeling.autoencoder import load_ae  # noqa: E402
from modeling.bagel import Bagel, BagelConfig, Qwen2ForCausalLM, SiglipVisionModel  # noqa: E402
from modeling.bagel.modeling_utils import MLPconnector, PositionEmbedding, TimestepEmbedder  # noqa: E402
from modeling.bagel.qwen2_navit import Qwen2MoTDecoderLayer  # noqa: E402
from modeling.bagel.siglip_navit import SiglipEncoderLayer, SiglipVisionTransformer  # noqa: E402

EXPERIMENT_NAME = "cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_overfit_v3"
CHECKPOINT_FORMAT = "full_backbone_lora_region1024_cotcore_fsdp_full_shard_resume_v3"
DEFAULT_MODEL = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_OUTPUT = BAGEL_ROOT / "experiments/counterfactual_cot/outputs" / EXPERIMENT_NAME
DEFAULT_MILESTONES = (0, 25, 50, 100, 200, 300, 400, 500)
_STOP_REQUESTED = False
_LOG_PATH: Path | None = None


class FSDPCompatibleLoRALinear(nn.Module):
    """BF16 LoRA required because every FSDP flat-parameter unit has one dtype."""

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear, got {type(base)!r}")
        if base.weight.dtype != torch.bfloat16:
            raise RuntimeError(f"FSDP v3 requires BF16 base linears, got {base.weight.dtype}")
        self.base = base
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout_p = float(dropout)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features, dtype=torch.bfloat16))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        update_input = x.to(dtype=self.lora_A.dtype)
        if self.dropout_p > 0.0 and self.training:
            update_input = torch.nn.functional.dropout(update_input, p=self.dropout_p)
        update = torch.nn.functional.linear(update_input, self.lora_A)
        update = torch.nn.functional.linear(update, self.lora_B)
        return base_out + update.to(dtype=base_out.dtype) * self.scaling


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _is_rank0() -> bool:
    return _rank() == 0


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
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


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    if not _is_rank0():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temp, path)


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    if _is_rank0():
        print(json.dumps({"event": "stop_requested", "signal": signum}), flush=True)


def _canonical_name(name: str) -> str:
    for fragment in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
        name = name.replace(fragment, "")
    return name.replace("..", ".").strip(".")


def _inject_fsdp_lora(model: nn.Module, target_names: Sequence[str], r: int, alpha: int, dropout: float) -> Dict[str, FSDPCompatibleLoRALinear]:
    wrappers: Dict[str, FSDPCompatibleLoRALinear] = {}
    for full_name in target_names:
        parent_name, _, child_name = full_name.rpartition(".")
        parent = model.get_submodule(parent_name)
        base = getattr(parent, child_name)
        wrapper = FSDPCompatibleLoRALinear(base, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, wrapper)
        wrappers[full_name] = wrapper
    return wrappers


def _build_cpu_model(args: argparse.Namespace, tokenizer_size: int, num_new_tokens: int) -> Tuple[Bagel, Dict[str, Any]]:
    llm_config = _load_llm_config(str(args.llm_path))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    vit_config = _load_vit_config(str(args.llm_path))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    language_model = Qwen2ForCausalLM(llm_config)
    if num_new_tokens > 0:
        language_model.resize_token_embeddings(tokenizer_size)
        language_model.config.vocab_size = tokenizer_size
        llm_config.vocab_size = tokenizer_size
    vit_model = SiglipVisionModel(vit_config)
    vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=_vae_config(),
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
        timestep_shift=1.0,
    )
    model = Bagel(language_model=language_model, vit_model=vit_model, config=config)
    report = _load_bagel_checkpoint(model, str(args.checkpoint_path))
    _validate_base_load_report(report)
    model.to(dtype=torch.bfloat16)
    return model, report


def _apply_activation_checkpointing(model: nn.Module, expected_layers: int) -> int:
    base_layers = sum(isinstance(module, Qwen2MoTDecoderLayer) for module in model.modules())
    if base_layers != expected_layers:
        raise RuntimeError(f"Expected {expected_layers} decoder layers before activation checkpointing, found {base_layers}")
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda module: isinstance(module, Qwen2MoTDecoderLayer),
    )
    wrappers = sum(isinstance(module, CheckpointWrapper) for module in model.modules())
    remaining = sum(isinstance(module, Qwen2MoTDecoderLayer) for module in model.modules())
    if wrappers != expected_layers or remaining != expected_layers:
        raise RuntimeError(f"Activation checkpoint mismatch: wrappers={wrappers}, layers={remaining}, expected={expected_layers}")
    return wrappers


def _fsdp_model(model: nn.Module, trainable_params: Sequence[nn.Parameter], device: torch.device) -> FSDP:
    # Bridge modules contain only ignored trainables. Wrapping them separately
    # leaves their parameters unmanaged and may strand them on CPU.
    classes = {Qwen2MoTDecoderLayer, SiglipEncoderLayer, SiglipVisionTransformer}
    return FSDP(
        model,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=classes),
        ignored_states=list(trainable_params),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        sync_module_states=False,
        limit_all_gathers=True,
        use_orig_params=True,
    )


def _validate_gpu_capacity(minimum_gib: float) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    total_gib = properties.total_memory / 1024**3
    if total_gib < minimum_gib:
        raise RuntimeError(f"GPU {properties.name} has only {total_gib:.2f} GiB; need >= {minimum_gib}")
    return {"name": properties.name, "total_memory_gib": round(total_gib, 2)}


def _memory_stats(device: torch.device) -> Dict[str, float]:
    gib = 1024**3
    return {
        "allocated_gib": round(torch.cuda.memory_allocated(device) / gib, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(device) / gib, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / gib, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / gib, 3),
    }


def _max_memory_stats(device: torch.device) -> Dict[str, float]:
    local = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)], dtype=torch.float64, device=device)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    gib = 1024**3
    return {"max_rank_peak_allocated_gib": round(float(local[0].item()) / gib, 3), "max_rank_peak_reserved_gib": round(float(local[1].item()) / gib, 3)}


def _enforce_memory_gate(args: argparse.Namespace, device: torch.device, stage: str) -> Dict[str, float]:
    report = _max_memory_stats(device)
    if report["max_rank_peak_reserved_gib"] >= args.max_peak_reserved_gib:
        raise RuntimeError(f"FSDP memory gate failed at {stage}: {report['max_rank_peak_reserved_gib']} >= {args.max_peak_reserved_gib} GiB")
    return report


def _checksum(parameters: Sequence[nn.Parameter], device: torch.device) -> torch.Tensor:
    value = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in parameters:
        value += parameter.detach().float().sum().double()
        value += parameter.detach().float().square().sum().double() * 1e-8
    return value


def _assert_replicated_trainables(parameters: Sequence[nn.Parameter], device: torch.device, stage: str) -> float:
    local = _checksum(parameters, device)
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    values = [float(item.item()) for item in gathered]
    if max(values) - min(values) > 1e-5 * max(1.0, abs(values[0])):
        raise RuntimeError(f"Replicated trainable checksum diverged at {stage}: {values}")
    return values[0]


def _assert_loss_agreement(loss: torch.Tensor) -> None:
    local = loss.detach().float()
    gathered = [torch.zeros_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    values = [float(item.item()) for item in gathered]
    if max(values) - min(values) > 1e-6 * max(1.0, abs(values[0])):
        raise RuntimeError(f"Rank loss mismatch: {values}")


def _sync_ignored_trainable_grads(parameters: Sequence[nn.Parameter]) -> None:
    """FSDP does not reduce gradients for explicitly ignored replicated states."""
    world_size = dist.get_world_size()
    for parameter in parameters:
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _atomic_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def _checkpoint_config(args: argparse.Namespace) -> Dict[str, Any]:
    values = _serializable_args(args)
    values.update(
        prompt_profile=PROMPT_PROFILE,
        region_objective="0.25_old_0.25_new_0.50_preserve",
        max_latent_size=64,
        fsdp_sharding="FULL_SHARD",
        fsdp_world_size=2,
        fsdp_use_orig_params=True,
        fsdp_trainables_replicated=True,
        lora_dtype="bfloat16",
    )
    return values


def _save_checkpoint(
    step: int,
    args: argparse.Namespace,
    trainable_named: Sequence[Tuple[str, nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    target_names: Sequence[str],
    trainable_report: Dict[str, Any],
    loss_history: Sequence[Dict[str, Any]],
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
            "config": _checkpoint_config(args),
            "trainable_state_dict": {name: parameter.detach().cpu() for name, parameter in trainable_named},
            "optimizer_state_dict": {key: value for key, value in _cpu_tree(optimizer.state_dict()).items()},
            "lora_target_names": list(target_names),
            "trainable_report": trainable_report,
            "loss_history": list(loss_history),
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()},
        }
        _atomic_save(path, payload)
        latest = checkpoint_dir / "latest.pt"
        if milestone:
            temp_link = latest.with_name(latest.name + ".tmp-link")
            if temp_link.exists():
                temp_link.unlink()
            os.link(path, temp_link)
            os.replace(temp_link, latest)
        _log("checkpoint_saved", step=step, path=str(path), latest=str(latest), bytes=path.stat().st_size)
    _barrier()


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _resolve_resume(args: argparse.Namespace) -> Path | None:
    if args.resume == "none":
        return None
    candidates = [args.output_dir / "checkpoints" / "latest.pt"] if args.resume == "auto" else [Path(args.resume)]
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
        raise RuntimeError(f"Invalid resume checkpoint: {args.resume}")
    return None


def _load_resume(path: Path, args: argparse.Namespace, trainable_named: Sequence[Tuple[str, nn.Parameter]], optimizer: torch.optim.Optimizer, target_names: Sequence[str]) -> Tuple[int, List[Dict[str, Any]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Unsupported checkpoint format: {payload.get('format')}")
    expected = _checkpoint_config(args)
    mismatches = {key: {"checkpoint": payload.get("config", {}).get(key), "current": value} for key, value in expected.items() if payload.get("config", {}).get(key) != value}
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
    del payload
    gc.collect()
    _log("resume_loaded", path=str(path), step=step, history_rows=len(history))
    return step, history


def _fixed_eval(model: nn.Module, vae_model: nn.Module, packer: Any, raw_dataset: Any, pair: Dict[str, Any], seed: int, device: torch.device, weights: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch, trace = _pack_single_pair(packer, raw_dataset, pair, seed)
    _validate_packed_trace(trace)
    data = _prepare_batch(batch, vae_model, device, torch.bfloat16)
    model.train()
    with torch.no_grad():
        loss, scalars = region_forward_loss(model, data, weights, _autocast)
    _assert_loss_agreement(loss)
    del data, loss
    return scalars, trace


def _autocast(_model: nn.Module) -> Any:
    return torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)


def _validate_topology(args: argparse.Namespace) -> Tuple[int, int, torch.device]:
    if not torch.cuda.is_available():
        raise RuntimeError("FSDP v3 requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if local_rank not in (0, 1) or world_size != 2:
        raise RuntimeError(f"FSDP v3 requires torchrun --nproc_per_node=2; LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    total_gib = properties.total_memory / 1024**3
    if total_gib < args.min_gpu_memory_gib:
        raise RuntimeError(f"GPU {local_rank} has only {total_gib:.2f} GiB; need >= {args.min_gpu_memory_gib}")
    gathered: List[Dict[str, Any]] = [dict() for _ in range(2)]
    dist.all_gather_object(gathered, {"rank": _rank(), "local_rank": local_rank, "name": properties.name, "total_memory_gib": round(total_gib, 2)})
    if _is_rank0():
        _log("fsdp_topology_validated", ranks=gathered, strategy="FULL_SHARD", use_orig_params=True)
    return _rank(), local_rank, device


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
    parser.add_argument("--milestone-steps", type=int, nargs="+", default=list(DEFAULT_MILESTONES))
    parser.add_argument("--eval-steps", type=int, nargs="+", default=list(DEFAULT_MILESTONES))
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
    args = parser.parse_args()
    args.pair_id = PAIR_ID
    required = {"image_size": 1024, "vit_image_size": 980, "vit_min_image_size": 980, "expected_num_tokens": 16384, "max_num_tokens_per_sample": 16384, "max_num_tokens": 16384, "num_samples": 1}
    mismatch = {key: {"actual": getattr(args, key), "required": value} for key, value in required.items() if getattr(args, key) != value}
    if mismatch or not args.activation_checkpointing or args.steps < 1 or args.checkpoint_every < 1:
        parser.error(f"Locked v3 configuration invalid: mismatch={json.dumps(mismatch)}, activation_checkpointing={args.activation_checkpointing}")
    return args


def main() -> None:
    global _LOG_PATH
    args = parse_args()
    rank, local_rank, device = _validate_topology(args)
    try:
        if _is_rank0():
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "checkpoints").mkdir(exist_ok=True)
            _LOG_PATH = args.output_dir / "train_log.jsonl"
        _barrier()
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)
        _validate_model_files(args)
        pair = _pick_and_validate_pair(load_jsonl(args.manifest))
        source = _resolve_pair_image(pair, "source")
        target = _resolve_pair_image(pair, "target")
        region_spec, region_weights = build_region_objective(source, target, EXPECTED_MASK)
        num_layers = _read_num_layers(args.llm_path)
        if _is_rank0():
            plan = {
                "experiment_name": EXPERIMENT_NAME,
                "pair_id": PAIR_ID,
                "input_structure": ["source_vae_1024", "source_vit_980", "source_prompt_plus_target_cot", "target_vae_1024_loss"],
                "prompt_profile": PROMPT_PROFILE,
                "source": str(source),
                "target": str(target),
                "region_objective": region_spec,
                "fsdp": {"world_size": 2, "sharding_strategy": "FULL_SHARD", "use_orig_params": True, "managed_params": "frozen_base_only", "replicated_params": "LoRA_and_bridge_trainables"},
                "args": _serializable_args(args),
            }
            _atomic_json(args.output_dir / "experiment_plan.json", plan)
            _atomic_json(args.output_dir / "region_spec.json", region_spec)
            _log("preflight_passed", pair_id=PAIR_ID, prompt_profile=PROMPT_PROFILE, expected_lora_wrappers=num_layers * len(TARGET_MODULE_SUFFIXES), region_token_masses=region_spec["token_masses"])
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        tokenizer, special_tokens, num_new_tokens = _load_tokenizer(str(args.llm_path))
        raw_dataset = _build_raw_sample_dataset(args, tokenizer)
        packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)
        pair_seed = _stable_pair_seed(PAIR_ID, args.seed)
        probe_batch, probe_trace = _pack_single_pair(packer, raw_dataset, pair, pair_seed)
        _validate_packed_trace(probe_trace)
        del probe_batch
        if _is_rank0():
            _log("packed_1024_validated", **probe_trace)
        model, base_report = _build_cpu_model(args, len(tokenizer), num_new_tokens)
        target_names = _iter_target_linear_names(model, list(range(num_layers)))
        if len(target_names) != num_layers * len(TARGET_MODULE_SUFFIXES):
            raise RuntimeError("Unexpected all-layer LoRA target count")
        wrappers = _inject_fsdp_lora(model, target_names, args.lora_r, args.lora_alpha, args.lora_dropout)
        trainable_groups = set_trainable(model, args.trainable_name_regex, wrappers)
        trainable_named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        trainable_params = [parameter for _, parameter in trainable_named]
        if any(parameter.dtype != torch.bfloat16 for parameter in trainable_params):
            raise RuntimeError("FSDP v3 requires all ignored trainables to be BF16")
        if not trainable_params:
            raise RuntimeError("No trainable parameters")
        # FSDP intentionally ignores the LoRA/bridge parameters. Move only these
        # small replicated tensors to their owning rank; never move the frozen base
        # model before FSDP has sharded it.
        with torch.no_grad():
            for parameter in trainable_params:
                parameter.data = parameter.data.to(device=device, non_blocking=True)
        misplaced = [name for name, parameter in trainable_named if parameter.device != device]
        if misplaced:
            raise RuntimeError(f"Ignored trainables were not moved to local GPU {device}: {misplaced[:10]}")
        fsdp_model = _fsdp_model(model, trainable_params, device)
        activation_layers = _apply_activation_checkpointing(fsdp_model, num_layers)
        if activation_layers != num_layers:
            raise RuntimeError("Not every decoder layer is activation checkpointed")
        vae_model, _ = load_ae(str(args.vae_path))
        vae_model.to(device=device, dtype=_dtype_for_precision(args.precision)).eval()
        trainable_checksum = _assert_replicated_trainables(trainable_params, device, "initialization")
        trainable_report = {
            "lora_wrappers": len(wrappers),
            "lora_tensors": len(trainable_groups["lora_trainable_names"]),
            "lora_parameters": sum(parameter.numel() for name, parameter in trainable_named if name in set(trainable_groups["lora_trainable_names"])),
            "existing_trainable_parameters": sum(parameter.numel() for name, parameter in trainable_named if name not in set(trainable_groups["lora_trainable_names"])),
            "total_trainable_parameters": sum(parameter.numel() for _, parameter in trainable_named),
            "activation_checkpointed_layers": activation_layers,
            "fsdp_ignored_trainables": len(trainable_params),
            "initial_trainable_checksum": trainable_checksum,
            "lora_target_names": target_names,
            **trainable_groups,
        }
        if _is_rank0():
            _atomic_json(args.output_dir / "trainable_audit.json", trainable_report)
            _log("model_ready", **{key: trainable_report[key] for key in ("lora_wrappers", "lora_parameters", "existing_trainable_parameters", "total_trainable_parameters", "activation_checkpointed_layers", "fsdp_ignored_trainables")}, memory=_memory_stats(device))
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        resume_path = _resolve_resume(args)
        start_step, history = (0, []) if resume_path is None else _load_resume(resume_path, args, trainable_named, optimizer, target_names)
        _assert_replicated_trainables(trainable_params, device, "resume_or_fresh")
        if start_step >= args.steps:
            _log("already_complete", checkpoint_step=start_step, requested_steps=args.steps)
            return
        milestones = {step for step in args.milestone_steps if 0 <= step <= args.steps}
        milestones.add(args.steps)
        eval_steps = {step for step in args.eval_steps if 0 <= step <= args.steps}
        fixed_seed = pair_seed + 1_000_000
        if start_step == 0 and resume_path is None:
            metrics, trace = _fixed_eval(fsdp_model, vae_model, packer, raw_dataset, pair, fixed_seed, device, region_weights)
            memory = _enforce_memory_gate(args, device, "fixed_eval_step0")
            _log("fixed_region_eval", step=0, **metrics, timesteps_hash=trace.get("packed_timesteps_hash"), memory=memory)
            _save_checkpoint(0, args, trainable_named, optimizer, target_names, trainable_report, history, 0 in milestones)
        fsdp_model.train()
        for step in range(start_step + 1, args.steps + 1):
            torch.manual_seed(pair_seed + step)
            torch.cuda.manual_seed_all(pair_seed + step)
            batch, trace = _pack_single_pair(packer, raw_dataset, pair, pair_seed + step)
            _validate_packed_trace(trace)
            data = _prepare_batch(batch, vae_model, device, torch.bfloat16)
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = region_forward_loss(fsdp_model, data, region_weights, _autocast)
            _assert_loss_agreement(loss)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            missing = [name for name, parameter in trainable_named if parameter.grad is None]
            nonfinite = [name for name, parameter in trainable_named if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
            if missing or nonfinite:
                raise RuntimeError(f"Gradient audit failed at step {step}: missing={missing[:10]}, nonfinite={nonfinite[:10]}")
            _sync_ignored_trainable_grads(trainable_params)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite grad norm at step {step}")
            optimizer.step()
            checksum = _assert_replicated_trainables(trainable_params, device, f"optimizer_step_{step}")
            memory = _enforce_memory_gate(args, device, f"train_step_{step}")
            row = {"step": step, **metrics, "grad_norm_pre_clip": float(grad_norm.detach().float().cpu()), "trainable_checksum": checksum, "step_seed": pair_seed + step, "timesteps_hash": trace.get("packed_timesteps_hash")}
            history.append(row)
            _log("train_step", **row, memory=memory)
            del data, loss
            milestone = step in milestones
            should_save = milestone or step % args.checkpoint_every == 0 or _STOP_REQUESTED
            if should_save:
                _save_checkpoint(step, args, trainable_named, optimizer, target_names, trainable_report, history, milestone)
            if step in eval_steps:
                fixed_metrics, fixed_trace = _fixed_eval(fsdp_model, vae_model, packer, raw_dataset, pair, fixed_seed, device, region_weights)
                memory = _enforce_memory_gate(args, device, f"fixed_eval_step{step}")
                _log("fixed_region_eval", step=step, **fixed_metrics, timesteps_hash=fixed_trace.get("packed_timesteps_hash"), memory=memory)
            stop = torch.tensor(int(_STOP_REQUESTED), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if int(stop.item()):
                _atomic_json(args.output_dir / "run_state.json", {"status": "interrupted", "step": step, "latest_checkpoint": str(args.output_dir / "checkpoints/latest.pt")})
                _log("stopped_safely", step=step)
                return
        _atomic_json(args.output_dir / "final_report.json", {"status": "completed", "experiment_name": EXPERIMENT_NAME, "steps": args.steps, "trainable_report": trainable_report, "base_load_report": base_report, "region_objective": region_spec, "loss_history": history, "fsdp": {"world_size": 2, "strategy": "FULL_SHARD", "use_orig_params": True}, "memory": _max_memory_stats(device)})
        _atomic_json(args.output_dir / "run_state.json", {"status": "completed", "step": args.steps, "latest_checkpoint": str(args.output_dir / "checkpoints/latest.pt")})
        _log("training_completed", steps=args.steps, final_region_mse=history[-1]["region_mse"], final_old_mse=history[-1]["old_mse"], final_new_mse=history[-1]["new_mse"], final_preserve_mse=history[-1]["preserve_mse"], memory=_max_memory_stats(device))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
