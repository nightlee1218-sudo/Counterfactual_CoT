#!/usr/bin/env python3
"""Gold40 AutoMask-v5 paired source/target-state FSDP FULL_SHARD trainer."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.full_backbone_lora_dual_target_cot1024_fsdp_v4 import train as v4_runtime
from experiments.counterfactual_cot.full_backbone_lora_region1024_fsdp_v3 import train as v3_runtime
from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.train import DEFAULT_TRAINABLE_REGEX, _read_num_layers, _validate_model_files
from experiments.counterfactual_cot.gold40_automask_dual_target_cot1024_fsdp_v5.paired_data import (
    STATE_IDS,
    build_state_dataset,
    load_automask_manifest,
    prompt_audit,
)
from experiments.counterfactual_cot.training.fixed_deterministic_eval import _build_singleton_packed_dataset, _pack_single_pair
from experiments.counterfactual_cot.training.gold40_overfit import _load_tokenizer, _prepare_batch, _serializable_args
from experiments.counterfactual_cot.training.lora_2step_smoke import TARGET_MODULE_SUFFIXES, _iter_target_linear_names, inject_lora, set_trainable
from modeling.autoencoder import load_ae

EXPERIMENT_NAME = "gold40_automask_v5_source_target_cot1024_fsdp_full_shard"
DEFAULT_MODEL = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_OUTPUT = BAGEL_ROOT / "experiments/counterfactual_cot/outputs" / EXPERIMENT_NAME
DEFAULT_MANIFEST = BAGEL_ROOT / "experiments/counterfactual_cot/gold40_automask_v5_data/artifacts/gold40_source_target_automask_v5.jsonl"
DEFAULT_OBJECTIVE_MODULE = "experiments.counterfactual_cot.gold40_automask_v5_data.automask_objective"
MIXTURE = {"native": 0.50, "high": 0.40, "pure_noise": 0.10}
SHIFT = 3.0
EXPECTED_TARGET_TIMESTEP_TOKENS = 4096
EXPECTED_NOISE_SHAPE = (8192, 64)
EXPECTED_ACTIVATION_CHECKPOINTS = 28
CHECKPOINT_FORMAT = "gold40_automask_v5_paired_fsdp"
RESUME_GUARD_KEYS = ("manifest", "objective_id", "objective_module", "record_count")


def _hash_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big") & 0x7FFFFFFF


def _rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _atomic_json(path: Path, payload: Any) -> None:
    if not _rank0():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp.replace(path)


def _log(event: str, **fields: Any) -> None:
    if _rank0():
        print(json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True, default=str), flush=True)


def _load_objective(module_name: str) -> Tuple[str, Any, Any]:
    """Load the separately-owned v5 paired API, adapting its branch API if needed."""
    module = importlib.import_module(module_name)
    objective_id = getattr(module, "OBJECTIVE_ID", None)
    build = getattr(module, "build_paired_automask_objective", None)
    forward = getattr(module, "paired_automask_forward_loss", None)
    if not isinstance(objective_id, str) or not objective_id:
        raise RuntimeError(f"{module_name}: missing non-empty OBJECTIVE_ID")
    if callable(build) and callable(forward):
        return objective_id, build, forward

    branch_build = getattr(module, "build_automask_objective", None)
    branch_forward = getattr(module, "automask_forward_loss", None)
    if not callable(branch_build) or not callable(branch_forward):
        raise RuntimeError(
            f"{module_name}: expected paired AutoMask API or build_automask_objective/automask_forward_loss"
        )

    def branch_record(record: Mapping[str, Any], state_id: str) -> Dict[str, Any]:
        branch = next(
            (item for item in record["paired_branches"] if item.get("branch_id") == state_id),
            None,
        )
        if not isinstance(branch, Mapping):
            raise RuntimeError(f"{record.get('pair_id')}: missing branch {state_id}")
        target_key = "source" if state_id == "source_state" else "target"
        target = record[target_key]
        mask = record.get("automatic_difference_mask")
        if not isinstance(mask, Mapping):
            raise RuntimeError(f"{record.get('pair_id')}: automatic_difference_mask is missing")
        return {
            "pair_id": record["pair_id"],
            "branch_id": state_id,
            "target": {
                "state": target_key,
                "image": branch["target_image"],
                "image_sha256": branch["target_image_sha256"],
            },
            "auto_candidate_mask": dict(mask),
        }

    def build_legacy_paired(record: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Mapping[str, torch.Tensor]]]:
        results = {state: branch_build(branch_record(record, state)) for state in STATE_IDS}
        return (
            {
                "objective_id": objective_id,
                "pair_id": record["pair_id"],
                "api_adapter": "branch_api_to_paired_api",
                "branches": {state: spec for state, (spec, _) in results.items()},
            },
            {state: weights for state, (_, weights) in results.items()},
        )

    def forward_legacy_paired(
        model: Any,
        data: Dict[str, Any],
        branch_id: str,
        branch_weights: Mapping[str, Mapping[str, torch.Tensor]],
        autocast_context: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if branch_id not in branch_weights:
            raise RuntimeError(f"missing AutoMask branch weights: {branch_id}")
        return branch_forward(model, data, branch_weights[branch_id], autocast_context)

    return objective_id, build_legacy_paired, forward_legacy_paired


def _schedule(seed: int, step: int) -> Dict[str, Any]:
    rng = np.random.default_rng(_hash_seed("gold40-automask-v5-timestep", seed, step))
    selector = float(rng.random())
    if selector < MIXTURE["native"]:
        raw, mode = float(rng.standard_normal()), "native"
        base = 1 / (1 + math.exp(-raw))
        effective = SHIFT * base / (1 + (SHIFT - 1) * base)
    elif selector < MIXTURE["native"] + MIXTURE["high"]:
        mode, effective = "high", float(rng.uniform(.85, .999))
        base = effective / (SHIFT - (SHIFT - 1) * effective)
        raw = math.log(base / (1 - base))
    else:
        mode, effective, raw = "pure_noise", 1.0, float("inf")
    return {
        "mode": mode,
        "effective_t": effective,
        "raw": raw,
        "pack_seed": _hash_seed("pack", seed, step),
        "noise_seed": _hash_seed("noise", seed, step),
    }


def _sample_pair(records: list[Dict[str, Any]], seed: int, step: int) -> Dict[str, Any]:
    """Stateless rank-identical sampling: exactly one paired record per macro step."""
    return records[_hash_seed("automask-v5-pair", seed, step) % len(records)]


def _resolve_resume(args: argparse.Namespace) -> Path | None:
    """Pick the newest usable checkpoint; returns None for a fresh run."""
    candidates: list[Path] = []
    if args.resume != "none":
        if args.resume != "auto":
            candidates.append(Path(args.resume))
        else:
            candidates.append(args.output_dir / "checkpoints" / "latest.pt")
            candidates.extend(sorted((args.output_dir / "checkpoints").glob("step_*.pt"), reverse=True))
    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            usable = payload.get("format") == CHECKPOINT_FORMAT and isinstance(payload.get("trainable_state_dict"), dict)
            step = payload.get("step") if usable else None
            del payload
            if usable:
                _log("resume_candidate", path=str(candidate), step=step)
                return candidate
        except Exception as exc:  # noqa: BLE001
            _log("checkpoint_rejected", path=str(candidate), error=repr(exc))
    if args.resume not in ("none", "auto"):
        raise RuntimeError(f"Invalid resume checkpoint: {args.resume}")
    return None


def _load_resume(
    path: Path,
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    trainable: Sequence[Tuple[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, list[Dict[str, Any]]]:
    """Rehydrate the replicated fp32 LoRA/bridge masters, the AdamW moments and the history."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError(f"Unsupported checkpoint format: {payload.get('format')}")
    saved_config = payload.get("config") or {}
    mismatches = {
        key: {"checkpoint": saved_config.get(key), "current": plan.get(key)}
        for key in RESUME_GUARD_KEYS
        if saved_config.get(key) != plan.get(key)
    }
    saved_args = (saved_config.get("args") or {}) if isinstance(saved_config.get("args"), dict) else {}
    for key in ("seed", "lr", "lora_r", "lora_alpha", "trainable_name_regex"):
        if saved_args.get(key) != getattr(args, key, None):
            mismatches[f"args.{key}"] = {"checkpoint": saved_args.get(key), "current": getattr(args, key, None)}
    if mismatches:
        raise RuntimeError(f"Resume config mismatch: {json.dumps(mismatches, default=str)}")
    named = dict(trainable)
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
    step = int(payload.get("step", 0))
    history = list(payload.get("history", []))
    del payload
    v3_runtime._assert_replicated_trainables([parameter for _, parameter in trainable], device, "resume_loaded")
    _log("resume_loaded", path=str(path), step=step, history_rows=len(history))
    return step, history


def _run_state(
    model: Any,
    vae: Any,
    packer: Any,
    dataset: Any,
    record: Dict[str, Any],
    schedule: Mapping[str, Any],
    device: torch.device,
    branch_weights: Mapping[str, Mapping[str, torch.Tensor]],
    forward_loss: Any,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    batch, trace = _pack_single_pair(packer, dataset, record, int(schedule["pack_seed"]))
    finite = torch.isfinite(batch.packed_timesteps)
    if int(finite.sum()) != EXPECTED_TARGET_TIMESTEP_TOKENS:
        raise RuntimeError(f"{record['pair_id']}: expected {EXPECTED_TARGET_TIMESTEP_TOKENS} target timestep tokens")
    batch.packed_timesteps = torch.where(
        finite,
        torch.full_like(batch.packed_timesteps, float(schedule["raw"])),
        batch.packed_timesteps,
    )
    data = _prepare_batch(batch, vae, device, torch.bfloat16)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(schedule["noise_seed"]))
    noise = torch.randn(EXPECTED_NOISE_SHAPE, device=device, dtype=data["padded_latent"].dtype, generator=generator)
    calls = 0

    def controlled_randn_like(value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal calls
        if tuple(value.shape) != EXPECTED_NOISE_SHAPE:
            raise RuntimeError(f"unexpected BAGEL noise shape {tuple(value.shape)}")
        calls += 1
        return noise.clone()

    cuda_index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[cuda_index], enabled=True), mock.patch.object(torch, "randn_like", side_effect=controlled_randn_like):
        torch.manual_seed(int(schedule["noise_seed"]))
        torch.cuda.manual_seed(int(schedule["noise_seed"]))
        loss, metrics = forward_loss(
            model,
            data,
            dataset.state_id,
            branch_weights,
            v3_runtime._autocast,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{record['pair_id']} {dataset.state_id}: non-finite loss")
        (loss * 0.5).backward()
    if calls != 1:
        raise RuntimeError(f"{record['pair_id']} {dataset.state_id}: expected one controlled noise draw, got {calls}")
    trace.update(
        state=dataset.state_id,
        pack_seed=int(schedule["pack_seed"]),
        noise_seed=int(schedule["noise_seed"]),
        noise_sha256=hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest(),
        raw_timestep=schedule["raw"],
        effective_t=schedule["effective_t"],
    )
    return metrics, trace


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="AutoMask-v5 ready paired Gold40 JSONL")
    parser.add_argument("--objective-module", default=DEFAULT_OBJECTIVE_MODULE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vae-path", type=Path, default=DEFAULT_MODEL / "ae.safetensors")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_MODEL / "ema.safetensors")
    parser.add_argument("--steps", type=int, default=960)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--trainable-name-regex", default=DEFAULT_TRAINABLE_REGEX)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--precision", default="bf16", choices=("bf16",))
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--vit-image-size", type=int, default=980)
    parser.add_argument("--vit-min-image-size", type=int, default=980)
    parser.add_argument("--expected-num-tokens", type=int, default=16384)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=16384)
    parser.add_argument("--max-num-tokens", type=int, default=16384)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--variant",
        default="paired_cf_mvp",
        choices=("paired_cf_mvp",),
        help="Internal BAGEL dataset variant; locked independently of the AutoMask-v5 experiment name.",
    )
    parser.add_argument("--run-name", default="automask_v5", help="Human-readable experiment label recorded in artifacts.")
    parser.add_argument("--min-gpu-memory-gib", type=float, default=38.0)
    parser.add_argument("--max-peak-reserved-gib", type=float, default=38.5)
    parser.add_argument(
        "--resume",
        default="auto",
        help="Resume source: 'auto' reuses the newest usable checkpoint under --output-dir, "
        "'none' forces a fresh run, or an explicit checkpoint path.",
    )
    args = parser.parse_args()
    locks = {
        "image_size": 1024,
        "vit_image_size": 980,
        "vit_min_image_size": 980,
        "expected_num_tokens": 16384,
        "max_num_tokens_per_sample": 16384,
        "max_num_tokens": 16384,
        "num_samples": 1,
        "lora_dropout": 0.0,
    }
    bad = {key: getattr(args, key) for key, expected in locks.items() if getattr(args, key) != expected}
    if bad or args.steps < 1 or args.checkpoint_every < 1:
        parser.error(f"locked Gold40 AutoMask-v5 configuration invalid: {bad}")
    if not args.preflight_only and not args.activation_checkpointing:
        parser.error("--activation-checkpointing is mandatory for 1024 FSDP")
    return args


def _preflight(args: argparse.Namespace) -> tuple[list[Dict[str, Any]], Dict[str, Any], Any, Any]:
    _validate_model_files(args)
    objective_id, build_objective, forward_loss = _load_objective(args.objective_module)
    records, audits = load_automask_manifest(args.manifest, require_files=True)
    objective_specs = {record["pair_id"]: build_objective(record)[0] for record in records}
    plan = {
        "experiment_name": EXPERIMENT_NAME,
        "manifest": str(args.manifest.resolve()),
        "record_count": len(records),
        "pair_ids": [item["pair_id"] for item in audits],
        "source_scene_ids": [item["source_scene_id"] for item in audits],
        "automask_only": True,
        "human_review_required": False,
        "paired_structure": list(STATE_IDS),
        "objective_id": objective_id,
        "objective_module": args.objective_module,
        "objective": objective_specs,
        "prompt_sha256": prompt_audit(records),
        "timestep_mixture": MIXTURE,
        "matched_pack_seed_raw_timestep_noise": True,
        "per_branch_loss_scale": 0.5,
        "optimizer_steps_per_macro_step": 1,
        "fsdp": {
            "world_size": 2,
            "strategy": "FULL_SHARD",
            "fp32_lora_and_bridge_masters": True,
            "activation_checkpointing_layers": EXPECTED_ACTIVATION_CHECKPOINTS,
        },
        "args": _serializable_args(args),
    }
    _atomic_json(args.output_dir / "preflight.json", plan)
    return records, plan, build_objective, forward_loss


def main() -> None:
    args = _parse_args()
    if args.preflight_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        records, _, _, _ = _preflight(args)
        _log("preflight_passed", records=len(records), manifest=str(args.manifest))
        return
    _, _, device = v3_runtime._validate_topology(args)
    try:
        if _rank0():
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "checkpoints").mkdir(exist_ok=True)
        _barrier()
        records, plan, build_objective, forward_loss = _preflight(args)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        tokenizer, special_tokens, added = _load_tokenizer(str(args.llm_path))
        packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)
        datasets = {state: build_state_dataset(args, tokenizer, state) for state in STATE_IDS}
        layers = _read_num_layers(args.llm_path)
        if layers != EXPECTED_ACTIVATION_CHECKPOINTS:
            raise RuntimeError(f"AutoMask-v5 requires 28 decoder layers, found {layers}")
        model, load_report = v3_runtime._build_cpu_model(args, len(tokenizer), added)
        model.timestep_shift = SHIFT
        model.config.timestep_shift = SHIFT
        target_names = _iter_target_linear_names(model, list(range(layers)))
        wrappers = inject_lora(model, target_names, args.lora_r, args.lora_alpha, 0.0)
        groups = set_trainable(model, args.trainable_name_regex, wrappers)
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("no LoRA/bridge trainables selected")
        for _, parameter in trainable:
            parameter.data = parameter.data.to(device=device, dtype=torch.float32)
        if any(parameter.dtype != torch.float32 for _, parameter in trainable):
            raise RuntimeError("FP32 LoRA/bridge master trainables required")
        hooks, bridge_report = v4_runtime._install_bridge_output_cast_hooks(model)
        fsdp = v3_runtime._fsdp_model(model, [parameter for _, parameter in trainable], device)
        activated = v3_runtime._apply_activation_checkpointing(fsdp, layers)
        if activated != EXPECTED_ACTIVATION_CHECKPOINTS:
            raise RuntimeError(f"expected {EXPECTED_ACTIVATION_CHECKPOINTS} activation-checkpointed layers, got {activated}")
        vae, _ = load_ae(str(args.vae_path))
        vae.to(device=device, dtype=torch.bfloat16).eval()
        optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=args.lr, weight_decay=args.weight_decay)
        audit = {
            "num_layers": layers,
            "lora_wrappers": len(wrappers),
            "lora_targets": target_names,
            "trainable_dtype": "float32",
            "fp32_master_scope": "LoRA and bridge trainables",
            "activation_checkpointed_layers": activated,
            "fsdp": "FULL_SHARD",
            "bridge_output_cast": bridge_report,
            **groups,
        }
        _atomic_json(args.output_dir / "trainable_audit.json", audit)
        _log("model_ready", records=len(records), trainables=len(trainable), lora_wrappers=len(wrappers))
        resume_path = _resolve_resume(args)
        if resume_path is None:
            start_step, history = 0, []
        else:
            start_step, history = _load_resume(resume_path, args, plan, trainable, optimizer, device)
            if start_step >= args.steps:
                _log("resume_already_complete", step=start_step, requested_steps=args.steps)
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(start_step + 1, args.steps + 1):
            record, schedule = _sample_pair(records, args.seed, step), _schedule(args.seed, step)
            _, branch_weights = build_objective(record)
            optimizer.zero_grad(set_to_none=True)
            result = {}
            for state in STATE_IDS:
                result[state] = _run_state(
                    fsdp,
                    vae,
                    packer,
                    datasets[state],
                    record,
                    schedule,
                    device,
                    branch_weights,
                    forward_loss,
                )
            source_trace, target_trace = result["source_state"][1], result["target_state"][1]
            for field in ("pack_seed", "noise_seed", "noise_sha256", "raw_timestep", "effective_t"):
                if source_trace[field] != target_trace[field]:
                    raise RuntimeError(f"state branches diverged on controlled {field}")
            missing = [name for name, parameter in trainable if parameter.grad is None]
            if missing:
                raise RuntimeError(f"missing gradients: {missing[:10]}")
            v3_runtime._sync_ignored_trainable_grads([parameter for _, parameter in trainable])
            norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], args.max_grad_norm)
            optimizer.step()
            checksum = v3_runtime._assert_replicated_trainables([parameter for _, parameter in trainable], device, f"step_{step}")
            peak_reserved_gib = torch.cuda.max_memory_reserved(device) / 2**30
            if dist.is_initialized():
                peak_tensor = torch.tensor([peak_reserved_gib], device=device)
                dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
                peak_reserved_gib = float(peak_tensor.item())
            row = {
                "gpu_peak_reserved_gib": round(peak_reserved_gib, 3),
                "step": step,
                "step": step,
                "pair_id": record["pair_id"],
                "source_scene_id": record["source_scene_id"],
                "intervention_type": record["intervention"]["intervention_type"],
                "macro_steps": 1,
                "micro_steps": 2,
                "optimizer_steps": 1,
                "branch_loss_scale": 0.5,
                "timestep_mode": schedule["mode"],
                "effective_t": schedule["effective_t"],
                "raw_timestep": schedule["raw"],
                "matched_pack_seed": schedule["pack_seed"],
                "matched_noise_sha256": source_trace["noise_sha256"],
                "grad_norm_pre_clip": float(norm.detach().float().cpu()),
                "trainable_checksum": checksum,
                **{
                    f"{state}_{key}": value
                    for state, (metrics, _) in result.items()
                    for key, value in metrics.items()
                },
            }
            history.append(row)
            _log("macro_step", **row)
            torch.cuda.reset_peak_memory_stats(device)
            if step % args.checkpoint_every == 0 or step == args.steps:
                payload = {
                    "format": CHECKPOINT_FORMAT,
                    "step": step,
                    "config": plan,
                    "trainable_state_dict": {name: parameter.detach().cpu() for name, parameter in trainable},
                    "optimizer_state_dict": v3_runtime._cpu_tree(optimizer.state_dict()),
                    "history": history,
                }
                if _rank0():
                    v3_runtime._atomic_save(args.output_dir / "checkpoints" / "latest.pt", payload)
                _barrier()
        _atomic_json(
            args.output_dir / "final_report.json",
            {
                "status": "completed",
                "base_load_report": load_report,
                "history": history,
                "fsdp": {"world_size": 2, "strategy": "FULL_SHARD"},
                "resume": {"from_step": start_step, "checkpoint": str(resume_path) if resume_path else None},
            },
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
