#!/usr/bin/env python3
"""Two-rank FSDP FULL_SHARD Gold40 source_state/target_state macro-step training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.full_backbone_lora_dual_target_cot1024_fsdp_v4 import train as v4
from experiments.counterfactual_cot.full_backbone_lora_region1024_fsdp_v3 import train as v3
from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.train import DEFAULT_TRAINABLE_REGEX, _read_num_layers, _validate_model_files
from experiments.counterfactual_cot.training.fixed_deterministic_eval import _build_singleton_packed_dataset, _pack_single_pair
from experiments.counterfactual_cot.training.gold40_overfit import _load_tokenizer, _prepare_batch, _serializable_args
from experiments.counterfactual_cot.training.lora_2step_smoke import TARGET_MODULE_SUFFIXES, _iter_target_linear_names, inject_lora, set_trainable
from experiments.counterfactual_cot.gold40_dual_target_cot_v4_data.paired_region_objective import (
    OBJECTIVE_ID,
    build_paired_region_objective,
    paired_region_forward_loss,
)
from experiments.counterfactual_cot.gold40_dual_target_cot1024_fsdp_v4.paired_data import (
    STATE_IDS,
    build_state_dataset,
    load_reviewed_manifest,
    objective_record,
    prompt_audit,
)
from modeling.autoencoder import load_ae

EXPERIMENT_NAME = "gold40_source_target_state_dual_target_cot1024_fsdp_full_shard_v4"
DEFAULT_MODEL = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_OUTPUT = BAGEL_ROOT / "experiments/counterfactual_cot/outputs" / EXPERIMENT_NAME
MIXTURE = {"native": .50, "high": .40, "pure_noise": .10}
SHIFT = 3.0


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


def _schedule(seed: int, step: int) -> Dict[str, Any]:
    rng = np.random.default_rng(_hash_seed("gold40-v4-timestep", seed, step))
    selector = float(rng.random())
    if selector < MIXTURE["native"]:
        raw, mode = float(rng.standard_normal()), "native"
        base = 1 / (1 + math.exp(-raw)); effective = SHIFT * base / (1 + (SHIFT - 1) * base)
    elif selector < MIXTURE["native"] + MIXTURE["high"]:
        mode, effective = "high", float(rng.uniform(.85, .999))
        base = effective / (SHIFT - (SHIFT - 1) * effective); raw = math.log(base / (1 - base))
    else:
        mode, effective, raw = "pure_noise", 1.0, float("inf")
    return {"mode": mode, "effective_t": effective, "raw": raw, "pack_seed": _hash_seed("pack", seed, step), "noise_seed": _hash_seed("noise", seed, step)}


def _sample_pair(records: list[Dict[str, Any]], seed: int, step: int) -> Dict[str, Any]:
    # Identical stateless draw on both ranks: every macro-step trains both states of one pair.
    return records[_hash_seed("pair", seed, step) % len(records)]


def _run_state(
    model: Any,
    vae: Any,
    packer: Any,
    dataset: Any,
    record: Dict[str, Any],
    schedule: Mapping[str, Any],
    device: torch.device,
    branch_weights: Mapping[str, Mapping[str, torch.Tensor]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    batch, trace = _pack_single_pair(packer, dataset, record, int(schedule["pack_seed"]))
    finite = torch.isfinite(batch.packed_timesteps)
    if int(finite.sum()) != 4096:
        raise RuntimeError(f"{record['pair_id']}: expected 4096 target timestep tokens")
    batch.packed_timesteps = torch.where(finite, torch.full_like(batch.packed_timesteps, float(schedule["raw"])), batch.packed_timesteps)
    data = _prepare_batch(batch, vae, device, torch.bfloat16)
    expected_shape = (8192, 64)
    generator = torch.Generator(device=device); generator.manual_seed(int(schedule["noise_seed"]))
    noise = torch.randn(expected_shape, device=device, dtype=data["padded_latent"].dtype, generator=generator)
    calls = 0
    def controlled_randn_like(value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal calls
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(f"unexpected BAGEL noise shape {tuple(value.shape)}")
        calls += 1
        return noise.clone()
    with torch.random.fork_rng(devices=[device.index], enabled=True), mock.patch.object(torch, "randn_like", side_effect=controlled_randn_like):
        torch.manual_seed(int(schedule["noise_seed"])); torch.cuda.manual_seed(int(schedule["noise_seed"]))
        loss, metrics = paired_region_forward_loss(
            model,
            data,
            dataset.state_id,
            branch_weights,
            v3._autocast,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite state loss")
        (loss * .5).backward()
    if calls != 1:
        raise RuntimeError(f"expected one controlled noise draw, found {calls}")
    trace.update(state=dataset.state_id, noise_sha256=hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest(), effective_t=schedule["effective_t"])
    return metrics, trace


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--manifest", type=Path, required=True, help="Reviewed approved train6/train24 Gold40 JSONL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_MODEL); parser.add_argument("--vae-path", type=Path, default=DEFAULT_MODEL / "ae.safetensors"); parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_MODEL / "ema.safetensors")
    parser.add_argument("--steps", type=int, default=500); parser.add_argument("--lr", type=float, default=3e-5); parser.add_argument("--weight-decay", type=float, default=0.0); parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8); parser.add_argument("--lora-alpha", type=int, default=16); parser.add_argument("--lora-dropout", type=float, default=0.0); parser.add_argument("--trainable-name-regex", default=DEFAULT_TRAINABLE_REGEX)
    parser.add_argument("--checkpoint-every", type=int, default=10); parser.add_argument("--seed", type=int, default=1234); parser.add_argument("--activation-checkpointing", action="store_true"); parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--precision", default="bf16", choices=("bf16",)); parser.add_argument("--image-size", type=int, default=1024); parser.add_argument("--vit-image-size", type=int, default=980); parser.add_argument("--vit-min-image-size", type=int, default=980); parser.add_argument("--expected-num-tokens", type=int, default=16384); parser.add_argument("--max-num-tokens-per-sample", type=int, default=16384); parser.add_argument("--max-num-tokens", type=int, default=16384); parser.add_argument("--num-samples", type=int, default=1); parser.add_argument("--variant", default="paired_cf_mvp"); parser.add_argument("--min-gpu-memory-gib", type=float, default=38.0); parser.add_argument("--max-peak-reserved-gib", type=float, default=38.5)
    args = parser.parse_args()
    locks = {"image_size": 1024, "vit_image_size": 980, "vit_min_image_size": 980, "expected_num_tokens": 16384, "max_num_tokens_per_sample": 16384, "max_num_tokens": 16384, "num_samples": 1, "lora_dropout": 0.0}
    bad = {key: getattr(args, key) for key, expected in locks.items() if getattr(args, key) != expected}
    if bad or args.steps < 1 or args.checkpoint_every < 1:
        parser.error(f"locked Gold40 configuration invalid: {bad}")
    if not args.preflight_only and not args.activation_checkpointing:
        parser.error("--activation-checkpointing is mandatory for 1024 FSDP")
    return args


def _preflight(args: argparse.Namespace) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    _validate_model_files(args)
    records, audits = load_reviewed_manifest(args.manifest, require_files=True)
    # Prove every reviewed record can create dynamic token partitions before allocating a model.
    objective = {
        record["pair_id"]: build_paired_region_objective(objective_record(record))[0]
        for record in records
    }
    plan = {"experiment_name": EXPERIMENT_NAME, "manifest": str(args.manifest), "record_count": len(records), "pair_ids": [item["pair_id"] for item in audits], "reviewed_approved_only": True, "paired_structure": list(STATE_IDS), "source_scene_ids": [item["source_scene_id"] for item in audits], "objective_id": OBJECTIVE_ID, "objective": objective, "prompt_sha256": prompt_audit(records), "timestep_mixture": MIXTURE, "matched_noise_and_timestep": True, "fsdp": {"world_size": 2, "strategy": "FULL_SHARD", "fp32_masters": True, "activation_checkpointing": True}, "args": _serializable_args(args)}
    _atomic_json(args.output_dir / "preflight.json", plan)
    return records, plan


def main() -> None:
    args = _parse_args()
    if args.preflight_only:
        args.output_dir.mkdir(parents=True, exist_ok=True); records, _ = _preflight(args); _log("preflight_passed", records=len(records), manifest=str(args.manifest)); return
    _, _, device = v3._validate_topology(args)
    try:
        if _rank0():
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "checkpoints").mkdir(exist_ok=True)
        _barrier(); records, plan = _preflight(args)
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        tokenizer, special_tokens, added = _load_tokenizer(str(args.llm_path)); packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)
        datasets = {state: build_state_dataset(args, tokenizer, state) for state in STATE_IDS}
        layers = _read_num_layers(args.llm_path); model, load_report = v3._build_cpu_model(args, len(tokenizer), added)
        model.timestep_shift = SHIFT; model.config.timestep_shift = SHIFT
        target_names = _iter_target_linear_names(model, list(range(layers))); wrappers = inject_lora(model, target_names, args.lora_r, args.lora_alpha, 0.0); groups = set_trainable(model, args.trainable_name_regex, wrappers)
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        for _, parameter in trainable: parameter.data = parameter.data.to(device=device, dtype=torch.float32)
        if any(parameter.dtype != torch.float32 for _, parameter in trainable): raise RuntimeError("FP32 master trainables required")
        hooks, bridge_report = v4._install_bridge_output_cast_hooks(model)
        fsdp = v3._fsdp_model(model, [parameter for _, parameter in trainable], device); activated = v3._apply_activation_checkpointing(fsdp, layers)
        if activated != layers: raise RuntimeError("all decoder layers must be activation checkpointed")
        vae, _ = load_ae(str(args.vae_path)); vae.to(device=device, dtype=torch.bfloat16).eval()
        optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=args.lr, weight_decay=args.weight_decay)
        audit = {"num_layers": layers, "lora_wrappers": len(wrappers), "lora_targets": target_names, "trainable_dtype": "float32", "activation_checkpointed_layers": activated, "fsdp": "FULL_SHARD", "bridge_output_cast": bridge_report, **groups}
        _atomic_json(args.output_dir / "trainable_audit.json", audit); _log("model_ready", records=len(records), trainables=len(trainable), lora_wrappers=len(wrappers))
        history = []
        for step in range(1, args.steps + 1):
            record, schedule = _sample_pair(records, args.seed, step), _schedule(args.seed, step)
            _, branch_weights = build_paired_region_objective(objective_record(record))
            optimizer.zero_grad(set_to_none=True)
            result = {
                state: _run_state(
                    fsdp,
                    vae,
                    packer,
                    datasets[state],
                    record,
                    schedule,
                    device,
                    branch_weights,
                )
                for state in STATE_IDS
            }
            if result["source_state"][1]["noise_sha256"] != result["target_state"][1]["noise_sha256"]: raise RuntimeError("state branches did not use identical noise")
            missing = [name for name, parameter in trainable if parameter.grad is None]
            if missing: raise RuntimeError(f"missing gradients: {missing[:10]}")
            v3._sync_ignored_trainable_grads([parameter for _, parameter in trainable]); norm = torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], args.max_grad_norm); optimizer.step(); checksum = v3._assert_replicated_trainables([parameter for _, parameter in trainable], device, f"step_{step}")
            row = {"step": step, "pair_id": record["pair_id"], "source_scene_id": record["source_scene_id"], "intervention_type": record["intervention"]["intervention_type"], "macro_steps": 1, "micro_steps": 2, "timestep_mode": schedule["mode"], "effective_t": schedule["effective_t"], "matched_noise_sha256": result["source_state"][1]["noise_sha256"], "grad_norm_pre_clip": float(norm.detach().float().cpu()), "trainable_checksum": checksum, **{f"{state}_{key}": value for state, (metrics, _) in result.items() for key, value in metrics.items()}}
            history.append(row); _log("macro_step", **row)
            if step % args.checkpoint_every == 0 or step == args.steps:
                payload = {"format": "gold40_dual_target_fsdp_v4", "step": step, "config": plan, "trainable_state_dict": {name: parameter.detach().cpu() for name, parameter in trainable}, "optimizer_state_dict": v3._cpu_tree(optimizer.state_dict()), "history": history}
                if _rank0(): v3._atomic_save(args.output_dir / "checkpoints" / "latest.pt", payload)
                _barrier()
        _atomic_json(args.output_dir / "final_report.json", {"status": "completed", "base_load_report": load_report, "history": history, "fsdp": {"world_size": 2, "strategy": "FULL_SHARD"}})
    finally:
        if dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
