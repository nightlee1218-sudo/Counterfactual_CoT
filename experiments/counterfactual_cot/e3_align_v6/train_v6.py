#!/usr/bin/env python3
"""Gold40 AutoMask-v6: v5 paired source/target-state FSDP SFT + token-region alignment.

Extends the v5 trainer (gold40_automask_dual_target_cot1024_fsdp_v5) with:

1. Alignment auxiliary loss (E3): on the E2-winning layers (13/6/16/14),
   ``L_total = L_SFT + lambda * L_align`` with
   ``L_align = -log sigmoid(s * (cos(h_tok, mu_in) - cos(h_tok, mu_out)))``.
   h_tok are the hidden states of the changed CoT value tokens in the packed
   state prompt; mu_in/mu_out are the layer's mass-weighted means over the
   target-latent positions inside/outside the v5 automatic edit partition
   (shared 64x64 row-major weights).  Extraction uses plain forward hooks on
   the align layers (references captured before FSDP/AC wrapping); the v3
   runtime's NON-REENTRANT activation checkpointing keeps the hook tensors
   autograd-connected via backward recompute (verified in the CPU selftest).
   lambda = 0.1 by default; ``--align-layers`` selects any subset (single-layer
   ablation); ``--align-margin-scale`` (default 1.0) is an inverse temperature;
   ``--align-detach-mu`` freezes the latent-side means (off by default: both
   sides train, matching the E2 criterion direction).

2. Anti-over-edit switches (``--overshoot-fix``, default on):
   ``--edit-loss-weight`` (default 0.4, vs v5's hardcoded 0.5) rebalances the
   region loss toward outside preservation, and ``--mask-debias`` (default on)
   computes region losses as unweighted means over hard region membership
   (w_edit >= 0.5) instead of continuous mask-mass means.  Switching the master
   switch off reproduces v5's objective exactly.

3. Optional general-T2I replay (default off): ``--general-replay-manifest``
   (JSONL: {"pair_id", "source": {"prompt", "cot_text", "image"}}) adds one
   plain-MSE standard_cot micro-step every ``--general-replay-every`` macro
   steps (default 4), branched at the same 0.5 loss scale.

4. Optional warm start: ``--init-trainable-from`` loads a v5/v6
   ``trainable_state_dict`` (e.g. final991 latest.pt) into the fp32 LoRA
   masters before FSDP wrapping; default off trains from the base weights.

Everything else (model build, full-layer LoRA r8/a16 fp32 masters, FSDP
FULL_SHARD world_size=2, mandatory 28-layer activation checkpointing,
matched-noise dual states at 0.5 branch scale, 16384-token packing, timestep
mixture/shift, deterministic per-step seeding) is reused verbatim from v5.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from unittest import mock

import numpy as np
import torch
import torch.distributed as dist

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.full_backbone_lora_region1024_fsdp_v3 import train as v3_runtime
from experiments.counterfactual_cot.gold40_automask_dual_target_cot1024_fsdp_v5 import train as v5_runtime
from experiments.counterfactual_cot.gold40_automask_dual_target_cot1024_fsdp_v5.paired_data import (
    STATE_IDS,
    build_state_dataset,
    load_automask_manifest,
    prompt_audit,
    render_state_prompt,
)
from experiments.counterfactual_cot.datasets.bagel_manifest_dataset import CounterfactualCoTDataset
from experiments.counterfactual_cot.e3_align_v6.align_loss import (
    ALIGN_DEFAULT_LAYERS,
    ALIGN_MIN_REGION_MASS,
    AlignHookManager,
    AlignTargetCache,
    align_loss_from_stash,
    build_region_weight_vectors,
    build_token_index,
    derive_prompt_layout,
    empty_align_metrics,
    state_changed_value,
    target_cot_occurrences,
    validate_replay_record,
)

EXPERIMENT_NAME = "gold40_automask_v6_align_fsdp_full_shard"
DEFAULT_MODEL = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_OUTPUT = BAGEL_ROOT / "experiments/counterfactual_cot/outputs" / EXPERIMENT_NAME
DEFAULT_MANIFEST = BAGEL_ROOT / "experiments/counterfactual_cot/final991_automask_v5_data/artifacts/splits/train.jsonl"
DEFAULT_OBJECTIVE_MODULE = "experiments.counterfactual_cot.e3_align_v6.align_objective"
CHECKPOINT_FORMAT = "gold40_automask_v6_align_fsdp"
REPLAY_NOISE_SHAPE = (4096, 64)
RESUME_GUARD_KEYS = (
    "manifest",
    "objective_id",
    "objective_module",
    "record_count",
    "align_lambda",
    "align_layers",
    "align_margin_scale",
    "align_detach_mu",
    "overshoot_edit_weight",
    "overshoot_mask_debias",
    "general_replay_manifest",
    "general_replay_every",
    "init_trainable_from",
)


def _parse_align_layers(value: str) -> List[int]:
    parts = [chunk.strip() for chunk in str(value).split(",") if chunk.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("align layers must be a comma-separated list like 13,6,16,14")
    try:
        layers = [int(chunk) for chunk in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"align layers must be integers: {value!r}") from exc
    if any(layer < 0 or layer > 27 for layer in layers):
        raise argparse.ArgumentTypeError("align layers must be decoder layer indices in [0, 27]")
    if len(set(layers)) != len(layers):
        raise argparse.ArgumentTypeError(f"align layers must be unique: {value!r}")
    return layers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="AutoMask-v5 ready paired JSONL (final991 train split by default)")
    parser.add_argument("--objective-module", default=DEFAULT_OBJECTIVE_MODULE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vae-path", type=Path, default=DEFAULT_MODEL / "ae.safetensors")
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_MODEL / "ema.safetensors")
    parser.add_argument("--steps", type=int, default=2973, help="optimizer steps; 2973 mirrors the final991 v5 run")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--trainable-name-regex", default=v5_runtime.DEFAULT_TRAINABLE_REGEX)
    parser.add_argument("--checkpoint-every", type=int, default=200)
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
    parser.add_argument("--run-name", default="automask_v6_align", help="Human-readable experiment label recorded in artifacts.")
    parser.add_argument("--min-gpu-memory-gib", type=float, default=38.0)
    parser.add_argument("--max-peak-reserved-gib", type=float, default=38.5)
    parser.add_argument(
        "--resume",
        default="auto",
        help="Resume source: 'auto' reuses the newest usable checkpoint under --output-dir, "
        "'none' forces a fresh run, or an explicit checkpoint path.",
    )
    # ---- E3 alignment ----
    parser.add_argument("--align-lambda", type=float, default=0.1, help="L_total = L_SFT + lambda * L_align")
    parser.add_argument(
        "--align-layers",
        type=_parse_align_layers,
        default=list(ALIGN_DEFAULT_LAYERS),
        help="decoder layers used for the alignment loss (default 13,6,16,14; pass one index for a single-layer ablation)",
    )
    parser.add_argument("--align-margin-scale", type=float, default=1.0, help="inverse temperature s in -log sigmoid(s*margin); must be > 0")
    parser.add_argument("--align-detach-mu", action="store_true", help="stop gradients on mu_in/mu_out (latent side frozen in the align term)")
    # ---- anti-over-edit ----
    parser.add_argument("--overshoot-fix", choices=("on", "off"), default="on", help="master switch; 'off' reproduces the v5 objective exactly")
    parser.add_argument("--edit-loss-weight", type=float, default=0.4, help="edit region weight when overshoot-fix=on (outside gets 1-w)")
    parser.add_argument("--mask-debias", choices=("on", "off"), default="on", help="hard-region unweighted means instead of continuous mask-mass means")
    # ---- general replay / warm start ----
    parser.add_argument("--general-replay-manifest", type=Path, default=None, help="optional general-T2I JSONL; default off")
    parser.add_argument("--general-replay-every", type=int, default=4, help="one replay micro-step every N macro steps")
    parser.add_argument("--init-trainable-from", type=Path, default=None, help="optional v5/v6 trainable_state_dict checkpoint for a warm start")
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
        parser.error(f"locked Gold40 AutoMask-v6 configuration invalid: {bad}")
    if not args.preflight_only and not args.activation_checkpointing:
        parser.error("--activation-checkpointing is mandatory for 1024 FSDP")
    if args.align_lambda < 0.0:
        parser.error("--align-lambda must be >= 0")
    if args.align_margin_scale <= 0.0:
        parser.error("--align-margin-scale must be > 0")
    if not (0.0 < args.edit_loss_weight < 1.0):
        parser.error("--edit-loss-weight must be in (0, 1)")
    if args.general_replay_every < 1:
        parser.error("--general-replay-every must be >= 1")

    args.effective_edit_weight = args.edit_loss_weight if args.overshoot_fix == "on" else 0.5
    args.effective_mask_debias = args.mask_debias == "on" and args.overshoot_fix == "on"
    args.align_layers_list = list(args.align_layers)
    return args


def _align_readiness_audit(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """String-level check that each state's changed value occurs in its Target CoT line."""
    ready = {state: 0 for state in STATE_IDS}
    missing: List[Dict[str, str]] = []
    for record in records:
        for state in STATE_IDS:
            value = state_changed_value(record, state)
            reason = "no_intervention_value" if not value else ""
            if not reason:
                prompt = render_state_prompt(record, state)
                occurrences, source = target_cot_occurrences(prompt, value)
                if occurrences:
                    ready[state] += 1
                    continue
                reason = f"span_not_found({source})"
            missing.append({"pair_id": record.get("pair_id", ""), "state": state, "reason": reason})
    return {
        "ready_counts": ready,
        "record_count": len(records),
        "missing_count": len(missing),
        "missing_sample": missing[:20],
    }


def _preflight(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Any, Any]:
    v5_runtime._validate_model_files(args)
    objective_id, build_objective, forward_loss = v5_runtime._load_objective(args.objective_module)
    objective_module = importlib.import_module(args.objective_module)
    if hasattr(objective_module, "configure_overshoot"):
        objective_module.configure_overshoot(
            edit_weight=args.effective_edit_weight,
            debias=args.effective_mask_debias,
        )
    records, audits = load_automask_manifest(args.manifest, require_files=True)
    objective_specs = {record["pair_id"]: build_objective(record)[0] for record in records}
    replay_plan: Dict[str, Any] = {"enabled": args.general_replay_manifest is not None, "manifest": None, "record_count": 0}
    if args.general_replay_manifest is not None:
        replay_records = [json.loads(line) for line in Path(args.general_replay_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        for record in replay_records:
            validate_replay_record(record)
        replay_plan.update(manifest=str(args.general_replay_manifest), record_count=len(replay_records), every=args.general_replay_every)
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
        "timestep_mixture": v5_runtime.MIXTURE,
        "matched_pack_seed_raw_timestep_noise": True,
        "per_branch_loss_scale": 0.5,
        "optimizer_steps_per_macro_step": 1,
        "fsdp": {
            "world_size": 2,
            "strategy": "FULL_SHARD",
            "fp32_lora_and_bridge_masters": True,
            "activation_checkpointing_layers": v5_runtime.EXPECTED_ACTIVATION_CHECKPOINTS,
        },
        "align": {
            "lambda": args.align_lambda,
            "layers": args.align_layers_list,
            "margin_scale": args.align_margin_scale,
            "detach_mu": bool(args.align_detach_mu),
            "formula": "-log sigmoid(s*(cos(h_tok, mu_in) - cos(h_tok, mu_out))), mean over tokens x layers, per branch scaled 0.5",
            "readiness": _align_readiness_audit(records),
        },
        "overshoot": {
            "master": args.overshoot_fix,
            "edit_weight": args.effective_edit_weight,
            "mask_debias": args.effective_mask_debias,
        },
        "replay": replay_plan,
        "init_trainable_from": str(args.init_trainable_from) if args.init_trainable_from else None,
        "args": v5_runtime._serializable_args(args),
    }
    v5_runtime._atomic_json(args.output_dir / "preflight.json", plan)
    return records, plan, build_objective, forward_loss


def _resolve_resume(args: argparse.Namespace) -> Optional[Path]:
    candidates: List[Path] = []
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
                v5_runtime._log("resume_candidate", path=str(candidate), step=step)
                return candidate
        except Exception as exc:  # noqa: BLE001
            v5_runtime._log("checkpoint_rejected", path=str(candidate), error=repr(exc))
    if args.resume not in ("none", "auto"):
        raise RuntimeError(f"Invalid resume checkpoint: {args.resume}")
    return None


def _load_resume(
    path: Path,
    args: argparse.Namespace,
    plan: Mapping[str, Any],
    trainable: Sequence[Tuple[str, torch.nn.Parameter]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[int, List[Dict[str, Any]]]:
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
    v5_runtime._log("resume_loaded", path=str(path), step=step, history_rows=len(history))
    return step, history


def _load_init_trainable(path: Path, trainable: Sequence[Tuple[str, torch.nn.Parameter]], device: torch.device) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("trainable_state_dict")
    if not isinstance(saved, dict) or not saved:
        raise RuntimeError(f"{path}: no trainable_state_dict to warm start from")
    named = dict(trainable)
    if set(saved) != set(named):
        only_saved = sorted(set(saved) - set(named))[:5]
        only_model = sorted(set(named) - set(saved))[:5]
        raise RuntimeError(f"init-trainable key mismatch: saved_only={only_saved} model_only={only_model}")
    for name, tensor in saved.items():
        if tuple(tensor.shape) != tuple(named[name].shape):
            raise RuntimeError(f"init-trainable shape mismatch for {name}")
        named[name].data.copy_(tensor.to(device=named[name].device, dtype=named[name].dtype))
    report = {
        "path": str(path),
        "format": payload.get("format"),
        "step": payload.get("step", payload.get("global_step")),
        "num_tensors": len(saved),
    }
    del payload
    v3_runtime._assert_replicated_trainables([parameter for _, parameter in trainable], device, "init_trainable_loaded")
    v5_runtime._log("init_trainable_loaded", **report)
    return report


def _prepare_align_state(
    args: argparse.Namespace,
    record: Mapping[str, Any],
    state_id: str,
    target_cache: AlignTargetCache,
    batch: Any,
    data: Dict[str, Any],
    branch_weights: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
    align_hooks: AlignHookManager,
) -> Dict[str, Any]:
    """Derive this forward's align targets and arm the hooks (or explain why not)."""
    if args.align_lambda <= 0.0:
        return {"enabled": False, "reason": "lambda_zero"}
    weights = branch_weights.get(state_id)
    if weights is None:
        return {"enabled": False, "reason": "no_branch_weights"}
    edit_mass = float(weights["edit"].sum())
    outside_mass = float(weights["outside"].sum())
    if edit_mass < ALIGN_MIN_REGION_MASS or outside_mass < ALIGN_MIN_REGION_MASS:
        return {"enabled": False, "reason": f"region_mass_in_out=({edit_mass:.2f},{outside_mass:.2f})"}
    value = state_changed_value(record, state_id)
    if not value:
        return {"enabled": False, "reason": "no_intervention_value"}
    prompt = render_state_prompt(record, state_id)
    entry = target_cache.entry(record["pair_id"], state_id, prompt, value)
    if not entry["spans"]:
        return {"enabled": False, "reason": "no_value_span", "span_source": entry["span_source"]}
    layout = derive_prompt_layout(
        batch.packed_text_ids,
        batch.packed_vit_token_indexes,
        batch.mse_loss_indexes,
        entry["text_ids"],
    )
    token_index = build_token_index(entry["spans"], layout["bos_pos"], device)
    mse_positions = data["mse_loss_indexes"].to(device=device, dtype=torch.long)
    w_in, w_out = build_region_weight_vectors(
        weights["edit"], weights["outside"], int(data["sequence_length"]), mse_positions, device, torch.bfloat16
    )
    align_hooks.configure_forward(token_index, w_in, w_out, detach_mu=bool(args.align_detach_mu))
    return {
        "enabled": True,
        "reason": "",
        "tokens": int(token_index.numel()),
        "spans": entry["spans"],
        "span_source": entry["span_source"],
    }


def _controlled_noise(shape: Tuple[int, int], noise_seed: int, device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(noise_seed))
    noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    state = {"calls": 0}

    def controlled_randn_like(value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if tuple(value.shape) != shape:
            raise RuntimeError(f"unexpected BAGEL noise shape {tuple(value.shape)} (expected {shape})")
        state["calls"] += 1
        return noise.clone()

    return noise, controlled_randn_like, state


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
    args: argparse.Namespace,
    align_hooks: AlignHookManager,
    target_cache: AlignTargetCache,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    state_id = dataset.state_id
    batch, trace = v5_runtime._pack_single_pair(packer, dataset, record, int(schedule["pack_seed"]))
    finite = torch.isfinite(batch.packed_timesteps)
    if int(finite.sum()) != v5_runtime.EXPECTED_TARGET_TIMESTEP_TOKENS:
        raise RuntimeError(f"{record['pair_id']}: expected {v5_runtime.EXPECTED_TARGET_TIMESTEP_TOKENS} target timestep tokens")
    batch.packed_timesteps = torch.where(
        finite,
        torch.full_like(batch.packed_timesteps, float(schedule["raw"])),
        batch.packed_timesteps,
    )
    data = v5_runtime._prepare_batch(batch, vae, device, torch.bfloat16)

    align_info = _prepare_align_state(args, record, state_id, target_cache, batch, data, branch_weights, device, align_hooks)
    noise, controlled_randn_like, noise_state = _controlled_noise(
        v5_runtime.EXPECTED_NOISE_SHAPE, int(schedule["noise_seed"]), device, data["padded_latent"].dtype
    )

    cuda_index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[cuda_index], enabled=True), mock.patch.object(
        torch, "randn_like", side_effect=controlled_randn_like
    ):
        torch.manual_seed(int(schedule["noise_seed"]))
        torch.cuda.manual_seed(int(schedule["noise_seed"]))
        sft_loss, metrics = forward_loss(model, data, state_id, branch_weights, v3_runtime._autocast)
        if align_hooks.configured:
            align_loss, align_metrics = align_loss_from_stash(align_hooks.stash, args.align_layers_list, args.align_margin_scale)
        else:
            align_loss, align_metrics = None, empty_align_metrics(align_info.get("reason", "not_configured"))
        total = sft_loss if align_loss is None else sft_loss + args.align_lambda * align_loss
        if not torch.isfinite(total):
            raise FloatingPointError(f"{record['pair_id']} {state_id}: non-finite loss")
        (total * 0.5).backward()
    if noise_state["calls"] != 1:
        raise RuntimeError(f"{record['pair_id']} {state_id}: expected one controlled noise draw, got {noise_state['calls']}")
    align_hooks.clear()

    merged = dict(metrics)
    merged.update(align_metrics)
    merged["align_lambda"] = float(args.align_lambda)
    if align_info.get("enabled"):
        merged["align_skipped"] = 0
        merged["align_span_source"] = align_info.get("span_source", "")
    trace.update(
        state=state_id,
        pack_seed=int(schedule["pack_seed"]),
        noise_seed=int(schedule["noise_seed"]),
        noise_sha256=hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest(),
        raw_timestep=schedule["raw"],
        effective_t=schedule["effective_t"],
        align_tokens=align_info.get("tokens", 0),
        align_spans=[list(span) for span in align_info.get("spans", [])],
        align_span_source=align_info.get("span_source", ""),
    )
    return merged, trace


def _run_replay_state(
    model: Any,
    vae: Any,
    packer: Any,
    dataset: Any,
    record: Dict[str, Any],
    schedule: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """One optional general-T2I micro-step: native MSE (+CE) only, no align, scale 0.5."""
    batch, _trace = v5_runtime._pack_single_pair(packer, dataset, record, int(schedule["pack_seed"]))
    finite = torch.isfinite(batch.packed_timesteps)
    if int(finite.sum()) != v5_runtime.EXPECTED_TARGET_TIMESTEP_TOKENS:
        raise RuntimeError(f"replay {record.get('pair_id')}: expected target timestep tokens")
    batch.packed_timesteps = torch.where(
        finite,
        torch.full_like(batch.packed_timesteps, float(schedule["raw"])),
        batch.packed_timesteps,
    )
    data = v5_runtime._prepare_batch(batch, vae, device, torch.bfloat16)
    noise, controlled_randn_like, noise_state = _controlled_noise(
        REPLAY_NOISE_SHAPE, int(schedule["noise_seed"]), device, data["padded_latent"].dtype
    )

    cuda_index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[cuda_index], enabled=True), mock.patch.object(
        torch, "randn_like", side_effect=controlled_randn_like
    ):
        torch.manual_seed(int(schedule["noise_seed"]))
        torch.cuda.manual_seed(int(schedule["noise_seed"]))
        with v3_runtime._autocast(model):
            output = model(**data)
        mse = output.get("mse")
        if mse is None:
            raise RuntimeError(f"replay {record.get('pair_id')}: model returned no mse")
        replay_loss = mse.float().mean()
        ce = output.get("ce")
        ce_loss = ce.float().mean() if ce is not None else None
        if ce_loss is not None:
            replay_loss = replay_loss + ce_loss
        if not torch.isfinite(replay_loss):
            raise FloatingPointError(f"replay {record.get('pair_id')}: non-finite loss")
        (replay_loss * 0.5).backward()
    if noise_state["calls"] != 1:
        raise RuntimeError(f"replay {record.get('pair_id')}: expected one controlled noise draw, got {noise_state['calls']}")
    return {
        "replay_total": float(replay_loss.detach().cpu()),
        "replay_mse": float(mse.float().mean().detach().cpu()),
        "replay_ce": float(ce_loss.detach().cpu()) if ce_loss is not None else 0.0,
    }


def _build_replay_dataset(args: argparse.Namespace, tokenizer: Any) -> "CounterfactualCoTDataset":
    from data.transforms import ImageTransform
    from experiments.counterfactual_cot.training.gold40_overfit import _transform_args

    class _ReplayStandardDataset(CounterfactualCoTDataset):
        def _paired_sample(self, pair: Dict[str, Any]):
            return self._standard_sample(pair, "source")

    return _ReplayStandardDataset(
        dataset_name="e3_general_replay",
        transform=ImageTransform(**_transform_args(1024, 1024, 16)),
        tokenizer=tokenizer,
        manifest_path=str(args.general_replay_manifest),
        variant="standard_cot",
        include_source_image=False,
        training_seed=args.seed,
        local_rank=0,
        world_size=1,
        num_workers=0,
        data_status=None,
        vit_transform=None,
        max_records=None,
    )


def main() -> None:
    args = _parse_args()
    if args.preflight_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        records, _, _, _ = _preflight(args)
        v5_runtime._log("preflight_passed", records=len(records), manifest=str(args.manifest))
        return
    _, _, device = v3_runtime._validate_topology(args)
    try:
        if v5_runtime._rank0():
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "checkpoints").mkdir(exist_ok=True)
        v5_runtime._barrier()
        records, plan, build_objective, forward_loss = _preflight(args)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        tokenizer, special_tokens, added = v5_runtime._load_tokenizer(str(args.llm_path))
        packer = v5_runtime._build_singleton_packed_dataset(args, tokenizer, special_tokens)
        datasets = {state: build_state_dataset(args, tokenizer, state) for state in STATE_IDS}
        replay_dataset = None
        if args.general_replay_manifest is not None:
            replay_dataset = _build_replay_dataset(args, tokenizer)
        layers = v5_runtime._read_num_layers(args.llm_path)
        if layers != v5_runtime.EXPECTED_ACTIVATION_CHECKPOINTS:
            raise RuntimeError(f"AutoMask-v6 requires 28 decoder layers, found {layers}")
        model, load_report = v3_runtime._build_cpu_model(args, len(tokenizer), added)
        model.timestep_shift = v5_runtime.SHIFT
        model.config.timestep_shift = v5_runtime.SHIFT
        target_names = v5_runtime._iter_target_linear_names(model, list(range(layers)))
        wrappers = v5_runtime.inject_lora(model, target_names, args.lora_r, args.lora_alpha, 0.0)
        groups = v5_runtime.set_trainable(model, args.trainable_name_regex, wrappers)
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("no LoRA/bridge trainables selected")
        for _, parameter in trainable:
            parameter.data = parameter.data.to(device=device, dtype=torch.float32)
        if any(parameter.dtype != torch.float32 for _, parameter in trainable):
            raise RuntimeError("FP32 LoRA/bridge master trainables required")
        init_report = None
        if args.init_trainable_from is not None:
            init_report = _load_init_trainable(args.init_trainable_from, trainable, device)
        # Align hooks are installed on module references captured BEFORE FSDP/AC
        # wrapping; the wrapped tree still forwards through these objects.
        if args.align_lambda > 0.0:
            align_layer_modules = [model.language_model.model.layers[index] for index in args.align_layers_list]
            align_hooks = AlignHookManager(align_layer_modules, args.align_layers_list)
            align_hooks.install()
        else:
            align_hooks = AlignHookManager([], [])
        target_cache = AlignTargetCache(tokenizer)
        hooks, bridge_report = v5_runtime._install_bridge_output_cast_hooks(model)
        fsdp = v3_runtime._fsdp_model(model, [parameter for _, parameter in trainable], device)
        activated = v3_runtime._apply_activation_checkpointing(fsdp, layers)
        if activated != v5_runtime.EXPECTED_ACTIVATION_CHECKPOINTS:
            raise RuntimeError(f"expected {v5_runtime.EXPECTED_ACTIVATION_CHECKPOINTS} activation-checkpointed layers, got {activated}")
        vae, _ = v5_runtime.load_ae(str(args.vae_path))
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
            "align_layers": args.align_layers_list,
            "init_trainable": init_report,
            **groups,
        }
        v5_runtime._atomic_json(args.output_dir / "trainable_audit.json", audit)
        v5_runtime._log(
            "model_ready",
            records=len(records),
            trainables=len(trainable),
            lora_wrappers=len(wrappers),
            align_layers=args.align_layers_list,
            align_lambda=args.align_lambda,
        )
        resume_path = _resolve_resume(args)
        if resume_path is None:
            start_step, history = 0, []
        else:
            start_step, history = _load_resume(resume_path, args, plan, trainable, optimizer, device)
            if start_step >= args.steps:
                v5_runtime._log("resume_already_complete", step=start_step, requested_steps=args.steps)
        torch.cuda.reset_peak_memory_stats(device)
        for step in range(start_step + 1, args.steps + 1):
            record, schedule = v5_runtime._sample_pair(records, args.seed, step), v5_runtime._schedule(args.seed, step)
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
                    args,
                    align_hooks,
                    target_cache,
                )
            source_trace, target_trace = result["source_state"][1], result["target_state"][1]
            for field in ("pack_seed", "noise_seed", "noise_sha256", "raw_timestep", "effective_t"):
                if source_trace[field] != target_trace[field]:
                    raise RuntimeError(f"state branches diverged on controlled {field}")
            row = {
                "gpu_peak_reserved_gib": None,
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
                "align_lambda": float(args.align_lambda),
                "replay_active": 0,
            }
            replay_metrics: Dict[str, float] = {}
            if replay_dataset is not None and step % args.general_replay_every == 0:
                replay_record = v5_runtime._sample_pair(replay_dataset.records, args.seed + 1_234_567, step)
                replay_schedule = v5_runtime._schedule(args.seed + 7_654_321, step)
                replay_metrics = _run_replay_state(fsdp, vae, packer, replay_dataset, replay_record, replay_schedule, device)
                replay_metrics["replay_active"] = 1
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
            if peak_reserved_gib > args.max_peak_reserved_gib:
                v5_runtime._log(
                    "peak_memory_over_budget",
                    step=step,
                    peak_gib=round(peak_reserved_gib, 3),
                    budget_gib=args.max_peak_reserved_gib,
                )
            row.update(
                {
                    "gpu_peak_reserved_gib": round(peak_reserved_gib, 3),
                    "grad_norm_pre_clip": float(norm.detach().float().cpu()),
                    "trainable_checksum": checksum,
                    **replay_metrics,
                    **{
                        f"{state}_{key}": value
                        for state, (metrics, _) in result.items()
                        for key, value in metrics.items()
                    },
                }
            )
            history.append(row)
            v5_runtime._log("macro_step", **row)
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
                if v5_runtime._rank0():
                    v5_runtime._atomic_save(args.output_dir / "checkpoints" / "latest.pt", payload)
                    if step == args.steps:
                        v5_runtime._atomic_save(args.output_dir / "checkpoints" / f"step_{step:06d}.pt", payload)
                v5_runtime._barrier()
        v5_runtime._atomic_json(
            args.output_dir / "final_report.json",
            {
                "status": "completed",
                "base_load_report": load_report,
                "init_trainable": init_report,
                "history": history,
                "fsdp": {"world_size": 2, "strategy": "FULL_SHARD"},
                "align": plan["align"],
                "overshoot": plan["overshoot"],
                "replay": plan["replay"],
                "resume": {"from_step": start_step, "checkpoint": str(resume_path) if resume_path else None},
            },
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
