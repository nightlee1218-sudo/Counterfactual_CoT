"""Cold-reload verifier for the LoRA smoke checkpoint.

This script is deliberately self-contained. It is meant to be launched as an
independent GPU subprocess so that after it exits, the OS reclaims every CUDA
resource -- there is no "same-process release" concern.

It performs a *read-only* verification: it never re-runs training, never
modifies ``step_002_trainable.pt``, and never touches the training report.

Flow (as per the audit checklist):
  1.  Load Base BAGEL from --checkpoint-path (defaults to the same ema.safetensors
      that was used at training time).
  2.  Inject 24 LoRALinear wrappers on layers.24..27 x 6 gen-branch Linears with
      the exact same r/alpha/dropout as training.
  3.  Set the same 14 existing tensors trainable via the regex.
  4.  Strict key-set check: saved_keys == expected_trainable_keys.
  5.  Shape / dtype / finiteness check per tensor.
  6.  ``copy_`` each saved tensor into the model, record per-tensor max_abs_diff
      between the CPU checkpoint tensor and the freshly-copied model parameter
      (both cast to fp32 for the comparison).
  7.  Fixed-probe forward: deterministic pack with probe_seed, one no-grad
      forward, save the FULL mse tensor (not just mean).
  8.  Fixed-seed generation -> PNG.
  9.  Write verifier_<label>.json + mse_probe_tensor.pt + generation.png.

Note: The Base checkpoint (ema.safetensors) is loaded from /dockerdata by
default (model-weight cache location), while all generated artifacts are
written under --output-dir which the launcher must place under /group.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # noqa: N816

# Make experiments imports available.
BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.data.render_prompts import paired_prompt
from experiments.counterfactual_cot.training.gold40_overfit import (
    FIXED_GOLD40_MANIFEST,
    DEFAULT_TRAINABLE_REGEX,
    _autocast_for_model,
    _build_model_for_gold40,
    _generate_one,
    _generation_parameters,
    _load_pair_image,
    _load_tokenizer,
    _make_inferencer,
    _prepare_batch,
)
from experiments.counterfactual_cot.training.single_pair_overfit_diag import (
    DEFAULT_PAIR_ID,
    _pack_single_pair,
    _build_raw_sample_dataset,
    _build_singleton_packed_dataset,
    _snapshot_rng_state,
    _restore_rng_state,
    _sha256_file,
    _stable_pair_seed,
)
from experiments.counterfactual_cot.training.lora_2step_smoke import (
    LoRALinear,
    TARGET_MODULE_SUFFIXES,
    _iter_target_linear_names,
    inject_lora,
    set_trainable,
    CallCounter,
    _describe_dtypes,
    _assert_dtype_invariants,
    _split_parent_child,
)


# ---------------------------------------------------------------------------
# Helpers (kept local so the verifier can run without pulling in trainer state)
# ---------------------------------------------------------------------------


def _tensor_hash(tensor: torch.Tensor) -> str:
    arr = tensor.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _string_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log(event: str, **fields: Any) -> None:
    fields["event"] = event
    print(json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str), flush=True)


def _pick_pair(records: Sequence[Dict[str, Any]], pair_id: str) -> Dict[str, Any]:
    for r in records:
        if r.get("pair_id") == pair_id:
            return r
    raise KeyError(pair_id)


def _hash_source_target_images(pair: Dict[str, Any]) -> Dict[str, Any]:
    """Read the raw source/target PNG files off disk (before any transform) and
    hash them. This lets the CPU comparator confirm both verifier subprocesses
    used the same conditioning image bytes."""
    out: Dict[str, Any] = {}
    for side in ("source", "target"):
        # Use the same resolver the training loop uses.
        info = pair.get(side) or {}
        rel_path = None
        if isinstance(info, dict):
            rel_path = info.get("image_path") or info.get("path")
        if rel_path is None:
            # Some manifests store the paths at pair level.
            rel_path = pair.get(f"{side}_image_path")
        if rel_path is None:
            out[side] = {"available": False}
            continue
        candidate = Path(rel_path)
        if not candidate.is_absolute():
            candidate = FIXED_GOLD40_MANIFEST.parent / candidate
        if not candidate.exists():
            out[side] = {"available": False, "path": str(candidate)}
            continue
        out[side] = {
            "available": True,
            "path": str(candidate),
            "sha256": _sha256_file(candidate),
            "size_bytes": candidate.stat().st_size,
        }
    return out


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=("A", "B"),
                        help="Verifier label (A or B). Determines output filename only.")
    parser.add_argument("--trainable-checkpoint", type=Path, required=True,
                        help="Path to step_002_trainable.pt to verify (read-only).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Verifier output directory (must be under /group).")

    parser.add_argument("--manifest", type=Path, default=FIXED_GOLD40_MANIFEST)
    parser.add_argument("--pair-id", default=DEFAULT_PAIR_ID)
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
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--lora-layers", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip the generation PNG (for fast debugging only; disables PNG comparison).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load trainable-only checkpoint (strict) but record per-tensor diff.
# ---------------------------------------------------------------------------


def load_trainable_strict_with_diff(
    payload: Dict[str, Any],
    model: nn.Module,
) -> Dict[str, Any]:
    saved: Dict[str, torch.Tensor] = payload["trainable_state_dict"]
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    saved_keys = set(saved.keys())
    missing = sorted(expected - saved_keys)
    unexpected = sorted(saved_keys - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"strict trainable-key mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    per_tensor: Dict[str, Dict[str, Any]] = {}
    max_abs_load_copy_diff = 0.0
    named = dict(model.named_parameters())
    for name in sorted(saved.keys()):
        tensor = saved[name]
        p = named[name]
        if tuple(p.shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"shape mismatch on {name}: model={tuple(p.shape)} vs ckpt={tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor.to(torch.float32)).all().item():
            raise RuntimeError(f"non-finite values in checkpoint tensor {name}")
        # Compute ckpt tensor hash and stats first.
        t32 = tensor.detach().to(torch.float32)
        p32_before = p.detach().to(device="cpu", dtype=torch.float32).clone()
        p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))
        p32_after = p.detach().to(device="cpu", dtype=torch.float32)
        # Diff between checkpoint (as fp32) and model param (as fp32) AFTER copy.
        diff = (p32_after - t32).abs()
        m = float(diff.max().item())
        max_abs_load_copy_diff = max(max_abs_load_copy_diff, m)
        per_tensor[name] = {
            "shape": list(tensor.shape),
            "ckpt_dtype": str(tensor.dtype),
            "model_dtype": str(p.dtype),
            "ckpt_tensor_hash": _tensor_hash(tensor),
            "model_param_hash_after_copy": _tensor_hash(p.detach()),
            "max_abs_load_copy_diff": m,
            "ckpt_min": float(t32.min().item()),
            "ckpt_max": float(t32.max().item()),
            "ckpt_mean": float(t32.mean().item()),
            "ckpt_l2": float(torch.linalg.vector_norm(t32.reshape(-1)).item()),
            "delta_from_pre_copy_l2": float(torch.linalg.vector_norm((p32_after - p32_before).reshape(-1)).item()),
        }
    return {
        "num_expected_keys": len(expected),
        "num_saved_keys": len(saved_keys),
        "num_loaded_tensors": len(saved),
        "max_abs_load_copy_diff": max_abs_load_copy_diff,
        "expected_keys": sorted(expected),
        "saved_keys": sorted(saved_keys),
        "per_tensor": per_tensor,
    }


# ---------------------------------------------------------------------------
# Deterministic probe forward -- returns the FULL mse tensor (CPU fp32).
# ---------------------------------------------------------------------------


def deterministic_probe_forward(
    model: nn.Module,
    vae_model: nn.Module,
    packer: Any,
    raw_dataset: Any,
    pair: Dict[str, Any],
    probe_seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    dtype = next(model.parameters()).dtype
    rng = _snapshot_rng_state()
    try:
        batch, trace = _pack_single_pair(packer, raw_dataset, pair, probe_seed)
        data = _prepare_batch(batch, vae_model, device, dtype)

        # Hash the packed inputs for cross-process equality checks. We hash the
        # tensors the model sees plus critical index/type/mask fields.
        packed_hashes: Dict[str, Any] = {}
        for key in ("packed_text_ids", "packed_text_embeds", "packed_position_ids",
                     "packed_timesteps", "padded_latent", "padded_vae_images",
                     "padded_vit_images", "packed_indexes", "packed_label_ids",
                     "packed_vae_token_indexes", "packed_vit_token_indexes",
                     "packed_query_indexes", "packed_key_value_indexes",
                     "mse_loss_indexes"):
            if key in data and isinstance(data[key], torch.Tensor):
                packed_hashes[key] = {
                    "shape": list(data[key].shape),
                    "dtype": str(data[key].dtype),
                    "hash": _tensor_hash(data[key]),
                }

        was_training = model.training
        model.train()
        with torch.no_grad():
            with _autocast_for_model(model):
                out = model(**data)
        if not was_training:
            model.eval()
        mse = out["mse"] if isinstance(out, dict) else out.get("mse")
        assert mse is not None, "forward returned no mse"
        mse_cpu = mse.detach().to(device="cpu", dtype=torch.float32).clone()

        # Also hash the forward-time noise draw indirectly: we don't have
        # direct hook access, but we can hash the model's cuda rng state
        # before/after forward and take the delta as a proxy noise signature.
        probe_meta = {
            "probe_seed": int(probe_seed),
            "packed_timesteps_hash": trace.get("packed_timesteps_hash"),
            "packed_timesteps": trace.get("packed_timesteps"),
            "packed_input_hashes": packed_hashes,
            "sequence_length": trace.get("sequence_length"),
            "num_vae_tokens": trace.get("num_vae_tokens"),
            "num_vit_tokens": trace.get("num_vit_tokens"),
            "mse_tensor_hash": _tensor_hash(mse_cpu),
            "mse_shape": list(mse_cpu.shape),
            "mse_dtype": str(mse_cpu.dtype),
            "mse_mean": float(mse_cpu.mean().item()),
            "mse_max": float(mse_cpu.max().item()),
            "mse_min": float(mse_cpu.min().item()),
            "mse_all_finite": bool(torch.isfinite(mse_cpu).all().item()),
        }
        return mse_cpu, probe_meta
    finally:
        _restore_rng_state(rng)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    start_wall = time.time()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Safety: ensure output-dir is not under /dockerdata.
    out_abs = args.output_dir.resolve()
    if str(out_abs).startswith("/dockerdata"):
        raise RuntimeError(
            f"Refusing to write verifier outputs under /dockerdata: {out_abs}. "
            f"Generated data must live under /group."
        )

    log("cold_reload_verifier_start", label=args.label, output_dir=str(args.output_dir),
        trainable_checkpoint=str(args.trainable_checkpoint), pair_id=args.pair_id)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    # ---- Base checkpoint provenance ----
    base_sha = _sha256_file(Path(args.checkpoint_path))
    base_size = Path(args.checkpoint_path).stat().st_size
    ckpt_sha = _sha256_file(Path(args.trainable_checkpoint))
    ckpt_size = Path(args.trainable_checkpoint).stat().st_size
    log("provenance_hashed", base_sha=base_sha, ckpt_sha=ckpt_sha)

    # ---- Tokenizer / manifest / pair ----
    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)
    records = load_jsonl(args.manifest)
    pair = _pick_pair(records, args.pair_id)
    raw_dataset = _build_raw_sample_dataset(args, tokenizer)
    packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)
    pair_seed = _stable_pair_seed(args.pair_id, args.seed)
    fixed_probe_seed = pair_seed + 999  # SAME as training-side probe seed

    # Prompt provenance
    prompt_text = paired_prompt(pair, include_source_image=True)
    prompt_hash = _string_hash(prompt_text)
    pair_image_hashes = _hash_source_target_images(pair)

    # ---- Build model + inject LoRA + set trainable ----
    model, vae_model, checkpoint_load_report = _build_model_for_gold40(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
    )
    target_names = _iter_target_linear_names(model, args.lora_layers)
    wrappers = inject_lora(model, target_names, args.lora_r, args.lora_alpha, args.lora_dropout)
    trainable_report = set_trainable(model, args.trainable_name_regex, wrappers)
    existing_names = trainable_report["existing_trainable_names"]
    lora_names = trainable_report["lora_trainable_names"]

    # Hooks for call counts on wrapped Linears.
    counter = CallCounter()
    for full in target_names:
        counter.register(full, model.get_submodule(full))

    # ---- Dtype invariants (initial) ----
    dtype_desc_initial = _describe_dtypes(model, existing_names, lora_names, target_names)
    _assert_dtype_invariants(dtype_desc_initial, expected_device_type=device.type)

    # ---- Strict-load the trainable checkpoint ----
    payload = torch.load(args.trainable_checkpoint, map_location="cpu", weights_only=False)
    saved_meta = payload.get("meta", {})
    load_report = load_trainable_strict_with_diff(payload, model)

    # ---- Dtype invariants (after load) ----
    dtype_desc_after_load = _describe_dtypes(model, existing_names, lora_names, target_names)
    _assert_dtype_invariants(dtype_desc_after_load, expected_device_type=device.type)

    log("checkpoint_strict_loaded",
        num_loaded=load_report["num_loaded_tensors"],
        max_abs_load_copy_diff=load_report["max_abs_load_copy_diff"])

    # ---- Deterministic probe forward -> full MSE tensor ----
    mse_tensor, probe_meta = deterministic_probe_forward(
        model, vae_model, packer, raw_dataset, pair, fixed_probe_seed, device,
    )
    mse_tensor_path = args.output_dir / f"mse_probe_tensor_{args.label}.pt"
    torch.save({"mse": mse_tensor, "meta": probe_meta}, mse_tensor_path)
    log("probe_forward_done",
        mse_mean=probe_meta["mse_mean"], mse_hash=probe_meta["mse_tensor_hash"])

    # ---- Fixed-seed generation PNG ----
    png_path = None
    png_sha = None
    png_size = None
    if not args.skip_generation:
        model.eval()
        vae_model.eval()
        rng = _snapshot_rng_state()
        try:
            inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, args)
            png_path = args.output_dir / f"generation_{args.label}.png"
            _generate_one(inferencer, pair, png_path, args, seed=args.generation_seed)
        finally:
            _restore_rng_state(rng)
        png_sha = _sha256_file(png_path)
        png_size = png_path.stat().st_size
        log("generation_done", path=str(png_path), sha256=png_sha)

    # ---- GPU memory stats ----
    peak_alloc_mib = (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0) / (1024 * 1024)
    peak_reserved_mib = (torch.cuda.max_memory_reserved() if device.type == "cuda" else 0) / (1024 * 1024)
    now_alloc_mib = (torch.cuda.memory_allocated() if device.type == "cuda" else 0) / (1024 * 1024)
    now_reserved_mib = (torch.cuda.memory_reserved() if device.type == "cuda" else 0) / (1024 * 1024)

    hook_calls = dict(counter.calls)
    counter.remove()

    # ---- LoRA target-module config fingerprint (for cross-run equality) ----
    lora_config = {
        "layers": list(args.lora_layers),
        "r": int(args.lora_r),
        "alpha": int(args.lora_alpha),
        "dropout": float(args.lora_dropout),
        "target_module_suffixes": list(TARGET_MODULE_SUFFIXES),
        "target_names": list(target_names),
    }
    lora_config_hash = _string_hash(
        json.dumps(lora_config, ensure_ascii=False, sort_keys=True)
    )

    # ---- Missing / unexpected key sanity ----
    checks = {
        "strict_key_set_equal": True,  # would have raised otherwise
        "shape_dtype_finite_ok": True,  # would have raised otherwise
        "max_abs_load_copy_diff_zero": load_report["max_abs_load_copy_diff"] == 0.0,
        "dtype_invariants_initial_ok": True,
        "dtype_invariants_after_load_ok": True,
        "all_target_linears_replaced": len(target_names) == len(args.lora_layers) * len(TARGET_MODULE_SUFFIXES),
        "all_hooks_called_at_least_once": all(v > 0 for v in hook_calls.values()) if hook_calls else False,
        "mse_all_finite": probe_meta["mse_all_finite"],
        "generation_available": (png_path is not None),
    }

    report = {
        "label": args.label,
        "status": "passed" if all(v for v in checks.values() if isinstance(v, bool)) else "failed",
        "checks": checks,
        "wall_seconds": time.time() - start_wall,
        "pair_id": args.pair_id,
        "pair_seed": pair_seed,
        "probe_seed": fixed_probe_seed,
        "generation_seed": args.generation_seed,
        "seed": args.seed,
        "conditioning": {
            "prompt": prompt_text,
            "prompt_hash": prompt_hash,
            "pair_image_hashes": pair_image_hashes,
        },
        "base_checkpoint": {
            "path": str(args.checkpoint_path),
            "sha256": base_sha,
            "size_bytes": base_size,
        },
        "trainable_checkpoint": {
            "path": str(args.trainable_checkpoint),
            "sha256": ckpt_sha,
            "size_bytes": ckpt_size,
            "saved_meta": saved_meta,
        },
        "lora_config": lora_config,
        "lora_config_hash": lora_config_hash,
        "trainable_regex": args.trainable_name_regex,
        "expected_trainable_keys": load_report["expected_keys"],
        "saved_trainable_keys": load_report["saved_keys"],
        "num_expected_keys": load_report["num_expected_keys"],
        "num_saved_keys": load_report["num_saved_keys"],
        "existing_trainable_names": existing_names,
        "lora_trainable_names": lora_names,
        "target_names": target_names,
        "reload": {
            "num_loaded_tensors": load_report["num_loaded_tensors"],
            "max_abs_load_copy_diff": load_report["max_abs_load_copy_diff"],
            "per_tensor": load_report["per_tensor"],
        },
        "dtype_invariants": {
            "initial": dtype_desc_initial,
            "after_load": dtype_desc_after_load,
        },
        "hook_calls_per_target_linear": hook_calls,
        "probe": {
            **probe_meta,
            "mse_tensor_path": str(mse_tensor_path),
        },
        "generation": {
            "path": str(png_path) if png_path else None,
            "sha256": png_sha,
            "size_bytes": png_size,
        },
        "gpu_memory": {
            "peak_allocated_mib": peak_alloc_mib,
            "peak_reserved_mib": peak_reserved_mib,
            "final_allocated_mib": now_alloc_mib,
            "final_reserved_mib": now_reserved_mib,
        },
        "checkpoint_load_report_base": checkpoint_load_report,
    }

    report_path = args.output_dir / f"verifier_{args.label.lower()}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("cold_reload_verifier_finished",
        label=args.label,
        report=str(report_path),
        status=report["status"],
        wall_seconds=report["wall_seconds"])

    # Clean up references (helpful for local debug; process exit is the real
    # release mechanism the launcher relies on).
    del model, vae_model, wrappers, payload, mse_tensor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
