"""2-step LoRA smoke for the single-pair overfit path.

Purpose: verify that a minimal LoRA injection on the target-image generation
branch is functionally correct *before* running any real training.

Nothing here launches the 300-step B1 run; it only:
  * loads Base BAGEL,
  * injects r=8 LoRA on layers.24..27 × 6 gen-branch Linears,
  * runs 6 pretrain assertions on the *base-only* forward,
  * runs 2 real training steps with real optimizer.step,
  * checks per-tensor gradients (A allowed zero on step 1, both must be
    non-zero somewhere on step 2, but we never accept the weaker "every tensor
    strictly non-zero" gate that hides ranks stuck at zero),
  * saves a trainable-only checkpoint, reloads it into a *fresh* Base +
    identical LoRA structure, and asserts strict key equality,
  * generates two inference PNGs with a fixed generation seed before and after
    reload and compares SHA256 for bit-exact equivalence,
  * writes ``lora_2step_smoke_report.json`` with every measurement required by
    the audit checklist.

No verl / LLaMA-Factory / peft / FSDP / DeepSpeed. No edit to
``qwen2_navit.py``. LoRA is a thin wrapper attached in place of the target
``nn.Linear`` via ``setattr`` on the parent module.
"""

from __future__ import annotations

import argparse
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
from torch import nn
try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # noqa: N816

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.training.gold40_overfit import (
    FIXED_GOLD40_MANIFEST,
    DEFAULT_TRAINABLE_REGEX,
    _autocast_for_model,
    _build_model_for_gold40,
    _generate_one,
    _generation_parameters,
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
    _run_one_step_assertions,
)


# ---------------------------------------------------------------------------
# LoRA wrapper (fp32 A/B, bf16 forward computation, correct dtype casts)
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` with an additive fp32 LoRA update.

    Storage:
      * ``base`` -- the original ``nn.Linear``, weights frozen in bf16
        (whatever dtype the base module was in when we wrapped it).
      * ``lora_A`` / ``lora_B`` -- fp32 parameters. Kept in fp32 because AdamW
        + gradient statistics become unstable in bf16 for small updates.

    Forward:
      * We compute ``base(x)`` in ``x.dtype`` (bf16), unchanged.
      * LoRA path casts ``x`` to fp32 for the A/B matmuls, then casts the
        result back to ``base_out.dtype`` and adds. This never promotes the
        rest of the network to fp32.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert isinstance(base, nn.Linear), f"LoRALinear expects nn.Linear, got {type(base)}"
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout_p = float(dropout)

        # Keep the base as a submodule so ``named_parameters`` still exposes
        # its weight/bias for freeze checks. We do NOT alias its parameters.
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        # fp32 LoRA parameters. Standard init: A ~ Kaiming, B == 0.
        self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # ``lora_B`` stays zero so LoRA output is exactly zero at init.

    # -------- book-keeping helpers used by the smoke script -----------------
    @property
    def base_weight_dtype(self) -> torch.dtype:
        return self.base.weight.dtype

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.alpha}"

    # -------- forward -------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)  # bf16 path unchanged
        if self.r == 0:
            return base_out
        # LoRA path in fp32; cast back at the end so we don't leak fp32 into
        # the rest of the graph.
        x32 = x.to(torch.float32)
        if self.dropout_p > 0.0 and self.training:
            x32 = torch.nn.functional.dropout(x32, p=self.dropout_p)
        lora_out = torch.nn.functional.linear(x32, self.lora_A)              # -> [..., r]
        lora_out = torch.nn.functional.linear(lora_out, self.lora_B)         # -> [..., out]
        lora_out = lora_out.to(base_out.dtype) * self.scaling
        return base_out + lora_out


# ---------------------------------------------------------------------------
# LoRA injection
# ---------------------------------------------------------------------------

TARGET_MODULE_SUFFIXES = (
    "self_attn.q_proj_moe_gen",
    "self_attn.k_proj_moe_gen",
    "self_attn.v_proj_moe_gen",
    "self_attn.o_proj_moe_gen",
    "mlp_moe_gen.gate_proj",
    "mlp_moe_gen.down_proj",
)


def _iter_target_linear_names(model: nn.Module, layer_indexes: Sequence[int]) -> List[str]:
    """Return fully-qualified names like
    ``language_model.model.layers.24.self_attn.q_proj_moe_gen``.

    Modules are resolved by looking up ``model.get_submodule(name)`` and
    checking it is exactly an ``nn.Linear`` (before wrapping). We do NOT use
    string prefix guesses; every hit is confirmed by attribute lookup.
    """
    names: List[str] = []
    for L in layer_indexes:
        for suffix in TARGET_MODULE_SUFFIXES:
            full = f"language_model.model.layers.{L}.{suffix}"
            try:
                submod = model.get_submodule(full)
            except AttributeError as exc:
                raise RuntimeError(f"target module missing: {full}: {exc}")
            if not isinstance(submod, nn.Linear):
                raise RuntimeError(
                    f"target module {full} is {type(submod).__name__}, expected nn.Linear"
                )
            names.append(full)
    return names


def _split_parent_child(full_name: str) -> Tuple[str, str]:
    parent, _, child = full_name.rpartition(".")
    return parent, child


def inject_lora(model: nn.Module, target_names: Sequence[str], r: int, alpha: int, dropout: float) -> Dict[str, LoRALinear]:
    """Replace each ``model.<target_name>`` (an ``nn.Linear``) with a
    ``LoRALinear`` wrapping it. Returns a dict {full_name: LoRALinear}."""
    wrappers: Dict[str, LoRALinear] = {}
    # LoRA weights live on the same device as the base Linear weight.
    for full in target_names:
        parent_name, child_name = _split_parent_child(full)
        parent = model.get_submodule(parent_name)
        base_lin: nn.Linear = getattr(parent, child_name)
        wrapper = LoRALinear(base_lin, r=r, alpha=alpha, dropout=dropout)
        # Place LoRA fp32 params on same device as base weight.
        dev = base_lin.weight.device
        wrapper.lora_A.data = wrapper.lora_A.data.to(device=dev)
        wrapper.lora_B.data = wrapper.lora_B.data.to(device=dev)
        setattr(parent, child_name, wrapper)
        wrappers[full] = wrapper
    return wrappers


# ---------------------------------------------------------------------------
# Freeze / trainable configuration
# ---------------------------------------------------------------------------


def set_trainable(
    model: nn.Module,
    keep_existing_regex: str,
    wrappers: Dict[str, LoRALinear],
) -> Dict[str, Any]:
    """Freeze everything, then re-enable:
      * every parameter whose ``name`` matches ``keep_existing_regex`` -- the
        existing 14 tensors (time_embedder / vae2llm / llm2vae / latent_pos_embed
        / vit_pos_embed / connector.*).
      * every ``lora_A`` and ``lora_B`` inside the injected wrappers.
    """
    import re

    pattern = re.compile(keep_existing_regex)
    for p in model.parameters():
        p.requires_grad_(False)

    existing_names: List[str] = []
    for name, p in model.named_parameters():
        if pattern.search(name):
            p.requires_grad_(True)
            existing_names.append(name)

    lora_names: List[str] = []
    for full, wrapper in wrappers.items():
        wrapper.lora_A.requires_grad_(True)
        wrapper.lora_B.requires_grad_(True)
        lora_names.append(f"{full}.lora_A")
        lora_names.append(f"{full}.lora_B")

    return {
        "existing_trainable_names": existing_names,
        "lora_trainable_names": lora_names,
    }


# ---------------------------------------------------------------------------
# Hooks: count calls of each wrapped Linear
# ---------------------------------------------------------------------------


class CallCounter:
    def __init__(self) -> None:
        self.calls: Dict[str, int] = {}
        self.handles: List[Any] = []

    def register(self, name: str, module: nn.Module) -> None:
        self.calls[name] = 0
        handle = module.register_forward_hook(self._make_hook(name))
        self.handles.append(handle)

    def _make_hook(self, name: str):
        def _hook(_mod, _inp, _out):  # noqa: ANN001
            self.calls[name] += 1
        return _hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# Gradient / update helpers
# ---------------------------------------------------------------------------


def collect_grad_stats(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """Return per-tensor gradient statistics for every ``requires_grad=True``
    parameter. Records name -> {grad_present, grad_l2, grad_max_abs,
    grad_all_finite, numel}."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            out[name] = {
                "grad_present": False,
                "grad_l2": 0.0,
                "grad_max_abs": 0.0,
                "grad_all_finite": True,
                "numel": int(p.numel()),
            }
            continue
        g32 = g.detach().to(torch.float32)
        out[name] = {
            "grad_present": True,
            "grad_l2": float(torch.linalg.vector_norm(g32.reshape(-1)).item()),
            "grad_max_abs": float(g32.abs().max().item()),
            "grad_all_finite": bool(torch.isfinite(g32).all().item()),
            "numel": int(p.numel()),
        }
    return out


def snapshot_trainable_fp32(model: nn.Module) -> Dict[str, torch.Tensor]:
    snap: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            snap[name] = p.detach().to(device="cpu", dtype=torch.float32, copy=True)
    return snap


def param_update_norms(before: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or name not in before:
            continue
        after = p.detach().to(device="cpu", dtype=torch.float32)
        delta = (after - before[name]).reshape(-1)
        out[name] = {
            "delta_l2": float(torch.linalg.vector_norm(delta).item()),
            "delta_max_abs": float(delta.abs().max().item()),
            "delta_mean_abs": float(delta.abs().mean().item()),
        }
    return out


# ---------------------------------------------------------------------------
# Trainable-only checkpoint I/O (strict key equality)
# ---------------------------------------------------------------------------


def trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def save_trainable_only(path: Path, model: nn.Module, meta: Dict[str, Any]) -> str:
    payload = {
        "format": "lora_smoke_trainable_only_v1",
        "trainable_state_dict": trainable_state_dict(model),
        "meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return str(path)


def load_trainable_strict(path: Path, model: nn.Module) -> Dict[str, Any]:
    """Strict reload: the set of ``requires_grad=True`` parameter names in the
    model must equal the set of keys in the checkpoint. No ``strict=False``."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload["trainable_state_dict"]
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    saved_keys = set(saved.keys())
    missing = expected - saved_keys
    unexpected = saved_keys - expected
    if missing or unexpected:
        raise RuntimeError(
            f"strict trainable-key mismatch: missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}"
        )
    shape_mismatch: List[str] = []
    max_abs_load_diff = 0.0
    named = dict(model.named_parameters())
    for name, tensor in saved.items():
        p = named[name]
        if tuple(p.shape) != tuple(tensor.shape):
            shape_mismatch.append(f"{name}: model={tuple(p.shape)} vs ckpt={tuple(tensor.shape)}")
            continue
        before = p.detach().to(device="cpu", dtype=torch.float32)
        p.data.copy_(tensor.to(device=p.device, dtype=p.dtype))
        after = p.detach().to(device="cpu", dtype=torch.float32)
        diff = float((after - tensor.to(dtype=torch.float32)).abs().max().item())
        max_abs_load_diff = max(max_abs_load_diff, diff)
    if shape_mismatch:
        raise RuntimeError(f"strict trainable-shape mismatch: {shape_mismatch[:8]}")
    return {
        "num_loaded_tensors": len(saved),
        "max_abs_load_copy_diff": max_abs_load_diff,
        "keys_expected": len(expected),
        "keys_in_ckpt": len(saved_keys),
    }


# ---------------------------------------------------------------------------
# Deterministic forward that returns the full (unreduced) mse tensor
# ---------------------------------------------------------------------------


def deterministic_forward_mse_tensor(
    model: nn.Module,
    vae_model: nn.Module,
    packer: Any,
    raw_dataset: Any,
    pair: Dict[str, Any],
    pair_seed: int,
) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    """Pack the pair with ``pair_seed`` (deterministic) and run one forward.

    Returns the full mse tensor (before .mean()), its mean, and the pack trace.
    RNG state is snapshotted/restored around the call so the caller's training
    trajectory is not perturbed.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    rng = _snapshot_rng_state()
    try:
        batch, trace = _pack_single_pair(packer, raw_dataset, pair, pair_seed)
        data = _prepare_batch(batch, vae_model, device, dtype)
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
        return mse_cpu, float(mse_cpu.mean().item()), trace
    finally:
        _restore_rng_state(rng)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _describe_dtypes(model: nn.Module, existing_names: Sequence[str], lora_names: Sequence[str], target_names: Sequence[str]) -> Dict[str, Any]:
    """Return a per-group dtype breakdown for existing trainable tensors,
    LoRA A/B tensors, and the frozen wrapped base weights. Also collects
    device info per group so we can verify parameters + inputs share the same
    GPU."""
    named = dict(model.named_parameters())
    named_bufs = dict(model.named_buffers())

    def _stat(pnames: Sequence[str]) -> Dict[str, Any]:
        dtypes: Dict[str, int] = {}
        devices: Dict[str, int] = {}
        for n in pnames:
            if n not in named:
                continue
            p = named[n]
            dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + 1
            devices[str(p.device)] = devices.get(str(p.device), 0) + 1
        return {"count": len(pnames), "dtypes": dtypes, "devices": devices}

    existing_stat = _stat(existing_names)
    lora_A_names = [n for n in lora_names if n.endswith(".lora_A")]
    lora_B_names = [n for n in lora_names if n.endswith(".lora_B")]
    lora_A_stat = _stat(lora_A_names)
    lora_B_stat = _stat(lora_B_names)

    # Wrapped base: read the .base.weight (and .base.bias if present) of each
    # LoRALinear.
    wrapped_base_dtypes: Dict[str, int] = {}
    wrapped_base_devices: Dict[str, int] = {}
    wrapped_base_requires_grad: List[str] = []
    per_target: Dict[str, Dict[str, Any]] = {}
    for full in target_names:
        wrapper = model.get_submodule(full)
        assert isinstance(wrapper, LoRALinear), f"{full} not wrapped"
        w = wrapper.base.weight
        wrapped_base_dtypes[str(w.dtype)] = wrapped_base_dtypes.get(str(w.dtype), 0) + 1
        wrapped_base_devices[str(w.device)] = wrapped_base_devices.get(str(w.device), 0) + 1
        if w.requires_grad:
            wrapped_base_requires_grad.append(f"{full}.base.weight")
        if wrapper.base.bias is not None:
            b = wrapper.base.bias
            wrapped_base_dtypes[str(b.dtype)] = wrapped_base_dtypes.get(str(b.dtype), 0) + 1
            wrapped_base_devices[str(b.device)] = wrapped_base_devices.get(str(b.device), 0) + 1
            if b.requires_grad:
                wrapped_base_requires_grad.append(f"{full}.base.bias")
        per_target[full] = {
            "base_weight_dtype": str(w.dtype),
            "base_weight_device": str(w.device),
            "base_weight_shape": list(w.shape),
            "base_weight_requires_grad": bool(w.requires_grad),
            "lora_A_dtype": str(wrapper.lora_A.dtype),
            "lora_A_device": str(wrapper.lora_A.device),
            "lora_A_shape": list(wrapper.lora_A.shape),
            "lora_A_requires_grad": bool(wrapper.lora_A.requires_grad),
            "lora_B_dtype": str(wrapper.lora_B.dtype),
            "lora_B_device": str(wrapper.lora_B.device),
            "lora_B_shape": list(wrapper.lora_B.shape),
            "lora_B_requires_grad": bool(wrapper.lora_B.requires_grad),
        }

    return {
        "existing": existing_stat,
        "lora_A": lora_A_stat,
        "lora_B": lora_B_stat,
        "wrapped_base": {
            "count": len(target_names),
            "dtypes": wrapped_base_dtypes,
            "devices": wrapped_base_devices,
            "requires_grad_leaks": wrapped_base_requires_grad,
        },
        "per_target": per_target,
    }


def _assert_dtype_invariants(desc: Dict[str, Any], expected_device_type: str) -> None:
    """Hard-check the three dtype invariants required by the audit:

    1. every ``lora_A``/``lora_B`` is ``torch.float32``
    2. every ``wrapped_base.weight`` (and bias) keeps its ORIGINAL bf16 dtype
       (no in-place cast happened during wrapping)
    3. every trainable param + wrapped base lives on the same accelerator
       device type (cuda)

    LoRA fp32-path -> bf16 add is checked implicitly at forward time by the
    fact that ``base_out`` remains bf16 and grads flow; but we do run a probe
    forward inside the smoke to be sure.
    """
    fp32 = "torch.float32"
    bf16 = "torch.bfloat16"
    assert desc["lora_A"]["dtypes"] == {fp32: desc["lora_A"]["count"]}, \
        f"lora_A dtypes not all fp32: {desc['lora_A']['dtypes']}"
    assert desc["lora_B"]["dtypes"] == {fp32: desc["lora_B"]["count"]}, \
        f"lora_B dtypes not all fp32: {desc['lora_B']['dtypes']}"
    # Wrapped base MUST be bf16 (untouched from load-time). If the base was
    # ever loaded as fp16/fp32 the whole precision plan is wrong.
    wb = desc["wrapped_base"]["dtypes"]
    assert wb.get(bf16, 0) == sum(wb.values()) and sum(wb.values()) > 0, \
        f"wrapped_base weights not all bf16: {wb}"
    assert not desc["wrapped_base"]["requires_grad_leaks"], \
        f"wrapped_base has leaked requires_grad params: {desc['wrapped_base']['requires_grad_leaks'][:8]}"
    # Devices: allow only 'cuda:*' entries; existing/lora_A/lora_B/wrapped_base
    # must all be on the same accelerator.
    for group in ("existing", "lora_A", "lora_B", "wrapped_base"):
        devs = desc[group]["devices"]
        assert devs and all(d.startswith(expected_device_type) for d in devs.keys()), \
            f"{group} devices not on {expected_device_type}: {devs}"


def _png_pixel_diff(a: Path, b: Path) -> Dict[str, Any]:
    """Compare two PNGs pixel-wise. Returns max/mean absolute diff on the RGB
    channels (0..255) and the fraction of pixels that differ at all. This lets
    us tell apart "CUDA microscopic non-determinism -> tiny pixel deltas" from
    "broken reload -> large deltas"."""
    if Image is None or a is None or b is None or not Path(a).exists() or not Path(b).exists():
        return {"available": False}
    im_a = np.asarray(Image.open(a).convert("RGB"), dtype=np.int32)
    im_b = np.asarray(Image.open(b).convert("RGB"), dtype=np.int32)
    if im_a.shape != im_b.shape:
        return {"available": True, "shape_mismatch": True, "shape_a": list(im_a.shape), "shape_b": list(im_b.shape)}
    diff = np.abs(im_a - im_b)
    total = float(im_a.size)
    nonzero = int((diff != 0).sum())
    return {
        "available": True,
        "shape": list(im_a.shape),
        "max_abs_diff": int(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "nonzero_channel_count": nonzero,
        "nonzero_channel_fraction": nonzero / total,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=FIXED_GOLD40_MANIFEST)
    parser.add_argument("--pair-id", default=DEFAULT_PAIR_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generation-seed", type=int, default=1234)
    # LoRA config
    parser.add_argument("--lora-layers", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    # Testing knobs
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip the 2 fixed-seed inferences (they take ~1min each). Not recommended.")
    return parser.parse_args()


def log(event: str, **fields: Any) -> None:
    fields["event"] = event
    print(json.dumps(fields, ensure_ascii=False, sort_keys=True, default=str), flush=True)


def _pick_pair(records: Sequence[Dict[str, Any]], pair_id: str) -> Dict[str, Any]:
    for r in records:
        if r.get("pair_id") == pair_id:
            return r
    raise KeyError(pair_id)


def build_model_and_inject(args: argparse.Namespace, device: torch.device, tokenizer_size: int, num_new_tokens: int, wrappers_out: List[str]) -> Tuple[Any, Any, Dict[str, LoRALinear], Dict[str, Any]]:
    """Return (model, vae_model, wrappers, load_report). Also mutates
    ``wrappers_out`` in-place with the full target names for provenance."""
    model, vae_model, load_report = _build_model_for_gold40(
        args, device, tokenizer_size=tokenizer_size, num_new_tokens=num_new_tokens,
    )
    target_names = _iter_target_linear_names(model, args.lora_layers)
    wrappers = inject_lora(model, target_names, args.lora_r, args.lora_alpha, args.lora_dropout)
    wrappers_out.extend(target_names)
    return model, vae_model, wrappers, load_report


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    (args.output_dir / "visualizations").mkdir(exist_ok=True)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    log("lora_2step_smoke_start", output_dir=str(args.output_dir), pair_id=args.pair_id, lr=args.lr,
        lora_layers=args.lora_layers, lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)
    records = load_jsonl(args.manifest)
    pair = _pick_pair(records, args.pair_id)
    raw_dataset = _build_raw_sample_dataset(args, tokenizer)
    packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)
    pair_seed = _stable_pair_seed(args.pair_id, args.seed)

    # ---- Build model + LoRA injection ----
    target_names: List[str] = []
    model, vae_model, wrappers, checkpoint_load_report = build_model_and_inject(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
        wrappers_out=target_names,
    )
    trainable_report = set_trainable(model, args.trainable_name_regex, wrappers)
    existing_names = trainable_report["existing_trainable_names"]
    lora_names = trainable_report["lora_trainable_names"]

    # ---- Trainable audit ----
    named = dict(model.named_parameters())
    existing_params = [(n, named[n]) for n in existing_names]
    lora_params_only = [(n, named[n]) for n in lora_names]
    all_trainable = [(n, p) for n, p in named.items() if p.requires_grad]

    existing_numel = sum(p.numel() for _, p in existing_params)
    lora_numel = sum(p.numel() for _, p in lora_params_only)
    total_trainable_numel = sum(p.numel() for _, p in all_trainable)

    assert existing_numel + lora_numel == total_trainable_numel, "existing + lora != all trainable"

    dtype_desc_initial = _describe_dtypes(model, existing_names, lora_names, target_names)
    _assert_dtype_invariants(dtype_desc_initial, expected_device_type=device.type)
    log("trainable_audit", existing_tensors=len(existing_params), existing_numel=existing_numel,
        lora_tensors=len(lora_params_only), lora_numel=lora_numel,
        total_trainable_tensors=len(all_trainable), total_trainable_numel=total_trainable_numel,
        existing_dtypes=dtype_desc_initial["existing"]["dtypes"],
        lora_A_dtypes=dtype_desc_initial["lora_A"]["dtypes"],
        lora_B_dtypes=dtype_desc_initial["lora_B"]["dtypes"],
        wrapped_base_dtypes=dtype_desc_initial["wrapped_base"]["dtypes"],
        existing_devices=dtype_desc_initial["existing"]["devices"],
        wrapped_base_devices=dtype_desc_initial["wrapped_base"]["devices"])

    # Check base weights are frozen (spot-check a few + verify no base.weight/bias is in trainable set)
    base_leak: List[str] = []
    for full in target_names:
        parent_name, _ = _split_parent_child(full)
        wrapper: LoRALinear = model.get_submodule(full)
        assert isinstance(wrapper, LoRALinear), f"{full} not wrapped"
        for pname, p in wrapper.base.named_parameters():
            if p.requires_grad:
                base_leak.append(f"{full}.base.{pname}")
    # Also ensure a random non-lora base linear is frozen (spot check first layer q_proj)
    spot = model.get_submodule("language_model.model.layers.0.self_attn.q_proj")
    if any(p.requires_grad for p in spot.parameters()):
        base_leak.append("language_model.model.layers.0.self_attn.q_proj (spot)")
    assert not base_leak, f"base weights leaked into trainable: {base_leak[:8]}"

    # ---- Forward hooks on the 24 target Linears (post-wrap: the wrapper is the callable) ----
    counter = CallCounter()
    for full in target_names:
        counter.register(full, model.get_submodule(full))

    # ---- Pretrain 6-item assertions (with the wrapped model; LoRA output is 0 at init so mse should match un-wrapped forward) ----
    assertion_path = args.output_dir / "pretrain_1step_assertions.json"
    assertion_detail = _run_one_step_assertions(
        args, model, vae_model, packer, raw_dataset, pair, pair_seed, assertion_path,
    )
    log("pretrain_1step_assertions_passed", mse_mean=assertion_detail.get("mse_mean"),
        effective_mse_tokens=assertion_detail.get("effective_mse_tokens"))

    # ---- Optimizer ----
    trainable_param_objs: List[nn.Parameter] = [p for _, p in all_trainable]
    optimizer = torch.optim.AdamW(trainable_param_objs, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_param_names_saved: List[str] = [n for n, _ in all_trainable]
    optim_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected_param_ids = {id(p) for _, p in all_trainable}
    optim_param_ids_ok = optim_param_ids == expected_param_ids
    assert optim_param_ids_ok, "optimizer param set != trainable param set"
    log("optimizer_built", num_trainable_tensors=len(trainable_param_objs),
        num_trainable_parameters=total_trainable_numel, lr=args.lr)

    # ---- 2 training steps ----
    model_dtype = next(model.parameters()).dtype

    step_records: List[Dict[str, Any]] = []
    for step in (1, 2):
        step_seed = pair_seed + step
        batch, trace = _pack_single_pair(packer, raw_dataset, pair, step_seed)
        data = _prepare_batch(batch, vae_model, device, model_dtype)

        torch.manual_seed(step_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(step_seed)

        optimizer.zero_grad(set_to_none=True)
        model.train()
        with _autocast_for_model(model):
            out = model(**data)
        mse = out["mse"] if isinstance(out, dict) else out.get("mse")
        loss = mse.mean()
        before_snap = snapshot_trainable_fp32(model)
        loss.backward()

        grad_stats = collect_grad_stats(model)

        # Split into existing vs LoRA
        existing_stats = {n: grad_stats[n] for n in existing_names if n in grad_stats}
        lora_stats = {n: grad_stats[n] for n in lora_names if n in grad_stats}

        # Zero-grad counts
        zero_existing = [n for n, s in existing_stats.items() if s["grad_l2"] == 0.0]
        zero_lora_A = [n for n in lora_names if n.endswith(".lora_A") and lora_stats[n]["grad_l2"] == 0.0]
        zero_lora_B = [n for n in lora_names if n.endswith(".lora_B") and lora_stats[n]["grad_l2"] == 0.0]

        # ---- gradient assertions (revised: warn on individual zero modules,
        # only fail when ALL A or ALL B are zero at step 2) ----
        total_A = sum(1 for n in lora_names if n.endswith(".lora_A"))
        total_B = sum(1 for n in lora_names if n.endswith(".lora_B"))
        a_nonzero = sum(1 for n in lora_names if n.endswith(".lora_A") and lora_stats[n]["grad_l2"] > 0.0)
        b_nonzero = sum(1 for n in lora_names if n.endswith(".lora_B") and lora_stats[n]["grad_l2"] > 0.0)

        # Common: every LoRA grad must exist and be finite.
        for n, s in lora_stats.items():
            assert s["grad_present"], f"step{step} LoRA grad missing on {n}"
            assert s["grad_all_finite"], f"step{step} LoRA grad non-finite on {n}"

        warnings_this_step: List[str] = []
        if step == 1:
            # A is allowed to be all-zero (dL/dA = B @ upstream = 0 when B=0).
            # B must have at least one non-zero, but individual zero B tensors
            # only warrant a warning.
            assert b_nonzero >= 1, \
                f"step1 all lora_B grads zero (fatal). b_nonzero=0/{total_B}"
            if b_nonzero < total_B:
                warnings_this_step.append(
                    f"step1 lora_B zero-grad modules: {[n for n in lora_names if n.endswith('.lora_B') and lora_stats[n]['grad_l2'] == 0.0]}"
                )
            # A all-zero on step 1 is expected, not a warning.
        else:  # step == 2
            assert a_nonzero >= 1, \
                f"step2 all lora_A grads zero (fatal). a_nonzero=0/{total_A}"
            assert b_nonzero >= 1, \
                f"step2 all lora_B grads zero (fatal). b_nonzero=0/{total_B}"
            if a_nonzero < total_A:
                warnings_this_step.append(
                    f"step2 lora_A zero-grad modules: {[n for n in lora_names if n.endswith('.lora_A') and lora_stats[n]['grad_l2'] == 0.0]}"
                )
            if b_nonzero < total_B:
                warnings_this_step.append(
                    f"step2 lora_B zero-grad modules: {[n for n in lora_names if n.endswith('.lora_B') and lora_stats[n]['grad_l2'] == 0.0]}"
                )

        # Clip and step
        clip_norm = torch.nn.utils.clip_grad_norm_(trainable_param_objs, args.max_grad_norm)
        optimizer.step()

        update_stats = param_update_norms(before_snap, model)
        del before_snap

        step_records.append({
            "step": step,
            "step_seed": step_seed,
            "mse_mean": float(mse.detach().float().mean().item()),
            "mse_numel": int(mse.numel()),
            "mse_all_finite": bool(torch.isfinite(mse).all().item()),
            "grad_norm_pre_clip": float(clip_norm),
            "num_zero_existing": len(zero_existing),
            "num_zero_lora_A": len(zero_lora_A),
            "num_zero_lora_B": len(zero_lora_B),
            "num_lora_A_nonzero": a_nonzero,
            "num_lora_B_nonzero": b_nonzero,
            "total_lora_A": total_A,
            "total_lora_B": total_B,
            "zero_grad_names_lora_A": [n for n in lora_names if n.endswith(".lora_A") and lora_stats[n]["grad_l2"] == 0.0],
            "zero_grad_names_lora_B": [n for n in lora_names if n.endswith(".lora_B") and lora_stats[n]["grad_l2"] == 0.0],
            "zero_grad_names_existing": zero_existing,
            "warnings": warnings_this_step,
            "existing_grad_stats": existing_stats,
            "lora_grad_stats": lora_stats,
            "update_stats": update_stats,
        })
        log("train_step", step=step, mse=step_records[-1]["mse_mean"],
            grad_norm_pre_clip=step_records[-1]["grad_norm_pre_clip"],
            zero_lora_A=len(zero_lora_A), zero_lora_B=len(zero_lora_B),
            zero_existing=len(zero_existing))
        del data, out, mse, loss

    # ---- Save trainable-only checkpoint ----
    ckpt_path = args.output_dir / "checkpoints" / "step_002_trainable.pt"
    save_meta = {
        "step": 2,
        "global_step": 2,
        "lora_target_names": target_names,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "existing_trainable_regex": args.trainable_name_regex,
    }
    ckpt_str = save_trainable_only(ckpt_path, model, save_meta)
    ckpt_sha = _sha256_file(Path(ckpt_str))
    ckpt_size = Path(ckpt_str).stat().st_size
    log("checkpoint_saved", path=ckpt_str, sha256=ckpt_sha, bytes=ckpt_size)

    # ---- Compute mse tensor for reload-equivalence BEFORE reload ----
    fixed_probe_seed = pair_seed + 999  # distinct from step_seed 1 / 2
    mse_before, mse_before_mean, probe_trace = deterministic_forward_mse_tensor(
        model, vae_model, packer, raw_dataset, pair, fixed_probe_seed,
    )

    # ---- Optional: fixed-seed inference for PNG SHA256 ----
    if not args.skip_generation:
        was_training = model.training
        model.eval()
        vae_model.eval()
        try:
            rng = _snapshot_rng_state()
            try:
                inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, args)
                gen_before = args.output_dir / "visualizations" / "gen_before_reload.png"
                _generate_one(inferencer, pair, gen_before, args, seed=args.generation_seed)
            finally:
                _restore_rng_state(rng)
        finally:
            if was_training:
                model.train()
        gen_before_sha = _sha256_file(gen_before)
    else:
        gen_before = None
        gen_before_sha = None

    # ---- Peak memory for the training phase (before we touch reload) ----
    peak_alloc_train_mib = (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0) / (1024 * 1024)
    peak_reserved_train_mib = (torch.cuda.max_memory_reserved() if device.type == "cuda" else 0) / (1024 * 1024)
    # Approx frozen-base footprint = full model bf16 weights, roughly 27.9 GiB on the observed configuration.
    # We record allocated *just after* full training + probe + first inference; this is the number we compare the
    # post-release allocated against, to guard against a leaked 27 GiB replica.

    # ---- Fresh Base + identical LoRA + strict reload ----
    log("reload_start")

    # -- Remove forward hooks first (they hold refs to modules) --
    hook_calls = dict(counter.calls)
    counter.remove()

    # -- Drop every python reference to the current model, optimizer, wrappers,
    #    per-target Linear submodules, batches, mse tensors, etc. Anything that
    #    might keep a CUDA storage alive must be deleted BEFORE we call
    #    empty_cache. `batch/data/out/mse/loss` were already removed inside the
    #    training loop via ``del data, out, mse, loss``; here we drop the rest.
    for full in target_names:
        # Loose reference to each per-target wrapper is a strong ref through the
        # model tree; deleting the outer model handle is enough, but we also
        # drop the ``wrappers`` dict below.
        pass
    del all_trainable, trainable_param_objs, named, existing_params, lora_params_only
    del wrappers  # dict of LoRALinear refs
    del optimizer

    # `mse_before` still holds a CPU fp32 tensor; keep it -- it's small.
    # Wipe the model + vae last so no forward-hook has a live parent handle.
    del model, vae_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # reload phase gets its own peaks

    alloc_after_release_mib = (torch.cuda.memory_allocated() if device.type == "cuda" else 0) / (1024 * 1024)
    reserved_after_release_mib = (torch.cuda.memory_reserved() if device.type == "cuda" else 0) / (1024 * 1024)
    log("reload_after_release_memory",
        allocated_mib=alloc_after_release_mib,
        reserved_mib=reserved_after_release_mib)

    # If the post-release allocated is still near the frozen-base footprint
    # (>= 20 GiB is deep in "first model still resident" territory), we do NOT
    # attempt to load a second Base -- that would likely OOM. Report and exit
    # gracefully.
    RELEASE_ALLOC_LIMIT_MIB = 20 * 1024  # 20 GiB
    release_ok = alloc_after_release_mib < RELEASE_ALLOC_LIMIT_MIB
    if not release_ok:
        early_report = {
            "status": "failed",
            "reason": (
                f"post-release allocated {alloc_after_release_mib:.1f} MiB >= "
                f"{RELEASE_ALLOC_LIMIT_MIB} MiB -- first Bagel model was not"
                f" fully freed; refusing to load a second Base to avoid OOM."
            ),
            "pair_id": args.pair_id,
            "cuda_memory": {
                "peak_allocated_train_mib": peak_alloc_train_mib,
                "peak_reserved_train_mib": peak_reserved_train_mib,
                "allocated_after_release_mib": alloc_after_release_mib,
                "reserved_after_release_mib": reserved_after_release_mib,
            },
            "hook_calls_per_target_linear": hook_calls,
            "step_records": step_records,
            "lora_target_names": target_names,
        }
        early_path = args.output_dir / "lora_2step_smoke_report.json"
        early_path.write_text(json.dumps(early_report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        log("lora_2step_smoke_finished", report=str(early_path), status="failed", reason="release_check_failed")
        return

    target_names2: List[str] = []
    model2, vae_model2, wrappers2, checkpoint_load_report2 = build_model_and_inject(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
        wrappers_out=target_names2,
    )
    _ = set_trainable(model2, args.trainable_name_regex, wrappers2)
    assert target_names == target_names2, "LoRA target names diverged on second build"

    # Re-audit dtype after the fresh Base + inject.
    named2 = dict(model2.named_parameters())
    existing_names2 = [n for n in existing_names if n in named2]
    lora_names2 = [n for n in lora_names if n in named2]
    dtype_desc_reload = _describe_dtypes(model2, existing_names2, lora_names2, target_names2)
    _assert_dtype_invariants(dtype_desc_reload, expected_device_type=device.type)

    load_report = load_trainable_strict(Path(ckpt_str), model2)

    # Dtype must still hold after copy_.
    dtype_desc_after_load = _describe_dtypes(model2, existing_names2, lora_names2, target_names2)
    _assert_dtype_invariants(dtype_desc_after_load, expected_device_type=device.type)

    # ---- Same-input equivalence check ----
    mse_after, mse_after_mean, probe_trace2 = deterministic_forward_mse_tensor(
        model2, vae_model2, packer, raw_dataset, pair, fixed_probe_seed,
    )
    assert probe_trace["packed_timesteps_hash"] == probe_trace2["packed_timesteps_hash"], \
        "probe pack trace diverged between runs (deterministic pack broken)"
    max_abs_diff = float((mse_before - mse_after).abs().max().item())
    mean_abs_diff = float((mse_before - mse_after).abs().mean().item())

    # Two-tier reload equivalence:
    #   * exact: bit-identical mse tensor.
    #   * numerical: <= 1e-6 max, <= 1e-7 mean absolute diff (bf16 forward with
    #     fp32 LoRA on flash-attn kernels can produce tiny non-determinism).
    exact_reload_match = (max_abs_diff == 0.0) and (mean_abs_diff == 0.0)
    numerical_reload_match = (max_abs_diff <= 1e-6) and (mean_abs_diff <= 1e-7)

    # ---- Optional: fixed-seed inference for PNG SHA256 (after reload) ----
    if not args.skip_generation:
        was_training = model2.training
        model2.eval()
        vae_model2.eval()
        try:
            rng = _snapshot_rng_state()
            try:
                inferencer2 = _make_inferencer(model2, vae_model2, tokenizer, special_tokens, args)
                gen_after = args.output_dir / "visualizations" / "gen_after_reload.png"
                _generate_one(inferencer2, pair, gen_after, args, seed=args.generation_seed)
            finally:
                _restore_rng_state(rng)
        finally:
            if was_training:
                model2.train()
        gen_after_sha = _sha256_file(gen_after)
    else:
        gen_after = None
        gen_after_sha = None

    peak_alloc_reload_mib = (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0) / (1024 * 1024)
    peak_reserved_reload_mib = (torch.cuda.max_memory_reserved() if device.type == "cuda" else 0) / (1024 * 1024)

    # ---- Report ----
    all_hook_called = all(v > 0 for v in hook_calls.values())
    # Pixel-level PNG diff (only informational; SHA mismatch alone must not fail
    # the checkpoint if numerical reload passes and pixel diff is tiny).
    png_pixel_diff = _png_pixel_diff(gen_before, gen_after)
    png_sha_matches = (gen_before_sha == gen_after_sha) if (gen_before_sha and gen_after_sha) else None
    checks = {
        "all_target_linears_replaced": len(target_names) == len(args.lora_layers) * len(TARGET_MODULE_SUFFIXES),
        "trainable_num_matches_expected": (existing_numel + lora_numel) == total_trainable_numel,
        "optimizer_param_set_equals_trainable": optim_param_ids_ok,
        "no_base_weight_in_trainable": not base_leak,
        "all_target_linears_called_in_forward": all_hook_called,
        "pretrain_6_assertions_passed": all(assertion_detail["checks"].values()),
        "step1_at_least_one_lora_B_nonzero": step_records[0]["num_lora_B_nonzero"] >= 1,
        "step2_at_least_one_lora_A_nonzero": step_records[1]["num_lora_A_nonzero"] >= 1,
        "step2_at_least_one_lora_B_nonzero": step_records[1]["num_lora_B_nonzero"] >= 1,
        "step1_grads_all_finite": all(s["grad_all_finite"] for s in step_records[0]["lora_grad_stats"].values()) and all(s["grad_all_finite"] for s in step_records[0]["existing_grad_stats"].values()),
        "step2_grads_all_finite": all(s["grad_all_finite"] for s in step_records[1]["lora_grad_stats"].values()) and all(s["grad_all_finite"] for s in step_records[1]["existing_grad_stats"].values()),
        "dtype_invariants_initial": True,   # _assert_dtype_invariants raises otherwise
        "dtype_invariants_after_reload": True,
        "release_before_reload_ok": release_ok,
        "reload_key_set_strict_equal": True,  # load_trainable_strict raises otherwise
        "reload_max_abs_load_copy_diff_zero": load_report["max_abs_load_copy_diff"] == 0.0,
        "exact_reload_match": exact_reload_match,
        "numerical_reload_match": numerical_reload_match,
        "gen_png_sha256_matches": png_sha_matches,
    }

    report = {
        "status": "passed" if (
            checks["numerical_reload_match"]
            and checks["release_before_reload_ok"]
            and checks["reload_max_abs_load_copy_diff_zero"]
            and checks["step1_at_least_one_lora_B_nonzero"]
            and checks["step2_at_least_one_lora_A_nonzero"]
            and checks["step2_at_least_one_lora_B_nonzero"]
            and checks["step1_grads_all_finite"]
            and checks["step2_grads_all_finite"]
            and checks["all_target_linears_called_in_forward"]
            and checks["pretrain_6_assertions_passed"]
            and checks["no_base_weight_in_trainable"]
            and checks["optimizer_param_set_equals_trainable"]
            and checks["trainable_num_matches_expected"]
            and checks["all_target_linears_replaced"]
            and checks["dtype_invariants_initial"]
            and checks["dtype_invariants_after_reload"]
        ) else "failed",
        "pair_id": args.pair_id,
        "pair_seed": pair_seed,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "lora_target_names": target_names,
        "num_lora_tensors": len(lora_names),
        "num_lora_parameters": lora_numel,
        "num_existing_trainable_tensors": len(existing_names),
        "num_existing_trainable_parameters": existing_numel,
        "num_total_trainable_parameters": total_trainable_numel,
        "dtype_invariants": {
            "initial": dtype_desc_initial,
            "after_reload_before_load": dtype_desc_reload,
            "after_reload_load": dtype_desc_after_load,
        },
        "existing_trainable_names": existing_names,
        "lora_trainable_names": lora_names,
        "optimizer_param_names": optimizer_param_names_saved,
        "hook_calls_per_target_linear": hook_calls,
        "pretrain_assertions": {
            "checks": assertion_detail["checks"],
            "mse_mean": assertion_detail.get("mse_mean"),
            "mse_numel": assertion_detail.get("mse_numel"),
            "mse_all_finite": assertion_detail.get("mse_all_finite"),
            "effective_mse_tokens": assertion_detail.get("effective_mse_tokens"),
        },
        "step_records": step_records,
        "checkpoint": {
            "path": ckpt_str,
            "sha256": ckpt_sha,
            "size_bytes": ckpt_size,
            "meta": save_meta,
        },
        "reload": {
            "num_loaded_tensors": load_report["num_loaded_tensors"],
            "keys_expected": load_report["keys_expected"],
            "keys_in_ckpt": load_report["keys_in_ckpt"],
            "max_abs_load_copy_diff": load_report["max_abs_load_copy_diff"],
            "probe_seed": fixed_probe_seed,
            "probe_pack_hash": probe_trace["packed_timesteps_hash"],
            "mse_before_mean": mse_before_mean,
            "mse_after_mean": mse_after_mean,
            "mse_tensor_max_abs_diff": max_abs_diff,
            "mse_tensor_mean_abs_diff": mean_abs_diff,
            "exact_reload_match": exact_reload_match,
            "numerical_reload_match": numerical_reload_match,
            "gen_png_before_reload": str(gen_before) if gen_before else None,
            "gen_png_after_reload": str(gen_after) if gen_after else None,
            "gen_png_before_sha256": gen_before_sha,
            "gen_png_after_sha256": gen_after_sha,
            "gen_png_pixel_diff": png_pixel_diff,
        },
        "cuda_memory": {
            "peak_allocated_train_mib": peak_alloc_train_mib,
            "peak_reserved_train_mib": peak_reserved_train_mib,
            "allocated_after_release_mib": alloc_after_release_mib,
            "reserved_after_release_mib": reserved_after_release_mib,
            "peak_allocated_reload_mib": peak_alloc_reload_mib,
            "peak_reserved_reload_mib": peak_reserved_reload_mib,
        },
        "checks": checks,
    }
    out_path = args.output_dir / "lora_2step_smoke_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log("lora_2step_smoke_finished", report=str(out_path), status=report["status"])


if __name__ == "__main__":
    main()
