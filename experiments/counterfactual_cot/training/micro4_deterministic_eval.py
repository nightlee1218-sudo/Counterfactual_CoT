"""micro4 Deterministic Optimization Eval  —  v6.1

Design contract
===============
* Does NOT modify any existing training file.
* Imports helpers from gold40_overfit / gold40_lora / lora_2step_smoke /
  smoke_train as-is.
* --preflight  : static + path checks only.  No model load, no CUDA forward.
  Writes preflight_report.json to PREFLIGHT_OUTPUT_DIR (separate from eval dir).
  Exits 0 on success.  Does NOT touch the formal eval output directory.
* --run-eval   : full deterministic eval.  Writes all v6 artefacts under
  EVAL_OUTPUT_DIR (must be empty or non-existent).

Directory separation
====================
PREFLIGHT_OUTPUT_DIR : .../gold40_v3_lora_micro4_deterministic_eval_preflight
  Contains only preflight_report.json.  May be re-run without clearing.
EVAL_OUTPUT_DIR      : .../gold40_v3_lora_micro4_deterministic_eval_v3
  Must be empty before --run-eval.  Contains all formal eval artefacts.
  (_v1/_v2 were polluted by failed runs; _v3 is the clean formal eval dir)

packed_timesteps vs forward-internal sigmoid
============================================
packed_timesteps in the batch is the RAW np.random.randn() scalar
(one value per vae_image, broadcast to all its latent tokens).
Inside bagel.py:Bagel.forward() the raw value is transformed:
    t_sigmoid = torch.sigmoid(packed_timesteps)          # squash to (0,1)
    t_shifted = shift * t_sigmoid / (1 + (shift-1)*t_sigmoid)  # logit-normal shift
The MSE target uses the shifted t_shifted, NOT the raw randn value.
The eval fixes the raw randn value (via np.random.seed before loader build)
and the noise tensor (via torch.manual_seed before each forward).

RNG seeding
===========
Phase 2 fixes four independent RNG streams in this order:
  1. random.seed(EVAL_SEED)          — Python global RNG
     Note: CounterfactualCoTDataset.set_epoch() uses random.Random(training_seed
     + epoch), which is an INSTANCE-level RNG, not the global random module.
     The global random.seed() call here does NOT affect set_epoch() behaviour;
     it is included for completeness and future-proofing only.
  2. np.random.seed(EVAL_SEED)       — NumPy global RNG
     Controls pack_sequence()'s np.random.randn() timestep sampling.
     num_workers=0 is hardcoded in _build_gold40_loader, so NumPy RNG runs
     entirely in the main process — no worker-fork seed isolation needed.
  3. torch.manual_seed(EVAL_SEED)    — Torch CPU RNG
  4. torch.cuda.manual_seed_all(EVAL_SEED) — Torch CUDA RNG (all devices)
     Items 3-4 cover the canonical-build phase; per-forward noise seed is
     set separately as NOISE_SEED_BASE + pair_idx * 1000 before each call.

Trajectory classification (v6.1 simplified)
============================================
sustained               : best_checkpoint_effect in {effective, partial} AND
                          (best includes step_400 OR step_400 <= best * 1.10)
                          i.e. step_400 is best OR within 10% of best
transient_then_regressed: best_checkpoint_effect in {effective, partial} AND
                          best excludes step_400 AND step_400 > best * 1.10
no_clear_improvement    : best_checkpoint_effect == not_demonstrated
inconclusive            : any integrity gate failed

Note: 'improved' (regression_from_best) and 'sustained_with_plateau' (trajectory)
are removed.  step_400 <= best * 1.10 is classified as 'stable' in
regression_from_best and contributes to 'sustained' in trajectory_pattern.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# BAGEL_ROOT bootstrap
# ---------------------------------------------------------------------------
BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

# ---------------------------------------------------------------------------
# Frozen paths & constants
# ---------------------------------------------------------------------------
MICRO4_CKPT_DIR = (
    BAGEL_ROOT
    / "experiments/counterfactual_cot/outputs/gold40_v3_lora_micro4_400steps"
)
MICRO4_MANIFEST = (
    BAGEL_ROOT
    / "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region"
    / "manifests/intervention_pairs_micro4_v1.jsonl"
)
MICRO4_MANIFEST_SHA256 = "2d0790de5c4ad3a66eb26f4b46b45c1ed35155edfc1bb91aeb869ac256494891"

# Formal eval output directory (must be empty before --run-eval)
# _v1/_v2/_v3 were polluted by failed runs; use _v4 as the clean formal eval dir.
DEFAULT_EVAL_OUTPUT_DIR = (
    BAGEL_ROOT
    / "experiments/counterfactual_cot/outputs"
    / "gold40_v3_lora_micro4_deterministic_eval_v4"
)

# Preflight-only output directory (separate; may be re-run without clearing)
DEFAULT_PREFLIGHT_OUTPUT_DIR = (
    BAGEL_ROOT
    / "experiments/counterfactual_cot/outputs"
    / "gold40_v3_lora_micro4_deterministic_eval_preflight"
)

# Keep DEFAULT_OUTPUT_DIR as alias for eval dir (used by --output-dir default)
DEFAULT_OUTPUT_DIR = DEFAULT_EVAL_OUTPUT_DIR

# Checkpoint manifest (written by training run)
CKPT_MANIFEST_PATH = MICRO4_CKPT_DIR / "checkpoint_manifest.jsonl"

# Expected SHA256 per checkpoint step (from checkpoint_manifest.jsonl)
CKPT_SHA256: Dict[str, str] = {
    "step_000": "4bf3c41e126008f4c495efe3fa8298f52ddad7c9b1d5105c9e9671c663278692",
    "step_010": "0a17b9da129d52a56083b940afaa22b0670c07cade5cef13f910f6656b9b9b6c",
    "step_025": "52220c2f9eced62b1e1674414d7ba0f6040ae42c16981a58f38c35ae03c72a84",
    "step_050": "f3933267f2df86457bf6aae38e8c60de6fc7dad5bff6e2a0e7a99bdd957ea797",
    "step_100": "a78b6de0775414f0ccfa1c0e2da6468fe6cc4fba22575c87640b9600f4058682",
    "step_200": "d2a094ed5cd3fe8c877a44b0d708939627c0991be3f58e4daf822dc118991d35",
    "step_300": "fd7021d78da0c279a782a72992e13e7482ee72a79a40baf625fdfe0effe5beec",
    "step_400": "ab84f0f60d6e7a1ebd8f47c396e46cd5fcf259b1e5be0bcc6ccd2cad05493e90",
}

EVAL_CHECKPOINTS: List[Tuple[str, Path]] = [
    (label, MICRO4_CKPT_DIR / "checkpoints" / f"{label}_trainable.pt")
    for label in CKPT_SHA256
]

PAIR_ORDER: List[str] = [
    "cf_source_city_street_object_replace_0000",
    "cf_source_classroom_object_replace_0000",
    "cf_source_city_street_absolute_position_change_v4_absolute_region_review_fix_v6_short_0000",
    "cf_source_classroom_absolute_position_change_v4_absolute_region_replacement_v4_0000",
]

EVAL_SEED = 9999
NOISE_SEED_BASE = 42
MAX_CANONICAL_SCAN_BATCHES = 20
EXPECTED_TRAINABLE_TENSORS = 62
EXPECTED_LORA_B_TENSORS = 24
LEGACY_STEP000_MSE = 0.5173   # from micro4_full_training_report.json (ref only)
REGRESSION_THRESHOLD = 1.10   # step_400 > best * 1.10 → regressed
TIE_TOLERANCE = 0.001         # |mse_a - mse_b| < tolerance → tied best

# ---------------------------------------------------------------------------
# Utility helpers (no model imports)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    tmp.replace(path)


def _append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Tensor / buffer digest  (reshape(-1) → view(uint8), path/dtype/shape in ctx)
# ---------------------------------------------------------------------------

def _tensor_digest(path: str, t: "torch.Tensor") -> Dict[str, Any]:
    import torch
    t_cpu = t.detach().cpu().contiguous()
    flat = t_cpu.reshape(-1)
    raw_bytes = flat.view(torch.uint8).numpy().tobytes() if flat.numel() > 0 else b""
    ctx = f"{path}|{str(t.dtype)}|{list(t.shape)}|".encode()
    sha = hashlib.sha256(ctx + raw_bytes).hexdigest()
    return {"sha256": sha, "path": path, "dtype": str(t.dtype),
            "shape": list(t.shape), "numel": t.numel()}


def _scalar_digest(path: str, value: Any) -> Dict[str, Any]:
    if isinstance(value, float):
        raw = struct.pack(">d", value)
        ctx = f"{path}|float|".encode()
        sha = hashlib.sha256(ctx + raw).hexdigest()
        return {"sha256": sha, "path": path, "type": "float", "value": repr(value)}
    ctx = f"{path}|{type(value).__name__}|{value}".encode()
    sha = hashlib.sha256(ctx).hexdigest()
    return {"sha256": sha, "path": path, "type": type(value).__name__, "value": value}


def _recursive_digest(obj: Any, path: str = "root") -> Dict[str, Dict]:
    """Full recursive digest supporting Tensor/dict/list/tuple/scalar/None."""
    import torch
    result: Dict[str, Dict] = {}
    if isinstance(obj, torch.Tensor):
        result[path] = _tensor_digest(path, obj)
    elif isinstance(obj, dict):
        for k, v in sorted(obj.items()):
            result.update(_recursive_digest(v, f"{path}.{k}"))
    elif isinstance(obj, (list, tuple)):
        bracket = ("[]" if isinstance(obj, list) else "()")
        for i, v in enumerate(obj):
            result.update(_recursive_digest(v, f"{path}{bracket[0]}{i}{bracket[1]}"))
    elif obj is None:
        ctx = f"{path}|None|None".encode()
        result[path] = {"sha256": hashlib.sha256(ctx).hexdigest(),
                        "path": path, "type": "None", "value": None}
    elif isinstance(obj, bool):
        result[path] = _scalar_digest(path, obj)
    elif isinstance(obj, (int, float, str)):
        result[path] = _scalar_digest(path, obj)
    else:
        ctx = f"{path}|{type(obj).__name__}|{repr(obj)}".encode()
        result[path] = {"sha256": hashlib.sha256(ctx).hexdigest(),
                        "path": path, "type": type(obj).__name__, "value": repr(obj)}
    return result


def _snapshot_named_buffers(model: "torch.nn.Module") -> Dict[str, Dict]:
    import torch
    snap: Dict[str, Dict] = {}
    for name, buf in model.named_buffers():
        if buf is None:
            snap[name] = {"sha256": "None", "dtype": "None", "shape": [], "numel": 0}
        else:
            buf_cpu = buf.detach().cpu().contiguous()
            flat = buf_cpu.reshape(-1)
            raw = flat.view(torch.uint8).numpy().tobytes() if flat.numel() > 0 else b""
            ctx = f"{name}|{str(buf.dtype)}|{list(buf.shape)}|".encode()
            sha = hashlib.sha256(ctx + raw).hexdigest()
            snap[name] = {"sha256": sha, "dtype": str(buf.dtype),
                          "shape": list(buf.shape), "numel": buf.numel()}
    return snap


# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------

class GateLog:
    """Accumulates gate results; any blocking failure sets failed=True."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []
        self.failed = False

    def record(self, gate_id: str, status: str, **kwargs: Any) -> None:
        entry = {"gate_id": gate_id, "status": status,
                 "recorded_at": _now_iso(), **kwargs}
        self.records.append(entry)
        if status not in ("passed", "ref_only", "frozen", "skipped"):
            self.failed = True

    def assert_gate(self, gate_id: str, condition: bool, msg: str,
                    **kwargs: Any) -> None:
        if condition:
            self.record(gate_id, "passed", **kwargs)
        else:
            self.record(gate_id, "FAILED", detail=msg, **kwargs)
            raise AssertionError(f"{gate_id} FAILED: {msg}")

    def to_dict(self) -> Dict[str, Any]:
        return {"gates": self.records, "any_failure": self.failed,
                "total": len(self.records)}


# ---------------------------------------------------------------------------
# G11: canonical immutability check
# ---------------------------------------------------------------------------

def _check_g11(gate_log: GateLog, canonical: Dict[str, Any],
               phase: str, ckpt_label: str) -> None:
    for pair_id in PAIR_ORDER:
        ci = canonical[pair_id]
        current = _recursive_digest(ci["data"])
        frozen = ci["field_digest"]
        frozen_keys = set(frozen.keys())
        current_keys = set(current.keys())
        only_frozen = frozen_keys - current_keys
        only_current = current_keys - frozen_keys
        gate_log.assert_gate(
            f"G11_{phase}_{ckpt_label}_{pair_id[:30]}_keyset",
            not only_frozen and not only_current,
            f"field set changed: only_frozen={sorted(only_frozen)[:3]}, "
            f"only_current={sorted(only_current)[:3]}",
        )
        sha_failures = [
            f"{fp}: frozen={frozen[fp]['sha256'][:12]}... "
            f"current={current[fp]['sha256'][:12]}..."
            for fp in sorted(frozen_keys & current_keys)
            if frozen[fp]["sha256"] != current[fp]["sha256"]
        ]
        gate_log.assert_gate(
            f"G11_{phase}_{ckpt_label}_{pair_id[:30]}_sha",
            not sha_failures,
            f"{len(sha_failures)} field(s) changed: " + "; ".join(sha_failures[:5]),
        )


# ---------------------------------------------------------------------------
# G13: named_buffer immutability check
# ---------------------------------------------------------------------------

def _check_g13(gate_log: GateLog, model: "torch.nn.Module",
               pre_snap: Dict[str, Dict], ckpt_label: str) -> None:
    post_snap = _snapshot_named_buffers(model)
    gate_log.assert_gate(
        f"G13_{ckpt_label}_keyset",
        set(post_snap.keys()) == set(pre_snap.keys()),
        f"buffer set changed",
    )
    for name, pre in pre_snap.items():
        post = post_snap.get(name, {})
        gate_log.assert_gate(
            f"G13_{ckpt_label}_{name[:40]}_sha",
            post.get("sha256") == pre.get("sha256"),
            f"buffer '{name}' changed: pre={pre.get('sha256','?')[:12]} "
            f"post={post.get('sha256','?')[:12]}",
        )


# ---------------------------------------------------------------------------
# G12: per-tensor exact copy verification
# ---------------------------------------------------------------------------

def _gate_g12_per_tensor(gate_log: GateLog, ckpt_path: Path,
                         model: "torch.nn.Module", ckpt_label: str) -> None:
    import torch
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = payload["trainable_state_dict"]
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}

    saved_keys = set(saved.keys())
    model_keys = set(named.keys())
    missing = model_keys - saved_keys
    unexpected = saved_keys - model_keys
    gate_log.assert_gate(
        f"G12_{ckpt_label}_keys",
        not missing and not unexpected,
        f"key mismatch missing={sorted(missing)[:5]} unexpected={sorted(unexpected)[:5]}",
        num_saved=len(saved_keys), num_model=len(model_keys),
    )

    failures: List[str] = []
    for key in sorted(saved_keys):
        ckpt_t = saved[key]
        model_p = named[key]
        if tuple(ckpt_t.shape) != tuple(model_p.shape):
            failures.append(f"{key}: shape ckpt={tuple(ckpt_t.shape)} model={tuple(model_p.shape)}")
            continue
        if ckpt_t.dtype != model_p.dtype:
            failures.append(f"{key}: dtype ckpt={ckpt_t.dtype} model={model_p.dtype}")
            continue
        model_cpu = model_p.detach().cpu()
        ckpt_cpu = ckpt_t.cpu()
        if not torch.equal(model_cpu, ckpt_cpu):
            diff = float((model_cpu.float() - ckpt_cpu.float()).abs().max().item())
            failures.append(f"{key}: torch.equal=False max_diff={diff:.6e}")

    gate_log.assert_gate(
        f"G12_{ckpt_label}_exact_copy",
        not failures,
        f"{len(failures)} tensor(s) failed: " + "; ".join(failures[:5]),
        num_checked=len(saved_keys),
    )


# ---------------------------------------------------------------------------
# Verdict computation
# ---------------------------------------------------------------------------

def _compute_verdict(all_results: Dict[str, Dict],
                     gate_log: GateLog) -> Dict[str, Any]:
    if gate_log.failed:
        return {
            "objective_optimization": {
                "final_checkpoint_effect": {"verdict": "inconclusive"},
                "best_checkpoint_effect": {"verdict": "inconclusive"},
                "regression_from_best": {"verdict": "inconclusive"},
                "trajectory_pattern": {"verdict": "inconclusive"},
            },
            "objective_overfit": {"verdict": "not_assessed"},
            "visual_capacity": {"verdict": "not_assessed"},
            "grounding": {"verdict": "not_assessed"},
        }

    step000_mse = all_results["step_000"]["macro_avg_mse"]
    step400_mse = all_results["step_400"]["macro_avg_mse"]

    def _effect(mse: float, base: float) -> str:
        if mse < base * 0.80:
            return "effective"
        if mse < base * 0.95:
            return "partial"
        return "not_demonstrated"

    final_effect = _effect(step400_mse, step000_mse)
    rel_400 = (step400_mse - step000_mse) / step000_mse

    # Best checkpoint (tie rule: |a-b| < TIE_TOLERANCE)
    best_mse = min(r["macro_avg_mse"] for r in all_results.values())
    best_labels = [
        lbl for lbl, r in all_results.items()
        if abs(r["macro_avg_mse"] - best_mse) < TIE_TOLERANCE
    ]
    best_effect = _effect(best_mse, step000_mse)
    rel_best = (best_mse - step000_mse) / step000_mse

    # Regression from best
    # 'improved' is removed: step_400 IS best → regression_verdict = 'stable'
    # (step_400 cannot be strictly below best by definition of best)
    reg_ratio = step400_mse / best_mse if best_mse > 0 else float("inf")
    if reg_ratio > REGRESSION_THRESHOLD:
        regression_verdict = "regressed"
    else:
        regression_verdict = "stable"   # covers both step_400==best and <=best*1.10

    # Trajectory pattern (v6.1 simplified: no 'improved', no 'sustained_with_plateau')
    # step_400 <= best * 1.10 (stable) → sustained
    # step_400 >  best * 1.10 (regressed) → transient_then_regressed
    if gate_log.failed:
        traj = "inconclusive"
    elif best_effect == "not_demonstrated":
        traj = "no_clear_improvement"
    elif best_effect in ("effective", "partial"):
        if regression_verdict == "regressed":
            traj = "transient_then_regressed"
        else:
            traj = "sustained"   # step_400 is best or within 10% of best
    else:
        traj = "inconclusive"

    return {
        "objective_optimization": {
            "final_checkpoint_effect": {
                "verdict": final_effect,
                "step000_macro_avg_mse": step000_mse,
                "step400_macro_avg_mse": step400_mse,
                "rel_change_step000_to_step400": round(rel_400, 6),
            },
            "best_checkpoint_effect": {
                "verdict": best_effect,
                "best_checkpoint": best_labels,
                "best_macro_avg_mse": best_mse,
                "rel_change_step000_to_best": round(rel_best, 6),
                "tie_detected": len(best_labels) > 1,
            },
            "regression_from_best": {
                "verdict": regression_verdict,
                "step400_macro_avg_mse": step400_mse,
                "best_macro_avg_mse": best_mse,
                "ratio_step400_over_best": round(reg_ratio, 6),
                "threshold": REGRESSION_THRESHOLD,
            },
            "trajectory_pattern": {"verdict": traj},
            "per_pair_step400_vs_step000": {
                pid: {
                    "step000": all_results["step_000"]["per_pair_mse"][pid],
                    "step400": all_results["step_400"]["per_pair_mse"][pid],
                    "rel_change": round(
                        (all_results["step_400"]["per_pair_mse"][pid]
                         - all_results["step_000"]["per_pair_mse"][pid])
                        / all_results["step_000"]["per_pair_mse"][pid], 6
                    ),
                }
                for pid in PAIR_ORDER
            },
        },
        "objective_overfit": {"verdict": "not_assessed"},
        "visual_capacity": {"verdict": "not_assessed"},
        "grounding": {"verdict": "not_assessed"},
    }


# ---------------------------------------------------------------------------
# eval_config_frozen content
# ---------------------------------------------------------------------------

def _build_eval_config_frozen(output_dir: Path) -> Dict[str, Any]:
    return {
        "schema": "micro4_deterministic_eval_v6",
        "created_at_iso": _now_iso(),
        "output_dir": str(output_dir),
        "manifest": str(MICRO4_MANIFEST),
        "manifest_sha256": MICRO4_MANIFEST_SHA256,
        "eval_seed": EVAL_SEED,
        "noise_seed_base": NOISE_SEED_BASE,
        "max_canonical_scan_batches": MAX_CANONICAL_SCAN_BATCHES,
        "expected_trainable_tensors": EXPECTED_TRAINABLE_TENSORS,
        "expected_lora_b_tensors": EXPECTED_LORA_B_TENSORS,
        "legacy_step000_mse_ref": LEGACY_STEP000_MSE,
        "regression_threshold": REGRESSION_THRESHOLD,
        "tie_tolerance": TIE_TOLERANCE,
        "pair_order": PAIR_ORDER,
        "eval_checkpoints": [
            {"label": lbl, "path": str(p), "expected_sha256": CKPT_SHA256[lbl]}
            for lbl, p in EVAL_CHECKPOINTS
        ],
        "rng_fix_order": [
            "1. random.seed(EVAL_SEED)  -- Python global RNG (does NOT affect "
            "CounterfactualCoTDataset.set_epoch which uses random.Random instance)",
            "2. np.random.seed(EVAL_SEED)  -- NumPy global RNG; controls "
            "pack_sequence np.random.randn() timestep; safe because num_workers=0",
            "3. torch.manual_seed(EVAL_SEED)  -- Torch CPU RNG",
            "4. torch.cuda.manual_seed_all(EVAL_SEED)  -- Torch CUDA RNG all devices",
        ],
        "num_workers": 0,
        "num_workers_note": (
            "hardcoded in _build_gold40_loader (PackedDataset num_workers=0, "
            "DataLoader num_workers=0); NumPy RNG runs entirely in main process; "
            "no worker-fork seed isolation needed"
        ),
        "random_seed_note": (
            "random.seed(EVAL_SEED) sets the Python global RNG. "
            "CounterfactualCoTDataset.set_epoch() uses random.Random("
            "training_seed + epoch) which is an INSTANCE-level RNG object, "
            "independent of the global random module state. "
            "The global seed call does NOT affect set_epoch() shuffle order; "
            "shuffle is controlled by _build_gold40_loader seed_offset=EVAL_SEED "
            "passed to packed_dataset.set_epoch(args.seed + seed_offset)."
        ),
        "packed_timesteps_note": (
            "packed_timesteps in batch = raw np.random.randn() scalar, "
            "broadcast to all latent tokens of that vae_image. "
            "Inside bagel.py:Bagel.forward() it is transformed: "
            "t_sig = sigmoid(packed_timesteps); "
            "t_shift = shift*t_sig/(1+(shift-1)*t_sig). "
            "MSE target uses t_shift, NOT the raw randn value."
        ),
        "kv_cache_note": (
            "model.train() routes to Qwen2Model.forward_train() which has no "
            "past_key_values/use_cache parameter. No KV cache is allocated."
        ),
        "optimization_verdict_criteria": {
            "schema_version": "micro4_opt_verdict_v5",
            "frozen_before_run": True,
            "objective_optimization": {
                "final_checkpoint_effect": {
                    "effective": "step_400 < step_000 * 0.80",
                    "partial": "step_000*0.80 <= step_400 < step_000*0.95",
                    "not_demonstrated": "step_400 >= step_000*0.95",
                    "inconclusive": "any integrity gate failed",
                },
                "best_checkpoint_effect": {
                    "effective": "best < step_000 * 0.80",
                    "partial": "step_000*0.80 <= best < step_000*0.95",
                    "not_demonstrated": "best >= step_000*0.95",
                    "inconclusive": "any integrity gate failed",
                    "tie_rule": f"abs(mse_a - mse_b) < {TIE_TOLERANCE} => tied",
                },
                "regression_from_best": {
                    "regressed": f"step_400 > best * {REGRESSION_THRESHOLD}",
                    "stable": (
                        f"step_400 <= best * {REGRESSION_THRESHOLD} "
                        "(covers step_400==best and step_400 within 10% of best)"
                    ),
                    "inconclusive": "any integrity gate failed",
                    "note": (
                        "'improved' removed: step_400 cannot be strictly below best "
                        "by definition; step_400==best is classified as 'stable'."
                    ),
                },
                "trajectory_pattern": {
                    "sustained": (
                        "best_effect in {effective,partial} AND "
                        f"step_400 <= best * {REGRESSION_THRESHOLD} "
                        "(step_400 is best or within 10% of best)"
                    ),
                    "transient_then_regressed": (
                        "best_effect in {effective,partial} AND "
                        f"step_400 > best * {REGRESSION_THRESHOLD}"
                    ),
                    "no_clear_improvement": "best_effect == not_demonstrated",
                    "inconclusive": "any integrity gate failed or no pattern matches",
                    "note": (
                        "'sustained_with_plateau' removed: step_400 <= best*1.10 "
                        "is now uniformly classified as 'sustained'."
                    ),
                },
            },
            "objective_overfit": {"current_value": "not_assessed"},
            "visual_capacity": {"current_value": "not_assessed"},
            "grounding": {"current_value": "not_assessed"},
        },
    }


# ---------------------------------------------------------------------------
# Frozen model asset paths (under /dockerdata)
# ---------------------------------------------------------------------------
MODEL_DIR = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
MODEL_EMA  = MODEL_DIR / "ema.safetensors"
MODEL_AE   = MODEL_DIR / "ae.safetensors"
MODEL_CONFIG       = MODEL_DIR / "config.json"
MODEL_LLM_CONFIG   = MODEL_DIR / "llm_config.json"
MODEL_VIT_CONFIG   = MODEL_DIR / "vit_config.json"
MODEL_TOKENIZER    = MODEL_DIR / "tokenizer.json"
MODEL_TOKENIZER_CFG = MODEL_DIR / "tokenizer_config.json"

# Frozen SHA256 for model assets (from交接文档)
MODEL_EMA_SHA256 = "0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3"
MODEL_AE_SHA256  = "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"

# Frozen manifest image paths (resolved from manifest final_image_root + image)
MANIFEST_IMAGES: List[Dict[str, str]] = [
    {
        "pair_id": "cf_source_city_street_object_replace_0000",
        "source": ("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
                   "images_gold_nanobanana_square/source/"
                   "source_city_street_object_replace_c82ecffb2e42.png"),
        "target": ("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
                   "images_gold_nanobanana_square/target/"
                   "cf_source_city_street_object_replace_0000.png"),
    },
    {
        "pair_id": "cf_source_classroom_object_replace_0000",
        "source": ("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
                   "images_gold_nanobanana_square/source/"
                   "source_classroom_object_replace_b036f640cf2e.png"),
        "target": ("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
                   "images_gold_nanobanana_square/target/"
                   "cf_source_classroom_object_replace_0000.png"),
    },
    {
        "pair_id": "cf_source_city_street_absolute_position_change_v4_absolute_region_review_fix_v6_short_0000",
        "source": ("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
                   "reruns/nanobanana_v4_absolute_region/images/source_regen/source/"
                   "source_city_street_absolute_position_change_v4_review_fix_v6_short_77af4c9a951f.png"),
        "target": ("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
                   "reruns/nanobanana_v4_absolute_region/images/source_regen/target/"
                   "cf_source_city_street_absolute_position_change_v4_absolute_region_review_fix_v6_short_0000.png"),
    },
    {
        "pair_id": "cf_source_classroom_absolute_position_change_v4_absolute_region_replacement_v4_0000",
        "source": ("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
                   "reruns/nanobanana_v4_absolute_region/images/source_regen/source/"
                   "source_classroom_absolute_position_change_v4_ed8fc97ea040.png"),
        "target": ("/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
                   "reruns/nanobanana_v4_absolute_region/images/source_regen/target/"
                   "cf_source_classroom_absolute_position_change_v4_absolute_region_replacement_v4_0000.png"),
    },
]

# Key source code paths consumed by eval
CODE_PATHS: List[str] = [
    str(BAGEL_ROOT / "experiments/counterfactual_cot/training/gold40_overfit.py"),
    str(BAGEL_ROOT / "experiments/counterfactual_cot/training/gold40_lora.py"),
    str(BAGEL_ROOT / "experiments/counterfactual_cot/training/lora_2step_smoke.py"),
    str(BAGEL_ROOT / "experiments/counterfactual_cot/training/smoke_train.py"),
    str(BAGEL_ROOT / "experiments/counterfactual_cot/datasets/bagel_manifest_dataset.py"),
    str(BAGEL_ROOT / "data/dataset_base.py"),
    str(BAGEL_ROOT / "modeling/bagel/bagel.py"),
    str(BAGEL_ROOT / "modeling/bagel/qwen2_navit.py"),
    str(BAGEL_ROOT / "modeling/autoencoder.py"),
]


# ---------------------------------------------------------------------------
# Preflight (static + path checks, no model load, no CUDA)
# ---------------------------------------------------------------------------

def run_preflight(preflight_dir: Path, eval_dir: Path) -> int:
    """Static/path preflight.  Returns 0 on success, 1 on failure.

    Writes preflight_report.json to preflight_dir (separate from eval_dir).
    Does NOT touch eval_dir.
    """
    report: Dict[str, Any] = {
        "mode": "preflight",
        "schema": "micro4_deterministic_eval_v6_preflight",
        "started_at": _now_iso(),
        "script_path": str(Path(__file__).resolve()),
        "bagel_root": str(BAGEL_ROOT),
        "preflight_output_dir": str(preflight_dir),
        "eval_output_dir": str(eval_dir),
        "checks": [],
        "paths_to_read": [],
        "paths_to_write": [],
    }
    ok = True

    def _chk(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        status = "passed" if passed else "FAILED"
        if not passed:
            ok = False
        report["checks"].append({"name": name, "status": status, "detail": detail})
        print(f"  [{status:6s}] {name}" + (f": {detail}" if detail else ""))

    print("=== micro4_deterministic_eval.py  preflight ===")

    # --- Directory separation ---
    _chk("preflight_dir_under_group",
         str(preflight_dir).startswith("/group"),
         str(preflight_dir))
    _chk("eval_dir_under_group",
         str(eval_dir).startswith("/group"),
         str(eval_dir))
    _chk("preflight_dir_differs_from_eval_dir",
         preflight_dir.resolve() != eval_dir.resolve(),
         f"preflight={preflight_dir} eval={eval_dir}")
    # Preflight dir may already exist (re-runnable).
    # Eval dir emptiness is enforced by --run-eval, not by preflight.
    # Preflight only pre-registers the eval dir path; it does NOT require it to be empty.
    eval_dir_empty = not eval_dir.exists() or not any(eval_dir.iterdir())
    report["checks"].append({
        "name": "eval_dir_empty_status",
        "status": "info",
        "detail": (
            f"eval_dir={eval_dir}  "
            f"{'empty or non-existent (ready for --run-eval)' if eval_dir_empty else 'NOT empty (--run-eval will refuse; clear before running eval)'}"
        ),
    })
    print(f"  [info  ] eval_dir_empty_status: {eval_dir}  "
          f"{'empty/non-existent' if eval_dir_empty else 'NOT empty (clear before --run-eval)'}")

    # --- Manifest ---
    _chk("manifest_exists", MICRO4_MANIFEST.exists(), str(MICRO4_MANIFEST))
    if MICRO4_MANIFEST.exists():
        sha = _sha256_file(MICRO4_MANIFEST)
        _chk("manifest_sha256", sha == MICRO4_MANIFEST_SHA256,
             f"got={sha} expected={MICRO4_MANIFEST_SHA256}")

    # --- Checkpoints ---
    for label, ckpt_path in EVAL_CHECKPOINTS:
        _chk(f"ckpt_exists_{label}", ckpt_path.exists(), str(ckpt_path))
        if ckpt_path.exists():
            sha = _sha256_file(ckpt_path)
            _chk(f"ckpt_sha256_{label}",
                 sha == CKPT_SHA256[label],
                 f"got={sha} expected={CKPT_SHA256[label]}")

    # --- checkpoint_manifest.jsonl ---
    _chk("ckpt_manifest_exists", CKPT_MANIFEST_PATH.exists(),
         str(CKPT_MANIFEST_PATH))

    # --- Pair order sanity ---
    _chk("pair_order_length_4", len(PAIR_ORDER) == 4, str(len(PAIR_ORDER)))
    _chk("pair_order_unique", len(set(PAIR_ORDER)) == 4, "")

    # --- Python imports (no model load) ---
    try:
        from experiments.counterfactual_cot.data.schema import load_jsonl  # noqa
        _chk("import_schema", True)
    except Exception as e:
        _chk("import_schema", False, str(e))

    try:
        from experiments.counterfactual_cot.training.gold40_overfit import (  # noqa
            _build_gold40_loader, _forward_loss, _prepare_batch,
        )
        _chk("import_gold40_overfit", True)
    except Exception as e:
        _chk("import_gold40_overfit", False, str(e))

    try:
        from experiments.counterfactual_cot.training.lora_2step_smoke import (  # noqa
            load_trainable_strict,
        )
        _chk("import_lora_2step_smoke", True)
    except Exception as e:
        _chk("import_lora_2step_smoke", False, str(e))

    try:
        from experiments.counterfactual_cot.training.gold40_lora import (  # noqa
            _build_model_with_lora,
        )
        _chk("import_gold40_lora", True)
    except Exception as e:
        _chk("import_gold40_lora", False, str(e))

    try:
        import torch  # noqa
        _chk("import_torch", True, f"version={torch.__version__}")
    except Exception as e:
        _chk("import_torch", False, str(e))

    try:
        import numpy as np_  # noqa
        _chk("import_numpy", True, f"version={np_.__version__}")
    except Exception as e:
        _chk("import_numpy", False, str(e))

    # --- Manifest records ---
    if MICRO4_MANIFEST.exists():
        try:
            from experiments.counterfactual_cot.data.schema import load_jsonl
            records = load_jsonl(MICRO4_MANIFEST)
            _chk("manifest_record_count_4", len(records) == 4,
                 f"got {len(records)}")
            found_ids = [r.get("pair_id", "") for r in records]
            missing_ids = [p for p in PAIR_ORDER if p not in found_ids]
            _chk("manifest_contains_all_pair_ids",
                 not missing_ids, f"missing={missing_ids}")
        except Exception as e:
            _chk("manifest_parse", False, str(e))

    # --- Manifest image existence (recursive verification of 4 source+target pairs) ---
    for img_entry in MANIFEST_IMAGES:
        pid_short = img_entry["pair_id"][:40]
        for side in ("source", "target"):
            img_path = Path(img_entry[side])
            exists = img_path.exists()
            size_bytes = img_path.stat().st_size if exists else 0
            _chk(f"manifest_image_{side}_{pid_short}",
                 exists and size_bytes > 0,
                 f"{img_path}  size={size_bytes}")

    # --- Model asset existence (under /dockerdata) ---
    _chk("model_dir_exists", MODEL_DIR.exists(), str(MODEL_DIR))
    for asset_path, frozen_sha, label in [
        (MODEL_EMA,  MODEL_EMA_SHA256,  "ema.safetensors"),
        (MODEL_AE,   MODEL_AE_SHA256,   "ae.safetensors"),
    ]:
        exists = asset_path.exists()
        size_bytes = asset_path.stat().st_size if exists else 0
        _chk(f"model_asset_exists_{label}",
             exists and size_bytes > 0,
             f"{asset_path}  size={size_bytes}")
        if exists:
            _chk(f"model_asset_sha256_{label}",
                 _sha256_file(asset_path) == frozen_sha,
                 f"expected={frozen_sha}")
    for cfg_path, label in [
        (MODEL_CONFIG,        "config.json"),
        (MODEL_LLM_CONFIG,    "llm_config.json"),
        (MODEL_VIT_CONFIG,    "vit_config.json"),
        (MODEL_TOKENIZER,     "tokenizer.json"),
        (MODEL_TOKENIZER_CFG, "tokenizer_config.json"),
    ]:
        exists = cfg_path.exists()
        size_bytes = cfg_path.stat().st_size if exists else 0
        _chk(f"model_config_exists_{label}",
             exists and size_bytes > 0,
             f"{cfg_path}  size={size_bytes}")

    # --- Code path existence ---
    for code_path in CODE_PATHS:
        p = Path(code_path)
        _chk(f"code_exists_{p.name}", p.exists(), code_path)

    # --- Paths to read (extended) ---
    paths_to_read_structured = {
        "manifest": str(MICRO4_MANIFEST),
        "checkpoint_manifest": str(CKPT_MANIFEST_PATH),
        "checkpoints": [str(p) for _, p in EVAL_CHECKPOINTS],
        "manifest_images": [
            {"pair_id": e["pair_id"], "source": e["source"], "target": e["target"]}
            for e in MANIFEST_IMAGES
        ],
        "model_assets": [
            {"path": str(MODEL_EMA),  "frozen_sha256": MODEL_EMA_SHA256,
             "size_bytes": MODEL_EMA.stat().st_size if MODEL_EMA.exists() else None},
            {"path": str(MODEL_AE),   "frozen_sha256": MODEL_AE_SHA256,
             "size_bytes": MODEL_AE.stat().st_size if MODEL_AE.exists() else None},
            {"path": str(MODEL_CONFIG),        "frozen_sha256": None,
             "size_bytes": MODEL_CONFIG.stat().st_size if MODEL_CONFIG.exists() else None},
            {"path": str(MODEL_LLM_CONFIG),    "frozen_sha256": None,
             "size_bytes": MODEL_LLM_CONFIG.stat().st_size if MODEL_LLM_CONFIG.exists() else None},
            {"path": str(MODEL_VIT_CONFIG),    "frozen_sha256": None,
             "size_bytes": MODEL_VIT_CONFIG.stat().st_size if MODEL_VIT_CONFIG.exists() else None},
            {"path": str(MODEL_TOKENIZER),     "frozen_sha256": None,
             "size_bytes": MODEL_TOKENIZER.stat().st_size if MODEL_TOKENIZER.exists() else None},
            {"path": str(MODEL_TOKENIZER_CFG), "frozen_sha256": None,
             "size_bytes": MODEL_TOKENIZER_CFG.stat().st_size if MODEL_TOKENIZER_CFG.exists() else None},
        ],
        "code_paths": CODE_PATHS,
    }
    report["paths_to_read"] = paths_to_read_structured
    # Flat list for backward compatibility
    report["paths_to_read_flat"] = (
        [str(MICRO4_MANIFEST), str(CKPT_MANIFEST_PATH)]
        + [str(p) for _, p in EVAL_CHECKPOINTS]
        + [e["source"] for e in MANIFEST_IMAGES]
        + [e["target"] for e in MANIFEST_IMAGES]
        + [str(MODEL_EMA), str(MODEL_AE), str(MODEL_CONFIG),
           str(MODEL_LLM_CONFIG), str(MODEL_VIT_CONFIG),
           str(MODEL_TOKENIZER), str(MODEL_TOKENIZER_CFG)]
        + CODE_PATHS
    )

    # --- Paths to write ---
    report["paths_to_write"] = {
        "preflight_dir": [
            str(preflight_dir / "preflight_report.json"),
        ],
        "eval_dir": [
            str(eval_dir / f)
            for f in [
                "eval_config_frozen.json",
                "canonical_inputs_manifest.json",
                "strict_load_records.json",
                "integrity_gates.json",
                "per_pair_mse_table.json",
                "macro_avg_table.json",
                "optimization_verdict.json",
                "eval_summary.json",
                "eval_log.jsonl",
            ]
        ],
    }

    # --- Finalize ---
    report["finished_at"] = _now_iso()
    report["overall_status"] = "passed" if ok else "FAILED"

    preflight_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = preflight_dir / "preflight_report.json"
    _atomic_write_json(preflight_path, report)
    sha = _sha256_file(preflight_path)

    print(f"\nPreflight {'PASSED' if ok else 'FAILED'}")
    print(f"Report   : {preflight_path}")
    print(f"SHA256   : {sha}")
    print(f"Checks   : {len(report['checks'])}")
    print(f"Eval dir (pre-registered, currently empty): {eval_dir}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Full deterministic eval (requires --run-eval)
# ---------------------------------------------------------------------------

def run_eval(output_dir: Path) -> int:
    """Full deterministic eval.  Returns 0 on success, non-zero on failure."""
    import torch

    gate_log = GateLog()
    all_results: Dict[str, Dict] = {}
    strict_load_records: Dict[str, Dict] = {}
    eval_log_path = output_dir / "eval_log.jsonl"

    def _log(event: str, **kw: Any) -> None:
        _append_jsonl(eval_log_path, {"event": event, "ts": _now_iso(), **kw})

    try:
        # ── Phase 0: pre-run gates ──────────────────────────────────────────

        # G_dir: non-empty output directory → refuse
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                f"G_dir FAIL: output directory is not empty: {output_dir}\n"
                "Remove or rename it before running eval."
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        # G1: manifest SHA256
        gate_log.assert_gate(
            "G1_manifest_exists", MICRO4_MANIFEST.exists(),
            f"manifest missing: {MICRO4_MANIFEST}",
        )
        gate_log.assert_gate(
            "G1_manifest_sha256",
            _sha256_file(MICRO4_MANIFEST) == MICRO4_MANIFEST_SHA256,
            f"manifest SHA256 mismatch",
        )

        # G2: all checkpoint files exist
        for label, ckpt_path in EVAL_CHECKPOINTS:
            gate_log.assert_gate(
                f"G2_ckpt_exists_{label}", ckpt_path.exists(),
                f"checkpoint missing: {ckpt_path}",
            )

        # G3: checkpoint SHA256
        for label, ckpt_path in EVAL_CHECKPOINTS:
            sha = _sha256_file(ckpt_path)
            gate_log.assert_gate(
                f"G3_ckpt_sha256_{label}",
                sha == CKPT_SHA256[label],
                f"SHA mismatch: got={sha} expected={CKPT_SHA256[label]}",
            )

        # G_config: write eval_config_frozen.json and seal SHA
        eval_config = _build_eval_config_frozen(output_dir)
        config_path = output_dir / "eval_config_frozen.json"
        _atomic_write_json(config_path, eval_config)
        g_config_sha256 = _sha256_file(config_path)
        gate_log.record("G_config", "frozen", sha256=g_config_sha256,
                        path=str(config_path))
        _log("phase0_done", g_config_sha256=g_config_sha256)

        # ── Phase 1: build args + load model ────────────────────────────────
        import argparse as _argparse
        from experiments.counterfactual_cot.training.gold40_lora import (
            _build_model_with_lora,
        )
        from experiments.counterfactual_cot.training.lora_2step_smoke import (
            load_trainable_strict,
        )
        from experiments.counterfactual_cot.training.gold40_overfit import (
            _build_gold40_loader, _forward_loss, _prepare_batch,
        )
        from experiments.counterfactual_cot.training.smoke_train import (
            _device, _load_tokenizer,
        )

        # Build a minimal args Namespace matching the micro4 training config
        # (values taken from gold40_v3_lora_micro4_400steps/run_config_resolved.json)
        args = _argparse.Namespace(
            llm_path="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT",
            vae_path="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors",
            checkpoint_path="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors",
            precision="bf16",
            device="cuda",
            model_device_map="single",
            device_map_max_memory="34GiB",
            lora_layers=[24, 25, 26, 27],
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            trainable_name_regex=(
                r"^(time_embedder|latent_pos_embed|vit_pos_embed|vae2llm|llm2vae|connector)\."
            ),
            image_size=256,
            vit_image_size=224,
            vit_min_image_size=224,
            expected_num_tokens=512,
            max_num_tokens_per_sample=1536,
            max_num_tokens=1536,
            seed=1234,
            variant="paired_cf_mvp",
            num_samples=4,
            require_paired_vit=True,
        )
        device = _device(args.device)
        tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)

        # Build model with LoRA (loads base weights from /dockerdata)
        model, vae_model, _wrappers, _target_names, _base_ckpt_report = \
            _build_model_with_lora(
                args, device,
                tokenizer_size=len(tokenizer),
                num_new_tokens=num_new_tokens,
            )

        # Freeze base parameters: set_trainable() marks only the 62 trainable
        # tensors (14 existing + 24 lora_A + 24 lora_B) as requires_grad=True.
        # This MUST happen before load_trainable_strict, which uses
        # {name for name, p in model.named_parameters() if p.requires_grad}
        # to determine the expected key set.
        from experiments.counterfactual_cot.training.gold40_lora import (
            _apply_trainable_and_audit,
        )
        _param_audit = _apply_trainable_and_audit(
            model, _wrappers, _target_names, args, device.type
        )
        _log("phase1_set_trainable_done",
             total_trainable=_param_audit["total_trainable_parameters"],
             existing_tensors=_param_audit["existing_tensors"],
             lora_tensors=_param_audit["lora_tensors"])

        # Load step_000 for G4/G5
        load_result_s0 = load_trainable_strict(EVAL_CHECKPOINTS[0][1], model)

        # G4: trainable tensor count
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        gate_log.assert_gate(
            "G4_trainable_count",
            len(trainable_names) == EXPECTED_TRAINABLE_TENSORS,
            f"got {len(trainable_names)}, expected {EXPECTED_TRAINABLE_TENSORS}",
        )

        # G5: lora_B all zero
        lora_b_nonzero = [
            n for n, p in model.named_parameters()
            if p.requires_grad and n.endswith(".lora_B")
            and float(p.detach().abs().max().item()) > 0.0
        ]
        gate_log.assert_gate(
            "G5_lora_b_all_zero",
            not lora_b_nonzero,
            f"non-zero lora_B: {lora_b_nonzero[:3]}",
        )

        # G12 for step_000 (explicit max_abs_load_copy_diff == 0)
        gate_log.assert_gate(
            "G12_step000_max_abs_diff_zero",
            load_result_s0["max_abs_load_copy_diff"] == 0.0,
            f"max_abs_load_copy_diff={load_result_s0['max_abs_load_copy_diff']}",
        )
        _gate_g12_per_tensor(gate_log, EVAL_CHECKPOINTS[0][1], model, "step_000")
        strict_load_records["step_000"] = {
            "load_trainable_strict": load_result_s0,
            "per_tensor_exact_copy": {"status": "passed", "num_keys_checked": 62},
        }
        _log("phase1_done")

        # ── Phase 2: fix RNG + build canonical eval inputs ──────────────────
        random.seed(EVAL_SEED)
        np.random.seed(EVAL_SEED)
        torch.manual_seed(EVAL_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(EVAL_SEED)

        loader = _build_gold40_loader(
            manifest_path=MICRO4_MANIFEST,
            tokenizer=tokenizer,
            special_tokens=special_tokens,
            args=args,
            seed_offset=EVAL_SEED,
        )
        iterator = iter(loader)

        collected: Dict[str, Any] = {}
        scan_positions: Dict[str, int] = {}
        scan_count = 0
        while len(collected) < 4:
            if scan_count >= MAX_CANONICAL_SCAN_BATCHES:
                raise RuntimeError(
                    f"G6 FAIL: scanned {scan_count} batches, "
                    f"only collected {len(collected)}/4 pairs"
                )
            try:
                raw_batch = next(iterator)
            except StopIteration:
                raise RuntimeError(
                    f"G6 FAIL: loader exhausted after {scan_count} batches, "
                    f"only collected {len(collected)}/4 pairs"
                )
            assert len(raw_batch.batch_data_indexes) == 1, \
                f"singlepack assertion: {len(raw_batch.batch_data_indexes)} samples"
            assert len(raw_batch.sample_lens) == 1, \
                f"sample_lens singlepack: {len(raw_batch.sample_lens)}"
            pid = raw_batch.batch_data_indexes[0]["pair_id"]
            if pid not in collected:
                collected[pid] = raw_batch
                scan_positions[pid] = scan_count
            scan_count += 1

        gate_log.assert_gate(
            "G6_all_pairs_collected",
            set(collected.keys()) == set(PAIR_ORDER),
            f"collected={set(collected.keys())} expected={set(PAIR_ORDER)}",
            scan_batches_used=scan_count,
        )

        # Build canonical inputs (VAE encode on GPU)
        device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        canonical: Dict[str, Dict] = {}
        for pair_idx, pair_id in enumerate(PAIR_ORDER):
            data = _prepare_batch(collected[pair_id], vae_model, device, model_dtype)
            field_digest = _recursive_digest(data)
            canonical[pair_id] = {
                "pair_id": pair_id,
                "pair_idx": pair_idx,
                "data": data,
                "noise_seed": NOISE_SEED_BASE + pair_idx * 1000,
                "field_digest": field_digest,
                "scan_batch_position": scan_positions[pair_id],
            }

        _atomic_write_json(
            output_dir / "canonical_inputs_manifest.json",
            {
                "schema": "micro4_canonical_inputs_v2",
                "created_at_iso": _now_iso(),
                "eval_seed": EVAL_SEED,
                "noise_seed_base": NOISE_SEED_BASE,
                "scan_batches_used": scan_count,
                "pairs": [
                    {
                        "pair_id": canonical[pid]["pair_id"],
                        "pair_idx": canonical[pid]["pair_idx"],
                        "noise_seed": canonical[pid]["noise_seed"],
                        "scan_batch_position": canonical[pid]["scan_batch_position"],
                        "field_digest_keys": sorted(canonical[pid]["field_digest"].keys()),
                    }
                    for pid in PAIR_ORDER
                ],
            },
        )
        _log("phase2_done", scan_batches_used=scan_count)

        # ── Phase 3: step_000 double-run G10 (with G11 + G13 protection) ────
        def _eval_one(label: str) -> Dict[str, float]:
            model.train()
            assert model.training, "model must be in train() mode"
            per_pair: Dict[str, float] = {}
            with torch.no_grad():
                for pid in PAIR_ORDER:
                    ci = canonical[pid]
                    torch.manual_seed(ci["noise_seed"])
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(ci["noise_seed"])
                    _, scalars = _forward_loss(model, ci["data"])
                    mse_val = scalars["mse"]
                    # Immediate finite gate (G8-instant)
                    assert math.isfinite(mse_val) and mse_val > 0, \
                        f"G8_instant FAIL [{label}][{pid}]: mse={mse_val}"
                    per_pair[pid] = mse_val
            return per_pair

        # G11 + G13 before run1
        _check_g11(gate_log, canonical, "pre_run1", "step_000")
        model.train()
        buffers_pre_run1 = _snapshot_named_buffers(model)

        run1 = _eval_one("step_000_run1")

        _check_g11(gate_log, canonical, "post_run1", "step_000")
        _check_g13(gate_log, model, buffers_pre_run1, "step_000_run1")

        # G11 + G13 before run2
        _check_g11(gate_log, canonical, "pre_run2", "step_000")
        model.train()
        buffers_pre_run2 = _snapshot_named_buffers(model)

        run2 = _eval_one("step_000_run2")

        _check_g11(gate_log, canonical, "post_run2", "step_000")
        _check_g13(gate_log, model, buffers_pre_run2, "step_000_run2")

        # G10: bit-exact
        for pid in PAIR_ORDER:
            diff = abs(run1[pid] - run2[pid])
            gate_log.assert_gate(
                f"G10_step000_bitexact_{pid[:30]}",
                diff == 0.0,
                f"run1={run1[pid]} run2={run2[pid]} diff={diff}",
            )

        # step_000 MSE > 0 gate
        for pid in PAIR_ORDER:
            gate_log.assert_gate(
                f"G_step000_mse_positive_{pid[:30]}",
                run1[pid] > 0,
                f"step_000 mse={run1[pid]} must be > 0",
            )

        _log("phase3_done", g10_passed=True)

        # ── Phase 4: all 8 checkpoints ──────────────────────────────────────
        for ckpt_label, ckpt_path in EVAL_CHECKPOINTS:

            # G12a: load_trainable_strict
            load_result = load_trainable_strict(ckpt_path, model)
            gate_log.assert_gate(
                f"G12_{ckpt_label}_num_loaded",
                load_result["num_loaded_tensors"] == EXPECTED_TRAINABLE_TENSORS,
                f"got {load_result['num_loaded_tensors']}",
            )
            gate_log.assert_gate(
                f"G12_{ckpt_label}_keys_match",
                load_result["keys_expected"] == load_result["keys_in_ckpt"] == EXPECTED_TRAINABLE_TENSORS,
                f"keys_expected={load_result['keys_expected']} keys_in_ckpt={load_result['keys_in_ckpt']}",
            )
            # G12 explicit max_abs_load_copy_diff == 0
            gate_log.assert_gate(
                f"G12_{ckpt_label}_max_abs_diff_zero",
                load_result["max_abs_load_copy_diff"] == 0.0,
                f"max_abs_load_copy_diff={load_result['max_abs_load_copy_diff']}",
            )

            # G12b: per-tensor exact copy
            _gate_g12_per_tensor(gate_log, ckpt_path, model, ckpt_label)
            strict_load_records[ckpt_label] = {
                "load_trainable_strict": load_result,
                "per_tensor_exact_copy": {"status": "passed", "num_keys_checked": 62},
            }

            # G11 pre + G13 pre
            _check_g11(gate_log, canonical, "pre", ckpt_label)
            model.train()
            buffers_pre = _snapshot_named_buffers(model)

            per_pair_mse = _eval_one(ckpt_label)

            # G11 post + G13 post
            _check_g11(gate_log, canonical, "post", ckpt_label)
            _check_g13(gate_log, model, buffers_pre, ckpt_label)

            # G14: step_000 third run vs G10 run1
            if ckpt_label == "step_000":
                for pid in PAIR_ORDER:
                    diff = abs(per_pair_mse[pid] - run1[pid])
                    gate_log.assert_gate(
                        f"G14_step000_vs_run1_{pid[:30]}",
                        diff == 0.0,
                        f"phase4={per_pair_mse[pid]} run1={run1[pid]} diff={diff}",
                    )

            macro_avg = sum(per_pair_mse.values()) / 4
            step000_macro = all_results.get("step_000", {}).get("macro_avg_mse", macro_avg)
            rel_change = (macro_avg - step000_macro) / step000_macro if step000_macro > 0 else 0.0

            all_results[ckpt_label] = {
                "per_pair_mse": per_pair_mse,
                "macro_avg_mse": macro_avg,
                "rel_change_vs_step000": round(rel_change, 6),
            }
            _log("ckpt_eval_done", ckpt_label=ckpt_label,
                 macro_avg_mse=macro_avg, per_pair_mse=per_pair_mse)

        # G8: all 32 MSE values finite and positive
        for lbl, res in all_results.items():
            for pid, mse in res["per_pair_mse"].items():
                gate_log.assert_gate(
                    f"G8_{lbl}_{pid[:30]}_finite_positive",
                    math.isfinite(mse) and mse > 0,
                    f"mse={mse}",
                )

        # G_step000_best_positive
        best_mse = min(r["macro_avg_mse"] for r in all_results.values())
        gate_log.assert_gate(
            "G_best_mse_positive",
            best_mse > 0,
            f"best_mse={best_mse}",
        )

        # G9: ref only
        g9_diff = abs(all_results["step_000"]["macro_avg_mse"] - LEGACY_STEP000_MSE)
        gate_log.record("G9_legacy_mse_ref", "ref_only",
                        diff=g9_diff, legacy=LEGACY_STEP000_MSE,
                        new=all_results["step_000"]["macro_avg_mse"])

        _log("phase4_done")

        # ── Phase 5: recheck eval_config_frozen.json SHA, write outputs ─────
        current_config_sha = _sha256_file(config_path)
        gate_log.assert_gate(
            "G_config_recheck",
            current_config_sha == g_config_sha256,
            f"eval_config_frozen.json tampered: "
            f"original={g_config_sha256} current={current_config_sha}",
        )

        # Compute verdict
        verdict = _compute_verdict(all_results, gate_log)
        verdict["integrity"] = {
            "g8_passed": not any(
                g["gate_id"].startswith("G8") and g["status"] == "FAILED"
                for g in gate_log.records
            ),
            "g9_ref_diff": g9_diff,
            "g10_passed": not any(
                g["gate_id"].startswith("G10") and g["status"] == "FAILED"
                for g in gate_log.records
            ),
            "g11_passed": not any(
                g["gate_id"].startswith("G11") and g["status"] == "FAILED"
                for g in gate_log.records
            ),
            "g12_all_passed": not any(
                g["gate_id"].startswith("G12") and g["status"] == "FAILED"
                for g in gate_log.records
            ),
            "g13_passed": not any(
                g["gate_id"].startswith("G13") and g["status"] == "FAILED"
                for g in gate_log.records
            ),
            "g14_passed": not any(
                g["gate_id"].startswith("G14") and g["status"] == "FAILED"
                for g in gate_log.records
            ),
            "g_config_sha256": g_config_sha256,
            "g_config_status": "frozen",
            "g_config_recheck": "passed",
        }
        verdict["criteria_ref"] = "eval_config_frozen.json:optimization_verdict_criteria"

        # per_pair_mse_table
        _atomic_write_json(
            output_dir / "per_pair_mse_table.json",
            {
                "schema": "micro4_per_pair_mse_v1",
                "pair_order": PAIR_ORDER,
                "rows": [
                    {
                        "checkpoint": lbl,
                        "step": int(lbl.split("_")[1]),
                        "per_pair_mse": res["per_pair_mse"],
                        "macro_avg_mse": res["macro_avg_mse"],
                        "rel_change_vs_step000": res["rel_change_vs_step000"],
                    }
                    for lbl, res in all_results.items()
                ],
            },
        )

        # macro_avg_table
        _atomic_write_json(
            output_dir / "macro_avg_table.json",
            {
                "step000_macro_avg_mse": all_results["step_000"]["macro_avg_mse"],
                "table": [
                    {
                        "step": int(lbl.split("_")[1]),
                        "checkpoint": lbl,
                        "macro_avg_mse": res["macro_avg_mse"],
                        "rel_change_vs_step000": res["rel_change_vs_step000"],
                    }
                    for lbl, res in all_results.items()
                ],
            },
        )

        _atomic_write_json(output_dir / "optimization_verdict.json", verdict)
        _atomic_write_json(output_dir / "strict_load_records.json", strict_load_records)
        _atomic_write_json(output_dir / "integrity_gates.json", gate_log.to_dict())

        best_mse_val = verdict["objective_optimization"]["best_checkpoint_effect"]["best_macro_avg_mse"]
        _atomic_write_json(
            output_dir / "eval_summary.json",
            {
                "schema": "micro4_eval_summary_v1",
                "finished_at": _now_iso(),
                "overall_status": "passed",
                "step000_macro_avg_mse": all_results["step_000"]["macro_avg_mse"],
                "step400_macro_avg_mse": all_results["step_400"]["macro_avg_mse"],
                "best_checkpoint": verdict["objective_optimization"]["best_checkpoint_effect"]["best_checkpoint"],
                "best_macro_avg_mse": best_mse_val,
                "trajectory_pattern": verdict["objective_optimization"]["trajectory_pattern"]["verdict"],
                "final_checkpoint_effect": verdict["objective_optimization"]["final_checkpoint_effect"]["verdict"],
                "g10_passed": verdict["integrity"]["g10_passed"],
                "g11_passed": verdict["integrity"]["g11_passed"],
                "g12_all_passed": verdict["integrity"]["g12_all_passed"],
                "g13_passed": verdict["integrity"]["g13_passed"],
                "g14_passed": verdict["integrity"]["g14_passed"],
            },
        )

        _log("phase5_done", overall_status="passed")
        print("=== eval PASSED ===")
        return 0

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n[FATAL] {exc}\n{tb}", file=sys.stderr)

        # Write failed integrity report
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(
                output_dir / "integrity_gates.json",
                {**gate_log.to_dict(), "fatal_exception": str(exc),
                 "traceback": tb, "failed_at": _now_iso()},
            )
            # Write inconclusive verdict
            _atomic_write_json(
                output_dir / "optimization_verdict.json",
                {
                    "schema": "micro4_opt_verdict_v5",
                    "objective_optimization": {
                        "final_checkpoint_effect": {"verdict": "inconclusive"},
                        "best_checkpoint_effect": {"verdict": "inconclusive"},
                        "regression_from_best": {"verdict": "inconclusive"},
                        "trajectory_pattern": {"verdict": "inconclusive"},
                    },
                    "objective_overfit": {"verdict": "not_assessed"},
                    "visual_capacity": {"verdict": "not_assessed"},
                    "grounding": {"verdict": "not_assessed"},
                    "integrity": {"overall": "FAILED", "exception": str(exc)},
                    "failed_at": _now_iso(),
                },
            )
            _log("fatal_exception", exception=str(exc))
        except Exception:
            pass

        return 2  # non-zero exit


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="micro4 Deterministic Optimization Eval v6.1"
    )
    p.add_argument(
        "--eval-dir", type=Path, default=DEFAULT_EVAL_OUTPUT_DIR,
        help="Formal eval output directory (must be under /group, must be empty)",
    )
    p.add_argument(
        "--preflight-dir", type=Path, default=DEFAULT_PREFLIGHT_OUTPUT_DIR,
        help="Preflight report directory (separate from eval dir, re-runnable)",
    )
    # Legacy alias kept for backward compatibility
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Alias for --eval-dir (deprecated, use --eval-dir)",
    )
    p.add_argument(
        "--preflight", action="store_true",
        help="Run static/path preflight only (no model load, no CUDA forward)",
    )
    p.add_argument(
        "--run-eval", action="store_true",
        help="Run full deterministic eval (requires GPU)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    # --output-dir is a legacy alias for --eval-dir
    eval_dir = (args.output_dir if args.output_dir is not None else args.eval_dir).resolve()
    preflight_dir = args.preflight_dir.resolve()

    for d, label in [(eval_dir, "eval-dir"), (preflight_dir, "preflight-dir")]:
        if not str(d).startswith("/group"):
            print(f"ERROR: {label} must be under /group, got: {d}",
                  file=sys.stderr)
            sys.exit(1)

    if args.preflight:
        sys.exit(run_preflight(preflight_dir, eval_dir))
    elif args.run_eval:
        sys.exit(run_eval(eval_dir))
    else:
        print("Specify --preflight or --run-eval", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
