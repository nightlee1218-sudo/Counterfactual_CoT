"""gold40 v3 LoRA C1 single-ball controlled experiment training.

Adapted from gold40_v3_lora_micro4_lr_probe.py for the C1 single-pair
controlled experiment (red ball, left_region->right_region).
Original script is unmodified.

Design contract:
  * Does NOT modify gold40_overfit.py, gold40_lora.py, lora_2step_smoke.py,
    smoke_train.py, bagel_manifest_dataset.py, or render_prompts.py.
  * Imports helpers from gold40_lora and lora_2step_smoke as-is.
  * Uses ONLY the C1 single-ball manifest (1 pair). Refuses to run on any other manifest.
  * Initialises from B_step000 checkpoint (62 tensors).
  * Saves checkpoints at steps 0, 10, 25, 50, 100, 200, 300, 400, 500.
  * Does NOT launch training without --acknowledge-full-training.
  * dry-run adds C1 canonical hard gate (c1_canonical_gate.json).

Four corrections vs v1 plan:
  1. sample_lens probe uses an independent CounterfactualCoTDataset instance,
     never consuming the training loader.
  2. Per-step train_log records actual pair_id from batch_data_indexes;
     post-training exposure_stats counts real occurrences per pair_id.
     epoch_within_cycle fields are removed.
  3. AdamW gate: before first optimizer.step, assert optimizer.state is empty
     (len==0) AND exactly 24 LoRA-B tensors are all-zero.
  4. micro4_vs_formal_B config diff: only manifest, num_samples, output_dir,
     eval_steps, save_steps are allowed to differ. All other fields must match
     formal B run_config_resolved.json exactly.

Modes:
  --dry-run   : build model, inject LoRA, run one no_grad forward, save
                step_000_trainable.pt, strict-reload, write dry-run report.
                Never calls optimizer.step.
  (default)   : full 400-step training. Requires --acknowledge-full-training
                AND a prior dry-run report.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

# ---------------------------------------------------------------------------
# Imports from frozen modules (do NOT modify those modules)
# ---------------------------------------------------------------------------
from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.datasets import CounterfactualCoTDataset
from experiments.counterfactual_cot.training.gold40_overfit import (
    _build_gold40_loader,
    _build_model_for_gold40,
    _forward_loss,
    _load_tokenizer,
    _mean_train_mse,
    _prepare_batch,
    _prepare_output_dir,
    _select_visualization_pairs,
)
from experiments.counterfactual_cot.training.gold40_lora import (
    _apply_trainable_and_audit,
    _append_jsonl,
    _atomic_write_json,
    _atomic_write_jsonl,
    _build_model_with_lora,
    _compare_trainable_checkpoints,
    _confirm_no_base_leak_in_optimizer,
    _dockerdata_ok,
    _now_iso,
    _save_step_trainable,
    _serialize_args_json,
    _sha256_file,
    _strict_reload_check,
    _write_git_provenance,
    _write_sequence_plan,
)
from experiments.counterfactual_cot.training.lora_2step_smoke import (
    load_trainable_strict,
    save_trainable_only,
    set_trainable,
)
from experiments.counterfactual_cot.training.smoke_train import (
    DEFAULT_MODEL_PATH,
    _device,
    _log_event,
    _path_size_gib,
    _set_seed,
    _inspect_raw_samples,
)
from data.transforms import ImageTransform
from experiments.counterfactual_cot.datasets import CounterfactualCoTDataset
from experiments.counterfactual_cot.training.fixed_deterministic_eval import (
    _build_raw_sample_dataset,
    _build_singleton_packed_dataset,
    _pack_single_pair,
    _stable_pair_seed,
    _set_deterministic_seed,
)

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

MICRO4_MANIFEST = (
    BAGEL_ROOT
    / "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region"
    / "manifests/intervention_pairs_single_ball_c1.jsonl"
)
MICRO4_MANIFEST_SHA256_EXPECT = "7306c0b1e60a2c6c269c29de29cea0894c06a651d32457f3127906ae4495593b"
MICRO4_NUM_PAIRS = 1
# B_step000 checkpoint (initialisation source)
B_STEP000_CKPT = (
    BAGEL_ROOT / "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_L24-27_r8_a16_lr1e6_singlepack_400steps/checkpoints/step_000_trainable.pt"
)
B_STEP000_SHA256_EXPECT = "0f138533db0bf98ddee14711e7664c798252b8e41710d5043303350cbe78baff"

# Formal B run_config_resolved.json (for config diff audit)
FORMAL_B_RUN_CONFIG = (
    BAGEL_ROOT / "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_L24-27_r8_a16_lr1e6_singlepack_400steps/run_config_resolved.json"
)

# micro4 output base
BASE_OUTPUT_STEM = BAGEL_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_lora_c1_single_ball"

# Checkpoint steps
MICRO4_SAVE_STEPS = (0, 10, 25, 50, 100, 200, 300, 400, 500)
MICRO4_EVAL_STEPS = (0, 10, 25, 50, 100, 200, 300, 400, 500)

# Fields allowed to differ between micro4 and formal B.
# Includes training hyperparams that legitimately differ AND
# run-mode control fields that exist only in micro4 (not in formal B).
MICRO4_ALLOWED_DIFF_KEYS = frozenset({
    # Legitimately different training config
    "manifest",
    "num_samples",
    "output_dir",
    "eval_steps",
    "save_steps",
    # lr-probe: lr and steps are swept across runs
    "lr",
    "steps",
    # Run-mode control fields: micro4-only, absent from formal B run_config
    "dry_run",
    "dry_run_report",
    "acknowledge_full_training",
    "allow_non_dockerdata",
    "resume_trainable_checkpoint",
})


# ---------------------------------------------------------------------------
# Frozen canonical fingerprint (evidence: single_pair_overfit_diag_ball_lr3e5_500steps)
# All values are directly read from historical artefacts; none are inferred.
# packed_timesteps_hash uses _tensor_hash algorithm (float32 numpy bytes -> sha256),
# matching fixed_deterministic_eval.py:_pack_single_pair.
# mse_indexes_sha256 is MISSING in historical artefacts (no full list saved);
# it is recorded per-run as a fingerprint only, not compared against history.
# target_prompt_hash is recorded per-run only, not a hard gate.
# ---------------------------------------------------------------------------
EXPECTED_CANONICAL: Dict[str, Any] = {
    "sequence_length":          1131,
    "num_vae_tokens":           512,
    "num_vit_tokens":           256,
    "mse_indexes_count":        256,
    "mse_indexes_min":          874,
    "mse_indexes_max":          1129,
    # packed_timesteps_hash: float32 numpy bytes -> sha256 (NOT json.dumps)
    "packed_timesteps_hash":    "4135670aa5e24d0175abaff975d3151f529955fa3cd488473f2b46cfecf78cb7",
    "source_prompt_hash":       "608213d94cb98602490a2e956d6a76c75eaac6042a9965f7d01f4dd4fc4e30a4",
    "source_cot_hash":          "6a8262eb6a459fd66db521d5eeb621b3d9400bbc9dece30b8618ec0748bf7035",
    "target_cot_hash":          "20452b4539ee898ef1400cc359920ee467e966e197fc43869fae98f9c2dffe1b",
    # mse_indexes_sha256: MISSING in historical artefacts; recorded per-run as fingerprint only
    # target_prompt_hash: recorded per-run only, not a hard gate
}

# ---------------------------------------------------------------------------
# Historical RNG contract (single_pair_overfit_diag_ball_lr3e5_500steps)
# ---------------------------------------------------------------------------
# pair_seed = _stable_pair_seed(pair_id, seed=1234) = 1460362567
# step_seed(s) = pair_seed + s  (s = 1..500)
# Each step: _set_deterministic_seed(step_seed) then _pack_single_pair
# torch.manual_seed(step_seed) before forward
C1_PAIR_ID = (
    "cf_source_living_room_play_absolute_position_change_"
    "v4_absolute_region_replacement_v4_0000"
)
C1_PAIR_SEED: int = 1460362567   # _stable_pair_seed(C1_PAIR_ID, 1234)
C1_STEPS: int = 500

# Historical single_pair_overfit report (frozen reference for RNG trajectory gate)
HIST_REPORT_PATH = (
    BAGEL_ROOT / "experiments/counterfactual_cot/outputs/"
    "single_pair_overfit_diag_ball_lr3e5_500steps/single_pair_overfit_report.json"
)
HIST_REPORT_SHA256 = "525e8911aa281faf0cb111c3cb319f3856dfc3f37ac3cd9e011d6452a7be0c43"

# Expected step-000 packed_timesteps_hash (pair_seed, no step offset)
EXPECTED_STEP000_PACKED_TS_HASH = "4135670aa5e24d0175abaff975d3151f529955fa3cd488473f2b46cfecf78cb7"

# Expected 500-step timesteps_hash sequence SHA256
# sha256(json.dumps([hash_step1, ..., hash_step500]))
EXPECTED_RNG_TRAJECTORY_SHA256 = "aaf8e52c5188d8f02fd17ed19cb2b5f62255c8befae2d5dc77c012e500d3bca4"

# Expected LoRA counts (inherited from formal B architecture)
EXPECTED_LORA_MODULES = 24
EXPECTED_LORA_B_TENSORS = 24
EXPECTED_TOTAL_TRAINABLE_TENSORS = 62  # 14 existing + 24 lora_A + 24 lora_B


# ---------------------------------------------------------------------------
# Shared canonical gate helpers (used by both dry-run and full-training)
# ---------------------------------------------------------------------------

def _extract_canonical_fields(
    batch: Any,
    probe: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract canonical fields from batch/probe/records.

    packed_timesteps_hash: float32 numpy bytes -> sha256
        (matches fixed_deterministic_eval._tensor_hash algorithm).
    mse_indexes_sha256_fingerprint: json.dumps -> sha256
        (per-run fingerprint only, not compared against EXPECTED_CANONICAL).
    target_prompt_hash: recorded only, not a hard gate.
    """
    import hashlib as _hlib
    import json as _jmod
    import numpy as _np

    # mse_indexes — correct attribute name is mse_loss_indexes (SimpleCustomBatch)
    try:
        mse_indexes = [int(i) for i in batch.mse_loss_indexes.tolist()]             if hasattr(batch, "mse_loss_indexes") and batch.mse_loss_indexes is not None else []
    except Exception:
        mse_indexes = []

    # packed_timesteps: float32 numpy bytes -> sha256
    try:
        _pts = batch.packed_timesteps             if hasattr(batch, "packed_timesteps") and batch.packed_timesteps is not None else None
    except Exception:
        _pts = None
    if _pts is not None:
        import torch as _torch
        arr = _pts.detach().to("cpu", dtype=_torch.float32).contiguous().numpy()
        packed_ts_hash = _hlib.sha256(arr.tobytes()).hexdigest()
    else:
        packed_ts_hash = ""

    # sequence_length
    try:
        seq_len = int(batch.sequence_length) if hasattr(batch, "sequence_length") else 0
    except Exception:
        seq_len = 0

    # num_vae_tokens / num_vit_tokens from batch directly (not from probe —
    # sample_lens_probe does not contain these fields)
    try:
        num_vae_tokens = int(len(batch.packed_vae_token_indexes))             if hasattr(batch, "packed_vae_token_indexes") and batch.packed_vae_token_indexes is not None             else None
    except Exception:
        num_vae_tokens = None
    try:
        num_vit_tokens = int(len(batch.packed_vit_token_indexes))             if hasattr(batch, "packed_vit_token_indexes") and batch.packed_vit_token_indexes is not None             else None
    except Exception:
        num_vit_tokens = None

    # prompt / CoT from records
    try:
        rec = records[0]
        src_cot    = rec.get("source", {}).get("cot_text", "")
        tgt_cot    = rec.get("target", {}).get("cot_text", "")
        src_prompt = rec.get("source", {}).get("prompt", "")
        tgt_prompt = rec.get("target", {}).get("prompt", "")
    except Exception:
        src_cot = tgt_cot = src_prompt = tgt_prompt = ""

    # mse structural checks
    mse_strictly_increasing = (
        all(mse_indexes[i] < mse_indexes[i + 1] for i in range(len(mse_indexes) - 1))
        if len(mse_indexes) > 1 else (len(mse_indexes) == 1)
    )
    mse_no_duplicates  = len(mse_indexes) == len(set(mse_indexes))
    mse_contiguous     = (
        mse_indexes == list(range(mse_indexes[0], mse_indexes[0] + len(mse_indexes)))
        if mse_indexes else False
    )

    # mse_indexes fingerprint (json.dumps, per-run only)
    mse_indexes_sha256_fp = _hlib.sha256(
        _jmod.dumps(mse_indexes, ensure_ascii=False).encode()
    ).hexdigest()

    return {
        "sequence_length":                seq_len,
        "num_vae_tokens":                 num_vae_tokens,
        "num_vit_tokens":                 num_vit_tokens,
        "mse_indexes_count":              len(mse_indexes),
        "mse_indexes_min":                min(mse_indexes) if mse_indexes else None,
        "mse_indexes_max":                max(mse_indexes) if mse_indexes else None,
        "mse_indexes_strictly_increasing": mse_strictly_increasing,
        "mse_indexes_no_duplicates":       mse_no_duplicates,
        "mse_indexes_contiguous":          mse_contiguous,
        "mse_indexes_full_list":           mse_indexes,
        "mse_indexes_sha256_fingerprint":  mse_indexes_sha256_fp,
        "packed_timesteps_hash":           packed_ts_hash,
        "source_cot_hash":    _hlib.sha256(src_cot.encode()).hexdigest(),
        "target_cot_hash":    _hlib.sha256(tgt_cot.encode()).hexdigest(),
        "source_prompt_hash": _hlib.sha256(src_prompt.encode()).hexdigest(),
        "target_prompt_hash": _hlib.sha256(tgt_prompt.encode()).hexdigest(),
    }


def _build_stable_fingerprint(fields: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Build stable canonical fingerprint (no created_at_iso) and its SHA256.

    Uses sort_keys=True for deterministic serialisation.
    """
    import hashlib as _hlib
    import json as _jmod

    fp = {
        "sequence_length":                fields["sequence_length"],
        "num_vae_tokens":                 fields["num_vae_tokens"],
        "num_vit_tokens":                 fields["num_vit_tokens"],
        "mse_indexes_count":              fields["mse_indexes_count"],
        "mse_indexes_min":                fields["mse_indexes_min"],
        "mse_indexes_max":                fields["mse_indexes_max"],
        "mse_indexes_contiguous":         fields["mse_indexes_contiguous"],
        "packed_timesteps_hash":          fields["packed_timesteps_hash"],
        "source_prompt_hash":             fields["source_prompt_hash"],
        "source_cot_hash":                fields["source_cot_hash"],
        "target_cot_hash":                fields["target_cot_hash"],
        "mse_indexes_sha256_fingerprint": fields["mse_indexes_sha256_fingerprint"],
    }
    fp_sha256 = _hlib.sha256(
        _jmod.dumps(fp, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return fp, fp_sha256


def _run_canonical_gate(
    fields: Dict[str, Any],
    gate_path: Path,
    pair_id: Optional[str],
    dryrun_stable_fp_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare fields against EXPECTED_CANONICAL; build and write gate JSON.

    Args:
        fields:                  output of _extract_canonical_fields()
        gate_path:               where to write the gate JSON atomically
        pair_id:                 batch pair_id for logging
        dryrun_stable_fp_sha256: if provided, also check stable fingerprint
                                 matches dry-run (used in full-training only)

    Returns the gate dict (with status, gate_failures, stable_fingerprint, …).
    Raises RuntimeError if gate fails.
    """
    stable_fp, stable_fp_sha256 = _build_stable_fingerprint(fields)

    failures: List[str] = []

    # Strict comparison against EXPECTED_CANONICAL
    for field, expected in EXPECTED_CANONICAL.items():
        actual = fields.get(field)
        if actual != expected:
            failures.append(f"{field}: expected={expected!r} actual={actual!r}")

    # mse structural checks
    if not fields["mse_indexes_strictly_increasing"]:
        failures.append("mse_indexes not strictly increasing")
    if not fields["mse_indexes_no_duplicates"]:
        failures.append("mse_indexes has duplicates")
    if not fields["mse_indexes_contiguous"]:
        failures.append("mse_indexes not contiguous")

    # existence checks
    if fields["sequence_length"] == 0:
        failures.append("sequence_length=0 (batch attribute missing)")
    if fields["num_vae_tokens"] is None:
        failures.append("num_vae_tokens not found in probe")
    if fields["num_vit_tokens"] is None:
        failures.append("num_vit_tokens not found in probe")
    if fields["mse_indexes_count"] == 0:
        failures.append("mse_indexes_count=0 (no MSE tokens found)")
    if not fields["packed_timesteps_hash"]:
        failures.append("packed_timesteps missing from batch")

    # cross-run fingerprint check (full-training only)
    if dryrun_stable_fp_sha256 is not None:
        if stable_fp_sha256 != dryrun_stable_fp_sha256:
            failures.append(
                f"stable_fingerprint_sha256 mismatch vs dry-run: "
                f"dryrun={dryrun_stable_fp_sha256!r} current={stable_fp_sha256!r}"
            )

    gate = {
        "pair_id":                    pair_id,
        "stable_fingerprint":         stable_fp,
        "stable_fingerprint_sha256":  stable_fp_sha256,
        "gate_failures":              failures,
        "status":                     "passed" if not failures else "failed",
        "created_at_iso":             _now_iso(),
    }
    # Embed all extracted fields for auditability
    gate.update(fields)
    _atomic_write_json(gate_path, gate)

    if failures:
        raise RuntimeError(
            f"C1 canonical gate FAILED - "
            f"first optimizer.step will not execute and "
            f"no step=1 train-log record will be written.\n"
            f"Failures: {failures}\n"
            f"See {gate_path}"
        )
    return gate

# ---------------------------------------------------------------------------
# RNG trajectory helpers (C1 historical contract)
# ---------------------------------------------------------------------------

def _load_hist_rng_trajectory() -> List[str]:
    """Load and verify the 500-step timesteps_hash sequence from the frozen
    historical single_pair_overfit report.  Raises RuntimeError on SHA256
    mismatch, malformed trajectory rows, or missing file.
    """
    if not HIST_REPORT_PATH.exists():
        raise RuntimeError(
            f"Historical report not found: {HIST_REPORT_PATH}\n"
            "This file is required for the C1 RNG trajectory gate."
        )
    actual_sha = _sha256_file(HIST_REPORT_PATH)
    if actual_sha != HIST_REPORT_SHA256:
        raise RuntimeError(
            f"Historical report SHA256 mismatch:\n"
            f"  expected: {HIST_REPORT_SHA256}\n"
            f"  actual:   {actual_sha}\n"
            f"  path:     {HIST_REPORT_PATH}"
        )
    with open(HIST_REPORT_PATH, encoding="utf-8") as f:
        report = json.load(f)
    if not isinstance(report, dict):
        raise RuntimeError(
            "Historical report validation failed: row=n/a step=n/a "
            f"field=loss_history parent report must be dict, got {type(report).__name__}"
        )
    loss_history = report.get("loss_history")
    if not isinstance(loss_history, list):
        raise RuntimeError(
            "Historical report validation failed: row=n/a step=n/a "
            f"field=loss_history must be list, got {type(loss_history).__name__}"
        )
    if len(loss_history) != C1_STEPS:
        raise RuntimeError(
            "Historical report validation failed: row=n/a step=n/a "
            f"field=loss_history length={len(loss_history)}, expected={C1_STEPS}"
        )

    rows_by_step: Dict[int, Dict[str, Any]] = {}
    first_row_by_step: Dict[int, int] = {}
    timesteps_hash_pattern = re.compile(r"^[0-9a-f]{64}$")
    for row_number, row in enumerate(loss_history, start=1):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"Historical report validation failed: row={row_number} step=n/a "
                f"field=row must be dict, got {type(row).__name__}"
            )

        step = row.get("step")
        if type(step) is not int:
            raise RuntimeError(
                f"Historical report validation failed: row={row_number} step={step!r} "
                f"field=step must be int (bool forbidden), got {type(step).__name__}"
            )
        if step < 1 or step > C1_STEPS:
            raise RuntimeError(
                f"Historical report validation failed: row={row_number} step={step} "
                f"field=step must be in 1..{C1_STEPS}"
            )
        if step in rows_by_step:
            raise RuntimeError(
                f"Historical report validation failed: row={row_number} step={step} "
                f"field=step duplicates row={first_row_by_step[step]}"
            )

        step_seed = row.get("step_seed")
        expected_step_seed = C1_PAIR_SEED + step
        if step_seed != expected_step_seed:
            raise RuntimeError(
                f"Historical report validation failed: row={row_number} step={step} "
                f"field=step_seed expected={expected_step_seed}, actual={step_seed!r}"
            )

        timesteps_hash = row.get("timesteps_hash")
        if not isinstance(timesteps_hash, str) or timesteps_hash_pattern.fullmatch(
            timesteps_hash
        ) is None:
            raise RuntimeError(
                f"Historical report validation failed: row={row_number} step={step} "
                "field=timesteps_hash must match ^[0-9a-f]{64}$, "
                f"actual={timesteps_hash!r}"
            )

        rows_by_step[step] = row
        first_row_by_step[step] = row_number

    expected_steps = set(range(1, C1_STEPS + 1))
    actual_steps = set(rows_by_step)
    if actual_steps != expected_steps:
        missing_steps = sorted(expected_steps - actual_steps)
        extra_steps = sorted(actual_steps - expected_steps)
        raise RuntimeError(
            "Historical report validation failed: row=n/a step=n/a field=step "
            f"must cover exactly 1..{C1_STEPS}; missing={missing_steps}, extra={extra_steps}"
        )

    return [rows_by_step[step]["timesteps_hash"] for step in range(1, C1_STEPS + 1)]


def _run_rng_trajectory_gate(
    packer: Any,
    raw_dataset: Any,
    pair: Dict[str, Any],
    gate_path: Path,
) -> Dict[str, Any]:
    """Without GPU forward: re-pack C1_STEPS batches using pair_seed+step,
    compute packed_timesteps_hash for each, compare against historical sequence.

    Returns gate dict; raises RuntimeError if any step mismatches.
    """
    import numpy as _np_traj
    import hashlib as _hlib_traj

    def _ts_hash(batch: Any) -> str:
        pts = getattr(batch, "packed_timesteps", None)
        if pts is None:
            return ""
        arr = pts.detach().to("cpu", dtype=torch.float32).contiguous().numpy()
        return _hlib_traj.sha256(arr.tobytes()).hexdigest()

    # Verify step-000 hash first (pair_seed, no step offset)
    _set_deterministic_seed(C1_PAIR_SEED)
    batch0, trace0 = _pack_single_pair(packer, raw_dataset, pair, C1_PAIR_SEED)
    step000_hash = trace0.get("packed_timesteps_hash", _ts_hash(batch0))
    step000_ok = (step000_hash == EXPECTED_STEP000_PACKED_TS_HASH)

    # Load historical 500-step sequence
    hist_hashes = _load_hist_rng_trajectory()

    # Re-pack 500 steps (CPU only, no model forward)
    current_hashes: List[str] = []
    first_mismatch_step: Optional[int] = None
    for step in range(1, C1_STEPS + 1):
        step_seed = C1_PAIR_SEED + step
        _set_deterministic_seed(step_seed)
        batch_s, trace_s = _pack_single_pair(packer, raw_dataset, pair, step_seed)
        h = trace_s.get("packed_timesteps_hash", _ts_hash(batch_s))
        current_hashes.append(h)
        if h != hist_hashes[step - 1] and first_mismatch_step is None:
            first_mismatch_step = step

    matched = sum(c == h for c, h in zip(current_hashes, hist_hashes))
    current_seq_sha = _hlib_traj.sha256(
        json.dumps(current_hashes, ensure_ascii=False).encode()
    ).hexdigest()
    hist_seq_sha = _hlib_traj.sha256(
        json.dumps(hist_hashes, ensure_ascii=False).encode()
    ).hexdigest()
    trajectory_ok = (first_mismatch_step is None) and step000_ok

    gate = {
        "status": "passed" if trajectory_ok else "failed",
        "step000_hash": step000_hash,
        "step000_expected": EXPECTED_STEP000_PACKED_TS_HASH,
        "step000_ok": step000_ok,
        "steps_matched": matched,
        "steps_total": C1_STEPS,
        "first_mismatch_step": first_mismatch_step,
        "current_trajectory_sha256": current_seq_sha,
        "hist_trajectory_sha256": hist_seq_sha,
        "expected_trajectory_sha256": EXPECTED_RNG_TRAJECTORY_SHA256,
        "trajectory_sha256_ok": current_seq_sha == EXPECTED_RNG_TRAJECTORY_SHA256,
        "hist_report_path": str(HIST_REPORT_PATH),
        "hist_report_sha256": HIST_REPORT_SHA256,
        "created_at_iso": _now_iso(),
    }
    _atomic_write_json(gate_path, gate)

    if not trajectory_ok:
        failures = []
        if not step000_ok:
            failures.append(
                f"step000 packed_timesteps_hash mismatch: "
                f"expected={EXPECTED_STEP000_PACKED_TS_HASH!r} actual={step000_hash!r}"
            )
        if first_mismatch_step is not None:
            failures.append(
                f"first mismatch at step={first_mismatch_step} "
                f"(matched {matched}/{C1_STEPS})"
            )
        raise RuntimeError(
            f"C1 RNG trajectory gate FAILED - training blocked.\n"
            + "\n".join(f"  {f}" for f in failures)
            + f"\nSee {gate_path}"
        )
    return gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_micro4_manifest(manifest: Path) -> None:
    if manifest.resolve() != MICRO4_MANIFEST.resolve():
        raise RuntimeError(
            f"C1 is locked to {MICRO4_MANIFEST}, got {manifest}."
        )
    if not manifest.exists():
        raise FileNotFoundError(f"micro4 manifest missing: {manifest}")
    sha = _sha256_file(manifest)
    if sha != MICRO4_MANIFEST_SHA256_EXPECT:
        raise RuntimeError(
            f"C1 manifest SHA mismatch: expected={MICRO4_MANIFEST_SHA256_EXPECT} observed={sha}"
        )


def _validate_b_step000(ckpt: Path) -> str:
    if not ckpt.exists():
        raise FileNotFoundError(f"B_step000 checkpoint missing: {ckpt}")
    sha = _sha256_file(ckpt)
    if sha != B_STEP000_SHA256_EXPECT:
        raise RuntimeError(
            f"B_step000 SHA mismatch: expected={B_STEP000_SHA256_EXPECT} observed={sha}"
        )
    return sha


# ---------------------------------------------------------------------------
# 1. Independent sample_lens probe (never consumes training loader)
# ---------------------------------------------------------------------------

def _sample_lens_probe(
    manifest: Path,
    tokenizer: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build an independent CounterfactualCoTDataset (separate from train loader)
    and iterate all 4 pairs once to record true token counts and sequence plans.
    Does NOT advance the training loader iterator.
    """
    transform = ImageTransform(
        max_image_size=args.image_size,
        min_image_size=args.image_size,
        image_stride=16,
    )
    vit_transform = ImageTransform(
        max_image_size=args.vit_image_size,
        min_image_size=args.vit_min_image_size,
        image_stride=14,
    )
    probe_dataset = CounterfactualCoTDataset(
        dataset_name="micro4_sample_lens_probe",
        transform=transform,
        tokenizer=tokenizer,
        manifest_path=str(manifest),
        variant=args.variant,
        include_source_image=True,
        training_seed=0,
        local_rank=0,
        world_size=1,
        num_workers=0,
        data_status=None,
        vit_transform=vit_transform,
        max_records=MICRO4_NUM_PAIRS,
    )
    samples = []
    iterator = iter(probe_dataset)
    for _ in range(MICRO4_NUM_PAIRS):
        s = next(iterator)
        num_tokens = s["num_tokens"]
        seq_plan = s["sequence_plan"]
        samples.append({
            "pair_id": s["data_indexes"].get("pair_id"),
            "mode": s["data_indexes"].get("mode"),
            "num_tokens": int(num_tokens),
            "sequence_plan": [
                {
                    "pos": i,
                    "type": item["type"],
                    "loss": bool(item["loss"]),
                    "enable_cfg": bool(item["enable_cfg"]),
                }
                for i, item in enumerate(seq_plan)
            ],
            "fits_in_expected_num_tokens": num_tokens <= args.expected_num_tokens,
            "fits_in_max_num_tokens_per_sample": num_tokens <= args.max_num_tokens_per_sample,
        })
    return {
        "event": "sample_lens_probe",
        "source": "independent_dataset_not_train_loader",
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "expected_num_tokens": args.expected_num_tokens,
        "max_num_tokens_per_sample": args.max_num_tokens_per_sample,
        "max_num_tokens": args.max_num_tokens,
        "num_pairs_probed": len(samples),
        "samples": samples,
        "packing_conclusion": (
            "All pairs exceed expected_num_tokens; PackedDataset produces "
            "1-sample packs (singlepack behaviour). effective_batch_size=1."
        ),
        "created_at_iso": _now_iso(),
    }


# ---------------------------------------------------------------------------
# 2. micro4_vs_formal_B config diff audit
# ---------------------------------------------------------------------------

def _micro4_vs_formal_b_diff(args: argparse.Namespace) -> Dict[str, Any]:
    """Compare micro4 resolved config against formal B run_config_resolved.json.
    Only MICRO4_ALLOWED_DIFF_KEYS are permitted to differ.
    """
    if not FORMAL_B_RUN_CONFIG.exists():
        return {
            "status": "skipped",
            "reason": f"formal_B run_config not found: {FORMAL_B_RUN_CONFIG}",
        }
    formal_b = json.loads(FORMAL_B_RUN_CONFIG.read_text(encoding="utf-8"))["args"]

    micro4_config = _serialize_args_json(args)

    # Fields to compare: union of both key sets, excluding micro4-only keys
    all_keys = set(formal_b.keys()) | set(micro4_config.keys())
    diffs: Dict[str, Any] = {}
    for key in sorted(all_keys):
        fb_val = formal_b.get(key, "__MISSING__")
        m4_val = micro4_config.get(key, "__MISSING__")
        if fb_val != m4_val:
            diffs[key] = {"formal_B": fb_val, "micro4": m4_val}

    forbidden_diffs = {k: v for k, v in diffs.items() if k not in MICRO4_ALLOWED_DIFF_KEYS}
    allowed_diffs = {k: v for k, v in diffs.items() if k in MICRO4_ALLOWED_DIFF_KEYS}

    status = "passed" if len(forbidden_diffs) == 0 else "failed"
    return {
        "status": status,
        "allowed_diff_keys": sorted(MICRO4_ALLOWED_DIFF_KEYS),
        "allowed_diffs": allowed_diffs,
        "forbidden_diffs": forbidden_diffs,
        "num_forbidden_diffs": len(forbidden_diffs),
        "formal_B_config_path": str(FORMAL_B_RUN_CONFIG),
        "created_at_iso": _now_iso(),
    }


# ---------------------------------------------------------------------------
# 3. AdamW gate: optimizer.state empty + 24 LoRA-B all-zero
# ---------------------------------------------------------------------------

def _adamw_pre_step_gate(
    optimizer: torch.optim.AdamW,
    model: nn.Module,
) -> Dict[str, Any]:
    """Assert optimizer.state is empty (no momentum buffers yet) and
    exactly EXPECTED_LORA_B_TENSORS LoRA-B tensors are all-zero.
    Must be called BEFORE the first optimizer.step().
    """
    failures: List[str] = []

    # Gate 1: optimizer.state must be empty
    num_state_entries = len(optimizer.state)
    g1 = num_state_entries == 0
    if not g1:
        failures.append(
            f"optimizer.state has {num_state_entries} entries before first step (expected 0)"
        )

    # Gate 2: find all LoRA-B tensors and verify all-zero
    lora_b_names = [
        name for name, p in model.named_parameters()
        if p.requires_grad and name.endswith(".lora_B")
    ]
    num_lora_b = len(lora_b_names)
    g2_count = num_lora_b == EXPECTED_LORA_B_TENSORS
    if not g2_count:
        failures.append(
            f"found {num_lora_b} LoRA-B tensors, expected {EXPECTED_LORA_B_TENSORS}"
        )

    lora_b_zero_check: List[Dict[str, Any]] = []
    all_lora_b_zero = True
    for name in lora_b_names:
        p = dict(model.named_parameters())[name]
        max_abs = float(p.data.abs().max().item())
        is_zero = max_abs == 0.0
        if not is_zero:
            all_lora_b_zero = False
            failures.append(f"LoRA-B tensor {name} is not zero (max_abs={max_abs})")
        lora_b_zero_check.append({
            "name": name,
            "shape": list(p.shape),
            "max_abs": max_abs,
            "is_zero": is_zero,
        })

    g2 = g2_count and all_lora_b_zero
    overall = "passed" if len(failures) == 0 else "failed"
    return {
        "gate": "adamw_pre_step_gate",
        "status": overall,
        "failures": failures,
        "G1_optimizer_state_empty": g1,
        "G1_num_state_entries": num_state_entries,
        "G2_lora_b_count_correct": g2_count,
        "G2_num_lora_b_found": num_lora_b,
        "G2_expected_lora_b": EXPECTED_LORA_B_TENSORS,
        "G2_all_lora_b_zero": all_lora_b_zero,
        "lora_b_zero_check": lora_b_zero_check,
        "created_at_iso": _now_iso(),
    }


# ---------------------------------------------------------------------------
# 4. Exposure audit (post-training, from actual train_log)
# ---------------------------------------------------------------------------

def _exposure_audit(train_log_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Read train_log.jsonl and count actual exposures per pair_id."""
    if not train_log_path.exists():
        return {"status": "skipped", "reason": "train_log.jsonl not found"}
    counts: Dict[str, int] = {}
    total_steps = 0
    with open(train_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            pid = entry.get("pair_id") or "__unknown__"
            counts[pid] = counts.get(pid, 0) + 1
            total_steps += 1
    formal_b_exposures_per_pair = 400 // 40  # = 10
    micro4_exposures_per_pair = total_steps // MICRO4_NUM_PAIRS if MICRO4_NUM_PAIRS > 0 else 0
    return {
        "event": "exposure_audit",
        "total_steps_logged": total_steps,
        "num_pairs": MICRO4_NUM_PAIRS,
        "exposures_per_pair_id": counts,
        "micro4_avg_exposures_per_pair": micro4_exposures_per_pair,
        "formal_B_exposures_per_pair": formal_b_exposures_per_pair,
        "exposure_ratio_micro4_vs_formal_B": (
            micro4_exposures_per_pair / formal_b_exposures_per_pair
            if formal_b_exposures_per_pair > 0 else None
        ),
        "equal_exposure_steps_would_be": formal_b_exposures_per_pair * MICRO4_NUM_PAIRS,
        "interpretation": (
            f"micro4 runs {args.steps} steps on {MICRO4_NUM_PAIRS} pairs = "
            f"~{micro4_exposures_per_pair} exposures/pair. "
            f"Formal B runs 400 steps on 40 pairs = {formal_b_exposures_per_pair} exposures/pair. "
            "micro4 is intentionally over-exposed for capacity diagnosis."
        ),
        "created_at_iso": _now_iso(),
    }


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------

def _run_dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    # G_dir: reject any existing non-empty output before preparation or writes.
    _out = Path(args.output_dir)
    if _out.exists() and (not _out.is_dir() or any(_out.iterdir())):
        raise RuntimeError(
            f"G_dir: output_dir already exists and is non-empty or is not a directory: {_out}"
        )

    _validate_micro4_manifest(args.manifest)
    b_step000_sha = _validate_b_step000(args.resume_trainable_checkpoint)

    _prepare_output_dir(args.output_dir)
    _atomic_write_json(args.output_dir / "run_config_resolved.json", {
        "mode": "dry_run",
        "args": _serialize_args_json(args),
        "created_at_iso": _now_iso(),
    })
    _write_git_provenance(args.output_dir)
    _write_sequence_plan(args.output_dir, args.manifest)
    _set_seed(args.seed)
    device = _device(args.device)

    records = load_jsonl(args.manifest)
    if len(records) != args.num_samples:
        raise ValueError(
            f"micro4 expects exactly {args.num_samples} records, found {len(records)}"
        )
    if not all(bool((r.get("human_review") or {}).get("accepted")) for r in records):
        raise ValueError("All micro4 records must have human_review.accepted=true")

    _dockerdata_ok(args.llm_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.vae_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.checkpoint_path, args.allow_non_dockerdata)

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)

    # 1. Independent sample_lens probe (before building train loader)
    probe = _sample_lens_probe(args.manifest, tokenizer, args)
    _atomic_write_json(args.output_dir / "sample_lens_probe.json", probe)

    # 2. Raw sequence report (uses _inspect_raw_samples from smoke_train)
    raw_sequence_report = _inspect_raw_samples(
        args.manifest, tokenizer, args.variant,
        args.image_size, args.vit_image_size, args.vit_min_image_size,
        MICRO4_NUM_PAIRS,
    )

    # 3. Build _pack_single_pair infrastructure (replaces train_loader for canonical batch)
    # C1 uses _stable_pair_seed / _pack_single_pair to exactly reproduce the
    # historical single_pair_overfit_diag_ball_lr3e5_500steps RNG contract.
    raw_dataset = _build_raw_sample_dataset(args, tokenizer)
    packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)
    pair = records[0]  # single-pair manifest

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, vae_model, wrappers, target_names, base_checkpoint_report = _build_model_with_lora(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
    )
    param_audit = _apply_trainable_and_audit(model, wrappers, target_names, args, device.type)

    # 4. micro4_vs_formal_B config diff audit
    config_diff = _micro4_vs_formal_b_diff(args)
    _atomic_write_json(args.output_dir / "micro4_vs_formal_b_config_diff.json", config_diff)
    if config_diff["status"] == "failed":
        raise RuntimeError(
            f"micro4_vs_formal_B config diff FAILED: {config_diff['forbidden_diffs']}"
        )

    # 5. Load B_step000 weights
    existing_names = param_audit["existing_names"]
    lora_names = param_audit["lora_names"]
    resume_report = load_trainable_strict(args.resume_trainable_checkpoint, model)
    _atomic_write_json(args.output_dir / "resume_from_b_step000.json", {
        "resume_from": str(args.resume_trainable_checkpoint),
        "resume_sha256": b_step000_sha,
        "num_tensors_loaded": resume_report["num_loaded_tensors"],
        "max_abs_load_copy_diff": resume_report["max_abs_load_copy_diff"],
        "created_at_iso": _now_iso(),
    })

    # 6. Canonical batch via _pack_single_pair(pair_seed) — deterministic, matches history
    model.train()
    vae_model.eval()
    model_dtype = next(model.parameters()).dtype
    batch, _canon_trace = _pack_single_pair(packer, raw_dataset, pair, C1_PAIR_SEED)

    # Capture real batch structure from SimpleCustomBatch attributes
    try:
        batch_data_indexes = batch.batch_data_indexes or [{}]
        batch_pair_id = batch_data_indexes[0].get("pair_id") if batch_data_indexes else None
        batch_sample_lens = list(batch.sample_lens) if hasattr(batch, "sample_lens") else []
        batch_sequence_length = int(batch.sequence_length) if hasattr(batch, "sequence_length") else 0
    except Exception:
        batch_pair_id = None
        batch_sample_lens = []
        batch_sequence_length = 0

    # C1 canonical gate (shared implementation)
    _canon_fields = _extract_canonical_fields(batch, probe, records)
    canonical_gate = _run_canonical_gate(
        _canon_fields,
        gate_path=args.output_dir / "c1_canonical_gate.json",
        pair_id=batch_pair_id,
        dryrun_stable_fp_sha256=None,  # dry-run: no prior fingerprint to compare
    )
    _stable_fingerprint     = canonical_gate["stable_fingerprint"]
    _stable_fingerprint_sha256 = canonical_gate["stable_fingerprint_sha256"]

    # 6b. RNG trajectory gate: write atomically, then bind its file SHA256.
    _rng_traj_gate_path = args.output_dir / "c1_rng_trajectory_gate.json"
    rng_traj_gate = _run_rng_trajectory_gate(
        packer, raw_dataset, pair,
        gate_path=_rng_traj_gate_path,
    )
    _rng_traj_gate_file_sha256 = _sha256_file(_rng_traj_gate_path)

    data = _prepare_batch(batch, vae_model, device, model_dtype)
    with torch.no_grad():
        loss, scalars = _forward_loss(model, data)
    mse_mean = float(scalars.get("mse", float(loss.detach().item())))
    mse_all_finite = torch.isfinite(loss).item()
    output_shape = list(loss.shape) if hasattr(loss, "shape") else []
    peak_alloc_mib = (torch.cuda.max_memory_allocated() if device.type == "cuda" else 0) / (1024 * 1024)
    peak_reserved_mib = (torch.cuda.max_memory_reserved() if device.type == "cuda" else 0) / (1024 * 1024)

    # 7. Save step_000 and strict-reload
    step000 = _save_step_trainable(
        0, model, args.output_dir, args, target_names, existing_names, lora_names,
    )
    reload_report = _strict_reload_check(Path(step000["path"]), model)

    # 8. Compare micro4 step_000 vs B_step000 (tensor-level, not file SHA)
    step000_vs_b = _compare_trainable_checkpoints(
        args.resume_trainable_checkpoint, Path(step000["path"])
    )
    _atomic_write_json(args.output_dir / "step000_vs_b_step000.json", step000_vs_b)
    if not step000_vs_b["all_tensors_equal"]:
        raise RuntimeError(
            f"micro4 step_000 diverges from B_step000: max_abs_diff={step000_vs_b['max_abs_diff']}"
        )

    # Compute script SHA256 for binding
    _script_sha256 = _sha256_file(Path(__file__))
    # Compute canonical gate file SHA256 for binding
    _gate_file_sha256 = _sha256_file(args.output_dir / "c1_canonical_gate.json")

    report = {
        "status": "passed",
        "mode": "dry_run",
        "created_at_iso": _now_iso(),
        "args": _serialize_args_json(args),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "b_step000_sha256": b_step000_sha,
        "script_sha256": _script_sha256,
        "canonical_gate_file_sha256": _gate_file_sha256,
        "rng_trajectory_gate_file_sha256": _rng_traj_gate_file_sha256,
        "stable_fingerprint": _stable_fingerprint,
        "stable_fingerprint_sha256": _stable_fingerprint_sha256,
        "base_checkpoint_report": base_checkpoint_report,
        "resume_report": resume_report,
        "sample_lens_probe": probe,
        "raw_sequence_report": raw_sequence_report,
        "first_batch_pair_id": batch_pair_id,
        "first_batch_sample_lens": batch_sample_lens,
        "first_batch_sequence_length": batch_sequence_length,
        "c1_canonical_gate": canonical_gate,
        "c1_rng_trajectory_gate": rng_traj_gate,
        "config_diff": config_diff,
        "param_audit": {k: v for k, v in param_audit.items() if k != "dtype_desc"},
        "forward": {
            "mse_mean": mse_mean,
            "mse_shape": output_shape,
            "mse_all_finite": mse_all_finite,
        },
        "step000_checkpoint": step000,
        "step000_strict_reload": reload_report,
        "step000_vs_b_step000": step000_vs_b,
        "cuda_memory_peak_mib": {
            "allocated": peak_alloc_mib,
            "reserved": peak_reserved_mib,
        },
    }
    _atomic_write_json(args.output_dir / "micro4_dry_run_report.json", report)
    return report


# ---------------------------------------------------------------------------
# Full training (400 steps)
# ---------------------------------------------------------------------------

def _run_full_training(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.acknowledge_full_training:
        raise RuntimeError(
            "Refusing to launch full training without --acknowledge-full-training. "
            "Run --dry-run first."
        )
    dry_run_report_path = Path(args.dry_run_report)
    if not dry_run_report_path.exists():
        raise RuntimeError(f"Missing dry-run report at {dry_run_report_path}. Run --dry-run first.")

    # C1 canonical gate must have passed in dry-run
    _c1_gate_from_dryrun = dry_run_report_path.parent / "c1_canonical_gate.json"
    if not _c1_gate_from_dryrun.exists():
        raise RuntimeError(
            f"C1 canonical gate file missing at {_c1_gate_from_dryrun}. "
            "Run --dry-run first to generate canonical gate."
        )
    import json as _json_gate
    _c1_gate = _json_gate.loads(_c1_gate_from_dryrun.read_text())
    if _c1_gate.get("status") != "passed":
        raise RuntimeError(
            f"C1 canonical gate status={_c1_gate.get('status')} - training blocked.\n"
            f"Failures: {_c1_gate.get('gate_failures')}\n"
            f"Fix canonical issues and re-run --dry-run."
        )

    # Binding verification: re-read dry-run report and check all binding fields
    import json as _json_bind
    _dryrun_report = _json_bind.loads(dry_run_report_path.read_text())
    _bind_failures = []
    # 1. Script SHA256
    _current_script_sha = _sha256_file(Path(__file__))
    _dryrun_script_sha = _dryrun_report.get("script_sha256", "")
    if _current_script_sha != _dryrun_script_sha:
        _bind_failures.append(
            f"script_sha256 mismatch: dry_run={_dryrun_script_sha!r} current={_current_script_sha!r}"
        )
    # 2. Manifest SHA256
    _current_manifest_sha = _sha256_file(args.manifest)
    _dryrun_manifest_sha = _dryrun_report.get("manifest_sha256", "")
    if _current_manifest_sha != _dryrun_manifest_sha:
        _bind_failures.append(
            f"manifest_sha256 mismatch: dry_run={_dryrun_manifest_sha!r} current={_current_manifest_sha!r}"
        )
    # 3. B_step000 SHA256
    _current_b_sha = _sha256_file(args.resume_trainable_checkpoint)
    _dryrun_b_sha = _dryrun_report.get("b_step000_sha256", "")
    if _current_b_sha != _dryrun_b_sha:
        _bind_failures.append(
            f"b_step000_sha256 mismatch: dry_run={_dryrun_b_sha!r} current={_current_b_sha!r}"
        )
    # 4. Canonical gate file SHA256
    _current_gate_sha = _sha256_file(_c1_gate_from_dryrun)
    _dryrun_gate_sha = _dryrun_report.get("canonical_gate_file_sha256", "")
    if _current_gate_sha != _dryrun_gate_sha:
        _bind_failures.append(
            f"canonical_gate_file_sha256 mismatch: dry_run={_dryrun_gate_sha!r} current={_current_gate_sha!r}"
        )
    # 5. Stable fingerprint SHA256
    _dryrun_stable_fp_sha = _dryrun_report.get("stable_fingerprint_sha256", "")
    _gate_stable_fp_sha = _c1_gate.get("stable_fingerprint_sha256", "")
    if _dryrun_stable_fp_sha != _gate_stable_fp_sha:
        _bind_failures.append(
            f"stable_fingerprint_sha256 mismatch between dry_run_report and canonical_gate: "
            f"report={_dryrun_stable_fp_sha!r} gate={_gate_stable_fp_sha!r}"
        )
    # 6. RNG trajectory gate file binding and direct hard checks.
    _rng_traj_gate_path = dry_run_report_path.parent / "c1_rng_trajectory_gate.json"
    if not _rng_traj_gate_path.exists():
        _bind_failures.append(
            f"c1_rng_trajectory_gate.json missing at {_rng_traj_gate_path}. "
            "Run --dry-run first."
        )
    else:
        _current_rng_gate_file_sha = _sha256_file(_rng_traj_gate_path)
        _dryrun_rng_gate_file_sha = _dryrun_report.get(
            "rng_trajectory_gate_file_sha256"
        )
        if _current_rng_gate_file_sha != _dryrun_rng_gate_file_sha:
            _bind_failures.append(
                "rng_trajectory_gate_file_sha256 mismatch: "
                f"dry_run={_dryrun_rng_gate_file_sha!r} "
                f"current={_current_rng_gate_file_sha!r}"
            )

        _rng_gate_data = json.loads(_rng_traj_gate_path.read_text())
        if not isinstance(_rng_gate_data, dict):
            _bind_failures.append(
                "c1_rng_trajectory_gate must be a JSON object, "
                f"got {type(_rng_gate_data).__name__}"
            )
        else:
            if _rng_gate_data.get("status") != "passed":
                _bind_failures.append(
                    "c1_rng_trajectory_gate field=status expected='passed', "
                    f"actual={_rng_gate_data.get('status')!r}"
                )
            if _rng_gate_data.get("step000_ok") is not True:
                _bind_failures.append(
                    "c1_rng_trajectory_gate field=step000_ok expected=True, "
                    f"actual={_rng_gate_data.get('step000_ok')!r}"
                )
            if _rng_gate_data.get("steps_matched") != C1_STEPS:
                _bind_failures.append(
                    f"c1_rng_trajectory_gate field=steps_matched expected={C1_STEPS}, "
                    f"actual={_rng_gate_data.get('steps_matched')!r}"
                )
            if _rng_gate_data.get("steps_total") != C1_STEPS:
                _bind_failures.append(
                    f"c1_rng_trajectory_gate field=steps_total expected={C1_STEPS}, "
                    f"actual={_rng_gate_data.get('steps_total')!r}"
                )
            if (
                "first_mismatch_step" not in _rng_gate_data
                or _rng_gate_data.get("first_mismatch_step") is not None
            ):
                _bind_failures.append(
                    "c1_rng_trajectory_gate field=first_mismatch_step expected=None, "
                    f"actual={_rng_gate_data.get('first_mismatch_step')!r}"
                )
            if _rng_gate_data.get("trajectory_sha256_ok") is not True:
                _bind_failures.append(
                    "c1_rng_trajectory_gate field=trajectory_sha256_ok expected=True, "
                    f"actual={_rng_gate_data.get('trajectory_sha256_ok')!r}"
                )
            for _field in (
                "current_trajectory_sha256",
                "hist_trajectory_sha256",
                "expected_trajectory_sha256",
            ):
                if _rng_gate_data.get(_field) != EXPECTED_RNG_TRAJECTORY_SHA256:
                    _bind_failures.append(
                        f"c1_rng_trajectory_gate field={_field} "
                        f"expected={EXPECTED_RNG_TRAJECTORY_SHA256!r}, "
                        f"actual={_rng_gate_data.get(_field)!r}"
                    )
            if _rng_gate_data.get("hist_report_sha256") != HIST_REPORT_SHA256:
                _bind_failures.append(
                    "c1_rng_trajectory_gate field=hist_report_sha256 "
                    f"expected={HIST_REPORT_SHA256!r}, "
                    f"actual={_rng_gate_data.get('hist_report_sha256')!r}"
                )

            _dryrun_embedded_rng_gate = _dryrun_report.get("c1_rng_trajectory_gate")
            if not isinstance(_dryrun_embedded_rng_gate, dict):
                _bind_failures.append(
                    "dry-run report field=c1_rng_trajectory_gate must be a JSON object"
                )
            elif _dryrun_embedded_rng_gate.get(
                "current_trajectory_sha256"
            ) != _rng_gate_data.get("current_trajectory_sha256"):
                _bind_failures.append(
                    "current_trajectory_sha256 mismatch between dry-run report and "
                    "c1_rng_trajectory_gate.json: "
                    f"report={_dryrun_embedded_rng_gate.get('current_trajectory_sha256')!r} "
                    f"gate={_rng_gate_data.get('current_trajectory_sha256')!r}"
                )

    if _bind_failures:
        raise RuntimeError(
            f"C1 dry-run binding verification FAILED - training blocked.\n"
            + "\n".join(f"  {f}" for f in _bind_failures)
        )
    _dryrun_stable_fingerprint = _dryrun_report.get("stable_fingerprint", {})
    _dryrun_stable_fp_sha256 = _dryrun_stable_fp_sha

    # G_dir: output_dir must not exist or must be empty to prevent accidental overwrites
    _out = Path(args.output_dir)
    if _out.exists() and any(_out.iterdir()):
        raise RuntimeError(
            f"G_dir: output_dir already exists and is non-empty: {_out}\n"
            "Remove or rename it before launching a new lr-probe run."
        )

    _validate_micro4_manifest(args.manifest)
    b_step000_sha = _validate_b_step000(args.resume_trainable_checkpoint)

    _prepare_output_dir(args.output_dir)
    _atomic_write_json(args.output_dir / "run_config_resolved.json", {
        "mode": "full_training",
        "args": _serialize_args_json(args),
        "created_at_iso": _now_iso(),
    })
    _write_git_provenance(args.output_dir)
    _write_sequence_plan(args.output_dir, args.manifest)
    _set_seed(args.seed)
    device = _device(args.device)

    records = load_jsonl(args.manifest)
    if len(records) != args.num_samples:
        raise ValueError(
            f"micro4 expects exactly {args.num_samples} records, found {len(records)}"
        )

    _dockerdata_ok(args.llm_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.vae_path, args.allow_non_dockerdata)
    _dockerdata_ok(args.checkpoint_path, args.allow_non_dockerdata)

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)

    # 1. Independent sample_lens probe (informational only, not used for canonical gate)
    probe = _sample_lens_probe(args.manifest, tokenizer, args)
    _atomic_write_json(args.output_dir / "sample_lens_probe.json", probe)

    # 2. Build _pack_single_pair infrastructure (C1 historical RNG contract)
    # eval_loader is kept for step-0 eval only; training batches come from _pack_single_pair.
    raw_dataset = _build_raw_sample_dataset(args, tokenizer)
    packer = _build_singleton_packed_dataset(args, tokenizer, special_tokens)
    eval_loader = _build_gold40_loader(
        args.manifest, tokenizer, special_tokens, args, seed_offset=100000
    )
    pair = records[0]  # single-pair manifest

    # Load historical 500-step hash sequence for per-step verification
    hist_hashes = _load_hist_rng_trajectory()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model, vae_model, wrappers, target_names, base_checkpoint_report = _build_model_with_lora(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
    )
    param_audit = _apply_trainable_and_audit(model, wrappers, target_names, args, device.type)

    # 3. Config diff audit
    config_diff = _micro4_vs_formal_b_diff(args)
    _atomic_write_json(args.output_dir / "micro4_vs_formal_b_config_diff.json", config_diff)
    if config_diff["status"] == "failed":
        raise RuntimeError(
            f"micro4_vs_formal_B config diff FAILED: {config_diff['forbidden_diffs']}"
        )

    existing_names = param_audit["existing_names"]
    lora_names = param_audit["lora_names"]

    # 4. Load B_step000 weights
    resume_report = load_trainable_strict(args.resume_trainable_checkpoint, model)
    _atomic_write_json(args.output_dir / "resume_from_b_step000.json", {
        "resume_from": str(args.resume_trainable_checkpoint),
        "resume_sha256": b_step000_sha,
        "num_tensors_loaded": resume_report["num_loaded_tensors"],
        "max_abs_load_copy_diff": resume_report["max_abs_load_copy_diff"],
        "global_step_after_resume": 0,
        "optimizer_state_initialized_fresh": True,
        "scheduler_present": False,
        "lr_at_step_1": args.lr,
        "created_at_iso": _now_iso(),
    })

    model.train()
    vae_model.eval()
    model_dtype = next(model.parameters()).dtype

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in trainable_params}
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr,
    )
    optim_audit = _confirm_no_base_leak_in_optimizer(optimizer, model, trainable_ids)

    # 5. Save step_000 BEFORE any optimizer.step
    step000 = _save_step_trainable(
        0, model, args.output_dir, args, target_names, existing_names, lora_names,
    )

    # 6. Compare step_000 vs B_step000 (tensor-level)
    step000_vs_b = _compare_trainable_checkpoints(
        args.resume_trainable_checkpoint, Path(step000["path"])
    )
    _atomic_write_json(args.output_dir / "step000_vs_b_step000.json", step000_vs_b)
    if not step000_vs_b["all_tensors_equal"]:
        raise RuntimeError(
            f"micro4 step_000 diverges from B_step000: max_abs_diff={step000_vs_b['max_abs_diff']}"
        )

    # 7. AdamW pre-step gate (before first optimizer.step)
    adamw_gate = _adamw_pre_step_gate(optimizer, model)
    _atomic_write_json(args.output_dir / "adamw_pre_step_gate.json", adamw_gate)
    if adamw_gate["status"] != "passed":
        raise RuntimeError(
            f"AdamW pre-step gate FAILED: {adamw_gate['failures']}"
        )

    # 8. Pre-step-1 canonical gate using _pack_single_pair(pair_seed)
    # step_0 eval first, then canonical gate on pair_seed batch.
    save_step_set = set(int(s) for s in args.save_steps)
    eval_step_set = set(int(s) for s in args.eval_steps)
    save_step_set.add(0)
    eval_step_set.add(0)

    checkpoint_records: List[Dict[str, Any]] = [step000]
    step0_eval = _mean_train_mse(model, vae_model, eval_loader, device, model_dtype, args.eval_max_batches)
    eval_records: List[Dict[str, Any]] = [{"step": 0, **step0_eval}]

    train_log_path = args.output_dir / "train_log.jsonl"

    # Canonical gate on pair_seed batch (before any forward/backward/optimizer.step)
    _canon_batch0, _canon_trace0 = _pack_single_pair(packer, raw_dataset, pair, C1_PAIR_SEED)
    try:
        _ft_pair_id = (_canon_batch0.batch_data_indexes or [{}])[0].get("pair_id")             if hasattr(_canon_batch0, "batch_data_indexes") else None
    except Exception:
        _ft_pair_id = None
    _ft_canon_fields = _extract_canonical_fields(_canon_batch0, probe, records)
    _pre_step1_gate = _run_canonical_gate(
        _ft_canon_fields,
        gate_path=args.output_dir / "c1_pre_step1_canonical_gate.json",
        pair_id=_ft_pair_id,
        dryrun_stable_fp_sha256=_dryrun_stable_fp_sha256,
    )
    # Gate passed: proceed to training loop.
    # Each step uses _pack_single_pair(step_seed=pair_seed+step).
    # train_loader/train_iterator are NOT used for training batches.

    for step in range(1, args.steps + 1):
        step_wall_start = time.time()
        step_seed = C1_PAIR_SEED + step

        # Pack this step's batch deterministically (matches historical RNG contract)
        batch, step_trace = _pack_single_pair(packer, raw_dataset, pair, step_seed)

        # Per-step RNG gate: verify timesteps_hash matches history BEFORE forward
        _step_ts_hash = step_trace.get("packed_timesteps_hash", "")
        _hist_ts_hash = hist_hashes[step - 1]
        if _step_ts_hash != _hist_ts_hash:
            raise RuntimeError(
                f"C1 per-step RNG gate FAILED at step={step}: "
                f"step={step} forward/backward/optimizer.step will not execute and "
                f"no step={step} train-log record will be written.\n"
                f"  expected (hist): {_hist_ts_hash!r}\n"
                f"  actual:          {_step_ts_hash!r}"
            )

        # Reset forward-time Torch/CUDA RNG (matches historical torch.manual_seed(step_seed))
        torch.manual_seed(step_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(step_seed)

        # Record actual pair_id and real pack shape from SimpleCustomBatch
        try:
            _bdx = batch.batch_data_indexes or [{}]
            actual_pair_id = _bdx[0].get("pair_id") if _bdx else None
            actual_sample_index = (_bdx[0].get("data_indexes") or [None, None])[0]
            actual_sample_lens = list(batch.sample_lens) if hasattr(batch, "sample_lens") else []
            actual_sequence_length = int(batch.sequence_length) if hasattr(batch, "sequence_length") else 0
        except Exception:
            actual_pair_id = None
            actual_sample_index = None
            actual_sample_lens = []
            actual_sequence_length = 0

        data = _prepare_batch(batch, vae_model, device, model_dtype)
        optimizer.zero_grad(set_to_none=True)
        loss, scalars = _forward_loss(model, data)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {scalars}")
        loss.backward()

        num_lora_A_nonzero = 0
        num_lora_B_nonzero = 0
        num_existing_nonzero = 0
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            g_l2 = float(
                torch.linalg.vector_norm(
                    p.grad.detach().to(torch.float32).reshape(-1)
                ).item()
            )
            if g_l2 > 0.0:
                if name.endswith(".lora_A"):
                    num_lora_A_nonzero += 1
                elif name.endswith(".lora_B"):
                    num_lora_B_nonzero += 1
                elif name in existing_names:
                    num_existing_nonzero += 1

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            .detach().cpu()
        )
        optimizer.step()

        _append_jsonl(train_log_path, {
            "global_step": int(step),
            "step_seed": int(step_seed),
            "timesteps_hash": _step_ts_hash,
            "pair_id": actual_pair_id,           # actual from batch, not inferred
            "sample_index_in_manifest": actual_sample_index,
            "batch_sample_lens": actual_sample_lens,   # real per-sample token counts in pack
            "batch_sequence_length": actual_sequence_length,  # total packed sequence length
            "loss": float(scalars.get("total", float(loss.detach().item()))),
            "mse": float(scalars.get("mse", float(loss.detach().item()))),
            "ce": float(scalars.get("ce", 0.0)),
            "grad_norm": float(grad_norm),
            "learning_rate": float(args.lr),
            "lora_A_nonzero_grads": int(num_lora_A_nonzero),
            "lora_B_nonzero_grads": int(num_lora_B_nonzero),
            "existing_nonzero_grads": int(num_existing_nonzero),
            "cuda_memory_allocated_mib": (
                torch.cuda.memory_allocated() if device.type == "cuda" else 0
            ) / (1024 * 1024),
            "wall_seconds_this_step": time.time() - step_wall_start,
            "created_at_iso": _now_iso(),
        })

        del data, loss, scalars
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if step in save_step_set:
            checkpoint_records.append(
                _save_step_trainable(
                    step, model, args.output_dir, args, target_names, existing_names, lora_names,
                )
            )
        if step in eval_step_set:
            evalr = _mean_train_mse(
                model, vae_model, eval_loader, device, model_dtype, args.eval_max_batches
            )
            eval_records.append({"step": step, **evalr})

    # 9. Post-training exposure audit from actual train_log
    exposure = _exposure_audit(train_log_path, args)
    _atomic_write_json(args.output_dir / "exposure_audit.json", exposure)

    peak_alloc_mib = (
        torch.cuda.max_memory_allocated() if device.type == "cuda" else 0
    ) / (1024 * 1024)
    peak_reserved_mib = (
        torch.cuda.max_memory_reserved() if device.type == "cuda" else 0
    ) / (1024 * 1024)

    report: Dict[str, Any] = {
        "status": "passed",
        "mode": "full_training",
        "created_at_iso": _now_iso(),
        "args": _serialize_args_json(args),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "b_step000_sha256": b_step000_sha,
        "base_checkpoint_report": base_checkpoint_report,
        "resume_report": resume_report,
        "config_diff": config_diff,
        "param_audit": {k: v for k, v in param_audit.items() if k != "dtype_desc"},
        "optim_audit": optim_audit,
        "adamw_pre_step_gate": adamw_gate,
        "step000_vs_b_step000": step000_vs_b,
        "checkpoint_records": checkpoint_records,
        "eval_records": eval_records,
        "exposure_audit": exposure,
        "cuda_memory_peak_mib": {
            "allocated": peak_alloc_mib,
            "reserved": peak_reserved_mib,
        },
    }
    _atomic_write_json(args.output_dir / "micro4_full_training_report.json", report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MICRO4_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-report", type=Path, default=None)
    parser.add_argument("--acknowledge-full-training", action="store_true")
    parser.add_argument("--resume-trainable-checkpoint", type=Path, default=B_STEP000_CKPT)
    # Base paths (inherit formal B defaults)
    parser.add_argument("--llm-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vae-path", default=str(Path(DEFAULT_MODEL_PATH) / "ae.safetensors"))
    parser.add_argument("--checkpoint-path", default=str(Path(DEFAULT_MODEL_PATH) / "ema.safetensors"))
    parser.add_argument("--allow-non-dockerdata", action="store_true")
    # Training config — must match formal B exactly (except allowed diffs)
    parser.add_argument("--variant", default="paired_cf_mvp", choices=["paired_cf_mvp"])
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=list(MICRO4_EVAL_STEPS))
    parser.add_argument("--save-steps", type=int, nargs="+", default=list(MICRO4_SAVE_STEPS))
    parser.add_argument("--eval-max-batches", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vit-image-size", type=int, default=224)
    parser.add_argument("--vit-min-image-size", type=int, default=224)
    parser.add_argument("--visual-image-size", type=int, default=1024)
    parser.add_argument("--visual-vae-min-image-size", type=int, default=512)
    parser.add_argument("--visual-vit-image-size", type=int, default=980)
    parser.add_argument("--visual-vit-min-image-size", type=int, default=224)
    # Must match formal B exactly
    parser.add_argument("--expected-num-tokens", type=int, default=512)
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=1536)
    parser.add_argument("--max-num-tokens", type=int, default=1536)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--model-device-map", default="single")
    parser.add_argument("--device-map-max-memory", default="34GiB")
    parser.add_argument("--trainable-name-regex",
                        default=r"^(time_embedder|latent_pos_embed|vit_pos_embed|vae2llm|llm2vae|connector)\.")
    parser.add_argument("--lora-layers", type=int, nargs="+", default=[24, 25, 26, 27])
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--cfg-img-scale", type=float, default=2.0)
    parser.add_argument("--cfg-interval-start", type=float, default=0.0)
    parser.add_argument("--cfg-interval-end", type=float, default=1.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    parser.add_argument("--num-timesteps", type=int, default=25)
    parser.add_argument("--cfg-renorm-min", type=float, default=0.0)
    parser.add_argument("--cfg-renorm-type", default="text_channel")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--visual-total", type=int, default=12)
    parser.add_argument("--visual-per-type", type=int, default=3)
    parser.add_argument("--no-visualizations", action="store_true")
    parser.add_argument("--require-paired-vit", dest="require_paired_vit",
                        action="store_true", default=True)
    args = parser.parse_args()

    # Resolve output dir
    if args.output_dir is None:
        if args.dry_run:
            args.output_dir = Path(str(BASE_OUTPUT_STEM) + "_preflight")
        else:
            args.output_dir = Path(str(BASE_OUTPUT_STEM) + "_lr3e5_500steps")

    # Resolve dry-run report path
    if args.dry_run_report is None:
        args.dry_run_report = Path(str(BASE_OUTPUT_STEM) + "_preflight/micro4_dry_run_report.json")

    # Normalise step lists
    args.eval_steps = sorted({int(s) for s in args.eval_steps if int(s) >= 0})
    args.save_steps = sorted({int(s) for s in args.save_steps if int(s) >= 0})
    if 0 not in args.eval_steps:
        args.eval_steps = [0] + args.eval_steps
    if 0 not in args.save_steps:
        args.save_steps = [0] + args.save_steps
    return args


def main() -> None:
    args = parse_args()
    try:
        if args.dry_run:
            report = _run_dry_run(args)
            print(json.dumps({
                "status": report["status"],
                "mode": "dry_run",
                "manifest_sha256": report["manifest_sha256"],
                "b_step000_sha256": report["b_step000_sha256"],
                "step000_path": report["step000_checkpoint"]["path"],
                "step000_sha256": report["step000_checkpoint"]["sha256"],
                "step000_vs_b_all_equal": report["step000_vs_b_step000"]["all_tensors_equal"],
                "step000_vs_b_max_abs_diff": report["step000_vs_b_step000"]["max_abs_diff"],
                "config_diff_status": report["config_diff"]["status"],
                "config_diff_forbidden": report["config_diff"]["forbidden_diffs"],
                "forward_mse": report["forward"]["mse_mean"],
                "forward_finite": report["forward"]["mse_all_finite"],
                "sample_lens": [
                    {"pair_id": s["pair_id"], "num_tokens": s["num_tokens"]}
                    for s in report["sample_lens_probe"]["samples"]
                ],
                "report_path": str(args.output_dir / "micro4_dry_run_report.json"),
            }, indent=2))
        else:
            report = _run_full_training(args)
            print(json.dumps({
                "status": report["status"],
                "mode": "full_training",
                "exposure_audit": report["exposure_audit"],
                "report_path": str(args.output_dir / "micro4_full_training_report.json"),
            }, indent=2))
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
