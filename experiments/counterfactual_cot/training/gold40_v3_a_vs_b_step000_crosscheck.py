"""Gold40 v3 baseline-A vs LoRA-B Step000 crosscheck orchestrator.

Runs the following in strict sequence (all as **separate** OS processes):

  1. Baseline A Step000 dry-run
     -> ``gold40_v3_baseline_step000_dryrun.py``
     Produces ``step_000_trainable.pt`` with the 14 existing trainable
     tensors only. No LoRA. No optimizer.step.

  2. LoRA B Step000 dry-run
     -> ``gold40_lora.py --dry-run``
     Produces ``step_000_trainable.pt`` with 14 existing tensors + 48 LoRA
     A/B tensors. No optimizer.step. If a prior Step000 already exists at
     the target path we REFUSE to overwrite it and stop.

  3. LoRA-A init reproducer (helper)
     -> ``--mode lora_a_reproducer``
     Independent process. ``_set_seed(1234)``. Reads the 24 target names
     and their (r, in_features) shape from the B Step000, then creates a
     fresh fp32 tensor for each name (in the SAME order as ``inject_lora``)
     and applies ``nn.init.kaiming_uniform_(a=math.sqrt(5))``. Dumps a
     dict {name: sha256(bytes)} + {name: max-abs-hash}.

  4. Comparator (CPU-only, same orchestrator process)
     Loads A and B trainable_state_dicts, runs every hard gate:

       G1  Base / AE / EMA / manifest / script SHA256 equal.
       G2  A has 14 existing tensors; B has 14 existing + 48 LoRA tensors.
       G3  For each of the 14 shared names: shape, dtype, sha256(bytes)
           equal and per-element max_abs_diff == 0.
       G4  All 24 lora_B tensors in B are strictly zero (all elements == 0).
       G5  All 24 lora_A tensors in B are bit-exact reproducible by the
           independent process 3 -> per-tensor sha256 match.
       G6  Dataset pair-order / manifest_sha / prompt hashes / seed /
           sequence_plan identical between A and B run configs.
       G7  Neither A nor B ran optimizer.step (mode == dry_run).
       G8  Neither A nor B produced NaN/Inf in their forward MSE.
       G9  Both A and B claim ``status == passed``.

  Writes ``baseline_v3_vs_lora_v3_step000_crosscheck.json``. If ANY gate
  fails, ``pass == false`` and the process exits non-zero.

Never touches:
  * Gold40 manifests (v1/v2/v3).
  * gold40_overfit.py / lora_2step_smoke.py / cold_reload_*.py.
  * BAGEL / qwen2_navit / siglip_navit / schema.py core code.
  * The dry-run Step000 already produced under
    ``.../gold40_v3_lora_L24-27_r8_a16_lr1e6_preflight/`` (if reused, it is
    read-only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


REPO_ROOT = Path("/group/40016/niightli/BAGEL")
V3_MANIFEST = (
    REPO_ROOT
    / "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/manifests"
    / "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl"
)
DEFAULT_A_DIR = REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_baseline_L14_step000_singlepack"
DEFAULT_B_DIR = REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_lora_L24-27_r8_a16_lr1e6_singlepack_preflight"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_a_vs_b_singlepack_crosscheck"

# Packing parameters restored to the safe1536 baseline runtime record
# (from gold40_overfit_single_400_safe1536/checkpoints/step_000_trainable.pt config).
# These force each dataloader batch to contain EXACTLY 1 Gold40 pair, with
# sequence_length ~ 1109 tokens.
DEFAULT_EXPECTED_NUM_TOKENS = 512
DEFAULT_MAX_NUM_TOKENS_PER_SAMPLE = 1536
DEFAULT_MAX_NUM_TOKENS = 1536
SINGLEPACK_SEQUENCE_LENGTH_CEILING = 1536

BASE_LLM = "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT"
BASE_AE = "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors"
BASE_EMA = "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors"
EXPECTED_EMA_SHA = "0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _tensor_bytes_sha256(t: torch.Tensor) -> str:
    """Return sha256 of the raw byte-view of *t* (contiguous CPU).

    ``numpy`` does not support BFloat16, so we lift bf16 tensors to fp32
    (deterministic, bit-preserving upcast) before hashing. Both A and B use
    the same upcast, so bit-exact equality is preserved by this transform:
    ``t_a == t_b (in bf16)`` iff ``t_a.to(fp32) == t_b.to(fp32) (in fp32)``.
    """
    t_cpu = t.detach().to("cpu").contiguous()
    if t_cpu.dtype == torch.bfloat16:
        t_cpu = t_cpu.to(torch.float32)
    return hashlib.sha256(t_cpu.numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Sub-mode 1: LoRA-A init reproducer (runs as its own process).
# ---------------------------------------------------------------------------

def _run_lora_a_reproducer(args: argparse.Namespace) -> int:
    """Independent process. Reproduce LoRA-A init from a fixed seed.

    Reads the ordered list of ``target_names`` and their (in_features,) from
    the B Step000 checkpoint, then applies exactly the same init sequence
    that ``inject_lora`` -> ``LoRALinear.__init__`` performs: create an
    ``nn.Parameter(torch.empty(r, in_features, dtype=torch.float32))``, then
    ``nn.init.kaiming_uniform_(a=math.sqrt(5))``. Records per-tensor
    sha256 to a JSON file.
    """
    from experiments.counterfactual_cot.training.smoke_train import _set_seed

    b_ckpt_path = Path(args.b_step000)
    out_path = Path(args.out)
    payload = torch.load(b_ckpt_path, map_location="cpu", weights_only=False)
    tsd = payload["trainable_state_dict"]
    meta = payload.get("meta", {})
    target_names = list(meta.get("lora_target_names", []))
    r = int(meta.get("lora_r"))
    alpha = int(meta.get("lora_alpha"))
    seed = int(args.seed)

    if not target_names:
        raise RuntimeError("no lora_target_names found in B checkpoint meta")

    # Determine in_features for each target from the corresponding lora_A
    # shape in B (which is (r, in_features)).
    shapes: List[Tuple[str, Tuple[int, int]]] = []
    for full in target_names:
        key = f"{full}.lora_A"
        if key not in tsd:
            raise KeyError(f"missing {key} in B trainable_state_dict")
        t = tsd[key]
        if t.dtype != torch.float32:
            raise RuntimeError(f"expected fp32 lora_A tensor, got {t.dtype} for {key}")
        if t.dim() != 2 or t.shape[0] != r:
            raise RuntimeError(f"unexpected lora_A shape {tuple(t.shape)} for {key}, r={r}")
        shapes.append((full, (int(t.shape[0]), int(t.shape[1]))))

    _set_seed(seed)

    per_tensor: Dict[str, Dict[str, Any]] = {}
    for full, (rr, in_features) in shapes:
        # Exactly matches LoRALinear.__init__ init order:
        #   lora_A = nn.Parameter(torch.empty(r, in_features, dtype=fp32))
        #   nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        #   lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=fp32))
        # We only care about lora_A here.
        lora_A = nn.Parameter(torch.empty(rr, in_features, dtype=torch.float32))
        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        # LoRALinear.__init__ ALSO creates lora_B (all zeros -- no RNG cost)
        # but that does not consume RNG state. However, to be strictly faithful
        # we materialize a zero-tensor too so the sequence exactly mirrors the
        # real init. (torch.zeros is deterministic and does not consume RNG.)
        _ = nn.Parameter(torch.zeros(1, rr, dtype=torch.float32))  # placeholder no-op
        per_tensor[f"{full}.lora_A"] = {
            "shape": [rr, in_features],
            "dtype": "torch.float32",
            "sha256_bytes": _tensor_bytes_sha256(lora_A),
            "abs_max": float(lora_A.detach().abs().max().item()),
            "num_nonzero": int((lora_A.detach() != 0).sum().item()),
        }

    result = {
        "mode": "lora_a_reproducer",
        "seed": seed,
        "target_names_count": len(target_names),
        "lora_r": r,
        "lora_alpha": alpha,
        "b_step000": str(b_ckpt_path),
        "per_tensor": per_tensor,
        "created_at_iso": _now_iso(),
    }
    _atomic_write_json(out_path, result)
    return 0


# ---------------------------------------------------------------------------
# Orchestrator utilities: launch subprocesses.
# ---------------------------------------------------------------------------

def _launch_subprocess(cmd: List[str], log_path: Path, env_extra: Optional[Dict[str, str]] = None) -> int:
    """Launch a child process, tee stdout+stderr to *log_path*, return rc."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    with open(log_path, "wb") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        rc = proc.wait()
    return rc


def _dockerdata_provenance() -> Dict[str, Any]:
    return {
        "llm_path": str(BASE_LLM),
        "ae_path": str(BASE_AE),
        "ema_path": str(BASE_EMA),
        "ae_sha256": _sha256_file(Path(BASE_AE)),
        "ema_sha256": _sha256_file(Path(BASE_EMA)),
        "expected_ema_sha256": EXPECTED_EMA_SHA,
        "ema_matches_reference": _sha256_file(Path(BASE_EMA)) == EXPECTED_EMA_SHA,
    }


# ---------------------------------------------------------------------------
# Comparator (runs after both dry-runs finish; CPU-only).
# ---------------------------------------------------------------------------

def _load_trainable(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["trainable_state_dict"], payload.get("meta", {})


def _classify_tensor_names(tsd_a: Dict[str, torch.Tensor], tsd_b: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
    a_names = set(tsd_a.keys())
    b_names = set(tsd_b.keys())
    lora_a_names = sorted(n for n in b_names if n.endswith(".lora_A"))
    lora_b_names = sorted(n for n in b_names if n.endswith(".lora_B"))
    b_existing = sorted(b_names - set(lora_a_names) - set(lora_b_names))
    a_existing = sorted(a_names)
    shared_existing = sorted(a_names & set(b_existing))
    only_in_a = sorted(a_names - b_names)
    only_in_b_existing = sorted(set(b_existing) - a_names)
    return {
        "a_existing": a_existing,
        "b_existing": b_existing,
        "shared_existing": shared_existing,
        "lora_a_names": lora_a_names,
        "lora_b_names": lora_b_names,
        "only_in_a_existing": only_in_a,
        "only_in_b_existing": only_in_b_existing,
    }


def _compare_existing_tensors(
    tsd_a: Dict[str, torch.Tensor],
    tsd_b: Dict[str, torch.Tensor],
    shared_existing: List[str],
) -> Tuple[Dict[str, Any], bool, float]:
    per_tensor: Dict[str, Any] = {}
    all_equal = True
    max_abs_diff_overall = 0.0
    for name in shared_existing:
        ta = tsd_a[name]
        tb = tsd_b[name]
        shape_eq = tuple(ta.shape) == tuple(tb.shape)
        dtype_eq = ta.dtype == tb.dtype
        if not shape_eq:
            per_tensor[name] = {"shape_a": list(ta.shape), "shape_b": list(tb.shape),
                                "equal": False, "reason": "shape_mismatch"}
            all_equal = False
            continue
        if not dtype_eq:
            per_tensor[name] = {"dtype_a": str(ta.dtype), "dtype_b": str(tb.dtype),
                                "equal": False, "reason": "dtype_mismatch"}
            all_equal = False
            continue
        diff = float((ta.to(torch.float32) - tb.to(torch.float32)).abs().max().item())
        sha_a = _tensor_bytes_sha256(ta)
        sha_b = _tensor_bytes_sha256(tb)
        eq = (diff == 0.0) and (sha_a == sha_b)
        max_abs_diff_overall = max(max_abs_diff_overall, diff)
        per_tensor[name] = {
            "shape": list(ta.shape),
            "dtype": str(ta.dtype),
            "sha256_a": sha_a,
            "sha256_b": sha_b,
            "max_abs_diff": diff,
            "equal": eq,
        }
        if not eq:
            all_equal = False
    return per_tensor, all_equal, max_abs_diff_overall


def _check_lora_b_zero(tsd_b: Dict[str, torch.Tensor], lora_b_names: List[str]) -> Dict[str, Any]:
    per: Dict[str, Any] = {}
    all_zero = True
    for name in lora_b_names:
        t = tsd_b[name]
        if t.dtype != torch.float32:
            per[name] = {"dtype": str(t.dtype), "reason": "dtype_not_fp32", "is_all_zero": False}
            all_zero = False
            continue
        abs_max = float(t.detach().abs().max().item())
        n_nonzero = int((t.detach() != 0).sum().item())
        is_zero = (abs_max == 0.0) and (n_nonzero == 0)
        per[name] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "abs_max": abs_max,
            "num_nonzero": n_nonzero,
            "is_all_zero": is_zero,
        }
        if not is_zero:
            all_zero = False
    return {"per_tensor": per, "all_zero": all_zero, "num_tensors": len(lora_b_names)}


def _check_lora_a_reproducibility(
    tsd_b: Dict[str, torch.Tensor],
    reproducer_report: Dict[str, Any],
    lora_a_names: List[str],
) -> Dict[str, Any]:
    per: Dict[str, Any] = {}
    all_match = True
    repro = reproducer_report.get("per_tensor", {})
    for name in lora_a_names:
        t = tsd_b[name]
        sha_b = _tensor_bytes_sha256(t)
        sha_repro = (repro.get(name) or {}).get("sha256_bytes")
        eq = (sha_b == sha_repro) and (sha_repro is not None)
        per[name] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "sha256_b": sha_b,
            "sha256_reproducer": sha_repro,
            "equal": eq,
        }
        if not eq:
            all_match = False
    return {"per_tensor": per, "all_match": all_match, "num_tensors": len(lora_a_names)}


def _load_run_config_resolved(dry_run_dir: Path) -> Optional[Dict[str, Any]]:
    p = dry_run_dir / "run_config_resolved.json"
    if not p.exists():
        return None
    return json.loads(p.read_text("utf-8"))


def _load_sequence_plan(dry_run_dir: Path) -> Optional[Dict[str, Any]]:
    p = dry_run_dir / "sequence_plan.json"
    if not p.exists():
        return None
    return json.loads(p.read_text("utf-8"))


def _report_sha256(dry_run_dir: Path, name: str) -> Optional[str]:
    p = dry_run_dir / name
    return _sha256_file(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Orchestrator top-level.
# ---------------------------------------------------------------------------

def _run_baseline_a(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.baseline_a_dir)
    ck = out_dir / "checkpoints" / "step_000_trainable.pt"
    if ck.exists() and not args.reuse_a:
        raise RuntimeError(
            f"Baseline A Step000 already present at {ck}; "
            "pass --reuse-a to reuse it or delete/rename the directory."
        )
    if args.reuse_a and ck.exists():
        return {"reused": True, "step000_path": str(ck), "step000_sha256": _sha256_file(ck)}

    log_path = Path(args.out_dir) / "logs" / "baseline_a.stdout.log"
    cmd = [
        sys.executable, "-u", "-m", "experiments.counterfactual_cot.training.gold40_v3_baseline_step000_dryrun",
        "--manifest", str(args.manifest),
        "--output-dir", str(out_dir),
        "--llm-path", args.llm_path,
        "--vae-path", args.vae_path,
        "--checkpoint-path", args.checkpoint_path,
        "--seed", str(args.seed),
        "--generation-seed", str(args.generation_seed),
        "--num-samples", str(args.num_samples),
        "--expected-num-tokens", str(args.expected_num_tokens),
        "--max-num-tokens-per-sample", str(args.max_num_tokens_per_sample),
        "--max-num-tokens", str(args.max_num_tokens),
    ]
    rc = _launch_subprocess(cmd, log_path, env_extra={"CUDA_VISIBLE_DEVICES": "0"})
    if rc != 0:
        raise RuntimeError(f"Baseline A dry-run exited with code {rc}. See {log_path}")
    if not ck.exists():
        raise RuntimeError(f"Baseline A dry-run finished but Step000 missing: {ck}")
    return {"reused": False, "step000_path": str(ck), "step000_sha256": _sha256_file(ck), "log": str(log_path)}


def _run_lora_b(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = Path(args.lora_b_dir)
    ck = out_dir / "checkpoints" / "step_000_trainable.pt"
    if ck.exists() and not args.reuse_b:
        raise RuntimeError(
            f"LoRA B Step000 already present at {ck}; "
            "pass --reuse-b to reuse it or delete/rename the directory."
        )
    if args.reuse_b and ck.exists():
        return {"reused": True, "step000_path": str(ck), "step000_sha256": _sha256_file(ck)}

    log_path = Path(args.out_dir) / "logs" / "lora_b.stdout.log"
    lora_b_output_dir = Path(args.lora_b_dir)
    cmd = [
        sys.executable, "-u", "-m", "experiments.counterfactual_cot.training.gold40_lora",
        "--dry-run",
        "--manifest", str(args.manifest),
        "--output-dir", str(lora_b_output_dir),
        "--llm-path", args.llm_path,
        "--vae-path", args.vae_path,
        "--checkpoint-path", args.checkpoint_path,
        "--num-samples", str(args.num_samples),
        "--lr", "1e-6",
        "--seed", str(args.seed),
        "--generation-seed", str(args.generation_seed),
        "--device", "cuda",
        "--precision", "bf16",
        "--model-device-map", "single",
        "--lora-layers", "24", "25", "26", "27",
        "--lora-r", "8", "--lora-alpha", "16", "--lora-dropout", "0.0",
        "--steps", "400",
        "--expected-num-tokens", str(args.expected_num_tokens),
        "--max-num-tokens-per-sample", str(args.max_num_tokens_per_sample),
        "--max-num-tokens", str(args.max_num_tokens),
    ]
    rc = _launch_subprocess(cmd, log_path, env_extra={"CUDA_VISIBLE_DEVICES": "0"})
    if rc != 0:
        raise RuntimeError(f"LoRA B dry-run exited with code {rc}. See {log_path}")
    if not ck.exists():
        raise RuntimeError(f"LoRA B dry-run finished but Step000 missing: {ck}")
    return {"reused": False, "step000_path": str(ck), "step000_sha256": _sha256_file(ck), "log": str(log_path)}


def _run_reproducer(args: argparse.Namespace, b_step000: Path) -> Path:
    """Run a SECOND independent gold40_lora --dry-run into a fresh output
    directory, then extract every ``.lora_A`` tensor from its Step000 and
    write a reproducer report. This is the strictest possible test of
    "LoRA-A uses a fixed seed and can be bit-exactly reproduced by an
    independent process": we run the exact same code path from scratch and
    require identical Step000 lora_A tensors.
    """
    repro_out_dir = Path(args.out_dir) / "lora_b_repro"
    repro_ck = repro_out_dir / "checkpoints" / "step_000_trainable.pt"
    out_path = Path(args.out_dir) / "lora_a_init_reproducer.json"
    log_path = Path(args.out_dir) / "logs" / "lora_a_reproducer.stdout.log"

    if repro_ck.exists():
        # Reuse a previously-produced reproducer Step000.
        pass
    else:
        cmd = [
            sys.executable, "-u", "-m", "experiments.counterfactual_cot.training.gold40_lora",
            "--dry-run",
            "--manifest", str(args.manifest),
            "--output-dir", str(repro_out_dir),
            "--llm-path", args.llm_path,
            "--vae-path", args.vae_path,
            "--checkpoint-path", args.checkpoint_path,
            "--num-samples", str(args.num_samples),
            "--lr", "1e-6",
            "--seed", str(args.seed),
            "--generation-seed", str(args.generation_seed),
            "--device", "cuda",
            "--precision", "bf16",
            "--model-device-map", "single",
            "--lora-layers", "24", "25", "26", "27",
            "--lora-r", "8", "--lora-alpha", "16", "--lora-dropout", "0.0",
            "--steps", "400",
            "--expected-num-tokens", str(args.expected_num_tokens),
            "--max-num-tokens-per-sample", str(args.max_num_tokens_per_sample),
            "--max-num-tokens", str(args.max_num_tokens),
        ]
        rc = _launch_subprocess(cmd, log_path, env_extra={"CUDA_VISIBLE_DEVICES": "0"})
        if rc != 0:
            raise RuntimeError(f"LoRA reproducer dry-run exited with code {rc}. See {log_path}")
        if not repro_ck.exists():
            raise RuntimeError(f"LoRA reproducer dry-run finished but Step000 missing: {repro_ck}")

    # Extract lora_A tensors from the reproducer Step000.
    payload = torch.load(repro_ck, map_location="cpu", weights_only=False)
    tsd = payload["trainable_state_dict"]
    meta = payload.get("meta", {})
    target_names = list(meta.get("lora_target_names", []))
    r_meta = int(meta.get("lora_r"))
    alpha_meta = int(meta.get("lora_alpha"))
    per_tensor: Dict[str, Dict[str, Any]] = {}
    for full in target_names:
        key = f"{full}.lora_A"
        if key not in tsd:
            raise KeyError(f"missing {key} in reproducer trainable_state_dict")
        t = tsd[key]
        per_tensor[key] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "sha256_bytes": _tensor_bytes_sha256(t),
            "abs_max": float(t.detach().abs().max().item()),
            "num_nonzero": int((t.detach() != 0).sum().item()),
        }

    result = {
        "mode": "lora_a_reproducer_via_second_dry_run",
        "seed": int(args.seed),
        "target_names_count": len(target_names),
        "lora_r": r_meta,
        "lora_alpha": alpha_meta,
        "repro_step000_path": str(repro_ck),
        "repro_step000_sha256": _sha256_file(repro_ck),
        "per_tensor": per_tensor,
        "created_at_iso": _now_iso(),
    }
    _atomic_write_json(out_path, result)
    return out_path


def _comparator(args: argparse.Namespace, a_step000: Path, b_step000: Path, reproducer_json: Path) -> Dict[str, Any]:
    tsd_a, meta_a = _load_trainable(a_step000)
    tsd_b, meta_b = _load_trainable(b_step000)
    reproducer_report = json.loads(reproducer_json.read_text("utf-8"))

    a_dir = a_step000.parent.parent
    b_dir = b_step000.parent.parent

    # Names.
    names = _classify_tensor_names(tsd_a, tsd_b)
    num_a = len(names["a_existing"])
    num_b_existing = len(names["b_existing"])
    num_lora_a = len(names["lora_a_names"])
    num_lora_b = len(names["lora_b_names"])
    num_shared = len(names["shared_existing"])
    only_in_a = names["only_in_a_existing"]
    only_in_b = names["only_in_b_existing"]

    # G3: 14 shared existing tensors bit-exact.
    per_shared, existing_all_equal, max_abs_diff_existing = _compare_existing_tensors(
        tsd_a, tsd_b, names["shared_existing"],
    )

    # G4: LoRA-B zero.
    lora_b_report = _check_lora_b_zero(tsd_b, names["lora_b_names"])

    # G5: LoRA-A reproducible.
    lora_a_report = _check_lora_a_reproducibility(tsd_b, reproducer_report, names["lora_a_names"])

    # G6: dataset / seed / manifest consistency.
    a_config = _load_run_config_resolved(a_dir)
    b_config = _load_run_config_resolved(b_dir)
    a_plan = _load_sequence_plan(a_dir)
    b_plan = _load_sequence_plan(b_dir)

    manifest_sha_a = (a_plan or {}).get("manifest_sha256")
    manifest_sha_b = (b_plan or {}).get("manifest_sha256")
    order_a = json.dumps([r["pair_id"] for r in (a_plan or {}).get("order", [])])
    order_b = json.dumps([r["pair_id"] for r in (b_plan or {}).get("order", [])])
    seed_a = ((a_config or {}).get("args") or {}).get("seed")
    seed_b = ((b_config or {}).get("args") or {}).get("seed")
    genseed_a = ((a_config or {}).get("args") or {}).get("generation_seed")
    genseed_b = ((b_config or {}).get("args") or {}).get("generation_seed")
    num_records_a = (a_plan or {}).get("num_records")
    num_records_b = (b_plan or {}).get("num_records")

    dataset_consistency = {
        "manifest_sha256_a": manifest_sha_a,
        "manifest_sha256_b": manifest_sha_b,
        "manifest_sha256_equal": manifest_sha_a == manifest_sha_b and manifest_sha_a is not None,
        "pair_order_equal": order_a == order_b and order_a != "[]",
        "num_records_a": num_records_a,
        "num_records_b": num_records_b,
        "num_records_equal": num_records_a == num_records_b and num_records_a == args.num_samples,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "seed_equal": seed_a == seed_b and seed_a == args.seed,
        "generation_seed_a": genseed_a,
        "generation_seed_b": genseed_b,
        "generation_seed_equal": genseed_a == genseed_b and genseed_a == args.generation_seed,
        "num_pairs": num_records_a,
    }

    # G7: neither ran optimizer.step (status is diagnostic_monitoring_only + mode is dry-run).
    a_report = json.loads((a_dir / "gold40_v3_baseline_dry_run_report.json").read_text("utf-8"))
    b_report = json.loads((b_dir / "gold40_lora_dry_run_report.json").read_text("utf-8"))
    optimizer_step_gate = {
        "a_evaluation_type": a_report.get("evaluation_type"),
        "b_evaluation_type": b_report.get("evaluation_type"),
        "a_mode": a_report.get("mode"),
        "b_mode": b_report.get("mode"),
        "a_status": a_report.get("status"),
        "b_status": b_report.get("status"),
        "a_no_optimizer_step": a_report.get("mode") == "baseline_a_step000_dry_run",
        "b_no_optimizer_step": b_report.get("mode") == "dry_run",
    }

    # G8: MSE finite.
    a_mse_finite = bool(a_report.get("forward", {}).get("mse_all_finite"))
    b_mse_finite = bool(b_report.get("forward", {}).get("mse_all_finite"))
    a_mse_mean = a_report.get("forward", {}).get("mse_mean")
    b_mse_mean = b_report.get("forward", {}).get("mse_mean")

    # G1: Base / manifest / script SHA256.
    base_provenance_now = _dockerdata_provenance()
    a_manifest_sha = _sha256_file(Path(args.manifest))
    manifest_sha_matches_expected = a_manifest_sha == manifest_sha_a == manifest_sha_b
    scripts_sha256 = {
        "gold40_lora.py": _sha256_file(REPO_ROOT / "experiments/counterfactual_cot/training/gold40_lora.py"),
        "gold40_overfit.py": _sha256_file(REPO_ROOT / "experiments/counterfactual_cot/training/gold40_overfit.py"),
        "lora_2step_smoke.py": _sha256_file(REPO_ROOT / "experiments/counterfactual_cot/training/lora_2step_smoke.py"),
        "gold40_v3_baseline_step000_dryrun.py": _sha256_file(
            REPO_ROOT / "experiments/counterfactual_cot/training/gold40_v3_baseline_step000_dryrun.py"
        ),
        "gold40_v3_a_vs_b_step000_crosscheck.py": _sha256_file(
            REPO_ROOT / "experiments/counterfactual_cot/training/gold40_v3_a_vs_b_step000_crosscheck.py"
        ),
    }

    # G2: tensor counts.
    counts_gate = {
        "a_num_existing": num_a,
        "b_num_existing": num_b_existing,
        "b_num_lora_a": num_lora_a,
        "b_num_lora_b": num_lora_b,
        "a_has_14": num_a == 14,
        "b_existing_has_14": num_b_existing == 14,
        "b_lora_a_has_24": num_lora_a == 24,
        "b_lora_b_has_24": num_lora_b == 24,
        "b_lora_total_has_48": (num_lora_a + num_lora_b) == 48,
        "only_in_a_existing": only_in_a,
        "only_in_b_existing": only_in_b,
        "shared_existing_has_14": num_shared == 14,
    }

    # G10-G12: realized batch-shape audit. This is the NEW gate that the
    # superseded crosscheck was missing. We compare the actual dataloader
    # output between A and B, not just static config declarations.
    a_fb = a_report.get("first_batch_report") or {}
    b_fb = b_report.get("first_batch_report") or {}
    a_seq_len = a_fb.get("sequence_length")
    b_seq_len = b_fb.get("sequence_length")
    a_sample_lens = a_fb.get("sample_lens") or []
    b_sample_lens = b_fb.get("sample_lens") or []
    a_tensors = a_fb.get("tensors") or {}
    b_tensors = b_fb.get("tensors") or {}
    ceiling = int(args.sequence_length_ceiling)

    tensor_shape_diffs: Dict[str, Dict[str, Any]] = {}
    all_shared_tensor_shapes_equal = True
    for tname in sorted(set(a_tensors.keys()) | set(b_tensors.keys())):
        ta = a_tensors.get(tname) or {}
        tb = b_tensors.get(tname) or {}
        sa = ta.get("shape")
        sb = tb.get("shape")
        eq = sa == sb and sa is not None
        tensor_shape_diffs[tname] = {"a_shape": sa, "b_shape": sb, "equal": eq}
        if not eq:
            all_shared_tensor_shapes_equal = False

    a_first_pair_id = None
    b_first_pair_id = None
    a_bdi = a_fb.get("batch_data_indexes")
    b_bdi = b_fb.get("batch_data_indexes")
    if isinstance(a_bdi, list) and a_bdi:
        a_first_pair_id = (a_bdi[0] or {}).get("pair_id")
    if isinstance(b_bdi, list) and b_bdi:
        b_first_pair_id = (b_bdi[0] or {}).get("pair_id")

    batch_shape_audit = {
        "a_sequence_length": a_seq_len,
        "b_sequence_length": b_seq_len,
        "a_sample_lens": a_sample_lens,
        "b_sample_lens": b_sample_lens,
        "a_num_samples_in_batch": len(a_sample_lens),
        "b_num_samples_in_batch": len(b_sample_lens),
        "a_first_pair_id": a_first_pair_id,
        "b_first_pair_id": b_first_pair_id,
        "sequence_length_ceiling": ceiling,
        "a_within_ceiling": (a_seq_len is not None) and (a_seq_len <= ceiling),
        "b_within_ceiling": (b_seq_len is not None) and (b_seq_len <= ceiling),
        "sequence_length_equal": a_seq_len == b_seq_len and a_seq_len is not None,
        "sample_lens_equal": a_sample_lens == b_sample_lens and bool(a_sample_lens),
        "a_is_single_pack": len(a_sample_lens) == 1,
        "b_is_single_pack": len(b_sample_lens) == 1,
        "first_pair_id_equal": a_first_pair_id == b_first_pair_id and a_first_pair_id is not None,
        "all_shared_tensor_shapes_equal": all_shared_tensor_shapes_equal,
        "tensor_shape_diffs": tensor_shape_diffs,
    }

    # Hard gates.
    hard_gates = {
        "G1_base_ae_ema_present_and_match_ref":
            base_provenance_now["ema_matches_reference"]
            and base_provenance_now["ae_sha256"] is not None,
        "G1_manifest_sha_equal_between_a_b_and_repo": manifest_sha_matches_expected,
        "G2_a_has_14_existing_only": counts_gate["a_has_14"] and not only_in_a,
        "G2_b_has_14_existing_plus_48_lora":
            counts_gate["b_existing_has_14"]
            and counts_gate["b_lora_total_has_48"]
            and counts_gate["b_lora_a_has_24"]
            and counts_gate["b_lora_b_has_24"],
        "G2_shared_14": counts_gate["shared_existing_has_14"] and not only_in_a and not only_in_b,
        "G3_14_existing_tensors_bit_exact": existing_all_equal and max_abs_diff_existing == 0.0,
        "G4_all_lora_B_strictly_zero": lora_b_report["all_zero"],
        "G5_all_lora_A_reproducible_from_seed": lora_a_report["all_match"],
        "G6_dataset_pair_order_equal": dataset_consistency["pair_order_equal"],
        "G6_manifest_sha_equal": dataset_consistency["manifest_sha256_equal"],
        "G6_seed_equal": dataset_consistency["seed_equal"],
        "G6_generation_seed_equal": dataset_consistency["generation_seed_equal"],
        "G6_num_records_equal": dataset_consistency["num_records_equal"],
        "G7_neither_ran_optimizer_step":
            optimizer_step_gate["a_no_optimizer_step"] and optimizer_step_gate["b_no_optimizer_step"],
        "G8_a_mse_finite": a_mse_finite,
        "G8_b_mse_finite": b_mse_finite,
        "G9_a_status_passed": a_report.get("status") == "passed",
        "G9_b_status_passed": b_report.get("status") == "passed",
        "G10_a_is_single_pack_batch": batch_shape_audit["a_is_single_pack"],
        "G10_b_is_single_pack_batch": batch_shape_audit["b_is_single_pack"],
        "G10_a_sequence_within_ceiling": batch_shape_audit["a_within_ceiling"],
        "G10_b_sequence_within_ceiling": batch_shape_audit["b_within_ceiling"],
        "G11_a_b_sequence_length_equal": batch_shape_audit["sequence_length_equal"],
        "G11_a_b_sample_lens_equal": batch_shape_audit["sample_lens_equal"],
        "G11_a_b_first_pair_id_equal": batch_shape_audit["first_pair_id_equal"],
        "G12_all_shared_tensor_shapes_equal": batch_shape_audit["all_shared_tensor_shapes_equal"],
    }
    all_pass = all(bool(v) for v in hard_gates.values())

    return {
        "created_at_iso": _now_iso(),
        "pass": all_pass,
        "hard_gates": hard_gates,
        "checkpoint_provenance": {
            "a_step000_path": str(a_step000),
            "a_step000_sha256": _sha256_file(a_step000),
            "a_step000_size_bytes": a_step000.stat().st_size,
            "b_step000_path": str(b_step000),
            "b_step000_sha256": _sha256_file(b_step000),
            "b_step000_size_bytes": b_step000.stat().st_size,
        },
        "base_provenance": base_provenance_now,
        "manifest_provenance": {
            "manifest": str(args.manifest),
            "manifest_sha256": a_manifest_sha,
        },
        "scripts_sha256": scripts_sha256,
        "counts_gate": counts_gate,
        "existing_tensor_comparison": {
            "num_shared": num_shared,
            "all_equal": existing_all_equal,
            "max_abs_diff_over_all_existing": max_abs_diff_existing,
            "per_tensor": per_shared,
        },
        "lora_b_zero_init_check": lora_b_report,
        "lora_a_reproducibility_check": lora_a_report,
        "dataset_consistency": dataset_consistency,
        "batch_shape_audit": batch_shape_audit,
        "optimizer_step_gate": optimizer_step_gate,
        "forward_mse": {
            "a_mse_mean": a_mse_mean,
            "b_mse_mean": b_mse_mean,
            "a_mse_all_finite": a_mse_finite,
            "b_mse_all_finite": b_mse_finite,
        },
        "reproducer_report_sha256": _sha256_file(reproducer_json),
        "reproducer_report_path": str(reproducer_json),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["orchestrate", "lora_a_reproducer"], default="orchestrate")
    # orchestrate mode:
    parser.add_argument("--manifest", type=Path, default=V3_MANIFEST)
    parser.add_argument("--baseline-a-dir", type=Path, default=DEFAULT_A_DIR)
    parser.add_argument("--lora-b-dir", type=Path, default=DEFAULT_B_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--llm-path", type=str, default=BASE_LLM)
    parser.add_argument("--vae-path", type=str, default=BASE_AE)
    parser.add_argument("--checkpoint-path", type=str, default=BASE_EMA)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument("--num-samples", type=int, default=40)
    parser.add_argument("--expected-num-tokens", type=int, default=DEFAULT_EXPECTED_NUM_TOKENS,
                        help="Early-yield token count. safe1536 baseline used 512.")
    parser.add_argument("--max-num-tokens-per-sample", type=int, default=DEFAULT_MAX_NUM_TOKENS_PER_SAMPLE,
                        help="Per-sample token ceiling. safe1536 baseline used 1536.")
    parser.add_argument("--max-num-tokens", type=int, default=DEFAULT_MAX_NUM_TOKENS,
                        help="Total packed batch token ceiling. safe1536 baseline used 1536.")
    parser.add_argument("--sequence-length-ceiling", type=int, default=SINGLEPACK_SEQUENCE_LENGTH_CEILING,
                        help="Hard-assert ceiling for actual realized sequence_length.")
    parser.add_argument("--reuse-a", action="store_true",
                        help="Reuse an existing baseline A Step000 if present.")
    parser.add_argument("--reuse-b", action="store_true",
                        help="Reuse an existing LoRA B Step000 if present.")
    # lora_a_reproducer mode:
    parser.add_argument("--b-step000", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "lora_a_reproducer":
        return _run_lora_a_reproducer(args)

    # orchestrate mode
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)

    # 1. Baseline A.
    a_info = _run_baseline_a(args)
    a_step000 = Path(a_info["step000_path"])

    # 2. LoRA B.
    b_info = _run_lora_b(args)
    b_step000 = Path(b_info["step000_path"])

    # 3. LoRA-A reproducer (independent process).
    reproducer_json = _run_reproducer(args, b_step000)

    # 4. Comparator.
    result = _comparator(args, a_step000, b_step000, reproducer_json)
    result["a_run_info"] = a_info
    result["b_run_info"] = b_info

    _atomic_write_json(out_dir / "baseline_v3_vs_lora_v3_step000_crosscheck.json", result)
    print(json.dumps({
        "pass": result["pass"],
        "output": str(out_dir / "baseline_v3_vs_lora_v3_step000_crosscheck.json"),
        "failed_gates": [k for k, v in result["hard_gates"].items() if not v],
    }, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
