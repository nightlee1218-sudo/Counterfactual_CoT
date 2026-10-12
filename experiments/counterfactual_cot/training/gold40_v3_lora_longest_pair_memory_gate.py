"""Gold40 v3 LoRA longest-pair memory gate on single A100-40GB.

Runs exactly TWO optimizer.step iterations on a real Gold40 v3 batch, with
step 2 forced to be the longest realized batch (max sequence_length under
single-pack packing). Records 15 memory checkpoints. Never launches 400
step training.

Reuses helpers from gold40_v3_lora_2step_smoke.py by import (no
modification to that file). The only functional change is the training
loop: step 1 is a normal single-pack draw (creates AdamW state), step 2
loops the dataloader until it yields the pre-registered longest pair_id.

Hard-gate policy is 18 items, matching the task spec.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path("/group/40016/niightli/BAGEL").resolve()
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from experiments.counterfactual_cot.training.gold40_lora import (  # noqa: E402
    _apply_trainable_and_audit,
    _atomic_write_json,
    _build_model_with_lora,
    _confirm_no_base_leak_in_optimizer,
    _cuda_memory_report,
    _diff_baseline_vs_lora,
    _dockerdata_ok,
    _log_event,
    _make_lora_smoke_meta,
    _now_iso,
    _path_size_gib,
    _prepare_output_dir,
    _save_step_trainable,
    _sha256_file,
    _strict_reload_check,
    _validate_lora_manifest,
    _write_git_provenance,
    _write_sequence_plan,
    BASELINE_CONFIG,
    FIXED_GOLD40_MANIFEST,
)
from experiments.counterfactual_cot.training.gold40_overfit import (  # noqa: E402
    DEFAULT_TRAINABLE_REGEX,
    _autocast_for_model,
    _build_gold40_loader,
    _load_tokenizer,
    _prepare_batch,
)
from experiments.counterfactual_cot.training.smoke_train import _batch_report, _set_seed  # noqa: E402
from experiments.counterfactual_cot.training.lora_2step_smoke import (  # noqa: E402
    LoRALinear,
    trainable_state_dict,
)
# Reuse smoke helpers verbatim (import only).
from experiments.counterfactual_cot.training.gold40_v3_lora_2step_smoke import (  # noqa: E402
    _audit_frozen_base,
    _cuda_memory_snapshot,
    _device,
    _diff_frozen_audits,
    _diff_param_snapshots,
    _global_grad_norm,
    _per_tensor_grad_snapshot,
    _per_tensor_param_snapshot,
    _tensor_fingerprint,
    V3_MANIFEST,
)


DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/counterfactual_cot/outputs/gold40_v3_lora_longest_pair_memory_gate_A100_40GB"
)

LONGEST_PAIR_ID_DEFAULT = (
    "cf_source_flower_shop_absolute_position_change_v4_absolute_region_wide_table_fix_v6_0000"
)
LONGEST_SEQUENCE_LENGTH_EXPECTED = 1270
SINGLEPACK_CEILING = 1536

# Correction reference: prior 2-step smoke report SHA (superseded loss claim).
PRIOR_SMOKE_REPORT_SHA = "93388c4c8cdbc0997f941c71396879dc8d616b1d0877d5901e5feed992fa8868"
PRIOR_CROSSCHECK_REPORT_SHA = "b0312d679dee9f35f763e142c6ec06e7cef999494e54aa57df20cba81acce991"
PRIOR_MEMORY_ROOT_CAUSE_SHA = "8c50deeea3330f06a5e49fdfe1f06dd15560f42cb0d142ef793cf804bbe3a425"


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--manifest", type=Path, default=V3_MANIFEST)
    p.add_argument("--llm-path", type=str, default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
    p.add_argument("--vae-path", type=str, default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors")
    p.add_argument("--checkpoint-path", type=str, default="/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--precision", type=str, default=BASELINE_CONFIG["precision"])
    p.add_argument("--model-device-map", type=str, default="single")
    p.add_argument("--device-map-max-memory", type=str, default="34GiB")

    p.add_argument("--num-samples", type=int, default=40)
    p.add_argument("--variant", type=str, default=BASELINE_CONFIG["variant"])
    p.add_argument("--image-size", type=int, default=BASELINE_CONFIG["image_size"])
    p.add_argument("--vit-image-size", type=int, default=BASELINE_CONFIG["vit_image_size"])
    p.add_argument("--vit-min-image-size", type=int, default=BASELINE_CONFIG["vit_min_image_size"])
    p.add_argument("--visual-image-size", type=int, default=BASELINE_CONFIG["visual_image_size"])
    p.add_argument("--visual-vae-min-image-size", type=int, default=BASELINE_CONFIG["visual_vae_min_image_size"])
    p.add_argument("--visual-vit-image-size", type=int, default=BASELINE_CONFIG["visual_vit_image_size"])
    p.add_argument("--visual-vit-min-image-size", type=int, default=BASELINE_CONFIG["visual_vit_min_image_size"])

    p.add_argument("--expected-num-tokens", type=int, default=BASELINE_CONFIG["expected_num_tokens"])
    p.add_argument("--max-num-tokens-per-sample", type=int, default=BASELINE_CONFIG["max_num_tokens_per_sample"])
    p.add_argument("--max-num-tokens", type=int, default=BASELINE_CONFIG["max_num_tokens"])

    p.add_argument("--seed", type=int, default=BASELINE_CONFIG["seed"])
    p.add_argument("--generation-seed", type=int, default=BASELINE_CONFIG["generation_seed"])

    p.add_argument("--lr", type=float, default=BASELINE_CONFIG["lr"])
    p.add_argument("--max-grad-norm", type=float, default=BASELINE_CONFIG["max_grad_norm"])
    p.add_argument("--adam-beta1", type=float, default=BASELINE_CONFIG["adam_betas"][0])
    p.add_argument("--adam-beta2", type=float, default=BASELINE_CONFIG["adam_betas"][1])
    p.add_argument("--adam-eps", type=float, default=BASELINE_CONFIG["adam_eps"])
    p.add_argument("--weight-decay", type=float, default=BASELINE_CONFIG["adam_weight_decay"])
    p.add_argument("--num-timesteps", type=int, default=BASELINE_CONFIG["num_timesteps"])
    p.add_argument("--timestep-shift", type=float, default=BASELINE_CONFIG["timestep_shift"])

    p.add_argument("--cfg-text-scale", type=float, default=BASELINE_CONFIG["cfg_text_scale"])
    p.add_argument("--cfg-img-scale", type=float, default=BASELINE_CONFIG["cfg_img_scale"])
    p.add_argument("--cfg-interval-start", type=float, default=BASELINE_CONFIG["cfg_interval_start"])
    p.add_argument("--cfg-interval-end", type=float, default=BASELINE_CONFIG["cfg_interval_end"])
    p.add_argument("--cfg-renorm-min", type=float, default=BASELINE_CONFIG["cfg_renorm_min"])
    p.add_argument("--cfg-renorm-type", type=str, default=BASELINE_CONFIG["cfg_renorm_type"])

    p.add_argument("--lora-layers", type=int, nargs="+", default=[24, 25, 26, 27])
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    p.add_argument("--trainable-name-regex", type=str, default=DEFAULT_TRAINABLE_REGEX)

    p.add_argument("--longest-pair-id", type=str, default=LONGEST_PAIR_ID_DEFAULT)
    p.add_argument("--longest-sequence-length-expected", type=int, default=LONGEST_SEQUENCE_LENGTH_EXPECTED)
    p.add_argument("--longest-pair-max-scans", type=int, default=800,
                   help="Maximum number of dataloader batches to draw while searching for the target longest pair.")

    p.add_argument("--steps", type=int, default=400)  # for baseline_vs_lora config parity only

    # Not used by this gate but kept for _diff_baseline_vs_lora parity.
    p.add_argument("--eval-max-batches", type=int, default=BASELINE_CONFIG["eval_max_batches"])
    p.add_argument("--log-every", type=int, default=BASELINE_CONFIG["log_every"])
    p.add_argument("--eval-steps", type=int, nargs="*", default=BASELINE_CONFIG["eval_steps"])
    p.add_argument("--save-steps", type=int, nargs="*", default=BASELINE_CONFIG["save_steps"])

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--allow-non-dockerdata", action="store_true", default=False)

    # Memory gate parameters.
    p.add_argument("--min-headroom-mib", type=int, default=4096,
                   help="Absolute minimum global reserved-memory headroom in MiB.")
    p.add_argument("--min-headroom-ratio", type=float, default=0.10,
                   help="Minimum ratio of headroom / total GPU memory.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fatal(output_dir: Path, why: str, exc: Optional[BaseException] = None) -> int:
    err = {
        "status": "failed",
        "kind": type(exc).__name__ if exc is not None else "AssertionError",
        "message": why,
        "traceback": traceback.format_exc() if exc is not None else "",
        "created_at_iso": _now_iso(),
    }
    _atomic_write_json(output_dir / "error_report.json", err)
    print(f"[longest-pair-gate] FAILED: {why}", file=sys.stderr)
    return 1


def _record_memory(device: torch.device, label: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    snap = _cuda_memory_snapshot(device)
    snap["label"] = label
    snap["at_iso"] = _now_iso()
    if extra:
        snap.update(extra)
    _log_event("mem_snapshot", **{k: snap.get(k) for k in ("label", "allocated_mib", "reserved_mib",
                                                             "peak_allocated_mib", "peak_reserved_mib",
                                                             "num_alloc_retries")})
    return snap


def _batch_shape_record(batch: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    rep = _batch_report(batch, data, torch.device("cuda"))
    return {
        "sequence_length": rep.get("sequence_length"),
        "sample_lens": list(rep.get("sample_lens") or []),
        "batch_data_indexes": rep.get("batch_data_indexes"),
        "tensor_shapes": {k: v.get("shape") for k, v in (rep.get("tensors") or {}).items()},
        "pair_id": ((rep.get("batch_data_indexes") or [{}])[0] or {}).get("pair_id"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    # --- Preflight ---
    _dockerdata_ok(args.llm_path, getattr(args, "allow_non_dockerdata", False))
    _dockerdata_ok(args.vae_path, getattr(args, "allow_non_dockerdata", False))
    _dockerdata_ok(args.checkpoint_path, getattr(args, "allow_non_dockerdata", False))
    _validate_lora_manifest(args.manifest)
    output_dir = args.output_dir
    _prepare_output_dir(output_dir)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # provenance
    _write_git_provenance(output_dir)
    _write_sequence_plan(output_dir, args.manifest)
    with open(output_dir / "run_config_resolved.json", "w") as f:
        json.dump({"args": vars(args), "created_at_iso": _now_iso()}, f, ensure_ascii=False, indent=2, default=str)

    device = _device(args.device)
    if device.type != "cuda":
        return _fatal(output_dir, f"expected cuda device, got {device}")

    total_memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    total_memory_mib = total_memory_bytes / (1024 * 1024)
    gpu_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    alloc_conf_raw = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    env_provenance = {
        "PYTORCH_CUDA_ALLOC_CONF_raw": alloc_conf_raw,
        "PYTORCH_CUDA_ALLOC_CONF_has_expandable_segments_true": "expandable_segments:true" in alloc_conf_raw.lower(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": total_memory_bytes,
        "gpu_total_memory_mib": total_memory_mib,
        "compute_capability": f"{cc[0]}.{cc[1]}",
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "created_at_iso": _now_iso(),
    }
    _atomic_write_json(output_dir / "environment_provenance.json", env_provenance)
    _log_event("gpu_probed", name=gpu_name, compute_capability=f"{cc[0]}.{cc[1]}",
               total_memory_bytes=total_memory_bytes, total_memory_mib=total_memory_mib,
               PYTORCH_CUDA_ALLOC_CONF=alloc_conf_raw,
               torch_version=torch.__version__, torch_cuda_version=torch.version.cuda)

    _set_seed(args.seed)

    memory_nodes: List[Dict[str, Any]] = []

    # 1. Base loaded (before Base weights land on GPU we can't record peak).
    torch.cuda.reset_peak_memory_stats()
    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)
    tokenizer_size = len(tokenizer)
    model, vae_model, wrappers, target_names, base_checkpoint_report = _build_model_with_lora(
        args, device, tokenizer_size=tokenizer_size, num_new_tokens=num_new_tokens,
    )
    memory_nodes.append(_record_memory(device, "1_base_and_vae_loaded"))

    # 2. Freeze + apply trainable regex + audit
    audit = _apply_trainable_and_audit(model, wrappers, target_names, args, device.type)
    existing_names: List[str] = list(audit["existing_names"])
    lora_names: List[str] = list(audit["lora_names"])
    memory_nodes.append(_record_memory(device, "2_lora_injected_and_trainables_set"))

    # config parity with baseline
    baseline_vs_lora_diff = _diff_baseline_vs_lora(args, audit)
    _atomic_write_json(output_dir / "baseline_vs_lora_config_diff.json", baseline_vs_lora_diff)
    if baseline_vs_lora_diff["status"] != "passed":
        return _fatal(output_dir,
                      f"baseline_vs_lora config diff has non-LoRA differences: {baseline_vs_lora_diff.get('non_lora_differences')}")

    # 3. Loader + optimizer
    train_loader = _build_gold40_loader(args.manifest, tokenizer, special_tokens, args, seed_offset=0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_ids = {id(p) for p in trainable_params}
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    optimizer_audit = _confirm_no_base_leak_in_optimizer(optimizer, model, trainable_ids)
    _atomic_write_json(output_dir / "optimizer_audit.json", optimizer_audit)
    memory_nodes.append(_record_memory(device, "3_optimizer_built"))

    # Frozen base audit BEFORE.
    frozen_before = _audit_frozen_base(model)
    _atomic_write_json(output_dir / "frozen_base_audit_before.json", frozen_before)
    gc.collect()
    torch.cuda.empty_cache()

    # ---------- Step 0: save trainable snapshot BEFORE any optimizer.step ----------
    step0_snap = _per_tensor_param_snapshot(model, existing_names, lora_names)
    _atomic_write_json(output_dir / "trainable_snapshot_step0.json", step0_snap)
    step0_ck = _save_step_trainable(0, model, output_dir, args, target_names, existing_names, lora_names)
    step0_reload = _strict_reload_check(Path(step0_ck["path"]), model)
    memory_nodes.append(_record_memory(device, "4_step000_saved"))

    # ================= Step 1: normal single-pack pair =================
    model_dtype = next(model.parameters()).dtype
    data_iter = iter(train_loader)
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader); batch = next(data_iter)
    data = _prepare_batch(batch, vae_model, device, model_dtype)
    step1_batch = _batch_shape_record(batch, data)
    if len(step1_batch["sample_lens"]) != 1 or (step1_batch["sequence_length"] or 0) > SINGLEPACK_CEILING:
        return _fatal(output_dir, f"step1 batch not single-pack: {step1_batch['sample_lens']}, seq={step1_batch['sequence_length']}")
    _log_event("step1_batch_ok", pair_id=step1_batch["pair_id"], sample_lens=step1_batch["sample_lens"],
               sequence_length=step1_batch["sequence_length"])

    optimizer.zero_grad(set_to_none=True)
    memory_nodes.append(_record_memory(device, "5_step1_before_forward", extra={"pair_id": step1_batch["pair_id"]}))
    with _autocast_for_model(model):
        out = model(**data)
    mse = out.get("mse") if isinstance(out, dict) else None
    if mse is None:
        return _fatal(output_dir, "step1 forward produced no mse")
    loss = mse.mean()
    if not torch.isfinite(loss).all().item():
        return _fatal(output_dir, f"step1 non-finite loss: {loss.item()}")
    loss_val_1 = float(loss.item())
    memory_nodes.append(_record_memory(device, "6_step1_after_forward",
                                       extra={"pair_id": step1_batch["pair_id"], "loss": loss_val_1}))

    loss.backward()
    memory_nodes.append(_record_memory(device, "7_step1_after_backward", extra={"pair_id": step1_batch["pair_id"]}))

    grad1 = _per_tensor_grad_snapshot(model, existing_names, lora_names)
    grad_norm1 = _global_grad_norm(trainable_params)
    if args.max_grad_norm and args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

    optimizer.step()
    memory_nodes.append(_record_memory(device, "8_step1_after_optimizer_step", extra={"pair_id": step1_batch["pair_id"]}))

    for name in list(existing_names) + list(lora_names):
        p = dict(model.named_parameters())[name]
        if not torch.isfinite(p).all().item():
            return _fatal(output_dir, f"step1 non-finite param after step: {name}")

    step1_snap = _per_tensor_param_snapshot(model, existing_names, lora_names)
    step1_ck = _save_step_trainable(1, model, output_dir, args, target_names, existing_names, lora_names)
    step1_reload = _strict_reload_check(Path(step1_ck["path"]), model)
    optimizer.zero_grad(set_to_none=True)
    memory_nodes.append(_record_memory(device, "9_step1_after_zero_grad", extra={"pair_id": step1_batch["pair_id"]}))

    step1_records = {
        "loss": loss_val_1,
        "grad_norm_pre_clip": grad_norm1,
        "batch": step1_batch,
        "checkpoint": step1_ck,
        "strict_reload": step1_reload,
        "num_lora_A_nonzero_grad": sum(1 for n in lora_names if n.endswith(".lora_A") and grad1.get(n, {}).get("is_nonzero")),
        "num_lora_B_nonzero_grad": sum(1 for n in lora_names if n.endswith(".lora_B") and grad1.get(n, {}).get("is_nonzero")),
        "num_existing_nonzero_grad": sum(1 for n in existing_names if grad1.get(n, {}).get("is_nonzero")),
    }

    del out, mse, loss, data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # ================= Step 2: LONGEST pair =================
    target_pair = args.longest_pair_id
    scanned_ids: List[str] = []
    longest_rep: Optional[Dict[str, Any]] = None
    longest_data: Optional[Dict[str, Any]] = None
    for scan_idx in range(1, args.longest_pair_max_scans + 1):
        try:
            b = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader); b = next(data_iter)
        d = _prepare_batch(b, vae_model, device, model_dtype)
        rep = _batch_shape_record(b, d)
        scanned_ids.append(rep["pair_id"])
        if rep["pair_id"] == target_pair:
            longest_data, longest_rep = d, rep
            _log_event("longest_pair_hit", scan_idx=scan_idx, pair_id=rep["pair_id"],
                       sequence_length=rep["sequence_length"])
            break
        # release before next scan
        del b, d
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if longest_rep is None or longest_data is None:
        return _fatal(output_dir, f"exhausted {args.longest_pair_max_scans} scans without hitting {target_pair}")

    if len(longest_rep["sample_lens"]) != 1:
        return _fatal(output_dir, f"longest pair not single-pack: {longest_rep}")
    if longest_rep["sequence_length"] != args.longest_sequence_length_expected:
        _log_event("longest_seq_length_mismatch",
                   expected=args.longest_sequence_length_expected,
                   observed=longest_rep["sequence_length"])
    if longest_rep["sequence_length"] > SINGLEPACK_CEILING:
        return _fatal(output_dir, f"longest sequence_length={longest_rep['sequence_length']} exceeds {SINGLEPACK_CEILING}")

    # Hard-assert the 6 pre-registered batch shapes for the longest pair. Any
    # deviation from these values means the pair drawn from the dataloader is
    # NOT the audited flower_shop pair, and the memory gate is invalidated.
    _lp_expected_shapes = {
        "sequence_length": 1270,
        "sample_lens": [1270],
        "packed_text_indexes": [502],
        "packed_vae_token_indexes": [512],
        "packed_vit_tokens": [256, 588],
        "padded_latent": [2, 16, 32, 32],
    }
    _lp_observed = {
        "sequence_length": longest_rep["sequence_length"],
        "sample_lens": longest_rep["sample_lens"],
        "packed_text_indexes": longest_rep["tensor_shapes"].get("packed_text_indexes"),
        "packed_vae_token_indexes": longest_rep["tensor_shapes"].get("packed_vae_token_indexes"),
        "packed_vit_tokens": longest_rep["tensor_shapes"].get("packed_vit_tokens"),
        "padded_latent": longest_rep["tensor_shapes"].get("padded_latent"),
    }
    _lp_mismatches = {k: (_lp_expected_shapes[k], _lp_observed[k])
                       for k in _lp_expected_shapes
                       if _lp_observed[k] != _lp_expected_shapes[k]}
    if _lp_mismatches:
        return _fatal(output_dir,
                      f"longest pair batch-shape hard-assertion failed: "
                      f"mismatches (expected vs observed) = {_lp_mismatches}")
    _log_event("longest_pair_shape_assertions_ok",
               pair_id=target_pair, expected=_lp_expected_shapes)

    # Sync + reset peak stats. Do NOT delete optimizer state.
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    memory_nodes.append(_record_memory(device, "10_step2_before_forward_after_reset_peak",
                                       extra={"pair_id": target_pair, "sequence_length": longest_rep["sequence_length"]}))

    with _autocast_for_model(model):
        out2 = model(**longest_data)
    mse2 = out2.get("mse") if isinstance(out2, dict) else None
    if mse2 is None:
        return _fatal(output_dir, "step2 forward produced no mse")
    loss2 = mse2.mean()
    if not torch.isfinite(loss2).all().item():
        return _fatal(output_dir, f"step2 non-finite loss: {loss2.item()}")
    loss_val_2 = float(loss2.item())
    memory_nodes.append(_record_memory(device, "11_step2_after_forward",
                                       extra={"pair_id": target_pair, "loss": loss_val_2}))

    loss2.backward()
    memory_nodes.append(_record_memory(device, "12_step2_after_backward", extra={"pair_id": target_pair}))

    grad2 = _per_tensor_grad_snapshot(model, existing_names, lora_names)
    grad_norm2 = _global_grad_norm(trainable_params)
    if args.max_grad_norm and args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)

    optimizer.step()
    memory_nodes.append(_record_memory(device, "13_step2_after_optimizer_step", extra={"pair_id": target_pair}))

    for name in list(existing_names) + list(lora_names):
        p = dict(model.named_parameters())[name]
        if not torch.isfinite(p).all().item():
            return _fatal(output_dir, f"step2 non-finite param after step: {name}")

    step2_snap = _per_tensor_param_snapshot(model, existing_names, lora_names)
    step2_ck = _save_step_trainable(2, model, output_dir, args, target_names, existing_names, lora_names)
    step2_reload = _strict_reload_check(Path(step2_ck["path"]), model)
    optimizer.zero_grad(set_to_none=True)
    memory_nodes.append(_record_memory(device, "14_step2_after_zero_grad", extra={"pair_id": target_pair}))

    # final global peak
    final_mem = _cuda_memory_snapshot(device)
    final_mem["label"] = "15_final_global"
    final_mem["at_iso"] = _now_iso()
    memory_nodes.append(final_mem)

    # ================= Post-training audits =================
    frozen_after = _audit_frozen_base(model)
    _atomic_write_json(output_dir / "frozen_base_audit_after.json", frozen_after)
    frozen_diff = _diff_frozen_audits(frozen_before, frozen_after)
    param_diff_02 = _diff_param_snapshots(step0_snap, step2_snap)

    del out2, mse2, loss2, longest_data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------- Hard gates -------------------
    # Take max reserved peak across step2 phases (indices 9..13 -> labels 10..14)
    step2_max_peak_reserved_mib = max(
        float(m.get("peak_reserved_mib", 0.0)) for m in memory_nodes[9:14]
    )
    step2_max_peak_allocated_mib = max(
        float(m.get("peak_allocated_mib", 0.0)) for m in memory_nodes[9:14]
    )
    step2_max_alloc_retries = max(
        int(m.get("num_alloc_retries", 0)) for m in memory_nodes[9:14]
    )
    global_peak_reserved_mib = max(float(m.get("peak_reserved_mib", 0.0)) for m in memory_nodes)
    global_peak_allocated_mib = max(float(m.get("peak_allocated_mib", 0.0)) for m in memory_nodes)
    global_alloc_retries = max(int(m.get("num_alloc_retries", 0)) for m in memory_nodes)

    memory_headroom_mib = total_memory_mib - step2_max_peak_reserved_mib
    memory_headroom_ratio = memory_headroom_mib / total_memory_mib

    step2_num_lora_A_nonzero = sum(1 for n in lora_names if n.endswith(".lora_A") and grad2.get(n, {}).get("is_nonzero"))
    step2_num_lora_B_nonzero = sum(1 for n in lora_names if n.endswith(".lora_B") and grad2.get(n, {}).get("is_nonzero"))
    step2_num_existing_nonzero = sum(1 for n in existing_names if grad2.get(n, {}).get("is_nonzero"))

    lora_a_names_set = set(n for n in lora_names if n.endswith(".lora_A"))
    lora_b_names_set = set(n for n in lora_names if n.endswith(".lora_B"))
    existing_names_set = set(existing_names)
    changed_names_list = list(param_diff_02.get("changed_names") or [])
    lora_a_changed = sum(1 for n in changed_names_list if n in lora_a_names_set)
    lora_b_changed = sum(1 for n in changed_names_list if n in lora_b_names_set)
    existing_changed = sum(1 for n in changed_names_list if n in existing_names_set)

    total_trainable_params = int(audit.get("total_trainable_parameters", 0))
    num_trainable_tensors = int(audit.get("existing_tensors", 0)) + int(audit.get("lora_tensors", 0))

    optim_no_leak = (
        (optimizer_audit.get("num_optimizer_params") == optimizer_audit.get("num_trainable_params"))
        and not optimizer_audit.get("frozen_leaked_into_optim")
    )

    hard_gates = {
        "G1_realized_shape_audit_present": (REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_singlepack_realized_shapes/gold40_v3_singlepack_all40_realized_shapes.json").exists(),
        "G2_longest_pair_selection_correct": longest_rep["pair_id"] == target_pair,
        "G3_step1_completed": step1_reload.get("max_abs_load_copy_diff") == 0.0,
        "G4_step2_longest_completed": step2_reload.get("max_abs_load_copy_diff") == 0.0,
        "G5_optimizer_step_exactly_2": True,
        "G6_longest_is_single_pack": len(longest_rep["sample_lens"]) == 1,
        "G7_longest_seq_within_ceiling": (longest_rep["sequence_length"] or 0) <= SINGLEPACK_CEILING,
        "G8_all_losses_and_grads_finite": True,
        "G9_trainable_tensors_62": num_trainable_tensors == 62,
        "G10_total_trainable_params_65_617_472": total_trainable_params == 65_617_472,
        "G11_no_base_leak_in_optimizer": optim_no_leak,
        "G12_frozen_base_unchanged": (
            frozen_diff.get("num_changed", -1) == 0
            and frozen_diff.get("num_added", -1) == 0
            and frozen_diff.get("num_removed", -1) == 0
        ),
        "G13_no_oom": True,
        "G14_no_nan_inf": True,
        "G15_num_alloc_retries_zero": global_alloc_retries == 0,
        "G16_headroom_mib_ge_4096": memory_headroom_mib >= float(args.min_headroom_mib),
        "G17_headroom_ratio_ge_10pct": memory_headroom_ratio >= float(args.min_headroom_ratio),
        "G18_step2_lora_grads_and_updates_nonzero": (
            step2_num_lora_A_nonzero >= 1 and step2_num_lora_B_nonzero >= 1
            and lora_a_changed >= 1 and lora_b_changed >= 1
        ),
    }
    all_pass = all(bool(v) for v in hard_gates.values())
    status = "passed" if all_pass else "failed"

    correction = {
        "prior_report_sha256_2step_smoke": PRIOR_SMOKE_REPORT_SHA,
        "prior_crosscheck_report_sha256": PRIOR_CROSSCHECK_REPORT_SHA,
        "prior_memory_root_cause_report_sha256": PRIOR_MEMORY_ROOT_CAUSE_SHA,
        "corrected_statement": (
            "The 2-step smoke reported step1 loss=0.6330 (pair="
            "cf_source_library_table_count_change_0000) and step2 loss=0.2390 (pair="
            "cf_source_home_kitchen_spatial_relation_change_0000). These two values "
            "come from DIFFERENT pairs and MUST NOT be described as 'same-pair loss "
            "descent' or 'two-step convergence evidence'. The correct reading is: "
            "the forward / backward / optimizer.step pipeline is operational and "
            "both steps yielded finite losses; no convergence claim is warranted."
        ),
        "note": "This correction lives only in the new report; the prior report file is preserved verbatim on disk.",
    }

    report = {
        "status": status,
        "created_at_iso": _now_iso(),
        "output_dir": str(output_dir),
        "gpu": {
            "name": gpu_name,
            "compute_capability": f"{cc[0]}.{cc[1]}",
            "total_memory_bytes": total_memory_bytes,
            "total_memory_mib": total_memory_mib,
        },
        "environment_provenance": env_provenance,
        "longest_pair": {
            "pair_id": target_pair,
            "sequence_length": longest_rep["sequence_length"],
            "sample_lens": longest_rep["sample_lens"],
            "batch_shapes": longest_rep["tensor_shapes"],
            "scans_used": len(scanned_ids),
            "first_scanned_pair_ids": scanned_ids[:20],
        },
        "trainable_audit": audit,
        "baseline_vs_lora_config_diff": baseline_vs_lora_diff,
        "optimizer_audit": optimizer_audit,
        "step0_checkpoint": step0_ck,
        "step0_strict_reload": step0_reload,
        "step1_record": step1_records,
        "step2_record": {
            "loss": loss_val_2,
            "grad_norm_pre_clip": grad_norm2,
            "batch": longest_rep,
            "checkpoint": step2_ck,
            "strict_reload": step2_reload,
            "num_lora_A_nonzero_grad": step2_num_lora_A_nonzero,
            "num_lora_B_nonzero_grad": step2_num_lora_B_nonzero,
            "num_existing_nonzero_grad": step2_num_existing_nonzero,
        },
        "param_update_step0_to_step2": param_diff_02,
        "param_update_summary_by_group": {
            "num_lora_A_changed": lora_a_changed,
            "num_lora_B_changed": lora_b_changed,
            "num_existing_changed": existing_changed,
            "changed_lora_A_names": sorted(n for n in changed_names_list if n in lora_a_names_set),
            "changed_lora_B_names": sorted(n for n in changed_names_list if n in lora_b_names_set),
            "changed_existing_names": sorted(n for n in changed_names_list if n in existing_names_set),
        },
        "frozen_base_audit_diff": frozen_diff,
        "memory_nodes": memory_nodes,
        "memory_summary": {
            "gpu_total_memory_mib": total_memory_mib,
            "global_peak_allocated_mib": global_peak_allocated_mib,
            "global_peak_reserved_mib": global_peak_reserved_mib,
            "step2_max_peak_allocated_mib": step2_max_peak_allocated_mib,
            "step2_max_peak_reserved_mib": step2_max_peak_reserved_mib,
            "memory_headroom_mib": memory_headroom_mib,
            "memory_headroom_ratio": memory_headroom_ratio,
            "num_alloc_retries_global": global_alloc_retries,
            "num_alloc_retries_step2": step2_max_alloc_retries,
        },
        "hard_gates": hard_gates,
        "correction": correction,
        "evaluation_type": "diagnostic_memory_gate_only_not_convergence_evidence",
    }

    report_path = output_dir / "gold40_v3_lora_longest_pair_memory_gate_report.json"
    _atomic_write_json(report_path, report)

    if not all_pass:
        _log_event("memory_gate_failed", failed_gates=[k for k, v in hard_gates.items() if not v])
        return 2
    _log_event("memory_gate_passed",
               headroom_mib=memory_headroom_mib, headroom_ratio=memory_headroom_ratio,
               step2_peak_reserved_mib=step2_max_peak_reserved_mib,
               global_peak_reserved_mib=global_peak_reserved_mib)
    return 0


if __name__ == "__main__":
    sys.exit(main())
