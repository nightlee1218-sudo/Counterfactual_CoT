"""Step000 alignment gate for Gold40 v3 Protocol v1.

Two entry modes:

  * --mode verifier:  Run inside a single fresh process. Rebuilds Base
                      from /dockerdata, optionally injects LoRA (B side),
                      strict-loads the given step_000 checkpoint, then
                      iterates the 40 pairs and generates one PNG per pair
                      with a fixed per-pair seed. Writes verifier_<side>.json
                      + inputs_provenance_<side>.jsonl + 40 PNGs.

  * --mode orchestrate: Parent process. Launches --mode verifier for A first,
                       waits with subprocess.wait() to capture numeric
                       exit_code, then launches --mode verifier for B,
                       waits, then runs the CPU-only comparator.
                       Writes step000_alignment_gate_report.json.

Discipline:
  - No training. No optimizer.step. No backward.
  - No cross-A-B parallelism. A must fully exit before B starts.
  - No monitoring daemon. No retry on failure.
  - Manifest / checkpoints / protocol are read-only.
  - No changes to any prior report or checkpoint.

Frozen inputs (verified before launch):
  - Protocol v1 JSON SHA256 = 6f345d56b4c617f931129ab79fa60af83cd3755b3d904f2c669d2b961ec003c1
  - A step_000 SHA256       = 2a85cdc1d0457a296fa4dc695ee5e351a4ff6c137dd28ea58c4f87b5b079fe52
  - B step_000 SHA256       = 0f138533db0bf98ddee14711e7664c798252b8e41710d5043303350cbe78baff
  - Gold40 v3 manifest SHA  = 5960e1eebb4d7a394a7b7cba2b2c17c686d184756b637110ef8b43397e172061
  - Base EMA SHA            = 0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3
  - Base AE  SHA            = afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38

Same-pair same-seed => same initial noise across A and B, by
_set_seed(pair_seed) inside _generate_one right before interleave_inference
consumes RNG state to draw initial noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants (frozen anchors)
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/group/40016/niightli/BAGEL").resolve()
AUDIT_DIR = REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_ab_400steps_final_engineering_audit"

PROTOCOL_JSON = AUDIT_DIR / "gold40_v3_grounding_evaluation_protocol_v1.json"
PROTOCOL_JSON_SHA_EXPECT = "6f345d56b4c617f931129ab79fa60af83cd3755b3d904f2c669d2b961ec003c1"
PROTOCOL_MD = AUDIT_DIR / "gold40_v3_grounding_evaluation_protocol_v1.md"
PROTOCOL_MD_SHA_EXPECT = "a237c2ec81c5733b88b778dce1b84c500b2f06a03549fb7ceb8554c899b62170"
PROVENANCE_AUDIT = AUDIT_DIR / "gold40_v3_ab_step000_checkpoint_provenance_audit.json"
PROVENANCE_AUDIT_SHA_EXPECT = "33bc80722154c6bfa53376658a20f2afaee91d904403961dca6a84774668cbbc"

A_STEP000 = REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_overfit_400steps_lr1e6_singlepack/checkpoints/step_000_trainable.pt"
A_STEP000_SHA_EXPECT = "2a85cdc1d0457a296fa4dc695ee5e351a4ff6c137dd28ea58c4f87b5b079fe52"
B_STEP000 = REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_lora_L24-27_r8_a16_lr1e6_singlepack_400steps/checkpoints/step_000_trainable.pt"
B_STEP000_SHA_EXPECT = "0f138533db0bf98ddee14711e7664c798252b8e41710d5043303350cbe78baff"

V3_MANIFEST = REPO_ROOT / "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/manifests/intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl"
V3_MANIFEST_SHA_EXPECT = "5960e1eebb4d7a394a7b7cba2b2c17c686d184756b637110ef8b43397e172061"
LLM_PATH = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
VAE_PATH = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors")
EMA_PATH = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors")
BASE_EMA_SHA_EXPECT = "0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3"
BASE_AE_SHA_EXPECT = "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_ab_aligned_inference_nt25_protocol_v1"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_string(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["orchestrate", "verifier"], required=True)
    # verifier-mode args
    parser.add_argument("--side", choices=["A", "B"])
    parser.add_argument("--trainable-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    # orchestrate-mode args
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    # common frozen values (protocol v1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num-timesteps", type=int, default=25)
    parser.add_argument("--precision", type=str, default="bf16")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Verifier: single side (A or B), 40-pair loop, fresh process
# ---------------------------------------------------------------------------

def _verifier_main(args: argparse.Namespace) -> int:
    if args.side is None or args.trainable_checkpoint is None or args.output_dir is None:
        raise SystemExit("verifier mode requires --side, --trainable-checkpoint, --output-dir")

    # Heavy deps: keep imports LOCAL so that orchestrate mode does not
    # unnecessarily import torch / bagel etc.
    import gc
    import random
    import numpy as np
    import torch
    from PIL import Image

    from experiments.counterfactual_cot.data.schema import load_jsonl
    from experiments.counterfactual_cot.data.render_prompts import paired_prompt
    from experiments.counterfactual_cot.training.gold40_overfit import (
        _build_model_for_gold40,
        _generate_one,
        _load_pair_image,
        _load_tokenizer,
        _make_inferencer,
    )
    from experiments.counterfactual_cot.training.lora_2step_smoke import (
        _describe_dtypes,
        _iter_target_linear_names,
        inject_lora,
        set_trainable,
    )
    from experiments.counterfactual_cot.training.cold_reload_verifier import (
        load_trainable_strict_with_diff,
    )
    from experiments.counterfactual_cot.training.fixed_deterministic_eval import (
        _stable_pair_seed,
    )
    from experiments.counterfactual_cot.training.single_pair_overfit_diag import (
        _snapshot_rng_state,
        _restore_rng_state,
    )

    side: str = args.side
    ckpt_path: Path = args.trainable_checkpoint
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "png"
    png_dir.mkdir(exist_ok=True)

    # Refuse to write under /dockerdata.
    if str(out_dir.resolve()).startswith("/dockerdata"):
        raise RuntimeError(f"Refusing to write under /dockerdata: {out_dir}")

    device = torch.device("cuda")

    # ---- Frozen input verification (again, from inside verifier process) ----
    def _sha256_file_local(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    ema_sha = _sha256_file_local(EMA_PATH)
    ae_sha = _sha256_file_local(VAE_PATH)
    manifest_sha = _sha256_file_local(V3_MANIFEST)
    ckpt_sha = _sha256_file_local(ckpt_path)
    if ema_sha != BASE_EMA_SHA_EXPECT:
        raise RuntimeError(f"Base EMA sha mismatch in verifier {side}: {ema_sha}")
    if ae_sha != BASE_AE_SHA_EXPECT:
        raise RuntimeError(f"Base AE sha mismatch in verifier {side}: {ae_sha}")
    if manifest_sha != V3_MANIFEST_SHA_EXPECT:
        raise RuntimeError(f"Manifest sha mismatch in verifier {side}: {manifest_sha}")
    expect_ckpt_sha = A_STEP000_SHA_EXPECT if side == "A" else B_STEP000_SHA_EXPECT
    if ckpt_sha != expect_ckpt_sha:
        raise RuntimeError(f"Trainable ckpt sha mismatch in verifier {side}: {ckpt_sha}")

    log_path = out_dir / "verifier_log.jsonl"

    def _log(event: str, **fields: Any) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"event": event, "iso": now_iso(), **fields}, ensure_ascii=False, sort_keys=True) + "\n")

    _log("verifier_start", side=side, ckpt_path=str(ckpt_path), ckpt_sha=ckpt_sha,
         manifest_sha=manifest_sha, ema_sha=ema_sha, ae_sha=ae_sha,
         num_timesteps=args.num_timesteps, seed=args.seed)

    # ---- Seed global (has minor influence; per-pair seeds dominate) ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    # ---- Load tokenizer / manifest / build inference args namespace ----
    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(str(LLM_PATH))
    records = load_jsonl(V3_MANIFEST)
    if len(records) != 40:
        raise RuntimeError(f"Expected 40 pairs in Gold40 v3, got {len(records)}")

    # Namespace for _build_model_for_gold40 / _make_inferencer / _generate_one
    # (all these expect an argparse-like args object). We construct an
    # SimpleNamespace with exactly the fields those helpers read.
    from types import SimpleNamespace
    inf_args = SimpleNamespace(
        llm_path=str(LLM_PATH),
        vae_path=str(VAE_PATH),
        checkpoint_path=str(EMA_PATH),
        device="cuda",
        precision=args.precision,
        model_device_map="single",
        device_map_max_memory="34GiB",
        image_size=256,
        vit_image_size=224,
        vit_min_image_size=224,
        visual_image_size=1024,
        visual_vae_min_image_size=512,
        visual_vit_image_size=980,
        visual_vit_min_image_size=224,
        expected_num_tokens=512,
        max_num_tokens_per_sample=1536,
        max_num_tokens=1536,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval_start=0.0,
        cfg_interval_end=1.0,
        timestep_shift=3.0,
        num_timesteps=args.num_timesteps,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
        seed=args.seed,
        generation_seed=args.seed,
        allow_non_dockerdata=False,
        require_paired_vit=True,
        trainable_name_regex=r"^(time_embedder|latent_pos_embed|vit_pos_embed|vae2llm|llm2vae|connector)\.",
        # LoRA fields used only on B side
        lora_layers=[24, 25, 26, 27],
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )

    # ---- Build model ----
    _log("model_build_start")
    model, vae_model, base_checkpoint_report = _build_model_for_gold40(
        inf_args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens,
    )
    _log("model_built")

    # ---- Inject LoRA only on B side ----
    if side == "B":
        target_names = _iter_target_linear_names(model, inf_args.lora_layers)
        wrappers = inject_lora(model, target_names,
                                inf_args.lora_r, inf_args.lora_alpha, inf_args.lora_dropout)
        trainable_report = set_trainable(model, inf_args.trainable_name_regex, wrappers)
        _log("lora_injected", num_target=len(target_names),
             num_existing=len(trainable_report["existing_trainable_names"]),
             num_lora=len(trainable_report["lora_trainable_names"]))
    else:
        trainable_report = set_trainable(model, inf_args.trainable_name_regex, {})
        _log("no_lora_injected_a_side",
             num_existing=len(trainable_report["existing_trainable_names"]))

    # ---- Strict load step_000 ----
    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    load_report = load_trainable_strict_with_diff(payload, model)
    _log("strict_loaded",
         num_loaded=load_report["num_loaded_tensors"],
         max_abs_load_copy_diff=load_report["max_abs_load_copy_diff"],
         missing=len(load_report.get("missing_keys", [])) if isinstance(load_report.get("missing_keys"), list) else load_report.get("missing_keys"),
         unexpected=len(load_report.get("unexpected_keys", [])) if isinstance(load_report.get("unexpected_keys"), list) else load_report.get("unexpected_keys"))

    if load_report["max_abs_load_copy_diff"] != 0.0:
        raise RuntimeError(f"strict load failed max_abs_load_copy_diff={load_report['max_abs_load_copy_diff']}")

    expected_loaded = 14 if side == "A" else 62
    if load_report["num_loaded_tensors"] != expected_loaded:
        raise RuntimeError(f"expected num_loaded={expected_loaded}, got {load_report['num_loaded_tensors']}")

    # ---- Build inferencer ----
    model.eval()
    vae_model.eval()
    inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, inf_args)

    # ---- Iterate 40 pairs ----
    inputs_manifest_path = out_dir / f"inputs_provenance_{side}.jsonl"
    per_pair_records: List[Dict[str, Any]] = []
    per_pair_manifest_rows: List[Dict[str, Any]] = []

    with open(inputs_manifest_path, "w", encoding="utf-8") as _mf:
        pass  # truncate

    for i, pair in enumerate(records):
        pair_id = pair.get("pair_id")
        pair_seed = _stable_pair_seed(pair_id, args.seed)

        # Compute input hashes deterministically. Note: _generate_one calls
        # _set_seed(seed) internally and then interleave_inference draws
        # initial noise. Therefore same (pair_id, args.seed) -> same
        # pair_seed -> same initial noise across A and B.
        prompt_text = paired_prompt(pair, include_source_image=True)
        prompt_sha = sha256_string(prompt_text)
        cot_text = str(pair.get("target", {}).get("cot_text", "")
                       or pair.get("target_cot_text", "")
                       or pair.get("cot_text", ""))
        cot_sha = sha256_string(cot_text)
        # Source / target image on-disk SHA
        source_img = _load_pair_image(pair, "source")
        # Save a stable byte representation for SHA: use raw bytes of the
        # source PNG on disk if available.
        src_pth = None
        tgt_pth = None
        # Recover file paths for source / target
        for key in ("source_image_path", "source_path", "path"):
            v = (pair.get("source") or {}).get("path") if isinstance(pair.get("source"), dict) else pair.get(key)
            if v:
                src_pth = v
                break
        for key in ("target_image_path", "target_path", "path"):
            v = (pair.get("target") or {}).get("path") if isinstance(pair.get("target"), dict) else pair.get(key)
            if v:
                tgt_pth = v
                break
        # Fall back to computing from PIL image bytes if paths unknown
        src_sha = _sha256_file_local(Path(src_pth)) if src_pth and Path(src_pth).exists() else None
        tgt_sha = _sha256_file_local(Path(tgt_pth)) if tgt_pth and Path(tgt_pth).exists() else None

        # Compute initial-noise SHA by rerunning the deterministic seed then
        # sampling the noise shape used by the inferencer (H, W, 3 latent).
        # This is a provenance signal only; the actual generation performs
        # the same seeding inside _generate_one.
        rng = _snapshot_rng_state()
        try:
            # Match _generate_one seeding: set global then draw a canonical
            # noise probe. We use a fixed-size probe to bind (pair_seed, seed).
            torch.manual_seed(pair_seed)
            np.random.seed(pair_seed)
            random.seed(pair_seed)
            torch.cuda.manual_seed_all(pair_seed)
            noise_probe = torch.randn(16, 128, 128, dtype=torch.float32).cpu().contiguous()
            noise_bytes = noise_probe.numpy().tobytes()
            noise_sha = sha256_bytes(noise_bytes)
        finally:
            _restore_rng_state(rng)

        # ---- Generate PNG ----
        png_path = png_dir / f"{i:03d}__{pair_id}__{side}.png"
        t0 = time.time()
        try:
            _generate_one(inferencer, pair, png_path, inf_args, seed=pair_seed)
            gen_status = "ok"
            gen_error = None
        except Exception as exc:  # noqa: BLE001
            gen_status = "error"
            gen_error = repr(exc)
            _log("gen_error", pair_index=i, pair_id=pair_id, error=gen_error)
            raise
        wall = time.time() - t0

        png_size = png_path.stat().st_size
        png_sha = _sha256_file_local(png_path)

        # PNG pixel content hash (bit-exact bytes independent of file metadata)
        pil = Image.open(png_path).convert("RGB")
        pixel_arr = np.asarray(pil, dtype=np.uint8).tobytes()
        pixel_sha = sha256_bytes(pixel_arr)

        row = {
            "pair_index": i,
            "condition": f"{side}_step000",
            "checkpoint_step": 0,
            "pair_id": pair_id,
            "seed": args.seed,
            "pair_seed": pair_seed,
            "intervention_type": (pair.get("intervention") or {}).get("intervention_type"),
            "intervention_field": (pair.get("intervention") or {}).get("field"),
            "source_image_sha256": src_sha,
            "target_image_sha256": tgt_sha,
            "prompt_sha256": prompt_sha,
            "cot_sha256": cot_sha,
            "noise_sha256": noise_sha,
            "checkpoint_sha256": ckpt_sha,
            "manifest_sha256": manifest_sha,
            "output_path": str(png_path),
            "output_png_size_bytes": png_size,
            "output_png_sha256": png_sha,
            "output_png_pixel_sha256": pixel_sha,
            "gen_wall_seconds": wall,
            "status": gen_status,
            "gen_error": gen_error,
        }
        per_pair_records.append(row)
        with open(inputs_manifest_path, "a", encoding="utf-8") as _mf:
            _mf.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

        _log("pair_done", pair_index=i, pair_id=pair_id, pair_seed=pair_seed,
             png_sha=png_sha, pixel_sha=pixel_sha, wall=wall)

    # ---- Aggregate + write verifier report ----
    peak_alloc = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    peak_reserved = float(torch.cuda.max_memory_reserved() / (1024 * 1024))
    num_alloc_retries = int(torch.cuda.memory_stats().get("num_alloc_retries", 0))

    verifier_report = {
        "status": "passed",
        "side": side,
        "created_at_iso": now_iso(),
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_size_bytes": ckpt_path.stat().st_size,
        "num_pairs_generated": len(per_pair_records),
        "manifest_sha256": manifest_sha,
        "ema_sha256": ema_sha,
        "ae_sha256": ae_sha,
        "seed": args.seed,
        "num_timesteps": args.num_timesteps,
        "precision": args.precision,
        "inference_config_signature": {
            "cfg_text_scale": inf_args.cfg_text_scale,
            "cfg_img_scale": inf_args.cfg_img_scale,
            "cfg_interval_start": inf_args.cfg_interval_start,
            "cfg_interval_end": inf_args.cfg_interval_end,
            "timestep_shift": inf_args.timestep_shift,
            "num_timesteps": inf_args.num_timesteps,
            "cfg_renorm_min": inf_args.cfg_renorm_min,
            "cfg_renorm_type": inf_args.cfg_renorm_type,
            "visual_image_size": inf_args.visual_image_size,
        },
        "base_checkpoint_report": base_checkpoint_report,
        "strict_load_report": load_report,
        "trainable_report": trainable_report,
        "gpu_memory": {
            "peak_allocated_mib": peak_alloc,
            "peak_reserved_mib": peak_reserved,
            "num_alloc_retries": num_alloc_retries,
        },
        "per_pair": per_pair_records,
    }
    _atomic_write_json(out_dir / f"verifier_{side}.json", verifier_report)

    _log("verifier_finished", num_pairs=len(per_pair_records),
         peak_allocated_mib=peak_alloc, peak_reserved_mib=peak_reserved,
         num_alloc_retries=num_alloc_retries)

    # Cleanly free GPU before exiting so the next verifier starts on a clean slate.
    del inferencer, model, vae_model, payload
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# ---------------------------------------------------------------------------
# Orchestrator: launch A verifier, wait, launch B verifier, wait, compare
# ---------------------------------------------------------------------------

def _run_verifier_subprocess(side: str, ckpt_path: Path, out_dir: Path,
                              log_path: Path, args: argparse.Namespace) -> int:
    """Launch the verifier as a fresh subprocess. Return numeric exit code."""
    env = os.environ.copy()
    env.update({
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONUNBUFFERED": "1",
    })
    cmd = [
        sys.executable, "-u", "-m",
        "experiments.counterfactual_cot.training.gold40_v3_step000_alignment_verifier",
        "--mode", "verifier",
        "--side", side,
        "--trainable-checkpoint", str(ckpt_path),
        "--output-dir", str(out_dir),
        "--seed", str(args.seed),
        "--num-timesteps", str(args.num_timesteps),
        "--precision", args.precision,
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env=env, cwd=str(REPO_ROOT))
        rc = proc.wait()
    return rc


def _comparator(a_dir: Path, b_dir: Path, output_report: Path,
                a_rc: int, b_rc: int, protocol_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """CPU-only comparator. Reads verifier_A.json + verifier_B.json + all PNGs."""
    from PIL import Image
    import numpy as np

    with open(a_dir / "verifier_A.json", "r", encoding="utf-8") as f:
        va = json.load(f)
    with open(b_dir / "verifier_B.json", "r", encoding="utf-8") as f:
        vb = json.load(f)

    a_rows = {r["pair_id"]: r for r in va["per_pair"]}
    b_rows = {r["pair_id"]: r for r in vb["per_pair"]}
    a_pids = list(a_rows.keys())
    b_pids = list(b_rows.keys())

    per_pair_compare: List[Dict[str, Any]] = []
    passed_pairs: List[str] = []
    failed_pairs: List[str] = []
    total_pairs = len(a_pids)

    for pid in a_pids:
        if pid not in b_rows:
            failed_pairs.append(pid)
            per_pair_compare.append({
                "pair_id": pid, "reason": "missing_in_b", "equal": False,
            })
            continue
        ra = a_rows[pid]
        rb = b_rows[pid]

        # Provenance checks
        src_eq = ra.get("source_image_sha256") == rb.get("source_image_sha256")
        tgt_eq = ra.get("target_image_sha256") == rb.get("target_image_sha256")
        prompt_eq = ra.get("prompt_sha256") == rb.get("prompt_sha256")
        cot_eq = ra.get("cot_sha256") == rb.get("cot_sha256")
        noise_eq = ra.get("noise_sha256") == rb.get("noise_sha256")
        seed_eq = ra.get("pair_seed") == rb.get("pair_seed")
        manifest_eq = ra.get("manifest_sha256") == rb.get("manifest_sha256")

        # PNG file SHA / pixel SHA / pixel max_abs_diff
        png_sha_eq = ra.get("output_png_sha256") == rb.get("output_png_sha256")
        pixel_sha_eq = ra.get("output_png_pixel_sha256") == rb.get("output_png_pixel_sha256")

        # Decode and compare pixel arrays for max_abs_diff
        max_abs_diff = None
        shape_eq = None
        try:
            pa = np.asarray(Image.open(ra["output_path"]).convert("RGB"), dtype=np.int16)
            pb = np.asarray(Image.open(rb["output_path"]).convert("RGB"), dtype=np.int16)
            shape_eq = pa.shape == pb.shape
            if shape_eq:
                max_abs_diff = int(np.abs(pa - pb).max())
        except Exception as exc:  # noqa: BLE001
            per_pair_compare.append({
                "pair_id": pid, "reason": f"pixel_read_error: {exc!r}", "equal": False,
            })
            failed_pairs.append(pid)
            continue

        all_ok = (
            src_eq and tgt_eq and prompt_eq and cot_eq and noise_eq
            and seed_eq and manifest_eq
            and png_sha_eq and pixel_sha_eq
            and shape_eq is True and max_abs_diff == 0
            and ra.get("status") == "ok" and rb.get("status") == "ok"
        )
        per_pair_compare.append({
            "pair_id": pid,
            "source_image_sha_equal": src_eq,
            "target_image_sha_equal": tgt_eq,
            "prompt_sha_equal": prompt_eq,
            "cot_sha_equal": cot_eq,
            "noise_sha_equal": noise_eq,
            "pair_seed_equal": seed_eq,
            "manifest_sha_equal": manifest_eq,
            "output_png_sha256_equal": png_sha_eq,
            "output_png_pixel_sha256_equal": pixel_sha_eq,
            "pixel_shape_equal": shape_eq,
            "pixel_max_abs_diff": max_abs_diff,
            "a_status": ra.get("status"),
            "b_status": rb.get("status"),
            "equal": all_ok,
            "png_a": ra["output_path"],
            "png_a_sha": ra.get("output_png_sha256"),
            "png_b": rb["output_path"],
            "png_b_sha": rb.get("output_png_sha256"),
        })
        if all_ok:
            passed_pairs.append(pid)
        else:
            failed_pairs.append(pid)

    verifiers_ok = (a_rc == 0 and b_rc == 0)
    all_pairs_ok = (len(failed_pairs) == 0 and total_pairs == 40)
    overall = "passed" if (verifiers_ok and all_pairs_ok) else "failed"

    report = {
        "status": overall,
        "created_at_iso": now_iso(),
        "frozen_inputs": protocol_snapshot,
        "verifier_A_exit_code": a_rc,
        "verifier_B_exit_code": b_rc,
        "verifier_A_report_path": str(a_dir / "verifier_A.json"),
        "verifier_B_report_path": str(b_dir / "verifier_B.json"),
        "verifier_A_report_sha256": sha256_file(a_dir / "verifier_A.json"),
        "verifier_B_report_sha256": sha256_file(b_dir / "verifier_B.json"),
        "verifier_A_num_pairs": va.get("num_pairs_generated"),
        "verifier_B_num_pairs": vb.get("num_pairs_generated"),
        "total_pairs": total_pairs,
        "passed_pairs": passed_pairs,
        "failed_pairs": failed_pairs,
        "num_passed_pairs": len(passed_pairs),
        "num_failed_pairs": len(failed_pairs),
        "gpu_memory_a": va.get("gpu_memory"),
        "gpu_memory_b": vb.get("gpu_memory"),
        "per_pair": per_pair_compare,
        "hard_gates": {
            "G01_verifier_A_exit_code_zero_numeric_captured": a_rc == 0,
            "G02_verifier_B_exit_code_zero_numeric_captured": b_rc == 0,
            "G03_verifier_A_generated_40_pairs": va.get("num_pairs_generated") == 40,
            "G04_verifier_B_generated_40_pairs": vb.get("num_pairs_generated") == 40,
            "G05_all_40_source_image_sha_equal": all(pp.get("source_image_sha_equal") for pp in per_pair_compare),
            "G06_all_40_target_image_sha_equal": all(pp.get("target_image_sha_equal") for pp in per_pair_compare),
            "G07_all_40_prompt_sha_equal": all(pp.get("prompt_sha_equal") for pp in per_pair_compare),
            "G08_all_40_cot_sha_equal": all(pp.get("cot_sha_equal") for pp in per_pair_compare),
            "G09_all_40_noise_sha_equal": all(pp.get("noise_sha_equal") for pp in per_pair_compare),
            "G10_all_40_pair_seed_equal": all(pp.get("pair_seed_equal") for pp in per_pair_compare),
            "G11_all_40_manifest_sha_equal": all(pp.get("manifest_sha_equal") for pp in per_pair_compare),
            "G12_all_40_png_sha_equal": all(pp.get("output_png_sha256_equal") for pp in per_pair_compare),
            "G13_all_40_pixel_sha_equal": all(pp.get("output_png_pixel_sha256_equal") for pp in per_pair_compare),
            "G14_all_40_pixel_shape_equal": all(pp.get("pixel_shape_equal") is True for pp in per_pair_compare),
            "G15_all_40_pixel_max_abs_diff_zero": all(pp.get("pixel_max_abs_diff") == 0 for pp in per_pair_compare),
            "G16_no_pair_status_error_a_side": all(pp.get("a_status") == "ok" for pp in per_pair_compare),
            "G17_no_pair_status_error_b_side": all(pp.get("b_status") == "ok" for pp in per_pair_compare),
        },
        "declarations": [
            "No training performed.",
            "No optimizer.step or backward performed.",
            "No Step400 image generated.",
            "No Judge invoked.",
            "No TS/PS/BA/LS/IQ scoring performed.",
            "Verifiers were run serially; B started only after A subprocess.wait() returned.",
            "No persistent monitoring daemon.",
            "No modification to any checkpoint, manifest, protocol, engineering audit, or adjudication addendum.",
        ],
    }
    _atomic_write_json(output_report, report)
    return report


def _orchestrate_main(args: argparse.Namespace) -> int:
    out_root: Path = args.out_root
    if out_root.exists():
        # Refuse to overwrite an existing non-empty output_dir.
        if any(out_root.iterdir()):
            raise SystemExit(
                f"out_root already exists and is non-empty: {out_root}. "
                f"Refusing to overwrite. Rename/archive it first."
            )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(exist_ok=True)

    # Frozen input verification
    frozen = {
        "protocol_v1_json": {"path": str(PROTOCOL_JSON), "sha256_expected": PROTOCOL_JSON_SHA_EXPECT, "sha256_observed": sha256_file(PROTOCOL_JSON)},
        "protocol_v1_md": {"path": str(PROTOCOL_MD), "sha256_expected": PROTOCOL_MD_SHA_EXPECT, "sha256_observed": sha256_file(PROTOCOL_MD)},
        "provenance_audit": {"path": str(PROVENANCE_AUDIT), "sha256_expected": PROVENANCE_AUDIT_SHA_EXPECT, "sha256_observed": sha256_file(PROVENANCE_AUDIT)},
        "a_step000": {"path": str(A_STEP000), "sha256_expected": A_STEP000_SHA_EXPECT, "sha256_observed": sha256_file(A_STEP000)},
        "b_step000": {"path": str(B_STEP000), "sha256_expected": B_STEP000_SHA_EXPECT, "sha256_observed": sha256_file(B_STEP000)},
        "v3_manifest": {"path": str(V3_MANIFEST), "sha256_expected": V3_MANIFEST_SHA_EXPECT, "sha256_observed": sha256_file(V3_MANIFEST)},
        "base_ema": {"path": str(EMA_PATH), "sha256_expected": BASE_EMA_SHA_EXPECT, "sha256_observed": sha256_file(EMA_PATH)},
        "base_ae": {"path": str(VAE_PATH), "sha256_expected": BASE_AE_SHA_EXPECT, "sha256_observed": sha256_file(VAE_PATH)},
    }
    for name, entry in frozen.items():
        if entry["sha256_observed"] != entry["sha256_expected"]:
            raise SystemExit(f"Frozen input SHA mismatch: {name}\n  expected {entry['sha256_expected']}\n  observed {entry['sha256_observed']}")

    _atomic_write_json(out_root / "frozen_inputs_verified.json", frozen)

    # Also snapshot protocol JSON into out_root for provenance.
    _atomic_write_json(out_root / "protocol_reference.json", {
        "protocol_json_path": str(PROTOCOL_JSON),
        "protocol_json_sha256": frozen["protocol_v1_json"]["sha256_observed"],
        "protocol_md_path": str(PROTOCOL_MD),
        "protocol_md_sha256": frozen["protocol_v1_md"]["sha256_observed"],
    })

    a_dir = out_root / "verifier_A"
    b_dir = out_root / "verifier_B"
    a_log = out_root / "logs/verifier_A.stdout.log"
    b_log = out_root / "logs/verifier_B.stdout.log"

    # ---- Verifier A ----
    print(f"[orchestrator] launching verifier A at {now_iso()}", flush=True)
    a_rc = _run_verifier_subprocess("A", A_STEP000, a_dir, a_log, args)
    print(f"[orchestrator] verifier A finished, exit_code={a_rc}", flush=True)
    if a_rc != 0:
        # Still write partial report so failure is documented; do NOT
        # continue to B by user's brief.
        partial = {
            "status": "failed",
            "reason": "verifier_A_exit_code_nonzero",
            "verifier_A_exit_code": a_rc,
            "verifier_A_log": str(a_log),
            "created_at_iso": now_iso(),
        }
        _atomic_write_json(out_root / "step000_alignment_gate_report.json", partial)
        return 1

    # ---- Verifier B ----
    print(f"[orchestrator] launching verifier B at {now_iso()}", flush=True)
    b_rc = _run_verifier_subprocess("B", B_STEP000, b_dir, b_log, args)
    print(f"[orchestrator] verifier B finished, exit_code={b_rc}", flush=True)
    if b_rc != 0:
        partial = {
            "status": "failed",
            "reason": "verifier_B_exit_code_nonzero",
            "verifier_A_exit_code": a_rc,
            "verifier_B_exit_code": b_rc,
            "verifier_A_log": str(a_log),
            "verifier_B_log": str(b_log),
            "created_at_iso": now_iso(),
        }
        _atomic_write_json(out_root / "step000_alignment_gate_report.json", partial)
        return 1

    # ---- Comparator ----
    print(f"[orchestrator] running comparator at {now_iso()}", flush=True)
    protocol_snapshot = {
        "protocol_v1_json_sha256": PROTOCOL_JSON_SHA_EXPECT,
        "protocol_v1_md_sha256": PROTOCOL_MD_SHA_EXPECT,
        "a_step000_sha256": A_STEP000_SHA_EXPECT,
        "b_step000_sha256": B_STEP000_SHA_EXPECT,
        "gold40_v3_manifest_sha256": V3_MANIFEST_SHA_EXPECT,
        "base_ema_sha256": BASE_EMA_SHA_EXPECT,
        "base_ae_sha256": BASE_AE_SHA_EXPECT,
        "provenance_audit_sha256": PROVENANCE_AUDIT_SHA_EXPECT,
        "seed": args.seed,
        "num_timesteps": args.num_timesteps,
        "precision": args.precision,
    }
    report_path = out_root / "step000_alignment_gate_report.json"
    report = _comparator(a_dir, b_dir, report_path, a_rc, b_rc, protocol_snapshot)
    print(json.dumps({
        "status": report["status"],
        "num_passed_pairs": report["num_passed_pairs"],
        "num_failed_pairs": report["num_failed_pairs"],
        "verifier_A_exit_code": report["verifier_A_exit_code"],
        "verifier_B_exit_code": report["verifier_B_exit_code"],
        "report_path": str(report_path),
    }, indent=2))
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    args = parse_args()
    if args.mode == "verifier":
        return _verifier_main(args)
    return _orchestrate_main(args)


if __name__ == "__main__":
    sys.exit(main())
