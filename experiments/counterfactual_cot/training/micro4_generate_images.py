#!/usr/bin/env python
"""micro4 diagnostic image generation (minimal deterministic set).

Generates Source | Golden | step_000 | step_400 for each of the 4 micro4 pairs
so the optimisation result can be reviewed visually.

Design notes
------------
* Mirrors the verified model-build path in ``micro4_deterministic_eval.py``
  (``_build_model_with_lora`` + ``_apply_trainable_and_audit``), then loads a
  trainable-only checkpoint with ``load_trainable_strict``.
* Uses the same inference recipe as ``gold40_overfit._generate_one``:
  ``paired_prompt(pair, include_source_image=True)`` and
  ``InterleaveInferencer.interleave_inference``.
* ``model.eval()`` is REQUIRED before generation: ``Qwen2Model.forward()``
  routes to ``forward_inference`` only in eval mode, and that path is the one
  that allocates/uses the KV cache. Staying in ``train()`` mode would route to
  ``forward_train`` and produce invalid output.
* Seed handling: each pair gets ``GEN_SEED_BASE + pair_index * SEED_STRIDE``,
  and the same seed is reused for every checkpoint. This makes the ONLY
  variable across step_000/step_400 the LoRA weights, which is the point.
* ``--preflight`` validates imports/paths/checkpoints without loading the model
  or touching CUDA.

Usage
-----
    # static / path preflight (no model load, no CUDA)
    python micro4_generate_images.py --preflight

    # generate the 8 diagnostic images
    python micro4_generate_images.py --generate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# BAGEL_ROOT bootstrap (mirrors micro4_deterministic_eval.py)
# ---------------------------------------------------------------------------
BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

# ---------------------------------------------------------------------------
# Frozen constants
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
MICRO4_MANIFEST_SHA256 = (
    "2d0790de5c4ad3a66eb26f4b46b45c1ed35155edfc1bb91aeb869ac256494891"
)

DEFAULT_OUTPUT_DIR = (
    BAGEL_ROOT
    / "experiments/counterfactual_cot/outputs/gold40_v3_lora_micro4_diag_images_v1"
)

# Checkpoints to visualise (minimal diagnostic set).
VIS_CHECKPOINTS: List[str] = ["step_000", "step_400"]

CKPT_SHA256: Dict[str, str] = {
    "step_000": "4bf3c41e126008f4c495efe3fa8298f52ddad7c9b1d5105c9e9671c663278692",
    "step_400": "ab84f0f60d6e7a1ebd8f47c396e46cd5fcf259b1e5be0bcc6ccd2cad05493e90",
}

PAIR_ORDER: List[str] = [
    "cf_source_city_street_object_replace_0000",
    "cf_source_classroom_object_replace_0000",
    "cf_source_city_street_absolute_position_change_v4_absolute_region_review_fix_v6_short_0000",
    "cf_source_classroom_absolute_position_change_v4_absolute_region_replacement_v4_0000",
]

EXPECTED_TRAINABLE_TENSORS = 62

# Generation config (matches gold40_overfit defaults / Protocol v1).
GEN_SEED_BASE = 1234
SEED_STRIDE = 1000
VISUAL_IMAGE_SIZE = 1024
NUM_TIMESTEPS = 25
CFG_TEXT_SCALE = 4.0
CFG_IMG_SCALE = 2.0
CFG_INTERVAL_START = 0.0
CFG_INTERVAL_END = 1.0
TIMESTEP_SHIFT = 3.0
CFG_RENORM_MIN = 0.0
CFG_RENORM_TYPE = "text_channel"


# ---------------------------------------------------------------------------
# Utilities (no model imports)
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def _resolve_image_path(pair: Dict[str, Any], path_value: str) -> Path:
    """Resolve a manifest image path (absolute, or relative to image root)."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    image_root = pair.get("final_image_root")
    if image_root:
        rooted = Path(image_root) / path
        if rooted.exists():
            return rooted
    return MICRO4_MANIFEST.parent / path


# ---------------------------------------------------------------------------
# Preflight (no model load, no CUDA)
# ---------------------------------------------------------------------------
def run_preflight(output_dir: Path) -> int:
    checks: List[Tuple[str, bool, str]] = []

    def _chk(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, bool(passed), detail))

    _chk("output_dir_under_group", str(BAGEL_ROOT) in str(output_dir),
         str(output_dir))

    # manifest
    _chk("manifest_exists", MICRO4_MANIFEST.exists(), str(MICRO4_MANIFEST))
    if MICRO4_MANIFEST.exists():
        got = _sha256_file(MICRO4_MANIFEST)
        _chk("manifest_sha256", got == MICRO4_MANIFEST_SHA256,
             f"got={got} expected={MICRO4_MANIFEST_SHA256}")

    # checkpoints
    for label in VIS_CHECKPOINTS:
        ckpt = MICRO4_CKPT_DIR / "checkpoints" / f"{label}_trainable.pt"
        _chk(f"ckpt_exists_{label}", ckpt.exists(), str(ckpt))
        if ckpt.exists():
            got = _sha256_file(ckpt)
            _chk(f"ckpt_sha256_{label}", got == CKPT_SHA256[label],
                 f"got={got} expected={CKPT_SHA256[label]}")

    # import health (catches interface drift without loading weights)
    try:
        from experiments.counterfactual_cot.training.gold40_overfit import (  # noqa: F401
            _make_inferencer, _generate_one, _resolve_image_path as _rp,
            _load_pair_image, _save_reference_image,
        )
        _chk("import_gold40_overfit", True)
    except Exception as exc:  # pragma: no cover
        _chk("import_gold40_overfit", False, f"{type(exc).__name__}: {exc}")

    try:
        from experiments.counterfactual_cot.training.gold40_lora import (  # noqa: F401
            _build_model_with_lora, _apply_trainable_and_audit,
        )
        _chk("import_gold40_lora", True)
    except Exception as exc:  # pragma: no cover
        _chk("import_gold40_lora", False, f"{type(exc).__name__}: {exc}")

    try:
        from experiments.counterfactual_cot.training.lora_2step_smoke import (  # noqa: F401
            load_trainable_strict,
        )
        _chk("import_load_trainable_strict", True)
    except Exception as exc:  # pragma: no cover
        _chk("import_load_trainable_strict", False, f"{type(exc).__name__}: {exc}")

    try:
        from experiments.counterfactual_cot.training.smoke_train import (  # noqa: F401
            _device, _load_tokenizer, _set_seed,
        )
        _chk("import_smoke_train", True)
    except Exception as exc:  # pragma: no cover
        _chk("import_smoke_train", False, f"{type(exc).__name__}: {exc}")

    try:
        from inferencer import InterleaveInferencer  # noqa: F401
        _chk("import_interleave_inferencer", True)
    except Exception as exc:  # pragma: no cover
        _chk("import_interleave_inferencer", False, f"{type(exc).__name__}: {exc}")

    try:
        import torch  # noqa: F401
        _chk("import_torch", True, f"version={torch.__version__}")
    except Exception as exc:  # pragma: no cover
        _chk("import_torch", False, f"{type(exc).__name__}: {exc}")

    # manifest contents + referenced images
    pairs: List[Dict[str, Any]] = []
    if MICRO4_MANIFEST.exists():
        from experiments.counterfactual_cot.data.schema import load_jsonl
        pairs = list(load_jsonl(MICRO4_MANIFEST))
        _chk("manifest_record_count_4", len(pairs) == 4, f"got {len(pairs)}")
        ids = {p["pair_id"] for p in pairs}
        missing = [p for p in PAIR_ORDER if p not in ids]
        _chk("manifest_contains_all_pair_ids", not missing, f"missing={missing}")

        img_missing: List[str] = []
        for pair in pairs:
            for side in ("source", "target"):
                p = _resolve_image_path(pair, pair[side]["image"])
                if not p.exists():
                    img_missing.append(f"{pair['pair_id']}:{side}:{p}")
        _chk("manifest_images_exist", not img_missing, f"missing={img_missing}")

    passed = sum(1 for _, ok, _ in checks if ok)
    failed = len(checks) - passed

    report = {
        "overall_status": "passed" if failed == 0 else "failed",
        "total_checks": len(checks),
        "passed": passed,
        "failed": failed,
        "checks": [{"name": n, "status": "passed" if ok else "failed",
                    "detail": d} for n, ok, d in checks],
        "paths_to_read": _preflight_read_paths(pairs),
        "paths_to_write": _preflight_write_paths(output_dir),
        "generation_config": {
            "gen_seed_base": GEN_SEED_BASE,
            "seed_stride": SEED_STRIDE,
            "num_timesteps": NUM_TIMESTEPS,
            "cfg_text_scale": CFG_TEXT_SCALE,
            "cfg_img_scale": CFG_IMG_SCALE,
            "cfg_interval": [CFG_INTERVAL_START, CFG_INTERVAL_END],
            "timestep_shift": TIMESTEP_SHIFT,
            "cfg_renorm_min": CFG_RENORM_MIN,
            "cfg_renorm_type": CFG_RENORM_TYPE,
            "visual_image_size": VISUAL_IMAGE_SIZE,
            "checkpoints": VIS_CHECKPOINTS,
        },
    }
    out = output_dir / "preflight_report.json"
    _atomic_write_json(out, report)
    for name, ok, detail in checks:
        mark = "passed" if ok else "FAILED"
        print(f"  [{mark}] {name}" + (f": {detail}" if detail else ""))
    print(f"\nPreflight {'PASSED' if failed == 0 else 'FAILED'} ({passed}/{len(checks)})")
    print(f"Report: {out}")
    print(f"SHA256: {_sha256_file(out)}")
    return 0 if failed == 0 else 1


def _preflight_read_paths(pairs: List[Dict[str, Any]]) -> List[str]:
    paths: List[str] = [str(MICRO4_MANIFEST)]
    for label in VIS_CHECKPOINTS:
        paths.append(str(MICRO4_CKPT_DIR / "checkpoints" / f"{label}_trainable.pt"))
    for pair in pairs:
        for side in ("source", "target"):
            paths.append(str(_resolve_image_path(pair, pair[side]["image"])))
    paths += [
        "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors",
        "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors",
        "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/llm_config.json",
        "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/vit_config.json",
        str(BAGEL_ROOT / "inferencer.py"),
        str(BAGEL_ROOT / "modeling/bagel/bagel.py"),
        str(BAGEL_ROOT / "modeling/bagel/qwen2_navit.py"),
        str(BAGEL_ROOT / "modeling/autoencoder.py"),
    ]
    return paths


def _preflight_write_paths(output_dir: Path) -> List[str]:
    names = [
        "preflight_report.json",
        "generation_manifest.json",
        "review.html",
        "contact_sheet.png",
    ]
    paths = [str(output_dir / n) for n in names]
    for label in VIS_CHECKPOINTS:
        for pair_id in PAIR_ORDER:
            paths.append(str(output_dir / "images" / f"{pair_id}__{label}.png"))
    return paths


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def run_generate(output_dir: Path) -> int:
    import torch
    from experiments.counterfactual_cot.data.schema import load_jsonl
    from experiments.counterfactual_cot.training.gold40_overfit import (
        _generate_one, _make_inferencer, _save_reference_image,
    )
    from experiments.counterfactual_cot.training.gold40_lora import (
        _apply_trainable_and_audit, _build_model_with_lora,
    )
    from experiments.counterfactual_cot.training.lora_2step_smoke import (
        load_trainable_strict,
    )
    from experiments.counterfactual_cot.training.smoke_train import (
        _device, _load_tokenizer,
    )

    # G_dir: refuse to run into a non-empty output directory
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"G_dir FAIL: output directory is not empty: {output_dir}\n"
            "Rename or clear it before generating."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # G1: manifest integrity
    got_manifest_sha = _sha256_file(MICRO4_MANIFEST)
    if got_manifest_sha != MICRO4_MANIFEST_SHA256:
        raise RuntimeError(
            f"G1 FAIL: manifest sha256 mismatch got={got_manifest_sha} "
            f"expected={MICRO4_MANIFEST_SHA256}"
        )

    # G2/G3: checkpoint presence + sha256
    for label in VIS_CHECKPOINTS:
        ckpt = MICRO4_CKPT_DIR / "checkpoints" / f"{label}_trainable.pt"
        if not ckpt.exists():
            raise RuntimeError(f"G2 FAIL: missing checkpoint {ckpt}")
        got = _sha256_file(ckpt)
        if got != CKPT_SHA256[label]:
            raise RuntimeError(
                f"G3 FAIL [{label}]: sha256 got={got} expected={CKPT_SHA256[label]}"
            )

    pairs = list(load_jsonl(MICRO4_MANIFEST))
    by_id = {p["pair_id"]: p for p in pairs}
    missing = [p for p in PAIR_ORDER if p not in by_id]
    if missing:
        raise RuntimeError(f"G6 FAIL: manifest missing pairs={missing}")
    ordered = [by_id[pid] for pid in PAIR_ORDER]

    # ---- Build model (mirrors verified eval path) -------------------------
    args = argparse.Namespace(
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
        visual_image_size=VISUAL_IMAGE_SIZE,
        visual_vae_min_image_size=512,
        visual_vit_image_size=980,
        visual_vit_min_image_size=224,
        expected_num_tokens=512,
        max_num_tokens_per_sample=1536,
        max_num_tokens=1536,
        seed=1234,
        variant="paired_cf_mvp",
        num_samples=4,
        require_paired_vit=True,
        cfg_text_scale=CFG_TEXT_SCALE,
        cfg_img_scale=CFG_IMG_SCALE,
        cfg_interval_start=CFG_INTERVAL_START,
        cfg_interval_end=CFG_INTERVAL_END,
        timestep_shift=TIMESTEP_SHIFT,
        num_timesteps=NUM_TIMESTEPS,
        cfg_renorm_min=CFG_RENORM_MIN,
        cfg_renorm_type=CFG_RENORM_TYPE,
    )

    device = _device(args.device)
    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(args.llm_path)
    model, vae_model, wrappers, target_names, _base_report = _build_model_with_lora(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens
    )
    audit = _apply_trainable_and_audit(model, wrappers, target_names, args, device.type)

    n_trainable = len({n for n, p in model.named_parameters() if p.requires_grad})
    if n_trainable != EXPECTED_TRAINABLE_TENSORS:
        raise RuntimeError(
            f"G4 FAIL: trainable tensors got={n_trainable} "
            f"expected={EXPECTED_TRAINABLE_TENSORS}"
        )
    print(f"[ok] model built, trainable tensors={n_trainable}, "
          f"lora_tensors={audit.get('lora_tensors')}")

    # ---- Reference images (Source + Golden per pair) ----------------------
    ref_dir = output_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for pair_idx, pair in enumerate(ordered):
        pid = pair["pair_id"]
        _save_reference_image(
            pair, "source", ref_dir / f"{pid}__source.png", VISUAL_IMAGE_SIZE
        )
        _save_reference_image(
            pair, "target", ref_dir / f"{pid}__golden.png", VISUAL_IMAGE_SIZE
        )

    # ---- Generate per checkpoint -----------------------------------------
    inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, args)
    records: List[Dict[str, Any]] = []

    for label in VIS_CHECKPOINTS:
        ckpt = MICRO4_CKPT_DIR / "checkpoints" / f"{label}_trainable.pt"
        load_result = load_trainable_strict(ckpt, model)
        if load_result["num_loaded_tensors"] != EXPECTED_TRAINABLE_TENSORS:
            raise RuntimeError(
                f"G12 FAIL [{label}]: num_loaded_tensors="
                f"{load_result['num_loaded_tensors']}"
            )
        if float(load_result["max_abs_load_copy_diff"]) != 0.0:
            raise RuntimeError(
                f"G12 FAIL [{label}]: max_abs_load_copy_diff="
                f"{load_result['max_abs_load_copy_diff']}"
            )
        print(f"[ok] loaded {label}: {load_result['num_loaded_tensors']} tensors, "
              f"max_diff={load_result['max_abs_load_copy_diff']}")

        # CRITICAL: eval() routes Qwen2Model.forward() to forward_inference
        # (the KV-cache path). train() would route to forward_train.
        model.eval()
        if model.training:
            raise RuntimeError("model must be in eval() mode for generation")

        for pair_idx, pair in enumerate(ordered):
            pid = pair["pair_id"]
            seed = GEN_SEED_BASE + pair_idx * SEED_STRIDE
            out_path = images_dir / f"{pid}__{label}.png"
            with torch.no_grad():
                _generate_one(
                    inferencer, pair, out_path, args, seed=seed,
                )
            if not out_path.exists():
                raise RuntimeError(f"G8 FAIL: no output written for {out_path}")
            size = out_path.stat().st_size
            if size <= 0:
                raise RuntimeError(f"G8 FAIL: empty png {out_path}")
            records.append({
                "pair_id": pid,
                "pair_index": pair_idx,
                "checkpoint": label,
                "checkpoint_sha256": CKPT_SHA256[label],
                "seed": seed,
                "num_timesteps": NUM_TIMESTEPS,
                "image": str(out_path),
                "size_bytes": size,
                "sha256": _sha256_file(out_path),
            })
            print(f"  [ok] {label} / {pid} -> {out_path.name} ({size} bytes)")

    _atomic_write_json(output_dir / "generation_manifest.json", {
        "schema": "micro4_diag_images_v1",
        "manifest_sha256": MICRO4_MANIFEST_SHA256,
        "generation_config": {
            "gen_seed_base": GEN_SEED_BASE,
            "seed_stride": SEED_STRIDE,
            "num_timesteps": NUM_TIMESTEPS,
            "cfg_text_scale": CFG_TEXT_SCALE,
            "cfg_img_scale": CFG_IMG_SCALE,
            "cfg_interval": [CFG_INTERVAL_START, CFG_INTERVAL_END],
            "timestep_shift": TIMESTEP_SHIFT,
            "cfg_renorm_min": CFG_RENORM_MIN,
            "cfg_renorm_type": CFG_RENORM_TYPE,
            "visual_image_size": VISUAL_IMAGE_SIZE,
            "prompt_mode": "paired_prompt(include_source_image=True)",
            "model_mode": "eval()",
        },
        "records": records,
    })

    _build_review(output_dir, ordered)
    print(f"\nGeneration complete. Output: {output_dir}")
    return 0


def _build_review(output_dir: Path, ordered: List[Dict[str, Any]]) -> None:
    """Build a per-pair contact sheet + review.html for human inspection."""
    from PIL import Image, ImageDraw, ImageFont

    tile = 512
    label_h = 34
    pad = 8

    def _font():
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
            )
        except Exception:
            return ImageFont.load_default()

    font = _font()
    sheets: List[Path] = []

    for pair in ordered:
        pid = pair["pair_id"]
        ref_dir = output_dir / "references"
        images_dir = output_dir / "images"
        entries = [
            ("Source", ref_dir / f"{pid}__source.png"),
            ("Golden", ref_dir / f"{pid}__golden.png"),
        ]
        for label in VIS_CHECKPOINTS:
            entries.append((label, images_dir / f"{pid}__{label}.png"))

        loaded = []
        for name, p in entries:
            if not p.exists():
                continue
            im = Image.open(p).convert("RGB")
            im = im.resize((tile, tile), Image.BICUBIC)
            loaded.append((name, im))
        if not loaded:
            continue

        total_w = tile * len(loaded) + pad * (len(loaded) + 1)
        total_h = tile + label_h + pad * 2
        canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        x = pad
        for name, im in loaded:
            canvas.paste(im, (x, label_h + pad))
            try:
                bbox = draw.textbbox((0, 0), name, font=font)
                tw = bbox[2] - bbox[0]
            except Exception:
                tw = len(name) * 9
            draw.text((x + (tile - tw) // 2, pad), name, fill=(20, 20, 20), font=font)
            x += tile + pad
        sheet = output_dir / f"contact_{pid}.png"
        canvas.save(sheet, format="PNG")
        sheets.append(sheet)

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>micro4 diagnostic images</title>",
        "<style>body{font-family:sans-serif;margin:24px;background:#fafafa}",
        "h2{margin-top:32px}img.contact{max-width:100%;border:1px solid #ccc}",
        ".meta{color:#555;font-size:13px}</style></head><body>",
        "<h1>micro4 Diagnostic Images</h1>",
        "<p class='meta'>Each row: Source | Golden (target) | step_000 | step_400. "
        "Only the LoRA weights differ between step_000 and step_400; "
        "the same seed and generation config are used for both.</p>",
    ]
    for pair in ordered:
        pid = pair["pair_id"]
        sheet = output_dir / f"contact_{pid}.png"
        if not sheet.exists():
            continue
        iv = pair.get("intervention", {})
        desc = (
            f"{iv.get('entity_id')}.{iv.get('field')}: "
            f"{iv.get('old_value')} -> {iv.get('new_value')}"
        )
        parts.append(f"<h2>{pid}</h2>")
        parts.append(f"<p class='meta'>Intervention: {desc}</p>")
        parts.append(
            f"<a href='{sheet.name}' target='_blank'>"
            f"<img class='contact' src='{sheet.name}' alt='{pid}'/></a>"
        )
    parts.append("</body></html>")
    (output_dir / "review.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"[ok] review.html with {len(sheets)} contact sheets")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate micro4 diagnostic images (step_000 vs step_400)."
    )
    parser.add_argument("--preflight", action="store_true",
                        help="Static/path checks only; no model load, no CUDA.")
    parser.add_argument("--generate", action="store_true",
                        help="Run generation (loads model, uses GPU).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.preflight and args.generate:
        raise SystemExit("Use either --preflight or --generate, not both.")
    if not args.preflight and not args.generate:
        raise SystemExit("Specify --preflight or --generate.")
    if args.preflight:
        return run_preflight(args.output_dir)
    return run_generate(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())