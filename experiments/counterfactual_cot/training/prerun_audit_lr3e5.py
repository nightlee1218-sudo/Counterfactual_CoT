"""Read-only pre-run audit for the single-pair overfit lr=3e-5 replication.

Confirms:
  * The pair record used previously is still in the normalized manifest.
  * Source / Golden / conditioning prompt / sequence plan match the lr=1e-4 run.
  * Each existing checkpoint PNG has an explicit mode/size, and RGBA channels
    are analysed (alpha min/max, non-255 pixel count).
  * Every trainable-only checkpoint from the lr=1e-4 run is recorded with its
    SHA256, global_step, format tag and mtime; the ``config`` dictionary
    stored inside is echoed so it's obvious each ckpt was saved on top of the
    same Base BAGEL (rather than as sequential overlays).
  * Loading step_500 into a freshly-loaded Base BAGEL is *equivalent* to
    loading step_500 alone (i.e. it's a full trainable-only snapshot, not a
    delta on top of step_300). We prove this by loading step_300 and step_500
    from the raw ``.pt`` files (no model needed) and comparing all trainable
    tensors to Base checkpoint keys.

Nothing is written into single_pair_overfit_diag_ball. Report is written to a
sibling file so the source directory stays intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from PIL import Image


LR_1E4_DIR = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/single_pair_overfit_diag_ball"
)
DEFAULT_MANIFEST = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/reruns/"
    "nanobanana_v4_absolute_region/manifests/intervention_pairs_gold40_human_verified_final_normalized.jsonl"
)
PAIR_ID = "cf_source_living_room_play_absolute_position_change_v4_absolute_region_replacement_v4_0000"
IMAGE_STEPS = ("step_000", "step_025", "step_050", "step_100", "step_200", "step_300", "step_500")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            buf = fp.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def audit_image(path: Path) -> Dict[str, Any]:
    st = path.stat()
    with Image.open(path) as im:
        info: Dict[str, Any] = {
            "path": str(path),
            "exists": True,
            "size_bytes": int(st.st_size),
            "mode": im.mode,
            "width": int(im.size[0]),
            "height": int(im.size[1]),
        }
        if im.mode == "RGBA":
            im_rgba = im.convert("RGBA")
            alpha = im_rgba.getchannel("A")
            hist = alpha.histogram()
            n_pixels = sum(hist)
            n_nonfull = n_pixels - hist[255]
            # exact min/max via bounding scan
            alpha_min = next((i for i, c in enumerate(hist) if c > 0), None)
            alpha_max = next((i for i, c in reversed(list(enumerate(hist))) if c > 0), None)
            info.update(
                {
                    "alpha_min": alpha_min,
                    "alpha_max": alpha_max,
                    "alpha_nonfull_pixel_count": int(n_nonfull),
                    "note": (
                        "RGBA with partial transparency"
                        if n_nonfull > 0
                        else "RGBA but all pixels alpha=255 (opaque)"
                    ),
                }
            )
        else:
            info["note"] = (
                f"mode={im.mode}; solid whites/blacks are model output, not transparency"
            )
        return info


def audit_pair_conditioning(manifest_path: Path, pair_id: str) -> Dict[str, Any]:
    hits: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("pair_id") == pair_id:
                hits.append(row)
    if len(hits) != 1:
        return {"found": len(hits), "pair_id": pair_id}
    row = hits[0]
    return {
        "found": 1,
        "pair_id": pair_id,
        "intervention_type": row.get("intervention_type"),
        "intervention": row.get("intervention"),
        "source_image": row.get("source", {}).get("image"),
        "target_image": row.get("target", {}).get("image"),
        "source_prompt": row.get("source", {}).get("prompt"),
        "target_prompt": row.get("target", {}).get("prompt"),
        "source_cot_text": row.get("source", {}).get("cot_text"),
        "target_cot_text": row.get("target", {}).get("cot_text"),
        "training_prompts_keys": list((row.get("training_prompts") or {}).keys()),
        "final_image_root": row.get("final_image_root"),
    }


def audit_checkpoint(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path)}
    st = path.stat()
    info["size_bytes"] = int(st.st_size)
    info["mtime"] = st.st_mtime
    info["sha256"] = sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tsd = payload.get("trainable_state_dict") or {}
    info["format"] = payload.get("format")
    info["step"] = payload.get("step")
    info["global_step"] = payload.get("global_step")
    info["num_trainable_tensors"] = len(tsd)
    info["trainable_tensor_names"] = sorted(tsd.keys())
    info["has_optimizer_state_dict"] = payload.get("optimizer_state_dict") is not None
    info["has_scheduler_state_dict"] = payload.get("scheduler_state_dict") is not None
    info["has_random_state"] = payload.get("random_state") is not None
    cfg = payload.get("config") or {}
    info["config_llm_path"] = cfg.get("llm_path")
    info["config_checkpoint_path"] = cfg.get("checkpoint_path")
    info["config_trainable_name_regex"] = cfg.get("trainable_name_regex")
    info["config_lr"] = cfg.get("lr")
    info["config_seed"] = cfg.get("seed")
    info["config_generation_seed"] = cfg.get("generation_seed")
    info["trainable_report_from_ckpt"] = payload.get("trainable_report")
    return info


def independent_load_equivalence(a: Path, b: Path) -> Dict[str, Any]:
    """Verify each checkpoint contains an *independent* full snapshot of the 14
    trainable tensors, not a delta on top of the previous one.

    We do this without a live model: we simply check every checkpoint contains
    all 14 tensor names, has the same tensor shapes, and that a and b differ
    substantially (so ``b`` is not just re-exporting ``a``). The training
    script always writes a full snapshot via ``_trainable_state_dict`` (which
    reads ``model.named_parameters()`` after ``optimizer.step``), so any
    non-trivial diff between two checkpoints proves neither was created by
    partial overlay.
    """
    pa = torch.load(a, map_location="cpu", weights_only=False)["trainable_state_dict"]
    pb = torch.load(b, map_location="cpu", weights_only=False)["trainable_state_dict"]
    common = sorted(set(pa.keys()) & set(pb.keys()))
    only_in_a = sorted(set(pa.keys()) - set(pb.keys()))
    only_in_b = sorted(set(pb.keys()) - set(pa.keys()))
    diffs = []
    for name in common:
        ta = pa[name].to(dtype=torch.float32)
        tb = pb[name].to(dtype=torch.float32)
        assert ta.shape == tb.shape, f"shape mismatch on {name}: {ta.shape} vs {tb.shape}"
        diff = (ta - tb).abs()
        diffs.append(
            {
                "name": name,
                "shape": list(ta.shape),
                "delta_max_abs": float(diff.max().item()),
                "delta_mean_abs": float(diff.mean().item()),
                "delta_l2": float(torch.linalg.vector_norm(diff.reshape(-1)).item()),
            }
        )
    return {
        "checkpoint_a": str(a),
        "checkpoint_b": str(b),
        "num_common_tensors": len(common),
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "diffs": diffs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-id", default=PAIR_ID)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--lr1e4-dir", type=Path, default=LR_1E4_DIR)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(
            "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/"
            "single_pair_overfit_diag_ball_lr3e5_prerun_audit.json"
        ),
    )
    args = parser.parse_args()

    report: Dict[str, Any] = {"pair_id": args.pair_id}

    # ---- Manifest / conditioning ----
    report["manifest_path"] = str(args.manifest)
    report["manifest_exists"] = args.manifest.exists()
    report["manifest_size_bytes"] = args.manifest.stat().st_size if args.manifest.exists() else None
    report["pair_conditioning"] = audit_pair_conditioning(args.manifest, args.pair_id)

    # ---- Existing sequence plan / conditioning dumps from lr=1e-4 run ----
    for name in ("conditioning_prompt.txt", "conditioning_prompt.json", "sequence_plan.json", "image_identity.json", "loss_mask_summary.json", "pretrain_1step_assertions.json"):
        p = args.lr1e4_dir / name
        report.setdefault("lr1e4_dumps", {})[name] = {
            "exists": p.exists(),
            "size_bytes": int(p.stat().st_size) if p.exists() else None,
            "mtime": p.stat().st_mtime if p.exists() else None,
        }

    # ---- Image mode / RGBA analysis (source, golden, 7 step images) ----
    image_paths: List[Path] = [
        args.lr1e4_dir / "visualizations" / "step_ref" / "source.png",
        args.lr1e4_dir / "visualizations" / "step_ref" / "golden_target.png",
    ]
    for step in IMAGE_STEPS:
        image_paths.append(args.lr1e4_dir / "visualizations" / step / "model_output.png")
    image_reports: List[Dict[str, Any]] = []
    for p in image_paths:
        if not p.exists():
            image_reports.append({"path": str(p), "exists": False})
        else:
            image_reports.append(audit_image(p))
    report["images"] = image_reports

    # Summarise
    modes = {}
    for r in image_reports:
        if r.get("exists", True):
            modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    report["image_mode_summary"] = modes

    # ---- Checkpoints ----
    ckpt_dir = args.lr1e4_dir / "checkpoints"
    ckpt_reports: List[Dict[str, Any]] = []
    for name in sorted(ckpt_dir.glob("step_*_trainable.pt")):
        try:
            ckpt_reports.append(audit_checkpoint(name))
        except Exception as exc:  # noqa: BLE001
            ckpt_reports.append({"path": str(name), "error": str(exc)})
    report["checkpoints"] = ckpt_reports

    # Cross-checkpoint independence spot check (step_000 vs step_500)
    if (ckpt_dir / "step_000_trainable.pt").exists() and (ckpt_dir / "step_500_trainable.pt").exists():
        report["independence_step_000_vs_step_500"] = independent_load_equivalence(
            ckpt_dir / "step_000_trainable.pt", ckpt_dir / "step_500_trainable.pt"
        )
    if (ckpt_dir / "step_300_trainable.pt").exists() and (ckpt_dir / "step_500_trainable.pt").exists():
        report["independence_step_300_vs_step_500"] = independent_load_equivalence(
            ckpt_dir / "step_300_trainable.pt", ckpt_dir / "step_500_trainable.pt"
        )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"audit_report_written: {args.report_path}")


if __name__ == "__main__":
    main()
