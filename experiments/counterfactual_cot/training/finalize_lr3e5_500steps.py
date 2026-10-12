"""End-of-run finalization for the lr=3e-5 500-step replication:

1. Compare every common trainable-only checkpoint (0/10/25/50/100/200/300)
   between OLD (lr=3e-5 300-step) and NEW (lr=3e-5 500-step): tensor-by-tensor
   exact equality is expected because the first 300 steps must be
   bit-for-bit deterministic given identical seeds + Base ckpt + code.

2. Compare every common visualization PNG (step_ref/source, step_ref/golden,
   step_000/010/025/050/100/200/300) by SHA256.

3. Compare every train_step row across the entire overlap (steps 1..300).

4. Build review.html + contact_sheet.png inside NEW dir.

5. Build a two-row lr=1e-4 vs lr=3e-5 comparison contact sheet using only
   the common save steps (0, 25, 50, 100, 200, 300, 500) plus Source & Golden,
   at outputs/lr1e4_vs_lr3e5_contact_sheet.png.

6. Emit a machine-readable final report at
   outputs/single_pair_overfit_diag_ball_lr3e5_500steps/final_report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont


OLD_LR3E5_DIR = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/"
    "single_pair_overfit_diag_ball_lr3e5"
)
NEW_LR3E5_DIR = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/"
    "single_pair_overfit_diag_ball_lr3e5_500steps"
)
LR1E4_DIR = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/"
    "single_pair_overfit_diag_ball"
)

# Save steps present in the new lr=3e-5 500-step run.
NEW_SAVE_STEPS = (0, 10, 25, 50, 100, 200, 300, 400, 500)
# Steps common with old lr=3e-5 300-step run (for strict determinism check).
OLD_COMMON_STEPS = (0, 10, 25, 50, 100, 200, 300)
# Steps common between lr=1e-4 (0/25/50/100/200/300/500) and lr=3e-5 (500-step)
# for the two-row comparison.
LR_COMPARE_STEPS = (0, 25, 50, 100, 200, 300, 500)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            buf = fp.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def parse_train_step_rows(log_path: Path) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = {}
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") != "train_step":
            continue
        rows[int(r["step"])] = r
    return rows


def compare_train_logs(old_log: Path, new_log: Path) -> Dict[str, Any]:
    old = parse_train_step_rows(old_log)
    new = parse_train_step_rows(new_log)
    overlap = sorted(set(old) & set(new))
    fields = ("mse", "ce", "total", "grad_l2_total", "grad_max_over_tensors",
              "grad_norm_after_clip", "step_seed", "timesteps_hash")
    exact_rows: List[int] = []
    mismatched_rows: List[Dict[str, Any]] = []
    for s in overlap:
        ok = True
        for k in fields:
            if old[s].get(k) != new[s].get(k):
                ok = False
                mismatched_rows.append({"step": s, "field": k, "old": old[s].get(k), "new": new[s].get(k)})
        if ok:
            exact_rows.append(s)
    return {
        "old_log": str(old_log),
        "new_log": str(new_log),
        "num_overlap_steps": len(overlap),
        "num_exact_match_rows": len(exact_rows),
        "num_mismatched_field_rows": len(mismatched_rows),
        "first_20_mismatches": mismatched_rows[:20],
        "step1_mse_old": old.get(1, {}).get("mse"),
        "step1_mse_new": new.get(1, {}).get("mse"),
    }


def compare_ckpt_pair(old_path: Path, new_path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "old_path": str(old_path),
        "new_path": str(new_path),
        "old_size": old_path.stat().st_size,
        "new_size": new_path.stat().st_size,
        "old_sha256": sha256_file(old_path),
        "new_sha256": sha256_file(new_path),
    }
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    new = torch.load(new_path, map_location="cpu", weights_only=False)
    old_ts = old["trainable_state_dict"]
    new_ts = new["trainable_state_dict"]
    common = sorted(set(old_ts) & set(new_ts))
    info["only_old"] = sorted(set(old_ts) - set(new_ts))
    info["only_new"] = sorted(set(new_ts) - set(old_ts))
    max_abs_all = 0.0
    all_identical = True
    per_tensor: List[Dict[str, Any]] = []
    for name in common:
        a = old_ts[name].to(dtype=torch.float32)
        b = new_ts[name].to(dtype=torch.float32)
        if a.shape != b.shape:
            all_identical = False
            per_tensor.append({"name": name, "same_shape": False})
            continue
        diff = (a - b).abs()
        mx = float(diff.max().item())
        exact = bool(torch.equal(a, b))
        per_tensor.append({
            "name": name,
            "shape": list(a.shape),
            "max_abs_diff": mx,
            "mean_abs_diff": float(diff.mean().item()),
            "exact_equal": exact,
        })
        if not exact:
            all_identical = False
        max_abs_all = max(max_abs_all, mx)
    info["max_abs_diff_over_14_tensors"] = max_abs_all
    info["all_14_tensors_exact_equal"] = all_identical
    info["per_tensor"] = per_tensor
    info["num_common_tensors"] = len(common)
    return info


def compare_all_step_ckpts(old_dir: Path, new_dir: Path, steps: Sequence[int]) -> List[Dict[str, Any]]:
    results = []
    for s in steps:
        old_p = old_dir / "checkpoints" / f"step_{s:03d}_trainable.pt"
        new_p = new_dir / "checkpoints" / f"step_{s:03d}_trainable.pt"
        if not (old_p.exists() and new_p.exists()):
            results.append({"step": s, "old_exists": old_p.exists(), "new_exists": new_p.exists()})
            continue
        row = compare_ckpt_pair(old_p, new_p)
        row["step"] = s
        # keep per_tensor detail summarised
        row["per_tensor_summary"] = {
            "num": len(row["per_tensor"]),
            "num_exact_equal": sum(1 for r in row["per_tensor"] if r.get("exact_equal")),
            "max_abs_diff": row["max_abs_diff_over_14_tensors"],
        }
        results.append(row)
    return results


def compare_all_step_pngs(old_dir: Path, new_dir: Path, steps: Sequence[int]) -> List[Dict[str, Any]]:
    results = []
    # step_ref images
    for name in ("source.png", "golden_target.png"):
        op = old_dir / "visualizations" / "step_ref" / name
        np_ = new_dir / "visualizations" / "step_ref" / name
        results.append({
            "kind": "ref",
            "file": name,
            "old_exists": op.exists(),
            "new_exists": np_.exists(),
            "old_sha256": sha256_file(op) if op.exists() else None,
            "new_sha256": sha256_file(np_) if np_.exists() else None,
            "old_size": op.stat().st_size if op.exists() else None,
            "new_size": np_.stat().st_size if np_.exists() else None,
        })
    for s in steps:
        op = old_dir / "visualizations" / f"step_{s:03d}" / "model_output.png"
        np_ = new_dir / "visualizations" / f"step_{s:03d}" / "model_output.png"
        results.append({
            "kind": "step",
            "step": s,
            "old_exists": op.exists(),
            "new_exists": np_.exists(),
            "old_sha256": sha256_file(op) if op.exists() else None,
            "new_sha256": sha256_file(np_) if np_.exists() else None,
            "old_size": op.stat().st_size if op.exists() else None,
            "new_size": np_.stat().st_size if np_.exists() else None,
        })
    for r in results:
        if r["old_sha256"] is not None and r["new_sha256"] is not None:
            r["sha256_match"] = r["old_sha256"] == r["new_sha256"]
        else:
            r["sha256_match"] = None
    return results


def build_two_row_contact_sheet(
    lr1e4_dir: Path,
    lr3e5_dir: Path,
    save_path: Path,
    steps: Sequence[int] = LR_COMPARE_STEPS,
    tile_size: int = 768,
    label_h: int = 46,
    row_label_w: int = 200,
    pad: int = 10,
) -> Path:
    """Build a 2-row contact sheet.

    Row 1: lr=1e-4     : Source | Step 000 | Step 025 | ... | Step 500 | Golden
    Row 2: lr=3e-5     : Source | Step 000 | Step 025 | ... | Step 500 | Golden
    """
    def row_specs(base: Path, row_label: str) -> List[Tuple[str, Path]]:
        specs: List[Tuple[str, Path]] = [
            ("Source", base / "visualizations" / "step_ref" / "source.png"),
        ]
        for s in steps:
            specs.append((f"Step {s:03d}", base / "visualizations" / f"step_{s:03d}" / "model_output.png"))
        specs.append(("Golden", base / "visualizations" / "step_ref" / "golden_target.png"))
        return specs

    def load_row(base: Path) -> List[Tuple[str, Image.Image]]:
        loaded: List[Tuple[str, Image.Image]] = []
        for label, p in row_specs(base, ""):
            if not p.exists():
                # placeholder image to keep column alignment
                im = Image.new("RGB", (tile_size, tile_size), (240, 200, 200))
                d = ImageDraw.Draw(im)
                d.text((30, tile_size // 2 - 10), f"missing: {label}", fill=(120, 0, 0))
                loaded.append((label, im))
                continue
            im = Image.open(p).convert("RGB")
            scale = tile_size / im.height
            new_w = int(round(im.width * scale))
            im = im.resize((new_w, tile_size), Image.BICUBIC)
            loaded.append((label, im))
        return loaded

    row1 = load_row(lr1e4_dir)
    row2 = load_row(lr3e5_dir)

    n_cols = len(row1)
    # Column widths: pick per-column max between two rows for aligned columns.
    col_widths = [max(row1[i][1].width, row2[i][1].width) for i in range(n_cols)]
    total_w = row_label_w + sum(col_widths) + pad * (n_cols + 2)
    total_h = label_h + 2 * (tile_size + pad) + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_row = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
        font_row = ImageFont.load_default()

    # Column labels (top row of the sheet)
    x = row_label_w + pad
    for i, cw in enumerate(col_widths):
        label = row1[i][0]
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(label) * 10
        draw.text((x + (cw - tw) // 2, pad), label, fill=(20, 20, 20), font=font)
        x += cw + pad

    # Row 1: lr=1e-4
    y = label_h + pad
    draw.text((pad, y + tile_size // 2 - 14), "lr=1e-4", fill=(20, 20, 20), font=font_row)
    x = row_label_w + pad
    for i, (label, im) in enumerate(row1):
        cw = col_widths[i]
        offset = (cw - im.width) // 2
        canvas.paste(im, (x + offset, y))
        x += cw + pad

    # Row 2: lr=3e-5
    y = label_h + pad + tile_size + pad
    draw.text((pad, y + tile_size // 2 - 14), "lr=3e-5", fill=(20, 20, 20), font=font_row)
    x = row_label_w + pad
    for i, (label, im) in enumerate(row2):
        cw = col_widths[i]
        offset = (cw - im.width) // 2
        canvas.paste(im, (x + offset, y))
        x += cw + pad

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path, format="PNG", optimize=False)
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path,
                        default=NEW_LR3E5_DIR / "final_report.json")
    parser.add_argument("--two-row-contact-out", type=Path,
                        default=NEW_LR3E5_DIR / "lr1e4_vs_lr3e5_contact_sheet.png")
    args = parser.parse_args()

    # Wait: rely on caller to only run this after training finishes.
    report_path_new = NEW_LR3E5_DIR / "single_pair_overfit_report.json"
    if not report_path_new.exists():
        raise SystemExit(f"NEW report not yet present: {report_path_new}. Training likely not finished.")

    final: Dict[str, Any] = {
        "old_dir": str(OLD_LR3E5_DIR),
        "new_dir": str(NEW_LR3E5_DIR),
        "lr1e4_dir": str(LR1E4_DIR),
        "new_save_steps": list(NEW_SAVE_STEPS),
        "old_common_steps": list(OLD_COMMON_STEPS),
    }

    # 1. train_step log comparison over the overlap (should be all 300)
    final["train_log_comparison"] = compare_train_logs(
        OLD_LR3E5_DIR / "logs" / "run.log",
        NEW_LR3E5_DIR / "logs" / "run.log",
    )

    # 2. tensor-by-tensor checkpoint comparison for common save steps
    final["checkpoint_comparison"] = compare_all_step_ckpts(
        OLD_LR3E5_DIR, NEW_LR3E5_DIR, OLD_COMMON_STEPS
    )

    # 3. PNG SHA256 comparison for common save steps + ref images
    final["png_comparison"] = compare_all_step_pngs(
        OLD_LR3E5_DIR, NEW_LR3E5_DIR, OLD_COMMON_STEPS
    )

    # 4. Build review.html + contact_sheet.png inside NEW dir by invoking
    # the existing script (which reads save_steps from the report json).
    print("[4] building review.html + contact_sheet.png in NEW dir...")
    r = subprocess.run(
        [sys.executable, "-u",
         "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/training/build_single_pair_overfit_review.py",
         "--output-dir", str(NEW_LR3E5_DIR)],
        capture_output=True, text=True, timeout=600,
    )
    final["build_review"] = {
        "returncode": r.returncode,
        "stdout": r.stdout,
        "stderr": r.stderr,
    }
    if r.returncode != 0:
        raise SystemExit(f"build_single_pair_overfit_review failed:\n{r.stderr}")

    # 5. Two-row contact sheet (lr=1e-4 vs lr=3e-5)
    print("[5] building two-row lr comparison contact sheet...")
    two_row = build_two_row_contact_sheet(LR1E4_DIR, NEW_LR3E5_DIR, args.two_row_contact_out)
    final["two_row_contact_sheet"] = str(two_row)

    # 6. Persist final report
    args.report_out.write_text(json.dumps(final, ensure_ascii=False, indent=2, default=str),
                               encoding="utf-8")
    print(f"[6] wrote final_report.json to {args.report_out}")

    # 7. Print a short human-readable summary at the end
    tlc = final["train_log_comparison"]
    print("=" * 80)
    print(f"train_step log overlap: {tlc['num_overlap_steps']}")
    print(f"  exact_match_rows    : {tlc['num_exact_match_rows']}")
    print(f"  mismatched_fields   : {tlc['num_mismatched_field_rows']}")
    print(f"  step1 mse old={tlc['step1_mse_old']} new={tlc['step1_mse_new']}")
    for row in final["checkpoint_comparison"]:
        s = row.get("step")
        print(f"ckpt step_{s:03d}: old_sha={row.get('old_sha256','?')[:16]}... new_sha={row.get('new_sha256','?')[:16]}...  all_14_exact_equal={row.get('all_14_tensors_exact_equal')}  max_abs_diff={row.get('max_abs_diff_over_14_tensors')}")
    for row in final["png_comparison"]:
        s = row.get("step") if row.get("kind") == "step" else row.get("file")
        print(f"png {row['kind']} {s}: old_sha={(row.get('old_sha256') or '?')[:16]}... new_sha={(row.get('new_sha256') or '?')[:16]}...  match={row.get('sha256_match')}")


if __name__ == "__main__":
    main()
