"""Generate a static review page + a full-resolution contact sheet for the
single-pair overfit diagnostic run.

Everything lives inside ``outputs/single_pair_overfit_diag_ball``. Paths in
the HTML are relative so the folder can be zipped and opened locally.

Nothing here re-runs inference or training; it only reads the existing PNGs
and ``single_pair_overfit_report.json``.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

DEFAULT_STEPS = (0, 25, 50, 100, 200, 300, 500)
ROLLING_WINDOW = 25


def resolve_steps(report: Dict[str, Any]) -> Tuple[int, ...]:
    """Prefer the ``save_steps`` list stored in the report so a run with a
    different cadence (e.g. lr=3e-5 uses 0/10/25/50/100/200/300) is handled
    without editing the script."""
    ss = report.get("save_steps")
    if isinstance(ss, (list, tuple)) and ss:
        return tuple(int(s) for s in ss)
    return DEFAULT_STEPS


def rolling_mean_at(mse: Sequence[float], step: int, window: int = ROLLING_WINDOW) -> Optional[float]:
    """Return the mean of ``mse[step-window+1 .. step]``. Steps are 1-indexed
    in the report (step 1 is index 0). ``step=0`` (no training yet) returns None."""
    if step <= 0:
        return None
    end = min(step, len(mse))
    start = max(0, end - window)
    window_vals = mse[start:end]
    if not window_vals:
        return None
    return sum(window_vals) / len(window_vals)


def load_report(report_path: Path) -> Dict[str, Any]:
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_review_html(
    output_dir: Path,
    report: Dict[str, Any],
    contact_sheet_rel: str,
) -> Path:
    loss_history = report.get("loss_history", [])
    mse_series = [row["mse"] for row in loss_history]
    generation_seed = report.get("args", {}).get("generation_seed", 1234)
    pair_id = report.get("pair_id", "")
    intervention = report.get("conditioning_prompt", {}).get("intervention", {})

    if isinstance(intervention, dict):
        atomic = intervention.get("atomic_change") or intervention.get("changes") or intervention
        intervention_repr = json.dumps(atomic, ensure_ascii=False)
    else:
        intervention_repr = str(intervention)

    source_rel = "visualizations/step_ref/source.png"
    golden_rel = "visualizations/step_ref/golden_target.png"

    steps = resolve_steps(report)
    columns: List[Tuple[str, str, str, Optional[float]]] = []
    columns.append(("Source", source_rel, "n/a", None))
    for step in steps:
        label = f"Step {step:03d}"
        rel = f"visualizations/step_{step:03d}/model_output.png"
        rmse = rolling_mean_at(mse_series, step, ROLLING_WINDOW)
        raw_mse = None
        if 1 <= step <= len(mse_series):
            raw_mse = mse_series[step - 1]
        columns.append((label, rel, "step", (rmse, raw_mse)))
    columns.append(("Golden Target", golden_rel, "golden", None))

    rows_html: List[str] = []
    # ---- Row 1: images ----
    img_cells = []
    for label, rel, kind, _ in columns:
        img_cells.append(
            f'<td class="cell {kind}"><div class="label">{html.escape(label)}</div>'
            f'<a href="{html.escape(rel)}" target="_blank"><img src="{html.escape(rel)}" alt="{html.escape(label)}"/></a></td>'
        )
    rows_html.append("<tr>" + "".join(img_cells) + "</tr>")

    # ---- Row 2: MSE info ----
    metric_cells = []
    for label, rel, kind, meta in columns:
        if kind == "step":
            rmse, raw_mse = meta
            rmse_str = "n/a" if rmse is None else f"{rmse:.4f}"
            raw_str = "n/a" if raw_mse is None else f"{raw_mse:.4f}"
            metric_cells.append(
                f'<td class="cell metrics"><div class="rolling">rolling-{ROLLING_WINDOW} MSE: <b>{rmse_str}</b></div>'
                f'<div class="raw">raw step MSE: {raw_str}</div>'
                f'<div class="seed">gen seed: {generation_seed}</div></td>'
            )
        elif kind == "golden":
            metric_cells.append('<td class="cell metrics golden"><div class="golden-note">ground truth</div></td>')
        else:
            metric_cells.append('<td class="cell metrics source"><div class="source-note">condition image</div></td>')
    rows_html.append("<tr>" + "".join(metric_cells) + "</tr>")

    # ---- Row 3: TS / PS / BA / LS / IQ / notes review inputs ----
    review_cells = []
    for label, rel, kind, _ in columns:
        if kind == "step":
            checkpoint_step = int(label.split()[-1])
            review_cells.append(
                f'<td class="cell review">'
                f'<div class="rlabel">TS</div><input type="text" name="ts_{checkpoint_step}" placeholder="0/1"/>'
                f'<div class="rlabel">PS</div><input type="text" name="ps_{checkpoint_step}" placeholder="0~1"/>'
                f'<div class="rlabel">BA</div><input type="text" name="ba_{checkpoint_step}" placeholder="0/1"/>'
                f'<div class="rlabel">LS</div><input type="text" name="ls_{checkpoint_step}" placeholder="0~1"/>'
                f'<div class="rlabel">IQ</div><input type="text" name="iq_{checkpoint_step}" placeholder="0~1"/>'
                f'<div class="rlabel">notes</div><textarea name="notes_{checkpoint_step}" rows="3"></textarea>'
                f'</td>'
            )
        else:
            review_cells.append('<td class="cell review empty"></td>')
    rows_html.append("<tr>" + "".join(review_cells) + "</tr>")

    contact_sheet_html = ""
    if contact_sheet_rel and (output_dir / contact_sheet_rel).exists():
        contact_sheet_html = (
            f'<h2>Full-resolution contact sheet</h2>'
            f'<p>Preview scaled to 100% width. Right-click / open in new tab for the raw 1024-per-tile image.</p>'
            f'<a href="{html.escape(contact_sheet_rel)}" target="_blank"><img class="contact" src="{html.escape(contact_sheet_rel)}" alt="contact sheet"/></a>'
        )

    intervention_text = html.escape(
        "ball_1.location left_region -> right_region"
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Single-pair Overfit Review - {html.escape(pair_id)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; margin: 24px; background:#fafafa; color:#222; }}
  h1 {{ font-size: 20px; margin-bottom: 4px; }}
  h2 {{ font-size: 16px; margin-top: 32px; }}
  .meta {{ background:#fff; border:1px solid #ddd; border-radius:6px; padding:12px 16px; margin-bottom:16px; font-size:13px; line-height:1.5; }}
  .meta code {{ background:#f0f0f0; padding:1px 4px; border-radius:3px; }}
  .intervention {{ font-size:14px; padding:8px 12px; background:#fff8e1; border-left:4px solid #ffb300; margin: 8px 0 16px 0; }}
  table.review {{ border-collapse: collapse; background:#fff; }}
  td.cell {{ border:1px solid #ddd; padding:8px; text-align:center; vertical-align:top; }}
  td.cell.source {{ background:#eef7ff; }}
  td.cell.golden {{ background:#eafaea; }}
  td.cell.step {{ background:#fff; }}
  .label {{ font-weight:600; margin-bottom:6px; font-size:13px; }}
  td.cell img {{ max-width: 320px; max-height: 320px; display:block; }}
  td.metrics {{ font-size:12px; line-height:1.5; }}
  td.metrics .rolling b {{ color:#c1272d; font-size:13px; }}
  td.metrics .raw {{ color:#888; }}
  td.metrics .seed {{ color:#888; font-family: ui-monospace, Menlo, Consolas, monospace; }}
  td.review .rlabel {{ margin-top:4px; font-size:11px; color:#666; text-align:left; }}
  td.review input, td.review textarea {{ width: 96%; box-sizing: border-box; font-size:12px; padding:2px 4px; }}
  td.review.empty {{ background:#f6f6f6; }}
  img.contact {{ max-width:100%; height:auto; display:block; margin-top:12px; border:1px solid #ccc; border-radius:4px; }}
  .fixed-seed {{ background:#eaf5ff; padding:6px 10px; border-radius:4px; font-family: ui-monospace, Menlo, Consolas, monospace; display:inline-block; font-size:12px; }}
</style>
</head>
<body>
<h1>Single-pair Overfit Review</h1>
<div class="meta">
  <div><b>pair_id</b> <code>{html.escape(pair_id)}</code></div>
  <div><b>manifest</b> <code>{html.escape(report.get("manifest", ""))}</code></div>
  <div><b>generation_seed</b> <span class="fixed-seed">{generation_seed} (identical across all step images)</span></div>
  <div><b>lr</b> <code>{report.get("args", {}).get("lr")}</code> &nbsp; <b>steps</b> <code>{report.get("steps")}</code> &nbsp; <b>save_steps</b> <code>{report.get("save_steps")}</code></div>
  <div><b>train_image_size</b> <code>{report.get("args", {}).get("image_size")}</code> &nbsp; <b>visual_image_size</b> <code>{report.get("args", {}).get("visual_image_size")}</code></div>
</div>

<div class="intervention"><b>Intervention:</b> {intervention_text} &nbsp;&nbsp; <span style="color:#666;">(atomic_change JSON: <code>{html.escape(intervention_repr)}</code>)</span></div>

<h2>Row 1 - images, Row 2 - MSE metrics, Row 3 - human review (TS / PS / BA / LS / IQ / notes)</h2>
<table class="review">
{chr(10).join(rows_html)}
</table>

<p style="font-size:12px; color:#666; margin-top:10px;">
  <b>TS</b>: target satisfied - is the ball in <code>right_region</code>? (0/1)
  &nbsp;|&nbsp; <b>PS</b>: preservation - sofa, background, floor, lighting kept? (0~1)
  &nbsp;|&nbsp; <b>BA</b>: change applied to <code>ball_1</code>? (0/1)
  &nbsp;|&nbsp; <b>LS</b>: change locally scoped to ball's spatial location? (0~1)
  &nbsp;|&nbsp; <b>IQ</b>: overall image quality (0~1)
</p>

{contact_sheet_html}

</body>
</html>
"""
    review_path = output_dir / "review.html"
    review_path.write_text(html_doc, encoding="utf-8")
    return review_path


def build_contact_sheet(
    output_dir: Path,
    steps: Sequence[int],
    tile_size: int = 1024,
    label_h: int = 44,
    pad: int = 10,
) -> Optional[Path]:
    """Assemble Source | step_XXX * N | Golden as a horizontal strip at
    original resolution (tiles resized to a common height for alignment)."""
    order: List[Tuple[str, str]] = [
        ("Source", "visualizations/step_ref/source.png"),
    ]
    for step in steps:
        order.append((f"Step {step:03d}", f"visualizations/step_{step:03d}/model_output.png"))
    order.append(("Golden", "visualizations/step_ref/golden_target.png"))
    imgs: List[Tuple[str, Image.Image]] = []
    for label, rel in order:
        p = output_dir / rel
        if not p.exists():
            print(f"[warn] missing image: {p}")
            continue
        im = Image.open(p).convert("RGB")
        # Scale to have the same height as tile_size while preserving aspect ratio.
        scale = tile_size / im.height
        new_w = int(round(im.width * scale))
        im = im.resize((new_w, tile_size), Image.BICUBIC)
        imgs.append((label, im))
    if not imgs:
        return None

    widths = [im.width for _, im in imgs]
    heights = [im.height for _, im in imgs]
    total_w = sum(widths) + pad * (len(imgs) + 1)
    total_h = max(heights) + label_h + pad * 2

    canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    x = pad
    for label, im in imgs:
        # tile
        canvas.paste(im, (x, label_h + pad))
        # label centered above tile
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(label) * 10
        draw.text((x + (im.width - tw) // 2, pad), label, fill=(20, 20, 20), font=font)
        x += im.width + pad

    contact_path = output_dir / "contact_sheet.png"
    canvas.save(contact_path, format="PNG", optimize=False)
    return contact_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/single_pair_overfit_diag_ball",
        type=Path,
    )
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    report_path = output_dir / "single_pair_overfit_report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    report = load_report(report_path)
    steps = resolve_steps(report)
    contact = build_contact_sheet(output_dir, steps)
    contact_rel = contact.name if contact else ""
    review_path = build_review_html(output_dir, report, contact_rel)
    print(f"steps: {steps}")
    print(f"contact_sheet: {contact}")
    print(f"review_html:  {review_path}")


if __name__ == "__main__":
    main()
