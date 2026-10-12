"""Generate human annotation blind package for Judge v1 calibration (v2 plan).

Produces (all under human_annotation_package/):
  review.html                     — static blind review page, 40 items,
                                    source/target/generated three-image layout,
                                    intervention + must_preserve visible,
                                    NO condition/step/LoRA/directory exposed
  annotator_1/annotation_form.json — empty scoring template for annotator 1
  annotator_2/annotation_form.json — empty scoring template for annotator 2
  package_manifest.json           — paths, SHA256, creation metadata

Blinding guarantees:
  - item_id only (item_000 … item_039), no pair_id, no condition
  - Generated image filenames renamed to generated_image.png in HTML
  - Source/target filenames renamed to source_image.png / target_image.png
  - No reference to A/B, step000/step400, LoRA, training, or output dirs
  - candidate_only mask is NOT shown and NOT referenced

No Judge invoked. No scores produced. No frozen asset modified.
Custody map is NOT embedded in any output file.
"""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/group/40016/niightli/BAGEL").resolve()
EVAL_DIR = REPO_ROOT / "experiments/counterfactual_cot/evaluation"
OUT_ROOT = EVAL_DIR / "human_annotation_package"

INFER_ROOT = REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_ab_aligned_inference_nt25_protocol_v1"
CUSTODY_MAP_PATH = EVAL_DIR / "judge_v1_calibration_plan_v2/judge_v1_custody_map.json"
CUSTODY_MAP_SHA_EXPECT = "8db9b9c592789b01d10e848e44ce4ff096625aba61d318e23db957af787714bc"
V2_PLAN_SHA_EXPECT = "b6c8068c54a804a38e09fa783ab4e59bf1129a9c8305c271633dc315f3dda524"
V2_PLAN_PATH = EVAL_DIR / "judge_v1_calibration_plan_v2/judge_v1_calibration_plan_v2.json"

V3_MANIFEST = (
    REPO_ROOT / "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/manifests/"
    "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl"
)

# Condition → verifier png directory
CONDITION_PNG_DIRS: Dict[str, Path] = {
    "A_step000": INFER_ROOT / "verifier_A/png",
    "A_step400": INFER_ROOT / "verifier_A_step400/png",
    "B_step000": INFER_ROOT / "verifier_B/png",
    "B_step400": INFER_ROOT / "verifier_B_step400/png",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _b64_img(p: Path) -> str:
    """Return base64-encoded PNG for embedding in HTML."""
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _load_manifest() -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with open(V3_MANIFEST, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r["pair_id"]] = r
    return records


def _resolve_generated_png(pair_id: str, condition: str) -> Optional[Path]:
    """Find the generated PNG for (pair_id, condition) in the verifier png dir."""
    png_dir = CONDITION_PNG_DIRS.get(condition)
    if not png_dir or not png_dir.exists():
        return None
    side = "A" if condition.startswith("A") else "B"
    # Filename pattern: NNN__<pair_id>__<side>.png
    for p in png_dir.iterdir():
        if p.suffix == ".png" and f"__{pair_id}__{side}" in p.name:
            return p
    return None


def _resolve_source_target(pair: Dict[str, Any]) -> tuple[Optional[Path], Optional[Path]]:
    root = pair.get("final_image_root", "")
    src_rel = (pair.get("source") or {}).get("image", "")
    tgt_rel = (pair.get("target") or {}).get("image", "")
    src = Path(root) / src_rel if root and src_rel else None
    tgt = Path(root) / tgt_rel if root and tgt_rel else None
    return (src if src and src.exists() else None,
            tgt if tgt and tgt.exists() else None)


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Blind Image Editing Evaluation — Calibration Set</title>
<style>
  body { font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; padding: 20px; }
  h1 { color: #333; border-bottom: 2px solid #666; padding-bottom: 8px; }
  .notice { background: #fff3cd; border: 1px solid #ffc107; padding: 12px 16px;
            border-radius: 4px; margin-bottom: 24px; font-size: 14px; }
  .item { background: #fff; border: 1px solid #ddd; border-radius: 6px;
          margin-bottom: 32px; padding: 20px; }
  .item-header { font-size: 18px; font-weight: bold; color: #222;
                 border-bottom: 1px solid #eee; padding-bottom: 8px; margin-bottom: 16px; }
  .images { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
  .img-block { flex: 1; min-width: 280px; text-align: center; }
  .img-block img { max-width: 100%; max-height: 400px; border: 1px solid #ccc;
                   border-radius: 4px; display: block; margin: 0 auto 6px; }
  .img-label { font-size: 13px; font-weight: bold; color: #555; }
  .intervention { background: #f0f4ff; border-left: 4px solid #4a6cf7;
                  padding: 10px 14px; border-radius: 0 4px 4px 0; margin-bottom: 12px; }
  .intervention h3 { margin: 0 0 6px; font-size: 14px; color: #333; }
  .intervention p { margin: 2px 0; font-size: 13px; font-family: monospace; }
  .preserve { background: #f0fff4; border-left: 4px solid #28a745;
              padding: 10px 14px; border-radius: 0 4px 4px 0; }
  .preserve h3 { margin: 0 0 6px; font-size: 14px; color: #333; }
  .preserve ul { margin: 4px 0; padding-left: 20px; font-size: 13px;
                 font-family: monospace; }
  .mask-notice { font-size: 12px; color: #c00; margin-top: 8px; font-style: italic; }
  .scoring-ref { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 4px;
                 padding: 10px 14px; margin-top: 12px; font-size: 12px; color: #555; }
</style>
</head>
<body>
<h1>Blind Image Editing Evaluation — Calibration Set (40 items)</h1>
<div class="notice">
  <strong>Annotator instructions:</strong>
  You are evaluating generated images for image editing quality.
  Each item shows: <em>Source Image</em> (original), <em>Target Image</em> (reference edit),
  and <em>Generated Image</em> (to evaluate).<br>
  Score each item using the JSON template provided separately.<br>
  <strong>Do NOT discuss scores with the other annotator before submitting.</strong><br>
  <strong>Do NOT use any pixel-difference mask as LS ground truth.</strong><br>
  No model names, training details, or conditions are shown — this is intentional.
</div>
"""

_HTML_TAIL = "</body></html>\n"


def _item_html(item_id: str, item_index: int,
               src_path: Optional[Path], tgt_path: Optional[Path],
               gen_path: Optional[Path],
               intervention: Dict[str, Any],
               must_preserve: List[str]) -> str:
    def _img_tag(p: Optional[Path], label: str) -> str:
        if p and p.exists():
            b64 = _b64_img(p)
            return (f'<div class="img-block">'
                    f'<img src="data:image/png;base64,{b64}" alt="{label}">'
                    f'<div class="img-label">{label}</div></div>')
        return (f'<div class="img-block">'
                f'<div style="height:200px;background:#eee;display:flex;'
                f'align-items:center;justify-content:center;color:#999;">'
                f'[{label} — not found]</div>'
                f'<div class="img-label">{label}</div></div>')

    iv = intervention
    preserve_items = "".join(f"<li>{atom}</li>" for atom in must_preserve)

    return f"""
<div class="item" id="{item_id}">
  <div class="item-header">Item {item_index:03d} &nbsp;|&nbsp; <code>{item_id}</code></div>
  <div class="images">
    {_img_tag(src_path, "Source Image")}
    {_img_tag(tgt_path, "Target Image")}
    {_img_tag(gen_path, "Generated Image")}
  </div>
  <div class="intervention">
    <h3>Intervention</h3>
    <p>Entity: <strong>{iv.get('entity_id','?')}</strong></p>
    <p>Field: {iv.get('field','?')}</p>
    <p>Required change: <strong>{iv.get('old_value','?')}</strong> → <strong>{iv.get('new_value','?')}</strong></p>
  </div>
  <div class="preserve">
    <h3>Preservation requirements (must_preserve atoms)</h3>
    <ul>{preserve_items}</ul>
    <p class="mask-notice">⚠ Do NOT use any pixel-difference mask as LS ground truth.
    Assess locality from source/target/generated images and the intervention description only.</p>
  </div>
  <div class="scoring-ref">
    Score this item in your annotation_form.json under key <code>"{item_id}"</code>.
    Fields: TS (0/1), PS (0.0–1.0), BA (0.0–1.0), LS (0.0–1.0), IQ (0.0–1.0),
    rationale per metric, PS_atom_decisions, score_status.
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Annotation form template
# ---------------------------------------------------------------------------

def _empty_item_template(item_id: str, must_preserve: List[str]) -> Dict[str, Any]:
    return {
        "item_id": item_id,
        "TS": None,
        "TS_rationale": None,
        "PS": None,
        "PS_rationale": None,
        "PS_atom_decisions": {atom: None for atom in must_preserve},
        "BA": None,
        "BA_rationale": None,
        "LS": None,
        "LS_rationale": None,
        "IQ": None,
        "IQ_rationale": None,
        "score_status": "not_scored",
        "annotator_notes": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    failures: List[str] = []

    # 1. Verify upstream anchors
    custody_sha = sha256_file(CUSTODY_MAP_PATH)
    if custody_sha != CUSTODY_MAP_SHA_EXPECT:
        print(f"STOP: custody_map SHA mismatch: expected={CUSTODY_MAP_SHA_EXPECT} observed={custody_sha}")
        return 1
    v2_sha = sha256_file(V2_PLAN_PATH)
    if v2_sha != V2_PLAN_SHA_EXPECT:
        print(f"STOP: v2 plan SHA mismatch: expected={V2_PLAN_SHA_EXPECT} observed={v2_sha}")
        return 1

    # 2. Load custody map (for item ordering only — pair_id/condition NOT in HTML)
    custody_data = json.loads(CUSTODY_MAP_PATH.read_text(encoding="utf-8"))
    custody_map: List[Dict[str, Any]] = custody_data["custody_map"]

    # 3. Load manifest
    manifest = _load_manifest()

    # 4. Prepare output dirs
    if OUT_ROOT.exists():
        # Refuse to overwrite non-empty package
        existing = list(OUT_ROOT.iterdir())
        if existing:
            print(f"STOP: output dir already exists and is non-empty: {OUT_ROOT}")
            return 1
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "annotator_1").mkdir()
    (OUT_ROOT / "annotator_2").mkdir()

    # 5. Build per-item data (blind — no pair_id/condition in HTML)
    html_items: List[str] = []
    form_items: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    for item_index, entry in enumerate(custody_map):
        item_id = entry["item_id"]
        pair_id = entry["pair_id"]       # used only to look up paths — NOT in HTML
        condition = entry["condition"]   # used only to find generated PNG — NOT in HTML

        pair = manifest.get(pair_id, {})
        iv = pair.get("intervention", {})
        pres = pair.get("preservation", {})
        must_preserve = pres.get("must_preserve", [])

        src_path, tgt_path = _resolve_source_target(pair)
        gen_path = _resolve_generated_png(pair_id, condition)

        # Track missing files
        if src_path is None:
            failures.append(f"{item_id}: source_image_missing for pair {pair_id}")
        if tgt_path is None:
            failures.append(f"{item_id}: target_image_missing for pair {pair_id}")
        if gen_path is None:
            failures.append(f"{item_id}: generated_png_missing for pair {pair_id} condition {condition}")

        # HTML item (blind)
        html_items.append(_item_html(
            item_id=item_id,
            item_index=item_index,
            src_path=src_path,
            tgt_path=tgt_path,
            gen_path=gen_path,
            intervention={
                "entity_id": iv.get("entity_id"),
                "field": iv.get("field"),
                "old_value": iv.get("old_value"),
                "new_value": iv.get("new_value"),
            },
            must_preserve=must_preserve,
        ))

        # Form template item
        form_items.append(_empty_item_template(item_id, must_preserve))

        # Manifest row (blind — no pair_id/condition)
        src_sha = sha256_file(src_path) if src_path else None
        tgt_sha = sha256_file(tgt_path) if tgt_path else None
        gen_sha = sha256_file(gen_path) if gen_path else None
        manifest_rows.append({
            "item_id": item_id,
            "item_index": item_index,
            "source_image_sha256": src_sha,
            "target_image_sha256": tgt_sha,
            "generated_image_sha256": gen_sha,
            "source_image_exists": src_path is not None,
            "target_image_exists": tgt_path is not None,
            "generated_image_exists": gen_path is not None,
            "num_must_preserve_atoms": len(must_preserve),
        })

    if failures:
        print(f"STOP: {len(failures)} missing files:\n" + "\n".join(failures[:10]))
        return 1

    # 6. Write HTML review page
    html_path = OUT_ROOT / "review.html"
    html_content = _HTML_HEAD + "\n".join(html_items) + _HTML_TAIL
    html_path.write_text(html_content, encoding="utf-8")
    html_sha = sha256_file(html_path)

    # 7. Write annotation forms
    form_header = {
        "annotator_id": None,
        "created_at_iso": now_iso(),
        "plan_version": "v2",
        "instructions": (
            "Fill in scores for each item. "
            "TS: 0 or 1. PS/BA/LS/IQ: float 0.0-1.0. "
            "If unable to score a metric, set it to null and set score_status to 'not_scored'. "
            "Do NOT use 0 as a placeholder for an unscored field. "
            "Do NOT use pixel-difference masks as LS ground truth. "
            "Do NOT discuss scores with the other annotator before submitting."
        ),
        "null_placeholder_prohibition": "Do NOT fill 0 for an unscored field. Use null.",
        "mask_prohibition": "Do NOT use candidate_only pixel-difference masks as LS ground truth.",
        "blinding_note": "item_id is the only identifier. Do not attempt to identify the model or condition.",
        "items": form_items,
    }

    a1_path = OUT_ROOT / "annotator_1/annotation_form.json"
    a2_path = OUT_ROOT / "annotator_2/annotation_form.json"

    form_a1 = dict(form_header)
    form_a1["annotator_id"] = "annotator_1"
    a1_path.write_text(
        json.dumps(form_a1, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )

    form_a2 = dict(form_header)
    form_a2["annotator_id"] = "annotator_2"
    a2_path.write_text(
        json.dumps(form_a2, ensure_ascii=False, indent=2, sort_keys=False),
        encoding="utf-8",
    )

    a1_sha = sha256_file(a1_path)
    a2_sha = sha256_file(a2_path)

    # 8. Write package manifest (blind — no pair_id/condition)
    pkg_manifest = {
        "package_type": "human_annotation_blind_package",
        "plan_version": "v2",
        "created_at_iso": now_iso(),
        "num_items": len(custody_map),
        "upstream_anchors": {
            "custody_map_sha256_verified": custody_sha == CUSTODY_MAP_SHA_EXPECT,
            "v2_plan_sha256_verified": v2_sha == V2_PLAN_SHA_EXPECT,
        },
        "files": {
            "review_html": {"path": str(html_path), "sha256": html_sha,
                            "size_bytes": html_path.stat().st_size},
            "annotator_1_form": {"path": str(a1_path), "sha256": a1_sha,
                                 "size_bytes": a1_path.stat().st_size},
            "annotator_2_form": {"path": str(a2_path), "sha256": a2_sha,
                                 "size_bytes": a2_path.stat().st_size},
        },
        "blinding_guarantees": [
            "HTML contains only item_id (item_000–item_039), no pair_id or condition.",
            "Generated image filenames are not shown; images are embedded as base64.",
            "Source/target filenames are not shown; images are embedded as base64.",
            "No reference to A/B, step000/step400, LoRA, training, or output directories.",
            "candidate_only masks are NOT shown and NOT referenced.",
            "Custody map is NOT embedded in any output file.",
        ],
        "items": manifest_rows,
        "declarations": [
            "No Judge invoked.",
            "No TS/PS/BA/LS/IQ scores produced.",
            "No model loaded.",
            "No GPU forward executed.",
            "No frozen asset modified.",
            "Custody map is sealed separately and must not be shared with annotators.",
        ],
    }
    pkg_path = OUT_ROOT / "package_manifest.json"
    pkg_path.write_text(
        json.dumps(pkg_manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pkg_sha = sha256_file(pkg_path)

    # 9. Print summary
    summary = {
        "overall_status": "passed",
        "num_items": len(custody_map),
        "review_html": {
            "path": str(html_path),
            "size_bytes": html_path.stat().st_size,
            "sha256": html_sha,
        },
        "annotator_1_form": {
            "path": str(a1_path),
            "size_bytes": a1_path.stat().st_size,
            "sha256": a1_sha,
        },
        "annotator_2_form": {
            "path": str(a2_path),
            "size_bytes": a2_path.stat().st_size,
            "sha256": a2_sha,
        },
        "package_manifest": {
            "path": str(pkg_path),
            "size_bytes": pkg_path.stat().st_size,
            "sha256": pkg_sha,
        },
        "declarations": [
            "No Judge invoked.",
            "No scores produced.",
            "No frozen asset modified.",
            "Custody map NOT embedded in any output.",
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
