"""data_v2 pipeline selftest — mock images, mock API, no network.

Runs the whole pipeline against synthetic fixtures:
  1. gen_intervention  : type mix, single-atomicity, must_preserve, move bboxes
  2. composite_target  : edit mode (background pixel-identity outside mask)
  3. composite_target  : move mode (relative_position)
  4. NanoBanana client : retry + seed-change logic via injected transport
  5. audit_v2          : gates pass on clean output, fail on tampered target
  6. run_pilot.sh      : end-to-end mock pilot, idempotent re-run

Usage:  python3 selftest_data_v2.py [--keep]
Exit code 0 iff all tests pass.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audit_v2  # noqa: E402
import composite_target as ct  # noqa: E402
import gen_intervention as gi  # noqa: E402
from common import load_jsonl, scene_atom_paths  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

IMG = 256


def _gradient_bg(h: int, w: int, tint: int) -> np.ndarray:
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    base = (yy * 160 + 40)
    img = np.stack([np.clip(base + tint, 0, 255)] * 3, axis=-1)
    return np.broadcast_to(img, (h, w, 3)).astype(np.uint8).copy()


def _draw_rect(img: np.ndarray, bbox, color) -> None:
    x1, y1, x2, y2 = bbox
    img[y1:y2, x1:x2] = color


def _draw_disk(img: np.ndarray, bbox, color) -> None:
    x1, y1, x2, y2 = bbox
    h, w = y2 - y1, x2 - x1
    yy, xx = np.ogrid[:h, :w]
    disk = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= (min(h, w) / 2) ** 2
    img[y1:y2, x1:x2][disk] = color


def make_source_image(path: Path, scene_idx: int) -> None:
    rng = np.random.default_rng(1000 + scene_idx)
    img = _gradient_bg(IMG, IMG, tint=int(scene_idx * 7))
    img = (img.astype(np.int16) + rng.integers(-3, 4, img.shape)).clip(0, 255).astype(np.uint8)
    # two objects per scene, deterministic bboxes
    _draw_rect(img, (60, 100, 120, 180), (178, 34, 34))       # "cup"
    _draw_disk(img, (150, 80, 215, 145), (30, 60, 200))       # "ball"
    Image.fromarray(img).save(path)


def make_source_manifest(root: Path, n_scenes: int = 8) -> Path:
    img_dir = root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for i in range(n_scenes):
        rel = f"scene_{i:03d}.png"
        make_source_image(img_dir / rel, i)
        records.append({
            "scene_id": f"s{i:03d}",
            "prompt": f"A photo of a red cup and a blue ball, scene {i}.",
            "image": rel,
            "cot": {
                "schema_version": "1.0",
                "entities": [
                    {"id": "cup", "category": "cup", "count": 1,
                     "attributes": {"color": "red", "material": "ceramic"},
                     "location": "center"},
                    {"id": "ball", "category": "ball", "count": 1,
                     "attributes": {"color": "blue", "size": "small"},
                     "location": "upper-right"},
                ],
                "background": {"description": "plain studio backdrop"},
            },
            "bboxes": {"cup": [60, 100, 120, 180], "ball": [150, 80, 215, 145]},
        })
    manifest = root / "source_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return manifest


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------

RESULTS: List[tuple] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def run_all(tmp: Path) -> bool:
    manifest = make_source_manifest(tmp)
    records = load_jsonl(manifest)
    image_root = tmp / "images"

    # ---- 1. gen_intervention ------------------------------------------------
    plans = gi.sample_plans(records, n_plans=40, seed=7, max_per_scene_subtype=3)
    dist = Counter(p["subtype"] for p in plans)
    check("gen: produces plans", len(plans) == 40, f"n={len(plans)} dist={dict(dist)}")
    check("gen: all five subtypes present", len(dist) == 5, f"{sorted(dist)}")
    check("gen: attribute weight (~30%)", 0.15 <= dist["attribute_intervention"] / len(plans) <= 0.45,
          f"{dist['attribute_intervention']}/40")
    check("gen: relative_position weight (~30%)",
          0.15 <= dist["relative_position"] / len(plans) <= 0.45,
          f"{dist['relative_position']}/40")
    check("gen: category down-weighted (~20%)",
          dist["category_replace"] / len(plans) <= 0.35, f"{dist['category_replace']}/40")
    check("gen: counting down-weighted (~10%)",
          dist["count_change"] / len(plans) <= 0.2, f"{dist['count_change']}/40")

    atom_ok, presv_ok, move_bbox_ok, move_relation_ok = True, True, True, True

    def atom_get(scene: Dict[str, Any], path: str) -> Any:
        eid, _, rest = path.partition(".")
        entity = next((e for e in scene.get("entities", []) if e["id"] == eid), None)
        if entity is None:
            return None
        if not rest:
            return entity
        cur: Any = entity
        for part in rest.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
        return cur

    for p in plans:
        src_scene, tgt_scene = p["source"]["cot"], p["target"]["cot"]
        all_paths = set(scene_atom_paths(src_scene)) | set(scene_atom_paths(tgt_scene))
        differing = [path for path in all_paths
                     if atom_get(src_scene, path) != atom_get(tgt_scene, path)]
        if len(differing) != 1:
            atom_ok = False
        allowed = p["preservation"]["allowed_to_change"]
        if len(allowed) != 1 or allowed[0] != f"{p['intervention']['entity_id']}.{p['intervention']['field']}":
            presv_ok = False
        if not set(allowed[0].split(".")[:1]) <= {p["intervention"]["entity_id"]}:
            presv_ok = False
        if p["subtype"] == "relative_position":
            iv = p["intervention"]
            if set(p["objects"]) != {iv["entity_id"], iv["new_value"][0]["target"]}:
                move_bbox_ok = False
            if iv["new_value"][0]["type"] not in ("left_of", "on_top_of"):
                move_relation_ok = False
            if p["compositing"]["mode"] != "move":
                move_relation_ok = False
        elif p["compositing"]["mode"] != "edit":
            move_relation_ok = False
    check("gen: single atomic change per plan", atom_ok)
    check("gen: must_preserve excludes exactly the allowed atom", presv_ok)
    check("gen: relative_position carries both object bboxes", move_bbox_ok)
    check("gen: compositing mode matches subtype", move_relation_ok)

    plans_path = tmp / "plans.jsonl"
    with plans_path.open("w", encoding="utf-8") as f:
        for p in plans:
            f.write(json.dumps(p) + "\n")

    # ---- 2/3. composite (mock provider, mock matting) ------------------------
    out_dir = tmp / "composite_out"
    summary = ct.run(plans_path, image_root, out_dir,
                     provider="mock", matting_name="mock", seed=7)
    check("composite: all pairs ok", summary["ok"] + summary["skipped"] == len(plans)
          and summary["failed"] == 0, json.dumps({k: summary[k] for k in ("ok", "skipped", "failed")}))

    bg_identical = True
    changed_out_zero = True
    for row in load_jsonl(out_dir / "manifest.jsonl"):
        src = np.asarray(Image.open(image_root / row["source"]["image"]).convert("RGB"))
        tgt = np.asarray(Image.open(out_dir / row["target"]["image"]).convert("RGB"))
        msk = np.asarray(Image.open(out_dir / row["target"]["mask"]).convert("L")) > 127
        if src.shape != tgt.shape:
            bg_identical = False
            continue
        keep = ~msk
        if not np.array_equal(src[keep], tgt[keep]):
            changed_out_zero = False
    check("composite: background pixels identical outside mask (all pairs)", changed_out_zero)

    audit_report = audit_v2.run_audit(out_dir / "manifest.jsonl", image_root,
                                      tmp / "audit_ok.json", sample_iou=10, seed=7)
    check("audit: clean pipeline passes all gates", audit_report["all_pass"],
          json.dumps(audit_report["summary"]))
    check("audit: bg_hist_l1 == 0 by construction",
          audit_report["summary"]["bg_hist_l1"]["max"] == 0.0)
    check("audit: changed_out_frac == 0 everywhere",
          audit_report["summary"]["changed_out_frac"]["n_nonzero"] == 0)
    check("audit: IoU spot-check produced values",
          (audit_report["summary"]["bbox_iou_sampled"]["n"] or 0) > 0)

    # ---- 4. nanobanana client retry/seed logic (injected transport) ----------
    calls: List[Dict[str, Any]] = []

    def flaky_transport(payload: Dict[str, Any]) -> Dict[str, Any]:
        calls.append({"seed": payload.get("seed")})
        if len(calls) < 3:  # two transient failures, then success
            return {"choices": []}  # triggers retry
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (10, 200, 30)).save(buf, format="PNG")
        import base64
        return {"choices": [{"message": {"images": [
            "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()]}}]}

    client = ct.NanoBananaEditClient(transport=flaky_transport, max_retries=3,
                                     retry_backoff_s=0.0)
    crop = Image.new("RGB", (64, 64), (5, 5, 5))
    edited = client.edit_crop(crop, "make it green", seed=99)
    check("api: flaky transport retried and succeeded",
          len(calls) == 3 and edited.size == (64, 64), f"calls={len(calls)}")

    calls.clear()

    def always_fail(payload: Dict[str, Any]) -> Dict[str, Any]:
        calls.append({"seed": payload.get("seed")})
        return {"choices": []}

    client2 = ct.NanoBananaEditClient(transport=always_fail, max_retries=2,
                                      retry_backoff_s=0.0)
    raised = False
    try:
        client2.edit_crop(crop, "x", seed=5)
    except RuntimeError:
        raised = True
    seeds = [c["seed"] for c in calls]
    check("api: hard failure retries then changes seed", raised and 5 in seeds and 10005 in seeds,
          f"seeds={seeds}")
    check("api: key guard", _key_guard(), "no NANOBANANA_KEY -> ValueError")

    # ---- 5. audit catches tampering ------------------------------------------
    tampered_dir = tmp / "tampered"
    shutil.copytree(out_dir, tampered_dir)
    mani = load_jsonl(tampered_dir / "manifest.jsonl")
    row = mani[0]
    tgt = np.asarray(Image.open(tampered_dir / row["target"]["image"]).convert("RGB")).copy()
    tgt[0:40, 0:40] = 255 - tgt[0:40, 0:40]  # stylize a background corner
    Image.fromarray(tgt).save(tampered_dir / row["target"]["image"])
    with (tampered_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for r in mani:
            f.write(json.dumps(r) + "\n")
    tampered_report = audit_v2.run_audit(tampered_dir / "manifest.jsonl", image_root,
                                         tmp / "audit_tampered.json", sample_iou=0)
    check("audit: tampered background fails the gate", not tampered_report["all_pass"],
          f"fail={tampered_report['summary']['n_fail_gate']}")

    # ---- 5b. engine_addit (mock fallback + distinct signature) ---------------
    import engine_addit as ea
    addit_client = ea.build_addit_client()
    check("addit: missing weights -> mock engine", isinstance(addit_client, ea.MockAdditEditClient))
    check("addit: real client raises cleanly when forced without weights",
          _addit_real_guard())
    crop_img = Image.new("RGB", (64, 64), (100, 100, 100))
    out_nb = ct.build_edit_client("mock").edit_crop(crop_img, "p", seed=1)
    out_ad = addit_client.edit_crop(crop_img, "p", seed=1)
    diff_engines = not np.array_equal(np.asarray(out_nb), np.asarray(out_ad))
    check("addit: engine outputs distinguishable from nanobanana mock", diff_engines)
    check("addit: deterministic across same seed",
          np.array_equal(np.asarray(addit_client.edit_crop(crop_img, "p", seed=1)),
                         np.asarray(out_ad)))
    check("addit: routed through composite factory",
          type(ct.build_edit_client("addit")).__name__ == "MockAdditEditClient")

    # ---- 5c. verify_checklist (mock VLM: pass + regen) ------------------------
    import verify_checklist as vc
    pair_plans = load_jsonl(out_dir / "manifest.jsonl")
    by_subtype: Dict[str, Any] = {}
    for p in pair_plans:
        by_subtype.setdefault(p["subtype"], p)
    checklist_ok = all(len(vc.build_checklist(p)) >= 2 for p in pair_plans)
    check("verify: every subtype yields change+preserve items", checklist_ok,
          f"subtypes={sorted(by_subtype)}")
    vrep = vc.run_verify(out_dir / "manifest.jsonl", image_root,
                         tmp / "verify_ok.json", vlm=vc.MockVlm())
    check("verify: clean pipeline all pass",
          vrep["summary"]["n_pass"] == vrep["summary"]["n_pairs"]
          and vrep["summary"]["n_regen"] == 0,
          json.dumps(vrep["summary"]))

    # fault injection -> regen flagged (fail every change* item of one pair)
    fault_pair = pair_plans[0]
    fault_ids = {i["id"] for i in vc.build_checklist(fault_pair)
                 if i["kind"].startswith("change")}
    faulty = vc.MockVlm(faults={fid: "wrong" for fid in fault_ids})
    frow = vc.verify_pair(fault_pair, image_root, out_dir, faulty, threshold=0.99)
    check("verify: fault injection flags regen",
          frow["status"] == "regen" and frow["n_pass"] < frow["n_items"],
          f"score={frow.get('score')} items={frow.get('n_items')}")

    # ---- 5d. integrity_guard (clean pass + overflow + erosion) -----------------
    import integrity_guard as ig
    irep = ig.run_guard(out_dir / "manifest.jsonl", image_root,
                        tmp / "integrity_ok.json", matting_name="mock")
    check("integrity: clean pipeline all ok",
          irep["summary"]["n_ok"] == irep["summary"]["n_pairs"]
          and irep["summary"]["n_needs_fix"] == 0,
          json.dumps(irep["summary"]))

    # overflow: tamper background outside mask
    orep = ig.run_guard(tampered_dir / "manifest.jsonl", image_root,
                        tmp / "integrity_overflow.json", matting_name="mock")
    check("integrity: blend overflow detected (needs_fix)",
          orep["summary"]["n_needs_fix"] >= 1
          and orep["summary"]["overflow_pairs"] >= 1,
          json.dumps(orep["summary"]))

    # erosion: stub matte that finds nothing in the object region
    class EmptyMatte:
        def extract_alpha(self, crop: Image.Image) -> np.ndarray:
            return np.zeros((crop.height, crop.width), dtype=np.uint8)
    erow = ig.check_pair(pair_plans[0], image_root, out_dir, EmptyMatte())
    check("integrity: foreground erosion detected (low IoU -> needs_fix)",
          erow["status"] == "needs_fix"
          and any("eroded" in r for r in erow.get("reasons", [])),
          f"iou={erow.get('fg_iou')} reasons={erow.get('reasons')}")

    # ---- 6. run_pilot.sh end-to-end (dual engine) + idempotent re-run ---------
    pilot_dir = tmp / "pilot"
    env = dict(os.environ)
    env.update({
        "SOURCE_MANIFEST": str(manifest),
        "IMAGE_ROOT": str(image_root),
        "PILOT_DIR": str(pilot_dir),
        "LOG_DIR": str(pilot_dir / "logs"),
    })
    r1 = subprocess.run(["bash", str(HERE / "run_pilot.sh"), "--n", "16", "--seed", "7"],
                        env=env, capture_output=True, text=True, timeout=900)
    check("pilot: first run exits 0 (all gates pass, both engines)", r1.returncode == 0,
          (r1.stderr or r1.stdout).strip()[-200:])
    ledger1 = (pilot_dir / "eng_nanobanana" / "ledger.jsonl").read_text()
    n_ok1 = sum(1 for line in ledger1.splitlines() if '"status": "ok"' in line)
    check("pilot: addit engine produced its own target set",
          (pilot_dir / "eng_addit" / "ledger.jsonl").exists()
          and len(load_jsonl(pilot_dir / "eng_addit" / "manifest.jsonl")) > 0)

    r2 = subprocess.run(["bash", str(HERE / "run_pilot.sh"), "--n", "16", "--seed", "7"],
                        env=env, capture_output=True, text=True, timeout=900)
    ledger2 = (pilot_dir / "eng_nanobanana" / "ledger.jsonl").read_text()
    n_ok2 = sum(1 for line in ledger2.splitlines() if '"status": "ok"' in line)
    check("pilot: re-run is idempotent (no new ok rows, exit 0)",
          r2.returncode == 0 and n_ok1 == n_ok2, f"ok1={n_ok1} ok2={n_ok2}")

    # merged pilot report with cross-engine stats
    preport = json.loads((pilot_dir / "pilot_report.json").read_text(encoding="utf-8"))
    check("pilot: pilot_report.json has both engines + audit pass",
          set(preport["per_engine"]) == {"nanobanana", "addit"}
          and preport["all_engines_audit_pass"]
          and "composite" in preport["per_engine"]["addit"],
          f"engines={sorted(preport['per_engine'])}")

    # nanobanana mode refuses without key
    env_nb = dict(env)
    env_nb.pop("NANOBANANA_KEY", None)
    env_nb["DATA_V2_PROVIDER"] = "nanobanana"
    r3 = subprocess.run(["bash", str(HERE / "run_pilot.sh"), "--n", "2"],
                        env=env_nb, capture_output=True, text=True, timeout=120)
    check("pilot: real-API mode refuses without NANOBANANA_KEY", r3.returncode != 0,
          r3.stderr.strip()[-120:])

    return all(ok for _, ok, _ in RESULTS)


def _addit_real_guard() -> bool:
    import engine_addit as ea
    saved_w = os.environ.pop("ADDIT_WEIGHTS_DIR", None)
    try:
        try:
            ea.build_addit_client(force_real=True)
            return False
        except RuntimeError:
            return True
    finally:
        if saved_w is not None:
            os.environ["ADDIT_WEIGHTS_DIR"] = saved_w


def _key_guard() -> bool:
    saved = os.environ.pop("NANOBANANA_KEY", None)
    try:
        try:
            ct.NanoBananaEditClient()
            return False
        except ValueError:
            return True
    finally:
        if saved is not None:
            os.environ["NANOBANANA_KEY"] = saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the tmp fixture dir")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="data_v2_selftest_"))
    print(f"fixtures: {tmp}")
    try:
        ok = run_all(tmp)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        ok = False
    n_pass = sum(1 for _, o, _ in RESULTS if o)
    print(f"\n{len(RESULTS)} checks: {n_pass} PASS, {len(RESULTS) - n_pass} FAIL")
    if not args.keep:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
