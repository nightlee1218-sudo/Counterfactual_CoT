#!/usr/bin/env python3
"""data_audit CPU selftest: synthetic 96x96 pairs, full CLI round trip.

Builds a miniature final991-automask-v5-style split (real record schema)
with deliberately defective pairs, runs style_audit.py, integrity_audit.py
and report.py as subprocesses, and asserts every flag fires exactly where
it should (clean pair -> no flags; churn -> bg_churn + concentration +
style flags; unchanged -> no_change_in_mask; empty/full mask; tampered sha;
missing mask file; unreferenced new_value; failed local check; wrong
nonzero count). Plus unit tests for the prompt-atomicity token rule
("inside:cup_1", "separate_from:shoebox_1", digits vs spelled-out counts,
"top_region" -> "upper", "unoccluded" -> "nothing overlaps") and a
pure-black-pixel regression (images with max-RGB==0 pixels must not crash
the saturation stats; 20261015 fix, 522/991 real pairs).

Usage: python3 selftest_data_audit.py   (exit 0 only if all pass)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASS, FAIL = 0, 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS {name}")
    else:
        FAIL += 1
        FAILURES.append(name)
        print(f"FAIL {name} {detail}")


def run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], capture_output=True,
                          text=True, cwd=str(HERE))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(tmp: Path) -> Path:
    import numpy as np
    from PIL import Image

    size = 96
    src = np.zeros((size, size, 3), dtype=np.uint8)
    src[:] = np.linspace(40, 200, size)[None, :, None]
    src[24:40, 24:40] = (200, 20, 20)
    square = np.zeros((size, size), dtype=bool)
    square[24:40, 24:40] = True
    everything = np.ones((size, size), dtype=bool)

    img_dir = tmp / "images"
    img_dir.mkdir(parents=True)
    splits_dir = tmp / "splits"
    splits_dir.mkdir()

    def save_img(arr, path):
        Image.fromarray(arr, "RGB").save(path)
        return sha256_file(path)

    def save_mask(mask_bool, path):
        Image.fromarray((mask_bool * 255).astype(np.uint8), "L").save(path)
        return sha256_file(path)

    source_path = img_dir / "source.png"
    src_sha = save_img(src, source_path)
    clean_tgt = src.copy()
    clean_tgt[24:40, 24:40] = (20, 200, 20)
    churn_tgt = np.clip(clean_tgt.astype(np.int16) + 25, 0, 255).astype(np.uint8)

    def base_case(pair_id, tgt_arr, mask_bool, new_value="green square"):
        tgt_path = img_dir / f"{pair_id}__target.png"
        tgt_sha = save_img(tgt_arr, tgt_path)
        mask_path = img_dir / f"{pair_id}__mask.png"
        mask_sha = save_mask(mask_bool, mask_path)
        nonzero = int(mask_bool.sum())
        return {
            "pair_id": pair_id,
            "intervention_type": "category_replacement",
            "intervention": {"field": "category", "operation": "replace",
                             "old_value": "red square", "new_value": new_value,
                             "entity_id": "square_1",
                             "intervention_type": "category_replacement",
                             "intervention_subtype": "plain_replace"},
            "source": {"image": str(source_path), "image_sha256": src_sha,
                       "prompt": "Create an image of a red square on a gray gradient."},
            "target": {"image": str(tgt_path), "image_sha256": tgt_sha,
                       "prompt": "Replace only the red square with a " + new_value
                                 + ". Keep the background pixel-consistent."},
            "automatic_difference_mask": {
                "path": str(mask_path), "sha256": mask_sha,
                "geometry": {"width": size, "height": size, "nonzero_pixels": nonzero},
                "source_candidate_mask": {"dilation_px": None,
                                          "input_field": "final991_training_mask",
                                          "threshold": 24}},
            "quality": {"accepted": False, "review_status": "auto_checked",
                        "local_checks": {"passed": True, "failures": [],
                                         "changed_pixel_ratio": 0.09,
                                         "mask_area_ratio": nonzero / size / size,
                                         "mean_absolute_difference": 5.0,
                                         "min_changed_ratio": 0.004,
                                         "source_size": [size, size],
                                         "target_size": [size, size]}},
        }

    cases = [base_case("clean_pair", clean_tgt, square),
             base_case("churn_pair", churn_tgt, square),
             base_case("unchanged_pair", src.copy(), square),
             base_case("empty_mask_pair", clean_tgt, ~everything),
             base_case("full_mask_pair", clean_tgt, everything)]
    cases.append(base_case("sha_tamper_pair", clean_tgt, square))
    cases[-1]["source"]["image_sha256"] = "0" * 64
    cases.append(base_case("missing_mask_pair", clean_tgt, square))
    (img_dir / "missing_mask_pair__mask.png").unlink()
    cases.append(base_case("atomicity_bad_pair", clean_tgt, square,
                           new_value="unicorn chariot"))
    cases[-1]["target"]["prompt"] = \
        "Replace only the red square with a green square. Keep the background pixel-consistent."
    cases.append(base_case("local_check_failed_pair", clean_tgt, square))
    cases[-1]["quality"]["local_checks"]["passed"] = False
    cases.append(base_case("nonzero_mismatch_pair", clean_tgt, square))
    cases[-1]["automatic_difference_mask"]["geometry"]["nonzero_pixels"] = 12345
    # pure-black-pixel regression (20261015 fix): saturation crashed with a
    # broadcast error on any image containing a max-RGB==0 pixel
    black_src = src.copy()
    black_src[0, 0] = (0, 0, 0)
    black_src[70:80, 70:80] = (0, 0, 0)
    black_path = img_dir / "black_source.png"
    black_sha = save_img(black_src, black_path)
    black_tgt = black_src.copy()
    black_tgt[24:40, 24:40] = (20, 200, 20)
    cases.append(base_case("black_pixel_pair", black_tgt, square))
    cases[-1]["source"]["image"] = str(black_path)
    cases[-1]["source"]["image_sha256"] = black_sha

    with (splits_dir / "val.jsonl").open("w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    return splits_dir


def test_atomicity_rule() -> None:
    spec = importlib.util.spec_from_file_location("ia_selftest",
                                                  HERE / "integrity_audit.py")
    ia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ia)
    check("atomicity_inside_cup",
          ia.new_value_ref_ratio("inside:cup_1",
                                 "... to inside the white ceramic rinse cup ...") >= 0.5)
    check("atomicity_separate_shoebox",
          ia.new_value_ref_ratio("separate_from:shoebox_1",
                                 "set them down beside the box, clearly outside it.") >= 0.5)
    check("atomicity_count_word",
          ia.new_value_ref_ratio(2, "shows exactly TWO notebooks, source shows three.") >= 0.5)
    check("atomicity_top_region",
          ia.new_value_ref_ratio("top_region",
                                 "move the remote to the upper area of the frame.") >= 0.5)
    check("atomicity_unoccluded",
          ia.new_value_ref_ratio("unoccluded",
                                 "TASK: separate the two objects so nothing overlaps.") >= 0.5)
    check("atomicity_unreferenced",
          ia.new_value_ref_ratio("unicorn chariot",
                                 "Replace only the red square with a green square.") < 0.5)


def run_audit_round(splits_dir: Path, out_dir: Path) -> dict:
    ps = run_cli(["style_audit.py", "--splits-dir", str(splits_dir),
                  "--splits", "val", "--out-dir", str(out_dir), "--workers", "2"])
    check("style_cli_exit0", ps.returncode == 0, ps.stderr[-400:])
    pi = run_cli(["integrity_audit.py", "--splits-dir", str(splits_dir),
                  "--splits", "val", "--out-dir", str(out_dir), "--workers", "2"])
    check("integrity_cli_exit0", pi.returncode == 0, pi.stderr[-400:])
    style = {r["pair_id"]: r for r in (json.loads(l) for l in
              (out_dir / "style_audit.jsonl").read_text().splitlines())}
    integ = {r["pair_id"]: r for r in (json.loads(l) for l in
              (out_dir / "integrity_audit.jsonl").read_text().splitlines())}
    return {"style": style, "integ": integ}


def test_style_rows(rows: dict) -> None:
    s = rows["style"]
    check("style_clean_bg_hist_zero", abs(s["clean_pair"]["bg_hist_l1"]) < 1e-9,
          f"got {s['clean_pair'].get('bg_hist_l1')}")
    check("style_clean_global_hist_positive", s["clean_pair"]["hist_l1"] > 0.02)
    check("style_churn_bg_hist_big", s["churn_pair"]["bg_hist_l1"] > 0.1)
    check("style_churn_brightness_shift", s["churn_pair"]["bg_val_delta"] > 0.03)
    u = s["unchanged_pair"]
    check("style_unchanged_zero_drift", u["hist_l1"] == 0.0
          and u["bg_hist_l1"] == 0.0 and u["sat_delta"] == 0.0
          and u["contrast_delta"] == 0.0)
    check("style_missing_mask_error_row", s["missing_mask_pair"]["error"] is not None)
    b = s["black_pixel_pair"]
    check("style_black_pixel_no_error", b["error"] is None, str(b.get("error")))
    check("style_black_pixel_finite_stats",
          isinstance(b["sat_mean_src"], float) and isinstance(b["bg_sat_delta"], float)
          and b["hist_l1"] >= 0.0 and b["bg_hist_l1"] >= 0.0
          and abs(b["sat_delta"]) < 0.1,
          f"sat_src={b.get('sat_mean_src')} bg_sat={b.get('bg_sat_delta')}")


def test_integrity_rows(rows: dict) -> None:
    i = rows["integ"]
    check("integ_clean_no_flags", i["clean_pair"]["flags"] == [],
          str(i["clean_pair"]["flags"]))
    check("integ_clean_metrics", i["clean_pair"]["changed_out_frac"] == 0.0
          and i["clean_pair"]["changed_in_frac"] == 1.0
          and i["clean_pair"]["precision_in"] == 1.0
          and i["clean_pair"]["iou_change_mask"] == 1.0)
    check("integ_churn_flags",
          {"bg_churn", "low_change_concentration"} <= set(i["churn_pair"]["flags"]),
          str(i["churn_pair"]["flags"]))
    check("integ_unchanged_no_change",
          "no_change_in_mask" in i["unchanged_pair"]["flags"],
          str(i["unchanged_pair"]["flags"]))
    check("integ_empty_mask", "mask_too_small" in i["empty_mask_pair"]["flags"]
          and i["empty_mask_pair"]["changed_in_frac"] is None)
    check("integ_full_mask", "mask_too_large" in i["full_mask_pair"]["flags"]
          and i["full_mask_pair"]["changed_out_frac"] is None)
    check("integ_sha_tamper", "source_sha_mismatch" in i["sha_tamper_pair"]["flags"])
    check("integ_missing_mask", "missing_difference_mask"
          in i["missing_mask_pair"]["flags"]
          and i["missing_mask_pair"]["error"] == "missing_files")
    check("integ_atomicity_bad", "new_value_unreferenced"
          in i["atomicity_bad_pair"]["flags"])
    check("integ_local_check_failed", "recorded_local_check_failed"
          in i["local_check_failed_pair"]["flags"])
    check("integ_nonzero_mismatch", "mask_nonzero_mismatch"
          in i["nonzero_mismatch_pair"]["flags"])
    check("integ_binary_mask_ok", "mask_not_binary" not in i["clean_pair"]["flags"])


def test_report(out_dir: Path) -> None:
    proc = run_cli(["report.py", "--audit-dir", str(out_dir)])
    check("report_cli_exit0", proc.returncode == 0, proc.stderr[-400:])
    problems = {json.loads(l)["pair_id"]: json.loads(l) for l in
                (out_dir / "problem_pairs.jsonl").read_text().splitlines()}
    check("report_problem_count", len(problems) == 9, f"got {sorted(problems)}")
    check("report_clean_absent", "clean_pair" not in problems
          and "black_pixel_pair" not in problems)
    check("report_churn_problems",
          {"bg_churn", "low_change_concentration", "bg_style_drift",
           "brightness_shift"} <= set(problems["churn_pair"]["problems"]),
          str(problems.get("churn_pair", {}).get("problems")))
    check("report_unchanged_problem",
          "no_change_in_mask" in problems["unchanged_pair"]["problems"])
    md = (out_dir / "audit_report.md").read_text()
    check("report_md_sections", all(x in md for x in
          ("Problem counts by flag", "Problems by split", "Metric percentiles",
           "Worst offenders", "v8 filter guidance")))
    summ = json.loads((out_dir / "audit_summary.json").read_text())
    check("report_summary_json", summ["n_pairs"] == 11
          and summ["n_problem_pairs"] == 9 and summ["n_clean"] == 2,
          json.dumps(summ)[:200])
    # threshold knob: raise bg-hist-l1 -> no bg_style_drift anywhere
    proc2 = run_cli(["report.py", "--audit-dir", str(out_dir),
                     "--bg-hist-l1", "5.0"])
    flags2 = {fl for l in (out_dir / "problem_pairs.jsonl").read_text().splitlines()
              for fl in json.loads(l)["problems"]}
    check("report_threshold_knob", proc2.returncode == 0
          and "bg_style_drift" not in flags2, str(sorted(flags2)))


def main() -> None:
    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        print("SKIP selftest_data_audit (numpy/PIL unavailable)")
        return
    tmp = Path(tempfile.mkdtemp(prefix="data_audit_selftest_"))
    try:
        test_atomicity_rule()
        splits_dir = build_dataset(tmp)
        out_dir = tmp / "audit_out"
        rows = run_audit_round(splits_dir, out_dir)
        test_style_rows(rows)
        test_integrity_rows(rows)
        test_report(out_dir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAIL:
        print(f"FAILED: {FAIL} ({', '.join(FAILURES)})")
        sys.exit(1)
    print(f"ALL PASS ({PASS} checks)")


if __name__ == "__main__":
    main()
