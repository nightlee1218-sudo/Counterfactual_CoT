#!/usr/bin/env python3
"""Freeze the Sentinel-6 subset from the final Gold40 manifest.

The script never edits Gold40. It writes one immutable-style subset per training
prompt profile plus a provenance report. The rendered prompt is stored in
``training_prompts.paired_edit`` so the existing BAGEL dataset path uses exactly
the same text that inference must use.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.data.prompt_profiles import render_prompt_profile


SOURCE_MANIFEST = (
    BAGEL_ROOT
    / "experiments/counterfactual_cot/gold40_final/manifest/"
    "intervention_pairs_gold40_final.jsonl"
)
SOURCE_MANIFEST_SHA256 = "e74a0dc713cd99628d4df296a54c9c367913682d3a88ed030ff12de321d3587c"
DEFAULT_OUTPUT_DIR = BAGEL_ROOT / "experiments/counterfactual_cot/data/sentinel6_final_v2"
PROFILES = ("delta_only_v2", "cot_core_v1")
PAIR_IDS = (
    "cf_source_library_table_attribute_replace_0000",
    "cf_source_seaside_path_count_change_0000",
    "cf_source_classroom_object_replace_0000",
    "cf_source_city_street_absolute_position_change_v4_absolute_region_review_fix_v6_short_0000",
    "cf_source_home_kitchen_spatial_relation_change_0000",
    "cf_source_living_room_play_absolute_position_change_v4_absolute_region_replacement_v4_0000",
)
EXPECTED_TYPE_COUNTS = {
    "attribute_replace": 1,
    "count_change": 1,
    "object_replace": 1,
    "spatial_position_change": 2,
    "spatial_relation_change": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(record: Dict[str, Any]) -> str:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_exclusive(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_image(record: Dict[str, Any], side: str) -> Path:
    value = Path(record[side]["image"])
    if value.is_absolute():
        return value
    image_root = record.get("final_image_root")
    if image_root:
        return Path(image_root) / value
    return SOURCE_MANIFEST.parent / value


def validate_source(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {record["pair_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Gold40 final contains duplicate pair_id values")
    missing = [pair_id for pair_id in PAIR_IDS if pair_id not in by_id]
    if missing:
        raise ValueError(f"Gold40 final is missing Sentinel-6 pairs: {missing}")

    selected = [by_id[pair_id] for pair_id in PAIR_IDS]
    counts = Counter(record["intervention"]["intervention_type"] for record in selected)
    if dict(counts) != EXPECTED_TYPE_COUNTS:
        raise ValueError(f"Unexpected Sentinel-6 intervention coverage: {dict(counts)}")
    rejected = [record["pair_id"] for record in selected if record.get("human_review", {}).get("accepted") is not True]
    if rejected:
        raise ValueError(f"Sentinel-6 contains records not accepted in Gold40 final: {rejected}")

    classroom = by_id["cf_source_classroom_object_replace_0000"]
    intervention = classroom["intervention"]
    if intervention.get("old_value") != "chair" or intervention.get("new_value") != "globe":
        raise ValueError(f"Final classroom contract must be chair->globe, got {intervention}")
    if "globe" not in classroom["target"]["prompt"].lower():
        raise ValueError("Final classroom target prompt does not mention globe")

    missing_images = []
    for record in selected:
        for side in ("source", "target"):
            image = resolve_image(record, side)
            if not image.is_file():
                missing_images.append(f"{record['pair_id']}:{side}:{image}")
    if missing_images:
        raise FileNotFoundError(f"Sentinel-6 image files are missing: {missing_images}")
    return selected


def build_profile_records(selected: List[Dict[str, Any]], profile: str) -> List[Dict[str, Any]]:
    output = []
    for source_record in selected:
        record = copy.deepcopy(source_record)
        rendered = render_prompt_profile(record, profile)
        record.setdefault("training_prompts", {})["paired_edit"] = rendered
        record["sentinel6_provenance"] = {
            "source_manifest": str(SOURCE_MANIFEST),
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_record_canonical_sha256": canonical_sha256(source_record),
            "prompt_profile": profile,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        }
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    actual_sha = sha256_file(SOURCE_MANIFEST)
    if actual_sha != SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            f"Gold40 final manifest hash mismatch: got={actual_sha} expected={SOURCE_MANIFEST_SHA256}"
        )
    selected = validate_source(load_jsonl(SOURCE_MANIFEST))

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Sentinel-6 directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_manifests: Dict[str, Dict[str, Any]] = {}
    for profile in PROFILES:
        path = args.output_dir / f"sentinel6_gold40_final__profile={profile}.jsonl"
        profile_records = build_profile_records(selected, profile)
        write_jsonl_exclusive(path, profile_records)
        output_manifests[profile] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "record_count": len(profile_records),
            "rendered_prompts": {
                record["pair_id"]: record["training_prompts"]["paired_edit"]
                for record in profile_records
            },
        }

    report = {
        "schema": "sentinel6_gold40_final_v2",
        "status": "frozen",
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": actual_sha,
        "pair_ids_in_order": list(PAIR_IDS),
        "intervention_type_counts": EXPECTED_TYPE_COUNTS,
        "classroom_object_replace_contract": {"old_value": "chair", "new_value": "globe"},
        "human_review_accepted_required": True,
        "output_manifests": output_manifests,
    }
    report_path = args.output_dir / "sentinel6_freeze_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "frozen",
        "source_manifest_sha256": actual_sha,
        "record_count": len(selected),
        "profiles": list(PROFILES),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
