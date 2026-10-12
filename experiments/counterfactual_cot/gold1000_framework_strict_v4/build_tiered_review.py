#!/usr/bin/env python3
"""Build a tiered human-review package so the 1000 pairs need not all be audited.

Tier 1 (must review): every non-auto_checked pair plus auto_checked pairs whose
automatic mask is below 1% -- the edit technically changed pixels but may be
semantically invisible or wrong.
Tier 2 (should review): auto_checked pairs whose mask exceeds 30% -- the edit
may have damaged must_preserve objects.
Tier 3 (sampled): ten pairs per intervention subtype drawn from the healthy
middle band, to verify that band's overall quality.

Outputs a compact CSV for filling (much easier than scrolling 1000 rows) plus an
HTML viewer ordered tier by tier.  Use ``merge_tiered_review.py`` afterwards to
write the decisions back into the full 1000-row review CSV.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from experiments.counterfactual_cot.data.schema import load_jsonl
from experiments.counterfactual_cot.data.review_export import export_review

TINY_MASK = 0.01
LARGE_MASK = 0.30
PER_SUBTYPE_SAMPLE = 10


def ratio(record: dict) -> float:
    checks = record.get("quality", {}).get("local_checks") or {}
    return float(checks.get("mask_area_ratio", 0.0))


def build_tiers(records: list[dict], seed: int) -> tuple[list[dict], dict[str, int]]:
    flagged = [r for r in records if r["quality"]["review_status"] != "auto_checked"]
    checked = [r for r in records if r["quality"]["review_status"] == "auto_checked"]
    tiny = [r for r in checked if ratio(r) < TINY_MASK]
    large = [r for r in checked if ratio(r) > LARGE_MASK]
    middle = [r for r in checked if TINY_MASK <= ratio(r) <= LARGE_MASK]

    tier1 = flagged + tiny
    tier2 = large

    by_subtype: dict[str, list[dict]] = defaultdict(list)
    for record in middle:
        by_subtype[str(record.get("subtype", "unknown"))].append(record)
    rng = random.Random(seed)
    tier3: list[dict] = []
    for subtype in sorted(by_subtype):
        pool = by_subtype[subtype]
        tier3.extend(rng.sample(pool, min(PER_SUBTYPE_SAMPLE, len(pool))))

    seen: set[str] = set()
    ordered: list[dict] = []
    for record in tier1 + tier2 + tier3:
        if record["pair_id"] in seen:
            continue
        seen.add(record["pair_id"])
        ordered.append(record)
    return ordered, {"tier1": len(tier1), "tier2": len(tier2), "tier3": len(tier3), "total": len(ordered)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()

    records = load_jsonl(args.manifest)
    ordered, counts = build_tiers(records, args.seed)

    # Tier labels are injected so the HTML details block shows the reason.
    n1, n2 = counts["tier1"], counts["tier1"] + counts["tier2"]
    for index, record in enumerate(ordered):
        tier = "tier1_must_review" if index < n1 else ("tier2_should_review" if index < n2 else "tier3_sampled")
        record["review_tier"] = tier

    export_review(ordered, args.output_dir, args.image_root)
    with (args.output_dir / "tier_index.json").open("w", encoding="utf-8") as handle:
        json.dump({"counts": counts, "seed": args.seed,
                   "thresholds": {"tiny_mask": TINY_MASK, "large_mask": LARGE_MASK},
                   "pair_ids": [r["pair_id"] for r in ordered]}, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output_dir": str(args.output_dir), **counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
