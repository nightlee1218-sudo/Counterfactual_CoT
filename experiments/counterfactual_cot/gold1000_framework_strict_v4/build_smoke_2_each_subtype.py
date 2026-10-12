#!/usr/bin/env python3
"""Select two pairs per subtype with 30 globally distinct source scenes."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any

SUBTYPES = (
    "same_affordance_replace",
    "shape_divergent_replace",
    "affordance_divergent_replace",
    "color",
    "material",
    "pattern",
    "occlusion_state",
    "functional_state",
    "1_to_2",
    "2_to_1",
    "2_to_3",
    "3_to_2",
    "absolute_horizontal",
    "absolute_vertical",
    "containment",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def entity_count(record: dict[str, Any]) -> int:
    return len(record["source"]["cot"]["entities"])


def representative_candidates(records: list[dict[str, Any]], subtype: str) -> list[dict[str, Any]]:
    by_count: dict[int, list[dict[str, Any]]] = {1: [], 2: [], 3: []}
    for record in records:
        if record["subtype"] == subtype:
            by_count[entity_count(record)].append(record)
    selected: list[dict[str, Any]] = []
    for count in (1, 2, 3):
        selected.extend(sorted(by_count[count], key=lambda record: record["pair_id"])[:12])
    return selected


def options_for_subtype(records: list[dict[str, Any]], subtype: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidates = representative_candidates(records, subtype)
    options = [
        pair for pair in itertools.combinations(candidates, 2)
        if pair[0]["source_scene_id"] != pair[1]["source_scene_id"]
    ]
    options.sort(key=lambda pair: (tuple(sorted(entity_count(record) for record in pair)), pair[0]["pair_id"], pair[1]["pair_id"]))
    if not options:
        raise SystemExit(f"FATAL: fewer than two distinct representative sources for {subtype}")
    return options


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    targets = Counter({1: 10, 2: 10, 3: 10})
    options = {subtype: options_for_subtype(records, subtype) for subtype in SUBTYPES}
    order = sorted(SUBTYPES, key=lambda subtype: len(options[subtype]))
    chosen: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    def search(index: int, used_sources: set[str], histogram: Counter[int]) -> bool:
        if index == len(order):
            return histogram == targets
        subtype = order[index]
        for pair in options[subtype]:
            pair_sources = {record["source_scene_id"] for record in pair}
            if used_sources & pair_sources:
                continue
            pair_histogram = Counter(entity_count(record) for record in pair)
            next_histogram = histogram + pair_histogram
            if any(next_histogram[count] > targets[count] for count in targets):
                continue
            chosen[subtype] = pair
            if search(index + 1, used_sources | pair_sources, next_histogram):
                return True
            del chosen[subtype]
        return False

    if not search(0, set(), Counter()):
        raise SystemExit("FATAL: cannot satisfy 10/10/10 entity-count coverage with 30 distinct sources")

    selected = [record for subtype in SUBTYPES for record in chosen[subtype]]
    subtype_counts = Counter(record["subtype"] for record in selected)
    used_sources = {record["source_scene_id"] for record in selected}
    entity_histogram = Counter(entity_count(record) for record in selected)
    if len(selected) != 30 or subtype_counts != Counter({subtype: 2 for subtype in SUBTYPES}):
        raise SystemExit(f"FATAL: invalid subtype sample: {dict(subtype_counts)}")
    if len(used_sources) != 30 or entity_histogram != targets:
        raise SystemExit(f"FATAL: invalid source or entity distribution: sources={len(used_sources)}, entities={dict(entity_histogram)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote 30 pairs with 30 distinct source scenes to {args.output}")
    print(f"Entity-count distribution: {dict(sorted(entity_histogram.items()))}")


if __name__ == "__main__":
    main()
