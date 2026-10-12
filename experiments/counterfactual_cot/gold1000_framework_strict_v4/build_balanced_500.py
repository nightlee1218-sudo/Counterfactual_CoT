#!/usr/bin/env python3
"""Build a deterministic, quota-balanced 500-pair Gold1000 pilot manifest."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

SUBTYPE_QUOTAS = {
    "same_affordance_replace": 40,
    "shape_divergent_replace": 42,
    "affordance_divergent_replace": 43,
    "color": 22,
    "material": 25,
    "pattern": 25,
    "occlusion_state": 28,
    "functional_state": 25,
    "1_to_2": 30,
    "2_to_1": 30,
    "2_to_3": 35,
    "3_to_2": 30,
    "absolute_horizontal": 38,
    "absolute_vertical": 37,
    "containment": 50,
}
ENTITY_TARGETS = {1: 100, 2: 250, 3: 150}
DOMAINS = {
    "indoor_counter": {"kitchen_counter", "bathroom_counter", "office_desk", "bedroom_nightstand"},
    "furniture_storage": {"bookshelf_study", "closet_wardrobe", "warehouse_shelf"},
    "commercial_display": {"grocery_display", "flower_shop", "toolshed_garage", "antique_curio_shop"},
    "outdoor_open": {"city_street", "parking_lot", "playground", "beach_scene"},
    "indoor_activity": {"living_room_play", "classroom", "dining_table"},
    "dense_closeup": {"pantry_shelf_closeup", "picnic_blanket"},
}
DOMAIN_OF_FAMILY = {family: domain for domain, families in DOMAINS.items() for family in families}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def entity_count(record: dict[str, Any]) -> int:
    return len(record["source"]["cot"]["entities"])


def hamilton_targets(counts: collections.Counter[str], total: int) -> dict[str, int]:
    denominator = sum(counts.values())
    exact = {key: total * value / denominator for key, value in counts.items()}
    result = {key: int(value) for key, value in exact.items()}
    remaining = total - sum(result.values())
    for key in sorted(counts, key=lambda item: (exact[item] - result[item], item), reverse=True)[:remaining]:
        result[key] += 1
    return result


def subtype_entity_plan(records: list[dict[str, Any]]) -> dict[str, dict[int, int]]:
    available: dict[str, collections.Counter[int]] = {}
    options: dict[str, list[tuple[int, int, int]]] = {}
    for subtype, quota in SUBTYPE_QUOTAS.items():
        available[subtype] = collections.Counter(entity_count(record) for record in records if record["subtype"] == subtype)
        patterns: list[tuple[int, int, int]] = []
        for one in range(min(quota, available[subtype][1]) + 1):
            for two in range(min(quota - one, available[subtype][2]) + 1):
                three = quota - one - two
                if 0 <= three <= available[subtype][3]:
                    patterns.append((one, two, three))
        if not patterns:
            raise SystemExit(f"FATAL: no entity-count allocation for {subtype}; available={dict(available[subtype])}")
        options[subtype] = patterns

    states: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {(0, 0, 0): []}
    for subtype in SUBTYPE_QUOTAS:
        next_states: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
        for prior, plan in states.items():
            for pattern in options[subtype]:
                total = tuple(prior[index] + pattern[index] for index in range(3))
                if total[0] > ENTITY_TARGETS[1] or total[1] > ENTITY_TARGETS[2] or total[2] > ENTITY_TARGETS[3]:
                    continue
                existing = next_states.get(total)
                if existing is None or pattern < existing[-1]:
                    next_states[total] = plan + [pattern]
        states = next_states
    target = (ENTITY_TARGETS[1], ENTITY_TARGETS[2], ENTITY_TARGETS[3])
    if target not in states:
        raise SystemExit(f"FATAL: cannot reach entity targets {ENTITY_TARGETS}")
    return {
        subtype: {1: pattern[0], 2: pattern[1], 3: pattern[2]}
        for subtype, pattern in zip(SUBTYPE_QUOTAS, states[target])
    }


def choose_records(records: list[dict[str, Any]], plan: dict[str, dict[int, int]]) -> list[dict[str, Any]]:
    """Use a binary MILP so quotas, complexity, domains and scene coverage hold together."""
    pair_count = len(records)
    source_ids = sorted({record["source_scene_id"] for record in records})
    source_index = {scene_id: index for index, scene_id in enumerate(source_ids)}
    variable_count = pair_count + len(source_ids)
    rows: list[tuple[dict[int, float], float, float]] = []

    def add_constraint(coefficients: dict[int, float], lower: float, upper: float) -> None:
        rows.append((coefficients, lower, upper))

    # Exact total, subtype quotas, and 1/2/3-entity curriculum.
    add_constraint({index: 1.0 for index in range(pair_count)}, 500, 500)
    for subtype, quota in SUBTYPE_QUOTAS.items():
        add_constraint({index: 1.0 for index, record in enumerate(records) if record["subtype"] == subtype}, quota, quota)
    for count, quota in ENTITY_TARGETS.items():
        add_constraint({index: 1.0 for index, record in enumerate(records) if entity_count(record) == count}, quota, quota)

    # All 20 official scenes appear. Domains use Hamilton half-sampling targets
    # from the full manifest, rather than merely satisfying the 25% ceiling.
    for family in sorted(DOMAIN_OF_FAMILY):
        add_constraint({index: 1.0 for index, record in enumerate(records) if record["template_family"] == family}, 1, np.inf)
    domain_targets = hamilton_targets(
        collections.Counter(DOMAIN_OF_FAMILY[record["template_family"]] for record in records), 500
    )
    for domain, target in domain_targets.items():
        add_constraint({index: 1.0 for index, record in enumerate(records) if DOMAIN_OF_FAMILY[record["template_family"]] == domain}, target, target)

    # y_scene is active when any selected pair uses that source. Minimizing y
    # saves source-image calls while source reuse remains capped by the full manifest.
    for index, record in enumerate(records):
        add_constraint({index: 1.0, pair_count + source_index[record["source_scene_id"]]: -1.0}, -np.inf, 0)

    matrix = lil_matrix((len(rows), variable_count), dtype=float)
    lower = np.empty(len(rows), dtype=float)
    upper = np.empty(len(rows), dtype=float)
    for row_index, (coefficients, lb, ub) in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
        lower[row_index] = lb
        upper[row_index] = ub

    # Source calls dominate cost; deterministic epsilon gives stable tie-breaking.
    objective = np.concatenate((np.arange(pair_count, dtype=float) * 1e-7, np.ones(len(source_ids))))
    result = milp(
        c=objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 300.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise SystemExit(f"FATAL: 500-pair MILP failed: status={result.status}; message={result.message}")
    selected = [record for index, record in enumerate(records) if result.x[index] > 0.5]
    if len(selected) != 500:
        raise SystemExit(f"FATAL: MILP selected {len(selected)} records")
    return sorted(selected, key=lambda record: record["pair_id"])


def audit(selected: list[dict[str, Any]]) -> dict[str, Any]:
    subtype_counts = collections.Counter(record["subtype"] for record in selected)
    entity_counts = collections.Counter(entity_count(record) for record in selected)
    family_counts = collections.Counter(record["template_family"] for record in selected)
    domain_counts = collections.Counter(DOMAIN_OF_FAMILY[record["template_family"]] for record in selected)
    source_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in selected:
        source_groups[record["source_scene_id"]].append(record)
    errors: list[str] = []
    if subtype_counts != collections.Counter(SUBTYPE_QUOTAS):
        errors.append(f"subtype_counts={dict(subtype_counts)}")
    if entity_counts != collections.Counter(ENTITY_TARGETS):
        errors.append(f"entity_counts={dict(entity_counts)}")
    if set(family_counts) != set(DOMAIN_OF_FAMILY):
        errors.append(f"missing_families={sorted(set(DOMAIN_OF_FAMILY) - set(family_counts))}")
    if any(count > 125 for count in domain_counts.values()):
        errors.append(f"domain_over_25pct={dict(domain_counts)}")
    if len({record["pair_id"] for record in selected}) != 500:
        errors.append("duplicate_pair_id")
    if len({(record["source"]["cot_text"], record["target"]["cot_text"]) for record in selected}) != 500:
        errors.append("duplicate_transition")
    for scene_id, group in source_groups.items():
        if len(group) > 3:
            errors.append(f"source_reuse_over_cap={scene_id}:{len(group)}")
        if len({record["source"]["cot_text"] for record in group}) != 1 or len({record["source"]["prompt"] for record in group}) != 1:
            errors.append(f"source_inconsistent={scene_id}")
    if errors:
        raise SystemExit("FATAL: " + "; ".join(errors))
    return {
        "records": len(selected),
        "subtype_counts": dict(sorted(subtype_counts.items())),
        "entity_count_distribution": dict(sorted(entity_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "source_scenes": len(source_groups),
        "source_reuse_distribution": dict(sorted(collections.Counter(len(group) for group in source_groups.values()).items())),
        "shared_pairs": sum(len(group) for group in source_groups.values() if len(group) > 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    records = load_jsonl(args.input)
    plan = subtype_entity_plan(records)
    selected = choose_records(records, plan)
    report = audit(selected)
    report["subtype_entity_plan"] = {subtype: {str(count): value for count, value in counts.items()} for subtype, counts in plan.items()}
    write_jsonl(args.output, selected)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
