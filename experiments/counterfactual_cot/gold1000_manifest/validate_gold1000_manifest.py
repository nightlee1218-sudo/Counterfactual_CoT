#!/usr/bin/env python3
"""Read-only pre-generation audit for the Gold1000 semantic manifest.

This script is deliberately non-mutating. It reports every contract violation that
must be fixed before any source/target image is generated, because image
generation is expensive and a semantic manifest error becomes an unrecoverable
dataset error once images and automatic masks exist.

Contract checked here
---------------------
1.  pair_id is globally unique.
2.  source_scene_id has exactly one source.prompt and one source.cot_text.
3.  source.cot and target.cot are structurally aligned: same entity ids in the
    same order, each entity has the same attribute keys.
4.  cot_text is the deterministic rendering of cot.
5.  source.cot_text and target.cot_text differ in exactly one atom.
6.  that atom corresponds to intervention.entity_id + intervention.field.
7.  the changed atom values equal intervention.old_value / new_value.
8.  intervention_type / intervention_subtype are stable and match dimension/subtype.
9.  preservation.allowed_to_change contains the changed atom; must_preserve
    covers every unchanged atom and never overlaps allowed_to_change.
10. location values stay inside the controlled region vocabulary.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent

LOCATION_VOCAB = {
    "left_region",
    "right_region",
    "center_region",
    "top_region",
    "bottom_region",
    "upper_left",
    "upper_right",
    "lower_left",
    "lower_right",
    "center",
}

DIMENSION_ALIASES = {
    "spatial_geometric": "spatial_geometric_intervention",
    "instance_cardinality": "instance_cardinality_intervention",
    "spatial_relational_intervention": "spatial_relation_change_intervention",
    "object_replace_intervention": "category_replacement",
}


def _read_jsonl(paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["__file"] = path.name
            record["__line"] = line_number
            records.append(record)
    return records


def _relation_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], Mapping):
        return f"{value[0].get('predicate')}:{value[0].get('target')}"
    return json.dumps(value, sort_keys=True)


def render_cot_text(cot: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for entity in cot.get("entities", []):
        eid = entity["id"]
        parts.append(f"{eid}.object={entity['category']}")
        parts.append(f"{eid}.count={entity['count']}")
        if "location" in entity:
            parts.append(f"{eid}.location={entity['location']}")
        for key, value in (entity.get("attributes") or {}).items():
            parts.append(f"{eid}.attribute.{key}={value}")
        for index, relation in enumerate(entity.get("relations") or []):
            parts.append(f"{eid}.relation.{index}={relation['predicate']}:{relation['target']}")
    background = cot.get("background")
    if isinstance(background, Mapping):
        for key, value in background.items():
            parts.append(f"background.{key}={value}")
    else:
        parts.append(f"background.scene={background}")
    return "; ".join(parts)


def _atoms(text: str) -> List[str]:
    return [atom.strip() for atom in text.split(";") if atom.strip()]


def _atom_key(atom: str) -> str:
    """Normalise a preservation entry to the cot_text atom vocabulary.

    cot_text is the single source of truth, so `category` renders as `.object`,
    `attributes.x` renders as `.attribute.x`, and `relations[0].predicate`
    renders as `.relation.0`.
    """
    key = atom.split("=", 1)[0]
    key = key.replace(".category", ".object").replace(".attributes.", ".attribute.")
    key = key.replace(".relations[0].predicate", ".relation.0").replace(".relations", ".relation.0")
    return key


def _expected_atom_key(intervention: Mapping[str, Any]) -> str:
    return _atom_key(f"{intervention['entity_id']}.{intervention['field']}")


def audit(records: Iterable[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    records = list(records)
    findings: List[Dict[str, Any]] = []
    counts: collections.Counter = collections.Counter()

    def add(code: str, **detail: Any) -> None:
        counts[code] += 1
        if counts[code] <= 25:
            findings.append({"code": code, **detail})

    by_pair: Dict[str, List[Mapping[str, Any]]] = collections.defaultdict(list)
    for record in records:
        by_pair[str(record.get("pair_id"))].append(record)
    for pair_id, group in by_pair.items():
        if len(group) > 1:
            add("duplicate_pair_id", pair_id=pair_id, locations=[f"{g['__file']}:{g['__line']}" for g in group])

    unique: List[Mapping[str, Any]] = [group[0] for group in by_pair.values()]

    scenes: Dict[str, List[Mapping[str, Any]]] = collections.defaultdict(list)
    for record in unique:
        scenes[str(record.get("source_scene_id"))].append(record)
    for scene_id, group in scenes.items():
        prompts = {record["source"]["prompt"] for record in group}
        cots = {record["source"]["cot_text"] for record in group}
        if len(prompts) > 1:
            add("scene_source_prompt_not_unique", scene_id=scene_id, variants=len(prompts))
        if len(cots) > 1:
            add("scene_source_cot_not_unique", scene_id=scene_id, variants=len(cots))

    for record in unique:
        pair_id = str(record.get("pair_id"))
        location = f"{record['__file']}:{record['__line']}"
        source = record.get("source") or {}
        target = record.get("target") or {}
        intervention = record.get("intervention") or {}
        preservation = record.get("preservation") or {}

        for side in ("source", "target"):
            cot = (record[side].get("cot") or {})
            if not isinstance(cot.get("background"), Mapping):
                add("background_not_object", pair_id=pair_id, side=side, location=location)
            expected = render_cot_text(cot)
            if expected != record[side].get("cot_text"):
                add("cot_text_not_canonical", pair_id=pair_id, side=side, location=location)

        source_entities = [e["id"] for e in (source.get("cot") or {}).get("entities", [])]
        target_entities = [e["id"] for e in (target.get("cot") or {}).get("entities", [])]
        if source_entities != target_entities:
            add("entity_alignment_mismatch", pair_id=pair_id, location=location)

        source_atoms = _atoms(str(source.get("cot_text") or ""))
        target_atoms = _atoms(str(target.get("cot_text") or ""))
        if len(source_atoms) != len(target_atoms) or len(source_atoms) == 0:
            add("atom_count_mismatch", pair_id=pair_id, location=location,
                source=len(source_atoms), target=len(target_atoms))
            continue
        diff = [(a, b) for a, b in zip(source_atoms, target_atoms) if a != b]
        if len(diff) != 1:
            add("not_single_atom_change", pair_id=pair_id, location=location, diffs=diff)
            continue
        old_atom, new_atom = diff[0]
        key = old_atom.split("=", 1)[0]
        expected_key = _expected_atom_key(intervention)
        if key != expected_key:
            add("changed_atom_key_mismatch", pair_id=pair_id, location=location,
                actual=key, expected=expected_key)
        old_value, new_value = old_atom.split("=", 1)[1], new_atom.split("=", 1)[1]
        if old_value != _relation_value(intervention.get("old_value")) or \
                new_value != _relation_value(intervention.get("new_value")):
            add("changed_values_mismatch_intervention", pair_id=pair_id, location=location,
                atom=f"{old_atom} -> {new_atom}",
                intervention=f"{intervention.get('old_value')} -> {intervention.get('new_value')}")
        if intervention.get("old_value") == intervention.get("new_value"):
            add("intervention_no_change", pair_id=pair_id, location=location)

        dimension = str(record.get("dimension"))
        normalized_dimension = DIMENSION_ALIASES.get(dimension, dimension)
        intervention_type = str(intervention.get("intervention_type"))
        normalized_type = DIMENSION_ALIASES.get(intervention_type, intervention_type)
        if normalized_dimension != normalized_type:
            add("dimension_type_inconsistent", pair_id=pair_id, location=location,
                dimension=dimension, intervention_type=intervention_type)
        if record.get("subtype") != intervention.get("intervention_subtype"):
            add("subtype_inconsistent", pair_id=pair_id, location=location,
                subtype=record.get("subtype"), intervention_subtype=intervention.get("intervention_subtype"))

        allowed = {_atom_key(a) for a in preservation.get("allowed_to_change", [])}
        preserve = {_atom_key(p) for p in preservation.get("must_preserve", [])}
        if allowed & preserve:
            add("preservation_overlap", pair_id=pair_id, location=location, overlap=sorted(allowed & preserve))
        if key not in allowed:
            add("changed_atom_not_allowed", pair_id=pair_id, location=location, atom=key)
        if key in preserve:
            add("changed_atom_in_must_preserve", pair_id=pair_id, location=location, atom=key)
        uncovered = [atom_key for atom in source_atoms
                     if (atom_key := _atom_key(atom)) != key
                     and atom_key not in preserve
                     and not atom.startswith("background.")]
        if uncovered:
            add("unchanged_atoms_not_preserved", pair_id=pair_id, location=location, atoms=uncovered)

        for side in ("source", "target"):
            for entity in (record[side].get("cot") or {}).get("entities", []):
                loc = entity.get("location")
                if loc is not None and loc not in LOCATION_VOCAB:
                    add("location_outside_vocabulary", pair_id=pair_id, side=side, location=loc)

    report = {
        "input_records": len(records),
        "unique_pair_ids": len(by_pair),
        "unique_scenes": len(scenes),
        "scene_size_histogram": dict(sorted(collections.Counter(len(v) for v in scenes.values()).items())),
        "dimension_histogram": dict(collections.Counter(str(r.get("dimension")) for r in unique)),
        "field_histogram": dict(collections.Counter(str(r.get("intervention", {}).get("field")) for r in unique)),
        "findings_total": sum(counts.values()),
        "finding_counts": dict(counts.most_common()),
        "ready_for_image_generation": sum(counts.values()) == 0,
    }
    return report, findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--report", type=Path, default=SCRIPT_DIR / "gold1000_manifest_audit.json")
    args = parser.parse_args()

    paths = sorted(p for p in args.manifest_dir.glob("*.jsonl") if p.name != "gold1000_manifest_audit.jsonl")
    records = _read_jsonl(paths)
    report, findings = audit(records)
    report["files"] = [p.name for p in paths]
    report["findings"] = findings
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "findings"}, ensure_ascii=False, indent=2))
    print(f"\nfull report: {args.report}")


if __name__ == "__main__":
    main()
