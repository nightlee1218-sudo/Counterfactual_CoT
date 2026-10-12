"""Full-manifest data audit for Gold40 (hard-gate version).

This script executes ONE-BY-ONE checks against every pair in the fixed
Gold40 manifest. Every check listed below is a HARD failure (audit exits with
code 1 if any pair fails any of them).

Hard checks (per pair):

  1.  ``pair_id`` present, non-empty, unique across the manifest.
  2.  ``source.image``, ``target.image`` and ``target.mask`` all resolve to
      real files on disk under ``pair.final_image_root`` (or manifest.parent
      when the manifest field is absolute).
  3.  Source and target images decode via PIL.
  4.  Both images are (1024, 1024) and source_size == target_size.
  5.  Source and target CoT parse as SceneGraph. Applying the intervention
      to the source scene reproduces the target scene EXACTLY.
      This script uses its own extended apply_intervention that supports
      the deep-path field ``relations.<i>.<key>`` (used by
      spatial_relation_change pairs in Gold40). We do NOT modify the shared
      schema.py -- only interpret the manifest here.
  6.  The structural diff between source and target scenes contains exactly
      one atom, and that atom equals ``<entity_id>.<field>`` (normalized).
  7.  ``preservation.allowed_to_change`` contains exactly one entry, equal
      to the normalized intervention atom.
  8.  ``preservation.must_preserve`` is a superset (up to trivial format
      normalization) of every source/target scene atom EXCEPT the
      intervention atom; must not contain the intervention atom; must not
      overlap ``allowed_to_change``. Extra atoms in ``must_preserve`` that
      cannot be resolved to any real scene atom are also a HARD failure.
  9.  Path resolution never walks through ``gold40_final/masks``.

Non-fatal warnings (recorded, do not affect pass/fail):

  * ``spatial_position_change`` with old/new values outside the canonical
    ``{left_region, right_region, upper_region, lower_region}`` set --
    e.g. ``center_inside_wicker_basket``, ``lower-left`` / ``lower-right``.
    Evaluator must special-case these pairs; training loss is untouched.
  * ``spatial_relation_change`` predicates on the known ``on->beside`` and
    ``beside->in_front_of`` axis (or any unexpected pair).
  * Stored ``cot_text`` disagreeing with a fresh ``render_cot_text`` run.

Attribute-field normalization:

  Manifest uses ``attributes.color`` in intervention.field. Some preservation
  lists in Gold40 use the singular alias ``attribute.color``. Both forms are
  normalized to the plural ``attributes.color`` before ANY comparison.

Usage:
    python -m experiments.counterfactual_cot.training.gold40_full_manifest_audit
    python -m experiments.counterfactual_cot.training.gold40_full_manifest_audit \
        --output /group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/gold40_full_audit/gold40_full_audit.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"PIL is required for image decoding checks: {exc}")

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.data.schema import (  # noqa: E402
    Intervention,
    SceneGraph,
    changed_scene_paths,
    load_jsonl,
    render_cot_text,
)


FIXED_GOLD40_MANIFEST = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
    "reruns/nanobanana_v4_absolute_region/manifests/"
    "intervention_pairs_gold40_human_verified_final_normalized.jsonl"
)
FORBIDDEN_MASK_PARENT_SUBSTR = "gold40_final/masks"
CANONICAL_REGIONS = {"left_region", "right_region", "upper_region", "lower_region"}
CANONICAL_RELATION_PAIRS = {("on", "beside"), ("beside", "in_front_of")}


# ---------------------------------------------------------------------------
# Local (audit-only) apply_intervention that supports every field shape used
# by the Gold40 manifest. We do NOT modify the shared schema.py.
# ---------------------------------------------------------------------------

def _parse_relation_deep_path(field: str) -> Optional[Tuple[int, str]]:
    """Parse ``relations.<i>.<key>``. Returns ``(index, key)`` or ``None``."""
    if not field.startswith("relations."):
        return None
    parts = field.split(".", 2)
    if len(parts) != 3:
        return None
    try:
        idx = int(parts[1])
    except ValueError:
        return None
    key = parts[2]
    return idx, key


def _get_field_extended(scene: SceneGraph, entity_id: str, field: str) -> Any:
    entity = scene.entities[scene.find_entity_index(entity_id)]
    if field == "category":
        return entity.category
    if field == "count":
        return entity.count
    if field == "location":
        return entity.location
    if field.startswith("attributes."):
        key = field.split(".", 1)[1]
        return entity.attributes.get(key)
    if field.startswith("attribute."):
        key = field.split(".", 1)[1]
        return entity.attributes.get(key)
    deep = _parse_relation_deep_path(field)
    if deep is not None:
        idx, key = deep
        if idx < 0 or idx >= len(entity.relations):
            return None
        return entity.relations[idx].get(key)
    if field == "relations":
        return entity.relations
    raise ValueError(f"Unsupported intervention field for audit: {field!r}")


def _set_field_extended(scene: SceneGraph, entity_id: str, field: str, value: Any) -> SceneGraph:
    scene_dict = copy.deepcopy(scene.to_dict())
    entity_index = SceneGraph.from_dict(scene_dict).find_entity_index(entity_id)
    entity_dict = scene_dict["entities"][entity_index]
    if field == "category":
        entity_dict["category"] = value
    elif field == "count":
        entity_dict["count"] = int(value)
    elif field == "location":
        entity_dict["location"] = value
    elif field.startswith("attributes."):
        key = field.split(".", 1)[1]
        entity_dict.setdefault("attributes", {})[key] = value
    elif field.startswith("attribute."):
        key = field.split(".", 1)[1]
        entity_dict.setdefault("attributes", {})[key] = value
    elif field == "relations":
        if not isinstance(value, list):
            raise ValueError("relations intervention value must be a list")
        entity_dict["relations"] = value
    else:
        deep = _parse_relation_deep_path(field)
        if deep is None:
            raise ValueError(f"Unsupported intervention field for audit: {field!r}")
        idx, key = deep
        rels = entity_dict.setdefault("relations", [])
        while len(rels) <= idx:
            rels.append({})
        rels[idx][key] = value
    return SceneGraph.from_dict(scene_dict)


def _apply_intervention_extended(source: SceneGraph, intervention: Intervention) -> SceneGraph:
    current_value = _get_field_extended(source, intervention.entity_id, intervention.field)
    if current_value != intervention.old_value:
        raise ValueError(
            f"Intervention old_value mismatch for {intervention.entity_id}.{intervention.field}: "
            f"expected {intervention.old_value!r}, found {current_value!r}"
        )
    return _set_field_extended(source, intervention.entity_id, intervention.field, intervention.new_value)


# ---------------------------------------------------------------------------
# Path resolution -- matches gold40_overfit._resolve_image_path.
# ---------------------------------------------------------------------------

def _resolve_image_path(pair: Dict[str, Any], path_value: str, manifest_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    image_root = pair.get("final_image_root")
    if image_root:
        rooted = Path(image_root) / path
        if rooted.exists():
            return rooted
    return manifest_dir / path


# ---------------------------------------------------------------------------
# Scene atoms enumeration used by must_preserve completeness check.
# ---------------------------------------------------------------------------

def _scene_atoms(scene: SceneGraph) -> List[str]:
    atoms: List[str] = []
    for entity in scene.entities:
        atoms.append(f"{entity.id}.category")
        if entity.count is not None:
            atoms.append(f"{entity.id}.count")
        if entity.location is not None:
            atoms.append(f"{entity.id}.location")
        for key in sorted(entity.attributes):
            atoms.append(f"{entity.id}.attributes.{key}")
        for relation_index in range(len(entity.relations)):
            atoms.append(f"{entity.id}.relations.{relation_index}.predicate")
            atoms.append(f"{entity.id}.relations.{relation_index}.target")
    for key in sorted(scene.background):
        atoms.append(f"background.{key}")
    return atoms


# ---------------------------------------------------------------------------
# Normalization helpers.
# ---------------------------------------------------------------------------

def _normalize_atom(atom: str) -> str:
    """Normalize atom form.

    * ``entity.attribute.color`` <-> ``entity.attributes.color``.
    * ``entity.relation.<i>.<key>`` <-> ``entity.relations.<i>.<key>``.
    * ``entity.relations[<i>].<key>`` -> ``entity.relations.<i>.<key>``
      (this normalizes the ``changed_scene_paths`` output which uses ``[i]``
      array indices).
    """
    # Convert bracket indexing ``foo[3]`` -> ``foo.3`` first.
    normalized_parts: List[str] = []
    for chunk in atom.split("."):
        if "[" in chunk and chunk.endswith("]"):
            base, _, idx = chunk[:-1].partition("[")
            if idx.isdigit():
                if base:
                    normalized_parts.append(base)
                normalized_parts.append(idx)
                continue
        normalized_parts.append(chunk)
    parts = normalized_parts
    for i, part in enumerate(parts):
        if part == "attribute":
            parts[i] = "attributes"
        elif part == "relation" and i + 1 < len(parts) and parts[i + 1].isdigit():
            parts[i] = "relations"
    return ".".join(parts)


def _normalize_field(field: str) -> str:
    """Normalize a bare intervention field: ``attribute.X`` -> ``attributes.X``."""
    if field.startswith("attribute."):
        return "attributes." + field.split(".", 1)[1]
    return field


# ---------------------------------------------------------------------------
# Per-pair audit
# ---------------------------------------------------------------------------

def _audit_pair(pair: Dict[str, Any], manifest_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "pair_id": pair.get("pair_id"),
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    def _record(name: str, ok: bool, detail: Any = None) -> None:
        result["checks"][name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            result["errors"].append(name)

    # --- 1) pair_id ---
    pair_id = pair.get("pair_id")
    _record("pair_id_present", isinstance(pair_id, str) and bool(pair_id))

    intervention_raw = pair.get("intervention")
    _record("intervention_is_dict", isinstance(intervention_raw, dict))
    if not isinstance(intervention_raw, dict):
        return result

    source_info = pair.get("source") or {}
    target_info = pair.get("target") or {}

    source_rel = source_info.get("image")
    target_rel = target_info.get("image")
    mask_rel = target_info.get("mask")

    _record("source_image_field_present", isinstance(source_rel, str) and bool(source_rel))
    _record("target_image_field_present", isinstance(target_rel, str) and bool(target_rel))
    _record("target_mask_field_present", isinstance(mask_rel, str) and bool(mask_rel))

    if not (isinstance(source_rel, str) and isinstance(target_rel, str) and isinstance(mask_rel, str)):
        return result

    source_path = _resolve_image_path(pair, source_rel, manifest_dir)
    target_path = _resolve_image_path(pair, target_rel, manifest_dir)
    mask_path = _resolve_image_path(pair, mask_rel, manifest_dir)

    _record("source_image_exists", source_path.exists(), {"resolved": str(source_path)})
    _record("target_image_exists", target_path.exists(), {"resolved": str(target_path)})
    _record("target_mask_exists", mask_path.exists(), {"resolved": str(mask_path)})

    for label, resolved in (
        ("source_image", source_path),
        ("target_image", target_path),
        ("target_mask", mask_path),
    ):
        _record(
            f"{label}_not_via_gold40_final_masks",
            FORBIDDEN_MASK_PARENT_SUBSTR not in str(resolved),
            {"resolved": str(resolved)},
        )

    def _decode_size(path: Path) -> Optional[Tuple[int, int]]:
        try:
            with Image.open(path) as image:
                image.load()
                return image.size
        except Exception as exc:  # noqa: BLE001
            result["errors"].append(f"decode_fail:{path}:{exc}")
            return None

    src_size = _decode_size(source_path) if source_path.exists() else None
    tgt_size = _decode_size(target_path) if target_path.exists() else None

    _record("source_decodes", src_size is not None, {"size": src_size})
    _record("target_decodes", tgt_size is not None, {"size": tgt_size})
    _record("source_size_is_1024x1024", src_size == (1024, 1024), {"size": src_size})
    _record("target_size_is_1024x1024", tgt_size == (1024, 1024), {"size": tgt_size})
    _record(
        "source_target_size_equal",
        (src_size is not None and src_size == tgt_size),
        {"source_size": src_size, "target_size": tgt_size},
    )

    source_cot = source_info.get("cot")
    target_cot = target_info.get("cot")
    _record("source_cot_dict", isinstance(source_cot, dict))
    _record("target_cot_dict", isinstance(target_cot, dict))

    try:
        source_scene = SceneGraph.from_dict(source_cot) if isinstance(source_cot, dict) else None
        target_scene = SceneGraph.from_dict(target_cot) if isinstance(target_cot, dict) else None
        _record("source_scene_parses", source_scene is not None)
        _record("target_scene_parses", target_scene is not None)
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(f"scene_parse_fail:{exc}")
        return result

    if source_scene is None or target_scene is None:
        return result

    try:
        intervention = Intervention.from_dict(intervention_raw)
        _record("intervention_parses", True)
    except Exception as exc:  # noqa: BLE001
        _record("intervention_parses", False, str(exc))
        return result

    try:
        expected_target_scene = _apply_intervention_extended(source_scene, intervention)
        _record("apply_intervention_ok", True)
    except Exception as exc:  # noqa: BLE001
        _record("apply_intervention_ok", False, str(exc))
        return result

    scene_reproduces = expected_target_scene.to_dict() == target_scene.to_dict()
    _record(
        "target_scene_matches_apply_intervention",
        scene_reproduces,
        {
            "diff_from_expected": None if scene_reproduces else changed_scene_paths(expected_target_scene, target_scene),
        },
    )

    # --- Scene diff analysis: exactly one atom, equal to intervention atom.
    diff_paths = changed_scene_paths(source_scene, target_scene)
    intervention_atom = _normalize_atom(f"{intervention.entity_id}.{_normalize_field(intervention.field)}")

    def _endpoint(diff_path: str) -> str:
        if diff_path.startswith("entities["):
            index_str, _, rest = diff_path[len("entities["):].partition("]")
            try:
                index = int(index_str)
                entity = source_scene.entities[index]
                if rest.startswith("."):
                    rest = rest[1:]
                return _normalize_atom(f"{entity.id}.{rest}")
            except (ValueError, IndexError):
                return diff_path
        return diff_path

    canonical_diff = sorted({_endpoint(p) for p in diff_paths})
    _record("single_field_diff", len(canonical_diff) == 1, {"diff": canonical_diff})
    _record(
        "single_field_diff_matches_intervention",
        len(canonical_diff) == 1 and canonical_diff[0] == intervention_atom,
        {"diff": canonical_diff, "intervention_atom": intervention_atom},
    )

    preservation = pair.get("preservation") or {}
    allowed = [_normalize_atom(str(x)) for x in preservation.get("allowed_to_change", [])]
    _record("allowed_to_change_single", len(allowed) == 1, {"allowed": allowed})
    _record(
        "allowed_to_change_matches_intervention",
        len(allowed) == 1 and allowed[0] == intervention_atom,
        {"allowed": allowed, "intervention_atom": intervention_atom},
    )

    must_preserve = [_normalize_atom(str(x)) for x in preservation.get("must_preserve", [])]

    source_atoms = set(_scene_atoms(source_scene))
    target_atoms = set(_scene_atoms(target_scene))
    # An atom present in EITHER source or target is real; must_preserve is
    # allowed to reference either side. This is a superset used for the
    # extra-atom (spurious atom) check.
    real_atoms = source_atoms | target_atoms
    # Atoms that exist in BOTH source and target and thus SHOULD be preserved
    # unless they are the intervention atom.
    common_atoms = source_atoms & target_atoms
    common_atoms.discard(intervention_atom)

    missing_from_preserve = sorted(common_atoms - set(must_preserve))
    extra_in_preserve = sorted(set(must_preserve) - real_atoms)  # atoms that do not exist anywhere
    intervention_atom_leaked = intervention_atom in set(must_preserve)
    allowed_leaked_into_preserve = any(a in must_preserve for a in allowed)

    # --- Hard gates for must_preserve completeness. ---
    _record(
        "must_preserve_no_intervention_atom",
        not intervention_atom_leaked,
        {"intervention_atom": intervention_atom, "leak": intervention_atom_leaked},
    )
    _record(
        "must_preserve_no_allowed_change_overlap",
        not allowed_leaked_into_preserve,
        {"allowed": allowed, "overlap": [a for a in allowed if a in must_preserve]},
    )
    _record(
        "must_preserve_covers_all_shared_atoms",
        len(missing_from_preserve) == 0,
        {"missing_atoms": missing_from_preserve},
    )
    _record(
        "must_preserve_no_spurious_atoms",
        len(extra_in_preserve) == 0,
        {"spurious_atoms": extra_in_preserve, "real_atoms_sample": sorted(real_atoms)[:20]},
    )
    _record(
        "must_preserve_present",
        len(must_preserve) >= 1,
        {"num": len(must_preserve)},
    )

    # --- Region / relation warnings (do NOT affect errors). ---
    intervention_type = intervention.intervention_type
    if intervention_type == "spatial_position_change":
        old_v = str(intervention.old_value)
        new_v = str(intervention.new_value)
        if old_v not in CANONICAL_REGIONS or new_v not in CANONICAL_REGIONS:
            result["warnings"].append({
                "kind": "non_canonical_region_semantics",
                "old_value": old_v,
                "new_value": new_v,
                "note": "evaluator must special-case this pair; native loss unchanged",
            })
    elif intervention_type == "spatial_relation_change":
        pair_tuple = (str(intervention.old_value), str(intervention.new_value))
        result["warnings"].append({
            "kind": (
                "known_relation_pair_needs_evaluator_handling"
                if pair_tuple in CANONICAL_RELATION_PAIRS
                else "unexpected_relation_pair"
            ),
            "old_value": intervention.old_value,
            "new_value": intervention.new_value,
        })

    # cot_text vs render_cot_text (warning only).
    recomputed_source_text = render_cot_text(source_scene)
    recomputed_target_text = render_cot_text(target_scene)
    if source_info.get("cot_text") != recomputed_source_text:
        result["warnings"].append({"kind": "source_cot_text_mismatch"})
    if target_info.get("cot_text") != recomputed_target_text:
        result["warnings"].append({"kind": "target_cot_text_mismatch"})

    return result


# ---------------------------------------------------------------------------
# Manifest-level orchestration
# ---------------------------------------------------------------------------

def _audit_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")

    records = load_jsonl(manifest_path)
    manifest_dir = manifest_path.parent

    ids = [str(r.get("pair_id") or "") for r in records]
    duplicates = sorted({pid for pid in ids if ids.count(pid) > 1 and pid})

    type_counts: Dict[str, int] = {}
    per_pair_results: List[Dict[str, Any]] = []
    for pair in records:
        t = str((pair.get("intervention") or {}).get("intervention_type", "unknown"))
        type_counts[t] = type_counts.get(t, 0) + 1
        per_pair_results.append(_audit_pair(pair, manifest_dir))

    num_pass = sum(1 for r in per_pair_results if not r["errors"])
    num_fail = sum(1 for r in per_pair_results if r["errors"])
    num_warn = sum(1 for r in per_pair_results if r["warnings"])

    must_preserve_stats = {
        "num_intervention_atom_leak": sum(
            1 for r in per_pair_results
            if not r["checks"].get("must_preserve_no_intervention_atom", {}).get("ok", True)
        ),
        "num_allowed_overlap": sum(
            1 for r in per_pair_results
            if not r["checks"].get("must_preserve_no_allowed_change_overlap", {}).get("ok", True)
        ),
        "num_missing_atoms": sum(
            1 for r in per_pair_results
            if not r["checks"].get("must_preserve_covers_all_shared_atoms", {}).get("ok", True)
        ),
        "num_spurious_atoms": sum(
            1 for r in per_pair_results
            if not r["checks"].get("must_preserve_no_spurious_atoms", {}).get("ok", True)
        ),
    }

    report: Dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "num_records": len(records),
        "duplicate_pair_ids": duplicates,
        "intervention_type_counts": dict(sorted(type_counts.items())),
        "num_pass": num_pass,
        "num_fail": num_fail,
        "num_warn": num_warn,
        "must_preserve_stats": must_preserve_stats,
        "overall_status": "passed" if num_fail == 0 and not duplicates else "failed",
        "created_at_iso": datetime.now(tz=timezone.utc).isoformat(),
        "pairs": per_pair_results,
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=FIXED_GOLD40_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
            "outputs/gold40_full_audit/gold40_full_audit.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = _audit_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[gold40_full_manifest_audit] wrote report -> {args.output}")
    print(
        f"[gold40_full_manifest_audit] pass={report['num_pass']} "
        f"fail={report['num_fail']} warn={report['num_warn']} "
        f"types={report['intervention_type_counts']}"
    )
    print(f"[gold40_full_manifest_audit] must_preserve_stats={report['must_preserve_stats']}")
    return 0 if report["overall_status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
