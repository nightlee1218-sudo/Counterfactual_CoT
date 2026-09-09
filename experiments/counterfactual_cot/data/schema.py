from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SUPPORTED_INTERVENTION_TYPES = {
    "object_replace",
    "attribute_replace",
    "count_change",
    "spatial_relation_change",
    "spatial_position_change",
}

SUPPORTED_OPERATIONS = {"replace", "set"}


@dataclass(frozen=True)
class SceneEntity:
    id: str
    category: str
    count: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None
    relations: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneEntity":
        if not isinstance(data, dict):
            raise ValueError("Scene entity must be an object")
        entity_id = str(data.get("id", "")).strip()
        category = str(data.get("category", "")).strip()
        if not entity_id:
            raise ValueError("Scene entity is missing id")
        if not category:
            raise ValueError(f"Scene entity {entity_id!r} is missing category")
        count = data.get("count")
        if count is not None:
            count = int(count)
            if count < 0:
                raise ValueError(f"Scene entity {entity_id!r} has negative count")
        attributes = data.get("attributes") or {}
        if not isinstance(attributes, dict):
            raise ValueError(f"Scene entity {entity_id!r} attributes must be an object")
        relations = data.get("relations") or []
        if not isinstance(relations, list):
            raise ValueError(f"Scene entity {entity_id!r} relations must be a list")
        location = data.get("location")
        return cls(
            id=entity_id,
            category=category,
            count=count,
            attributes=dict(attributes),
            location=None if location is None else str(location),
            relations=list(relations),
        )

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"id": self.id, "category": self.category}
        if self.count is not None:
            data["count"] = self.count
        if self.attributes:
            data["attributes"] = self.attributes
        if self.location is not None:
            data["location"] = self.location
        if self.relations:
            data["relations"] = self.relations
        return data


@dataclass(frozen=True)
class SceneGraph:
    entities: List[SceneEntity]
    background: Dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneGraph":
        if not isinstance(data, dict):
            raise ValueError("Scene graph must be an object")
        entities_raw = data.get("entities")
        if not isinstance(entities_raw, list) or not entities_raw:
            raise ValueError("Scene graph must contain a non-empty entities list")
        entities = [SceneEntity.from_dict(item) for item in entities_raw]
        entity_ids = [entity.id for entity in entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Scene graph contains duplicate entity ids")
        background = data.get("background") or {}
        if not isinstance(background, dict):
            raise ValueError("Scene graph background must be an object")
        return cls(
            entities=entities,
            background=dict(background),
            schema_version=str(data.get("schema_version", "1.0")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "entities": [entity.to_dict() for entity in self.entities],
            "background": self.background,
        }

    def find_entity_index(self, entity_id: str) -> int:
        for index, entity in enumerate(self.entities):
            if entity.id == entity_id:
                return index
        raise ValueError(f"Entity {entity_id!r} not found in scene graph")


@dataclass(frozen=True)
class Intervention:
    operation: str
    intervention_type: str
    entity_id: str
    field: str
    old_value: Any
    new_value: Any

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intervention":
        if not isinstance(data, dict):
            raise ValueError("Intervention must be an object")
        operation = str(data.get("operation", "")).strip()
        intervention_type = str(data.get("intervention_type", "")).strip()
        entity_id = str(data.get("entity_id", "")).strip()
        field_name = str(data.get("field", "")).strip()
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported intervention operation: {operation!r}")
        if intervention_type not in SUPPORTED_INTERVENTION_TYPES:
            raise ValueError(f"Unsupported intervention type: {intervention_type!r}")
        if not entity_id:
            raise ValueError("Intervention is missing entity_id")
        if not field_name:
            raise ValueError("Intervention is missing field")
        if "old_value" not in data or "new_value" not in data:
            raise ValueError("Intervention must contain old_value and new_value")
        return cls(
            operation=operation,
            intervention_type=intervention_type,
            entity_id=entity_id,
            field=field_name,
            old_value=data["old_value"],
            new_value=data["new_value"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "intervention_type": self.intervention_type,
            "entity_id": self.entity_id,
            "field": self.field,
            "old_value": self.old_value,
            "new_value": self.new_value,
        }


@dataclass(frozen=True)
class PreservationSpec:
    must_preserve: List[str] = field(default_factory=list)
    allowed_to_change: List[str] = field(default_factory=list)
    must_revalidate: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PreservationSpec":
        data = data or {}
        return cls(
            must_preserve=[str(item) for item in data.get("must_preserve", [])],
            allowed_to_change=[str(item) for item in data.get("allowed_to_change", [])],
            must_revalidate=[str(item) for item in data.get("must_revalidate", [])],
        )

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "must_preserve": list(self.must_preserve),
            "allowed_to_change": list(self.allowed_to_change),
            "must_revalidate": list(self.must_revalidate),
        }


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def render_cot_text(scene: SceneGraph) -> str:
    parts: List[str] = []
    for entity in scene.entities:
        prefix = entity.id
        parts.append(f"{prefix}.object={entity.category}")
        if entity.count is not None:
            parts.append(f"{prefix}.count={entity.count}")
        if entity.location is not None:
            parts.append(f"{prefix}.location={entity.location}")
        for key in sorted(entity.attributes):
            parts.append(f"{prefix}.attribute.{key}={entity.attributes[key]}")
        for relation_index, relation in enumerate(entity.relations):
            predicate = relation.get("predicate", "")
            target = relation.get("target", "")
            parts.append(f"{prefix}.relation.{relation_index}={predicate}:{target}")
    for key in sorted(scene.background):
        parts.append(f"background.{key}={scene.background[key]}")
    return "; ".join(parts)


def get_field(scene: SceneGraph, entity_id: str, field_path: str) -> Any:
    entity = scene.entities[scene.find_entity_index(entity_id)]
    if field_path == "category":
        return entity.category
    if field_path == "count":
        return entity.count
    if field_path == "location":
        return entity.location
    if field_path.startswith("attributes."):
        key = field_path.split(".", 1)[1]
        return entity.attributes.get(key)
    if field_path == "relations":
        return entity.relations
    raise ValueError(f"Unsupported intervention field: {field_path!r}")


def set_field(scene: SceneGraph, entity_id: str, field_path: str, value: Any) -> SceneGraph:
    scene_dict = copy.deepcopy(scene.to_dict())
    entity_index = SceneGraph.from_dict(scene_dict).find_entity_index(entity_id)
    entity_dict = scene_dict["entities"][entity_index]
    if field_path == "category":
        entity_dict["category"] = value
    elif field_path == "count":
        entity_dict["count"] = int(value)
    elif field_path == "location":
        entity_dict["location"] = value
    elif field_path.startswith("attributes."):
        key = field_path.split(".", 1)[1]
        entity_dict.setdefault("attributes", {})[key] = value
    elif field_path == "relations":
        if not isinstance(value, list):
            raise ValueError("relations intervention value must be a list")
        entity_dict["relations"] = value
    else:
        raise ValueError(f"Unsupported intervention field: {field_path!r}")
    return SceneGraph.from_dict(scene_dict)


def apply_intervention(source_scene: SceneGraph, intervention: Intervention) -> SceneGraph:
    current_value = get_field(source_scene, intervention.entity_id, intervention.field)
    if current_value != intervention.old_value:
        raise ValueError(
            f"Intervention old_value mismatch for {intervention.entity_id}.{intervention.field}: "
            f"expected {intervention.old_value!r}, found {current_value!r}"
        )
    return set_field(source_scene, intervention.entity_id, intervention.field, intervention.new_value)


def changed_scene_paths(source: SceneGraph, target: SceneGraph) -> List[str]:
    source_dict = source.to_dict()
    target_dict = target.to_dict()
    changes: List[str] = []

    def walk(prefix: str, left: Any, right: Any) -> None:
        if left == right:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                walk(f"{prefix}.{key}" if prefix else str(key), left.get(key), right.get(key))
            return
        if isinstance(left, list) and isinstance(right, list):
            max_len = max(len(left), len(right))
            for index in range(max_len):
                l_item = left[index] if index < len(left) else None
                r_item = right[index] if index < len(right) else None
                walk(f"{prefix}[{index}]", l_item, r_item)
            return
        changes.append(prefix)

    walk("", source_dict, target_dict)
    return changes


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
