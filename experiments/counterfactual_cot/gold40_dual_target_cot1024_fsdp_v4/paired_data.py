"""Reviewed Gold40 source_state/target_state manifest contract and BAGEL branches."""
from __future__ import annotations

import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from data.transforms import ImageTransform
from experiments.counterfactual_cot.datasets.bagel_manifest_dataset import CounterfactualCoTDataset
from experiments.counterfactual_cot.training.gold40_overfit import _transform_args

STATE_IDS = ("source_state", "target_state")
PROMPT_PROFILE = "gold40_source_condition_state_target_cot_v1"
MODE = "[MODE=GOLD40_SOURCE_TARGET_STATE]"


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _roi(value: Any, label: str) -> List[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{label} must be [left, top, right, bottom]")
    values = [int(item) for item in value]
    left, top, right, bottom = values
    if not (0 <= left < right <= 1024 and 0 <= top < bottom <= 1024):
        raise ValueError(f"{label} out of 1024x1024 bounds: {values}")
    return values


def _branches(record: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    branches = record.get("paired_branches")
    if not isinstance(branches, list) or len(branches) != 2:
        raise ValueError("paired_branches must contain exactly two records")
    by_id = {item.get("branch_id"): item for item in branches if isinstance(item, Mapping)}
    if set(by_id) != set(STATE_IDS):
        raise ValueError("paired_branches must contain source_state and target_state exactly once")
    return by_id


def review_and_regions(record: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    review = record.get("review")
    if (
        not isinstance(review, Mapping)
        or review.get("approved") is not True
        or review.get("status") != "approved"
    ):
        raise ValueError("record review must explicitly be approved")
    if record.get("training_readiness", {}).get("ready_for_training") is not True:
        raise ValueError("record training_readiness.ready_for_training must be true")
    candidate_mask = record.get("dynamic_candidate_mask")
    if not isinstance(candidate_mask, Mapping) or candidate_mask.get("candidate_only") is not True:
        raise ValueError("record must retain its dynamic_candidate_mask")
    if candidate_mask.get("review_status") != "approved":
        raise ValueError("dynamic_candidate_mask.review_status must be approved")
    mask_path = Path(_text(candidate_mask.get("path"), "dynamic_candidate_mask.path"))
    if not mask_path.is_file():
        raise FileNotFoundError(f"reviewed dynamic candidate mask is missing: {mask_path}")
    regions = record.get("dynamic_regions")
    if not isinstance(regions, Mapping):
        raise ValueError("dynamic_regions is missing")
    reviewed_masks = []
    for field in ("source_changed_roi", "target_changed_roi", "shared_changed_roi"):
        value = regions.get(field)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"dynamic_regions.{field} must be an approved mask object")
        path = Path(_text(value.get("path"), f"dynamic_regions.{field}.path"))
        digest = value.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"dynamic_regions.{field}.sha256 must be a SHA256 hex digest")
        if not path.is_file():
            raise FileNotFoundError(f"reviewed ROI mask is missing: {path}")
        reviewed_masks.append(field)
    if not reviewed_masks:
        raise ValueError("dynamic_regions must contain at least one reviewed binary ROI mask")
    if not (set(reviewed_masks) & {"source_changed_roi", "shared_changed_roi"}):
        raise ValueError("source state has no reviewed changed-region support")
    if not (set(reviewed_masks) & {"target_changed_roi", "shared_changed_roi"}):
        raise ValueError("target state has no reviewed changed-region support")
    return review, regions


def validate_record(record: Mapping[str, Any], *, require_files: bool = True) -> Dict[str, Any]:
    pair_id = _text(record.get("pair_id"), "pair_id")
    if record.get("macro_pair_id") != f"dual_cot::{pair_id}":
        raise ValueError(f"{pair_id}: macro_pair_id must equal dual_cot::<pair_id>")
    _text(record.get("source_scene_id"), f"{pair_id}.source_scene_id")
    intervention = record.get("intervention")
    if not isinstance(intervention, Mapping) or not _text(intervention.get("intervention_type"), f"{pair_id}.intervention_type"):
        raise ValueError(f"{pair_id}: intervention_type missing")
    review, regions = review_and_regions(record)
    branches = _branches(record)
    source, target = record.get("source"), record.get("target")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise ValueError(f"{pair_id}: source and target records are required")
    for state_id, state in (("source_state", source), ("target_state", target)):
        branch = branches[state_id]
        if branch.get("target_state") != ("source" if state_id == "source_state" else "target"):
            raise ValueError(f"{pair_id}: {state_id}.target_state mismatch")
        for field, branch_field in (("image", "target_image"), ("cot_text", "target_cot_text")):
            expected, actual = _text(state.get(field), f"{pair_id}.{state_id}.{field}"), _text(branch.get(branch_field), f"{pair_id}.{state_id}.{branch_field}")
            if expected != actual:
                raise ValueError(f"{pair_id}: {state_id} {branch_field} differs from nested {field}")
        _text(state.get("prompt"), f"{pair_id}.{state_id}.prompt")
        if require_files and not Path(state["image"]).is_file():
            raise FileNotFoundError(f"{pair_id}: missing {state_id} image {state['image']}")
    return {
        "pair_id": pair_id,
        "source_scene_id": record["source_scene_id"],
        "intervention_type": intervention["intervention_type"],
        "template_family": record.get("template_family"),
        "reviewer": review.get("reviewer"),
        "regions": {
            key: regions[key]
            for key in ("source_changed_roi", "target_changed_roi", "shared_changed_roi")
            if regions.get(key) is not None
        },
    }


def load_reviewed_manifest(path: Path, *, require_files: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    if not records:
        raise ValueError(f"{path}: no manifest records")
    seen, audits = set(), []
    for record in records:
        audit = validate_record(record, require_files=require_files)
        if audit["pair_id"] in seen:
            raise ValueError(f"duplicate pair_id: {audit['pair_id']}")
        seen.add(audit["pair_id"])
        audits.append(audit)
    return records, audits


def objective_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Project the reviewed split contract onto the audited objective's stable API.

    The split keeps review and dynamic-region metadata nested; the objective also
    accepts its legacy top-level projections.  Copying prevents any mutation of the
    provenance-bearing manifest record.
    """
    validate_record(record, require_files=True)
    projected = copy.deepcopy(dict(record))
    projected["human_review"] = {**projected["review"], **projected["dynamic_regions"]}
    return projected


def render_state_prompt(record: Mapping[str, Any], state_id: str) -> str:
    branch = _branches(record)[state_id]
    source = record["source"]
    return "\n".join((MODE, f"Source Prompt: {_text(source['prompt'], 'source.prompt')}", f"Source CoT: {_text(source['cot_text'], 'source.cot_text')}", f"Target CoT: {_text(branch['target_cot_text'], state_id + '.target_cot_text')}", "Generate the target image by strictly following Target CoT."))


class Gold40StateDataset(CounterfactualCoTDataset):
    """Build one deterministic source-conditioned target-state branch."""
    def __init__(self, *args: Any, state_id: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if state_id not in STATE_IDS:
            raise ValueError(f"invalid state_id: {state_id}")
        self.state_id = state_id

    def _paired_sample(self, record: Dict[str, Any]):
        data = self._init_sample()
        source = record["source"]
        branch = _branches(record)[self.state_id]
        source_image = self._load_image(source["image"], image_root=record.get("final_image_root"))
        data = self._add_vae_image(data, source_image, need_loss=False, enable_cfg=True)
        data = self._add_vit_image(data, source_image, enable_cfg=True)
        data = self._add_text(data, render_state_prompt(record, self.state_id), need_loss=False, enable_cfg=True)
        data = self._add_vae_image(data, self._load_image(branch["target_image"], image_root=record.get("final_image_root")), need_loss=True, enable_cfg=False)
        return data, self.state_id, f"{PROMPT_PROFILE}:{self.state_id}"


def build_state_dataset(args: Namespace, tokenizer: Any, state_id: str) -> Gold40StateDataset:
    return Gold40StateDataset(
        dataset_name=f"gold40_{state_id}", transform=ImageTransform(**_transform_args(1024, 1024, 16)),
        tokenizer=tokenizer, manifest_path=str(args.manifest), variant="paired_cf_mvp", include_source_image=True,
        training_seed=args.seed, local_rank=0, world_size=1, num_workers=0, data_status=None,
        vit_transform=ImageTransform(**_transform_args(980, 980, 14)), max_records=None, state_id=state_id,
    )


def prompt_audit(records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    return {record["pair_id"]: {state: hashlib.sha256(render_state_prompt(record, state).encode()).hexdigest() for state in STATE_IDS} for record in records}
