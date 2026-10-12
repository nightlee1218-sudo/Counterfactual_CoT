"""AutoMask-v5 Gold40 paired-record contract and branch datasets.

This loader deliberately consumes machine-ready AutoMask records.  It neither reads,
requires, nor synthesizes any human-review metadata.
"""
from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from data.transforms import ImageTransform
from experiments.counterfactual_cot.datasets.bagel_manifest_dataset import CounterfactualCoTDataset
from experiments.counterfactual_cot.training.gold40_overfit import _transform_args

STATE_IDS = ("source_state", "target_state")
PROMPT_PROFILE = "gold40_automask_source_condition_state_target_cot_v5"
MODE = "[MODE=GOLD40_AUTOMASK_SOURCE_TARGET_STATE_V5]"


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _branches(record: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    branches = record.get("paired_branches")
    if not isinstance(branches, list) or len(branches) != 2:
        raise ValueError("paired_branches must contain exactly two records")
    by_id = {item.get("branch_id"): item for item in branches if isinstance(item, Mapping)}
    if set(by_id) != set(STATE_IDS):
        raise ValueError("paired_branches must contain source_state and target_state exactly once")
    return by_id


def _automask(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve the sole v5 automatic-difference mask payload."""
    value = record.get("automatic_difference_mask")
    if isinstance(value, Mapping) and isinstance(value.get("path"), str) and value["path"].strip():
        return value
    raise ValueError("AutoMask-v5 record requires automatic_difference_mask metadata")


def validate_record(record: Mapping[str, Any], *, require_files: bool = True) -> Dict[str, Any]:
    pair_id = _text(record.get("pair_id"), "pair_id")
    if record.get("schema_version") != "gold40_automask_v5":
        raise ValueError(f"{pair_id}: schema_version must be gold40_automask_v5")
    forbidden = [field for field in ("review", "human_review", "reviewed_mask", "dynamic_regions", "dynamic_candidate_mask") if field in record]
    if forbidden:
        raise ValueError(f"{pair_id}: AutoMask-v5 must not contain review/candidate fields: {forbidden}")
    if record.get("macro_pair_id") != f"dual_cot::{pair_id}":
        raise ValueError(f"{pair_id}: invalid macro_pair_id")
    source_scene_id = _text(record.get("source_scene_id"), f"{pair_id}.source_scene_id")
    intervention = _mapping(record.get("intervention"), f"{pair_id}.intervention")
    intervention_type = _text(intervention.get("intervention_type"), f"{pair_id}.intervention_type")
    readiness = _mapping(record.get("training_readiness"), f"{pair_id}.training_readiness")
    if readiness.get("ready_for_training") is not True or readiness.get("mode") != "automatic_difference_mask":
        raise ValueError(f"{pair_id}: training_readiness must permit automatic_difference_mask training")
    if readiness.get("human_review_used") is not False:
        raise ValueError(f"{pair_id}: training_readiness.human_review_used must be false")
    provenance = _mapping(record.get("mask_provenance"), f"{pair_id}.mask_provenance")
    expected_provenance = {"kind": "automatic_rgb_difference_mask", "semantic_annotation": False, "state_specific": False}
    if {key: provenance.get(key) for key in expected_provenance} != expected_provenance:
        raise ValueError(f"{pair_id}: invalid automatic mask provenance")
    mask = _automask(record)
    mask_path = Path(_text(mask.get("path", mask.get("mask_path")), f"{pair_id}.automask.path"))
    digest = mask.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{pair_id}: automask.sha256 must be a SHA256 digest")
    if require_files and not mask_path.is_file():
        raise FileNotFoundError(f"{pair_id}: missing AutoMask asset {mask_path}")
    branches = _branches(record)
    source, target = _mapping(record.get("source"), f"{pair_id}.source"), _mapping(record.get("target"), f"{pair_id}.target")
    for state_id, state, target_state in (
        ("source_state", source, "source"),
        ("target_state", target, "target"),
    ):
        branch = branches[state_id]
        if branch.get("target_state") != target_state:
            raise ValueError(f"{pair_id}: {state_id}.target_state mismatch")
        image = _text(state.get("image"), f"{pair_id}.{state_id}.image")
        cot = _text(state.get("cot_text"), f"{pair_id}.{state_id}.cot_text")
        _text(state.get("prompt"), f"{pair_id}.{state_id}.prompt")
        if _text(branch.get("target_image"), f"{pair_id}.{state_id}.target_image") != image:
            raise ValueError(f"{pair_id}: {state_id} target image differs from paired state image")
        if _text(branch.get("target_cot_text"), f"{pair_id}.{state_id}.target_cot_text") != cot:
            raise ValueError(f"{pair_id}: {state_id} target CoT differs from paired state CoT")
        if require_files and not Path(image).is_file():
            raise FileNotFoundError(f"{pair_id}: missing {state_id} image {image}")
    return {
        "pair_id": pair_id,
        "source_scene_id": source_scene_id,
        "intervention_type": intervention_type,
        "automask_path": str(mask_path),
        "automask_sha256": digest,
        "automask_kind": str(mask.get("kind", mask.get("method", "automask"))),
    }


def load_automask_manifest(path: Path, *, require_files: bool = True) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: record must be an object")
            records.append(value)
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


def render_state_prompt(record: Mapping[str, Any], state_id: str) -> str:
    branch = _branches(record)[state_id]
    source = _mapping(record.get("source"), "source")
    return "\n".join((
        MODE,
        f"Source Prompt: {_text(source['prompt'], 'source.prompt')}",
        f"Source CoT: {_text(source['cot_text'], 'source.cot_text')}",
        f"Target CoT: {_text(branch['target_cot_text'], state_id + '.target_cot_text')}",
        "Generate the target image by strictly following Target CoT.",
    ))


class Gold40AutoMaskStateDataset(CounterfactualCoTDataset):
    """One source-conditioned target state, selected only from the current macro pair."""

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
        data = self._add_vae_image(
            data,
            self._load_image(branch["target_image"], image_root=record.get("final_image_root")),
            need_loss=True,
            enable_cfg=False,
        )
        return data, self.state_id, f"{PROMPT_PROFILE}:{self.state_id}"


def build_state_dataset(args: Namespace, tokenizer: Any, state_id: str) -> Gold40AutoMaskStateDataset:
    return Gold40AutoMaskStateDataset(
        dataset_name=f"gold40_automask_v5_{state_id}",
        transform=ImageTransform(**_transform_args(1024, 1024, 16)),
        tokenizer=tokenizer,
        manifest_path=str(args.manifest),
        variant="paired_cf_mvp",
        include_source_image=True,
        training_seed=args.seed,
        local_rank=0,
        world_size=1,
        num_workers=0,
        data_status=None,
        vit_transform=ImageTransform(**_transform_args(980, 980, 14)),
        max_records=None,
        state_id=state_id,
    )


def prompt_audit(records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, str]]:
    return {
        record["pair_id"]: {
            state: hashlib.sha256(render_state_prompt(record, state).encode()).hexdigest()
            for state in STATE_IDS
        }
        for record in records
    }
