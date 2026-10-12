from __future__ import annotations

import hashlib
from argparse import Namespace
from typing import Any, Dict, Tuple

from data.transforms import ImageTransform
from experiments.counterfactual_cot.datasets.bagel_manifest_dataset import CounterfactualCoTDataset
from experiments.counterfactual_cot.training.gold40_overfit import _transform_args

PROMPT_PROFILE = "source_prompt_source_cot_target_cot_switch_v1"
BRANCHES: Tuple[str, str] = ("left", "right")
MODE = "[MODE=COUNTERFACTUAL_EDIT_DUAL_TARGET_COT]"
FINAL_INSTRUCTION = "Generate the target image by strictly following Target CoT."


def _required(pair: Dict[str, Any], side: str, field: str) -> str:
    value = pair.get(side, {}).get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing non-empty pair[{side!r}][{field!r}]")
    return value.strip()


def render_dual_target_prompt(pair: Dict[str, Any], cot_side: str) -> str:
    if cot_side not in BRANCHES:
        raise ValueError(f"cot_side must be one of {BRANCHES}, got {cot_side!r}")
    source_prompt = _required(pair, "source", "prompt")
    source_cot = _required(pair, "source", "cot_text")
    target_cot = _required(pair, "source" if cot_side == "left" else "target", "cot_text")
    return (
        f"{MODE}\n"
        f"Source Prompt: {source_prompt}\n"
        f"Source CoT: {source_cot}\n"
        f"Target CoT: {target_cot}\n"
        f"{FINAL_INSTRUCTION}"
    )


def prompt_manifest(pair: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    prompts = {}
    for branch in BRANCHES:
        text = render_dual_target_prompt(pair, branch)
        prompts[branch] = {
            "branch": branch,
            "target_image_side": "source" if branch == "left" else "target",
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    left_lines = prompts["left"]["text"].splitlines()
    right_lines = prompts["right"]["text"].splitlines()
    if len(left_lines) != len(right_lines) or left_lines[:3] != right_lines[:3] or left_lines[-1] != right_lines[-1]:
        raise RuntimeError("Left/right prompts differ outside Target CoT")
    left_cot = _required(pair, "source", "cot_text")
    right_cot = _required(pair, "target", "cot_text")
    left_fields = [field.strip() for field in left_cot.split(";")]
    right_fields = [field.strip() for field in right_cot.split(";")]
    differences = [(left, right) for left, right in zip(left_fields, right_fields) if left != right]
    if len(left_fields) != len(right_fields) or differences != [
        ("ball_1.location=left_region", "ball_1.location=right_region")
    ]:
        raise RuntimeError(f"Target CoTs must differ only in ball_1.location: {differences}")
    return prompts


class DualTargetCoTDataset(CounterfactualCoTDataset):
    """One branch of the same-source, left/right Target-CoT paired objective."""

    def __init__(self, *args: Any, target_side: str, cot_side: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if target_side not in BRANCHES or cot_side not in BRANCHES:
            raise ValueError(f"Invalid branch target={target_side!r}, cot={cot_side!r}")
        self.target_side = target_side
        self.cot_side = cot_side

    def _paired_sample(self, pair: Dict[str, Any]):
        data = self._init_sample()
        image_root = pair.get("final_image_root")
        source_image = self._load_image(pair["source"]["image"], image_root=image_root)
        data = self._add_vae_image(data, source_image, need_loss=False, enable_cfg=True)
        data = self._add_vit_image(data, source_image, enable_cfg=True)
        prompt = render_dual_target_prompt(pair, self.cot_side)
        required = ("Source Prompt:", "Source CoT:", "Target CoT:")
        forbidden = ("Target Prompt:", "Intervention:", "Preserve:")
        if not all(field in prompt for field in required) or any(field in prompt for field in forbidden):
            raise RuntimeError(f"Invalid dual Target-CoT prompt: {prompt}")
        data = self._add_text(data, prompt, need_loss=False, enable_cfg=True)
        target_record_side = "source" if self.target_side == "left" else "target"
        target_image = self._load_image(pair[target_record_side]["image"], image_root=image_root)
        data = self._add_vae_image(data, target_image, need_loss=True, enable_cfg=False)
        return data, self.target_side, f"{PROMPT_PROFILE}:{self.target_side}_target:{self.cot_side}_cot"


def build_branch_dataset(
    args: Namespace,
    tokenizer: Any,
    target_side: str,
    cot_side: str,
) -> DualTargetCoTDataset:
    return DualTargetCoTDataset(
        dataset_name=f"counterfactual_cot_dual_target_{target_side}_target_{cot_side}_cot",
        transform=ImageTransform(**_transform_args(args.image_size, args.image_size, 16)),
        tokenizer=tokenizer,
        manifest_path=str(args.manifest),
        variant=args.variant,
        include_source_image=True,
        training_seed=args.seed,
        local_rank=0,
        world_size=1,
        num_workers=0,
        data_status=None,
        vit_transform=ImageTransform(**_transform_args(args.vit_image_size, args.vit_min_image_size, 14)),
        max_records=1,
        target_side=target_side,
        cot_side=cot_side,
    )
