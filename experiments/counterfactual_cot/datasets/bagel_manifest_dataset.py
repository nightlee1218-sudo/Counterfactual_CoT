from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image

from data.data_utils import pil_img2rgb
from data.distributed_iterable_dataset import DistributedIterableDataset
from experiments.counterfactual_cot.data.render_prompts import paired_prompt, standard_prompt
from experiments.counterfactual_cot.data.schema import load_jsonl


class CounterfactualCoTDataset(DistributedIterableDataset):
    """BAGEL-compatible JSONL dataset for counterfactual CoT experiments.

    Each yielded sample follows BAGEL's existing `PackedDataset` contract:
    `image_tensor_list`, `text_ids_list`, `sequence_plan`, `num_tokens`, and
    `data_indexes`.  No BAGEL model internals are reimplemented here.
    """

    VALID_VARIANTS = {"standard_cot", "independent_cf", "paired_cf_mvp", "paired_cf_generation"}

    def __init__(
        self,
        dataset_name: str,
        transform,
        tokenizer,
        manifest_path: str,
        variant: str = "paired_cf_mvp",
        include_source_image: bool = True,
        training_seed: int = 42,
        local_rank: int = 0,
        world_size: int = 1,
        num_workers: int = 8,
        data_status=None,
        vit_transform=None,
        num_used_data=None,
        max_records: Optional[int] = None,
    ) -> None:
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VALID_VARIANTS)}, got {variant!r}")
        self.transform = transform
        self.vit_transform = vit_transform
        self.tokenizer = tokenizer
        self.variant = variant
        self.include_source_image = include_source_image
        self.training_seed = int(training_seed)
        self.data_status = data_status
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.records = load_jsonl(self.manifest_path)
        if max_records is not None:
            self.records = self.records[: int(max_records)]
        if not self.records:
            raise ValueError(f"Manifest contains no records: {manifest_path}")
        self.indices = list(range(len(self.records)))
        self.current_epoch = 0
        self.set_epoch(0)

    def set_epoch(self, seed: int = 0) -> None:
        self.current_epoch = int(seed)
        indices = list(self.indices)
        rng = random.Random(self.training_seed + self.current_epoch)
        rng.shuffle(indices)
        self.data_paths_per_rank = indices[self.local_rank::self.world_size]
        self.num_files_per_rank = len(self.data_paths_per_rank)
        if self.num_files_per_rank == 0:
            raise ValueError(
                f"{self.dataset_name}: rank {self.local_rank} received no records; "
                f"records={len(indices)}, world_size={self.world_size}"
            )

    def get_data_paths(self, *args, **kwargs):
        return self.indices

    def get_data_paths_per_worker(self):
        info = torch.utils.data.get_worker_info()
        if info is None:
            return self.data_paths_per_rank, 0
        return self.data_paths_per_rank[info.id::info.num_workers], info.id

    @staticmethod
    def _resume_start_idx(data_status: Any, worker_id: int) -> int:
        if data_status is None or worker_id not in data_status:
            return 0
        status = data_status[worker_id]
        if isinstance(status, dict) and "data_indexes" in status:
            status = status["data_indexes"]
        if isinstance(status, (list, tuple)) and len(status) >= 2:
            return int(status[1]) + 1
        return int(status) + 1

    def _resolve(self, path_value: str, image_root: Optional[str] = None) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        if image_root:
            rooted_path = Path(image_root) / path
            if rooted_path.exists():
                return rooted_path
        return self.root / path

    def _load_image(self, path_value: str, image_root: Optional[str] = None) -> Image.Image:
        path = self._resolve(path_value, image_root=image_root)
        return pil_img2rgb(Image.open(path))

    def _add_text(self, data: Dict[str, Any], text: str, need_loss: bool = False, enable_cfg: bool = True) -> Dict[str, Any]:
        text_ids = self.tokenizer.encode(text)
        data["text_ids_list"].append(text_ids)
        data["num_tokens"] += len(text_ids)
        data["sequence_plan"].append(
            {
                "type": "text",
                "enable_cfg": int(enable_cfg),
                "loss": int(need_loss),
                "special_token_loss": 0,
                "special_token_label": None,
            }
        )
        return data

    def _add_vae_image(self, data: Dict[str, Any], image: Image.Image, need_loss: bool, enable_cfg: bool) -> Dict[str, Any]:
        image_tensor = self.transform(image)
        height, width = image_tensor.shape[1:]
        data["num_tokens"] += width * height // self.transform.stride ** 2
        data["image_tensor_list"].append(image_tensor)
        data["sequence_plan"].append(
            {
                "type": "vae_image",
                "enable_cfg": int(enable_cfg),
                "loss": int(need_loss),
                "special_token_loss": 0,
                "special_token_label": None,
            }
        )
        return data

    def _add_vit_image(self, data: Dict[str, Any], image: Image.Image, enable_cfg: bool) -> Dict[str, Any]:
        if self.vit_transform is None:
            return data
        image_tensor = self.vit_transform(image)
        height, width = image_tensor.shape[1:]
        data["num_tokens"] += width * height // self.vit_transform.stride ** 2
        data["image_tensor_list"].append(image_tensor)
        data["sequence_plan"].append(
            {
                "type": "vit_image",
                "enable_cfg": int(enable_cfg),
                "loss": 0,
                "special_token_loss": 0,
                "special_token_label": None,
            }
        )
        return data

    @staticmethod
    def _init_sample() -> Dict[str, Any]:
        return {"sequence_plan": [], "text_ids_list": [], "image_tensor_list": [], "num_tokens": 0}

    def _standard_sample(self, pair: Dict[str, Any], side: str) -> Tuple[Dict[str, Any], str, str]:
        data = self._init_sample()
        entry = pair[side]
        image_root = pair.get("final_image_root")
        prompt = standard_prompt(entry["prompt"], entry["cot_text"])
        data = self._add_text(data, prompt, need_loss=False, enable_cfg=True)
        data = self._add_vae_image(data, self._load_image(entry["image"], image_root=image_root), need_loss=True, enable_cfg=False)
        return data, side, "standard_cot"

    def _independent_edit_sample(self, pair: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
        data = self._init_sample()
        image_root = pair.get("final_image_root")
        source_image = self._load_image(pair["source"]["image"], image_root=image_root)
        data = self._add_vae_image(data, source_image, need_loss=False, enable_cfg=True)
        data = self._add_vit_image(data, source_image, enable_cfg=True)
        prompt = standard_prompt(pair["target"]["prompt"], pair["target"]["cot_text"])
        data = self._add_text(data, prompt, need_loss=False, enable_cfg=True)
        data = self._add_vae_image(data, self._load_image(pair["target"]["image"], image_root=image_root), need_loss=True, enable_cfg=False)
        return data, "target", "independent_cf"

    def _paired_sample(self, pair: Dict[str, Any]) -> Tuple[Dict[str, Any], str, str]:
        data = self._init_sample()
        image_root = pair.get("final_image_root")
        include_source = self.include_source_image and self.variant == "paired_cf_mvp"
        if include_source:
            source_image = self._load_image(pair["source"]["image"], image_root=image_root)
            data = self._add_vae_image(data, source_image, need_loss=False, enable_cfg=True)
            data = self._add_vit_image(data, source_image, enable_cfg=True)
        prompt = pair.get("training_prompts", {}).get("paired_edit" if include_source else "paired_generation")
        if not prompt:
            prompt = paired_prompt(pair, include_source_image=include_source)
        data = self._add_text(data, prompt, need_loss=False, enable_cfg=True)
        data = self._add_vae_image(data, self._load_image(pair["target"]["image"], image_root=image_root), need_loss=True, enable_cfg=False)
        return data, "target", self.variant

    def _make_sample(self, pair: Dict[str, Any], sample_index: int, epoch: int) -> Tuple[Dict[str, Any], str, str]:
        if self.variant == "standard_cot":
            return self._standard_sample(pair, "source")
        if self.variant == "independent_cf":
            return self._independent_edit_sample(pair)
        return self._paired_sample(pair)

    def __iter__(self):
        indices_per_worker, worker_id = self.get_data_paths_per_worker()
        start_idx = self._resume_start_idx(self.data_status, worker_id)
        epoch = self.current_epoch
        while True:
            for step, sample_index in enumerate(indices_per_worker[start_idx:], start=start_idx):
                pair = self.records[sample_index]
                try:
                    data, side, mode = self._make_sample(pair, sample_index, epoch)
                except Exception as exc:
                    print(f"[{self.dataset_name}] failed to build pair {pair.get('pair_id')}: {exc}")
                    continue
                data["data_indexes"] = {
                    "data_indexes": [sample_index, step],
                    "worker_id": worker_id,
                    "dataset_name": self.dataset_name,
                    "pair_id": pair.get("pair_id", ""),
                    "mode": mode,
                    "side": side,
                }
                yield data
            start_idx = 0
            epoch += 1
            self.set_epoch(epoch)
            indices_per_worker, worker_id = self.get_data_paths_per_worker()
            print(f"[{self.dataset_name}] worker-{worker_id} completed epoch {epoch - 1}")
