#!/usr/bin/env python3
"""Run BAGEL's official GenEval evaluator under mmdet 3.3.0.

Two incompatibilities have to be bridged, everything else is the repo's own code
(thresholds, NMS, ``evaluate()`` clause logic) reused verbatim:

1. Weight naming. The checkpoint downloaded by
   ``eval/gen/geneval/evaluation/download_models.sh`` is the 2022 mmdet-2.x
   release. mmdet 3.x renamed the Mask2Former submodules:
       ...attentions.0.attn.*   -> ...self_attn.attn.*
       ...attentions.1.attn.*   -> ...cross_attn.attn.*
       ...attentions.0.<leaf>   -> ...self_attn.<leaf>      (pixel decoder)
       ...ffns.0.layers.*       -> ...ffn.layers.*
   The remap is applied then loaded with ``strict=True`` so a silent mismatch
   cannot slip through.

2. Return type. mmdet 2.x returned ``(bboxes_per_class, masks_per_class)`` as
   lists indexed by class; mmdet 3.x returns a ``DetDataSample``. The official
   script indexes ``bbox[class_index]``, so the sample is converted back into
   that legacy layout, ordered by the class list in ``object_names.txt``.

This module monkeypatches ``init_detector`` / ``inference_detector`` inside
``evaluate_images`` and then calls its ``main``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

BAGEL_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = BAGEL_ROOT / "eval/gen/geneval/evaluation"

DEFAULT_MODEL_PATH = BAGEL_ROOT / "eval/gen/geneval/model"
DEFAULT_CONFIG = (
    BAGEL_ROOT
    / "eval/gen/geneval/mmdet_configs/configs/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco.py"
)


def remap_key(key: str) -> str:
    """mmdet 2.x (2022 ckpt) -> mmdet 3.x Mask2Former key names.

    The 0/1 ordering of ``attentions`` is NOT self/cross as one would assume:
    it was verified empirically on real photographs. Mapping ``attentions.0`` to
    ``self_attn`` yields all-zero confidences; only the assignment below produces
    sane detections (e.g. 0.97 bed / 0.95 book on a bedroom photo).
    """
    if ".attentions.0.attn." in key:
        key = key.replace(".attentions.0.attn.", ".cross_attn.attn.")
    elif ".attentions.1.attn." in key:
        key = key.replace(".attentions.1.attn.", ".self_attn.attn.")
    elif ".attentions.0." in key:
        key = key.replace(".attentions.0.", ".self_attn.")
    if ".ffns.0.layers." in key:
        key = key.replace(".ffns.0.layers.", ".ffn.layers.")
    return key


def build_remapped_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(str(path), map_location="cpu")
    raw = payload.get("state_dict", payload)
    return {remap_key(k): v for k, v in raw.items()}


def make_init_detector(model_path: Path):
    from mmdet.apis import init_detector as _init

    def init_detector(config: str, checkpoint: str, device: str = "cuda:0"):
        if not checkpoint:
            return _init(config, checkpoint, device=device)
        state = build_remapped_state(Path(checkpoint))
        model = _init(config, None, device=device)
        missing, unexpected = model.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"strict load failed: missing={len(missing)} unexpected={len(unexpected)}"
            )
        print(
            json.dumps(
                {"event": "detector_loaded", "tensors": len(state), "remapped": True}
            ),
            flush=True,
        )
        return model

    return init_detector


def make_inference_detector():
    from mmdet.apis import inference_detector as _infer
    from mmdet.structures import DetDataSample

    def inference_detector(model, filepath):
        """Return the mmdet-2.x layout: (bboxes_per_class, masks_per_class)."""
        result = _infer(model, filepath)
        sample = result[0] if isinstance(result, (list, tuple)) else result
        if not isinstance(sample, DetDataSample):
            return result

        instances = sample.pred_instances
        bboxes = instances.bboxes.cpu().numpy()
        scores = instances.scores.cpu().numpy()
        labels = instances.labels.cpu().numpy()
        if hasattr(instances, "masks") and instances.masks is not None:
            masks = instances.masks.cpu().numpy().astype(bool)
        else:
            masks = None

        # model class order (COCO) -> index used by the official script
        names = list(getattr(model, "dataset_meta", {}).get("classes", []))
        n_classes = len(ei_classnames)
        by_class: list[list[np.ndarray]] = [[] for _ in range(n_classes)]
        seg_by_class: list[list[np.ndarray]] = [[] for _ in range(n_classes)]
        for i in range(len(bboxes)):
            name = names[labels[i]] if labels[i] < len(names) else None
            try:
                idx = ei_classnames.index(name)
            except ValueError:
                continue
            by_class[idx].append(
                np.array(
                    [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], scores[i]],
                    dtype=np.float32,
                )
            )
            if masks is not None:
                seg_by_class[idx].append(masks[i])
        bbox_out = [np.stack(v) if v else np.zeros((0, 5), dtype=np.float32) for v in by_class]
        segm_out = [np.stack(v) if v else np.zeros((0, 0), dtype=bool) for v in seg_by_class]
        return bbox_out, segm_out

    return inference_detector


ei_classnames: list[str] = []


def main() -> None:
    global ei_classnames
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("imagedir", type=str)
    parser.add_argument("--outfile", type=str, default="results.jsonl")
    parser.add_argument("--model-config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--options", nargs="*", type=str, default=[])
    args = parser.parse_args()
    args.options = dict(opt.split("=", 1) for opt in args.options)

    sys.path.insert(0, str(EVAL_DIR))
    import evaluate_images as ei  # noqa: E402

    with open(EVAL_DIR / "object_names.txt") as handle:
        ei_classnames = [line.strip() for line in handle]

    ei.init_detector = make_init_detector(Path(args.model_path))
    ei.inference_detector = make_inference_detector()
    ei.args = args

    # replicate the module-level block that evaluate_images runs under __main__
    ei.object_detector, (ei.clip_model, ei.transform, ei.tokenizer), loaded_names = ei.load_models(
        args
    )
    if loaded_names != ei_classnames:
        raise RuntimeError("class name list mismatch between object_names.txt and load_models")
    ei.classnames = loaded_names
    ei.THRESHOLD = float(args.options.get("threshold", 0.3))
    ei.COUNTING_THRESHOLD = float(args.options.get("counting_threshold", 0.9))
    ei.MAX_OBJECTS = int(args.options.get("max_objects", 16))
    ei.NMS_THRESHOLD = float(args.options.get("max_overlap", 1.0))
    ei.POSITION_THRESHOLD = float(args.options.get("position_threshold", 0.1))

    ei.main(args)


if __name__ == "__main__":
    main()
