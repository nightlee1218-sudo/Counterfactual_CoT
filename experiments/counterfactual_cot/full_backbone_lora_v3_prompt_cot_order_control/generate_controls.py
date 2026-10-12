#!/usr/bin/env python3
"""Generate the three prompt/CoT conditions for v3 step 0 and step 500."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

BAGEL_ROOT = Path(__file__).resolve().parents[3]
if str(BAGEL_ROOT) not in sys.path:
    sys.path.insert(0, str(BAGEL_ROOT))

from experiments.counterfactual_cot.data.schema import load_jsonl  # noqa: E402
from experiments.counterfactual_cot.full_backbone_lora_region1024_v2.train import (  # noqa: E402
    MANIFEST,
    _pick_and_validate_pair,
)
from experiments.counterfactual_cot.full_backbone_lora_v3_prompt_cot_order_control.prompt_conditions import (  # noqa: E402
    CONDITION_IDS,
    condition_manifest,
)
from experiments.counterfactual_cot.full_backbone_lora_v3_strict_step0_step500_control.generate_controls import (  # noqa: E402
    CHECKPOINTS,
    EXPECTED_FORMAT,
    TRAIN_OUTPUT,
    _file_identity,
    _load_checkpoint_identity,
    _validate_pair,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent
EVALUATE = EXPERIMENT_DIR / "evaluate.py"
UPSTREAM_EVALUATOR = BAGEL_ROOT / "experiments/counterfactual_cot/full_backbone_lora_region1024_fsdp_v3/evaluate.py"
V3_TRAIN = BAGEL_ROOT / "experiments/counterfactual_cot/full_backbone_lora_region1024_fsdp_v3/train.py"
DEFAULT_MODEL = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
DEFAULT_OUTPUT = BAGEL_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "cf_singlepair_fullL0-27_lora_r8_promptCotOrder_strict_step0_step500_control_v1"
)
SEEDS = (1234, 2234, 3234)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(temporary, path)


def _tree_identity(root: Path, suffixes: Tuple[str, ...]) -> Dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix in suffixes)
    if not files:
        raise RuntimeError(f"No runtime files found under {root}")
    digest = hashlib.sha256()
    total_size = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        identity = _file_identity(path)
        total_size += int(identity["size_bytes"])
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(identity["sha256"].encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(root.resolve()),
        "num_files": len(files),
        "total_size_bytes": total_size,
        "tree_sha256": digest.hexdigest(),
    }


def _runtime_identities(model_dir: Path) -> Dict[str, Dict[str, Any]]:
    files = {
        "base_checkpoint": model_dir / "ema.safetensors",
        "vae_checkpoint": model_dir / "ae.safetensors",
        "manifest": MANIFEST,
        "root_inferencer": BAGEL_ROOT / "inferencer.py",
        "local_evaluator": EVALUATE,
        "local_prompt_conditions": EXPERIMENT_DIR / "prompt_conditions.py",
        "local_generator": Path(__file__),
        "local_analyzer": EXPERIMENT_DIR / "analyze_controls.py",
        "upstream_evaluator": UPSTREAM_EVALUATOR,
        "v3_train": V3_TRAIN,
    }
    identities: Dict[str, Dict[str, Any]] = {name: _file_identity(path) for name, path in files.items()}
    identities.update({
        "model_metadata_tree": _tree_identity(model_dir, (".json", ".txt", ".model")),
        "data_code_tree": _tree_identity(BAGEL_ROOT / "data", (".py",)),
        "modeling_code_tree": _tree_identity(BAGEL_ROOT / "modeling", (".py",)),
        "counterfactual_common_code_tree": _tree_identity(
            BAGEL_ROOT / "experiments/counterfactual_cot", (".py",)
        ),
        "software": _software_identity(),
    })
    return identities


def _software_identity() -> Dict[str, str]:
    import PIL
    import torch
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "pillow": PIL.__version__,
    }


def _validate_report(
    report_path: Path,
    step: int,
    checkpoint: Dict[str, Any],
    expected_prompts: Dict[str, Dict[str, Any]],
    runtime: Dict[str, Dict[str, Any]],
    run_id: str,
) -> Dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    step_dir = report_path.parent.resolve()
    if report.get("status") != "completed" or int(report.get("step", -1)) != step:
        raise RuntimeError(f"Incomplete or wrong-step report: {report_path}")
    if Path(report.get("checkpoint", "")).resolve() != Path(checkpoint["path"]):
        raise RuntimeError(f"Report checkpoint mismatch at step {step}")
    checkpoint_report = report.get("checkpoint_report", {})
    if int(checkpoint_report.get("step", -1)) != step or checkpoint_report.get("checkpoint_format") != EXPECTED_FORMAT:
        raise RuntimeError(f"Checkpoint report mismatch at step {step}")
    actual_manifest = report.get("prompt_manifest", {})
    expected_text = {condition: expected_prompts[condition]["text"] for condition in CONDITION_IDS}
    if actual_manifest != expected_text:
        raise RuntimeError(f"Prompt manifest mismatch at step {step}")
    generations = report.get("generations", [])
    expected_keys = {(condition, seed) for condition in CONDITION_IDS for seed in SEEDS}
    actual_keys = {(str(row.get("profile")), int(row.get("seed", -1))) for row in generations}
    if len(generations) != 9 or actual_keys != expected_keys:
        raise RuntimeError(f"Generation matrix mismatch at step {step}: {actual_keys ^ expected_keys}")
    files: List[Dict[str, Any]] = []
    for row in generations:
        condition = str(row["profile"])
        if row.get("prompt") != expected_text[condition] or float(row.get("cfg_img_scale", -1)) != 2.0:
            raise RuntimeError(f"Generation condition mismatch at step {step}: {condition}")
        path = Path(row["path"]).resolve()
        if step_dir not in path.parents:
            raise RuntimeError(f"Generation escaped step directory: {path}")
        identity = _file_identity(path)
        identity.update(condition=condition, seed=int(row["seed"]), prompt_sha256=expected_prompts[condition]["sha256"])
        files.append(identity)
    for role in ("source_visualization", "target_visualization", "contact_sheet"):
        path = Path(report[role]).resolve()
        if step_dir not in path.parents:
            raise RuntimeError(f"Report asset escaped step directory: {path}")
        identity = _file_identity(path)
        identity["role"] = role
        files.append(identity)
    if _runtime_identities(Path(checkpoint["locked_config"]["llm_path"])) != runtime:
        raise RuntimeError("Runtime assets changed during generation")
    return {
        "run_id": run_id,
        "step": step,
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_path": checkpoint["path"],
        "report": _file_identity(report_path),
        "files": files,
        "runtime_identities": runtime,
        "condition_manifest": expected_prompts,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gpu-id", default="0")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        args.output_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise RuntimeError(f"Strict output directory already exists: {args.output_dir}; use a new directory") from exc
    pair = _pick_and_validate_pair(load_jsonl(MANIFEST))
    prompts = condition_manifest(pair)
    checkpoints = {str(step): _load_checkpoint_identity(path, step) for step, path in CHECKPOINTS.items()}
    _validate_pair(checkpoints["0"], checkpoints["500"])
    runtime = _runtime_identities(args.model_dir)
    run_id = str(uuid.uuid4())
    protocol: Dict[str, Any] = {
        "status": "generating",
        "run_id": run_id,
        "created_at": _utc_now(),
        "ancestry_scope": (
            "Legacy checkpoints contain no run UUID. Same-run ancestry is strongly supported by identical experiment/config/schema, "
            "zero LoRA-B and empty history at step0, all LoRA-B updated plus contiguous 1..500 history at step500, and ordered timestamps; "
            "it cannot be proven retrospectively by a cryptographic parent identifier."
        ),
        "research_questions": {
            "A_vs_B": "Does swapping Target Prompt and Source CoT order change obedience?",
            "A_vs_C": "With Target Prompt first, does replacing conflicting Source CoT with aligned Target CoT help?",
            "step0_vs_step500": "Did v3 optimization change each condition relative to zero-update behavior?",
        },
        "conditions": prompts,
        "checkpoint_identities": checkpoints,
        "runtime_identities": runtime,
        "locked_generation": {
            "conditions": list(CONDITION_IDS),
            "seeds": list(SEEDS),
            "visual_image_size": 1024,
            "visual_vae_min_image_size": 1024,
            "visual_vit_image_size": 980,
            "visual_vit_min_image_size": 980,
            "cfg_text_scale": 4.0,
            "cfg_img_scale": 2.0,
            "cfg_interval": [0.0, 1.0],
            "timestep_shift": 3.0,
            "num_timesteps": 25,
            "cfg_renorm_min": 0.0,
            "cfg_renorm_type": "text_channel",
            "cfg_ablation": False,
        },
        "execution_order": [0, 500],
        "independent_process_per_checkpoint": True,
        "commands": [],
        "generation_receipts": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output_dir / "control_protocol.json", protocol)
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu_id),
        PYTHONUNBUFFERED="1",
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128",
    )
    for step in (0, 500):
        if _load_checkpoint_identity(CHECKPOINTS[step], step) != checkpoints[str(step)]:
            raise RuntimeError(f"Checkpoint changed before cold-load generation at step {step}")
        if _runtime_identities(args.model_dir) != runtime:
            raise RuntimeError(f"Runtime assets changed before cold-load generation at step {step}")
        command = [
            sys.executable, str(EVALUATE),
            "--llm-path", str(args.model_dir),
            "--vae-path", str(args.model_dir / "ae.safetensors"),
            "--checkpoint-path", str(args.model_dir / "ema.safetensors"),
            "--output-dir", str(TRAIN_OUTPUT),
            "--eval-dir", str(args.output_dir / "generations"),
            "--resume-checkpoint", str(CHECKPOINTS[step]),
            "--profiles", *CONDITION_IDS,
            "--seeds", *[str(seed) for seed in SEEDS],
            "--visual-image-size", "1024",
            "--visual-vae-min-image-size", "1024",
            "--visual-vit-image-size", "980",
            "--visual-vit-min-image-size", "980",
            "--cfg-text-scale", "4.0",
            "--cfg-img-scale", "2.0",
            "--cfg-interval-start", "0.0",
            "--cfg-interval-end", "1.0",
            "--timestep-shift", "3.0",
            "--num-timesteps", "25",
            "--cfg-renorm-min", "0.0",
            "--cfg-renorm-type", "text_channel",
            "--skip-cfg-ablation",
        ]
        protocol["commands"].append(command)
        _atomic_json(args.output_dir / "control_protocol.json", protocol)
        with (args.output_dir / f"generate_step_{step:06d}.log").open("w", encoding="utf-8") as handle:
            subprocess.run(command, cwd=BAGEL_ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
        if _load_checkpoint_identity(CHECKPOINTS[step], step) != checkpoints[str(step)]:
            raise RuntimeError(f"Checkpoint changed during cold-load generation at step {step}")
        if _runtime_identities(args.model_dir) != runtime:
            raise RuntimeError(f"Runtime assets changed during cold-load generation at step {step}")
        report_path = args.output_dir / "generations" / f"step_{step:06d}" / "evaluation_report.json"
        receipt = _validate_report(report_path, step, checkpoints[str(step)], prompts, runtime, run_id)
        receipt_path = args.output_dir / "receipts" / f"step_{step:06d}.json"
        _atomic_json(receipt_path, receipt)
        protocol["generation_receipts"][str(step)] = _file_identity(receipt_path)
        _atomic_json(args.output_dir / "control_protocol.json", protocol)
    protocol["status"] = "completed"
    protocol["completed_at"] = _utc_now()
    _atomic_json(args.output_dir / "control_protocol.json", protocol)
    print(json.dumps({"status": "completed", "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
