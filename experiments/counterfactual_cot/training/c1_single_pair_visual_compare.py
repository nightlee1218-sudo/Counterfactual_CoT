"""Strict C1 single-pair visual comparison driver.

Modes:
  * preflight: CPU-only frozen-input validation. Writes only the dedicated
    preflight directory.
  * generate: parent orchestrator. Runs one fresh worker subprocess per
    architecture/checkpoint, enforces the Step000 bit-exact gate, then runs
    Step400 and Step500 only after that gate passes.
  * worker: internal cold-load mode used by the orchestrator.

This program never trains, calls backward, or calls optimizer.step. It never
retries, deletes, clears, or overwrites an existing non-empty output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple


REPO_ROOT = Path("/group/40016/niightli/BAGEL").resolve()
SCRIPT_PATH = Path(__file__).resolve()
MODULE_NAME = "experiments.counterfactual_cot.training.c1_single_pair_visual_compare"

FORMAL_OUTPUT_DIR = REPO_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_c1_visual_compare_v1"
)
PREFLIGHT_OUTPUT_DIR = REPO_ROOT / (
    "experiments/counterfactual_cot/outputs/"
    "gold40_v3_lora_c1_visual_compare_preflight_v1"
)

MANIFEST_PATH = REPO_ROOT / (
    "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/"
    "manifests/intervention_pairs_single_ball_c1.jsonl"
)
MANIFEST_SHA256 = "7306c0b1e60a2c6c269c29de29cea0894c06a651d32457f3127906ae4495593b"
PAIR_ID = "cf_source_living_room_play_absolute_position_change_v4_absolute_region_replacement_v4_0000"
IMAGE_ROOT = REPO_ROOT / "experiments/counterfactual_cot/gold40_final"
SOURCE_PATH = IMAGE_ROOT / f"source/{PAIR_ID}__source.png"
TARGET_PATH = IMAGE_ROOT / f"target/{PAIR_ID}__target.png"
SOURCE_SHA256 = "71d756241d84721b7885c68bd4da650bf4873277d48be363fb6990d74ace1539"
TARGET_SHA256 = "8500c4401b5dd17ca7aa06d78bf9f3e1dbab3ceb67904c0a2903e034477f9dd1"
SOURCE_SIZE_BYTES = 1_731_076
TARGET_SIZE_BYTES = 1_723_696
IMAGE_MODE = "RGB"
IMAGE_DIMENSIONS = (1024, 1024)

SOURCE_PROMPT = (
    "One red toy ball must be fully visible, separate and unobstructed in the left_region area. "
    "One gray sofa must be fully visible, separate and unobstructed in the center area. "
    "The scene takes place in a cozy living room, with a light rug, under warm indoor light. "
    "All listed objects must be complete and countable. Do not add any extra instance of the "
    "target object in the background, reflection, poster, screen, outside the frame, or partially "
    "cropped at the edge. The gray sofa must stay centered and clearly separate from the ball."
)
TARGET_PROMPT = (
    "One red toy ball must be fully visible, separate and unobstructed in the right_region area. "
    "One gray sofa must be fully visible, separate and unobstructed in the center area. "
    "The scene takes place in a cozy living room, with a light rug, under warm indoor light. "
    "All listed objects must be complete and countable. Do not add any extra instance of the "
    "target object in the background, reflection, poster, screen, outside the frame, or partially "
    "cropped at the edge. Move only the red toy ball; keep the gray sofa centered."
)
SOURCE_COT = (
    "ball_1.object=toy ball; ball_1.count=1; ball_1.location=left_region; "
    "ball_1.attribute.color=red; sofa_1.object=gray sofa; sofa_1.count=1; "
    "sofa_1.location=center; background.floor=light rug; "
    "background.lighting=warm indoor light; background.scene=cozy living room"
)
TARGET_COT = (
    "ball_1.object=toy ball; ball_1.count=1; ball_1.location=right_region; "
    "ball_1.attribute.color=red; sofa_1.object=gray sofa; sofa_1.count=1; "
    "sofa_1.location=center; background.floor=light rug; "
    "background.lighting=warm indoor light; background.scene=cozy living room"
)
SOURCE_PROMPT_SHA256 = "608213d94cb98602490a2e956d6a76c75eaac6042a9965f7d01f4dd4fc4e30a4"
TARGET_PROMPT_SHA256 = "f2f49ddcdf0db2b2c22e7e652c9240f8529b069d411495cd5e7c5180318dca4c"
SOURCE_COT_SHA256 = "6a8262eb6a459fd66db521d5eeb621b3d9400bbc9dece30b8618ec0748bf7035"
TARGET_COT_SHA256 = "20452b4539ee898ef1400cc359920ee467e966e197fc43869fae98f9c2dffe1b"

INTERVENTION = {
    "intervention_type": "spatial_position_change",
    "entity_id": "ball_1",
    "field": "location",
    "old_value": "left_region",
    "new_value": "right_region",
}
MUST_PRESERVE = [
    "ball_1.category",
    "ball_1.count",
    "ball_1.attributes.color",
    "sofa_1.category",
    "sofa_1.count",
    "sofa_1.location",
    "background.scene",
    "background.floor",
    "background.lighting",
]
ALLOWED_TO_CHANGE = ["ball_1.location"]
PAIRED_PROMPT = (
    f"[MODE=COUNTERFACTUAL_EDIT]\nSource Prompt: {SOURCE_PROMPT}\n"
    f"Source CoT: {SOURCE_COT}\n"
    "Intervention: spatial_position_change: ball_1.location left_region -> right_region\n"
    f"Target CoT: {TARGET_COT}\n"
    "Preserve: ball_1.category, ball_1.count, ball_1.attributes.color, "
    "sofa_1.category, sofa_1.count, sofa_1.location, background.scene, "
    "background.floor, background.lighting. Allowed changes: ball_1.location.\n"
    "Generate the target image."
)
PAIRED_PROMPT_SHA256 = "2dcc6fefcd98f201357663a3fc0422f2464ce69a3fb00266e6b443cf4209578a"

LLM_PATH = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
AE_PATH = LLM_PATH / "ae.safetensors"
EMA_PATH = LLM_PATH / "ema.safetensors"
AE_SHA256 = "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38"
EMA_SHA256 = "0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3"

ARCHITECTURES = ("historical_no_lora", "c1_lora")
STEPS = (0, 400, 500)
SEEDS = (1234, 2234, 3234)
CHECKPOINTS: Dict[Tuple[str, int], Dict[str, Any]] = {
    ("historical_no_lora", 0): {
        "path": REPO_ROOT / "experiments/counterfactual_cot/outputs/single_pair_overfit_diag_ball_lr3e5_500steps/checkpoints/step_000_trainable.pt",
        "size_bytes": 126_926_722,
        "sha256": "6ab8a39e65fa15eec8f72c923a52d8340ef8b3e18135ede91c6626be392764da",
    },
    ("historical_no_lora", 400): {
        "path": REPO_ROOT / "experiments/counterfactual_cot/outputs/single_pair_overfit_diag_ball_lr3e5_500steps/checkpoints/step_400_trainable.pt",
        "size_bytes": 380_831_762,
        "sha256": "30546255d0f564ee540c200361bdd0000025a22ef44a4ef04471a72e6efeed8a",
    },
    ("historical_no_lora", 500): {
        "path": REPO_ROOT / "experiments/counterfactual_cot/outputs/single_pair_overfit_diag_ball_lr3e5_500steps/checkpoints/step_500_trainable.pt",
        "size_bytes": 380_850_386,
        "sha256": "4836a58411940a4956fb1dc230a0b907dab627cec7422d07bc4a12e97bf05fc8",
    },
    ("c1_lora", 0): {
        "path": REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_lora_c1_single_ball_p4d_v1/checkpoints/step_000_trainable.pt",
        "size_bytes": 135_587_762,
        "sha256": "0af1669d6ad50f7354b850917401a6862970221f63c786311f12fb5aad924001",
    },
    ("c1_lora", 400): {
        "path": REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_lora_c1_single_ball_p4d_v1/checkpoints/step_400_trainable.pt",
        "size_bytes": 135_587_762,
        "sha256": "58a69f9b3a1dd34569c56ebe0131ec2e0a4edd36d87f1018fc6889be59620995",
    },
    ("c1_lora", 500): {
        "path": REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_lora_c1_single_ball_p4d_v1/checkpoints/step_500_trainable.pt",
        "size_bytes": 135_587_762,
        "sha256": "ca15aef699e9255e305a865fb3ea113531f101f021f60a58c87a92fed4a602ac",
    },
}

FROZEN_INFERENCE = {
    "seeds": list(SEEDS),
    "steps": list(STEPS),
    "num_timesteps": 25,
    "cfg_text_scale": 4.0,
    "cfg_img_scale": 2.0,
    "cfg_interval": [0.0, 1.0],
    "timestep_shift": 3.0,
    "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "text_channel",
    "image_size": 1024,
    "precision": "bf16",
    "model_mode": "eval",
    "prompt_mode": "paired_prompt(include_source_image=True)",
    "think": False,
    "understanding_output": False,
    "lora_layers": [24, 25, 26, 27],
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
}

TRAINABLE_REGEX = r"^(time_embedder|latent_pos_embed|vit_pos_embed|vae2llm|llm2vae|connector)\."
EXPECTED_EXISTING_TENSORS = 14
EXPECTED_LORA_TENSORS = 48
EXPECTED_LORA_MODULES = 24
EXPECTED_LORA_TOTAL_TENSORS = 62
EXPECTED_EMA_LOADED_TENSORS = 1221
EXPECTED_OUTPUT_COUNT = 18
RUN_TOKEN_ENV = "C1_VISUAL_COMPARE_RUN_TOKEN"
RUN_STATE_NAME = "run_state.json"


class _ClaimedOutputRootError(RuntimeError):
    """Raised after this process created the exclusive output-root claim."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(
    path: Path,
    label: str,
    *,
    require_leaf: bool,
) -> Path:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if current == absolute and require_leaf:
                raise RuntimeError(f"{label} is missing: {absolute}")
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError(f"{label} contains a symlink component: {current}")
    if require_leaf:
        metadata = absolute.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {absolute}")
    return absolute


def _require_empty_or_absent(path: Path, label: str) -> None:
    absolute = _assert_no_symlink_components(path, label, require_leaf=False)
    try:
        metadata = absolute.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(metadata.st_mode) or any(absolute.iterdir()):
        raise RuntimeError(
            f"G_dir {label}: path exists and is non-empty or is not a directory: {absolute}. "
            "Refusing to delete, clear, or overwrite it."
        )


def _claim_output_root(path: Path, label: str) -> Path:
    _require_empty_or_absent(path, label)
    absolute = _lexical_absolute(path)
    try:
        absolute.lstat()
    except FileNotFoundError:
        absolute.mkdir(parents=True, exist_ok=False)
    _assert_no_symlink_components(absolute, label, require_leaf=False)
    metadata = absolute.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"G_dir {label} is not a directory: {absolute}")
    claim_path = absolute / ".c1_visual_compare_claim"
    descriptor = os.open(claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "created_at_iso": _now_iso()}, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        _assert_no_symlink_components(absolute, label, require_leaf=False)
        unexpected = sorted(item.name for item in absolute.iterdir() if item != claim_path)
        if unexpected:
            raise RuntimeError(
                f"G_dir {label}: unexpected entries appeared during exclusive claim: {unexpected}"
            )
    except BaseException as exc:  # noqa: BLE001
        raise _ClaimedOutputRootError(
            f"G_dir {label} failed after exclusive claim creation: {exc}"
        ) from exc
    return claim_path


def _binding(path: Path, expected_sha256: str, expected_size: int | None = None) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Frozen file is missing: {path}")
    observed_size = path.stat().st_size
    observed_sha256 = _sha256_file(path)
    if expected_size is not None and observed_size != expected_size:
        raise RuntimeError(
            f"Frozen file size mismatch for {path}: expected={expected_size}, observed={observed_size}"
        )
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"Frozen file SHA256 mismatch for {path}: "
            f"expected={expected_sha256}, observed={observed_sha256}"
        )
    return {
        "path": str(path),
        "size_bytes": observed_size,
        "sha256_expected": expected_sha256,
        "sha256_observed": observed_sha256,
        "match": True,
    }


def _load_unique_pair() -> Dict[str, Any]:
    rows = [json.loads(line) for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Manifest must contain exactly one row, got {len(rows)}")
    pair = rows[0]
    if pair.get("pair_id") != PAIR_ID:
        raise RuntimeError(f"pair_id mismatch: expected={PAIR_ID!r}, actual={pair.get('pair_id')!r}")
    return pair


def _verify_text(label: str, actual: Any, expected: str, expected_sha256: str) -> Dict[str, Any]:
    if actual != expected:
        raise RuntimeError(f"Frozen {label} text mismatch")
    observed_sha256 = _sha256_text(actual)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"Frozen {label} SHA256 mismatch: expected={expected_sha256}, observed={observed_sha256}"
        )
    return {
        "text": actual,
        "sha256_expected": expected_sha256,
        "sha256_observed": observed_sha256,
        "match": True,
    }


def _verify_image(path: Path, expected_sha256: str, expected_size: int) -> Dict[str, Any]:
    from PIL import Image

    result = _binding(path, expected_sha256, expected_size)
    with Image.open(path) as image:
        observed_dimensions = tuple(image.size)
        observed_mode = image.mode
    if observed_dimensions != IMAGE_DIMENSIONS or observed_mode != IMAGE_MODE:
        raise RuntimeError(
            f"Frozen image metadata mismatch for {path}: expected={IMAGE_DIMENSIONS}/{IMAGE_MODE}, "
            f"observed={observed_dimensions}/{observed_mode}"
        )
    result.update({"dimensions": list(observed_dimensions), "mode": observed_mode})
    return result


def _verify_frozen_inputs() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bindings: Dict[str, Any] = {
        "manifest": _binding(MANIFEST_PATH, MANIFEST_SHA256),
        "ae": _binding(AE_PATH, AE_SHA256),
        "ema": _binding(EMA_PATH, EMA_SHA256),
        "script": {
            "path": str(SCRIPT_PATH),
            "size_bytes": SCRIPT_PATH.stat().st_size,
            "sha256": _sha256_file(SCRIPT_PATH),
        },
        "checkpoints": {},
    }
    for architecture in ARCHITECTURES:
        bindings["checkpoints"][architecture] = {}
        for step in STEPS:
            spec = CHECKPOINTS[(architecture, step)]
            bindings["checkpoints"][architecture][str(step)] = _binding(
                spec["path"], spec["sha256"], spec["size_bytes"]
            )

    pair = _load_unique_pair()
    if pair.get("final_image_root") != str(IMAGE_ROOT):
        raise RuntimeError(
            f"final_image_root mismatch: expected={str(IMAGE_ROOT)!r}, "
            f"actual={pair.get('final_image_root')!r}"
        )
    expected_relative_images = {
        "source": f"source/{PAIR_ID}__source.png",
        "target": f"target/{PAIR_ID}__target.png",
    }
    for side, expected_relative in expected_relative_images.items():
        if pair.get(side, {}).get("image") != expected_relative:
            raise RuntimeError(
                f"{side}.image mismatch: expected={expected_relative!r}, "
                f"actual={pair.get(side, {}).get('image')!r}"
            )

    intervention = pair.get("intervention", {})
    for field, expected in INTERVENTION.items():
        if intervention.get(field) != expected:
            raise RuntimeError(
                f"intervention.{field} mismatch: expected={expected!r}, actual={intervention.get(field)!r}"
            )
    preservation = pair.get("preservation", {})
    if preservation.get("must_preserve") != MUST_PRESERVE:
        raise RuntimeError("preservation.must_preserve mismatch")
    if preservation.get("allowed_to_change") != ALLOWED_TO_CHANGE:
        raise RuntimeError("preservation.allowed_to_change mismatch")

    bindings["pair_id"] = PAIR_ID
    bindings["source_image"] = _verify_image(SOURCE_PATH, SOURCE_SHA256, SOURCE_SIZE_BYTES)
    bindings["target_image"] = _verify_image(TARGET_PATH, TARGET_SHA256, TARGET_SIZE_BYTES)
    bindings["source_prompt"] = _verify_text(
        "source prompt", pair.get("source", {}).get("prompt"), SOURCE_PROMPT, SOURCE_PROMPT_SHA256
    )
    bindings["target_prompt"] = _verify_text(
        "target prompt", pair.get("target", {}).get("prompt"), TARGET_PROMPT, TARGET_PROMPT_SHA256
    )
    bindings["source_cot"] = _verify_text(
        "source CoT", pair.get("source", {}).get("cot_text"), SOURCE_COT, SOURCE_COT_SHA256
    )
    bindings["target_cot"] = _verify_text(
        "target CoT", pair.get("target", {}).get("cot_text"), TARGET_COT, TARGET_COT_SHA256
    )
    bindings["paired_prompt"] = {
        "text": PAIRED_PROMPT,
        "sha256_expected": PAIRED_PROMPT_SHA256,
        "sha256_observed": _sha256_text(PAIRED_PROMPT),
        "match": _sha256_text(PAIRED_PROMPT) == PAIRED_PROMPT_SHA256,
    }
    if not bindings["paired_prompt"]["match"]:
        raise RuntimeError("Frozen paired prompt self-binding mismatch")
    return bindings, pair


def _runtime_parent() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "argv": list(sys.argv),
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _runtime_worker(torch: Any) -> Dict[str, Any]:
    gpu: Dict[str, Any] = {"available": bool(torch.cuda.is_available())}
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu.update({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": int(properties.total_memory),
            "capability": list(torch.cuda.get_device_capability(index)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })
    return {
        **_runtime_parent(),
        "pytorch_version": torch.__version__,
        "pytorch_compiled_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": gpu,
    }


def _run_state_path(output_root: Path = FORMAL_OUTPUT_DIR) -> Path:
    return output_root / RUN_STATE_NAME


def _publish_run_state(
    run_token: str,
    *,
    phase: str,
    authorized_condition: Tuple[str, int] | None,
    completed_conditions: Sequence[Tuple[str, int]],
) -> None:
    state = {
        "parent_pid": os.getpid(),
        "run_token_sha256": _sha256_text(run_token),
        "phase": phase,
        "authorized_condition": list(authorized_condition) if authorized_condition else None,
        "completed_conditions": [list(condition) for condition in completed_conditions],
        "updated_at_iso": _now_iso(),
    }
    _atomic_write_json(_run_state_path(), state)


def _verify_worker_authorization(architecture: str, step: int) -> Dict[str, Any]:
    token = os.environ.get(RUN_TOKEN_ENV)
    if not token:
        raise RuntimeError("Internal worker authorization token is missing")
    state_path = _run_state_path()
    if not state_path.is_file():
        raise RuntimeError(f"Internal worker run state is missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected_parent_pid = state.get("parent_pid")
    if type(expected_parent_pid) is not int or os.getppid() != expected_parent_pid:
        raise RuntimeError(
            f"Internal worker parent mismatch: expected={expected_parent_pid!r}, actual={os.getppid()}"
        )
    if state.get("run_token_sha256") != _sha256_text(token):
        raise RuntimeError("Internal worker authorization token mismatch")
    if state.get("authorized_condition") != [architecture, step]:
        raise RuntimeError(
            "Internal worker condition is not authorized by the parent: "
            f"authorized={state.get('authorized_condition')!r}, requested={[architecture, step]!r}"
        )
    expected_phase = "step000" if step == 0 else "later"
    if state.get("phase") != expected_phase:
        raise RuntimeError(
            f"Internal worker phase mismatch: expected={expected_phase!r}, actual={state.get('phase')!r}"
        )
    return {
        "state_path": str(state_path),
        "parent_pid": expected_parent_pid,
        "worker_parent_pid": os.getppid(),
        "phase": state["phase"],
        "authorized_condition": state["authorized_condition"],
        "run_token_sha256": state["run_token_sha256"],
    }


def _inference_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        llm_path=str(LLM_PATH),
        vae_path=str(AE_PATH),
        checkpoint_path=str(EMA_PATH),
        device="cuda",
        precision="bf16",
        model_device_map="single",
        device_map_max_memory="34GiB",
        image_size=256,
        vit_image_size=224,
        vit_min_image_size=224,
        visual_image_size=1024,
        visual_vae_min_image_size=512,
        visual_vit_image_size=980,
        visual_vit_min_image_size=224,
        expected_num_tokens=512,
        max_num_tokens_per_sample=1536,
        max_num_tokens=1536,
        cfg_text_scale=4.0,
        cfg_img_scale=2.0,
        cfg_interval_start=0.0,
        cfg_interval_end=1.0,
        timestep_shift=3.0,
        num_timesteps=25,
        cfg_renorm_min=0.0,
        cfg_renorm_type="text_channel",
        seed=SEEDS[0],
        generation_seed=SEEDS[0],
        allow_non_dockerdata=False,
        require_paired_vit=True,
        trainable_name_regex=TRAINABLE_REGEX,
        lora_layers=[24, 25, 26, 27],
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.0,
    )


def _strict_load_trainable(payload: Any, model: Any, architecture: str, torch: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"Checkpoint payload must be dict, got {type(payload).__name__}")
    if "trainable_state_dict" not in payload:
        raise RuntimeError("Checkpoint payload is missing trainable_state_dict")
    saved = payload["trainable_state_dict"]
    if not isinstance(saved, dict):
        raise RuntimeError(f"trainable_state_dict must be dict, got {type(saved).__name__}")

    named = dict(model.named_parameters())
    expected = {name for name, parameter in named.items() if parameter.requires_grad}
    saved_keys = set(saved)
    expected_count = EXPECTED_EXISTING_TENSORS if architecture == "historical_no_lora" else EXPECTED_LORA_TOTAL_TENSORS
    if len(expected) != expected_count:
        raise RuntimeError(
            f"Expected trainable key-set count={expected_count} for {architecture}, got {len(expected)}"
        )
    missing = sorted(expected - saved_keys)
    unexpected = sorted(saved_keys - expected)
    if missing or unexpected:
        raise RuntimeError(f"Strict trainable key-set mismatch: missing={missing}, unexpected={unexpected}")

    per_tensor: Dict[str, Any] = {}
    checkpoint_dtype_counts: Dict[str, int] = {}
    model_dtype_counts: Dict[str, int] = {}
    max_abs_copy_diff = 0.0
    for name in sorted(saved):
        tensor = saved[name]
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"Checkpoint trainable_state_dict[{name!r}] is not a tensor")
        parameter = named[name]
        expected_dtype = torch.float32 if architecture == "c1_lora" and name.endswith((".lora_A", ".lora_B")) else torch.bfloat16
        if tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"Checkpoint dtype mismatch for {name}: expected={expected_dtype}, actual={tensor.dtype}"
            )
        if parameter.dtype != expected_dtype:
            raise RuntimeError(
                f"Model dtype mismatch for {name}: expected={expected_dtype}, actual={parameter.dtype}"
            )
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"Shape mismatch for {name}: model={tuple(parameter.shape)}, checkpoint={tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor.to(dtype=torch.float32)).all().item()):
            raise RuntimeError(f"Non-finite checkpoint tensor: {name}")
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
        copied = parameter.detach().to(device="cpu", dtype=torch.float32)
        original = tensor.detach().to(device="cpu", dtype=torch.float32)
        copy_diff = float((copied - original).abs().max().item())
        if copy_diff != 0.0:
            raise RuntimeError(f"Copy-diff mismatch for {name}: max_abs_diff={copy_diff}")
        max_abs_copy_diff = max(max_abs_copy_diff, copy_diff)
        checkpoint_dtype_counts[str(tensor.dtype)] = checkpoint_dtype_counts.get(str(tensor.dtype), 0) + 1
        model_dtype_counts[str(parameter.dtype)] = model_dtype_counts.get(str(parameter.dtype), 0) + 1
        per_tensor[name] = {
            "shape": list(tensor.shape),
            "checkpoint_dtype": str(tensor.dtype),
            "model_dtype": str(parameter.dtype),
            "finite": True,
            "max_abs_load_copy_diff": copy_diff,
        }

    expected_dtypes = {"torch.bfloat16": 14}
    if architecture == "c1_lora":
        expected_dtypes["torch.float32"] = 48
    if checkpoint_dtype_counts != expected_dtypes or model_dtype_counts != expected_dtypes:
        raise RuntimeError(
            f"Dtype count mismatch for {architecture}: expected={expected_dtypes}, "
            f"checkpoint={checkpoint_dtype_counts}, model={model_dtype_counts}"
        )
    return {
        "payload_source": "payload[trainable_state_dict] only",
        "ignored_payload_keys": sorted(set(payload) - {"trainable_state_dict"}),
        "expected_key_count": expected_count,
        "saved_key_count": len(saved_keys),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "checkpoint_dtype_counts": checkpoint_dtype_counts,
        "model_dtype_counts": model_dtype_counts,
        "max_abs_load_copy_diff": max_abs_copy_diff,
        "per_tensor": per_tensor,
    }


def _expected_output_path(
    output_root: Path,
    architecture: str,
    step: int,
    seed: int,
) -> Path:
    return _lexical_absolute(
        output_root
        / "workers"
        / f"{architecture}__step_{step:03d}"
        / f"seed_{seed}.png"
    )


def _validate_model_output_path(
    raw_path: Any,
    architecture: str,
    step: int,
    seed: int,
    *,
    output_root: Path = FORMAL_OUTPUT_DIR,
) -> Path:
    if not isinstance(raw_path, str):
        raise RuntimeError(
            f"Output path must be str for {architecture}/step={step}/seed={seed}, "
            f"got {type(raw_path).__name__}"
        )
    path = Path(raw_path)
    expected = _expected_output_path(output_root, architecture, step, seed)
    lexical = _lexical_absolute(path)
    if lexical != expected or raw_path != str(expected):
        raise RuntimeError(
            f"Output path mismatch for {architecture}/step={step}/seed={seed}: "
            f"expected={expected}, actual={raw_path!r}"
        )
    _assert_no_symlink_components(lexical, "model output path", require_leaf=True)
    expected_worker_dir = _lexical_absolute(expected.parent)
    formal_root = _lexical_absolute(output_root)
    resolved = lexical.resolve(strict=True)
    resolved_worker_dir = expected_worker_dir.resolve(strict=True)
    resolved_formal_root = formal_root.resolve(strict=True)
    if resolved.parent != resolved_worker_dir:
        raise RuntimeError(
            f"Output escaped expected worker directory: output={resolved}, "
            f"worker_dir={resolved_worker_dir}"
        )
    try:
        resolved.relative_to(resolved_formal_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Output escaped formal root: output={resolved}, root={resolved_formal_root}"
        ) from exc
    if resolved.name != f"seed_{seed}.png":
        raise RuntimeError(f"Output filename mismatch: {resolved.name}")
    return resolved


def _image_record(path: Path) -> Dict[str, Any]:
    from PIL import Image

    _assert_no_symlink_components(path, "PNG image", require_leaf=True)
    with Image.open(path) as image:
        native_mode = image.mode
        if native_mode != IMAGE_MODE:
            raise RuntimeError(
                f"PNG native mode mismatch for {path}: expected={IMAGE_MODE}, actual={native_mode}"
            )
        rgb = image.convert("RGB")
        pixel_bytes = rgb.tobytes()
        dimensions = tuple(rgb.size)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "png_sha256": _sha256_file(path),
        "rgb_pixel_sha256": hashlib.sha256(pixel_bytes).hexdigest(),
        "dimensions": list(dimensions),
        "shape": [dimensions[1], dimensions[0], 3],
        "native_mode": native_mode,
        "mode": native_mode,
    }


def _worker_run(architecture: str, step: int, checkpoint: Path, output_dir: Path) -> Dict[str, Any]:
    import random

    import numpy as np
    import torch

    if architecture not in ARCHITECTURES or step not in STEPS:
        raise RuntimeError(f"Invalid frozen worker condition: architecture={architecture!r}, step={step!r}")
    spec = CHECKPOINTS[(architecture, step)]
    if checkpoint.resolve() != spec["path"].resolve():
        raise RuntimeError(f"Worker checkpoint path mismatch: expected={spec['path']}, actual={checkpoint}")
    expected_output_dir = FORMAL_OUTPUT_DIR / "workers" / f"{architecture}__step_{step:03d}"
    if output_dir.resolve() != expected_output_dir.resolve():
        raise RuntimeError(
            f"Worker output path mismatch: expected={expected_output_dir}, actual={output_dir}"
        )
    _assert_no_symlink_components(
        output_dir, "worker output directory", require_leaf=False
    )
    if not output_dir.is_dir() or any(output_dir.iterdir()):
        raise RuntimeError(f"Worker must own a newly claimed empty directory: {output_dir}")
    authorization = _verify_worker_authorization(architecture, step)

    bindings, pair = _verify_frozen_inputs()
    checkpoint_binding = bindings["checkpoints"][architecture][str(step)]

    from experiments.counterfactual_cot.data.render_prompts import paired_prompt
    from experiments.counterfactual_cot.training.gold40_overfit import (
        _build_model_for_gold40,
        _generate_one,
        _load_tokenizer,
        _make_inferencer,
    )
    from experiments.counterfactual_cot.training.lora_2step_smoke import (
        TARGET_MODULE_SUFFIXES,
        _iter_target_linear_names,
        inject_lora,
        set_trainable,
    )

    rendered_prompt = paired_prompt(pair, include_source_image=True)
    if rendered_prompt != PAIRED_PROMPT or _sha256_text(rendered_prompt) != PAIRED_PROMPT_SHA256:
        raise RuntimeError("paired_prompt(include_source_image=True) binding mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in generation worker")

    args = _inference_namespace()
    torch.manual_seed(SEEDS[0])
    np.random.seed(SEEDS[0])
    random.seed(SEEDS[0])
    torch.cuda.manual_seed_all(SEEDS[0])
    torch.cuda.reset_peak_memory_stats()

    tokenizer, special_tokens, num_new_tokens = _load_tokenizer(str(LLM_PATH))
    device = torch.device("cuda")
    model, vae_model, base_checkpoint_report = _build_model_for_gold40(
        args, device, tokenizer_size=len(tokenizer), num_new_tokens=num_new_tokens
    )
    if not isinstance(base_checkpoint_report, dict):
        raise RuntimeError(
            "EMA loader did not return a checkpoint compatibility report"
        )
    for field in ("num_loaded_tensors", "num_missing_keys", "num_unexpected_keys"):
        if type(base_checkpoint_report.get(field)) is not int:
            raise RuntimeError(
                f"EMA checkpoint report field={field} must be int (bool forbidden), "
                f"actual={base_checkpoint_report.get(field)!r}"
            )
    if base_checkpoint_report.get("checkpoint_path") != str(EMA_PATH):
        raise RuntimeError(
            "EMA checkpoint report path mismatch: "
            f"expected={str(EMA_PATH)!r}, "
            f"actual={base_checkpoint_report.get('checkpoint_path')!r}"
        )
    if base_checkpoint_report["num_loaded_tensors"] != EXPECTED_EMA_LOADED_TENSORS:
        raise RuntimeError(
            "EMA loaded tensor count mismatch: "
            f"expected={EXPECTED_EMA_LOADED_TENSORS}, "
            f"actual={base_checkpoint_report['num_loaded_tensors']}"
        )
    expected_ema_missing_keys = {
        "latent_pos_embed.pos_embed",
        "vit_pos_embed.pos_embed",
    }
    ema_missing_sample = base_checkpoint_report.get("missing_keys_sample")
    ema_unexpected_sample = base_checkpoint_report.get("unexpected_keys_sample")
    if (
        base_checkpoint_report["num_missing_keys"] != len(expected_ema_missing_keys)
        or not isinstance(ema_missing_sample, list)
        or set(ema_missing_sample) != expected_ema_missing_keys
    ):
        raise RuntimeError(
            "EMA loader missing-key contract mismatch: "
            f"expected={sorted(expected_ema_missing_keys)}, "
            f"count={base_checkpoint_report['num_missing_keys']}, "
            f"sample={ema_missing_sample!r}"
        )
    if (
        base_checkpoint_report["num_unexpected_keys"] != 0
        or not isinstance(ema_unexpected_sample, list)
        or ema_unexpected_sample
    ):
        raise RuntimeError(
            "EMA loader reported unexpected keys: "
            f"count={base_checkpoint_report['num_unexpected_keys']}, "
            f"sample={ema_unexpected_sample!r}"
        )
    loader_compatibility_audit = {
        "ema": {
            "status": "passed_with_expected_fixed_position_omissions",
            "source": "_build_model_for_gold40 checkpoint report",
            "checkpoint_path": str(EMA_PATH),
            "num_loaded_tensors": EXPECTED_EMA_LOADED_TENSORS,
            "num_missing_keys": len(expected_ema_missing_keys),
            "expected_missing_keys": sorted(expected_ema_missing_keys),
            "num_unexpected_keys": 0,
        },
        "ae": {
            "status": "not_exposed_by_external_loader",
            "source": "modeling.autoencoder.load_ae returns (model, params) only",
            "limitation": (
                "AE file SHA256 is frozen, but missing/unexpected load keys are not "
                "returned by the external loader and cannot be hard-checked here."
            ),
        },
    }
    if architecture == "historical_no_lora":
        target_names: List[str] = []
        wrappers: Dict[str, Any] = {}
    else:
        target_names = _iter_target_linear_names(model, args.lora_layers)
        if len(target_names) != EXPECTED_LORA_MODULES:
            raise RuntimeError(
                f"LoRA target count mismatch: expected={EXPECTED_LORA_MODULES}, actual={len(target_names)}"
            )
        if len(target_names) != len(args.lora_layers) * len(TARGET_MODULE_SUFFIXES):
            raise RuntimeError("LoRA target matrix is not exactly L24-L27 x target suffixes")
        wrappers = inject_lora(model, target_names, args.lora_r, args.lora_alpha, args.lora_dropout)

    trainable_report = set_trainable(model, TRAINABLE_REGEX, wrappers)
    existing_names = list(trainable_report["existing_trainable_names"])
    lora_names = list(trainable_report["lora_trainable_names"])
    if len(existing_names) != EXPECTED_EXISTING_TENSORS:
        raise RuntimeError(
            f"Existing trainable count mismatch: expected=14, actual={len(existing_names)}"
        )
    expected_lora = 0 if architecture == "historical_no_lora" else EXPECTED_LORA_TENSORS
    if len(lora_names) != expected_lora:
        raise RuntimeError(
            f"LoRA trainable count mismatch for {architecture}: expected={expected_lora}, actual={len(lora_names)}"
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    strict_load_report = _strict_load_trainable(payload, model, architecture, torch)
    model.eval()
    vae_model.eval()
    if model.training or vae_model.training:
        raise RuntimeError("model.eval()/vae_model.eval() did not take effect")
    inferencer = _make_inferencer(model, vae_model, tokenizer, special_tokens, args)

    outputs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        output_path = output_dir / f"seed_{seed}.png"
        try:
            output_path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError(f"Refusing to overwrite output image: {output_path}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng_reset = {
            "python_seed": seed,
            "numpy_seed": seed,
            "torch_cpu_seed": seed,
            "torch_cuda_seed": seed if torch.cuda.is_available() else None,
        }
        _generate_one(
            inferencer,
            pair,
            output_path,
            args,
            seed=seed,
            prompt=rendered_prompt,
        )
        image = _image_record(output_path)
        if (
            tuple(image["dimensions"]) != IMAGE_DIMENSIONS
            or image["native_mode"] != IMAGE_MODE
        ):
            raise RuntimeError(f"Generated image metadata mismatch: {image}")
        outputs.append({
            "architecture": architecture,
            "step": step,
            "seed": seed,
            "worker_pid": os.getpid(),
            "rng_reset": rng_reset,
            "checkpoint": checkpoint_binding,
            **image,
        })

    return {
        "status": "passed",
        "failures": [],
        "architecture": architecture,
        "step": step,
        "seeds": list(SEEDS),
        "pid": os.getpid(),
        "authorization": authorization,
        "checkpoint": checkpoint_binding,
        "runtime": _runtime_worker(torch),
        "frozen_inference": FROZEN_INFERENCE,
        "strict_load_report": strict_load_report,
        "trainable_report": {
            "existing_names": existing_names,
            "lora_names": lora_names,
            "lora_target_names": target_names,
            "existing_count": len(existing_names),
            "lora_count": len(lora_names),
        },
        "base_checkpoint_report": base_checkpoint_report,
        "loader_compatibility_audit": loader_compatibility_audit,
        "model_eval": not model.training,
        "vae_eval": not vae_model.training,
        "outputs": outputs,
        "gpu_memory": {
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 * 1024),
        },
    }


def _collect_worker_partial_outputs(
    output_dir: Path,
    architecture: str,
    step: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    checkpoint_spec = CHECKPOINTS[(architecture, step)]
    checkpoint_record = {
        "path": str(checkpoint_spec["path"]),
        "size_bytes": checkpoint_spec["size_bytes"],
        "sha256_expected": checkpoint_spec["sha256"],
        "sha256_observed": checkpoint_spec["sha256"],
        "match": True,
    }
    for seed in SEEDS:
        path = output_dir / f"seed_{seed}.png"
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        except BaseException as stat_exc:  # noqa: BLE001
            records.append({
                "partial": True,
                "architecture": architecture,
                "step": step,
                "seed": seed,
                "worker_pid": os.getpid(),
                "checkpoint": dict(checkpoint_record),
                "path": str(path),
                "image_record_error": (
                    f"lstat {type(stat_exc).__name__}: {stat_exc}"
                ),
            })
            continue
        record: Dict[str, Any] = {
            "partial": True,
            "architecture": architecture,
            "step": step,
            "seed": seed,
            "worker_pid": os.getpid(),
            "rng_reset": {
                "python_seed": seed,
                "numpy_seed": seed,
                "torch_cpu_seed": seed,
                "torch_cuda_seed": seed,
            },
            "checkpoint": dict(checkpoint_record),
            "path": str(path),
        }
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            record["image_record_error"] = (
                "partial output is a symlink or is not a regular file"
            )
        else:
            try:
                record.update(_image_record(path))
            except BaseException as image_exc:  # noqa: BLE001
                record.update({
                    "size_bytes": metadata.st_size,
                    "image_record_error": (
                        f"{type(image_exc).__name__}: {image_exc}"
                    ),
                })
                try:
                    record["png_sha256"] = _sha256_file(path)
                except BaseException as sha_exc:  # noqa: BLE001
                    record["png_sha256_error"] = (
                        f"{type(sha_exc).__name__}: {sha_exc}"
                    )
        records.append(record)
    return records


def _worker_main(args: argparse.Namespace) -> int:
    output_dir = args.worker_output_dir
    claimed = False
    try:
        # Authorization must pass before the worker can create or write anything.
        authorization = _verify_worker_authorization(args.architecture, args.step)
        expected_parent = _lexical_absolute(FORMAL_OUTPUT_DIR / "workers")
        lexical_output_dir = _lexical_absolute(output_dir)
        if lexical_output_dir.parent != expected_parent:
            raise RuntimeError(f"Worker output parent mismatch: {output_dir.parent}")
        _assert_no_symlink_components(
            expected_parent, "worker output parent", require_leaf=False
        )
        try:
            lexical_output_dir.lstat()
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError(f"Worker output directory already exists: {lexical_output_dir}")
        lexical_output_dir.mkdir(parents=False, exist_ok=False)
        output_dir = lexical_output_dir
        claimed = True
        report = _worker_run(args.architecture, args.step, args.checkpoint, output_dir)
        if report.get("authorization") != authorization:
            raise RuntimeError("Worker authorization changed during execution")
        _atomic_write_json(output_dir / "worker_report.json", report)
        return 0
    except BaseException as exc:  # noqa: BLE001
        failure_traceback = traceback.format_exc()
        if claimed:
            torch_module = sys.modules.get("torch")
            runtime = (
                _runtime_worker(torch_module)
                if torch_module is not None
                else {**_runtime_parent(), "pytorch_version": None, "gpu": {"available": None}}
            )
            try:
                _atomic_write_json(output_dir / "worker_report.json", {
                    "status": "failed",
                    "failures": [f"{type(exc).__name__}: {exc}"],
                    "architecture": args.architecture,
                    "step": args.step,
                    "seeds": list(SEEDS),
                    "pid": os.getpid(),
                    "checkpoint": str(args.checkpoint),
                    "runtime": runtime,
                    "outputs_present_at_failure": _collect_worker_partial_outputs(
                        output_dir,
                        args.architecture,
                        args.step,
                    ),
                    "traceback": failure_traceback,
                    "failure_semantics": (
                        "This worker stopped. The parent orchestrator must record whether "
                        "Step400/500 generation will execute. No retry, deletion, or overwrite occurred."
                    ),
                })
            except Exception as report_exc:  # noqa: BLE001
                print(
                    f"Failed to atomically write worker failure report: {report_exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
        print(failure_traceback, file=sys.stderr, flush=True)
        return 1


def _copy_references(output_root: Path) -> Dict[str, Any]:
    reference_dir = output_root / "references"
    reference_dir.mkdir(parents=True, exist_ok=False)
    records: Dict[str, Any] = {}
    for side, source, expected_sha in (
        ("source", SOURCE_PATH, SOURCE_SHA256),
        ("target", TARGET_PATH, TARGET_SHA256),
    ):
        destination = reference_dir / f"{side}.png"
        before_sha = _sha256_file(source)
        shutil.copyfile(source, destination)
        after_sha = _sha256_file(destination)
        if before_sha != expected_sha or after_sha != before_sha:
            raise RuntimeError(
                f"Reference copy SHA mismatch for {side}: expected={expected_sha}, "
                f"before={before_sha}, after={after_sha}"
            )
        records[side] = {
            "source_path": str(source),
            "copied_path": str(destination),
            "sha256_before_copy": before_sha,
            "sha256_after_copy": after_sha,
            "match": True,
        }
    return records


def _run_worker_subprocess(
    architecture: str,
    step: int,
    output_root: Path,
    run_token: str,
) -> Dict[str, Any]:
    spec = CHECKPOINTS[(architecture, step)]
    worker_dir = output_root / "workers" / f"{architecture}__step_{step:03d}"
    log_path = output_root / "logs" / f"{architecture}__step_{step:03d}.log"
    command = [
        sys.executable,
        "-u",
        "-m",
        MODULE_NAME,
        "--mode",
        "worker",
        "--architecture",
        architecture,
        "--step",
        str(step),
        "--checkpoint",
        str(spec["path"]),
        "--worker-output-dir",
        str(worker_dir),
    ]
    environment = os.environ.copy()
    environment.update({
        "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        RUN_TOKEN_ENV: run_token,
    })
    with log_path.open("xb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        pid = process.pid
        exit_code = process.wait()
    report_path = worker_dir / "worker_report.json"
    envelope: Dict[str, Any] = {
        "architecture": architecture,
        "step": step,
        "pid": pid,
        "command": command,
        "exit_code": exit_code,
        "log_path": str(log_path),
        "log_sha256": _sha256_file(log_path),
        "worker_report_path": str(report_path),
        "worker_report_sha256": None,
        "worker_report": None,
        "worker_report_parse_error": None,
        "pid_matches_worker_report": False,
    }
    try:
        metadata = report_path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("worker report is a symlink or is not a regular file")
        report_bytes = report_path.read_bytes()
        envelope["worker_report_sha256"] = hashlib.sha256(report_bytes).hexdigest()
        parsed_report = json.loads(report_bytes.decode("utf-8"))
        if not isinstance(parsed_report, dict):
            raise RuntimeError(
                f"worker report must be a JSON object, got {type(parsed_report).__name__}"
            )
        envelope["worker_report"] = parsed_report
        envelope["pid_matches_worker_report"] = parsed_report.get("pid") == pid
    except BaseException as report_exc:  # noqa: BLE001
        envelope["worker_report_parse_error"] = (
            f"{type(report_exc).__name__}: {report_exc}"
        )
    return envelope


def _index_worker_outputs(
    run: Dict[str, Any],
    architecture: str,
    step: int,
) -> Dict[int, Dict[str, Any]]:
    condition_label = f"{architecture}/step={step}"
    if not isinstance(run, dict):
        raise RuntimeError(f"Worker run for {condition_label} must be dict")
    if (
        run.get("architecture") != architecture
        or type(run.get("step")) is not int
        or run.get("step") != step
    ):
        raise RuntimeError(
            f"Worker run identity mismatch for {condition_label}: "
            f"architecture={run.get('architecture')!r}, step={run.get('step')!r}"
        )
    if type(run.get("pid")) is not int:
        raise RuntimeError(
            f"Worker run PID for {condition_label} must be int (bool forbidden)"
        )
    worker_report = run.get("worker_report")
    if not isinstance(worker_report, dict):
        raise RuntimeError(f"worker_report for {condition_label} must be dict")
    if (
        worker_report.get("status") != "passed"
        or worker_report.get("failures") != []
        or worker_report.get("architecture") != architecture
        or type(worker_report.get("step")) is not int
        or worker_report.get("step") != step
        or type(worker_report.get("pid")) is not int
        or worker_report.get("pid") != run["pid"]
        or worker_report.get("seeds") != list(SEEDS)
    ):
        raise RuntimeError(
            f"Worker report identity mismatch for {condition_label}: "
            f"architecture={worker_report.get('architecture')!r}, "
            f"step={worker_report.get('step')!r}, pid={worker_report.get('pid')!r}, "
            f"expected_pid={run['pid']}"
        )
    outputs = worker_report.get("outputs")
    if not isinstance(outputs, list):
        raise RuntimeError(f"outputs for {condition_label} must be list")
    if len(outputs) != len(SEEDS):
        raise RuntimeError(
            f"outputs length for {condition_label} must be {len(SEEDS)}, "
            f"actual={len(outputs)}"
        )

    indexed: Dict[int, Dict[str, Any]] = {}
    for row_number, item in enumerate(outputs, start=1):
        if not isinstance(item, dict):
            raise RuntimeError(
                f"Output row={row_number} for {condition_label} must be dict"
            )
        seed = item.get("seed")
        if type(seed) is not int:
            raise RuntimeError(
                f"Output row={row_number} field=seed for {condition_label} "
                f"must be int (bool forbidden), actual={seed!r}"
            )
        if seed in indexed:
            raise RuntimeError(
                f"Output row={row_number} has duplicate seed={seed} for {condition_label}"
            )
        if (
            item.get("architecture") != architecture
            or type(item.get("step")) is not int
            or item.get("step") != step
            or type(item.get("worker_pid")) is not int
            or item.get("worker_pid") != run["pid"]
        ):
            raise RuntimeError(
                f"Output row={row_number} identity mismatch for {condition_label}: "
                f"architecture={item.get('architecture')!r}, step={item.get('step')!r}, "
                f"worker_pid={item.get('worker_pid')!r}, expected_pid={run['pid']}"
            )
        expected_rng_reset = {
            "python_seed": seed,
            "numpy_seed": seed,
            "torch_cpu_seed": seed,
            "torch_cuda_seed": seed,
        }
        if item.get("rng_reset") != expected_rng_reset:
            raise RuntimeError(
                f"Output row={row_number} RNG reset mismatch for {condition_label}: "
                f"expected={expected_rng_reset}, actual={item.get('rng_reset')!r}"
            )
        checkpoint = item.get("checkpoint")
        checkpoint_spec = CHECKPOINTS[(architecture, step)]
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("path") != str(checkpoint_spec["path"])
            or checkpoint.get("size_bytes") != checkpoint_spec["size_bytes"]
            or checkpoint.get("sha256_expected") != checkpoint_spec["sha256"]
            or checkpoint.get("sha256_observed") != checkpoint_spec["sha256"]
            or checkpoint.get("match") is not True
        ):
            raise RuntimeError(
                f"Output row={row_number} checkpoint binding mismatch for {condition_label}"
            )
        validated_path = _validate_model_output_path(
            item.get("path"), architecture, step, seed
        )
        item = dict(item)
        item["path"] = str(validated_path)
        indexed[seed] = item

    if set(indexed) != set(SEEDS):
        raise RuntimeError(
            f"Seed set mismatch for {condition_label}: "
            f"expected={list(SEEDS)}, actual={sorted(indexed)}"
        )
    return indexed


def _compare_step000(worker_runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from PIL import Image, ImageChops

    if not isinstance(worker_runs, (list, tuple)) or len(worker_runs) != len(ARCHITECTURES):
        raise RuntimeError(
            f"Step000 requires exactly {len(ARCHITECTURES)} worker runs, "
            f"actual={len(worker_runs) if isinstance(worker_runs, (list, tuple)) else 'non-sequence'}"
        )
    runs_by_architecture: Dict[str, Dict[str, Any]] = {}
    for run in worker_runs:
        if not isinstance(run, dict):
            raise RuntimeError("Each Step000 worker run must be dict")
        architecture = run.get("architecture")
        if architecture not in ARCHITECTURES:
            raise RuntimeError(f"Unexpected Step000 architecture: {architecture!r}")
        if architecture in runs_by_architecture:
            raise RuntimeError(f"Duplicate Step000 worker run for architecture={architecture}")
        runs_by_architecture[architecture] = run
    if set(runs_by_architecture) != set(ARCHITECTURES):
        raise RuntimeError(
            f"Step000 architecture set mismatch: actual={sorted(runs_by_architecture)}"
        )

    historical = _index_worker_outputs(
        runs_by_architecture["historical_no_lora"], "historical_no_lora", 0
    )
    c1 = _index_worker_outputs(
        runs_by_architecture["c1_lora"], "c1_lora", 0
    )
    comparisons: List[Dict[str, Any]] = []
    metadata_fields = (
        "size_bytes",
        "png_sha256",
        "rgb_pixel_sha256",
        "dimensions",
        "shape",
        "native_mode",
        "mode",
    )
    for seed in SEEDS:
        left = historical[seed]
        right = c1[seed]
        left_path = _validate_model_output_path(
            left["path"], "historical_no_lora", 0, seed
        )
        right_path = _validate_model_output_path(right["path"], "c1_lora", 0, seed)
        left_actual = _image_record(left_path)
        right_actual = _image_record(right_path)
        left_report_matches_file = all(
            left.get(field) == left_actual[field] for field in metadata_fields
        )
        right_report_matches_file = all(
            right.get(field) == right_actual[field] for field in metadata_fields
        )
        with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
            if left_image.mode != IMAGE_MODE or right_image.mode != IMAGE_MODE:
                raise RuntimeError(
                    f"Step000 native mode mismatch at seed={seed}: "
                    f"historical={left_image.mode}, c1_lora={right_image.mode}"
                )
            left_rgb = left_image.convert("RGB")
            right_rgb = right_image.convert("RGB")
            shape_equal = left_rgb.size == right_rgb.size
            max_abs_diff = None
            if shape_equal:
                extrema = ImageChops.difference(left_rgb, right_rgb).getextrema()
                max_abs_diff = max(channel_max for _, channel_max in extrema)
        png_equal = left_actual["png_sha256"] == right_actual["png_sha256"]
        pixel_equal = left_actual["rgb_pixel_sha256"] == right_actual["rgb_pixel_sha256"]
        passed = (
            left_report_matches_file
            and right_report_matches_file
            and png_equal
            and pixel_equal
            and shape_equal
            and max_abs_diff == 0
        )
        comparisons.append({
            "seed": seed,
            "historical_output": left_actual,
            "c1_lora_output": right_actual,
            "historical_report_matches_file": left_report_matches_file,
            "c1_lora_report_matches_file": right_report_matches_file,
            "png_sha256_equal": png_equal,
            "rgb_pixel_sha256_equal": pixel_equal,
            "shape_equal": shape_equal,
            "pixel_max_abs_diff": max_abs_diff,
            "passed": passed,
        })
    return comparisons


def _base_report(bindings: Dict[str, Any], references: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return {
        "status": "running",
        "failures": [],
        "created_at_iso": _now_iso(),
        "script": bindings["script"],
        "frozen_bindings": bindings,
        "frozen_inference": FROZEN_INFERENCE,
        "runtime_parent": _runtime_parent(),
        "references": references,
        "worker_runs": [],
        "step000_comparisons": [],
        "generation_manifest": [],
        "completeness": {
            "expected_model_outputs": EXPECTED_OUTPUT_COUNT,
            "actual_model_outputs": 0,
            "one_to_one": False,
        },
    }


def _collect_generation_manifest(worker_runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in worker_runs:
        if not isinstance(run, dict):
            continue
        worker_report = run.get("worker_report")
        if not isinstance(worker_report, dict):
            continue
        for field, partial in (
            ("outputs", False),
            ("outputs_present_at_failure", True),
        ):
            outputs = worker_report.get(field)
            if not isinstance(outputs, list):
                continue
            for output in outputs:
                if not isinstance(output, dict):
                    continue
                architecture = output.get("architecture")
                step = output.get("step")
                seed = output.get("seed")
                if (
                    architecture in ARCHITECTURES
                    and architecture == run.get("architecture")
                    and type(step) is int
                    and step in STEPS
                    and step == run.get("step")
                    and type(seed) is int
                    and seed in SEEDS
                    and type(output.get("worker_pid")) is int
                    and output.get("worker_pid") == run.get("pid")
                ):
                    row = dict(output)
                    row["partial"] = partial
                    rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            row["architecture"],
            row["step"],
            row["seed"],
            row["partial"],
        ),
    )


def _write_failed_report(
    report_path: Path,
    report: Dict[str, Any],
    failure: str,
    *,
    step000_failure: bool,
) -> None:
    report["status"] = "failed"
    failures = report.setdefault("failures", [])
    if not isinstance(failures, list):
        failures = [f"preexisting_failures_field={failures!r}"]
        report["failures"] = failures
    failures.append(failure)
    report["failure_semantics"] = (
        "Step400/500 generation will not execute."
        if step000_failure
        else "Previously completed outputs are preserved; no retry, deletion, or overwrite will occur."
    )
    report["failure_report_guarantee"] = (
        "Best-effort atomic naming under normal writable-filesystem conditions; "
        "this is not a guarantee against a fully unwritable filesystem or power loss."
    )
    worker_runs = report.get("worker_runs", [])
    if isinstance(worker_runs, list) and (FORMAL_OUTPUT_DIR / "workers").is_dir():
        observed_manifest = _collect_generation_manifest(worker_runs)
        complete_manifest = [
            row for row in observed_manifest if row.get("partial") is False
        ]
        partial_manifest = [
            row for row in observed_manifest if row.get("partial") is True
        ]
        report["generation_manifest"] = observed_manifest
        report["generation_manifest_status"] = {
            "status": "partial",
            "expected_model_outputs": EXPECTED_OUTPUT_COUNT,
            "observed_model_outputs": len(observed_manifest),
            "complete_model_outputs": len(complete_manifest),
            "partial_model_outputs": len(partial_manifest),
            "partial_outputs_are_not_formal_matrix_entries": True,
            "success_authority": "visual_compare_report.json status=passed only",
        }
        try:
            report["completeness"] = _completeness(
                complete_manifest, FORMAL_OUTPUT_DIR, worker_runs
            )
        except BaseException as completeness_exc:  # noqa: BLE001
            failures.append(
                "failure_scene_completeness_error: "
                f"{type(completeness_exc).__name__}: {completeness_exc}"
            )
    report["updated_at_iso"] = _now_iso()
    _atomic_write_json(report_path, report)


def _best_effort_write_failed_report(
    report_path: Path,
    report: Dict[str, Any],
    failure: str,
    *,
    step000_failure: bool,
) -> bool:
    try:
        _write_failed_report(
            report_path,
            report,
            failure,
            step000_failure=step000_failure,
        )
        return True
    except BaseException as report_exc:  # noqa: BLE001
        fallback = {
            "status": "failed",
            "failures": [failure, f"failed_report_write_error: {type(report_exc).__name__}: {report_exc}"],
            "failure_semantics": (
                "Step400/500 generation will not execute."
                if step000_failure
                else "Previously completed outputs are preserved; no retry, deletion, or overwrite will occur."
            ),
            "failure_report_guarantee": (
                "Best-effort atomic naming under normal writable-filesystem conditions; "
                "this is not a guarantee against a fully unwritable filesystem or power loss."
            ),
            "updated_at_iso": _now_iso(),
            "traceback": report.get("traceback"),
        }
        try:
            _atomic_write_json(report_path, fallback)
            return True
        except BaseException as fallback_exc:  # noqa: BLE001
            print(
                "Unable to persist failed visual comparison report under current "
                f"filesystem conditions: primary={report_exc!r}, fallback={fallback_exc!r}",
                file=sys.stderr,
                flush=True,
            )
            return False


def _completeness(
    manifest: Sequence[Dict[str, Any]],
    output_root: Path,
    worker_runs: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    expected_conditions = {
        (architecture, step, seed)
        for architecture in ARCHITECTURES
        for step in STEPS
        for seed in SEEDS
    }
    expected_paths = {
        _expected_output_path(output_root, architecture, step, seed): (
            architecture,
            step,
            seed,
        )
        for architecture, step, seed in expected_conditions
    }
    file_failures: List[str] = []
    observed_condition_rows: List[Tuple[str, int, int]] = []
    manifest_paths: List[Path] = []
    metadata_fields = (
        "size_bytes",
        "png_sha256",
        "rgb_pixel_sha256",
        "dimensions",
        "shape",
        "native_mode",
        "mode",
    )

    for row_number, row in enumerate(manifest, start=1):
        if not isinstance(row, dict):
            file_failures.append(
                f"manifest_row_not_dict:row={row_number}:type={type(row).__name__}"
            )
            continue
        architecture = row.get("architecture")
        step = row.get("step")
        seed = row.get("seed")
        if architecture not in ARCHITECTURES:
            file_failures.append(
                f"manifest_architecture_invalid:row={row_number}:actual={architecture!r}"
            )
            continue
        if type(step) is not int or step not in STEPS:
            file_failures.append(
                f"manifest_step_invalid:row={row_number}:actual={step!r}"
            )
            continue
        if type(seed) is not int or seed not in SEEDS:
            file_failures.append(
                f"manifest_seed_invalid:row={row_number}:actual={seed!r}"
            )
            continue
        condition = (architecture, step, seed)
        observed_condition_rows.append(condition)
        try:
            validated_path = _validate_model_output_path(
                row.get("path"), architecture, step, seed, output_root=output_root
            )
        except BaseException as exc:  # noqa: BLE001
            file_failures.append(
                f"manifest_path_invalid:row={row_number}:condition={condition!r}:"
                f"{type(exc).__name__}:{exc}"
            )
            continue
        manifest_paths.append(_lexical_absolute(Path(row["path"])))
        try:
            actual = _image_record(validated_path)
        except BaseException as exc:  # noqa: BLE001
            file_failures.append(
                f"image_record_invalid:row={row_number}:condition={condition!r}:"
                f"{type(exc).__name__}:{exc}"
            )
            continue
        for field in metadata_fields:
            if row.get(field) != actual[field]:
                file_failures.append(
                    f"metadata_mismatch:{validated_path}:{field}:"
                    f"report={row.get(field)!r}:actual={actual[field]!r}"
                )

    observed_conditions = set(observed_condition_rows)
    duplicate_condition_count = len(observed_condition_rows) - len(observed_conditions)
    duplicate_manifest_path_count = len(manifest_paths) - len(set(manifest_paths))
    if duplicate_condition_count:
        file_failures.append(
            f"duplicate_conditions:count={duplicate_condition_count}"
        )
    if duplicate_manifest_path_count:
        file_failures.append(
            f"duplicate_manifest_paths:count={duplicate_manifest_path_count}"
        )

    workers_root = _lexical_absolute(output_root / "workers")
    actual_png_paths: set[Path] = set()
    if not workers_root.is_dir():
        file_failures.append(f"workers_root_missing:{workers_root}")
    else:
        try:
            _assert_no_symlink_components(
                workers_root, "workers root", require_leaf=False
            )
        except BaseException as exc:  # noqa: BLE001
            file_failures.append(f"workers_root_invalid:{type(exc).__name__}:{exc}")
        for raw_path in workers_root.rglob("*"):
            try:
                metadata = raw_path.lstat()
            except FileNotFoundError:
                file_failures.append(f"filesystem_entry_disappeared:{raw_path}")
                continue
            if stat.S_ISLNK(metadata.st_mode):
                file_failures.append(f"symlink_forbidden:{raw_path}")
                continue
            if raw_path.suffix.lower() != ".png":
                continue
            if not stat.S_ISREG(metadata.st_mode):
                file_failures.append(f"png_not_regular_file:{raw_path}")
                continue
            lexical = _lexical_absolute(raw_path)
            try:
                _assert_no_symlink_components(
                    lexical, "filesystem PNG", require_leaf=True
                )
                resolved = lexical.resolve(strict=True)
                resolved.relative_to(_lexical_absolute(output_root).resolve(strict=True))
            except BaseException as exc:  # noqa: BLE001
                file_failures.append(
                    f"filesystem_png_invalid:{raw_path}:{type(exc).__name__}:{exc}"
                )
                continue
            actual_png_paths.add(lexical)

    expected_path_set = set(expected_paths)
    actual_path_set = set(actual_png_paths)
    manifest_path_set = set(manifest_paths)
    for path in sorted(expected_path_set - actual_path_set):
        file_failures.append(f"missing_png:{path}")
    for path in sorted(actual_path_set - expected_path_set):
        file_failures.append(f"unexpected_or_orphan_png:{path}")
    for path in sorted(expected_path_set - manifest_path_set):
        file_failures.append(f"expected_png_missing_from_manifest:{path}")
    for path in sorted(manifest_path_set - expected_path_set):
        file_failures.append(f"manifest_path_not_expected:{path}")

    expected_worker_conditions = {
        (architecture, step) for architecture in ARCHITECTURES for step in STEPS
    }
    actual_worker_condition_rows: List[Tuple[Any, Any]] = []
    worker_pids: List[Any] = []
    worker_schema_ok = True
    for row_number, run in enumerate(worker_runs, start=1):
        if not isinstance(run, dict):
            worker_schema_ok = False
            file_failures.append(f"worker_run_not_dict:row={row_number}")
            continue
        architecture = run.get("architecture")
        step = run.get("step")
        pid = run.get("pid")
        worker_pids.append(pid)
        if (
            architecture not in ARCHITECTURES
            or type(step) is not int
            or step not in STEPS
            or type(pid) is not int
        ):
            worker_schema_ok = False
            file_failures.append(f"worker_run_schema_invalid:row={row_number}")
            continue
        actual_worker_condition_rows.append((architecture, step))
        try:
            _index_worker_outputs(run, architecture, step)
        except BaseException as exc:  # noqa: BLE001
            worker_schema_ok = False
            file_failures.append(
                f"worker_outputs_invalid:row={row_number}:"
                f"{type(exc).__name__}:{exc}"
            )
    actual_worker_conditions = set(actual_worker_condition_rows)
    worker_integrity = (
        worker_schema_ok
        and len(worker_runs) == 6
        and len(actual_worker_condition_rows) == len(actual_worker_conditions)
        and actual_worker_conditions == expected_worker_conditions
        and len(set(worker_pids)) == 6
        and all(run.get("exit_code") == 0 for run in worker_runs if isinstance(run, dict))
        and all(
            run.get("pid_matches_worker_report") is True
            for run in worker_runs
            if isinstance(run, dict)
        )
    )
    one_to_one = (
        len(manifest) == EXPECTED_OUTPUT_COUNT
        and len(observed_condition_rows) == EXPECTED_OUTPUT_COUNT
        and observed_conditions == expected_conditions
        and duplicate_condition_count == 0
        and duplicate_manifest_path_count == 0
        and len(actual_path_set) == EXPECTED_OUTPUT_COUNT
        and actual_path_set == expected_path_set
        and manifest_path_set == expected_path_set
        and not file_failures
        and worker_integrity
    )
    return {
        "expected_model_outputs": EXPECTED_OUTPUT_COUNT,
        "actual_model_outputs": len(manifest),
        "filesystem_png_count": len(actual_path_set),
        "expected_conditions": [list(item) for item in sorted(expected_conditions)],
        "actual_conditions": [list(item) for item in sorted(observed_conditions)],
        "missing_conditions": [list(item) for item in sorted(expected_conditions - observed_conditions)],
        "unexpected_conditions": [list(item) for item in sorted(observed_conditions - expected_conditions)],
        "expected_png_paths": [str(path) for path in sorted(expected_path_set)],
        "actual_png_paths": [str(path) for path in sorted(actual_path_set)],
        "manifest_png_paths": [str(path) for path in sorted(manifest_path_set)],
        "duplicate_condition_count": duplicate_condition_count,
        "duplicate_manifest_path_count": duplicate_manifest_path_count,
        "file_failures": file_failures,
        "worker_integrity": worker_integrity,
        "expected_worker_conditions": [list(item) for item in sorted(expected_worker_conditions)],
        "actual_worker_conditions": [list(item) for item in sorted(actual_worker_conditions)],
        "worker_pids": worker_pids,
        "one_to_one": one_to_one,
    }


def _preflight_main() -> int:
    if PREFLIGHT_OUTPUT_DIR.resolve() == FORMAL_OUTPUT_DIR.resolve():
        raise RuntimeError("Preflight output directory must be separate from formal output directory")
    _claim_output_root(PREFLIGHT_OUTPUT_DIR, "preflight_output_dir")
    report_path = PREFLIGHT_OUTPUT_DIR / "preflight_report.json"
    try:
        bindings, _ = _verify_frozen_inputs()
        report = {
            "status": "passed",
            "failures": [],
            "mode": "preflight",
            "created_at_iso": _now_iso(),
            "formal_output_dir_untouched": str(FORMAL_OUTPUT_DIR),
            "frozen_bindings": bindings,
            "frozen_inference": FROZEN_INFERENCE,
            "runtime_parent": _runtime_parent(),
            "declarations": [
                "No model was loaded.",
                "No GPU was accessed.",
                "No image was generated.",
                "The formal output directory was not written.",
            ],
        }
        _atomic_write_json(report_path, report)
        return 0
    except BaseException as exc:  # noqa: BLE001
        _atomic_write_json(report_path, {
            "status": "failed",
            "failures": [f"{type(exc).__name__}: {exc}"],
            "mode": "preflight",
            "traceback": traceback.format_exc(),
            "created_at_iso": _now_iso(),
            "formal_output_dir_untouched": str(FORMAL_OUTPUT_DIR),
        })
        return 1


def _generate_main() -> int:
    report_path = FORMAL_OUTPUT_DIR / "visual_compare_report.json"
    report: Dict[str, Any] = {}
    run_token: str | None = None
    completed_conditions: List[Tuple[str, int]] = []
    step000_phase = True
    claimed = False

    # No formal failure-report obligation exists until this process creates the claim.
    try:
        try:
            _claim_output_root(FORMAL_OUTPUT_DIR, "formal_output_dir")
            claimed = True
        except _ClaimedOutputRootError:
            claimed = True
            raise
        report = {
            "status": "running",
            "failures": [],
            "phase": "claimed",
            "created_at_iso": _now_iso(),
            "worker_runs": [],
            "step000_comparisons": [],
            "generation_manifest": [],
            "generation_manifest_status": {
                "status": "partial",
                "expected_model_outputs": EXPECTED_OUTPUT_COUNT,
                "observed_model_outputs": 0,
                "success_authority": "visual_compare_report.json status=passed only",
            },
        }
        _atomic_write_json(report_path, report)

        run_token = secrets.token_hex(32)
        report["script"] = {
            "path": str(SCRIPT_PATH),
            "size_bytes": SCRIPT_PATH.stat().st_size,
            "sha256": _sha256_file(SCRIPT_PATH),
        }
        report["runtime_parent"] = _runtime_parent()
        report["run_authorization"] = {
            "parent_pid": os.getpid(),
            "run_token_sha256": _sha256_text(run_token),
            "state_path": str(_run_state_path()),
        }
        _atomic_write_json(report_path, report)

        (FORMAL_OUTPUT_DIR / "logs").mkdir(exist_ok=False)
        (FORMAL_OUTPUT_DIR / "workers").mkdir(exist_ok=False)
        _publish_run_state(
            run_token,
            phase="step000",
            authorized_condition=None,
            completed_conditions=completed_conditions,
        )
        bindings, _ = _verify_frozen_inputs()
        references = _copy_references(FORMAL_OUTPUT_DIR)
        report = _base_report(bindings, references)
        report["phase"] = "step000"
        report["run_authorization"] = {
            "parent_pid": os.getpid(),
            "run_token_sha256": _sha256_text(run_token),
            "state_path": str(_run_state_path()),
        }
        report["generation_manifest_status"] = {
            "status": "partial",
            "expected_model_outputs": EXPECTED_OUTPUT_COUNT,
            "observed_model_outputs": 0,
            "success_authority": "visual_compare_report.json status=passed only",
        }
        _atomic_write_json(report_path, report)

        for architecture in ARCHITECTURES:
            condition = (architecture, 0)
            _publish_run_state(
                run_token,
                phase="step000",
                authorized_condition=condition,
                completed_conditions=completed_conditions,
            )
            run = _run_worker_subprocess(
                architecture, 0, FORMAL_OUTPUT_DIR, run_token
            )
            report["worker_runs"].append(run)
            worker_ok = (
                run["exit_code"] == 0
                and run["worker_report_parse_error"] is None
                and run["pid_matches_worker_report"] is True
                and isinstance(run["worker_report"], dict)
                and run["worker_report"].get("status") == "passed"
            )
            if not worker_ok:
                raise RuntimeError(
                    f"Step000 worker failed: architecture={architecture}, "
                    f"exit_code={run['exit_code']}; Step400/500 generation will not execute."
                )
            completed_conditions.append(condition)
            _publish_run_state(
                run_token,
                phase="step000",
                authorized_condition=None,
                completed_conditions=completed_conditions,
            )
            _atomic_write_json(report_path, report)

        comparisons = _compare_step000(report["worker_runs"])
        report["step000_comparisons"] = comparisons
        report["step000_gate_passed"] = len(comparisons) == len(SEEDS) and all(
            comparison["passed"] for comparison in comparisons
        )
        if not report["step000_gate_passed"]:
            raise RuntimeError(
                "Step000 bit-exact gate failed for one or more preregistered seeds; "
                "Step400/500 generation will not execute."
            )
        _publish_run_state(
            run_token,
            phase="later",
            authorized_condition=None,
            completed_conditions=completed_conditions,
        )
        report["phase"] = "later"
        _atomic_write_json(report_path, report)

        step000_phase = False
        for step in (400, 500):
            for architecture in ARCHITECTURES:
                condition = (architecture, step)
                _publish_run_state(
                    run_token,
                    phase="later",
                    authorized_condition=condition,
                    completed_conditions=completed_conditions,
                )
                run = _run_worker_subprocess(
                    architecture, step, FORMAL_OUTPUT_DIR, run_token
                )
                report["worker_runs"].append(run)
                worker_ok = (
                    run["exit_code"] == 0
                    and run["worker_report_parse_error"] is None
                    and run["pid_matches_worker_report"] is True
                    and isinstance(run["worker_report"], dict)
                    and run["worker_report"].get("status") == "passed"
                )
                if not worker_ok:
                    raise RuntimeError(
                        f"Later-phase worker failed after preserving earlier outputs: "
                        f"architecture={architecture}, step={step}, "
                        f"exit_code={run['exit_code']}"
                    )
                completed_conditions.append(condition)
                _publish_run_state(
                    run_token,
                    phase="later",
                    authorized_condition=None,
                    completed_conditions=completed_conditions,
                )
                _atomic_write_json(report_path, report)

        generation_manifest = _collect_generation_manifest(report["worker_runs"])
        completeness = _completeness(
            generation_manifest, FORMAL_OUTPUT_DIR, report["worker_runs"]
        )
        report["generation_manifest"] = generation_manifest
        report["completeness"] = completeness
        report["generation_manifest_status"] = {
            "status": "partial",
            "expected_model_outputs": EXPECTED_OUTPUT_COUNT,
            "observed_model_outputs": len(generation_manifest),
            "success_authority": "visual_compare_report.json status=passed only",
        }
        if not completeness["one_to_one"]:
            raise RuntimeError(
                f"18-output completeness check failed: {completeness}"
            )

        manifest_path = FORMAL_OUTPUT_DIR / "generation_manifest.json"
        manifest_payload = {
            "status": "complete",
            "expected_model_outputs": EXPECTED_OUTPUT_COUNT,
            "observed_model_outputs": len(generation_manifest),
            "success_authority": "visual_compare_report.json status=passed only",
            "outputs": generation_manifest,
        }
        _atomic_write_json(manifest_path, manifest_payload)
        report["generation_manifest_status"] = {
            "status": "complete",
            "expected_model_outputs": EXPECTED_OUTPUT_COUNT,
            "observed_model_outputs": len(generation_manifest),
            "success_authority": "visual_compare_report.json status=passed only",
        }
        report["generation_manifest_file"] = {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
        }
        _publish_run_state(
            run_token,
            phase="passed",
            authorized_condition=None,
            completed_conditions=completed_conditions,
        )
        report["run_state"] = {
            "path": str(_run_state_path()),
            "sha256": _sha256_file(_run_state_path()),
        }
        report["phase"] = "complete"
        report["status"] = "passed"
        report["failures"] = []
        report["completed_at_iso"] = _now_iso()
        report["declarations"] = [
            "Exactly six fresh worker subprocesses were used: one per architecture/checkpoint.",
            "Step000 passed for every preregistered seed before Step400/500 workers launched.",
            "All Step400 and Step500 conditions were retained; no seed or checkpoint was selected by result.",
            "generation_manifest.json is not success authority; only visual_compare_report.json status=passed is authoritative.",
            "No training, backward, or optimizer.step was performed.",
        ]
        _atomic_write_json(report_path, report)
        return 0
    except BaseException as exc:  # noqa: BLE001
        if not claimed:
            raise
        report.setdefault("status", "running")
        report.setdefault("failures", [])
        report.setdefault("worker_runs", [])
        report["traceback"] = traceback.format_exc()
        if run_token is not None:
            try:
                _publish_run_state(
                    run_token,
                    phase="failed",
                    authorized_condition=None,
                    completed_conditions=completed_conditions,
                )
            except BaseException as state_exc:  # noqa: BLE001
                failures = report.setdefault("failures", [])
                if isinstance(failures, list):
                    failures.append(f"run_state_failure: {state_exc!r}")
        _best_effort_write_failed_report(
            report_path,
            report,
            f"{type(exc).__name__}: {exc}",
            step000_failure=step000_phase,
        )
        return 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "generate", "worker"))
    parser.add_argument("--architecture", choices=ARCHITECTURES, help=argparse.SUPPRESS)
    parser.add_argument("--step", type=int, choices=STEPS, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    worker_values = (args.architecture, args.step, args.checkpoint, args.worker_output_dir)
    if args.mode == "worker" and any(value is None for value in worker_values):
        parser.error("worker mode requires internal architecture, step, checkpoint, and worker output")
    if args.mode != "worker" and any(value is not None for value in worker_values):
        parser.error("worker-only arguments are forbidden in parent modes")
    return args


def main() -> int:
    args = _parse_args()
    if args.mode == "preflight":
        return _preflight_main()
    if args.mode == "generate":
        return _generate_main()
    return _worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
