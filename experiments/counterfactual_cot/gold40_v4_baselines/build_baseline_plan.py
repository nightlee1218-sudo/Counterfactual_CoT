#!/usr/bin/env python3
"""Generate auditable Gold40 v4 baseline plans; this tool never starts training.

The paired-v4 main method has a reviewed-split-aware trainer. Other ablations
remain deliberately fail-fast until their own implementations are supplied and
audited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = ROOT / "experiments/counterfactual_cot/outputs"
APPROVED = {"approved", "accepted", "reviewed", "human_verified", "verified", "pass", "passed"}
REJECTED = {"candidate", "unreviewed", "pending", "rejected", "manual_review_required", "needs_review"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"Reviewed manifest is empty: {path}")
    return rows


def _review_values(value: Any, is_status_value: bool = False) -> Iterable[str]:
    status_fields = {
        "status", "review_status", "review_decision", "approval_status", "review_state", "review", "approval",
    }
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = key.lower()
            if lowered in {"approved", "ready_for_training"} and nested is True:
                yield "approved"
            yield from _review_values(nested, is_status_value or lowered in status_fields)
    elif isinstance(value, list):
        for nested in value:
            yield from _review_values(nested, is_status_value)
    elif isinstance(value, str) and is_status_value:
        yield value.strip().lower()


def _record_id(record: dict[str, Any], index: int) -> str:
    for key in ("split_record_id", "macro_pair_id", "pair_id", "id"):
        if isinstance(record.get(key), str) and record[key]:
            return record[key]
    return f"row:{index}"


def _validate_reviewed(path: Path, role: str) -> dict[str, Any]:
    rows = _load_jsonl(path)
    rejected: list[str] = []
    missing_approval: list[str] = []
    seen: set[str] = set()
    duplicate_ids: list[str] = []
    for index, row in enumerate(rows, 1):
        row_id = _record_id(row, index)
        if row_id in seen:
            duplicate_ids.append(row_id)
        seen.add(row_id)
        statuses = set(_review_values(row))
        if statuses & REJECTED:
            rejected.append(f"{row_id}={sorted(statuses & REJECTED)}")
        if not statuses & APPROVED:
            missing_approval.append(row_id)
    if rejected or missing_approval or duplicate_ids:
        fragments = []
        if rejected:
            fragments.append(f"rejected/non-final review statuses: {rejected[:5]}")
        if missing_approval:
            fragments.append(f"records without explicit approved review status: {missing_approval[:5]}")
        if duplicate_ids:
            fragments.append(f"duplicate record identifiers: {duplicate_ids[:5]}")
        raise ValueError(f"{role} manifest is not an approved reviewed split: " + "; ".join(fragments))
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "records": len(rows),
        "record_id_examples": [_record_id(row, index) for index, row in enumerate(rows[:5], 1)],
        "review_gate": "every JSONL record has an explicit final approved review status; candidate/unreviewed/pending statuses are rejected",
    }


def _metadata_manifests(path: Path) -> tuple[Path | None, Path | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    def resolve(keys: tuple[str, ...]) -> Path | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str):
                candidate = Path(value)
                candidate = candidate if candidate.is_absolute() else path.parent / candidate
                if candidate.is_file():
                    return candidate.resolve()
        return None
    return resolve(("train_manifest", "train_manifest_path", "reviewed_train_manifest")), resolve(("eval_manifest", "evaluation_manifest", "holdout_manifest", "test_manifest"))


def _first_existing(directory: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_split(split: Path, train_override: Path | None, eval_override: Path | None) -> tuple[Path, Path]:
    train_path, eval_path = train_override, eval_override
    if split.is_file() and split.suffix == ".jsonl":
        train_path = train_path or split.resolve()
    elif split.is_file() and split.suffix == ".json":
        metadata_train, metadata_eval = _metadata_manifests(split)
        train_path, eval_path = train_path or metadata_train, eval_path or metadata_eval
    elif split.is_dir():
        for metadata_name in ("reviewed_split.json", "split.json", "split_metadata.json", "manifest.json"):
            metadata = split / metadata_name
            if metadata.is_file():
                metadata_train, metadata_eval = _metadata_manifests(metadata)
                train_path, eval_path = train_path or metadata_train, eval_path or metadata_eval
                if train_path and eval_path:
                    break
        train_path = train_path or _first_existing(split, (
            "reviewed_train.jsonl", "train_manifest.jsonl", "train.jsonl", "gold40_train_reviewed_v4.jsonl",
        ))
        eval_path = eval_path or _first_existing(split, (
            "reviewed_eval.jsonl", "eval_manifest.jsonl", "evaluation_manifest.jsonl", "holdout_manifest.jsonl", "test_manifest.jsonl", "eval.jsonl", "test.jsonl",
        ))
        if not train_path or not eval_path:
            jsonls = sorted(split.glob("*.jsonl"))
            train_candidates = [item for item in jsonls if re.search(r"(^|[_-])train([_.-]|$)", item.name, re.I)]
            eval_candidates = [item for item in jsonls if re.search(r"eval|holdout|test", item.name, re.I)]
            train_path = train_path or (train_candidates[0].resolve() if len(train_candidates) == 1 else None)
            eval_path = eval_path or (eval_candidates[0].resolve() if len(eval_candidates) == 1 else None)
    else:
        raise ValueError(f"--reviewed-split does not exist: {split}")
    if not train_path or not eval_path:
        raise ValueError("Could not resolve both reviewed train and eval manifests. Use a split directory with reviewed_train.jsonl/reviewed_eval.jsonl, metadata keys train_manifest/eval_manifest, or pass --train-manifest and --eval-manifest.")
    if train_path.resolve() == eval_path.resolve():
        raise ValueError("Train and evaluation manifests must be distinct to prevent evaluation leakage.")
    return train_path.resolve(), eval_path.resolve()


def _baseline_specs(output_root: Path) -> list[dict[str, Any]]:
    unavailable = "No audited split-aware training entrypoint exists for this ablation. The registered Gold40 general trainer implements paired_v4_main only; launcher intentionally exits before torchrun."
    common_eval = {
        "manifest": "reviewed_eval_manifest",
        "protocol": "cold-load base and selected checkpoints; use fixed held-out cases and matched deterministic seeds/noise; retain per-case artifacts, hashes, and blinded human review.",
        "primary_metrics": ["approved semantic edit success", "preservation success", "image-quality/manual-review pass"],
    }
    return [
        {
            "id": "v3_single_target",
            "label": "v3 single-target baseline",
            "input_manifest": "reviewed_train_manifest (target-state records only; one branch per optimizer update)",
            "training_strategy_difference": "Removes v4 source/target paired macro-step. Each target-state example is optimized independently; there is no matched source-state branch, same-update parameter sharing, or matched-noise constraint.",
            "output_dir": str(output_root / "gold40_v4_baselines" / "v3_single_target"),
            "expected_evaluation": {**common_eval, "condition_test": "Evaluate target-state edit following. Report paired CoT-switch results only as a non-causal diagnostic because this baseline never trains the counterfactual pair."},
            "implementation": {"state": "unimplemented_fail_fast", "reason": unavailable, "related_legacy_entry": "experiments/counterfactual_cot/full_backbone_lora_region1024_fsdp_v3/train.py"},
        },
        {
            "id": "paired_v4_main",
            "label": "paired v4 main method",
            "input_manifest": "reviewed_train_manifest (paired source_state + target_state branches per macro pair)",
            "training_strategy_difference": "Reference method: 0.5 source-state loss plus 0.5 target-state loss before one update, identical pack seed/timestep/forward noise across branches, and native/high/pure-noise mixture 50/40/10.",
            "output_dir": str(output_root / "gold40_v4_baselines" / "paired_v4_main"),
            "expected_evaluation": {**common_eval, "condition_test": "At t=1 and matched noise, compare correct and swapped Target-CoT for both source and target states; report correct-vs-swapped loss gap and paired switch success only when both conditions pass."},
            "implementation": {
                "state": "implemented_training_only",
                "entrypoint": "experiments/counterfactual_cot/gold40_dual_target_cot1024_fsdp_v4/train.py",
                "manifest_argument": "--manifest",
                "launcher_behavior": "torchrun --standalone --nproc_per_node=2; generator does not execute it",
                "evaluation_state": "protocol_specified_but_no_registered_split-aware evaluator",
            },
        },
        {
            "id": "no_paired_branch",
            "label": "no paired branch ablation",
            "input_manifest": "reviewed_train_manifest flattened into independently scheduled source_state and target_state branch records",
            "training_strategy_difference": "Uses the same branch content as v4 but performs independent optimizer updates, removing joint 0.5/0.5 macro-steps, same-update parameters, and the paired branch gradient comparison.",
            "output_dir": str(output_root / "gold40_v4_baselines" / "no_paired_branch"),
            "expected_evaluation": {**common_eval, "condition_test": "Run the same two-state correct-vs-swapped Target-CoT matrix. Pairwise switch success is reported descriptively, not as evidence of paired training."},
            "implementation": {"state": "unimplemented_fail_fast", "reason": unavailable},
        },
        {
            "id": "independent_noise",
            "label": "independent-noise paired ablation",
            "input_manifest": "reviewed_train_manifest (paired branches retained)",
            "training_strategy_difference": "Retains v4 paired 0.5/0.5 macro-steps but independently samples forward noise for source_state and target_state; it removes the same-noise causal control while keeping all other mixture probabilities, including pure-noise.",
            "output_dir": str(output_root / "gold40_v4_baselines" / "independent_noise"),
            "expected_evaluation": {**common_eval, "condition_test": "Evaluation itself remains matched-noise and uses the v4 switch matrix, so effects are comparable to the main method."},
            "implementation": {"state": "unimplemented_fail_fast", "reason": unavailable},
        },
        {
            "id": "no_pure_noise",
            "label": "no-pure-noise paired ablation",
            "input_manifest": "reviewed_train_manifest (paired branches retained)",
            "training_strategy_difference": "Retains v4 paired branches and same-noise control, but removes t=1 pure-noise updates; native/high modes are renormalized from 50/40 to 5/9 and 4/9. This is an alternative member of independent_noise_or_no_pure_noise, not combined with independent_noise.",
            "output_dir": str(output_root / "gold40_v4_baselines" / "no_pure_noise"),
            "expected_evaluation": {**common_eval, "condition_test": "Evaluate the usual matched-noise switch matrix, with a prespecified t=1 subset to test the missing pure-noise training exposure."},
            "implementation": {"state": "unimplemented_fail_fast", "reason": unavailable},
        },
        {
            "id": "cot_shuffle",
            "label": "Target-CoT shuffle control",
            "input_manifest": "reviewed_train_manifest (paired branches retained; deterministic within-macro-pair Target-CoT permutation)",
            "training_strategy_difference": "Retains v4 images, paired macro-steps, loss weights, and matched noise, but assigns the opposite branch Target-CoT to each target image. The seed-derived permutation must be checkpointed/audited; source/target image supervision is not shuffled.",
            "output_dir": str(output_root / "gold40_v4_baselines" / "cot_shuffle"),
            "expected_evaluation": {**common_eval, "condition_test": "Use the same correct-vs-swapped matrix. A strong correct-condition advantage is not expected; any such result requires leakage investigation before interpretation."},
            "implementation": {"state": "unimplemented_fail_fast", "reason": unavailable},
        },
    ]


def _launcher(plan_sha256: str, baseline: dict[str, Any], train: dict[str, Any], evaluation: dict[str, Any]) -> str:
    header = f'''#!/usr/bin/env bash
# Generated by build_baseline_plan.py; generation itself never starts training.
set -euo pipefail
NPROC_PER_NODE=2
PLAN_SHA256="{plan_sha256}"
BASELINE_ID="{baseline["id"]}"
TRAIN_MANIFEST="{train["path"]}"
EVAL_MANIFEST="{evaluation["path"]}"
'''
    if baseline["id"] != "paired_v4_main":
        return header + f'''OUTPUT_DIR="{baseline["output_dir"]}"
printf '%s\\n' "Refusing to launch $BASELINE_ID (plan $PLAN_SHA256)."
printf '%s\\n' "nproc_per_node=$NPROC_PER_NODE"
printf '%s\\n' "Reviewed train manifest: $TRAIN_MANIFEST"
printf '%s\\n' "Reviewed eval manifest:  $EVAL_MANIFEST"
printf '%s\\n' "Requested output directory: $OUTPUT_DIR"
printf '%s\\n' "FAIL-FAST: no audited split-aware training implementation is registered for this baseline."
printf '%s\\n' "Do not replace this guard with a torchrun invocation until an implementation and its manifest contract are reviewed."
exit 2
'''
    return header + f'''# Manually running this template starts only the implemented paired-v4 training method.
ROOT="{ROOT}"
ENTRYPOINT="$ROOT/experiments/counterfactual_cot/gold40_dual_target_cot1024_fsdp_v4/train.py"
MODEL_DIR="${{MODEL_DIR:-/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT}}"
GPU_IDS="${{GPU_IDS:-0,1}}"
OUTPUT_DIR="${{OUTPUT_DIR:-{baseline["output_dir"]}}}"
STEPS="${{STEPS:-500}}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export NCCL_ASYNC_ERROR_HANDLING=1
printf '%s\\n' "Launching $BASELINE_ID with nproc_per_node=$NPROC_PER_NODE (plan $PLAN_SHA256)."
printf '%s\\n' "Held-out evaluation manifest is locked in the plan and is not consumed by the training entrypoint: $EVAL_MANIFEST"
cd "$ROOT"
exec torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" "$ENTRYPOINT" \\
  --manifest "$TRAIN_MANIFEST" \\
  --output-dir "$OUTPUT_DIR" \\
  --llm-path "$MODEL_DIR" \\
  --vae-path "$MODEL_DIR/ae.safetensors" \\
  --checkpoint-path "$MODEL_DIR/ema.safetensors" \\
  --steps "$STEPS" \\
  --lr 3e-5 \\
  --lora-r 8 \\
  --lora-alpha 16 \\
  --activation-checkpointing \\
  --checkpoint-every 10 \\
  --seed 1234 \\
  --variant paired_v4_main
'''


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate auditable Gold40 v4 baseline plans and fail-fast launchers; never trains.")
    parser.add_argument("--reviewed-split", required=True, type=Path, help="Reviewed split directory, split metadata JSON, or JSONL paired with explicit --eval-manifest.")
    parser.add_argument("--train-manifest", type=Path, help="Explicit reviewed training JSONL override.")
    parser.add_argument("--eval-manifest", type=Path, help="Explicit reviewed held-out evaluation JSONL override.")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for experimental_plan.json and launchers/.")
    parser.add_argument("--force", action="store_true", help="Allow replacing generated plan/template files in --output-dir.")
    args = parser.parse_args()
    try:
        train_path, eval_path = _resolve_split(args.reviewed_split.resolve(), args.train_manifest, args.eval_manifest)
        train, evaluation = _validate_reviewed(train_path, "train"), _validate_reviewed(eval_path, "evaluation")
        output_dir = args.output_dir.resolve()
        plan_path = output_dir / "experimental_plan.json"
        launch_dir = output_dir / "launchers"
        if (plan_path.exists() or launch_dir.exists()) and not args.force:
            raise ValueError(f"Refusing to overwrite existing generated artifacts in {output_dir}; use --force after auditing the diff.")
        specs = _baseline_specs(DEFAULT_OUTPUT_ROOT)
        plan = {
            "schema_version": "gold40_v4_baseline_plan_v1",
            "created_at": _utc_now(),
            "generator": str(Path(__file__).resolve()),
            "training_started": False,
            "nproc_per_node": 2,
            "reviewed_split": {"train": train, "evaluation": evaluation},
            "global_evaluation_contract": "All variants must use the identical reviewed held-out manifest and locked seed/noise schedule. Report automated metrics with per-case evidence and blinded human review; do not compare outputs across changed manifests.",
            "baselines": specs,
        }
        serialized = json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        plan_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        output_dir.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(serialized, encoding="utf-8")
        launch_dir.mkdir(parents=True, exist_ok=True)
        for baseline in specs:
            path = launch_dir / f"launch_{baseline['id']}.sh"
            path.write_text(_launcher(plan_sha256, baseline, train, evaluation), encoding="utf-8")
            path.chmod(path.stat().st_mode | 0o111)
        print(json.dumps({"status": "generated_no_training_started", "plan": str(plan_path), "plan_sha256": plan_sha256, "launchers": str(launch_dir), "baseline_count": len(specs)}, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
