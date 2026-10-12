#!/usr/bin/env python3
"""Build a non-destructive index and symlink view for counterfactual-CoT outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DEFAULT_VIEW = OUTPUTS / "_organized_view_v1"
DEFAULT_JSONL = OUTPUTS / "experiment_index_v1.jsonl"
DEFAULT_MARKDOWN = OUTPUTS / "EXPERIMENT_INDEX_V1.md"

CATEGORY_LABELS = {
    "00_key_evidence": "关键证据与当前入口",
    "10_dataset_and_provenance": "数据、配置与来源",
    "20_training_runs": "训练运行与 checkpoint",
    "30_inference_and_visualization": "推理、诊断图与可视化",
    "40_evaluation_and_comparisons": "固定评估与对照",
    "50_preflight_dryrun_smoke": "Preflight、dry-run、smoke 与资源门禁",
    "60_engineering_audits": "工程审计、对齐与复现检查",
    "70_failed_partial_historical": "失败、不完整与历史问题记录",
    "80_uncertain_unclassified": "信息不足，待按报告确认",
}

KEY_OVERRIDES: dict[str, dict[str, Any]] = {
    "gold40_v3_overfit_400steps_lr1e6_singlepack": {
        "display_name": "dataset=gold40_v3__activity=training__method=bridge14_no_lora__objective=full_image_flow_mse__lr=1e-6__steps=400__role=A_capacity_diagnostic",
        "category": "20_training_runs",
        "lifecycle_status": "completed",
        "evidence_level": "optimization_and_training_artifacts_not_semantic_success",
        "dataset_scope": "gold40_v3_training_set_40_pairs",
        "method": "14_bridge_head_tail_tensors_no_lora",
        "objective": "full_image_flow_matching_mse",
        "learning_rate": 1e-6,
        "steps": 400,
        "evidence_boundary": "Gold40 is the training set; this run is a capacity diagnostic, not generalization evidence.",
    },
    "gold40_v3_lora_L24-27_r8_a16_lr1e6_singlepack_400steps": {
        "display_name": "dataset=gold40_v3__activity=training__method=bridge14_plus_lora_L24-27_r8_a16__objective=full_image_flow_mse__lr=1e-6__steps=400__role=B_capacity_diagnostic",
        "category": "20_training_runs",
        "lifecycle_status": "completed",
        "evidence_level": "engineering_verified_training_artifacts_not_semantic_success",
        "dataset_scope": "gold40_v3_training_set_40_pairs",
        "method": "14_bridge_head_tail_tensors_plus_L24-27_LoRA_r8_a16",
        "objective": "full_image_flow_matching_mse",
        "learning_rate": 1e-6,
        "steps": 400,
        "evidence_boundary": "The run verifies LoRA training and reload; it does not establish visual grounding.",
    },
    "gold40_v3_ab_aligned_inference_nt25_protocol_v1": {
        "display_name": "dataset=gold40_v3__activity=aligned_inference__conditions=A_bridge14_and_B_bridge14_plus_lora__steps=000_and_400__sampling=nt25_seed1234_cfgT4_cfgI2_1024__role=capacity_visual_audit",
        "category": "00_key_evidence",
        "lifecycle_status": "completed",
        "evidence_level": "aligned_visual_outputs_unscored",
        "dataset_scope": "gold40_v3_training_set_40_pairs_all_5_intervention_types",
        "method": "A_bridge14_vs_B_bridge14_plus_LoRA",
        "objective": "aligned_inference_only",
        "learning_rate": 1e-6,
        "steps": [0, 400],
        "evidence_boundary": "A/B changes trainable parameterization, not CoT conditioning. Images are training-set capacity outputs, not generalization results.",
    },
    "gold40_v3_lora_micro4_400steps": {
        "display_name": "dataset=micro4_object_and_position__activity=training__method=bridge14_plus_lora_L24-27_r8_a16__objective=full_image_flow_mse__lr=1e-6__steps=400__role=low_lr_capacity_probe",
        "category": "20_training_runs",
        "lifecycle_status": "completed",
        "evidence_level": "training_artifacts",
        "dataset_scope": "micro4_training_subset_object_and_position",
        "method": "14_bridge_head_tail_tensors_plus_L24-27_LoRA_r8_a16",
        "objective": "full_image_flow_matching_mse",
        "learning_rate": 1e-6,
        "steps": 400,
        "evidence_boundary": "Four training pairs only; use as an optimization probe.",
    },
    "gold40_v3_lora_micro4_deterministic_eval_v4": {
        "display_name": "dataset=micro4_object_and_position__activity=fixed_mse_evaluation__method=bridge14_plus_lora_L24-27_r8_a16__lr=1e-6__steps=000_to_400__result=3.77pct_mse_improvement_below_5pct_gate",
        "category": "40_evaluation_and_comparisons",
        "lifecycle_status": "completed",
        "evidence_level": "optimization_metric_only",
        "dataset_scope": "micro4_training_subset_object_and_position",
        "method": "14_bridge_head_tail_tensors_plus_L24-27_LoRA_r8_a16",
        "objective": "fixed_flow_matching_mse_evaluation",
        "learning_rate": 1e-6,
        "steps": [0, 400],
        "evidence_boundary": "MSE is not a semantic grounding metric.",
    },
    "gold40_v3_lora_micro4_lr3e5_500steps": {
        "display_name": "dataset=micro4_object_and_position__activity=training__method=bridge14_plus_lora_L24-27_r8_a16__objective=full_image_flow_mse__lr=3e-5__steps=500__role=high_lr_capacity_probe",
        "category": "20_training_runs",
        "lifecycle_status": "completed",
        "evidence_level": "training_artifacts",
        "dataset_scope": "micro4_training_subset_object_and_position",
        "method": "14_bridge_head_tail_tensors_plus_L24-27_LoRA_r8_a16",
        "objective": "full_image_flow_matching_mse",
        "learning_rate": 3e-5,
        "steps": 500,
        "evidence_boundary": "Four training pairs only; use as an optimization probe.",
    },
    "gold40_v3_lora_micro4_lr3e5_500steps_deterministic_eval_v1": {
        "display_name": "dataset=micro4_object_and_position__activity=fixed_mse_evaluation__method=bridge14_plus_lora_L24-27_r8_a16__lr=3e-5__steps=000_to_500__result=56.61pct_mse_improvement",
        "category": "00_key_evidence",
        "lifecycle_status": "completed",
        "evidence_level": "optimization_metric_only",
        "dataset_scope": "micro4_training_subset_object_and_position",
        "method": "14_bridge_head_tail_tensors_plus_L24-27_LoRA_r8_a16",
        "objective": "fixed_flow_matching_mse_evaluation",
        "learning_rate": 3e-5,
        "steps": [0, 500],
        "evidence_boundary": "Strong objective fitting does not establish visual grounding.",
    },
    "gold40_v3_lora_micro4_lr3e5_500steps_diag_images_v1": {
        "display_name": "dataset=micro4_object_and_position__activity=diagnostic_inference__method=bridge14_plus_lora_L24-27_r8_a16__lr=3e-5__checkpoints=000_010_100_300_500__sampling=nt25_cfgT4_cfgI2_1024",
        "category": "00_key_evidence",
        "lifecycle_status": "completed",
        "evidence_level": "visual_outputs_unscored",
        "dataset_scope": "micro4_training_subset_object_and_position",
        "method": "14_bridge_head_tail_tensors_plus_L24-27_LoRA_r8_a16",
        "objective": "diagnostic_inference",
        "learning_rate": 3e-5,
        "steps": [0, 10, 100, 300, 500],
        "evidence_boundary": "Diagnostic images are not formal semantic scores.",
    },
    "single_pair_overfit_diag_ball": {
        "display_name": "dataset=C1_single_pair_living_room_red_ball_left_to_right__activity=training_and_inference__method=bridge14_no_lora__objective=full_image_flow_mse__lr=1e-4__steps=500__finding=target_addition_without_reliable_source_deletion",
        "category": "00_key_evidence",
        "lifecycle_status": "completed",
        "evidence_level": "single_pair_positive_control_and_failure_case",
        "dataset_scope": "C1_single_training_pair_spatial_position_change",
        "method": "14_bridge_head_tail_tensors_no_lora",
        "objective": "full_image_flow_matching_mse",
        "learning_rate": 1e-4,
        "steps": 500,
        "evidence_boundary": "Single-pair mechanism case; not representative of all Gold40 tasks.",
    },
    "single_pair_overfit_diag_ball_lr3e5_500steps": {
        "display_name": "dataset=C1_single_pair_living_room_red_ball_left_to_right__activity=reproduced_training_and_inference__method=bridge14_no_lora__objective=full_image_flow_mse__lr=3e-5__steps=500__finding=no_reliable_source_deletion",
        "category": "00_key_evidence",
        "lifecycle_status": "completed_reproduced",
        "evidence_level": "single_pair_duration_and_learning_rate_diagnostic",
        "dataset_scope": "C1_single_training_pair_spatial_position_change",
        "method": "14_bridge_head_tail_tensors_no_lora",
        "objective": "full_image_flow_matching_mse",
        "learning_rate": 3e-5,
        "steps": 500,
        "evidence_boundary": "Single-pair mechanism case; the first 300 training rows were exactly reproduced.",
    },
    "gold40_v3_lora_c1_single_ball_region_split_v1_500steps": {
        "display_name": "dataset=C1_single_pair_living_room_red_ball_left_to_right__activity=training__method=bridge14_plus_lora_L24-27_r8_a16__objective=region_split_edit50_preserve50__lr=1e-6__steps=500__finding=edit_mse_best_gain_0.339pct",
        "category": "00_key_evidence",
        "lifecycle_status": "completed",
        "evidence_level": "loss_ablation_optimization_and_visual_diagnostic",
        "dataset_scope": "C1_single_training_pair_spatial_position_change",
        "method": "14_bridge_head_tail_tensors_plus_L24-27_LoRA_r8_a16",
        "objective": "region_split_edit_0.5_preserve_0.5",
        "learning_rate": 1e-6,
        "steps": 500,
        "evidence_boundary": "Shows that this region weighting at this LR is insufficient; it does not reject all region-aware objectives.",
    },
    "c1_prompt_ablation_v1": {
        "display_name": "dataset=C1_single_pair_living_room_red_ball_left_to_right__activity=prompt_ood_inference_ablation__profiles=6__checkpoints=4__generation_seeds=3__images=72__boundary=only_legacy_profile_train_matched",
        "category": "00_key_evidence",
        "lifecycle_status": "completed",
        "evidence_level": "prompt_ood_visual_diagnostic_unscored",
        "dataset_scope": "C1_single_training_pair_spatial_position_change",
        "method": "four_existing_checkpoints_six_prompt_profiles",
        "objective": "inference_prompt_ablation",
        "learning_rate": None,
        "steps": None,
        "evidence_boundary": "Only legacy_paired_v1 matches training; other profiles are inference-time OOD diagnostics.",
    },
    "gold40_v3_ab_400steps_final_engineering_audit": {
        "display_name": "dataset=gold40_v3__activity=engineering_audit__comparison=A_bridge14_vs_B_bridge14_plus_lora__steps=000_to_400__boundary=mse_comparable_visual_effect_not_established",
        "category": "60_engineering_audits",
        "lifecycle_status": "completed_with_evidence_limits",
        "evidence_level": "engineering_only",
        "dataset_scope": "gold40_v3_training_set_40_pairs",
        "method": "A_bridge14_vs_B_bridge14_plus_LoRA",
        "objective": "engineering_integrity_audit",
        "learning_rate": 1e-6,
        "steps": [0, 400],
        "evidence_boundary": "The original visualization path was not comparable; later aligned inference is the valid visual-output source.",
    },
}

AB_LINKS = {
    "dataset=gold40_v3__condition=A_bridge14_no_lora__checkpoint_step=000__images=40__protocol=nt25_seed1234_cfgT4_cfgI2_1024": "gold40_v3_ab_aligned_inference_nt25_protocol_v1/verifier_A/png",
    "dataset=gold40_v3__condition=A_bridge14_no_lora__checkpoint_step=400__images=40__protocol=nt25_seed1234_cfgT4_cfgI2_1024": "gold40_v3_ab_aligned_inference_nt25_protocol_v1/verifier_A_step400/png",
    "dataset=gold40_v3__condition=B_bridge14_plus_lora_L24-27_r8_a16__checkpoint_step=000__images=40__protocol=nt25_seed1234_cfgT4_cfgI2_1024": "gold40_v3_ab_aligned_inference_nt25_protocol_v1/verifier_B/png",
    "dataset=gold40_v3__condition=B_bridge14_plus_lora_L24-27_r8_a16__checkpoint_step=400__images=40__protocol=nt25_seed1234_cfgT4_cfgI2_1024": "gold40_v3_ab_aligned_inference_nt25_protocol_v1/verifier_B_step400/png",
}


def classify(name: str) -> tuple[str, str, str]:
    lower = name.lower()
    if any(token in lower for token in ("crash", "oom", "failure", "bug")):
        return "70_failed_partial_historical", "failed_or_bug_record", "failure_record_not_result"
    if "partial" in lower:
        return "70_failed_partial_historical", "partial", "partial_record_not_complete_result"
    if name in {"gold40_v2", "gold40_v3", "gold40_full_audit", "base_provenance", "gpu_info"} or any(
        token in lower for token in ("config_diff", "lr_provenance", "realized_shapes")
    ):
        return "10_dataset_and_provenance", "reference", "dataset_or_configuration_provenance"
    if any(token in lower for token in ("preflight", "dryrun", "smoke", "memory_gate", "rollback_verify")):
        return "50_preflight_dryrun_smoke", "diagnostic_only", "engineering_gate_not_scientific_result"
    if any(token in lower for token in ("audit", "crosscheck", "launch_gate", "cold_reload", "verification")):
        return "60_engineering_audits", "diagnostic_only", "engineering_only"
    if any(token in lower for token in ("deterministic_eval", "acceptance_eval", "fixed_deterministic_eval")):
        return "40_evaluation_and_comparisons", "unknown_completion_read_report", "evaluation_metric_or_package"
    if any(token in lower for token in ("aligned_inference", "visual_compare", "diag_images", "prompt_ablation", "conditioning_diagnostics")):
        return "30_inference_and_visualization", "unknown_completion_read_report", "visual_outputs_not_automatically_semantic_success"
    if any(token in lower for token in ("steps400", "400steps", "500steps", "overfit", "single_ball", "single_pair")):
        return "20_training_runs", "unknown_completion_read_report", "training_artifacts_not_automatically_semantic_success"
    return "80_uncertain_unclassified", "unknown", "unknown_read_report_before_use"


def infer_fields(name: str) -> dict[str, Any]:
    lower = name.lower()
    if "micro4" in lower:
        dataset_scope = "micro4_subset"
    elif "c1" in lower or "single_pair" in lower or "single_ball" in lower:
        dataset_scope = "C1_single_pair"
    elif "gold40_v3" in lower:
        dataset_scope = "gold40_v3"
    elif "gold40" in lower:
        dataset_scope = "gold40_unspecified_version"
    else:
        dataset_scope = "unknown"

    if "lora" in lower:
        method = "LoRA_present_details_from_name_or_report"
    elif "baseline" in lower or "overfit" in lower:
        method = "no_LoRA_or_unspecified_read_report"
    else:
        method = "unknown"

    if "region_split" in lower:
        objective = "region_split"
    elif any(token in lower for token in ("training", "overfit", "steps", "single_ball", "single_pair")):
        objective = "unspecified_training_objective_read_report"
    else:
        objective = "not_applicable_or_unknown"

    lr: float | None = None
    lr_match = re.search(r"lr(\d+)e(\d+)", lower)
    if lr_match:
        lr = int(lr_match.group(1)) * 10 ** (-int(lr_match.group(2)))

    steps: int | None = None
    for pattern in (r"(\d+)steps", r"steps(\d+)"):
        match = re.search(pattern, lower)
        if match:
            steps = int(match.group(1))
            break

    return {
        "dataset_scope": dataset_scope,
        "method": method,
        "objective": objective,
        "learning_rate": lr,
        "steps": steps,
    }


def inventory(path: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    bytes_total = 0
    files_total = 0
    checkpoint_count = 0
    report_paths: list[str] = []
    for root, _, files in os.walk(path):
        root_path = Path(root)
        for filename in files:
            file_path = root_path / filename
            try:
                stat = file_path.stat()
            except OSError:
                continue
            files_total += 1
            bytes_total += stat.st_size
            suffix = file_path.suffix.lower() or "[no_suffix]"
            counts[suffix] += 1
            if suffix in {".pt", ".safetensors"}:
                checkpoint_count += 1
            if len(report_paths) < 12 and suffix in {".json", ".jsonl", ".md"} and any(
                token in filename.lower() for token in ("report", "summary", "audit", "config", "manifest", "gate")
            ):
                report_paths.append(str(file_path.relative_to(ROOT)))
    return {
        "file_count": files_total,
        "size_bytes": bytes_total,
        "size_gib": round(bytes_total / (1024**3), 4),
        "checkpoint_count": checkpoint_count,
        "extension_counts": dict(sorted(counts.items())),
        "representative_reports": report_paths,
    }


def safe_symlink(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def link_name(display_name: str, source_name: str, max_bytes: int = 220) -> str:
    """Keep detailed names readable while respecting the common 255-byte limit."""
    if len(display_name.encode("utf-8")) <= max_bytes:
        return display_name
    digest = hashlib.sha256(display_name.encode("utf-8")).hexdigest()[:10]
    prefix_budget = max_bytes - len(digest) - len("__id=")
    prefix = display_name.encode("utf-8")[:prefix_budget].decode("utf-8", errors="ignore").rstrip("_=-")
    return f"{prefix}__id={digest}"


def build_entry(path: Path) -> dict[str, Any]:
    name = path.name
    category, lifecycle, evidence = classify(name)
    entry: dict[str, Any] = {
        "source_name": name,
        "source_path": str(path),
        "source_path_relative": str(path.relative_to(ROOT)),
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "display_name": (
            f"source={name}__dataset={infer_fields(name)['dataset_scope']}__activity=unknown_read_report"
            f"__lifecycle={lifecycle}__evidence={evidence}"
        ),
        "lifecycle_status": lifecycle,
        "evidence_level": evidence,
        "evidence_boundary": "Automatically classified from directory name and file inventory; read the referenced report before scientific use.",
        "deletion_policy": "preserve_source_no_deletion",
        **infer_fields(name),
        **inventory(path),
    }
    if name in KEY_OVERRIDES:
        entry.update(KEY_OVERRIDES[name])
        entry["category_label"] = CATEGORY_LABELS[entry["category"]]
    return entry


def markdown(entries: list[dict[str, Any]], view: Path) -> str:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_category[entry["category"]].append(entry)

    lines = [
        "# Counterfactual-CoT 输出索引 V1",
        "",
        "> 本索引不移动、不复制、不删除任何历史输出；`_organized_view_v1` 只包含软链接。",
        "> `passed`、`completed` 或工程门通过不等于视觉语义成功。所有科学使用必须同时查看 `evidence_level` 与 `evidence_boundary`。",
        "",
        "## 最常用：Gold40 A/B 对齐图片",
        "",
        "| 条件 | 方法 | Step | 图片路径 | 证据边界 |",
        "|---|---|---:|---|---|",
        f"| A | 14 bridge/head-tail tensors，无 LoRA | 0 | `{OUTPUTS / AB_LINKS[next(iter(AB_LINKS))]}` | Step-0 对齐基线 |",
        f"| A | 14 bridge/head-tail tensors，无 LoRA | 400 | `{OUTPUTS / list(AB_LINKS.values())[1]}` | Gold40 训练集容量输出 |",
        f"| B | A + L24-27 LoRA r8/a16 | 0 | `{OUTPUTS / list(AB_LINKS.values())[2]}` | 与 A Step-0 逐像素一致 |",
        f"| B | A + L24-27 LoRA r8/a16 | 400 | `{OUTPUTS / list(AB_LINKS.values())[3]}` | Gold40 训练集容量输出；不是 CoT 消融 |",
        "",
        "注意：Step-400 verifier 的部分 `per_pair.condition` 历史字段仍写成 Step-0；应以目录、顶层 `checkpoint_path` 和 `checkpoint_sha256` 为准。",
        "",
        "## 索引规则",
        "",
        f"- 源目录：`{OUTPUTS}`",
        f"- 分类视图：`{view}`",
        f"- JSONL：`{DEFAULT_JSONL}`",
        "- `70_failed_partial_historical` 仍然保留，只表示不能当作完整科学结果。",
        "- `80_uncertain_unclassified` 表示信息不足，不根据名字猜测实验含义。",
        "- 自动分类条目的字段含 `unknown` 时，应打开 `representative_reports` 再使用。",
        "",
    ]

    for category in CATEGORY_LABELS:
        group = sorted(by_category.get(category, []), key=lambda item: item["source_name"].lower())
        lines.extend([f"## {category}：{CATEGORY_LABELS[category]}（{len(group)}）", ""])
        if not group:
            lines.extend(["暂无。", ""])
            continue
        lines.extend([
            "| 客观显示名 | 原目录 | 状态 | 证据级别 | 大小 GiB | Checkpoints |",
            "|---|---|---|---|---:|---:|",
        ])
        for item in group:
            lines.append(
                f"| `{item['display_name']}` | `{item['source_name']}` | `{item['lifecycle_status']}` | "
                f"`{item['evidence_level']}` | {item['size_gib']:.4f} | {item['checkpoint_count']} |"
            )
        lines.append("")

    lines.extend([
        "## 使用原则",
        "",
        "1. 先从 `00_key_evidence` 进入当前有效证据。",
        "2. 训练目录和评估目录分开使用；不要从目录名推断视觉成功。",
        "3. Gold40 是训练集，相关输出只能称为 capacity/overfit diagnostic。",
        "4. A/B 比较的是可训练参数集合，不是 Paired-CoT 与 Independent 的核心科学对照。",
        "5. 本索引不提供删除候选，也不会修改任何源文件。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, default=DEFAULT_VIEW)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    for generated in (args.view, args.jsonl, args.markdown):
        if generated.exists() or generated.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing generated path: {generated}")

    source_dirs = sorted(
        path for path in OUTPUTS.iterdir() if path.is_dir() and not path.name.startswith("_organized_view")
    )
    entries = [build_entry(path) for path in source_dirs]

    args.view.mkdir(parents=True, exist_ok=False)
    for category in CATEGORY_LABELS:
        (args.view / category).mkdir()

    for entry in entries:
        filesystem_name = link_name(entry["display_name"], entry["source_name"])
        link = args.view / entry["category"] / filesystem_name
        safe_symlink(Path(entry["source_path"]), link)
        entry["organized_view_path"] = str(link)
        entry["organized_view_link_name"] = filesystem_name

    ab_dir = args.view / "00_key_evidence" / "gold40_aligned_A_B_image_sets"
    ab_dir.mkdir()
    for label, relative_target in AB_LINKS.items():
        safe_symlink(OUTPUTS / relative_target, ab_dir / label)

    args.jsonl.write_text(
        "".join(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )
    args.markdown.write_text(markdown(entries, args.view), encoding="utf-8")

    counts = Counter(entry["category"] for entry in entries)
    print(json.dumps({
        "status": "created",
        "source_directories_indexed": len(entries),
        "source_files_deleted": 0,
        "source_directories_moved": 0,
        "view": str(args.view),
        "jsonl": str(args.jsonl),
        "markdown": str(args.markdown),
        "category_counts": dict(sorted(counts.items())),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
