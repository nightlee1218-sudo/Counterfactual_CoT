# Gold40 AutoMask-v5 计划与评估

此目录独立于 `gold40_v4_baselines/`，不会修改或复用 v4 baseline 的启动器。它只提供：

- `build_automask_v5_plan.py`：生成 AutoMask-v5 主方法和 `paired_v5_mask_free`（配对但无掩码）消融的可审计计划与启动模板；
- `audit_gold40_automask_v5.py`：审计已生成结果。

主方法已有已验证的独立 trainer：`gold40_automask_dual_target_cot1024_fsdp_v5/train.py`。生成的 `launch_automask_v5_main.sh` 因此会使用 `torchrun --standalone --nproc_per_node=2` 运行该入口；计划生成本身绝不会启动训练。`paired_v5_mask_free` 仍没有已注册的 objective/trainer，故其 launcher 会清晰地 `exit 2` 且不调用 `torchrun`。不能把 v4 trainer 或主方法 trainer 冒充无掩码消融。

## AutoMask-v5 manifest 契约

训练和 held-out 清单均为 JSONL，或 JSON 数组/`{"records": [...]}`。每条记录至少需要：

```json
{
  "schema_version": "gold40_automask_v5_v1",
  "pair_id": "example_0001",
  "source_scene_id": "scene_01",
  "source": {"image": "/absolute/source.png"},
  "target": {"image": "/absolute/target.png"},
  "automatic_difference_mask": {
    "path": "/absolute/automask.png",
    "sha256": "<64-char-sha256>",
    "threshold": 24,
    "dilation_px": 9,
    "candidate_only": true
  },
  "intervention_type": "attribute_change"
}
```

正式 trainer 契约使用 `automatic_difference_mask`；计划和审计也兼容带明确 v5 标识的 `automask` 形式，以便读取适配器输出。计划器要求唯一的 `pair_id`、非空 `source_scene_id` 和自动掩码路径，并拒绝 train/eval 之间的 `source_scene_id` 泄漏。它锁定清单的绝对路径和 SHA256，不会把自动掩码伪装成人工审核的 v4 ROI。

## 生成实验计划和启动模板

```bash
cd /group/40016/niightli/BAGEL
python experiments/counterfactual_cot/gold40_automask_v5_baselines/build_automask_v5_plan.py \
  --train-auto-manifest /absolute/path/to/train_automask_v5.jsonl \
  --eval-auto-manifest /absolute/path/to/test_automask_v5.jsonl \
  --output-dir /absolute/path/to/gold40_automask_v5_plan
```

输出包括 `experimental_plan.json` 以及：

- `launchers/launch_automask_v5_main.sh`
- `launchers/launch_paired_v5_mask_free.sh`

这是启动模板：`launch_automask_v5_main.sh` 会调用已实现的两卡 trainer；`launch_paired_v5_mask_free.sh` 仍会 fail fast，直到实际无掩码 objective/trainer 存在。

## 同种子生成审计

审计严格分离三类证据：

1. `--v5-auto-manifest`：AutoMask-v5 掩码以及 source/target reference images；
2. `--generation-manifest`：每个 `(pair_id, seed, state)` 的输出路径；
3. `--automated-semantic-label-manifest`：同一 key 的显式自动语义布尔标签。

生成和标签键必须完全相同；每个 `(pair_id, seed)` 必须同时有 `source_state` 和 `target_state`。标签记录示例（`automated_semantic_correct`、`semantic_correct` 或兼容的 `state_correct` 三者恰有一个）：

```json
{"pair_id":"example_0001","seed":1234,"state":"source_state","automated_semantic_correct":true}
{"pair_id":"example_0001","seed":1234,"state":"target_state","automated_semantic_correct":true}
```

```bash
cd /group/40016/niightli/BAGEL
python experiments/counterfactual_cot/gold40_automask_v5_baselines/audit_gold40_automask_v5.py \
  --v5-auto-manifest /absolute/path/to/test_automask_v5.jsonl \
  --generation-manifest /absolute/path/to/generation_results.jsonl \
  --automated-semantic-label-manifest /absolute/path/to/automated_semantic_labels.jsonl \
  --output /absolute/path/to/automask_v5_audit.json \
  --bootstrap-samples 10000 --bootstrap-seed 0
```

报告给出 source accuracy、target accuracy、同种子 `same_seed_switch_accuracy`（两状态的显式语义标签都为真）、mask 内 `edit_mae` 与 mask 外 `outside_mae`，及总体、按 intervention type、按 pair 的 percentile bootstrap 95% CI。总体和类型汇总按 `pair_id` cluster 重采样并保留该 pair 的所有 seed；单 pair 汇总按 `pair_id + seed` 重采样。

`edit_mae` 与 `outside_mae` 都是 `[0,1]` RGB 像素诊断：前者在 AutoMask-v5 区域内比对相应 source/target reference，后者在区域外比对 source reference。它们绝不用于推断语义成功；语义成功只读取独立的显式自动语义标签。
