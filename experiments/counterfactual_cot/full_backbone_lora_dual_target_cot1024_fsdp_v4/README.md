# Round 4: Dual Target-CoT Paired Training

实验名：`cf_singlepair_fullL0-27_lora_r8_dualTargetCoT1024_fsdpFullShard_overfit_v4`

## 核心改动

v1/v3 中 Target CoT 在全部训练 step 都固定为 right，模型可以将整段文本视作常量。v4 对同一 Source 同时训练两个互斥分支：

```text
Left branch:
  Source Image = left
  Source Prompt = left
  Source CoT = left
  Target CoT = left
  Target Image = source image (left)

Right branch:
  Source Image = same left image
  Source Prompt = same left prompt
  Source CoT = same left CoT
  Target CoT = right
  Target Image = golden target (right)
```

两个 Prompt 经过代码门禁，除 `ball_1.location=left_region/right_region` 外完全一致；不包含 Target Prompt、Intervention、delta 指令或独立 Preservation 文本。

## Paired macro-step

一个 optimizer macro-step 顺序执行：

1. `0.5 * left loss` backward；
2. `0.5 * right loss` backward；
3. 双 rank ignored LoRA/bridge 梯度平均；
4. 一次 gradient clip 和 AdamW update。

Left/Right 使用相同 pack seed、相同 timestep 和同一预生成 forward noise tensor；训练逐步记录 noise SHA256 并要求两个分支完全一致。两分支在同一次 optimizer update 前使用相同模型参数。500 macro-step 等于 1000 次 1024 forward/backward，预计耗时约为 v3 的两倍。

## Timestep mixture

每个 macro-step 的模式由 `(seed, macro_step)` 的 SHA256 无状态派生，断点恢复无需额外 sampler state：

- 50%：原生 raw-normal timestep，并经过 BAGEL shift `3.0`；
- 40%：反解 raw logit，使 BAGEL shift 后的实际 `t ∈ [0.85, 0.999]`；
- 10%：纯噪声 `t=1.0`。

纯噪声时 left/right noisy latent 相同，区分监督目标的唯一输入变量是 Target CoT。

## Loss

v4-A 为保持单因素归因，继续使用 v3 权重：

```text
0.25 * left_roi_mse + 0.25 * right_roi_mse + 0.50 * outside_mse
```

- Left target：左 ROI 有球，右 ROI 为空；
- Right target：左 ROI 去球，右 ROI 有球；
- Outside：监督各自 branch 的真实 target flow。

若 v4-A 已产生 CoT 切换趋势但右侧编辑不完整，再另开 v4-B 比较 `0.40/0.40/0.20`；当前目录不混入该变量。

## 条件依赖诊断

在 macro-step `0/100/250/500` 使用相同 `t=1`、相同 noise，评估四个组合：

- left target + left CoT（正确）；
- left target + right CoT（交换）；
- right target + right CoT（正确）；
- right target + left CoT（交换）。

记录 `swapped loss - correct loss`。若 gap 仍接近 0，说明模型仍忽略 Target CoT，即使训练 loss 下降也不能判定成功。

## 训练架构

- VAE/output `1024×1024`，ViT `980×980`；
- 全 `L0-L27`、168 个 generation-branch LoRA；
- `r=8`、`alpha=16`，LoRA dropout 固定 0；
- LoRA 与 bridge trainable 参数及 AdamW 状态使用 FP32 master 精度，forward 由 BF16 autocast 执行；五个写入 LLM token 序列的 bridge 顶层输出会在实验侧显式 cast 为 BF16，`llm2vae` regression head 保持 FP32；
- 2×A100 40GB；
- FSDP `FULL_SHARD`、`use_orig_params=True`；
- 28 层 non-reentrant activation checkpointing；
- 原子 checkpoint、每 10 macro-step 自动恢复点；
- 所有产物写入持久化 `group`。

## 第一步：2 macro-step smoke

```bash
cd /group/40016/niightli/BAGEL
GPU_IDS=0,1 bash experiments/counterfactual_cot/full_backbone_lora_dual_target_cot1024_fsdp_v4/run_smoke_2xa100_40gb.sh
```

必须验证：

- left/right pack 均为 8192 VAE token、4900 ViT token；
- 168 LoRA wrapper、28 activation-checkpointed layer；
- 每个 macro-step 都有 left/right 两组有限 loss；
- 两分支 `effective_t`、noise seed、pack seed 相同；
- 梯度完整有限，rank loss/checksum 一致；
- fixed pure-noise correct-vs-swapped 四组合均成功运行；
- `max_rank_peak_reserved_gib < 38.5`；
- `step_000002.pt` 完整保存。

Smoke 输出：

`experiments/counterfactual_cot/outputs/cf_singlepair_fullL0-27_lora_r8_dualTargetCoT1024_fsdpFullShard_smoke_v4_r3`

## 第二步：500 macro-step

Smoke 审计通过后：

```bash
cd /group/40016/niightli/BAGEL
nohup env GPU_IDS=0,1 \
  bash experiments/counterfactual_cot/full_backbone_lora_dual_target_cot1024_fsdp_v4/run_train500_2xa100_40gb.sh \
  > experiments/counterfactual_cot/outputs/round4_dual_target_cot_launcher.log 2>&1 &
echo $!
```

正式输出：

`experiments/counterfactual_cot/outputs/cf_singlepair_fullL0-27_lora_r8_dualTargetCoT1024_fsdpFullShard_overfit_v4`

中断后重新执行同一命令即可从完整 macro-step 恢复。

## 第三步：16-seed 同噪声切换评估

训练完成后，确认一张 A100 40GB 空闲：

```bash
cd /group/40016/niightli/BAGEL
GPU_ID=0 bash experiments/counterfactual_cot/full_backbone_lora_dual_target_cot1024_fsdp_v4/run_evaluate.sh
```

该脚本分别冷加载 step 0 和 step 500。每个固定 seed 用同一初始噪声生成：

```text
Target CoT=left  -> 应输出左球
Target CoT=right -> 应输出右球
```

主要结果：

- `evaluation_dual_target_cot_switch/step_000000/paired_switch_contact_sheet.jpg`
- `evaluation_dual_target_cot_switch/step_000500/paired_switch_contact_sheet.jpg`
- `evaluation_dual_target_cot_switch/analysis/switch_metrics.json`

严格成功要求：

- Left 输出主红球中心 `x < 270`；
- Right 输出主红球中心 `x >= 810`；
- 同 seed 两者同时正确才记为 paired switch success；
- 必须人工复核恰好一个球与场景质量。

建议最终门禁：Left 至少 `14/16`、Right 至少 `12/16`、paired switch 至少 `12/16`。