# V3 Prompt/CoT Order and Alignment Control

本实验不重新训练。它对同一 v3 的 `step_000000.pt` 与 `step_000500.pt` 做严格同条件冷加载推理，比较三种文本条件：

- A `target_prompt_then_source_cot_v1`：Target Prompt(right) → Source CoT(left)
- B `source_cot_then_target_prompt_v1`：Source CoT(left) → Target Prompt(right)
- C `target_prompt_then_target_cot_v1`：Target Prompt(right) → Target CoT(right)

三者使用相同 mode header 和中性结尾 `Generate the target image.`，均不包含 Source Prompt、Intervention 或独立 Preservation 段。

## 可回答的问题

1. A vs B：字段内容和标签完全相同，只有顺序变化，属于严格顺序对照。
2. A vs C：Target Prompt 均在第一位，第二字段由冲突的 Source CoT(left) 改为一致的 Target CoT(right)。这是“CoT 来源标签 + 内容一致性”的联合效应，不是纯内容效应。
3. Step0 vs Step500：训练是否改变每种条件的服从行为。

Target Prompt 本身含有 `Move only the red toy ball`，因此任一成功都可能来自自然语言动作指令。C 中两个字段都要求 right，成功也不能单独证明模型必须使用 Target CoT。

## 锁定配置

- checkpoints：同一 v3 的 step 0 / step 500；
- 每个 checkpoint 独立冷启动进程；
- seeds：1234 / 2234 / 3234；
- VAE/output 1024，ViT 980；
- text CFG 4.0，image CFG 2.0；
- CFG interval [0,1]，timestep shift 3.0，25 sampling steps；
- 不运行额外 CFG ablation。

协议会校验 checkpoint schema、168 个 LoRA-B 的零更新/全更新状态、连续 1..500 history，并记录基础权重、代码、prompt 和每张图片 SHA256。已有输出目录不会被覆盖。

## 运行

确认一张 A100 40GB 空闲：

```bash
cd /group/40016/niightli/BAGEL
nohup env GPU_ID=0 \
  bash experiments/counterfactual_cot/full_backbone_lora_v3_prompt_cot_order_control/run_strict_control.sh \
  > experiments/counterfactual_cot/outputs/v3_prompt_cot_order_control_launcher.log 2>&1 &
echo $!
```

输出目录：

`experiments/counterfactual_cot/outputs/cf_singlepair_fullL0-27_lora_r8_promptCotOrder_strict_step0_step500_control_v1`

重点结果：

- `analysis/paired_metrics.json`：每条件的 step0→step500 训练效应；
- `analysis/contrasts.csv`：A/B 顺序效应、A/C 一致性效应及 difference-in-differences；
- `analysis/diagnostic_step0_conditions.jpg` / `diagnostic_step500_conditions.jpg`：同 checkpoint 下三条件横向对比；
- `analysis/diagnostic_conditions/*.jpg`：每条件的 step0/step500 配对图；
- `analysis/blind_review/`：隐藏 checkpoint 身份的 A/B 人工盲审材料。

只有 3 个 seed，所有 contrast 仅用于机制诊断，不应宣称统计显著性。
