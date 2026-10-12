# V3 Strict Step-0 vs Step-500 Control

该目录用于回答：在完全相同的基础权重、输入 pair、prompt 文本、seed、CFG、分辨率和采样器设置下，v3 的 500 次优化相对零更新 step 0 到底改善了什么。

## 对照定义

- Baseline：同一次 v3 训练在任何 optimizer update 之前保存的 `step_000000.pt`。
- Treatment：同一次 v3 训练完成后的 `step_000500.pt`。
- Step 0 是“零更新 checkpoint”，不是另一份独立纯 base checkpoint；LoRA 已注入，但 B 初始为零。
- 每个 checkpoint 使用独立冷启动 Python 进程，避免模型、CUDA allocator 和 inferencer 缓存交叉污染。
- 两个进程顺序使用同一物理 GPU。

锁定条件：

- 六种 prompt profile；
- seeds：`1234, 2234, 3234`；
- `1024` VAE/output、`980` ViT；
- `cfg_text=4.0`、`cfg_img=2.0`；
- CFG interval `[0,1]`；
- timestep shift `3.0`；
- 25 sampling steps；
- 不混入 image-CFG ablation。

`control_protocol.json` 会记录唯一 run ID、两个 checkpoint 的 SHA256、格式、step、trainable schema/LoRA target 摘要、锁定训练配置、基础权重与评估代码 SHA256、实际执行命令。还会检查 step0 的 168 个 LoRA-B 全为零、history 为空，以及 step500 的 168 个 LoRA-B 均已更新、history 为连续 1..500。旧 checkpoint 没有 run UUID，因此历史血缘只能由这些内部一致性证据强支持，不能追溯性地做密码学证明。

每个 step 生成后都会写独立 receipt，将 checkpoint SHA、evaluation report、prompt/profile/seed 和每张 PNG 的 SHA256 绑定。分析前会重新核对全部 receipt；陈旧、被替换或混合的结果会被拒绝。固定输出目录若已有 protocol 或 generations，脚本也会拒绝覆盖，重跑必须使用新的 `--output-dir`。

## 运行

确认一张 A100 40GB 空闲后：

```bash
cd /group/40016/niightli/BAGEL
nohup env GPU_ID=0 \
  bash experiments/counterfactual_cot/full_backbone_lora_v3_strict_step0_step500_control/run_strict_control.sh \
  > experiments/counterfactual_cot/outputs/v3_strict_control_launcher.log 2>&1 &
echo $!
```

两个 checkpoint 会分别完整冷加载，耗时大致为原多 Prompt 推理的两倍。无需持续监测。

输出目录：

`experiments/counterfactual_cot/outputs/cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_strict_step0_step500_control_v1`

## 结果结构

- `control_protocol.json`：对照身份和所有锁定条件；
- `generations/step_000000/`：零更新输出；
- `generations/step_000500/`：500-step 输出；
- `analysis/paired_metrics.json`：逐 profile/seed 配对指标与汇总；
- `analysis/paired_metrics.csv`：便于表格分析的长表；
- `receipts/`：逐 checkpoint 的报告、图片与 runtime SHA256 收据；
- `analysis/diagnostic_contact_sheets/<profile>.jpg`：带 step 标签和指标的诊断图；
- `analysis/diagnostic_all_profiles_contact_sheet.jpg`：完整带标签配对图；
- `analysis/blind_review/blind_ab_contact_sheet.jpg`：不显示 step 身份或自动指标的 A/B 盲审图；
- `analysis/blind_review/manual_review_template.json`：先填写的盲审表；
- `analysis/blind_review/blind_key.json`：完成盲审后才能打开的 A/B→checkpoint 映射。

## 指标与结论边界

自动指标：

1. old/new changed-pixel mask 内相对 golden target 的 RGB MAE；
2. preserve 区域相对 source 的 RGB MAE；
3. old/new ROI 中的透明红色像素代理；
4. step0 -> step500 的逐 seed 配对变化。

红球代理以 source→golden target 的红色像素变化量为标尺：source 的进度为 0，golden target 的进度为 1。旧区移除进度与新区新增进度都达到 50% 才记为代理成功。该阈值只用于一致性筛查，不替代人工判断；像素 MAE 对小幅空间偏移敏感，也不能单独证明语义正确。

主研究结论必须优先看 `cot_core_v1`：它与训练条件一致。`delta_only_v2` 的改善表示显式编辑能力增强，但不能证明 Target CoT 遵循。若 step500 相对 step0 只改善 delta-only，而 cot-core 没有改善，应判定为“训练增强了直接编辑/目标图拟合，但尚未验证 Target CoT 控制”。
