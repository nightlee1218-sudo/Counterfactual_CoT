# `cf_singlepair_fullL0-27_gen6_lora_r8_overfit_v1`

## 实验边界

- 只训练锁定的 living-room / `ball_1` pair。
- 相比旧实验，只扩大 LoRA 层覆盖：从 `L24-L27` 改为 `L0-L27`。
- 每层仍只注入相同的 6 个 generation-branch Linear：`q/k/v/o_proj_moe_gen`、`mlp_moe_gen.gate_proj/down_proj`，共 `28 × 6 = 168` 个 wrapper。
- 保留旧实验已有的 bridge trainable 参数，保持对照变量一致。
- 不修改 BAGEL 主干实现，不引入 FSDP、DeepSpeed 或模型并行。

## 显存决策

当前为两张 A100 40GB。先使用单张 A100，并对全部 28 个 `Qwen2MoTDecoderLayer` 启用 non-reentrant activation checkpointing；第二张卡暂不使用。已有 `L24-L27` 实测峰值约 `34.5 GiB allocated / 35.5 GiB reserved`，因此必须先执行 2-step smoke，以实测峰值决定是否能安全运行正式实验。

若 smoke OOM 或 `peak_reserved_gib >= 38.5`，不要启动正式任务；后备方案是另建双卡 FSDP `FULL_SHARD` 版本，而不是普通 `device_map`。

## 操作顺序

先前台执行 2-step 显存/梯度 smoke（独立输出，不污染正式实验）：

```bash
cd /group/40016/niightli/BAGEL
GPU_ID=0 bash experiments/counterfactual_cot/full_backbone_lora_overfit_v1/run_smoke_a100_40gb_ac.sh
```

smoke 成功后，把终端末尾输出发回审计。只有确认 `peak_reserved_gib < 38.5` 后，才后台启动正式 500-step overfit：

```bash
cd /group/40016/niightli/BAGEL
nohup env GPU_ID=0 bash experiments/counterfactual_cot/full_backbone_lora_overfit_v1/run_single_a100_40gb_ac.sh \
  > /dockerdata/niightli/BAGEL/full_lora_ac_overfit_launcher.log 2>&1 &
echo $!
```

该命令启动后即可退出，不需要持续监测。训练完成后，再单独冷加载生成结果图：

```bash
cd /group/40016/niightli/BAGEL
bash experiments/counterfactual_cot/full_backbone_lora_overfit_v1/run_evaluate.sh
```

## 中断与恢复

- `run_single_a100.sh` 默认 `--resume auto`。
- 每 10 step 原子覆盖 `checkpoints/latest.pt`；里程碑 `0/25/50/100/200/300/500` 额外保留。
- 检查点包含全部 trainable 参数、AdamW 状态、step、loss history 和 RNG 状态。
- `SIGTERM`/`Ctrl-C` 会在当前 step 完成后保存再退出；意外退出最多损失 9 step。
- 恢复时直接重新执行同一条 `run_single_a100.sh`，不得修改学习率、LoRA rank、模型路径或 pair。

## 审计产物

- `experiment_plan.json`：锁定 pair、源/目标图、28 层和 168 targets。
- `trainable_audit.json`：实际 trainable 参数数量及完整名称。
- `train_log.jsonl`：逐 step loss、梯度范数、显存，以及固定 timestep 的 milestone MSE。
- `checkpoints/latest.pt`：自动恢复点。
- `final_report.json`：训练完成摘要。
- `evaluation/step_XXXXXX/{source,golden_target,model_output}.png`：最终视觉对比。
