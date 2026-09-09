# Round 3: 2×A100 40GB FSDP FULL_SHARD

实验名：`cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_overfit_v3`

## 固定实验条件

- 条件输入：`Source Image + Source Prompt + Target CoT`（`cot_core_v1`）。
- 训练/推理：VAE `1024×1024`、ViT `980×980`，target latent 为 `64×64=4096` token。
- region loss：`0.25 old + 0.25 new + 0.50 preserve`。
- 全 `L0-L27`、168 个 generation-branch LoRA，`r=8`、`alpha=16`；保留 bridge trainable 参数。
- 两张 A100 40GB，`torchrun --nproc_per_node=2`，FSDP `FULL_SHARD`、`use_orig_params=True`。
- VAE 在两张卡各复制一份；冻结主干参数由 FSDP 分片。
- LoRA 与 bridge trainable 参数故意作为 FSDP ignored states 在两 rank 复制，以保留原始参数名、避免 FSDP flat-parameter 混合冻结/可训练约束，并使完整 checkpoint 可直接冷加载评估。
- 两个 rank 每 step 使用**同一个**锁定 pair、相同 seed；每 step 校验 loss 和 trainable checksum 一致，避免拆分后两卡训练出不同权重。
- 所有输出写入持久化 `group`，不写入一次性 SSD。

## 断点续传

checkpoint 仅由 rank 0 原子写入，内容包括完整 LoRA/bridge 参数、AdamW 状态、训练历史、RNG、FSDP/分辨率/prompt/region-loss 配置。两 rank 恢复时都从同一完整 checkpoint 读入，因此同一命令可安全 `--resume auto`。

不要使用 v2 的 checkpoint 恢复 v3，也不要用 v3 checkpoint 恢复 v2。

## 第一步：两卡 2-step smoke

确认两张 A100 空闲后，前台执行：

```bash
cd /group/40016/niightli/BAGEL
GPU_IDS=0,1 bash experiments/counterfactual_cot/full_backbone_lora_region1024_fsdp_v3/run_smoke_2xa100_40gb.sh
```

Smoke 必须同时满足：

- 两个 rank 均通过 `fsdp_topology_validated`；
- `num_vae_tokens=8192`、`num_vit_tokens=4900`；
- `lora_wrappers=168`、`activation_checkpointed_layers=28`；
- old/new/preserve 都是有限值；
- 没有 `Gradient audit failed` 或 `Rank loss mismatch`；
- 没有 `Replicated trainable checksum diverged`；
- `max_rank_peak_reserved_gib < 38.5`；
- 保存 `step_000002.pt` 后任务正常结束。

Smoke 输出：

`experiments/counterfactual_cot/outputs/cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_smoke_v3`

## 第二步：500-step 正式训练

只有 smoke 通过后运行：

```bash
cd /group/40016/niightli/BAGEL
nohup env GPU_IDS=0,1 bash experiments/counterfactual_cot/full_backbone_lora_region1024_fsdp_v3/run_train500_2xa100_40gb.sh \
  > experiments/counterfactual_cot/outputs/round3_fsdp_launcher.log 2>&1 &
echo $!
```

每 10 step 保存 `latest.pt`，并保留 `0/25/50/100/200/300/400/500`。中断后重新执行同一命令即可恢复。

正式输出：

`experiments/counterfactual_cot/outputs/cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_fsdpFullShard_overfit_v3`

## 第三步：单卡冷加载评估

完成 500 step 后：

```bash
cd /group/40016/niightli/BAGEL
GPU_ID=0 bash experiments/counterfactual_cot/full_backbone_lora_region1024_fsdp_v3/run_evaluate.sh
```

评估会加载 rank-0 产出的完整 trainable checkpoint，在单卡生成六种 prompt、三个 seed 及 image-CFG 诊断；不依赖 FSDP 运行时。