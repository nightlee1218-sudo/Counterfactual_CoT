# Round 2: full-backbone LoRA + CoT-core + region loss at 1024

实验名：`cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_overfit_v2`

## 与第一轮的差异

- 从原始 BAGEL base 重新初始化，不继承第一轮 checkpoint。
- 训练文本固定为 `cot_core_v1`：仅 `Source Prompt + Target CoT`，不包含 Source CoT、Target Prompt、显式 Intervention 或 Preservation 文本。
- 输入结构：Source VAE 1024 + Source ViT 980 + CoT-core 文本 + Target VAE 1024。
- Target latent grid 为 `64×64=4096` token，`max_latent_size=64`，总 token 预算为 16384。
- loss 为 `0.25*old_mse + 0.25*new_mse + 0.50*preserve_mse`；三个区域分别按自身 soft token mass 归一化。
- preserve 区域仍监督 golden target 的 flow velocity，不把 source latent 当 target。
- 全部 `L0-L27` 的 168 个 generation-branch LoRA wrapper，`r=8, alpha=16`；保留既有 bridge trainable 参数。
- 全部 28 个 decoder layer 使用 non-reentrant activation checkpointing。
- 所有实验产物直接写入 `/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs`，不会写到一次性 SSD。

## 第一步：2-step smoke

前台执行：

```bash
cd /group/40016/niightli/BAGEL
GPU_ID=0 bash experiments/counterfactual_cot/full_backbone_lora_region1024_v2/run_smoke_a100_40gb.sh
```

只有同时满足以下条件才运行正式实验：

- `num_vae_tokens=8192`、`num_vit_tokens=4900`；
- `sequence_length` 在 13000–16384；
- `lora_wrappers=168`、`activation_checkpointed_layers=28`；
- old/new/preserve loss 均为有限值；
- 无 gradient audit 错误；
- `peak_reserved_gib < 38.5`；
- step 0 和 step 2 使用相同 fixed timestep hash。

Smoke 输出：

`experiments/counterfactual_cot/outputs/cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_smoke_v2`

## 第二步：500-step 正式训练

Smoke 通过后后台启动：

```bash
cd /group/40016/niightli/BAGEL
nohup env GPU_ID=0 bash experiments/counterfactual_cot/full_backbone_lora_region1024_v2/run_train500_a100_40gb.sh \
  > experiments/counterfactual_cot/outputs/round2_region1024_launcher.log 2>&1 &
echo $!
```

无需持续监测。每 10 step 原子更新 `latest.pt`，并保留 `0/25/50/100/200/300/400/500`。收到 `SIGTERM`/`Ctrl-C` 后会在当前 step 结束时保存；重新执行相同命令会 `--resume auto`。

正式输出：

`experiments/counterfactual_cot/outputs/cf_singlepair_fullL0-27_lora_r8_regionSplit1024_cotCore_overfit_v2`

## 第三步：多 Prompt 推理

训练完成后执行：

```bash
cd /group/40016/niightli/BAGEL
GPU_ID=0 bash experiments/counterfactual_cot/full_backbone_lora_region1024_v2/run_evaluate.sh
```

默认生成：

- Prompt：`cot_core_v1`、`legacy_paired_v1`、`full_paired_v2`、`full_without_target_cot_v1`、`target_prompt_only_v1`、`delta_only_v2`
- Seeds：`1234`、`2234`、`3234`
- `cot_core_v1` 与 `delta_only_v2` 的额外 `cfg_img_scale=1.0/1.5` 诊断
- 1024×1024 输出和 contact sheet

主结论只依据 `cot_core_v1`；其他 prompt 是鲁棒性诊断。Target-CoT 因果对照暂不包含在本轮代码中。
