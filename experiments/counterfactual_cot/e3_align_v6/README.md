# E3 — token-region 对齐辅助损失训练 v6（`e3_align_v6/`）

在 v5 automask dual SFT（final991 配方：`gold40_automask_dual_target_cot1024_fsdp_v5`，
896 对 train split，2973 步，LoRA r8/α16 fp32 masters，FSDP FULL_SHARD 双卡，28 层强制
activation checkpointing）基础上，加入 E2 探针判据驱动的对齐辅助损失：

```
L_total = L_SFT + λ · L_align ,  λ = 0.1 (默认)
L_align = −log σ( s · ( cos(h_tok, μ_in) − cos(h_tok, μ_out) ) )
```

- `h_tok`：训练 packed 序列 prompt（`render_state_prompt`）中 **Target CoT 行内被锚定的
  干预值 token**（target_state 用 `new_value`，source_state 用 `old_value`）在对齐层的
  输出 hidden state；
- `μ_in / μ_out`：同一层输出在 **target latent 4096 个位置**（64×64 row-major）上按 v5
  objective 的 edit/outside 连续权重做质量加权平均；
- 对齐层默认 **L13/L6/L16/L14**（E2 配对 bootstrap 的 final991 显著层），对 (token, layer)
  取平均；每 state 分支仍按 v5 的 `0.5` 缩放后 backward。

E2 判据回顾：final991 相对 base 在上述层的 margin（cos(h_tok, μ_mask内) − cos(h_tok,
μ_mask外)）显著为正——模型学到了「变化 token ↔ 图像编辑区域」对应机制。v6 把该判据变成
可微目标与 SFT 共同优化（token 侧经共享 attention 的 value 混流回连 gen LoRA 参数，
latent 侧直接作用于 gen 流）。

## 与 v5 的差异清单

| 项 | v5 | v6 |
|---|---|---|
| 损失 | 0.5·edit_mse + 0.5·outside_mse + ce（连续 mask 权重） | 同结构，区域项可重配；另加 λ·L_align |
| objective 模块 | `gold40_automask_v5_data.automask_objective` | `e3_align_v6.align_objective`（分区/审计逐字复用 v5；`configure_overshoot` 注入开关） |
| 区域损失权重 | 硬编码 0.5/0.5 | `--overshoot-fix on` 时 `--edit-loss-weight 0.4`（outside 0.6） |
| 区域均值 | 连续质量加权 | `--mask-debias on` 时硬分区（w_edit≥0.5）无权均值；空硬区自动回退连续模式并打标 |
| 对齐 hook | 无 | 4 层 forward hook（`align_loss.AlignHookManager`）；λ=0 时完全不启用 |
| 通用回放 | 无 | `--general-replay-manifest`（默认关）每 N 步插入一个 standard_cot 微步 |
| warm start | 无 | `--init-trainable-from` 可载 v5/v6 `trainable_state_dict` |
| checkpoint 格式 | `gold40_automask_v5_paired_fsdp` | `gold40_automask_v6_align_fsdp`（互不兼容；resume guard 增加 align/overshoot/replay/warmstart 全部配置键） |
| steps 默认 | 960 | 2973（对齐 final991 实际训练量） |
| manifest 默认 | 40 对 gold40 | `final991_automask_v5_data/artifacts/splits/train.jsonl`（896 对，final991 同源） |

模型构建、LoRA 注入（仅 gen 专家投影 `q/k/v/o_proj_moe_gen` + `mlp_moe_gen.gate/down`）、
fp32 masters、FSDP/AC、matched-noise 双 state、timestep 混合（native .50 / high .40 /
pure_noise .10，SHIFT=3.0）、16384 token 打包、逐步确定性 seed 全部与 v5 相同。

## hook 机制与 activation checkpointing 的交互（自测 #11 验证）

- v3 训练运行时用 `apply_activation_checkpointing` + `CheckpointImpl.NO_REENTRANT` 包装
  全部 28 层。非重入实现的前向**仍带 autograd 运行**，只是经 saved-tensors-hooks 丢弃
  保存的中间激活、反向按需重算整段。
- 因此在 align 层上注册普通 `forward_hook`，把 (S, D) 层输出立即降维成小张量
  （`index_select` 取 h_tok + GEMV 求 μ），提取物**保持梯度连接**；反向经标准重算机制
  回传，不需要绕开或替换 checkpoint，也不改动任何共享训练代码。
- hook 以「首次写入为准」防止重算阶段覆盖 stash；每次 backward 后清空。层引用在
  FSDP/AC 包装**之前**捕获（`model.language_model.model.layers[i]` 对象身份包装后不变）。
- `selftest_v6.py` 用同样 API 构造 3 层玩具网络验证：AC+hook 的输入与全部参数梯度与
  非 checkpoint 基线一致（atol 1e-5）。
- `config.freeze_und` 未出现在模型配置中（默认 False），und 流不被 detach；LoRA 只注入
  gen 投影，token 侧梯度经共享 attention 的 value 混流进入这些可训练参数，梯度路径成立。

## 位置与 mask 映射（沿用 E2 已验证结论）

- packed 布局（v5 state 样本；packer cfg dropout 全 0 → 每步必有 prompt）：
  `[SOI + 4096 src-vae + EOI][SOI + 4900 src-vit + EOI][bos + prompt + eos][SOI + 4096 tgt-vae + EOI]`
- 训练走原生 forward 路径（**非** E2 的推理 monkeypatch），位置从 batch 张量鲁棒推导并
  强制校验：`n_text = mse_loss_indexes[0] − max(vit_indexes) − 5`，
  `bos_pos = max(vit_indexes) + 2`，并断言 `packed_text_ids[5:5+n_text] ==
  tokenizer.encode(state prompt)`（布局漂移立即报错）。
- 变化 token span：复用 E2 `find_changed_occurrences`（`=`/`:` 锚定）+ 前缀编码映射，限定
  在 prompt 的 **Target CoT:** 行内（模型被指令遵循的行），取全部锚定出现的并集；
  preflight 统计两 state 的 span 命中率写入 plan，训练中未命中则该步跳过对齐并记录原因。
- mask→in/out：直接复用 `build_paired_automask_objective` 的 64×64 连续分区
  （adaptive_avg_pool2d，row-major，双 state 共享同一张量），不二次读图。
- μ 用 GEMV（`(S,) @ (S, D)`）计算，不物化 (4096, D) 的大中间张量。

## 反编辑过冲开关（默认开，均可关）

`--overshoot-fix off` 逐字复现 v5 目标（0.5/0.5 连续权重），便于因果对照。

1. **edit 权重重配**（`--edit-loss-weight 0.4`，outside 0.6）：final991 的过冲表现为
   编辑外溢/背景被改动，加大 outside 保留压力。
2. **mask 内去偏**（`--mask-debias on`）：automask 是 9px 膨胀后的差分 mask，连续质量
   均值把弱掩码边缘 token 同时稀释进 edit 与 outside 两侧；硬分区下区域均值改为无权
   平均——编辑区信号不被 halo 稀释，halo token 得到完整 outside 惩罚。硬侧为空时自动
   回退连续模式（metrics 记 `region_mode=debias_fallback_continuous`）。
3. 区域损失的重组校验随模式切换：连续模式保留 v5 质量恒等式；硬模式用计数恒等式
   `native == (n_in·mean_in + n_out·mean_out)/4096`（精确成立）。

## 超参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--align-lambda` | 0.1 | 对齐项系数；0 = 完全禁用（不装 hook） |
| `--align-layers` | 13,6,16,14 | 对齐层；传单层即单层消融（如 `--align-layers 13`） |
| `--align-margin-scale` | 1.0 | σ 的逆温度 s（margin∈(−2,2)） |
| `--align-detach-mu` | off | on 时 μ 侧停梯度（仅 token 侧训练） |
| `--overshoot-fix` | on | off 复现 v5 objective |
| `--edit-loss-weight` | 0.4 | 仅 overshoot-fix=on 生效 |
| `--mask-debias` | on | 仅 overshoot-fix=on 生效 |
| `--general-replay-manifest` | 无（关） | JSONL：`{"pair_id", "source": {"prompt","cot_text","image"}}` |
| `--general-replay-every` | 4 | 每 N 个 macro step 插 1 个回放微步（native MSE+CE，0.5 缩放，无对齐） |
| `--init-trainable-from` | 无 | warm start：载 v5/v6 `trainable_state_dict`（键/形状严格匹配） |

history 行新增字段（带 `source_state_` / `target_state_` 前缀）：`align_loss`、
`align_margin`、`align_cos_in/out`、`align_tokens`、`align_skipped`、
`align_margin_L{13,6,16,14}`、`align_span_source`；objective 侧新增 `region_mode`、
`edit_weight`、`debias`；回放步新增 `replay_active/total/mse/ce`。

## 显存与吞吐预算（2× A100-40GB）

- v5 final991 实测峰值 reserved **38.029 GiB**（预算上限 38.5）。
- v6 增量：每层提取物 ≤ (n_tok×3584 + 2×3584) bf16 ≈ **数十 KB**（4 层 × 2 state 同时
  存活也 < 1 MB）；GEMV 无大临时张量；反向重算多执行 4×2 次 hook 提取（~0.5 GFLOP 级）。
  「保留 4 层完整层输出」的替代方案（~512 MB）已被该设计规避。
- 预计峰值增量 **< 0.1 GiB**，仍在 38.5 GiB 帽内；逐步峰值超帽时打
  `peak_memory_over_budget` 日志（不中断）。step 时间预计 +1~3%（prompt 重编码有
  per-(pair,state) 缓存，重复采样的 pair 直接命中）。

## CPU 自测（无 GPU、无模型权重）

`python3 selftest_v6.py` —— 20 项，**20/20 通过**：

- 损失数值：margin=+1/√8 精确重建 `−log σ(margin)`、符号序、尺度温度、bf16 容差、
  多层平均恒等、skip 路径、`empty_align_metrics`；
- 梯度：叶子梯度有限且互相独立；`detach_mu` 隔离 μ 侧、token 侧梯度与不 detach 一致；
- **AC parity（关键）**：`checkpoint_wrapper` NO_REENTRANT + hook 提取物与非 checkpoint
  基线的输入与全部参数梯度一致（atol 1e-5）；重算阶段不换 stash；unconfigured hook 零副作用；
- objective：开关全关时与 v5 `automask_forward_loss` 数值逐项一致；debias 手算对照与
  计数恒等式；全图掩码回退路径；`configure_overshoot` 参数校验；分区与 v5 逐张量相等；
- 布局：合成 packed 布局的正/负推导（含漂移必报错）、region 权重向量落位、多 span 索引；
- span：真实 40 对 manifest 记录 + 本地真实 tokenizer，解码重建出
  `right_region`（span=(241,243)）；缓存编码计数；
- CLI：默认值、`--overshoot-fix off` 的 effective 开关、λ=0、单层消融、全部非法参数
  退出、必选 `--activation-checkpointing`、resume guard 键覆盖。

## 启动（GPU 机器）

```bash
# 0) CPU 自测（本机即可）
bash run_v6.sh selftest
# 1) 只跑 manifest/objective/对齐命中率审计
bash run_v6.sh preflight
# 2) 2 步冒烟（throwaway 输出目录 gold40_automask_v6_align_smoke）
bash run_v6.sh smoke
# 3) 正式训练（nohup 双卡 torchrun + 30s watchdog 写 ../wb_e3_progress.txt）
bash run_v6.sh train
bash run_v6.sh status      # 随时查看进度/最新 macro_step 行
```

消融示例：`EXTRA_ARGS="--align-lambda 0 --overshoot-fix off" bash run_v6.sh smoke`
（复现 v5 配方对照）；`EXTRA_ARGS="--align-layers 13" bash run_v6.sh train`（单层）。

warm start 示例：
`EXTRA_ARGS="--init-trainable-from <...>/final991_automask_v5_train2973/checkpoints/latest.pt" bash run_v6.sh train`
（注意：从 final991 续训与从 base 新训是两种不同的实验语义，README 默认从 base 训练）。

## checkpoint / resume

- 每 `--checkpoint-every`（默认 200）步写 `checkpoints/latest.pt`；训练完成时另存
  `checkpoints/step_002973.pt`。`--resume auto` 断点续跑；resume guard 校验 manifest、
  objective、对齐/过冲/回放/warm-start 全部配置与 (seed, lr, lora_r, lora_alpha, regex)。
- v5 与 v6 checkpoint 格式互不识别；v6 的 `trainable_state_dict` 键与 v5 完全一致，
  E2 探针的 LoRA 直载流程对 v6 checkpoint 同样适用（探针重跑即可评估对齐训练效果）。

## 已知限制与风险

1. **FSDP×AC×辅助损失路径的组合**未在 CPU 上端到端验证（FSDP 需要 GPU）；hook 不改变
   FSDP/AC 的结构，`run_v6.sh smoke` 是第一道真实验证。若 smoke 出现
   `missing gradients` 或 recompute 相关报错，优先尝试 `--align-detach-mu`（把 μ 侧
   从梯度路径剥离）并将日志反馈。
2. source_state 分支的 in-mask 语义：automask 是双图差分，mask 同时覆盖新旧两个位置；
   source 分支的 `old_value` token 与「两处位置」的 μ_in 对齐，信号比 target 分支弱
   （E2 判据即在此约定下测得，保持一致）。
3. 连续 μ 的 GEMV 在 bf16 下累积（CUDA bf16 GEMV 内部 fp32 累加），对 margin 的数值
   噪声 ~1e-2 量级，远小于训练的随机性；如需更稳可在 hook 内先 `hidden.float()`
   （代价：一次性 (S, D) fp32 临时张量 ~235 MB，不建议默认开）。
4. 通用回放仅在显式提供 manifest 时生效；回放记录不做图像存在性预检（训练时报错跳过
   由 `_pack_single_pair` 的异常路径决定），建议先 `bash run_v6.sh preflight` 审计。

## 文件清单

- `align_loss.py` — span 定位（E2 移植）、packed 布局推导、`AlignHookManager`、
  对齐损失数学、回放记录校验（仅依赖 torch/stdlib）
- `align_objective.py` — v6 objective：v5 分区逐字复用 + `configure_overshoot`
  （edit 权重重配 / 硬分区去偏）+ 分模式重组校验
- `train_v6.py` — 训练主脚本（v5 忠实改造 + 对齐插入 + replay/warm-start/resume guard）
- `selftest_v6.py` — 20 项 CPU 自测
- `run_v6.sh` — selftest/preflight/smoke/train/status 启动器（后台 + 日志 + watchdog）
- `README.md` — 本文件

