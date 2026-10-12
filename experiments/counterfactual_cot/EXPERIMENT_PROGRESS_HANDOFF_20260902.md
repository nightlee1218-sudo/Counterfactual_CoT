# Counterfactual CoT 图像编辑实验：完整进度、证据与交接说明

> 更新时间：2026-09-02  
> 项目根目录：`/group/40016/niightli/BAGEL`  
> 实验目录：`/group/40016/niightli/BAGEL/experiments/counterfactual_cot`  
> 文档用途：供合作同学及其 Agent/模型快速、准确地理解当前研究问题、完整实验时间线、已经实现的工程、已获得的证据、证据边界、当前瓶颈与暂缓执行的后续方案。

---

## 0. 最重要的结论摘要

本项目研究 BAGEL 是否能够在给定 Source image 和反事实条件时，按照目标 CoT/编辑要求改变一个指定视觉原子，同时保持其他内容不变。当前最典型的 C1 任务是：将客厅中唯一的红色玩具球从 `left_region` 移到 `right_region`，保持灰色沙发、背景、视角、照明和球的颜色、类别、数量不变。

完整成功必须同时满足：

```text
右侧有红球
左侧旧球消失
红球总数恰好为 1
沙发、背景、视角和照明基本保持
```

截至目前，最可信的总体结论是：

1. 训练、梯度、optimizer、checkpoint、LoRA 注入和推理链路均已通过多轮工程审计，不能把失败简单归因为“代码没有训练”或“checkpoint 没加载”。
2. 最早的单 Pair、仅训练 14 个头尾/桥接 tensors、`lr=1e-4`、500-step 实验，曾明确在右侧生成红球，证明模型并非完全没有空间响应能力；但左侧旧球通常没有删除，结果更像“新增/复制右球”，而非真正移动。
3. 在 Transformer 主干 L24–27 加入 LoRA 后，`lr=1e-6` 的优化过弱；Micro4 固定评估只改善 3.77%，未达到预设 5% 门槛。
4. 将 Micro4 学习率提高到 `3e-5` 后，固定 flow-MSE 改善 56.61%，证明 LoRA 联合训练链路和容量能够显著拟合训练目标；但视觉编辑没有同步可靠改善，说明“MSE 下降”与“视觉 Grounding 成立”是两件事。
5. C1 Region-split loss 将编辑区和保持区分别归一化并各占 50%，成功排除了“背景面积自动支配全图 loss”的计算问题；但固定 `edit_mse` 到最佳 Step 400 只改善约 0.339%，说明仅提高编辑区权重不足以解决问题。
6. Prompt-only 消融中，只有直接动作式 `delta_only_v2` 出现较明显的右侧响应迹象；`legacy_paired_v1`、`cot_core_v1`、完整 `full_paired_v2`、Target-Prompt-only 等均未表现出稳定完整成功。不同 checkpoint 不是逐像素相同，但核心语义行为近似，说明当前 LoRA 参数更新没有转化为明显的目标语义增益。
7. 当前主要瓶颈更可能是：旧状态删除、数量守恒、Source image 强锚定、训练强度/学习率、训练与推理 Prompt 不匹配，以及 Target CoT 到视觉动作之间缺少 Grounding；问题不只是背景稀释，也不能仅用“LoRA 容量不足”解释。
8. 下一阶段训练方案已经设计，但按当前要求暂缓执行。应先把已有结果汇报、评分和归档清楚，再决定是否启动。

---

## 1. 研究问题与实验目标的演变

### 1.1 原始项目目标

项目最初构建了三类协议：

1. `Standard-CoT`：`Prompt + CoT -> source image`；
2. `Independent CF Edit`：输入 Source image、Target Prompt 和 Target CoT，但不提供 Source CoT、Intervention 或 Preservation；
3. `Paired CF-CoT MVP`：输入 Source image、Source Prompt/CoT、Intervention、Target CoT 和 Preservation，监督 Target image。

原始核心比较是：在相同 Base checkpoint、训练步数、尺寸、随机种子和数据预算下，`Paired CF-CoT` 是否优于 `Independent CF Edit`。

原始设计说明位于：

```text
experiments/counterfactual_cot/README.md
```

### 1.2 当前被扶正后的核心问题

在后续审计中发现，历史 `paired_prompt()` 实际没有向模型提供自然语言 `Target Prompt`，而是提供：

```text
Source image
Source Prompt
Source CoT
Intervention
Target CoT
Preservation
```

这使得“模型是否真正 follow Target CoT”无法被单独识别：即使成功，也可能依赖 Intervention、Preservation 或样本记忆；即使失败，也可能是 Source Prompt/Source CoT 的旧状态压过目标状态。

因此当前将研究目标区分为两个层次：

#### 目标 A：任务能力上限

输入所有一致信息：

```text
Source image
Source Prompt
Source CoT
Intervention
Target Prompt
Target CoT
Preservation
```

它回答“信息给足后模型是否有能力完成编辑”。即使成功，也不能单独证明 Target CoT 具有因果控制作用。

#### 目标 B：Target CoT 跟随能力

核心输入限定为：

```text
Source image
Source Prompt
Target CoT
```

不额外提供 Source CoT、Intervention、Target Prompt 或 Preservation。它回答“Target CoT 是否能够覆盖 Source image 和 Source Prompt 中的旧状态”。这是当前最核心的科学问题。

---

## 2. 数据、监督与成功定义

### 2.1 数据管线

项目实现了从受控 SceneGraph 到 Source/Target Pair 的完整流程：

1. 从模板生成 Source Prompt/CoT；
2. 构造 Source registry、标准化 CoT、去重；
3. 使用 Base BAGEL 生成 Source image；
4. 人工审核 Source；
5. 确定性地产生单原子 Intervention；
6. 使用同一 Source image 生成局部编辑 Target；
7. 生成候选差分 mask、ledger 和审核页面；
8. 按 TS/PS/BA/LS/IQ 做人工质量门禁；
9. 冻结 manifest 和图片血缘。

历史数据流程和命令见：

```text
experiments/counterfactual_cot/README.md
```

### 2.2 训练监督到底是什么

当前模型没有对 Target CoT 做文本 CE 监督。训练输入和监督关系是：

```text
Source VAE tokens：条件，不计算 loss
Source ViT tokens：条件，不计算 loss
Prompt/CoT text：条件，不计算文本 CE
Target VAE tokens：唯一直接接受图像 flow-matching loss 的部分
```

BAGEL 对 Target latent 构造带噪状态并预测 flow velocity。逐元素 MSE 的形状为：

```text
[256, 64]
```

含义是 256 个 Target VAE tokens（16×16），每个 token 64 个 latent 维度。

因此，Target CoT 当前只是条件输入；所谓“模型 follow CoT”，只能通过 Target image 的视觉行为间接判断，不能从训练 loss 本身直接推出。

### 2.3 正式成功标准

全项目人工 Grounding 标准为：

```text
TS == 1
PS >= 0.8
BA >= 0.8
LS >= 0.7
IQ >= 0.6
```

含义：

- `TS`：目标原子是否成功；
- `PS`：Preservation atoms 保持比例；
- `BA`：背景/非编辑内容保持；
- `LS`：编辑局部性；
- `IQ`：图像质量。

C1 单球任务使用更直观的主成功规则：

```text
right_ball_present == true
AND left_ball_present == false
AND ball_count == 1
```

右侧新增球但左球仍存在不能算成功。

---

## 3. 第一阶段：仅训练 14 个模型头尾/桥接 tensors

前三轮没有训练 Transformer 主干。共同 trainable regex 为：

```text
^(time_embedder|latent_pos_embed|vit_pos_embed|vae2llm|llm2vae|connector)\.
```

共 14 个 tensors、63,454,784 个参数，主要包括时间步嵌入、位置嵌入、VAE/LLM 双向映射和连接层。

### 3.1 第一轮：Gold40 全量，400 steps

配置：

```text
数据：Gold40 全量
steps：400
lr：1e-6（脚本默认及后续冻结基线）
训练范围：14 个头尾/桥接 tensors
主干：冻结
objective：原生全图 flow-matching MSE
```

主要入口与结果：

```text
training/gold40_overfit.py
outputs/gold40_overfit_single_400_safe1536/gold40_overfit_report.json
outputs/gold40_overfit_single_400_safe1536_acceptance_eval/
```

数值结果：

```text
Step 0 mean MSE：   0.652545
Step 400 mean MSE： 0.347381
相对下降：          46.765%
工程状态：          passed
```

解释：训练链路和 MSE 优化成立，但在 C1 示例中，Step 400 输出仍主要表现为左侧单球，没有明确执行左移右。该阶段证明了全量训练能够拟合 flow target，不能证明空间编辑 Grounding 成功。

### 3.2 第二轮：C1 单 Pair，500 steps，lr=1e-4

配置：

```text
Pair：cf_source_living_room_play_absolute_position_change_v4_absolute_region_replacement_v4_0000
任务：red toy ball, left_region -> right_region
steps：500
lr：1e-4
训练范围：相同 14 tensors
```

结果目录与报告：

```text
outputs/single_pair_overfit_diag_ball/
outputs/single_pair_overfit_diag_ball/single_pair_overfit_report.json
```

关键结果：

- 右侧出现了红球，是早期实验中最明确的目标位置响应；
- 左侧旧球未可靠删除，常形成左右两个球；
- 证明头尾/桥接参数已经能够推动目标语义变化；
- 但模型学到的更像“右侧新增对象”，而非“移动并保持数量为 1”。

这轮是当前极其重要的正控制：模型并非完全没有空间编辑能力，真正困难集中在旧对象删除和数量守恒。

### 3.3 第三轮：同一 C1 Pair，300 steps，lr=3e-5

配置：

```text
同一 Pair
steps：300
lr：3e-5
训练范围：相同 14 tensors
```

结果目录：

```text
outputs/single_pair_overfit_diag_ball_lr3e5/
```

之后又进行了严格可复现的 500-step 延长：

```text
outputs/single_pair_overfit_diag_ball_lr3e5_500steps/
outputs/single_pair_overfit_diag_ball_lr3e5_500steps/final_report.json
```

300-step 旧运行与 500-step 新运行的前 300 个 train-log rows 完全一致；共同 checkpoint 的 14 个 tensors 逐张量完全一致，证明延长实验不是另一次随机训练。

视觉观察：没有可靠解决左球删除问题，后期画质出现模糊/退化。该结论目前属于人工定性观察，不是正式盲评分数。

阶段结论：降低学习率没有自动解决双球和旧对象删除；延长训练也不保证视觉语义改善。

---

## 4. 第二阶段：在主干 L24–27 注入 LoRA

### 4.1 LoRA 配置和真实训练范围

配置：

```text
Transformer layers：24, 25, 26, 27
LoRA rank：8
LoRA alpha：16
LoRA dropout：0
LoRA modules：24
LoRA tensors：48（每个 module 的 A/B）
LoRA parameters：2,162,688
```

LoRA 阶段仍继续训练原来的 14 个头尾/桥接 tensors。因此总训练范围为：

| 参数组 | tensors | 参数量 |
|---|---:|---:|
| 既有头尾/桥接参数 | 14 | 63,454,784 |
| L24–27 LoRA | 48 | 2,162,688 |
| 合计 | 62 | 65,617,472 |

主干原权重冻结，只更新附加 LoRA A/B 参数。需要避免错误表述为“只训练 LoRA”或“LoRA 只有 62 万参数”。

### 4.2 Formal Gold40 LoRA，400 steps，lr=1e-6

配置：

```text
数据：Gold40 v3，40 pairs
steps：400
lr：1e-6
LoRA：L24–27, r=8, alpha=16
训练范围：14 existing tensors + 48 LoRA tensors
objective：全图平均 flow-MSE
```

结果：

```text
outputs/gold40_v3_lora_L24-27_r8_a16_lr1e6_singlepack_400steps/
outputs/gold40_v3_lora_L24-27_r8_a16_lr1e6_singlepack_400steps/gold40_lora_full_training_report.json
```

工程结论：

- LoRA 注入、梯度、optimizer、checkpoint 和 reload 均通过；
- 62 个 tensors 正常训练；
- 正式训练完成，报告 `status=passed`。

视觉证据边界：已经生成评估图和 annotation form，但正式人工 TS/PS/BA/LS/IQ 仍未填写。当前“偏模糊/卡通、位置和对象替换未稳定学会、颜色变化可能来自基础先验”只能记录为非正式观察，不能写成正式量化结论。

---

## 5. Micro4 最小容量与学习率诊断

### 5.1 Micro4，lr=1e-6，400 steps

动机：怀疑 40 个 Pair 的训练信号被不同任务平均，因此缩小到 4 个 Pair，并使用固定 deterministic eval 排除噪声差异。

结果目录：

```text
outputs/gold40_v3_lora_micro4_400steps/
outputs/gold40_v3_lora_micro4_deterministic_eval_v4/
outputs/gold40_v3_lora_micro4_deterministic_eval_v4/macro_avg_table.json
```

Macro MSE：

```text
Step 0：   0.4703174
Step 400： 0.4525991
相对改善：3.7673%
预设门槛：至少改善 5%
```

逐 Pair：

| Pair 类型 | 相对变化 |
|---|---:|
| City object replacement | 改善 8.81% |
| Classroom object replacement | 改善 4.59% |
| City position change | 恶化 1.83% |
| Classroom position change | 改善 3.02% |

结论：`lr=1e-6` 下优化信号较弱、跨 Pair 不一致，未达到预注册 5% 门槛。不能写成“完全没有下降”，但可以写成“优化未被证明成立，因此无法继续评价容量和 Grounding”。

### 5.2 Micro4，lr=3e-5，500 steps

动机：直接检验上一轮是否主要受学习率过低限制。

结果目录：

```text
outputs/gold40_v3_lora_micro4_lr3e5_500steps/
outputs/gold40_v3_lora_micro4_lr3e5_500steps_deterministic_eval_v1/
outputs/gold40_v3_lora_micro4_lr3e5_500steps_deterministic_eval_v1/macro_avg_table.json
outputs/gold40_v3_lora_micro4_lr3e5_500steps_diag_images_v1/
```

固定 Macro MSE：

| Step | Macro MSE | 相对 Step 0 |
|---:|---:|---:|
| 0 | 0.4703174 | 0% |
| 10 | 0.3098549 | 改善 34.12% |
| 100 | 0.2700343 | 改善 42.58% |
| 300 | 0.2407908 | 改善 48.80% |
| 400 | 0.2235401 | 改善 52.47% |
| 500 | 0.2040561 | 改善 56.61% |

结论：较高学习率显著解决了 MSE 优化不足，证明 LoRA 联合训练链路具有拟合能力。但视觉诊断没有显示与 MSE 同步的稳定语义改进；Step 300/500 的清晰度和回归原图等现象目前是非正式人工观察。正式结论只能是：

> 数值优化成立，不等于视觉 Grounding 成立。

---

## 6. C1 单球严格血缘 LoRA 对照

### 6.1 设计目的与血缘控制

为了排除数据不一致、初始化不一致和随机训练轨迹不一致，选取历史 C1 单球 Pair，并绑定：

- Pair ID；
- Source/Target 文件及 SHA256；
- Prompt/CoT/Intervention/Preservation；
- Formal B Step 0 的 62 个 tensors；
- 训练 packing、timestep 和 noise RNG trajectory；
- optimizer 类型与 checkpoint 保存计划。

500-step RNG trajectory 完全匹配历史基准。需要注意：RNG gate 证明训练随机轨迹可复现，不代表当前实验继承了历史 `lr=3e-5` 权重，也不能说唯一变量只有 LoRA；当前学习率、可训练参数集合和历史 14-tensor 实验并不完全相同。

### 6.2 C1 Full-image LoRA，500 steps，lr=1e-6

结果目录：

```text
outputs/gold40_v3_lora_c1_single_ball_p4d_v1/
outputs/gold40_v3_lora_c1_single_ball_p4d_v1/micro4_full_training_report.json
```

Legacy online eval：

| Step | mean MSE |
|---:|---:|
| 0 | 0.359286 |
| 10 | 0.325171 |
| 25 | 0.238215 |
| 50 | 0.264594 |
| 100 | 0.268067 |
| 200 | 0.240741 |
| 300 | 0.283922 |
| 400 | 0.212200 |
| 500 | 0.268593 |

Step 400 相对 Step 0 改善约 40.94%；Step 500 相对 Step 400 回升/恶化约 26.58%。但这一组是 legacy online eval，没有像后续 Region-split 一样明确绑定同一个固定 eval seed 和 timestep hash，不能称为严格固定输入的 checkpoint 排名。

视觉上没有稳定完成左球移右。结论是训练存在数值变化和非单调轨迹，但未证明视觉 Grounding。

---

## 7. Region-split 新损失实验

### 7.1 动机

C1 编辑区只占约 7.4614%，保持区占约 92.5386%。原生全图平均 MSE 可近似理解为：

```text
0.0746 × edit_mse + 0.9254 × preserve_mse
```

因此提出“编辑区梯度被大面积背景稀释”的假设。

### 7.2 Mask 构造

固定 Source/Target 上构造：

1. RGB 最大通道差值阈值 24；
2. 膨胀 9 pixels；
3. 3-pixel median filter；
4. old-ball ROI 和 new-ball ROI 先取 union；
5. candidate difference mask 与 ROI union 取 intersection。

准确公式：

```text
M_edit = M_difference ∩ (ROI_old ∪ ROI_new)
```

再平均池化到 16×16、共 256 个 Target VAE token 的软权重。

### 7.3 Loss 公式

对每个 token 的 64 个 latent 维度先求均值，再计算：

```text
edit_mse     = sum(w_i * E_i) / sum(w_i)
preserve_mse = sum((1-w_i) * E_i) / sum(1-w_i)
L_region     = 0.5 * edit_mse + 0.5 * preserve_mse
```

编辑区单位面积监督相对原全图 loss 增强约 6.70 倍；保持区单位面积约为原来的 0.54 倍。保持区仍有监督，并非只训练球区域。

### 7.4 工程和数学审计

实现：

```text
training/gold40_v3_lora_c1_single_ball_region_split.py
```

结果：

```text
outputs/gold40_v3_lora_c1_single_ball_region_split_v1_preflight/
outputs/gold40_v3_lora_c1_single_ball_region_split_v1_500steps/
```

已验证：

- Source/Target SHA256 固定；
- candidate mask 和 token mask 可逐像素/逐 token 重建；
- old/new 两侧 ROI 都非空；
- edit/preserve 分区可严格重组 native MSE；
- 输出形状严格为 `[256,64]`，没有错误监督 Source tokens；
- Step 0 加载 62/62 tensors；
- 与历史 B Step 0 逐 tensor 完全相等；
- 500/500 RNG trajectory 完全匹配；
- AdamW、checkpoint 和 dry-run 报告身份绑定通过；
- 9 个 checkpoints（0/10/25/50/100/200/300/400/500）齐全；
- `train_log.jsonl` 完整；
- 无 NaN/Inf；
- 正式报告 `status=passed`。

### 7.5 固定评估结果

所有 checkpoint 使用相同：

```text
eval_seed = 1460362567
packed_timesteps_hash = 4135670a...
```

关键点：

| Step | edit_mse | preserve_mse | Region total |
|---:|---:|---:|---:|
| 0 | 0.660556 | 0.315938 | 0.488247 |
| 50 | 0.661073 | 0.315345 | 0.488209 |
| 100 | 0.663146 | 0.317451 | 0.490298 |
| 200 | 0.660280 | 0.316540 | 0.488410 |
| 300 | 0.660556 | 0.315731 | 0.488143 |
| 400 | 0.658319 | 0.315379 | 0.486849 |
| 500 | 0.659837 | 0.315694 | 0.487766 |

Step 400 相对 Step 0：

```text
edit_mse 改善约 0.339%
Region total 改善约 0.286%
```

Step 500 又轻微回退。

结论：Region-split 成功消除了背景面积自动主导 loss 的计算机制，但编辑区自身仍几乎学不动。因此“只要解决背景稀释即可成功”的解释被明显削弱。由于本轮仍使用 `lr=1e-6`，不能完全否定背景稀释的影响；准确结论是，在当前学习率、容量和 500-step 设置下，单独提高编辑区权重不足以解决问题。

---

## 8. C1 Full-image 与 Region-split 视觉比较

实现：

```text
training/c1_region_split_visual_compare.py
```

输出：

```text
outputs/gold40_v3_lora_c1_region_split_visual_compare_v1/
outputs/gold40_v3_lora_c1_region_split_visual_compare_v1/visual_compare_report.json
```

固定 Source、Legacy Prompt、seeds、CFG、采样步数和模型加载协议，对比 Full-image 与 Region-split 的 Step 0/400/500。两组 Step 0 已验证逐像素一致。

结果：Full-image 和 Region-split 的后期 checkpoint 都没有稳定把红球从左侧移动到右侧。checkpoint 产生了低层视觉变化，但没有产生明确的目标语义变化。

---

## 9. Prompt 协议审计与 Prompt-only 消融

### 9.1 为什么要做

历史 `legacy_paired_v1` 没有包含 Target Prompt，并且同时包含多个目标信号，不能单独判断模型是否 follow Target CoT。因此新增版本化 Prompt profiles，不修改 legacy 函数，避免破坏旧实验可复现性。

实现：

```text
data/prompt_profiles.py
training/c1_prompt_ablation_visual_compare.py
evaluation/summarize_c1_prompt_ablation.py
tests/test_prompt_profiles.py
```

### 9.2 六个 Prompt profiles

| Profile | 输入文本组件 | 目的 |
|---|---|---|
| `legacy_paired_v1` | Source Prompt + Source CoT + Intervention + Target CoT + Preservation | 历史训练协议 |
| `cot_core_v1` | Source Prompt + Target CoT | 核心 CoT 跟随测试 |
| `full_paired_v2` | Source Prompt/CoT + Intervention + Target Prompt/CoT + Preservation | 信息完整的能力上限 |
| `full_without_target_cot_v1` | 与 `full_paired_v2` 相同，但删除 Target CoT | Target CoT 严格消融 |
| `target_prompt_only_v1` | Target Prompt | 自然语言目标控制 |
| `delta_only_v2` | 直接移动、删除旧球、保持数量和背景的动作指令 | 直接编辑正控制 |

命名中的 `v1/v2` 是每个 Prompt 模板自己的版本，不是训练轮次或效果等级：

- `legacy_paired_v1` 是历史 paired 第一版；
- `full_paired_v2` 是加入 Target Prompt 的 paired 第二版；
- `delta_only_v2` 是在旧 direct-delta 思路上补充“删除旧球、保持一个球”的第二版；
- 其他 `v1` 是对应 profile 的首次冻结定义。

### 9.3 实验矩阵

四个 checkpoint：

```text
common_step000
full_image_step500
region_split_step400
region_split_step500
```

六个 Prompt、三个 seeds：

```text
1234, 2234, 3234
```

总计：

```text
4 × 6 × 3 = 72 张图
```

输出：

```text
outputs/c1_prompt_ablation_v1/
outputs/c1_prompt_ablation_v1/prompt_manifest.json
outputs/c1_prompt_ablation_v1/prompt_ablation_report.json
outputs/c1_prompt_ablation_v1/blind_manual_review.json
outputs/c1_prompt_ablation_v1/blind_review_key.json
outputs/c1_prompt_ablation_v1/contact_sheets_v1/
```

工程结果：72/72 图片、24/24 workers 全部完成，`failures=[]`，报告 `status=passed`。这里的 `passed` 只代表生成流程完整，不代表语义成功。

### 9.4 当前视觉观察

盲评表目前仍未正式填写，所以下列只能写成非正式观察：

- `legacy_paired_v1`：没有稳定移动球；
- `cot_core_v1`：没有明显 follow Target CoT；
- `full_paired_v2`：即使信息全部给足也没有稳定成功；
- `full_without_target_cot_v1`：同样未稳定成功；
- `target_prompt_only_v1`：自然语言 Target Prompt 也未稳定成功；
- `delta_only_v2`：唯一出现较明显的成功迹象/右侧响应，但仍需确认左球是否删除和总数是否为 1。

不同 checkpoint 的输出不是逐像素一致，但同 Prompt、同 seed 下的核心语义行为高度接近。当前更合理的判断是：训练确实更新了参数，但没有转化成明显、稳定的 Prompt-conditioned 空间编辑能力。

### 9.5 证据边界

只有 `legacy_paired_v1` 与当前 C1 checkpoint 的训练 Prompt 匹配。其余五个 Prompt 都是在推理阶段第一次使用，属于 Prompt OOD/鲁棒性诊断。

因此，当前 Prompt-only 消融不能证明：

- 使用 `cot_core_v1` 重新训练也一定失败；
- Target CoT 永远没有作用；
- `delta_only_v2` 用于训练一定成功。

它只能证明：使用 Legacy Prompt 训练得到的现有 checkpoint，不能零样本稳定响应这些新 Prompt。

---

## 10. 工程上已经完成的工作与投入

为避免把随机波动、文件错载或评估不一致误判为研究结论，已完成以下工程：

1. 冻结数据 manifest、Pair ID、Source/Target 路径和 SHA256；
2. 对历史单 Pair 训练做 300-step/500-step 日志、checkpoint 和图片逐项复现核对；
3. 对 C1 LoRA 建立 Step 0 逐 tensor 身份校验；
4. 对 500-step packing/noise/timestep RNG trajectory 做逐步 hash 对齐；
5. 对 EMA 基础模型加载数量、预期 missing positional tensors 和 unexpected tensors 建立 gate；
6. 对 AdamW 配置、optimizer 参数集合和 LoRA target modules 建立审计；
7. 对 Region mask 构造、ROI、面积、token pooling 和 native-MSE 重组做数学验证；
8. 将正式训练和 dry-run 报告做 SHA256 身份绑定；
9. 保存密集 checkpoint 轨迹，避免只观察最终 Step 500；
10. 建立固定 deterministic eval，区分训练 batch loss 与可比的 checkpoint 指标；
11. 建立版本化 Prompt renderer，保留 Legacy Prompt 的 bit-exact 行为；
12. 建立 Prompt-only parent/worker 生成、checkpoint/Prompt/seed/PNG SHA 审计；
13. 建立匿名随机顺序盲评模板和独立解盲 key；
14. 建立人工评分完整性、图片 SHA、generation report 身份校验和汇总脚本；
15. 生成按 checkpoint 和按 Prompt 分组的 contact sheets，便于人工比较。

这些工作的重要意义是：目前的负面结果不是来自一个未经核验的单次训练，而是在数据、初始化、RNG、loss 数学关系、checkpoint 加载和推理输入均被持续审计后得到的。

---

## 11. 当前最可信的科学结论

### 11.1 已经证明

1. BAGEL 的训练和推理工程链路有效；
2. 早期 `lr=1e-4` 的 14-tensor 单 Pair 训练能够在右侧生成目标球；
3. 主要困难不是“完全生成不了右球”，而是删除左球并保持数量为 1；
4. L24–27 LoRA 已正确注入、获得梯度、更新并从 checkpoint 恢复；
5. `lr=1e-6` 对当前 LoRA 联合训练过弱；
6. `lr=3e-5` 能显著降低固定 flow-MSE；
7. flow-MSE 大幅下降不等价于视觉 Grounding 成功；
8. 背景面积稀释不是当前失败的唯一原因；
9. Region 50/50 加权在 `lr=1e-6` 下不足以推动 edit region 收敛；
10. 直接动作式 Prompt 比结构化状态式 CoT 更容易触发右侧响应；
11. 当前 LoRA checkpoint 的低层参数/像素变化没有形成明显的目标语义增益。

### 11.2 尚未证明

1. 用 `cot_core_v1` 直接训练后是否能学会 Target CoT；
2. Target CoT 是否对视觉结果有独立因果控制作用；
3. 较高学习率的 C1 LoRA 是否能恢复早期右球响应并改善旧球删除；
4. Source image CFG 是否过强；
5. 训练和推理分辨率错配是否是重要因素；
6. 将 old-ball 删除、新-ball 生成、背景保持拆成三部分监督是否有效；
7. 扩大 LoRA 层数或 rank 是否必要；
8. 当前任何结果是否能泛化到未见 Pair；
9. Formal Gold40 的正式 TS/PS/BA/LS/IQ 结果，因为人工评分尚未完成。

---

## 12. 当前瓶颈的优先级判断

按现有证据，优先级从高到低为：

1. **训练强度/学习率不足**：`1e-6` 的 Region fixed edit-MSE 几乎不动，而 Micro4 `3e-5` 可大幅优化 MSE；
2. **旧状态删除和数量守恒困难**：早期高学习率实验和当前 `delta_only_v2` 都更容易新增右球，而不是删除左球；
3. **Target CoT 缺少视觉 Grounding**：结构化状态描述没有表现出直接动作指令同等的响应；
4. **Source image 锚定过强**：Source VAE/ViT 双路条件加 image CFG 可能鼓励复制左球；
5. **train/eval Prompt 不匹配**：当前非 Legacy profiles 都是推理 OOD；
6. **训练/推理分辨率差异**：弱训练信号可能在高分辨率推理中进一步消失；
7. **loss 仍不够针对删除/生成解耦**：Region-split 把旧球和新球合并为一个 edit 区，无法分别诊断删除和生成。

目前没有足够证据把“LoRA 容量不足”排在第一位，因为高学习率 Micro4 已显示显著拟合能力，且早期仅训练头尾参数就能生成右球。

---

## 13. 2026-09-03 精简裁决与下一步实验

### 13.1 审计范围冻结

不再重复数据血缘、LoRA 注入、梯度、optimizer、checkpoint、reload、RNG、Step-0 对齐和 Region-split 数学审计。关键历史运行已登记到：

```text
evaluation/experiment_registry.jsonl
```

以下结论直接复用：`lr=1e-6` 过弱；`lr=3e-5` 能显著降低固定 MSE；500 steps 不保证语义改善；当前 50/50 Region-split 不是充分解；MSE 下降不等于 Grounding；C1 暴露的“新增而不删除”是机制案例而非全项目结论。

### 13.2 零训练 Gold40 证据回收

先复用 `outputs/gold40_v3_ab_aligned_inference_nt25_protocol_v1/` 中现有 160 张图，不重新生成、不重新加载 checkpoint。按五类干预各选 2 条，共 10 条，比较 A/B 的 Step 0 与 Step 400：

- `attribute_replace`：目标属性出现、原属性消失、对象身份保持；
- `count_change`：目标数量正确、无额外对象；
- `object_replace`：新对象出现、旧对象消失、位置保持；
- `spatial_position_change`：目标位置正确、原位置清空、数量守恒；
- `spatial_relation_change`：目标关系成立、原关系解除、对象保持。

只做一次结构化自动评分；不做双人盲评。每项记录 `target_gain`、`source_removal`、`preservation`、`quality` 和 `uncertain`。Gold40 是训练集，因此结果只用于选择方向和样本，不能声称泛化。

### 13.3 唯一优先的新训练：Sentinel-6 短程训练匹配门

样本覆盖全部五类任务和至少五个场景族，C1 只作为历史锚点：

```text
attribute: cf_source_library_table_attribute_replace_0000
count:     cf_source_seaside_path_count_change_0000
object:    cf_source_classroom_object_replace_0000
position:  cf_source_city_street_absolute_position_change_v4_absolute_region_review_fix_v6_short_0000
relation:  cf_source_home_kitchen_spatial_relation_change_0000
anchor:    cf_source_living_room_play_absolute_position_change_v4_absolute_region_replacement_v4_0000
```

只比较两个训练匹配条件：

```text
delta_only_v2：直接动作指令，作为编辑能力正控制
cot_core_v1：Source Prompt + Target CoT，检验 CoT grounding
```

冻结配置：

```text
初始化：同一个已验证 Step 0
训练参数：14 bridge/head-tail tensors + L24-27 LoRA r8/a16
lr：3e-5
steps：50
checkpoint：0 / 10 / 25 / 50
训练 seed：先仅 1 个
推理：仅 Step 0 与 Step 50，各 3 个固定 generation seeds
推理协议：num_timesteps=25, cfg_text=4.0, cfg_img=2.0, 1024×1024
loss：先沿用原生 full-image flow-MSE，不重复 Region-split
```

唯一实验变量是训练 Prompt profile；训练和推理必须使用同一 profile。中间 checkpoint 只算固定 MSE，不生成额外图片。总训练仅 2 个短程 run，总新图为 `6 × 2 × 2 × 3 = 72` 张。

### 13.4 进展判据与硬停止规则

单样本联合成功要求：目标状态出现、旧状态消失、数量/关系正确、非编辑内容可接受。阶段性进展要求：

```text
至少 3/5 个 intervention types 在 Step 50 相对 Step 0 改善；
且其中每类至少 2/3 generation seeds 联合成功；
且不能只表现为新增目标而旧状态残留。
```

结果直接决定下一步：

- `delta_only_v2` 成功、`cot_core_v1` 失败：编辑链有能力，瓶颈是 CoT grounding；停止调 loss/LoRA，集中改 CoT 表示与训练。
- 两者都成功：此前主要受低学习率和 Prompt mismatch 限制；下一轮再做 Independent、Paired-no-CoT、Paired-CoT 公平对照。
- 两者 MSE 都下降但语义都失败：确认 objective-metric mismatch；停止扩大 Gold40 和 CoT sweep，转向 source-state overwrite 目标。
- 两者 MSE 都不降：只检查本轮 Prompt 数据视图和优化契约，不搜索 rank、层数或 500 steps。
- 仅 C1 改善：判定为个例，不升级项目结论。

只有命中前三种中的明确分叉才算本轮完成；不以“又生成了一批图片”作为进展。

---

## 14. 关键代码与结果索引

### 数据与 Prompt

```text
experiments/counterfactual_cot/README.md
experiments/counterfactual_cot/data/render_prompts.py
experiments/counterfactual_cot/data/prompt_profiles.py
experiments/counterfactual_cot/datasets/bagel_manifest_dataset.py
```

### 早期 14-tensor 训练

```text
experiments/counterfactual_cot/training/gold40_overfit.py
experiments/counterfactual_cot/outputs/gold40_overfit_single_400_safe1536/
experiments/counterfactual_cot/outputs/single_pair_overfit_diag_ball/
experiments/counterfactual_cot/outputs/single_pair_overfit_diag_ball_lr3e5/
experiments/counterfactual_cot/outputs/single_pair_overfit_diag_ball_lr3e5_500steps/
```

### LoRA、Micro4 与 C1

```text
experiments/counterfactual_cot/training/gold40_lora.py
experiments/counterfactual_cot/outputs/gold40_v3_lora_L24-27_r8_a16_lr1e6_singlepack_400steps/
experiments/counterfactual_cot/outputs/gold40_v3_lora_micro4_400steps/
experiments/counterfactual_cot/outputs/gold40_v3_lora_micro4_deterministic_eval_v4/
experiments/counterfactual_cot/outputs/gold40_v3_lora_micro4_lr3e5_500steps/
experiments/counterfactual_cot/outputs/gold40_v3_lora_micro4_lr3e5_500steps_deterministic_eval_v1/
experiments/counterfactual_cot/outputs/gold40_v3_lora_c1_single_ball_p4d_v1/
```

### Region-split

```text
experiments/counterfactual_cot/training/gold40_v3_lora_c1_single_ball_region_split.py
experiments/counterfactual_cot/outputs/gold40_v3_lora_c1_single_ball_region_split_v1_preflight/
experiments/counterfactual_cot/outputs/gold40_v3_lora_c1_single_ball_region_split_v1_500steps/
experiments/counterfactual_cot/training/c1_region_split_visual_compare.py
experiments/counterfactual_cot/outputs/gold40_v3_lora_c1_region_split_visual_compare_v1/
```

### Prompt ablation

```text
experiments/counterfactual_cot/data/prompt_profiles.py
experiments/counterfactual_cot/training/c1_prompt_ablation_visual_compare.py
experiments/counterfactual_cot/evaluation/summarize_c1_prompt_ablation.py
experiments/counterfactual_cot/tests/test_prompt_profiles.py
experiments/counterfactual_cot/outputs/c1_prompt_ablation_v1/
experiments/counterfactual_cot/outputs/c1_prompt_ablation_v1/contact_sheets_v1/
```

最建议先看的对比图：

```text
contact_sheets_v1/by_prompt__delta_only_v2.jpg
contact_sheets_v1/by_prompt__cot_core_v1.jpg
contact_sheets_v1/by_prompt__full_paired_v2.jpg
contact_sheets_v1/by_prompt__full_without_target_cot_v1.jpg
```

---

## 15. 交接给下一个 Agent 时必须遵守的规则

1. 不覆盖或重命名任何历史 output、checkpoint、manifest 或报告；
2. 不修改 `legacy_paired_v1` 的语义；新 Prompt 必须使用新版本名；
3. 不把 `status=passed` 理解为视觉成功，它通常只代表工程流程通过；
4. 不把训练 batch loss 或 flow-MSE 降低等同于 Grounding 成功；
5. 不把未填写的人工评分当成 0，也不能据此计算正式成功率；
6. 区分训练 Prompt 与推理 Prompt；Prompt-only OOD 结果不能冒充新 Prompt 训练结果；
7. Full-image 和 Region-split 不是两个模型架构，只是 loss reduction 不同；
8. Region mask 只参与训练 loss，推理时不输入模型；
9. C1 成功必须同时检查右球、左球删除和球数为 1；
10. 所有 checkpoint 比较必须固定 Source、Prompt、seed、CFG、采样步数和模型加载方式；
11. 新训练必须新建输出目录、保存 Step 0、绑定 Prompt/数据/checkpoint/code SHA；
12. 当前后续训练计划处于“暂缓”状态，未经明确要求不要启动；
13. 当前不安排双人 blind review；先使用一次结构化自动评分做方向门禁，低于内部标准的结果不进入正式评测；
14. 当前所有结果主要是训练集容量/过拟合诊断，不应写成泛化结论。

---

## 16. 最终汇报用一句话

> 我们从仅训练 BAGEL 头尾桥接参数开始，先在 Gold40 和单 Pair 上确认训练链路与空间响应能力，再在主干 L24–27 注入 LoRA，通过 Formal Gold40、Micro4 低/高学习率、C1 严格 RNG 血缘、Region-split 新损失和 72 张 Prompt-only 消融逐层排查；目前已证明高学习率能够显著优化 flow-MSE、早期单 Pair 能在右侧生成球、背景稀释不是唯一原因，但尚未解决左球删除、数量守恒和 Target CoT 到视觉动作的 Grounding，且现有 LoRA checkpoint 没有表现出可靠语义增益。工程审计已经较完整，下一步应先做 Source-anchor/分辨率诊断和高学习率短程正控制，而不是盲目延长当前低学习率训练；该计划目前暂缓执行。
