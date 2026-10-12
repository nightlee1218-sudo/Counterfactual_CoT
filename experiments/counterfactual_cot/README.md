# Counterfactual CoT Grounding

本目录在开发机中的目标位置为：

```text
/group/40016/niightli/BAGEL/experiments/counterfactual_cot
```

所有 Python 模块命令都应从 BAGEL 根目录 `/group/40016/niightli/BAGEL` 执行。首期训练数据使用受控模板自建；COCO 不作为首期来源，R2I-Bench 与 GenEval 只用于最终独立测试。

## 正确流程

1. `generate_source_scenes` 从受控模板生成多场景 source Prompt/CoT；当前候选文件为 `data/source_scenes_gold40.jsonl`，覆盖 10 类场景，四种预期干预各 10 条。
2. 人工审核 source JSONL 中 Prompt 与 CoT 是否一致；这里只审核文本和结构，不审核尚未生成的图片。
3. `build_source_registry` 标准化 CoT、生成 `cot_text` 和 `scene_hash`、去重。
4. `generate_source_images` 使用原始 Base BAGEL，`think=False`，输入固定 Prompt 与我们构造的固定 CoT。每个 `scene_id` 只生成一次 source 图。
5. 人工审核 `images/review/source_review.csv`，使用 `apply_source_reviews` 把结果写回 registry。正式干预生成默认只接受 `source_quality.accepted=true` 的场景。
6. `generate_interventions` 根据每条 source 的 `primary_intervention_type` 和 `primary_entity_id` 确定性地只改一个 CoT 原子，并自动把其他全部原子列入 preservation。
7. `generate_pair_images` 只生成 target：Nano Banana Pro 读取同一张 source 图做局部编辑，不从零重画；同时生成候选像素差 mask、写 ledger、做本地检查并导出 CSV/HTML。
8. 人工给 TS/PS/BA/LS/IQ 打分。自动检查永远不会设置 `accepted=true`。
9. 过滤 accepted pair，按 `scene_id` 切分 train/val/test，冻结 manifest 并校验。
10. 依次执行单 batch smoke、20–50 对 Gold overfit、约 1000 对 Pilot，再开始正式对照训练。

## Source Prompt/CoT

从 BAGEL 根目录生成可审核的 40 条候选数据：

```bash
python -m experiments.counterfactual_cot.data.generate_source_scenes \
  --output experiments/counterfactual_cot/data/source_scenes_gold40.jsonl \
  --seed 20260817
```

生成器不调用外部 LLM，也不产生图片。它先构造结构化 SceneGraph，再确定性渲染 Prompt，因此 Prompt 与 CoT 共用同一个事实来源。候选集覆盖厨房、客厅、商店、城市街道、花园、教室、花店、图书馆、湖边和海滨等场景。每条记录的 `metadata` 指定后续应修改的主要实体和干预类型。

## Source 图

```bash
python -m experiments.counterfactual_cot.data.build_source_registry \
  --input /path/source_scenes.jsonl \
  --output /path/source_registry.jsonl

python -m experiments.counterfactual_cot.data.generate_source_images \
  --source-registry /path/source_registry.jsonl \
  --output-registry /path/source_registry_with_images.jsonl \
  --image-root /group/40016/niightli/BAGEL/experiments/counterfactual_cot/images_gold \
  --bagel-root /group/40016/niightli/BAGEL \
  --model-path models/BAGEL-7B-MoT \
  --backend base_bagel \
  --resume
```

不使用 GPU 时可将 `--backend` 改为 `mock`，仅测试流程。

人工编辑 `images_gold/review/source_review.csv` 的 `accepted` 和 `review_notes` 后执行：

```bash
python -m experiments.counterfactual_cot.data.apply_source_reviews \
  --source-registry /path/source_registry_with_images.jsonl \
  --review-csv /group/40016/niightli/BAGEL/experiments/counterfactual_cot/images_gold/review/source_review.csv \
  --output-registry /path/source_registry_reviewed.jsonl
```

## Intervention 与 Target 图

```bash
python -m experiments.counterfactual_cot.data.generate_interventions \
  --source-registry /path/source_registry_reviewed.jsonl \
  --output /path/intervention_pairs.jsonl \
  --max-per-scene 1

export CFCOT_IMAGE_API_KEY="你的 Venus 代理 token"
python -m experiments.counterfactual_cot.data.generate_pair_images \
  --manifest /path/intervention_pairs.jsonl \
  --output-manifest /path/pairs_with_images_smoke.jsonl \
  --image-root /group/40016/niightli/BAGEL/experiments/counterfactual_cot/images_gold \
  --provider nano_banana_pro \
  --model gemini-3-pro-image \
  --api-key-env CFCOT_IMAGE_API_KEY \
  --resume \
  --max-pairs 2 \
  --timeout-s 120 \
  --max-retries 3 \
  --rate-limit-qps 0.2
```

如果 source 图片位于另一目录，额外传 `--input-root`。manifest 中图片路径统一相对 `image-root` 保存。

验证：

```bash
python -m experiments.counterfactual_cot.data.validate_manifest \
  --manifest /path/pairs_with_images_smoke.jsonl \
  --root /group/40016/niightli/BAGEL/experiments/counterfactual_cot/images_gold
```

## 三个训练组

`Standard-CoT` 使用普通非反事实 source：`Prompt + CoT -> source image`。

`Independent CF Edit` 使用与 Paired 相同的 source image 和 target image，但输入只包含 source image、target Prompt 与 target CoT，不提供 intervention、source CoT 或 preservation。它控制相同图片和相同编辑模态带来的收益。

`Paired CF-CoT MVP` 输入 source image、source Prompt/CoT、intervention、target CoT 和 preservation，监督 target image；当前损失仍是 BAGEL 原生图像生成 MSE，不宣称已实现 mask preservation loss 或 visual-delta loss。

关键结论依赖 `Paired > Independent`。三组必须使用相同 Base checkpoint、训练步数、图像尺寸、随机种子和合理匹配的数据预算。

## 人工质量门禁

本地检查只验证图片可读、尺寸一致、source 与 target 不完全相同、候选 mask 非空且不覆盖整图。TS、PS、BA、LS、IQ 是语义判断，必须由独立 Judge 加人工复核获得。生成模型不能同时担任最终 Judge。测试集应全部人工核验。

只有满足以下条件并人工设置 `accepted=true, review_status=accepted` 的 pair 才能进入训练：

```text
TS = 1.0
PS >= 0.8
BA >= 0.8
LS >= 0.7
IQ >= 0.6
```