# Gold40 v4 Baseline Plan Generator

该目录只生成可审计的实验计划与启动模板，**不会启动训练**，也不会修改任何训练实现。

它要求互不重叠的、已人工审核的 train/eval JSONL；每条记录必须有明确的最终批准状态（例如 `review_status: "approved"`）。任何 `candidate`、`unreviewed`、`pending` 或缺失批准状态都会被拒绝。

```bash
cd /group/40016/niightli/BAGEL
python experiments/counterfactual_cot/gold40_v4_baselines/build_baseline_plan.py \
  --reviewed-split /absolute/path/to/reviewed_split_directory \
  --output-dir /absolute/path/to/gold40_v4_baseline_plan
```

目录输入可使用 `reviewed_split.json`（含 `train_manifest` 和 `eval_manifest`）、`reviewed_train.jsonl` 与 `reviewed_eval.jsonl`，或 split-builder 默认写出的 `train.jsonl` 与 `test.jsonl`。也可以显式指定：

```bash
python experiments/counterfactual_cot/gold40_v4_baselines/build_baseline_plan.py \
  --reviewed-split /absolute/path/to/reviewed_train.jsonl \
  --eval-manifest /absolute/path/to/reviewed_eval.jsonl \
  --output-dir /absolute/path/to/gold40_v4_baseline_plan
```

输出的 `experimental_plan.json` 锁定输入清单的绝对路径、SHA256、记录数、评估协议和各 baseline 的预期输出目录。`launchers/*.sh` 固定 `nproc_per_node=2`：`launch_paired_v4_main.sh` 对应已审核的 split-aware 两卡训练入口，只有手动运行该模板才会训练；其余消融均以状态码 2 明确 fail-fast，直到各自的训练实现和 manifest 契约经审核。当前没有注册 split-aware 评估入口，因此评估协议只锁定在计划中，不会伪造评估调用。
