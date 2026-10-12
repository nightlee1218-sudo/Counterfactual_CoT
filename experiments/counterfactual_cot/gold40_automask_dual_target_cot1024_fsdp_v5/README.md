# Gold40 AutoMask-v5 paired CoT1024 FSDP

This is an independent v5 training entrypoint.  It consumes the v5 **AutoMask ready** paired manifest and intentionally does not require, inspect, or synthesize human-review fields.

## Invariants

- One macro step samples one arbitrary paired record (not a hard-coded single pair), then processes `source_state` and `target_state` in that order.
- Both branches use the same stateless pack seed, raw diffusion timestep, and identically seeded controlled noise; each backward is scaled by `0.5`, followed by exactly one optimizer step.
- Timestep mixture is native/high/pure-noise = 50/40/10.
- Training uses two-rank FSDP `FULL_SHARD`, BF16 compute, FP32 LoRA/bridge trainable masters, and exactly 28 activation-checkpointed decoder layers.
- The objective API is independently owned at `experiments.counterfactual_cot.gold40_automask_v5_data.automask_objective`, exposing `OBJECTIVE_ID`, `build_paired_automask_objective(record)`, and `paired_automask_forward_loss(...)`.

The default manifest is:

```text
/group/40016/niightli/BAGEL/experiments/counterfactual_cot/gold40_automask_v5_data/artifacts/gold40_source_target_automask_v5.jsonl
```

A ready record must provide `training_readiness.ready_for_training: true`, AutoMask path/hash metadata, source/target assets, and exactly the two paired branches.  The loader never consults `review` or `human_review`.

## CPU-only preflight

Preflight validates model assets, all manifest assets, and every per-pair AutoMask objective before it allocates a CUDA model:

```bash
cd /group/40016/niightli/BAGEL
python experiments/counterfactual_cot/gold40_automask_dual_target_cot1024_fsdp_v5/train.py \
  --manifest /absolute/path/to/gold40_source_target_automask_v5.jsonl \
  --output-dir /tmp/gold40-automask-v5-preflight \
  --preflight-only
```

## Two A100 40GB runs

Smoke (2 macro steps):

```bash
MANIFEST=/absolute/path/to/gold40_source_target_automask_v5.jsonl \
bash experiments/counterfactual_cot/gold40_automask_dual_target_cot1024_fsdp_v5/run_smoke_2xa100_40gb.sh
```

Formal 960-macro-step run:

```bash
MANIFEST=/absolute/path/to/gold40_source_target_automask_v5.jsonl \
bash experiments/counterfactual_cot/gold40_automask_dual_target_cot1024_fsdp_v5/run_train960_2xa100_40gb.sh
```

Set `GPU_IDS`, `MODEL_DIR`, and `OUTPUT_DIR` to override defaults.  These scripts only launch training; they do not auto-run.
