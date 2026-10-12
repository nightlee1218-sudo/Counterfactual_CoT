# Gold40 v5 auto-mask objective

`automask_objective.py` provides the standalone AutoMask-v5 spatial-weight objective for one macro pair:

- `build_automask_objective(record)` validates the v5 macro record and returns its auditable shared 64×64 CPU `edit`/`outside` token partition.
- `build_paired_automask_objective(record)` returns the same partition under both `source_state` and `target_state` for the paired trainer.
- `paired_automask_forward_loss(model, data, branch_id, branch_weights, autocast_context)` calculates one branch loss; `automask_forward_loss(...)` is the equivalent unpaired primitive.

The accepted v5 record must have `schema_version="gold40_automask_v5"`, `pair_id`, matching `macro_pair_id`, `source_scene_id`, `intervention_type`, `source`, `target`, both `paired_branches`, and only the official `automatic_difference_mask` field. It must also declare `training_readiness={"ready_for_training": true, "mode": "automatic_difference_mask"}` and `mask_provenance={"kind": "automatic_rgb_difference_mask", "semantic_annotation": false, "state_specific": false}`. Human review fields and legacy candidate-mask fields are rejected.

The objective validates the automatic mask path/SHA256, native 1024×1024 `L` encoding, and nonempty pixels. It validates each `paired_branches` target path/SHA256 against its own real `source` or `target` endpoint before training. The mask is adaptive-average-pooled to 64×64; `outside = 1 - edit`, which is the sole unique token partition. Both branches reuse that same spatial partition, but the trainer must pack each branch from its own audited `spec["branches"][branch_id]["target_image"]` so each loss uses its real target flow.

Loss is `0.50*edit_mse + 0.50*outside_mse + ce`. The module rejects empty partitions and audits exact reconstruction of native target-flow MSE from the spatial partition.

The automatic difference mask is **only an automatic spatial loss weight**. It is **not** a human semantic ROI, does not claim semantic correctness, and must not be presented as manual review evidence.

## Build the v5 manifest and grouped splits

The adapter reads the existing 40-pair v4 candidate manifest and its locked automatic
RGB-difference masks. It writes a separate v5 manifest without changing v4 data or
writing any artificial approval fields:

```bash
cd /group/40016/niightli/BAGEL
python experiments/counterfactual_cot/gold40_automask_v5_data/build_gold40_automask_v5_manifest.py

python experiments/counterfactual_cot/gold40_automask_v5_data/build_gold40_automask_v5_splits.py \
  --smoke-train-pairs 8
```

Outputs:

```text
artifacts/gold40_source_target_automask_v5.jsonl
artifacts/gold40_automask_v5_audit.json
artifacts/splits/train.jsonl       # 24 pairs / 6 source scenes
artifacts/splits/val.jsonl         # 8 pairs / 2 source scenes
artifacts/splits/test.jsonl        # 8 pairs / 2 source scenes
artifacts/splits/smoke_train_8_pairs.jsonl
artifacts/splits/split_audit.json
```

All split records remain whole-source-scene groups. A 6-pair smoke request is rejected
because Gold40 currently has four pairs per source scene.
