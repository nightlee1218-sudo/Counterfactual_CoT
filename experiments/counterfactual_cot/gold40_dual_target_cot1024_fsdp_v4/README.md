# Gold40 dual-state CoT1024 FSDP v4

Only an approved reviewed split is accepted. Every record must retain the candidate payload, human-reviewed mutually disjoint `source_changed_roi` (source-only), `target_changed_roi` (target-only), and optional `shared_changed_roi` binary masks with SHA256. `shared_changed_roi` is required for a same-location state replacement. Records also require `review.approved: true`, `training_readiness.ready_for_training: true`, `source_scene_id`, and paired `source_state`/`target_state` branches. `dynamic_regions.*_ltrb` are compatibility metadata only; optimization pools the reviewed binary masks directly to the 64×64 target-flow grid.

Preflight does not allocate CUDA model memory:

```bash
cd /group/40016/niightli/BAGEL
python experiments/counterfactual_cot/gold40_dual_target_cot1024_fsdp_v4/train.py \
  --manifest /absolute/path/to/gold40_smoke_train8_reviewed.jsonl \
  --output-dir /tmp/gold40-preflight --preflight-only
```

Two A100 smoke run (does not run automatically):

```bash
MANIFEST=/absolute/path/to/gold40_smoke_train8_reviewed.jsonl \
bash experiments/counterfactual_cot/gold40_dual_target_cot1024_fsdp_v4/run_smoke_2xa100_40gb.sh
```

Gold40 currently has four pairs per source scene, so use a whole-scene 4- or 8-pair
smoke manifest; a six-pair smoke split is deliberately rejected to prevent scene leakage.

Use `run_train500_2xa100_40gb.sh` with the reviewed train24 manifest for formal training. Each macro-step statelessly samples one pair then backpropagates both target states with identical packed timestep and forward noise. The objective API is `paired_region_objective.build_paired_region_objective(record)`.
