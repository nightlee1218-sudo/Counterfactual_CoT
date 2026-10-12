"""Summarize saved_checkpoints, gradient-clip / NaN / OOM stats from the
NEW run's single_pair_overfit_report.json.
"""
import json
import math
from pathlib import Path

NEW_DIR = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/"
    "single_pair_overfit_diag_ball_lr3e5_500steps"
)


def main() -> None:
    r = json.loads((NEW_DIR / "single_pair_overfit_report.json").read_text())
    print("=== saved_checkpoints ===")
    for c in r["saved_checkpoints"]:
        print(f"  step={c['global_step']:>3}  size={c['checkpoint_bytes']}  sha256={c['checkpoint_sha256']}")
    print()
    lh = r["loss_history"]
    n_nan = sum(1 for row in lh if math.isnan(row["mse"]) or math.isinf(row["mse"]))
    n_clip_needed = sum(1 for row in lh if row["grad_l2_total"] > 1.0)
    n_actually_clipped = sum(
        1 for row in lh
        if row.get("grad_norm_after_clip") is not None
        and abs(float(row["grad_norm_after_clip"]) - 1.0) < 1e-3
    )
    print(f"n_steps                       = {len(lh)}")
    print(f"n_nan_or_inf_mse              = {n_nan}")
    print(f"n_grad_l2_total > 1.0         = {n_clip_needed}")
    print(f"n_grad_norm_after_clip ~= 1.0 = {n_actually_clipped} (max_grad_norm=1.0)")
    print(f"max_grad_l2_total_over_run    = {max(row['grad_l2_total'] for row in lh):.4f}")
    print(f"min_grad_l2_total_over_run    = {min(row['grad_l2_total'] for row in lh):.4f}")
    print(f"max_grad_max_over_tensors     = {max(row['grad_max_over_tensors'] for row in lh):.4f}")
    print(f"status                        = {r.get('status')}")
    print(f"pair_id                       = {r.get('pair_id')}")
    print(f"effective_mse_tokens          = {r['loss_mask_summary'].get('effective_mse_tokens')}")
    print(f"generation_seed               = {r.get('args', {}).get('generation_seed')}")
    print(f"seed                          = {r.get('args', {}).get('seed')}")
    print(f"lr                            = {r.get('args', {}).get('lr')}")
    print(f"num_timesteps                 = {r.get('args', {}).get('num_timesteps')}")
    print(f"precision                     = {r.get('args', {}).get('precision')}")
    print(f"max_grad_norm                 = {r.get('args', {}).get('max_grad_norm')}")
    print(f"trainable_name_regex          = {r.get('args', {}).get('trainable_name_regex')}")
    print(f"trainable_parameters          = {r.get('trainable_report', {}).get('trainable_parameters')}")
    print(f"trainable_tensors             = {r.get('trainable_report', {}).get('trainable_tensors')}")

    # print step_1 exact numbers to confirm 0.23699234426021576
    print()
    print("=== step_1 line ===")
    row1 = next(row for row in lh if row["step"] == 1)
    print(json.dumps(row1, indent=2))


if __name__ == "__main__":
    main()
