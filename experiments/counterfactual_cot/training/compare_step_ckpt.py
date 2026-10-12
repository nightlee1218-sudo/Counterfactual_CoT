"""Compare the fresh lr=3e-5 500-step run's Step0 checkpoint tensor-by-tensor
against the old lr=3e-5 300-step Step0 checkpoint.

Also compares the ``loss_history`` from the two runs up to the earliest common
save step present so far in the new run's ``single_pair_overfit_report.json``
(created only at end-of-run) — since the report isn't present yet during
training, we don't touch it here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch


OLD_CKPT = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/"
    "single_pair_overfit_diag_ball_lr3e5/checkpoints/step_000_trainable.pt"
)
NEW_CKPT = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/"
    "single_pair_overfit_diag_ball_lr3e5_500steps/checkpoints/step_000_trainable.pt"
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=OLD_CKPT)
    parser.add_argument("--new", type=Path, default=NEW_CKPT)
    args = parser.parse_args()

    print(f"OLD path: {args.old}  size={args.old.stat().st_size}  sha256={sha256_file(args.old)}")
    print(f"NEW path: {args.new}  size={args.new.stat().st_size}  sha256={sha256_file(args.new)}")

    old = torch.load(args.old, map_location="cpu", weights_only=False)
    new = torch.load(args.new, map_location="cpu", weights_only=False)

    for key in ("format", "step", "global_step"):
        print(f"[{key}] old={old.get(key)} new={new.get(key)}  match={old.get(key)==new.get(key)}")

    old_ts = old["trainable_state_dict"]
    new_ts = new["trainable_state_dict"]
    common = sorted(set(old_ts.keys()) & set(new_ts.keys()))
    only_old = sorted(set(old_ts.keys()) - set(new_ts.keys()))
    only_new = sorted(set(new_ts.keys()) - set(old_ts.keys()))
    print(f"num_common_tensors = {len(common)}  only_old={only_old} only_new={only_new}")

    max_abs_all = 0.0
    all_identical = True
    per_tensor = []
    for name in common:
        a = old_ts[name].to(dtype=torch.float32)
        b = new_ts[name].to(dtype=torch.float32)
        if a.shape != b.shape:
            per_tensor.append({"name": name, "same_shape": False})
            all_identical = False
            continue
        diff = (a - b).abs()
        mx = float(diff.max().item())
        mean = float(diff.mean().item())
        exact = bool(torch.equal(a, b))
        per_tensor.append({"name": name, "shape": list(a.shape), "max_abs_diff": mx, "mean_abs_diff": mean, "exact_equal": exact})
        if not exact:
            all_identical = False
        max_abs_all = max(max_abs_all, mx)
    print(f"MAX_ABS_DIFF_ACROSS_ALL_14 = {max_abs_all}")
    print(f"ALL_TENSORS_EXACT_EQUAL   = {all_identical}")
    for row in per_tensor:
        print(json.dumps(row))

    # Config differences we DO expect: steps / save_steps / output_dir
    old_cfg = old.get("config") or {}
    new_cfg = new.get("config") or {}
    for key in sorted(set(old_cfg) | set(new_cfg)):
        if old_cfg.get(key) != new_cfg.get(key):
            print(f"CONFIG_DIFF: {key}: old={old_cfg.get(key)!r} new={new_cfg.get(key)!r}")


if __name__ == "__main__":
    main()
