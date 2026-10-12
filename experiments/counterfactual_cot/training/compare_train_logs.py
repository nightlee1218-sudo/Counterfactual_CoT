"""Compare per-step train_step events between old lr=3e-5 300-step run and
the new lr=3e-5 500-step run. Every step should match bit-for-bit for the
first 300 steps if RNG / data / optimizer are truly deterministic across the
two runs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


OLD_LOG = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/"
    "single_pair_overfit_diag_ball_lr3e5/logs/run.log"
)
NEW_LOG = Path(
    "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/"
    "single_pair_overfit_diag_ball_lr3e5_500steps/logs/run.log"
)

FIELDS = ("mse", "ce", "total", "grad_l2_total", "grad_max_over_tensors", "grad_norm_after_clip", "step_seed", "timesteps_hash")


def parse_train_steps(path: Path):
    rows = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") != "train_step":
            continue
        rows[int(r["step"])] = r
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-step", type=int, default=None,
                        help="Compare up to this step (inclusive). Default: min of the two log lengths.")
    parser.add_argument("--summary-every", type=int, default=25)
    args = parser.parse_args()

    old = parse_train_steps(OLD_LOG)
    new = parse_train_steps(NEW_LOG)
    print(f"old steps: min={min(old)} max={max(old)}  n={len(old)}")
    print(f"new steps: min={min(new)} max={max(new)}  n={len(new)}")

    max_step = args.max_step if args.max_step is not None else min(max(old), max(new))
    n_checked = 0
    n_exact = 0
    diffs = []
    for step in range(1, max_step + 1):
        if step not in old or step not in new:
            continue
        n_checked += 1
        ok = True
        for k in FIELDS:
            a = old[step].get(k)
            b = new[step].get(k)
            if a != b:
                ok = False
                diffs.append({"step": step, "field": k, "old": a, "new": b})
        if ok:
            n_exact += 1
        if step % args.summary_every == 0 or step in (1, 10, 25, 50, 100, 200, 300):
            marker = "OK" if ok else "MISMATCH"
            print(f"step {step:03d}: {marker}  mse_old={old[step]['mse']}  mse_new={new[step]['mse']}  seed_old={old[step]['step_seed']}  seed_new={new[step]['step_seed']}  tshash_old={old[step]['timesteps_hash'][:16]}...  tshash_new={new[step]['timesteps_hash'][:16]}...")
    print(f"\ncompared {n_checked} steps, exact_match_rows = {n_exact}")
    if diffs:
        print(f"\nFIRST 40 DIFFS:")
        for d in diffs[:40]:
            print(json.dumps(d))
    else:
        print("no per-step diffs found")


if __name__ == "__main__":
    main()
