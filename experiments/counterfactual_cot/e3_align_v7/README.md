# E3 v7: thrust-fixed token-region alignment (lambda 0.3 + temperature tau + strength reg + plateau guard)

v6 completed 2973 steps but its target-state align loss hugged the
log(2)=0.6931 plateau: 1683/2973 macro steps inside 0.69+-0.005 and the
margin improved only 0.001 -> 0.017 — the alignment objective had no
usable gradient at the margins the run actually sat at.  v7 fixes the
thrust with four parameterised changes; everything else is v6-verbatim.

## Files

| file | role |
|---|---|
| `train_v7.py` | trainer core (from `train_v6.py`, cp + targeted diffs: +262/-53) |
| `align_objective.py` | v6 spatial objective + live region-tensor stash for the strength regularizer |
| `selftest_v7.py` | CPU suite: 24 tests ALL PASS (incl. temperature/strength/plateau numerics) |
| `run_v7.sh` | launcher: `selftest\|preflight\|smoke\|train\|status`, plateau ALERT surfaced |
| `README.md` | this file |

`align_loss.py` is NOT copied: v7 imports it from `e3_align_v6.align_loss`
unchanged — the v7 temperature is exactly that module's `margin_scale`
(= 1/tau), so the battle-tested NO_REENTRANT AC hook code stays
single-sourced.  `exp/e3-align-v7` is branched from `exp/e3-align-v6`, so
both directories always coexist in this branch and on the server tree.

## The four changes (all switchable)

### 1. lambda 0.1 -> 0.3 (`--align-lambda`, default 0.3)

### 2. Temperature in L_align (`--align-tau`, default 0.1)

```
L_align = -log sigmoid((cos(h_tok, mu_in) - cos(h_tok, mu_out)) / tau)
```

The v6 margin scale was 1.0; v7 uses margin scale = 1/tau = 10.
Math: d(-log sigmoid(m/tau))/dm = -(1/tau)·sigmoid(-m/tau); at the small
v6-observed margins (m ~ 0.001-0.006) this is ~1/(2·tau), i.e. the plateau
gradient is amplified by ~1/tau = 10x.  Verified in selftest 21 at m=0.005:
tau=0.1 -> |grad| 4.875 vs untampered 0.4987 = 9.77x; **combined with
lambda 0.1->0.3 the effective align-gradient scale-up vs v6 is ~29x**
(the "tens of times" magnitude).  Note: the *pure* temperature amplification
limit at m->0 is exactly 1/tau = 10x, not 20x — reaching ~20x would need
tau~0.05 (`--align-tau 0.05`).  `--align-margin-scale` remains as an
explicit ablation override (default None = 1/tau).

### 3. Explicit strength regularizer (`--strength-reg on|off`, default on)

```
L_str = w_s * | edit_mse / outside_mse.detach() - strength_target |
        w_s = 0.05 (--strength-weight), target = 1.0 (--strength-target)
```

This REPLACES v6's proven-ineffective edit/outside loss-weight rebalancing
(0.4/0.6): with strength-reg on the region weights are forced back to the
v5 0.5/0.5 and `--edit-loss-weight != 0.5` is rejected by the CLI.  The
strength estimate is built from the per-branch edit/outside mask MSE
statistics training already computes every step (target_state branch only;
v6 observed edit/outside ~ 2, i.e. under-applying the edit).  The outside
term is the DETACHED anchor — the regularizer can never be reduced by
worsening the outside region (selftest 22/23 verify value, gradient
directions and the anchor property).  The live tensors come from the v7
objective's `branch_region_tensors` stash.  `--mask-debias` is kept
(default on) and `--overshoot-fix off` still reproduces v5 exactly.

### 4. Plateau watchdog (`--plateau-guard on|off`, default on)

If the target-state align loss of NON-SKIPPED branches never leaves
`0.69+-0.005` (`--plateau-center`/`--plateau-eps`) within
`--plateau-window` (1000) consecutive steps AND shows no downward trend
(mean second-half drop >= `--plateau-trend-eps` 1e-4 counts as leaving):
rank 0 writes an ALERT block into `--progress-file`
(`wb_e3_progress_v7.txt`), saves an emergency `latest.pt`, writes
final_report with status `plateau_alert_stopped` + a `plateau_alert`
payload, and training stops — no more 16h runs on a dead loss.
`run_v7.sh`'s watchdog stops overwriting the progress file once the alert
lands and surfaces it.  History is re-seeded on resume, and skipped
branches (align_skipped=1, which log align_loss 0.0) are excluded.

## What is v6-verbatim

Layers 13/6/16/14, gold40 data + strict seed audits, per-state matched
noise at 0.5 branch scale, full-layer LoRA r8/a16 fp32 masters, FSDP
FULL_SHARD world_size=2, mandatory 28-layer NO_REENTRANT activation
checkpointing (hook recompute parity — selftest 11), 16384-token packing,
general-T2I replay, warm start (`--init-trainable-from`, v5/v6/v7 keys all
load), debug AMP dtype mode, resume guard (now also guarding tau +
strength + plateau keys), atomic saves, final_report history.

New metrics in history rows: `target_state_strength_est`,
`target_state_strength_reg_loss`; final_report gains `strength_reg`,
`plateau_guard`, `plateau_alert` sections.

## Run (GPU instance, 2x A100)

```bash
bash run_v7.sh selftest   # CPU only (24 tests)
bash run_v7.sh preflight
bash run_v7.sh smoke      # 2 steps, throwaway dir
bash run_v7.sh train      # 2973 steps + watchdog; ALERT path included
bash run_v7.sh status
```

Overrides via `EXTRA_ARGS` (e.g. `--align-tau 0.05`, `--strength-reg off`,
`--init-trainable-from .../final991.../latest.pt`).  Default run trains
from base weights (same as v6); the checkpoint format is
`gold40_automask_v7_align_fsdp`.

## Selftest summary

24 tests ALL PASS, including the two requested numerical verifications:
- temperature: grad at m=0.005 amplified 9.77x (limit 1/tau=10x); with
  lambda 0.3 vs v6's 0.1: ~29x total;
- strength: L_str value from live objective tensors == w_s*|E/O-1| built
  from the logged statistics; under-edit gradient reduces edit_mse; the
  outside anchor receives NO gradient;
- plateau: band escape / trend escape / window / alert payload semantics.
