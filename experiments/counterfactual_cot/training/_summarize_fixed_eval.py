import json
from pathlib import Path

p = Path('/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/gold40_fixed_deterministic_eval/fixed_eval_report.json')
r = json.load(open(p))

lines = []
lines.append(f"status = {r['status']}")
lines.append(f"step_labels = {r['step_labels']}")
replays = r['replays']
lines.append(f"num_replays = {len(replays)}")

r0 = {x['step_label']: x for x in replays[0]}
r1 = {x['step_label']: x for x in replays[1]}

lines.append("")
lines.append("=== mean_mse per replay per checkpoint ===")
lines.append(f"{'step_label':<12} {'replay0':>10} {'replay1':>10} {'|r0-r1|':>10}")
for lbl in r['step_labels']:
    a = r0[lbl]['mean_mse']
    b = r1[lbl]['mean_mse']
    lines.append(f"{lbl:<12} {a:>10.6f} {b:>10.6f} {abs(a-b):>10.2e}")

lines.append("")
lines.append("=== per intervention type (replay 0) ===")
types = sorted({t for x in replays[0] for t in x['mean_mse_by_intervention_type'].keys()})
lines.append(f"{'step_label':<12} " + ' '.join(f"{t[:14]:>15}" for t in types))
for lbl in r['step_labels']:
    row = r0[lbl]['mean_mse_by_intervention_type']
    lines.append(f"{lbl:<12} " + ' '.join(f"{row.get(t, float('nan')):>15.6f}" for t in types))

lines.append("")
lines.append("=== checkpoint metadata ===")
for lbl in r['step_labels']:
    x = r0[lbl]
    sha = x.get('trainable_checkpoint_sha256')
    gs = x.get('global_step')
    ckpt = x.get('trainable_checkpoint')
    sha_s = (sha[:16] if sha else 'none')
    ckpt_s = ckpt if ckpt else '(base only)'
    lines.append(f"{lbl:<12} global_step={gs} sha256={sha_s}... ckpt={ckpt_s}")

lines.append("")
lines.append("=== consistency (replay 0 vs replay 1) ===")
cons = r.get('consistency', {})
for lbl in r['step_labels']:
    if lbl in cons:
        c = cons[lbl]
        lines.append(f"{lbl:<12} max_abs_diff={c['max_abs_diff']:.3e}  mean_abs_diff={c['mean_abs_diff']:.3e}")

lines.append("")
lines.append("=== per-pair reproducibility (replay 0 vs 1) ===")
ts_ok = ns_ok = total = 0
for lbl, c in cons.items():
    for pid, d in c['per_pair'].items():
        total += 1
        if d['timesteps_match']:
            ts_ok += 1
        if d['noise_match']:
            ns_ok += 1
lines.append(f"timesteps_match: {ts_ok}/{total}    noise_match: {ns_ok}/{total}")

lines.append("")
lines.append("=== per-pair MSE table (replay 0) ===")
pair_ids = [p['pair_id'] for p in replays[0][0]['per_pair']]
header_cols = ' '.join(f"{lbl:>10}" for lbl in r['step_labels'])
lines.append(f"{'pair_id':<70} {header_cols}   intervention_type")
pair_type = {p['pair_id']: p['intervention_type'] for p in replays[0][0]['per_pair']}
per_label_pair_mse = {lbl: {p['pair_id']: p['mse'] for p in r0[lbl]['per_pair']} for lbl in r['step_labels']}
for pid in pair_ids:
    row = ' '.join(f"{per_label_pair_mse[lbl][pid]:>10.4f}" for lbl in r['step_labels'])
    lines.append(f"{pid:<70} {row}   {pair_type.get(pid, '')}")

lines.append("")
lines.append("=== review pairs (source/step_XXX/golden shown in review.html) ===")
for rp in r['review_pairs']:
    lines.append(f"  {rp['pair_id']}  [{rp['intervention_type']}]")
lines.append("")
lines.append(f"review_html = {r['review_html']}")

avg = []
for lbl in r['step_labels']:
    a = r0[lbl]['mean_mse']
    b = r1[lbl]['mean_mse']
    avg.append((lbl, (a + b) / 2.0))
avg.sort(key=lambda x: x[1])
lines.append("")
lines.append("=== ranking by mean_mse averaged over both replays (lower is better) ===")
for lbl, v in avg:
    lines.append(f"{lbl:<12} avg_mean_mse={v:.6f}")

out_path = Path('/group/40016/niightli/BAGEL/experiments/counterfactual_cot/outputs/gold40_fixed_deterministic_eval/summary.txt')
out_path.write_text("\n".join(lines), encoding='utf-8')
print(out_path)
print("LINES=", len(lines))
