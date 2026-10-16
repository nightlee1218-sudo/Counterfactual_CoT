"""pilot_report: merge per-engine stage reports into one pilot_report.json.

Reads, per engine dir (eng_<name>/):
    ledger.jsonl (composite), integrity.json, verify.json, audit_report.json
and the shared plans.jsonl; emits a cross-engine comparison:
    per-engine pass/regen/needs_fix/gate stats + changed_px distribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

if __package__ in (None, ""):
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parent))
from common import load_jsonl, write_json_atomic  # noqa: E402


def _engine_stats(engine_dir: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"engine_dir": str(engine_dir)}

    ledger = engine_dir / "ledger.jsonl"
    if ledger.exists():
        rows = load_jsonl(ledger)
        ok = [r for r in rows if r.get("status") == "ok"]
        changed = [r.get("changed_px", 0) for r in ok]
        stats["composite"] = {
            "n_planned": len(rows),
            "n_ok": len(ok),
            "n_failed": sum(1 for r in rows if r.get("status") == "failed"),
            "changed_px": {"mean": float(np.mean(changed)) if changed else None,
                           "min": int(np.min(changed)) if changed else None,
                           "max": int(np.max(changed)) if changed else None},
        }

    for name, fname, keys in (
        ("integrity", "integrity.json",
         ["n_pairs", "n_ok", "n_needs_fix", "n_error", "overflow_pairs"]),
        ("verify", "verify.json",
         ["n_pairs", "n_pass", "n_regen", "n_error", "score_mean", "item_pass_rate"]),
        ("audit", "audit_report.json",
         ["n_records", "n_pass", "n_fail_gate", "n_error"]),
    ):
        path = engine_dir / fname
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        src = payload.get("summary", payload)
        stats[name] = {k: src.get(k) for k in keys if k in src}
        if name == "audit":
            stats["audit"]["all_pass"] = bool(payload.get("all_pass"))
    return stats


def build_report(pilot_dir: Path, engines: List[str]) -> Dict[str, Any]:
    plans_path = pilot_dir / "plans.jsonl"
    n_plans = len(load_jsonl(plans_path)) if plans_path.exists() else 0
    per_engine = {}
    for engine in engines:
        engine_dir = pilot_dir / f"eng_{engine}"
        if engine_dir.exists():
            per_engine[engine] = _engine_stats(engine_dir)

    cross = {}
    eng_a, eng_b = (engines + ["", ""])[:2]
    if eng_a and eng_b and eng_a in per_engine and eng_b in per_engine:
        # changed-px overlap needs per-pair ledgers; compare summary stats
        a, b = per_engine[eng_a], per_engine[eng_b]
        for stage in ("composite", "integrity", "verify", "audit"):
            if stage in a and stage in b:
                cross[stage] = {eng_a: a[stage], eng_b: b[stage]}

    return {
        "schema_version": "counterfactual_data_v2_pilot_report/1.0",
        "pilot_dir": str(pilot_dir),
        "n_plans": n_plans,
        "engines": engines,
        "per_engine": per_engine,
        "cross_engine": cross,
        "all_engines_audit_pass": all(
            e.get("audit", {}).get("all_pass") is True for e in per_engine.values()
        ) if per_engine else False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-dir", required=True, type=Path)
    ap.add_argument("--engines", nargs="+", default=["nanobanana", "addit"])
    args = ap.parse_args()
    report = build_report(args.pilot_dir, args.engines)
    out = args.pilot_dir / "pilot_report.json"
    write_json_atomic(out, report)
    print(json.dumps({"per_engine_keys": {k: list(v) for k, v in report["per_engine"].items()},
                      "all_engines_audit_pass": report["all_engines_audit_pass"],
                      "out": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
