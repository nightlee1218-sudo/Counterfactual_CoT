"""Static equivalence audit for gold40_v3_overfit vs gold40_overfit.

This is a read-only static analysis: it produces a JSON report that proves
the new entrypoint delegates to the original ``gold40_overfit.main`` and
does NOT copy or re-implement any training-loop / dataloader / optimizer /
forward / loss / checkpointing / evaluation logic.

It is invoked once, offline, before the launch gate. It does not import
torch or load any BAGEL / VAE model.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path("/group/40016/niightli/BAGEL").resolve()
NEW_ENTRY = REPO_ROOT / "experiments/counterfactual_cot/training/gold40_v3_overfit.py"
ORIGINAL = REPO_ROOT / "experiments/counterfactual_cot/training/gold40_overfit.py"

# Symbols whose reimplementation in the new entry would violate the "no
# training-loop copy" contract. If ANY of these appear as function/class
# definitions in the new entry, the audit fails.
FORBIDDEN_DEFS: Set[str] = {
    "run",
    "parse_args",
    "_train_step",
    "_run_step",
    "_run_eval",
    "_prepare_batch",
    "_build_gold40_loader",
    "_build_model_for_gold40",
    "_load_tokenizer",
    "_autocast_for_model",
    "_register_gold40_dataset",
    "_diff_baseline_vs_lora",
    "_apply_trainable_and_audit",
    "_confirm_no_base_leak_in_optimizer",
    "_save_step_trainable",
    "_strict_reload_check",
    "_prepare_output_dir",
    "_transform_args",
    "collate_wrapper",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def collect_top_level_defs(source: str) -> Dict[str, List[str]]:
    """Return {"functions": [...], "classes": [...]} at module top level."""
    tree = ast.parse(source)
    functions: List[str] = []
    classes: List[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
    return {"functions": functions, "classes": classes}


def find_calls(source: str, target_name: str) -> int:
    """Count occurrences of ``target_name`` as a call name in Call nodes.
    Matches both ``target_name(...)`` and ``x.target_name(...)``.
    """
    tree = ast.parse(source)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == target_name:
                count += 1
            elif isinstance(func, ast.Attribute) and func.attr == target_name:
                count += 1
    return count


def find_imports_of_backend(source: str) -> List[str]:
    tree = ast.parse(source)
    hits: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            name_hits = [a.name for a in node.names if "gold40_overfit" in a.name]
            if "gold40_overfit" in mod or name_hits:
                aliases = ", ".join((a.asname or a.name) for a in node.names)
                hits.append(f"from {mod} import {aliases}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if "gold40_overfit" in a.name:
                    hits.append(f"import {a.name}" + (f" as {a.asname}" if a.asname else ""))
    return hits


def line_count(p: Path) -> int:
    return sum(1 for _ in open(p, "rb"))


def build_report() -> Dict[str, Any]:
    new_src = NEW_ENTRY.read_text(encoding="utf-8")
    orig_src = ORIGINAL.read_text(encoding="utf-8")

    new_defs = collect_top_level_defs(new_src)
    orig_defs = collect_top_level_defs(orig_src)

    forbidden_hit = [n for n in new_defs["functions"] + new_defs["classes"] if n in FORBIDDEN_DEFS]
    imports_of_backend = find_imports_of_backend(new_src)
    calls_to_backend_main = find_calls(new_src, "main")
    swap_hits = re.findall(r"gold40_overfit\.FIXED_GOLD40_MANIFEST\s*=", new_src)
    try_finally_hits = re.findall(r"\bfinally\s*:", new_src)
    v3_sha_hit = "5960e1eebb4d7a394a7b7cba2b2c17c686d184756b637110ef8b43397e172061" in new_src
    overfit_sha_hit = "dcb0f8a798b5b151ad712287a282d1cf0e1b5388c3c552e8de1ca2f5a19d5d3b" in new_src

    checks: Dict[str, Any] = {}
    checks["c01_new_entry_defines_no_forbidden_symbol"] = not forbidden_hit
    checks["c02_new_entry_imports_backend_module"] = any("gold40_overfit" in s for s in imports_of_backend)
    checks["c03_new_entry_calls_backend_main"] = calls_to_backend_main >= 1
    checks["c04_new_entry_swaps_FIXED_GOLD40_MANIFEST"] = len(swap_hits) >= 2  # swap + restore
    checks["c05_new_entry_has_try_finally"] = len(try_finally_hits) >= 1
    checks["c06_new_entry_hardcodes_v3_sha256"] = v3_sha_hit
    checks["c07_new_entry_hardcodes_backend_sha256"] = overfit_sha_hit
    checks["c08_backend_file_unchanged"] = sha256_file(ORIGINAL) == "dcb0f8a798b5b151ad712287a282d1cf0e1b5388c3c552e8de1ca2f5a19d5d3b"
    checks["c09_new_entry_is_short"] = line_count(NEW_ENTRY) <= 300  # sanity: not a training loop
    checks["c10_new_entry_line_count_less_than_backend"] = line_count(NEW_ENTRY) < line_count(ORIGINAL)

    status = "passed" if all(bool(v) for v in checks.values()) else "failed"

    return {
        "status": status,
        "created_at_iso": now_iso(),
        "new_entry": {
            "path": str(NEW_ENTRY),
            "sha256": sha256_file(NEW_ENTRY),
            "line_count": line_count(NEW_ENTRY),
            "top_level_functions": new_defs["functions"],
            "top_level_classes": new_defs["classes"],
            "forbidden_hits": forbidden_hit,
            "imports_of_backend": imports_of_backend,
            "num_calls_to_symbol_main": calls_to_backend_main,
            "num_swap_assignments_to_FIXED_GOLD40_MANIFEST": len(swap_hits),
            "num_try_finally_blocks": len(try_finally_hits),
            "v3_manifest_sha256_hardcoded": v3_sha_hit,
            "backend_sha256_hardcoded": overfit_sha_hit,
        },
        "backend": {
            "path": str(ORIGINAL),
            "sha256_expected": "dcb0f8a798b5b151ad712287a282d1cf0e1b5388c3c552e8de1ca2f5a19d5d3b",
            "sha256_observed": sha256_file(ORIGINAL),
            "line_count": line_count(ORIGINAL),
            "top_level_functions_count": len(orig_defs["functions"]),
            "top_level_classes_count": len(orig_defs["classes"]),
        },
        "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "contract_summary": (
            "gold40_v3_overfit delegates to gold40_overfit.main() and only "
            "monkey-patches the module attribute FIXED_GOLD40_MANIFEST to a "
            "SHA256-verified Gold40 v3 Path inside a try/finally. It defines "
            "no forbidden training-loop / dataloader / optimizer / evaluator "
            "symbol at module top level."
        ),
    }


def main() -> int:
    out_dir = REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_ab_400steps_launch_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    out = out_dir / "gold40_v3_overfit_static_equivalence_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "path": str(out),
        "sha256": sha256_file(out),
        "size_bytes": out.stat().st_size,
        "failed_checks": report["failed_checks"],
    }, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
