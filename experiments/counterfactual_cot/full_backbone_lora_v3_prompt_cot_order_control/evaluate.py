#!/usr/bin/env python3
"""Thin wrapper around the audited v3 evaluator for three local prompt conditions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from prompt_conditions import CONDITION_IDS, render_prompt_condition

BAGEL_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_EVALUATOR = BAGEL_ROOT / "experiments/counterfactual_cot/full_backbone_lora_region1024_fsdp_v3/evaluate.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("v3_upstream_evaluate_for_order_control", UPSTREAM_EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {UPSTREAM_EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for symbol in ("PROFILES", "render_prompt_profile", "parse_args", "main"):
        if not hasattr(module, symbol):
            raise RuntimeError(f"Upstream evaluator interface changed: missing {symbol}")
    module.PROFILES = CONDITION_IDS
    module.render_prompt_profile = render_prompt_condition
    module.main()


if __name__ == "__main__":
    main()
