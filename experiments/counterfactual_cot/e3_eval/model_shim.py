#!/usr/bin/env python3
"""E3 eval shim: inject a LoRA-checkpoint model alias into an existing script.

The E1 generation driver (e1_general_t2i/generate_t2i.py) and the E2 probe
(e2_token_probe/probe.py) both resolve ``--model`` through a module-level
``MODELS`` dict at parse/runtime (``choices=list(MODELS)`` built inside
``parse_args``/``main``).  This shim loads the target script as a module,
registers

    MODELS["v6"] = {"trainable_checkpoint": <ckpt>, "skip_trainable_reload": False}

and then calls its ``main()`` with the forwarded argv — adding v6 (E3
trainable_state_dict checkpoint, key-compatible with the v5 loaders) WITHOUT
modifying any e1/e2 file (those live on other branches: exp/e2-token-probe,
feature/e1-general-t2i).

Usage:
  python3 model_shim.py --script ../e2_token_probe/probe.py \
      --alias v6 --ckpt <...>/latest.pt -- \
      --model v6 --out-dir <...> --limit 1
  python3 model_shim.py --script ../e1_general_t2i/generate_t2i.py \
      --alias v6 --ckpt <...>/latest.pt -- \
      --model v6 --output-dir <...> --seed 1234

``--dry-run`` prints the injected entry + forwarded argv and exits (used by the
CPU selftest; the target module is still imported, so its import side effects
apply).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_script_as_module(script_path: Path):
    if not script_path.is_file():
        raise SystemExit(f"model_shim: target script not found: {script_path}")
    spec = importlib.util.spec_from_file_location(f"e3_shim_{script_path.stem}", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"model_shim: cannot load {script_path} as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True, help="target .py with a MODELS dict and main()")
    parser.add_argument("--alias", required=True, help="model alias to register, e.g. v6")
    parser.add_argument("--ckpt", type=Path, required=True, help="trainable_state_dict checkpoint (latest.pt)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="args after '--' are forwarded to the target")
    args = parser.parse_args()

    forwarded = list(args.rest)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]

    module = load_script_as_module(args.script.resolve())
    models = getattr(module, "MODELS", None)
    if not isinstance(models, dict):
        raise SystemExit("model_shim: target script has no module-level MODELS dict")
    if not hasattr(module, "main"):
        raise SystemExit("model_shim: target script has no main()")
    if not args.ckpt.is_file():
        raise SystemExit(f"model_shim: checkpoint not found: {args.ckpt}")

    models[args.alias] = {
        "trainable_checkpoint": args.ckpt.resolve(),
        "skip_trainable_reload": False,
        "shim_alias": args.alias,
    }
    sys.argv = [str(args.script.resolve())] + forwarded
    if args.dry_run:
        print(json.dumps({
            "event": "shim_dry_run",
            "script": str(args.script.resolve()),
            "alias": args.alias,
            "ckpt": str(args.ckpt.resolve()),
            "registered": sorted(models),
            "forwarded_argv": forwarded,
        }, ensure_ascii=False))
        return
    module.main()


if __name__ == "__main__":
    main()
