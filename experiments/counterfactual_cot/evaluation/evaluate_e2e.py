from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate external R2I-Bench/GenEval results without running benchmark selection.")
    parser.add_argument("--r2i-result", type=Path)
    parser.add_argument("--geneval-result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {}
    if args.r2i_result and args.r2i_result.exists():
        report["r2i"] = json.loads(args.r2i_result.read_text(encoding="utf-8"))
    if args.geneval_result and args.geneval_result.exists():
        report["geneval"] = json.loads(args.geneval_result.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote E2E aggregate to {args.output}")


if __name__ == "__main__":
    main()
