from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_optional(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate counterfactual CoT experiment result JSON files.")
    parser.add_argument("--planning", type=Path)
    parser.add_argument("--grounding", type=Path)
    parser.add_argument("--bootstrap", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "planning": load_optional(args.planning),
        "grounding": load_optional(args.grounding),
        "paired_bootstrap": load_optional(args.bootstrap),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote aggregate report to {args.output}")


if __name__ == "__main__":
    main()
