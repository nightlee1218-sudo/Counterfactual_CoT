from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.counterfactual_cot.data.schema import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create predicted-CoT JSONL records; oracle fallback is explicit.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--oracle-fallback", action="store_true")
    args = parser.parse_args()
    rows = []
    for pair in load_jsonl(args.manifest):
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "prompt": pair["target"]["prompt"],
                "predicted_cot": pair["target"]["cot"] if args.oracle_fallback else None,
                "predicted_cot_text": pair["target"]["cot_text"] if args.oracle_fallback else "",
                "mode": "oracle_fallback" if args.oracle_fallback else "pending_model_prediction",
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} CoT prediction records to {args.output}")


if __name__ == "__main__":
    main()
