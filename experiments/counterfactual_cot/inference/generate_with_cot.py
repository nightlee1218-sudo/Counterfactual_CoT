from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from experiments.counterfactual_cot.data.render_prompts import paired_prompt, standard_prompt
from experiments.counterfactual_cot.data.schema import load_jsonl


def build_prompt(pair: Dict[str, Any], mode: str) -> str:
    if mode == "direct":
        return pair["target"]["prompt"]
    if mode in {"oracle_cot", "self_cot"}:
        return standard_prompt(pair["target"]["prompt"], pair["target"]["cot_text"])
    if mode == "counterfactual_edit":
        return paired_prompt(pair, include_source_image=True)
    if mode == "conflict_cot":
        return standard_prompt(pair["source"]["prompt"], pair["target"]["cot_text"])
    if mode == "shuffled_cot":
        tokens = list(reversed(pair["target"]["cot_text"].split("; ")))
        return standard_prompt(pair["target"]["prompt"], "; ".join(tokens))
    if mode == "random_cot":
        return standard_prompt(pair["target"]["prompt"], "object=random; count=1; location=center")
    if mode == "synonym_cot":
        cot_text = pair["target"]["cot_text"].replace("object=", "entity=").replace("location=", "position=")
        return standard_prompt(pair["target"]["prompt"], cot_text)
    raise ValueError(f"Unsupported inference mode: {mode}")


def build_records(manifest: Path, output_dir: Path, checkpoint: str, modes: List[str], seeds: List[int]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for pair in load_jsonl(manifest):
        for mode in modes:
            for seed in seeds:
                sample_id = f"{pair['pair_id']}__{mode}__seed{seed}"
                records.append(
                    {
                        "sample_id": sample_id,
                        "pair_id": pair["pair_id"],
                        "condition": mode,
                        "checkpoint": checkpoint,
                        "prompt": build_prompt(pair, mode),
                        "predicted_cot": None,
                        "provided_cot": pair["target"].get("cot"),
                        "intervention": pair.get("intervention"),
                        "seed": seed,
                        "generation_parameters": {},
                        "source_image": pair["source"].get("image"),
                        "output_image": str(output_dir / f"{sample_id}.png"),
                        "latency_ms": 0,
                        "created_at_unix": time.time(),
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build counterfactual CoT inference jobs/records.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modes", nargs="+", default=["direct", "oracle_cot", "counterfactual_edit"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--records", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(args.manifest, args.output_dir, args.checkpoint, args.modes, args.seeds)
    args.records.parent.mkdir(parents=True, exist_ok=True)
    with args.records.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} inference records to {args.records}")


if __name__ == "__main__":
    main()
