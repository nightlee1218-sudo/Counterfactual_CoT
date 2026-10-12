from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set

from experiments.counterfactual_cot.data.schema import SceneGraph, load_jsonl


def atoms(scene: Dict[str, Any]) -> Set[str]:
    graph = SceneGraph.from_dict(scene)
    output: Set[str] = set()
    for entity in graph.entities:
        output.add(f"entity:{entity.id}:category:{entity.category}")
        if entity.count is not None:
            output.add(f"entity:{entity.id}:count:{entity.count}")
        if entity.location:
            output.add(f"entity:{entity.id}:location:{entity.location}")
        for key, value in entity.attributes.items():
            output.add(f"entity:{entity.id}:attr:{key}:{value}")
    for key, value in graph.background.items():
        output.add(f"background:{key}:{value}")
    return output


def score_record(pred: Dict[str, Any], gold_by_pair: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    pair_id = pred["pair_id"]
    gold_scene = gold_by_pair[pair_id]["target"]["cot"]
    pred_scene = pred.get("predicted_cot")
    valid_json = pred_scene is not None
    if not valid_json:
        return {"pair_id": pair_id, "valid_json": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    gold_atoms = atoms(gold_scene)
    pred_atoms = atoms(pred_scene)
    tp = len(gold_atoms & pred_atoms)
    precision = tp / max(1, len(pred_atoms))
    recall = tp / max(1, len(gold_atoms))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {"pair_id": pair_id, "valid_json": 1, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predicted structured CoT planning quality.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    gold_by_pair = {item["pair_id"]: item for item in load_jsonl(args.manifest)}
    rows = [score_record(item, gold_by_pair) for item in load_jsonl(args.predictions)]
    summary = {
        "num_records": len(rows),
        "valid_json_rate": sum(row["valid_json"] for row in rows) / max(1, len(rows)),
        "entity_atom_f1": sum(row["f1"] for row in rows) / max(1, len(rows)),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
