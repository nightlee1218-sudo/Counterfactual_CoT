from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def load_metric(path: Path, metric: str, method_a: str, method_b: str) -> Dict[str, Tuple[float, float]]:
    by_scene = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            scene_id = str(row.get("scene_id") or "").strip()
            if not scene_id:
                raise ValueError(f"{path}:{row_number}: missing required scene_id for paired bootstrap")
            method = row["method"]
            if method in {method_a, method_b}:
                by_scene[scene_id][method] = float(row[metric])
    paired = {}
    for scene_id, values in by_scene.items():
        if method_a in values and method_b in values:
            paired[scene_id] = (values[method_a], values[method_b])
    return paired


def bootstrap(paired: Dict[str, Tuple[float, float]], num_samples: int, seed: int) -> Dict[str, float]:
    rng = random.Random(seed)
    scene_ids = list(paired)
    if not scene_ids:
        raise ValueError("No paired scenes available for bootstrap")
    observed = sum(paired[s][0] - paired[s][1] for s in scene_ids) / len(scene_ids)
    samples: List[float] = []
    for _ in range(num_samples):
        draw = [rng.choice(scene_ids) for _ in scene_ids]
        samples.append(sum(paired[s][0] - paired[s][1] for s in draw) / len(draw))
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return {"mean_diff": observed, "ci95_low": lo, "ci95_high": hi, "num_scenes": len(scene_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap over source-scene clusters.")
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--metric", default="grounded_success")
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = bootstrap(load_metric(args.scores, args.metric, args.method_a, args.method_b), args.num_samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
