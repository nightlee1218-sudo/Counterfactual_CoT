#!/usr/bin/env python3
"""E1 general T2I benchmark: build prompt subsets from official T2I-CompBench data.

Subsets (official prompts, sampled with seed 1234):
  - attribute_binding: color + shape + texture, 50 each = 150
  - spatial_relations: spatial, 150
  - counting: numeracy, 150

Rationale: BAGEL-7B-MoT is weakest on T2I-CompBench spatial and attribute-style
binding; numeracy covers counting. GenAI-Bench / DPG official data were not
retrievable in this environment (parquet scan limit / no prompt-only mirror),
noted in README.md.

Usage:
  python3 build_prompts.py [--out prompts] [--per-subset 150]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "bench_raw"
SEED = 1234
PER_SUBSET = 150


def load_lines(name: str) -> list[str]:
    lines = [ln.strip() for ln in (RAW / f"{name}.txt").read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "prompts")
    parser.add_argument("--per-subset", type=int, default=PER_SUBSET)
    args = parser.parse_args()

    rng = random.Random(SEED)
    args.out.mkdir(parents=True, exist_ok=True)

    plan = {
        "attribute_binding": {"color": args.per_subset // 3,
                              "shape": args.per_subset // 3,
                              "texture": args.per_subset - 2 * (args.per_subset // 3)},
        "spatial_relations": {"spatial": args.per_subset},
        "counting": {"numeracy": args.per_subset},
    }

    provenance = {"seed": SEED, "source": "T2I-CompBench (Karine-Huang/T2I-CompBench, examples/dataset/*.txt)",
                  "subsets": {}}
    for subset, cats in plan.items():
        rows = []
        for cat, n in cats.items():
            lines = load_lines(cat)
            sampled = rng.sample(lines, n)
            for prompt in sampled:
                rows.append({
                    "item_id": f"{subset}__{cat}__{len(rows):04d}",
                    "subset": subset,
                    "source_category": cat,
                    "prompt": prompt,
                })
        out_path = args.out / f"{subset}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        provenance["subsets"][subset] = {"count": len(rows), "categories": {c: n for c, n in cats.items()}}
        print(f"{subset}: {len(rows)} prompts -> {out_path}")

    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
