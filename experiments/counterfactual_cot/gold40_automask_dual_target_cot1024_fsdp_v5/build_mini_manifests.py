#!/usr/bin/env python3
"""Build the v2 mini-benchmark manifests: GEdit-36 / WISE-36 / GenEval-36.

Everything is deterministic: fixed RNG seed, explicit per-stratum quotas, and a
written audit so the exact sample set can be reproduced later.

Design notes
------------
* GEdit-36 : stratified over all 11 ``task_type`` values in the EN split of
  ``stepfun-ai/GEdit-Bench``. The 12 items used by the v1 mini-eval are pinned
  as ``gedit_00..gedit_11`` so the v1<->v2 calibration is exact.
* WISE-36  : stratified over the 3 super-groups (cultural / spatio-temporal /
  natural-science); within a group items are drawn round-robin across
  ``Subcategory`` to maximise subcategory coverage.
* GenEval-36 : stratified over the 6 ``tag`` values; the 12 v1 prompts are
  pinned as ``geneval_00..geneval_11``.

Outputs (under --out):
  gedit36.jsonl  wise36.jsonl  geneval36.jsonl  sampling_audit.json
  sources/gedit/<key>.png      (source images, reused from v1 when possible)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BAGEL_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = BAGEL_ROOT / "experiments/counterfactual_cot/outputs/mini_eval_v2_20260918/manifests"
V1_MANIFEST_DIR = BAGEL_ROOT / "experiments/counterfactual_cot/outputs/mini_eval_manifests"
V1_GEDIT_SOURCES = V1_MANIFEST_DIR / "gedit_sources"

SAMPLING_SEED = 20260918

# ---------------------------------------------------------------------------
# Quotas
# ---------------------------------------------------------------------------
# 11 task_type values, all covered. Proportional to the EN split (604 items)
# with a floor of 3 per type, then trimmed to sum exactly to 36.
GEDIT_QUOTA: dict[str, int] = {
    "text_change": 5,
    "ps_human": 4,
    "style_change": 3,
    "subject-add": 3,
    "subject-replace": 3,
    "subject-remove": 3,
    "background_change": 3,
    "color_alter": 3,
    "material_alter": 3,
    "motion_change": 3,
    "tone_transfer": 3,
}

WISE_GROUP_OF_CATEGORY = {
    "Cultural knowledge": "cultural_common_sense",
    "time": "spatio_temporal_reasoning",
    "Space": "spatio_temporal_reasoning",
    "Biology": "natural_science",
    "Chemistry": "natural_science",
    "Physical Knowledge": "natural_science",
}
WISE_QUOTA: dict[str, int] = {
    "cultural_common_sense": 14,
    "spatio_temporal_reasoning": 11,
    "natural_science": 11,
}

GENEVAL_QUOTA: dict[str, int] = {
    "position": 7,
    "color_attr": 7,
    "two_object": 6,
    "colors": 6,
    "single_object": 5,
    "counting": 5,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# GEdit
# ---------------------------------------------------------------------------
def load_v1_gedit_keys() -> list[dict[str, Any]]:
    path = V1_MANIFEST_DIR / "gedit_mini.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_gedit(out_dir: Path, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from datasets import load_dataset  # imported lazily: needs network on first call

    dataset = load_dataset("stepfun-ai/GEdit-Bench")["train"]
    items = [d for d in dataset if d["instruction_language"] == "en"]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_type[item["task_type"]].append(item)

    v1_rows = load_v1_gedit_keys()
    v1_keys = [r["meta"]["key"] for r in v1_rows]
    if len(set(GEDIT_QUOTA)) != 11:
        raise RuntimeError("GEDIT_QUOTA must cover 11 task types")
    if sum(GEDIT_QUOTA.values()) != 36:
        raise RuntimeError(f"GEDIT_QUOTA must sum to 36, got {sum(GEDIT_QUOTA.values())}")

    # 1) pin the v1 items
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    per_type_used: Counter = Counter()
    by_key = {i["key"]: i for i in items}
    for row in v1_rows:
        key = row["meta"]["key"]
        item = by_key.get(key)
        if item is None:
            raise RuntimeError(f"v1 gedit key {key} not found in EN split")
        task_type = item["task_type"]
        if per_type_used[task_type] >= GEDIT_QUOTA[task_type]:
            raise RuntimeError(f"v1 items overflow quota for {task_type}")
        selected.append(item)
        used.add(key)
        per_type_used[task_type] += 1

    # 2) fill remaining quota per type
    for task_type, quota in GEDIT_QUOTA.items():
        pool = [i for i in by_type.get(task_type, []) if i["key"] not in used]
        if len(pool) < quota - per_type_used[task_type]:
            raise RuntimeError(f"not enough EN items for {task_type}")
        rng.shuffle(pool)
        take = pool[: quota - per_type_used[task_type]]
        selected.extend(take)
        for item in take:
            used.add(item["key"])

    # v1 items keep their original order at the front; the rest follow
    front = [i for i in selected if i["key"] in set(v1_keys)]
    front.sort(key=lambda i: v1_keys.index(i["key"]))
    rest = [i for i in selected if i["key"] not in set(v1_keys)]
    rest.sort(key=lambda i: (i["task_type"], i["key"]))
    ordered = front + rest

    # 3) dump source images (reuse the v1 files byte-for-byte when available)
    src_dir = out_dir / "sources" / "gedit"
    src_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(ordered):
        key = item["key"]
        dest = src_dir / f"{key}.png"
        v1_file = V1_GEDIT_SOURCES / f"{key}.png"
        if v1_file.is_file() and not dest.exists():
            shutil.copyfile(v1_file, dest)
        if not dest.is_file():
            item["input_image"].convert("RGB").save(dest)
        rows.append(
            {
                "id": f"gedit_{index:02d}",
                "task": "edit",
                "prompt": item["instruction"],
                "image_path": str(dest.resolve()),
                "meta": {
                    "task_type": item["task_type"],
                    "key": key,
                    "instruction_language": item["instruction_language"],
                    "intersection_exist": bool(item["Intersection_exist"]),
                    "in_v1": key in set(v1_keys),
                },
            }
        )

    audit = {
        "source": "stepfun-ai/GEdit-Bench (train, instruction_language==en)",
        "n_total_en": len(items),
        "n_selected": len(rows),
        "quota": GEDIT_QUOTA,
        "en_distribution": dict(Counter(i["task_type"] for i in items)),
        "selected_distribution": dict(Counter(r["meta"]["task_type"] for r in rows)),
        "v1_pinned": len(front),
    }
    return rows, audit


# ---------------------------------------------------------------------------
# WISE
# ---------------------------------------------------------------------------
def build_wise(out_dir: Path, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads((BAGEL_ROOT / "eval/gen/wise/final_data.json").read_text(encoding="utf-8"))
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        group = WISE_GROUP_OF_CATEGORY.get(item["Category"])
        if group is None:
            raise RuntimeError(f"unmapped WISE category: {item['Category']}")
        by_group[group].append(item)

    selected: list[dict[str, Any]] = []
    for group, quota in WISE_QUOTA.items():
        # round-robin over subcategories to maximise coverage
        by_sub: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in by_group[group]:
            by_sub[item["Subcategory"]].append(item)
        subs = sorted(by_sub)
        for sub in subs:
            rng.shuffle(by_sub[sub])
        picked: list[dict[str, Any]] = []
        cursor = 0
        while len(picked) < quota:
            sub = subs[cursor % len(subs)]
            if by_sub[sub]:
                picked.append(by_sub[sub].pop())
            cursor += 1
            if cursor > 10000:
                raise RuntimeError("round-robin failed to fill quota")
        selected.extend(picked)

    selected.sort(key=lambda i: int(i["prompt_id"]))
    rows = [
        {
            "id": f"wise_{int(item['prompt_id']):04d}",
            "task": "t2i",
            "prompt": item["Prompt"],
            "image_path": None,
            "meta": {
                "group": WISE_GROUP_OF_CATEGORY[item["Category"]],
                "category": item["Category"],
                "subcategory": item["Subcategory"],
                "hint": item.get("Hint", ""),
                "explanation": item.get("Explanation", ""),
                "prompt_id": int(item["prompt_id"]),
            },
        }
        for item in selected
    ]
    audit = {
        "source": "eval/gen/wise/final_data.json",
        "n_total": len(data),
        "n_selected": len(rows),
        "quota": WISE_QUOTA,
        "selected_group_distribution": dict(Counter(r["meta"]["group"] for r in rows)),
        "selected_category_distribution": dict(Counter(r["meta"]["category"] for r in rows)),
        "subcategories_covered": len({r["meta"]["subcategory"] for r in rows}),
    }
    return rows, audit


# ---------------------------------------------------------------------------
# GenEval
# ---------------------------------------------------------------------------
def build_geneval(out_dir: Path, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = BAGEL_ROOT / "eval/gen/geneval/prompts/evaluation_metadata.jsonl"
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    v1_path = V1_MANIFEST_DIR / "geneval_mini.jsonl"
    v1_prompts: list[str] = []
    if v1_path.exists():
        v1_prompts = [
            json.loads(l)["prompt"] for l in v1_path.read_text(encoding="utf-8").splitlines() if l.strip()
        ]

    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_tag[item["tag"]].append(item)
    for tag in by_tag:
        rng.shuffle(by_tag[tag])

    used_prompts: set[str] = set()
    selected: list[dict[str, Any]] = []
    # 1) pin the v1 prompts
    by_prompt = {i["prompt"]: i for i in items}
    for prompt in v1_prompts:
        item = by_prompt.get(prompt)
        if item is not None:
            selected.append(item)
            used_prompts.add(prompt)
    # 2) fill by tag quota
    for tag, quota in GENEVAL_QUOTA.items():
        have = sum(1 for i in selected if i["tag"] == tag)
        pool = [i for i in by_tag.get(tag, []) if i["prompt"] not in used_prompts]
        for item in pool[: max(0, quota - have)]:
            selected.append(item)
            used_prompts.add(item["prompt"])

    front = [i for i in selected if i["prompt"] in set(v1_prompts)]
    front.sort(key=lambda i: v1_prompts.index(i["prompt"]))
    rest = [i for i in selected if i["prompt"] not in set(v1_prompts)]
    rest.sort(key=lambda i: (i["tag"], i["prompt"]))
    ordered = front + rest

    rows = [
        {
            "id": f"geneval_{index:02d}",
            "task": "t2i",
            "prompt": item["prompt"],
            "image_path": None,
            "meta": {
                "tag": item["tag"],
                "include": item["include"],
                "in_v1": item["prompt"] in set(v1_prompts),
            },
        }
        for index, item in enumerate(ordered)
    ]
    audit = {
        "source": "eval/gen/geneval/prompts/evaluation_metadata.jsonl",
        "n_total": len(items),
        "n_selected": len(rows),
        "quota": GENEVAL_QUOTA,
        "total_distribution": dict(Counter(i["tag"] for i in items)),
        "selected_distribution": dict(Counter(r["meta"]["tag"] for r in rows)),
        "v1_pinned": len(front),
    }
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SAMPLING_SEED)
    parser.add_argument(
        "--only", choices=("gedit", "wise", "geneval"), nargs="*", help="build a subset of manifests"
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    audit: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampling_seed": args.seed,
        "python": sys.version.split()[0],
    }
    targets = args.only or ["gedit", "wise", "geneval"]

    builders = {"gedit": build_gedit, "wise": build_wise, "geneval": build_geneval}
    for name in targets:
        rows, bench_audit = builders[name](args.out, rng)
        write_jsonl(args.out / f"{name}36.jsonl", rows)
        audit[name] = bench_audit
        audit[name]["manifest_sha256"] = sha256_file(args.out / f"{name}36.jsonl")
        print(json.dumps({"built": name, "n": len(rows)}, ensure_ascii=False), flush=True)

    (args.out / "sampling_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"event": "done", "out": str(args.out.resolve())}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
