"""Gold40 v3 realized-batch shape audit under single-pack packing.

This script enumerates the ENTIRE Gold40 v3 training loader under the
production single-pack packing config (expected_num_tokens=512,
max_num_tokens_per_sample=1536, max_num_tokens=1536), yielding one
DataLoader batch per Gold40 pair. It records the realized sequence_length,
sample_lens, packed tensor shapes, prompt / cot hashes, and the pair_id
in dataloader-yield order.

Nothing here loads a BAGEL model, VAE, or executes a GPU forward. This is
a pure loader-side audit.

Hard assertions:

  * len(sample_lens) == 1 for every yielded batch (single-pack invariant).
  * sequence_length <= 1536.
  * All 40 Gold40 pair_ids are enumerated exactly once.

Outputs:

  gold40_v3_singlepack_all40_realized_shapes.json
  gold40_v3_singlepack_longest_pair_selection.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path("/group/40016/niightli/BAGEL").resolve()
sys.path.insert(0, str(REPO_ROOT))

# Reuse the exact production loader + tokenizer path so the realized batch
# is bit-identical to what gold40_lora / gold40_v3_lora_2step_smoke sees.
from experiments.counterfactual_cot.training.gold40_lora import BASELINE_CONFIG  # type: ignore
from experiments.counterfactual_cot.training.gold40_overfit import _build_gold40_loader  # type: ignore
from experiments.counterfactual_cot.training.smoke_train import _load_tokenizer  # type: ignore

V3_MANIFEST = (
    REPO_ROOT
    / "experiments/counterfactual_cot/reruns/nanobanana_v4_absolute_region/manifests/"
    "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl"
)
BASE_LLM = "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT"
BASE_AE = "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ae.safetensors"
BASE_EMA = "/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT/ema.safetensors"
SEED = 1234
NUM_SAMPLES = 40
SINGLEPACK_CEILING = 1536


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_shape(t: Any) -> List[int]:
    try:
        return list(t.shape)
    except AttributeError:
        return []


def _attr(batch: Any, name: str, default: Any = None) -> Any:
    return getattr(batch, name, default)


def _extract_batch(batch: Any) -> Dict[str, Any]:
    sample_lens = list(_attr(batch, "sample_lens") or [])
    sequence_length = _attr(batch, "sequence_length")

    bdi = _attr(batch, "batch_data_indexes") or []
    pair_id = None
    dataset_index = None
    if isinstance(bdi, list) and bdi:
        entry = bdi[0]
        if isinstance(entry, dict):
            pair_id = entry.get("pair_id")
            data_indexes = entry.get("data_indexes")
            if isinstance(data_indexes, (list, tuple)) and data_indexes:
                dataset_index = int(data_indexes[0])

    # Loader-side raw shapes. padded_latent is produced later by VAE encode
    # (GPU); we deliberately do not run it, and instead record padded_images
    # which the collator already produced.
    padded_images = _attr(batch, "padded_images")
    pvls = _attr(batch, "patchified_vae_latent_shapes")

    shapes = {
        "packed_position_ids": _tensor_shape(_attr(batch, "packed_position_ids")),
        "packed_vae_token_indexes": _tensor_shape(_attr(batch, "packed_vae_token_indexes")),
        "padded_images": _tensor_shape(padded_images),
        "patchified_vae_latent_shapes_len": len(pvls) if pvls is not None else None,
        "patchified_vae_latent_shapes": [list(x) if not isinstance(x, list) else x for x in (pvls or [])],
        "packed_vit_tokens": _tensor_shape(_attr(batch, "packed_vit_tokens")),
        "packed_vit_token_indexes": _tensor_shape(_attr(batch, "packed_vit_token_indexes")),
        "packed_text_ids": _tensor_shape(_attr(batch, "packed_text_ids")),
        "packed_text_indexes": _tensor_shape(_attr(batch, "packed_text_indexes")),
        "packed_timesteps": _tensor_shape(_attr(batch, "packed_timesteps")),
        "mse_loss_indexes": _tensor_shape(_attr(batch, "mse_loss_indexes")),
        "packed_latent_position_ids": _tensor_shape(_attr(batch, "packed_latent_position_ids")),
        "vit_token_seqlens": _tensor_shape(_attr(batch, "vit_token_seqlens")),
    }

    text_ids_hash = None
    packed_text_ids = _attr(batch, "packed_text_ids")
    if packed_text_ids is not None:
        try:
            arr = packed_text_ids.detach().cpu().contiguous().view(-1).tolist()
            text_ids_hash = hashlib.sha256(repr(arr).encode("utf-8")).hexdigest()
        except Exception:
            text_ids_hash = None

    return {
        "pair_id": pair_id,
        "dataset_index": dataset_index,
        "sample_lens": sample_lens,
        "sequence_length": int(sequence_length) if sequence_length is not None else None,
        "shapes": shapes,
        "packed_text_ids_sha256": text_ids_hash,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=V3_MANIFEST)
    p.add_argument("--llm-path", type=str, default=BASE_LLM)
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "experiments/counterfactual_cot/outputs/gold40_v3_singlepack_realized_shapes")
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    p.add_argument("--seed", type=int, default=SEED)
    # Loader knobs (identical to BASELINE_CONFIG post-correction).
    p.add_argument("--expected-num-tokens", type=int, default=BASELINE_CONFIG["expected_num_tokens"])
    p.add_argument("--max-num-tokens-per-sample", type=int, default=BASELINE_CONFIG["max_num_tokens_per_sample"])
    p.add_argument("--max-num-tokens", type=int, default=BASELINE_CONFIG["max_num_tokens"])
    p.add_argument("--image-size", type=int, default=BASELINE_CONFIG["image_size"])
    p.add_argument("--vit-image-size", type=int, default=BASELINE_CONFIG["vit_image_size"])
    p.add_argument("--vit-min-image-size", type=int, default=BASELINE_CONFIG["vit_min_image_size"])
    p.add_argument("--variant", type=str, default=BASELINE_CONFIG["variant"])
    return p.parse_args()


def _run(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, special_tokens, _num_new_tokens = _load_tokenizer(args.llm_path)
    loader = _build_gold40_loader(args.manifest, tokenizer, special_tokens, args, seed_offset=0)

    # PackedDataset in this repo does weighted-random sampling with
    # replacement. To cover all 40 unique Gold40 pairs, we iterate until
    # every pair_id has been observed at least once (with a safety cap).
    per_pair_records: List[Dict[str, Any]] = []
    pair_id_first_seen: Dict[str, Dict[str, Any]] = {}
    all_yield_pair_ids: List[str] = []
    it = iter(loader)
    max_iters = args.num_samples * 20  # 800 for num_samples=40
    step = 0
    while len(pair_id_first_seen) < args.num_samples and step < max_iters:
        step += 1
        batch = next(it)
        rec = _extract_batch(batch)
        rec["yield_index"] = step

        # Hard assertions.
        if len(rec["sample_lens"]) != 1:
            raise RuntimeError(
                f"step{step}: sample_lens={rec['sample_lens']} not single-pack. Refusing to proceed."
            )
        if rec["sequence_length"] is None or int(rec["sequence_length"]) > SINGLEPACK_CEILING:
            raise RuntimeError(
                f"step{step}: sequence_length={rec['sequence_length']} exceeds {SINGLEPACK_CEILING}"
            )
        if rec["pair_id"] is None:
            raise RuntimeError(f"step{step}: batch missing pair_id")

        all_yield_pair_ids.append(rec["pair_id"])
        per_pair_records.append(rec)
        # First observation of each pair defines its recorded shapes.
        if rec["pair_id"] not in pair_id_first_seen:
            pair_id_first_seen[rec["pair_id"]] = rec

    unique_pair_ids = sorted(pair_id_first_seen.keys())
    coverage_ok = len(unique_pair_ids) == args.num_samples

    # Longest pair selection is done over the FIRST-SEEN record per unique pair
    # (each pair contributes one record; sequence_length is deterministic per
    # pair given the fixed seed and Gold40 v3 manifest).
    per_unique_records = list(pair_id_first_seen.values())

    # Longest pair selection by the declared priority.
    def _selection_key(r: Dict[str, Any]) -> Tuple[int, int, int, int]:
        return (
            -int(r["sequence_length"]),                      # 1. seq_len max
            -int(r["shapes"]["packed_text_indexes"][0] if r["shapes"]["packed_text_indexes"] else 0),   # 2. text max
            -int(r["shapes"]["packed_vit_tokens"][0] if r["shapes"]["packed_vit_tokens"] else 0),        # 3. vit max
            int(r["yield_index"]),                            # 4. earliest yield index
        )

    ranked = sorted(per_unique_records, key=_selection_key)
    longest = ranked[0]

    seq_lens = [int(r["sequence_length"]) for r in per_unique_records]
    seq_summary = {
        "min": min(seq_lens),
        "max": max(seq_lens),
        "mean": sum(seq_lens) / len(seq_lens),
        "count": len(seq_lens),
        "min_pair_id": per_unique_records[seq_lens.index(min(seq_lens))]["pair_id"],
        "max_pair_id": per_unique_records[seq_lens.index(max(seq_lens))]["pair_id"],
    }

    manifest_sha = _sha256_file(args.manifest)
    script_sha = _sha256_file(Path(__file__).resolve())

    all_shapes_report = {
        "created_at_iso": _now_iso(),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": script_sha,
        "loader_config": {
            "expected_num_tokens": int(args.expected_num_tokens),
            "max_num_tokens_per_sample": int(args.max_num_tokens_per_sample),
            "max_num_tokens": int(args.max_num_tokens),
            "image_size": int(args.image_size),
            "vit_image_size": int(args.vit_image_size),
            "vit_min_image_size": int(args.vit_min_image_size),
            "variant": args.variant,
            "seed": int(args.seed),
            "singlepack_ceiling": SINGLEPACK_CEILING,
        },
        "num_samples_expected": int(args.num_samples),
        "num_iterations_used_for_coverage": step,
        "num_pairs_yielded_total": len(per_pair_records),
        "num_unique_pairs_observed": len(unique_pair_ids),
        "coverage_ok": coverage_ok,
        "unique_pair_ids": unique_pair_ids,
        "sampling_order_first_100": all_yield_pair_ids[:100],
        "sequence_length_summary": seq_summary,
        "per_pair_records_unique": per_unique_records,
        "rank_by_priority": [
            {"rank": i + 1, "pair_id": r["pair_id"], "sequence_length": r["sequence_length"],
             "yield_index": r["yield_index"],
             "packed_text_indexes_len": r["shapes"]["packed_text_indexes"][0] if r["shapes"]["packed_text_indexes"] else None,
             "packed_vit_tokens_len": r["shapes"]["packed_vit_tokens"][0] if r["shapes"]["packed_vit_tokens"] else None}
            for i, r in enumerate(ranked)
        ],
    }

    all_shapes_path = out_dir / "gold40_v3_singlepack_all40_realized_shapes.json"
    with open(all_shapes_path, "w") as f:
        json.dump(all_shapes_report, f, ensure_ascii=False, indent=2, default=str)

    longest_report = {
        "created_at_iso": _now_iso(),
        "manifest": str(args.manifest),
        "manifest_sha256": manifest_sha,
        "selection_priority": [
            "1. maximum sequence_length",
            "2. tie-break: maximum packed_text_indexes length",
            "3. tie-break: maximum packed_vit_tokens length",
            "4. tie-break: earliest dataloader yield index",
        ],
        "longest_pair": longest,
        "selection_key_of_longest": {
            "seq_len": int(longest["sequence_length"]),
            "packed_text_indexes_len": longest["shapes"]["packed_text_indexes"][0] if longest["shapes"]["packed_text_indexes"] else None,
            "packed_vit_tokens_len": longest["shapes"]["packed_vit_tokens"][0] if longest["shapes"]["packed_vit_tokens"] else None,
            "yield_index": int(longest["yield_index"]),
        },
        "all_seq_lengths": seq_lens,
    }
    longest_path = out_dir / "gold40_v3_singlepack_longest_pair_selection.json"
    with open(longest_path, "w") as f:
        json.dump(longest_report, f, ensure_ascii=False, indent=2, default=str)

    print(json.dumps({
        "status": "passed",
        "num_iterations_used": step,
        "num_pairs_yielded_total": len(per_pair_records),
        "num_unique_pairs_observed": len(unique_pair_ids),
        "coverage_ok": coverage_ok,
        "longest_pair_id": longest["pair_id"],
        "longest_sequence_length": longest["sequence_length"],
        "longest_yield_index": longest["yield_index"],
        "seq_summary": seq_summary,
        "all_shapes_report_path": str(all_shapes_path),
        "longest_report_path": str(longest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_run(_parse_args()))
