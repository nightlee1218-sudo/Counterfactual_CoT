from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .image_providers import ProviderTransientResponseError, build_provider
from .image_quality import apply_local_check_to_quality, inspect_pair_images, min_changed_ratio_for
from .mask_utils import create_candidate_mask
from .review_export import export_review
from .schema import canonical_json, load_jsonl, write_jsonl


def target_edit_instruction(pair: Dict[str, Any]) -> str:
    """Render a visual-only edit instruction; never expose schema IDs or raw CoT atoms."""
    return (
        "[MODE=STRICT_LOCAL_EDIT_V2]\n"
        "Edit the provided source photograph locally and preserve its camera, framing, lighting, background, and every unrelated object.\n"
        "Apply only this visible target requirement:\n"
        f"{pair['target']['prompt']}\n\n"
        "Do not use or render internal scene-graph IDs, coordinate labels, or serialized plans. "
        "Do not regenerate the whole scene, add extra objects, duplicate objects, or reinterpret a physical relation loosely. "
        "Return only one edited square image."
    )


_BOOST_COUNT_ADD = (
    "\n\nMANDATORY RE-DO: the previous attempt added nothing, which is a failure. "
    "You must insert the missing new instance(s) into the scene: fully visible, the same size as the existing ones, clearly countable. "
    "A result identical or nearly identical to the source image is a failure."
)
_BOOST_COUNT_REMOVE = (
    "\n\nMANDATORY RE-DO: the previous attempt removed nothing, which is a failure. "
    "You must completely erase the extra instance(s) and let the bare surface show through where they stood: no outline, no shadow, no leftover. "
    "A result identical or nearly identical to the source image is a failure."
)
_BOOST_OCCLUSION = (
    "\n\nMANDATORY RE-DO: the previous attempt left the two objects untouched, which is a failure. "
    "You must physically pick up the occluding object: erase it from its spot, fill the area with the uninterrupted background, "
    "and redraw it either clearly in front of the other object with a visible overlap, or clearly far away from it with no overlap, exactly as instructed above. "
    "The relative arrangement of the two objects must visibly differ from the source image."
)
_BOOST_MOVE = (
    "\n\nMANDATORY RE-DO: the previous attempt changed almost nothing, which is a failure. "
    "You must physically relocate the object: erase it from its original spot first (let the surface behind it continue naturally) "
    "and then redraw the same object at the new position with the same size and orientation. "
    "The object's center must shift by at least one third of the frame; even a large object must be moved entirely. "
    "The object's screen position must clearly differ from the source image."
)
_BOOST_REPLACE = (
    "\n\nMANDATORY RE-DO: the previous attempt kept the old object, which is a failure. "
    "You must erase the original object completely and draw the new category in the same spot with a clearly different silhouette. "
    "An abstract blob or an unidentifiable shape is also a failure."
)
_BOOST_ATTRIBUTE = (
    "\n\nMANDATORY RE-DO: the previous attempt changed almost nothing, which is a failure. "
    "You must make the requested state change clearly visible on the object itself while keeping silhouette, count, and placement unchanged."
)

_COUNT_ADD_SUBTYPES = {"1_to_2", "1_to_3", "2_to_3"}
_COUNT_REMOVE_SUBTYPES = {"2_to_1", "3_to_1", "3_to_2"}


def _boost_instruction(instruction: str, pair: Dict[str, Any], attempt: int = 1) -> str:
    """Subtype-aware retry boost: the generic 'relocate' text was wrong for
    count/occlusion/replace failures.  Later attempts escalate."""
    subtype = str(pair.get("subtype", ""))
    intervention = pair.get("intervention", {}) if isinstance(pair.get("intervention"), dict) else {}
    old_value, new_value = intervention.get("old_value"), intervention.get("new_value")
    if subtype in _COUNT_ADD_SUBTYPES or (
        isinstance(old_value, int) and isinstance(new_value, int) and new_value > old_value
    ):
        boost = _BOOST_COUNT_ADD
    elif subtype in _COUNT_REMOVE_SUBTYPES or (
        isinstance(old_value, int) and isinstance(new_value, int) and new_value < old_value
    ):
        boost = _BOOST_COUNT_REMOVE
    elif subtype == "occlusion_state":
        boost = _BOOST_OCCLUSION
    elif subtype in ("absolute_horizontal", "absolute_vertical"):
        boost = _BOOST_MOVE
    elif subtype.endswith("_replace"):
        boost = _BOOST_REPLACE
    else:
        boost = _BOOST_ATTRIBUTE
    if attempt >= 2:
        boost += (
            f"\n\nFINAL WARNING: this is retry {attempt + 1} and every previous output was identical to the "
            "source image, which is a hard failure. The output image MUST be clearly different from the "
            "source image, or the whole request fails."
        )
    return instruction.rstrip() + boost


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _successes(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(item.get("input_hash")): item for item in load_jsonl(path) if item.get("status") == "success"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Edit target images from pre-generated source images and update the pair manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--provider", choices=["mock", "nano_banana_pro"], default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--fallback-model", default=None, help="Try once on this model when the primary returns an empty completion.")
    parser.add_argument("--api-key-env", default="CFCOT_IMAGE_API_KEY")
    parser.add_argument("--endpoint", default="http://v2.open.venus.oa.com/llmproxy/chat/completions")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=180)
    # Same fail-fast policy as source generation: do not stall a batch on one
    # pair; re-run with --resume to fill only the gaps.
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-s", type=float, default=3.0)
    parser.add_argument("--rate-limit-qps", type=float, default=0.2)
    parser.add_argument("--mask-threshold", type=int, default=24)
    parser.add_argument("--retry-no-change", action="store_true",
                        help="Retry with a stronger instruction when no visible change is detected (or the mask is empty).")
    parser.add_argument("--no-change-retries", type=int, default=1,
                        help="Number of additional edit attempts per pair when the output shows no visible change.")
    args = parser.parse_args()

    input_root = args.input_root or args.image_root
    model = args.model or ("gemini-3-pro-image" if args.provider == "nano_banana_pro" else "mock-local")
    kwargs: Dict[str, Any] = {"model": model}
    if args.provider == "nano_banana_pro":
        kwargs.update({
            "api_key_env": args.api_key_env,
            "endpoint": args.endpoint,
            "timeout_s": args.timeout_s,
            "max_retries": args.max_retries,
            "retry_backoff_s": args.retry_backoff_s,
            "rate_limit_qps": args.rate_limit_qps,
        })
    provider = build_provider(args.provider, **kwargs)
    ledger_path = args.image_root / "ledgers" / "image_api_ledger.jsonl"
    failures_path = args.image_root / "failures" / "image_api_failures.jsonl"
    successes = _successes(ledger_path)
    records = load_jsonl(args.manifest)
    if args.max_pairs is not None:
        records = records[: args.max_pairs]
    outputs: List[Dict[str, Any]] = []

    for pair in records:
        pair = dict(pair)
        pair_id = str(pair["pair_id"])
        source_path = _resolve(input_root, str(pair.get("source", {}).get("image", "")))
        instruction = target_edit_instruction(pair)
        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.exists() else "missing"

        def _input_hash(model: str) -> str:
            payload = {
                "pair_id": pair_id,
                "intervention": pair.get("intervention"),
                "target_cot": pair.get("target", {}).get("cot"),
                "instruction": instruction,
                "source_sha256": source_digest,
                "provider": provider.name,
                "model": model,
            }
            return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

        input_hash = _input_hash(provider.model)
        # Legacy runs recorded the success under the fallback-model hash whenever
        # the primary model returned an empty completion (and a provider-pollution
        # bug then kept that model for the rest of the chunk).  Accept either hash
        # so --resume does not regenerate pairs that only ever succeeded via the
        # fallback model.
        fallback_hash = _input_hash(args.fallback_model) if args.fallback_model else None
        request_id = f"target_{pair_id}_{input_hash[:12]}"
        target_relative = Path("target") / f"{pair_id}__target.png"
        mask_relative = Path("masks") / f"{pair_id}__mask.png"
        target_path = args.image_root / target_relative
        mask_path = args.image_root / mask_relative
        for legacy in (
            args.image_root / "target" / f"{pair_id}.png",
            args.image_root / "masks" / f"{pair_id}.png",
        ):
            if legacy.exists():
                legacy.unlink()
        started = time.monotonic()
        resume_hit = None
        if args.resume and target_path.exists():
            if input_hash in successes:
                resume_hit = successes[input_hash]
            elif fallback_hash is not None and fallback_hash in successes:
                resume_hit = successes[fallback_hash]
        target_needs_regeneration = resume_hit is None
        try:
            if not source_path.exists():
                raise FileNotFoundError(f"Pre-generated source image is missing: {source_path}")
            used_provider = provider
            used_hash = input_hash
            if target_needs_regeneration:
                if target_path.exists():
                    target_path.unlink()
                if mask_path.exists():
                    mask_path.unlink()
                try:
                    response_meta = provider.edit_target(source_path, instruction, target_path)
                except ProviderTransientResponseError:
                    if not args.fallback_model:
                        raise
                    fallback = build_provider(
                        args.provider,
                        model=args.fallback_model,
                        api_key_env=args.api_key_env,
                        endpoint=args.endpoint,
                        timeout_s=args.timeout_s,
                        max_retries=1,
                        retry_backoff_s=args.retry_backoff_s,
                        rate_limit_qps=args.rate_limit_qps,
                    ) if args.provider == "nano_banana_pro" else provider
                    response_meta = fallback.edit_target(source_path, instruction, target_path)
                    # Local only: never overwrite `provider`, or every later pair in
                    # this chunk would silently switch model and hash basis.
                    used_provider = fallback
                    used_hash = fallback_hash
                _append_jsonl(ledger_path, {
                    "request_id": request_id,
                    "pair_id": pair_id,
                    "role": "target",
                    "provider": used_provider.name,
                    "model": used_provider.model,
                    "input_hash": used_hash,
                    "status": "success",
                    "prompt": instruction,
                    "prompt_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                    "source_image": pair["source"]["image"],
                    "source_sha256": source_digest,
                    "output_image": str(target_relative),
                    "mime_type": response_meta.get("mime_type", "image/png"),
                    "response_id": response_meta.get("response_id"),
                    "latency_s": time.monotonic() - started,
                    "created_at_unix": time.time(),
                })
            else:
                used_provider = type("P", (), {"name": provider.name, "model": str(resume_hit.get("model", provider.model))})()
                used_hash = str(resume_hit.get("input_hash", input_hash))
            # Relocation edits (move / occlusion) are the weakest case for image
            # editors.  Retry loop: an identical-looking output ("no visible
            # change") or an empty mask (target == source) triggers a stronger
            # re-edit, up to --no-change-retries extra attempts.
            min_ratio = float(pair.get("quality_hint", {}).get("min_changed_ratio")) if pair.get("quality_hint") else min_changed_ratio_for(pair.get("subtype"))
            check = None
            for attempt in range(1 + (args.no_change_retries if args.retry_no_change else 0)):
                if attempt > 0:
                    boost_text = _boost_instruction(instruction, pair, attempt)
                    boost_started = time.monotonic()
                    boost_provider = provider
                    try:
                        boost_provider.edit_target(source_path, boost_text, target_path)
                    except ProviderTransientResponseError:
                        if not args.fallback_model:
                            raise
                        boost_fallback = build_provider(
                            args.provider,
                            model=args.fallback_model,
                            api_key_env=args.api_key_env,
                            endpoint=args.endpoint,
                            timeout_s=args.timeout_s,
                            max_retries=1,
                            retry_backoff_s=args.retry_backoff_s,
                            rate_limit_qps=args.rate_limit_qps,
                        ) if args.provider == "nano_banana_pro" else provider
                        boost_fallback.edit_target(source_path, boost_text, target_path)
                        boost_provider = boost_fallback
                    _append_jsonl(ledger_path, {
                        "request_id": f"{request_id}_boost{attempt}",
                        "pair_id": pair_id,
                        "role": "target",
                        "provider": boost_provider.name,
                        "model": boost_provider.model,
                        "input_hash": used_hash,
                        "status": "success",
                        "prompt": boost_text,
                        "prompt_sha256": hashlib.sha256(boost_text.encode("utf-8")).hexdigest(),
                        "source_image": pair["source"]["image"],
                        "source_sha256": source_digest,
                        "output_image": str(target_relative),
                        "mime_type": "image/png",
                        "response_id": None,
                        "latency_s": time.monotonic() - boost_started,
                        "created_at_unix": time.time(),
                    })
                if target_needs_regeneration or not mask_path.exists() or attempt > 0:
                    try:
                        create_candidate_mask(source_path, target_path, mask_path, threshold=args.mask_threshold)
                    except ValueError:
                        # Empty mask = target identical to source.  Retry with a
                        # stronger instruction instead of failing immediately.
                        if attempt + 1 < 1 + (args.no_change_retries if args.retry_no_change else 0):
                            continue
                        raise
                check = inspect_pair_images(source_path, target_path, mask_path, min_ratio)
                if "no_visible_change" not in check.get("failures", []):
                    break
            pair.setdefault("target", {})["image"] = str(target_relative)
            pair["target"]["mask"] = str(mask_relative)
            pair["generation"] = {
                "target_generation_mode": "edit",
                "provider": used_provider.name,
                "model": used_provider.model,
                "request_ids": [request_id],
                "target_request_id": request_id,
                "input_hash": used_hash,
                "created_at_unix": time.time(),
                "parameters": {
                    "endpoint": "venus_llmproxy_chat_completions" if args.provider == "nano_banana_pro" else "local_mock",
                    "timeout_s": args.timeout_s,
                    "max_retries": args.max_retries,
                },
            }
            pair["quality"] = apply_local_check_to_quality(pair.get("quality", {}), check)
        except Exception as exc:
            quality = dict(pair.get("quality", {}))
            quality["accepted"] = False
            quality["review_status"] = "generation_failed"
            quality["failure_reason"] = str(exc)
            pair["quality"] = quality
            _append_jsonl(failures_path, {
                "request_id": request_id,
                "pair_id": pair_id,
                "role": "target",
                "provider": provider.name,
                "model": provider.model,
                "input_hash": input_hash,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "created_at_unix": time.time(),
            })
        outputs.append(pair)
        write_jsonl(args.output_manifest, outputs)

    write_jsonl(args.output_manifest, outputs)
    export_review(outputs, args.image_root / "review", args.image_root)
    print(f"Wrote {len(outputs)} pairs to {args.output_manifest}. Automatic checks never set accepted=true.")


if __name__ == "__main__":
    main()
