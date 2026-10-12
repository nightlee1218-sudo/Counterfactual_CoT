from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

from .image_providers import build_provider
from .schema import canonical_json, load_jsonl, write_jsonl


def source_conditioning_text(record: Dict[str, Any]) -> str:
    return (
        "[MODE=SOURCE_ORACLE_COT]\n"
        "Generate one clean synthetic image that strictly follows both the scene description and the fixed scene plan.\n"
        f"Scene description: {record['prompt']}\n"
        f"Fixed scene plan: {record['cot_text']}\n"
        "Use a simple composition, no text, no watermark, and make exact object count and location clearly visible."
    )


def _hash_input(record: Dict[str, Any], model: str, parameters: Dict[str, Any]) -> str:
    payload = {
        "scene_hash": record.get("scene_hash"),
        "prompt": record.get("prompt"),
        "cot": record.get("cot"),
        "model": model,
        "parameters": parameters,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _load_successes(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(item.get("input_hash")): item for item in load_jsonl(path) if item.get("status") == "success"}


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve_bagel_root(value: Path | None) -> Path:
    if value is not None:
        root = value.resolve()
    else:
        deployed = Path(__file__).resolve().parents[3]
        sibling = Path(__file__).resolve().parents[2] / "BAGEL"
        root = deployed if (deployed / "modeling").exists() else sibling
    if not (root / "modeling").exists():
        raise ValueError(f"Not a BAGEL code root: {root}")
    return root


class BaseBagelSourceGenerator:
    def __init__(self, bagel_root: Path, model_path: Path) -> None:
        import torch
        from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch

        sys.path.insert(0, str(bagel_root))
        from data.data_utils import add_special_tokens
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
        from modeling.qwen2 import Qwen2Tokenizer

        model_path = model_path if model_path.is_absolute() else bagel_root / model_path
        llm_config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1
        vae_model, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))
        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)
        tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)
        device_map = infer_auto_device_map(
            model,
            max_memory={index: "80GiB" for index in range(torch.cuda.device_count())},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )
        shared = ["language_model.model.embed_tokens", "time_embedder", "latent_pos_embed", "vae2llm", "llm2vae", "connector", "vit_pos_embed"]
        first_device = device_map.get(shared[0], "cuda:0")
        for name in shared:
            device_map[name] = first_device if name in device_map else "cuda:0"
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(model_path / "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder=str(bagel_root / "offload_source_generation"),
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
        self.inferencer = InterleaveInferencer(model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids)

    def generate(self, text: str, output_path: Path, parameters: Dict[str, Any]) -> None:
        result = self.inferencer(
            text=text,
            think=False,
            image_shapes=(parameters["height"], parameters["width"]),
            num_timesteps=parameters["num_timesteps"],
            cfg_text_scale=parameters["cfg_text_scale"],
            timestep_shift=parameters["timestep_shift"],
        )
        image = result.get("image")
        if image is None:
            raise RuntimeError("Base BAGEL returned no image")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, format="PNG")


def _mock_generate(record: Dict[str, Any], output_path: Path, width: int, height: int) -> None:
    image = Image.new("RGB", (width, height), (238, 234, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((width // 4, height // 4, 3 * width // 4, 3 * height // 4), fill=(235, 190, 60))
    draw.text((20, 20), str(record.get("scene_id", "scene")), fill=(20, 20, 20))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate source images once per source scene using Base BAGEL or an image API with fixed external CoT.")
    parser.add_argument("--source-registry", required=True, type=Path)
    parser.add_argument("--output-registry", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--bagel-root", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=Path("models/BAGEL-7B-MoT"))
    parser.add_argument("--backend", choices=["base_bagel", "nano_banana_pro", "mock"], default="base_bagel")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default="CFCOT_IMAGE_API_KEY")
    parser.add_argument("--endpoint", default="http://v2.open.venus.oa.com/llmproxy/chat/completions")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-s", type=float, default=2.0)
    parser.add_argument("--rate-limit-qps", type=float, default=0.2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--cfg-text-scale", type=float, default=4.0)
    parser.add_argument("--timestep-shift", type=float, default=3.0)
    args = parser.parse_args()

    args.image_root.mkdir(parents=True, exist_ok=True)
    ledger_path = args.image_root / "ledgers" / "source_generation.jsonl"
    failures_path = args.image_root / "failures" / "source_generation.jsonl"
    successes = _load_successes(ledger_path)
    parameters = {
        "width": args.width,
        "height": args.height,
        "num_timesteps": args.num_timesteps,
        "cfg_text_scale": args.cfg_text_scale,
        "timestep_shift": args.timestep_shift,
        "think": False,
    }
    if args.backend == "mock":
        model_name = "mock-local"
    elif args.backend == "nano_banana_pro":
        model_name = args.model or "gemini-3-pro-image"
        parameters.update({
            "endpoint": "venus_llmproxy_chat_completions",
            "timeout_s": args.timeout_s,
            "max_retries": args.max_retries,
        })
    else:
        model_name = str(args.model_path)
    generator = None
    records = load_jsonl(args.source_registry)
    if args.max_scenes is not None:
        records = records[: args.max_scenes]
    outputs: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        input_hash = _hash_input(item, model_name, parameters)
        request_id = f"source_{item['scene_id']}_{input_hash[:12]}"
        relative_path = Path("source") / f"{item['scene_id']}_{item['scene_hash'][:12]}.png"
        output_path = args.image_root / relative_path
        if args.resume and input_hash in successes and output_path.exists():
            item["image"] = str(relative_path)
            item["source_generation"] = successes[input_hash].get("generation", {})
            item.setdefault("source_quality", {"accepted": False, "review_status": "pending"})
            outputs.append(item)
            continue
        started = time.monotonic()
        try:
            prompt = source_conditioning_text(item)
            response_meta: Dict[str, Any] = {}
            if args.backend == "mock":
                _mock_generate(item, output_path, args.width, args.height)
                response_meta = {"mime_type": "image/png", "mock": True}
            elif args.backend == "nano_banana_pro":
                if generator is None:
                    generator = build_provider(
                        "nano_banana_pro",
                        model=model_name,
                        api_key_env=args.api_key_env,
                        endpoint=args.endpoint,
                        timeout_s=args.timeout_s,
                        max_retries=args.max_retries,
                        retry_backoff_s=args.retry_backoff_s,
                        rate_limit_qps=args.rate_limit_qps,
                    )
                response_meta = generator.generate_source(prompt, output_path)
            else:
                if generator is None:
                    generator = BaseBagelSourceGenerator(_resolve_bagel_root(args.bagel_root), args.model_path)
                generator.generate(prompt, output_path, parameters)
                response_meta = {"mime_type": "image/png", "mock": False}
            generation = {
                "provider": args.backend,
                "model": model_name,
                "request_id": request_id,
                "input_hash": input_hash,
                "think": False,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "parameters": parameters,
                "response_id": response_meta.get("response_id"),
                "mime_type": response_meta.get("mime_type", "image/png"),
            }
            item["image"] = str(relative_path)
            item["source_generation"] = generation
            item["source_quality"] = {"accepted": False, "review_status": "pending", "review_notes": ""}
            _append_jsonl(ledger_path, {
                "request_id": request_id,
                "scene_id": item["scene_id"],
                "input_hash": input_hash,
                "status": "success",
                "output_image": str(relative_path),
                "latency_s": time.monotonic() - started,
                "generation": generation,
            })
        except Exception as exc:
            item.setdefault("source_quality", {})["accepted"] = False
            item["source_quality"]["review_status"] = "generation_failed"
            _append_jsonl(failures_path, {
                "request_id": request_id,
                "scene_id": item.get("scene_id"),
                "input_hash": input_hash,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "created_at_unix": time.time(),
            })
        outputs.append(item)
        write_jsonl(args.output_registry, outputs)
    write_jsonl(args.output_registry, outputs)
    review_path = args.image_root / "review" / "source_review.csv"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["scene_id", "template_family", "prompt", "cot_text", "image", "accepted", "review_notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in outputs:
            writer.writerow({
                "scene_id": item.get("scene_id", ""),
                "template_family": item.get("template_family", ""),
                "prompt": item.get("prompt", ""),
                "cot_text": item.get("cot_text", ""),
                "image": item.get("image", ""),
                "accepted": item.get("source_quality", {}).get("accepted", False),
                "review_notes": item.get("source_quality", {}).get("review_notes", ""),
            })
    print(f"Wrote {len(outputs)} source records to {args.output_registry}. Review CSV: {review_path}")


if __name__ == "__main__":
    main()
