from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict

from .schema import load_jsonl, write_jsonl


class ImageAPIClient:
    def __init__(self, model: str, api_key_env: str, dry_run: bool = False, timeout_s: int = 120) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env, "")
        self.dry_run = dry_run
        self.timeout_s = timeout_s
        if not dry_run and not self.api_key:
            raise ValueError(f"Missing API key in environment variable {api_key_env}")

    def generate_source(self, pair: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        if self.dry_run:
            return {"request_id": request_id, "model": self.model, "status": "dry_run", "role": "source"}
        raise NotImplementedError("Wire provider-specific source image generation here.")

    def generate_target_edit(self, pair: Dict[str, Any], request_id: str) -> Dict[str, Any]:
        if self.dry_run:
            return {
                "request_id": request_id,
                "model": self.model,
                "status": "dry_run",
                "role": "target",
                "target_generation_mode": "edit",
            }
        raise NotImplementedError("Wire provider-specific target image editing here.")


def load_ledger(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(item["request_id"]): item for item in load_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate source/target images with resumable ledger.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--role", choices=["source", "target"], required=True)
    parser.add_argument("--model", default="provider/image-model/version")
    parser.add_argument("--api-key-env", default="CFCOT_IMAGE_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = ImageAPIClient(args.model, args.api_key_env, dry_run=args.dry_run)
    ledger = load_ledger(args.ledger)
    outputs = list(ledger.values())
    for pair in load_jsonl(args.manifest):
        request_id = f"image_{args.role}_{pair['pair_id']}"
        if request_id in ledger:
            continue
        result = client.generate_source(pair, request_id) if args.role == "source" else client.generate_target_edit(pair, request_id)
        result["created_at_unix"] = time.time()
        outputs.append(result)
        write_jsonl(args.ledger, outputs)
    print(f"Ledger contains {len(outputs)} image API records at {args.ledger}")


if __name__ == "__main__":
    main()
