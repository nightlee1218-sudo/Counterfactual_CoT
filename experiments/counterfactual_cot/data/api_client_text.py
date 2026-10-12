from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .schema import load_jsonl, write_jsonl


class TextAPIClient:
    def __init__(self, model: str, api_key_env: str, dry_run: bool = False, timeout_s: int = 60) -> None:
        self.model = model
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env, "")
        self.dry_run = dry_run
        self.timeout_s = timeout_s
        if not dry_run and not self.api_key:
            raise ValueError(f"Missing API key in environment variable {api_key_env}")

    def parse_prompt_to_scene_graph(self, prompt: str, request_id: str) -> Dict[str, Any]:
        if self.dry_run:
            return {
                "request_id": request_id,
                "model": self.model,
                "raw_response": None,
                "parsed": None,
                "status": "dry_run",
            }
        raise NotImplementedError("Wire provider-specific text API here; do not call it during training.")


def load_ledger(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(item["request_id"]): item for item in load_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text-side scene graph metadata with resumable ledger.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--model", default="provider/model/version")
    parser.add_argument("--api-key-env", default="CFCOT_TEXT_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    client = TextAPIClient(args.model, args.api_key_env, dry_run=args.dry_run)
    ledger = load_ledger(args.ledger)
    outputs = list(ledger.values())
    for index, record in enumerate(load_jsonl(args.input)):
        request_id = f"text_{record.get('scene_id', index)}"
        if request_id in ledger:
            continue
        result = client.parse_prompt_to_scene_graph(str(record.get("prompt", "")), request_id)
        result["created_at_unix"] = time.time()
        outputs.append(result)
        write_jsonl(args.ledger, outputs)
    print(f"Ledger contains {len(outputs)} text API records at {args.ledger}")


if __name__ == "__main__":
    main()
