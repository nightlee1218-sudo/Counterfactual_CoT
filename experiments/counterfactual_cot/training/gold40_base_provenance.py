"""Record provenance for the frozen Base BAGEL-7B-MoT under /dockerdata.

This is a *read-only* utility. It performs:

  1. ``os.stat`` + ``os.path.realpath`` on every requested path.
  2. Streaming SHA256 (buffered) on the three artefacts we must fingerprint.
  3. Cross-check the EMA SHA256 against ``EXPECTED_EMA_SHA256``.
  4. Atomically writes ``base_provenance.json`` **regardless of pass/fail**,
     so the operator always has a machine-readable record of what was
     observed. On failure, the report contains ``status="failed"``, an
     ``error`` object with the fail reason, and a ``finished_items`` list
     describing which files were already hashed before the failure.

Failure modes:

  * Any of the requested paths does not exist -> status=failed, missing item
    is recorded.
  * SHA256 read raises -> status=failed, error captured.
  * EMA SHA256 does not match ``EXPECTED_EMA_SHA256`` -> status=failed,
    both got/expected values are recorded (and both AE and EMA hashes are
    preserved so the operator can compare visually).

Exit codes:
  0 - status=passed.
  1 - status=failed (report is still written).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


REQUIRED_BASE_DIR = Path("/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT")
REQUIRED_AE = REQUIRED_BASE_DIR / "ae.safetensors"
REQUIRED_EMA = REQUIRED_BASE_DIR / "ema.safetensors"

# Known EMA SHA256 recorded by the training-side smoke run at
# outputs/lora_2step_smoke_lr3e5_L24-27_r8/cold_reload/orchestrator.log
# and outputs/step0_conditioning_diagnostics/diagnostic_manifest.jsonl.
EXPECTED_EMA_SHA256 = "0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3"


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _stat_record(requested: Path) -> Dict[str, Any]:
    if not requested.exists():
        raise FileNotFoundError(f"required path does not exist: {requested}")
    resolved = Path(os.path.realpath(str(requested)))
    st = requested.lstat()
    is_symlink = stat.S_ISLNK(st.st_mode)
    real_st = requested.stat()
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "is_symlink": bool(is_symlink),
        "size_bytes": int(real_st.st_size),
        "mtime_unix": float(real_st.st_mtime),
        "mtime_iso": datetime.fromtimestamp(real_st.st_mtime, tz=timezone.utc).isoformat(),
        "mode": oct(real_st.st_mode),
        "uid": int(real_st.st_uid),
        "gid": int(real_st.st_gid),
    }


def _list_base_dir(base_dir: Path) -> Dict[str, Any]:
    entries = []
    for child in sorted(base_dir.iterdir()):
        entries.append(
            {
                "name": child.name,
                "is_dir": child.is_dir(),
                "size_bytes": child.stat().st_size if child.is_file() else None,
                "is_symlink": child.is_symlink(),
            }
        )
    return {"num_entries": len(entries), "entries": entries}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/group/40016/niightli/BAGEL/experiments/counterfactual_cot/"
            "outputs/base_provenance/base_provenance.json"
        ),
        help="Where to write base_provenance.json. Must live under /group.",
    )
    parser.add_argument("--llm-path", type=Path, default=REQUIRED_BASE_DIR)
    parser.add_argument("--vae-path", type=Path, default=REQUIRED_AE)
    parser.add_argument("--ema-path", type=Path, default=REQUIRED_EMA)
    parser.add_argument("--expected-ema-sha256", default=EXPECTED_EMA_SHA256)
    parser.add_argument("--allow-non-dockerdata", action="store_true",
                        help="Skip the /dockerdata guardrail. Only for local debug.")
    return parser.parse_args()


def _finalize(payload: Dict[str, Any], output_path: Path, passed: bool) -> int:
    payload["status"] = "passed" if passed else "failed"
    _atomic_write_json(output_path, payload)
    print(json.dumps({
        "status": payload["status"],
        "output": str(output_path),
        "ae_sha256": (payload.get("vae_safetensors") or {}).get("sha256"),
        "ema_sha256": (payload.get("ema_safetensors") or {}).get("sha256"),
        "ema_matches_reference": (payload.get("ema_safetensors") or {}).get("sha256_matches_reference"),
        "error": payload.get("error"),
    }, indent=2))
    return 0 if passed else 1


def main() -> int:
    args = parse_args()

    payload: Dict[str, Any] = {
        "created_at_iso": datetime.now(tz=timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "expected_ema_sha256": args.expected_ema_sha256 or None,
        "requested_paths": {
            "llm_path": str(args.llm_path),
            "vae_path": str(args.vae_path),
            "ema_path": str(args.ema_path),
        },
        "allow_non_dockerdata": bool(args.allow_non_dockerdata),
        "finished_items": [],
        "llm_dir": None,
        "vae_safetensors": None,
        "ema_safetensors": None,
        "error": None,
    }

    output_abs = args.output.resolve()
    if str(output_abs).startswith("/dockerdata"):
        payload["error"] = {
            "kind": "output_under_dockerdata",
            "message": f"--output must NOT live under /dockerdata: {output_abs}",
        }
        # Still write the report to /tmp so it is not silently lost.
        try:
            _atomic_write_json(Path("/tmp/base_provenance_fallback.json"), payload)
        except Exception:  # noqa: BLE001
            pass
        return _finalize(payload, output_abs, passed=False)

    # /dockerdata guardrail (unless explicitly overridden).
    for label, path in (("llm_path", args.llm_path), ("vae_path", args.vae_path), ("ema_path", args.ema_path)):
        if not str(path).startswith("/dockerdata/") and not args.allow_non_dockerdata:
            payload["error"] = {
                "kind": "not_under_dockerdata",
                "message": f"{label} must live under /dockerdata (got: {path}). "
                           "Pass --allow-non-dockerdata only for local debug.",
                "offending_label": label,
                "offending_path": str(path),
            }
            return _finalize(payload, output_abs, passed=False)

    t0 = time.time()

    # LLM directory stat + listing.
    try:
        payload["llm_dir"] = {**_stat_record(args.llm_path), "listing": _list_base_dir(args.llm_path)}
        payload["finished_items"].append("llm_dir_stat")
    except Exception as exc:  # noqa: BLE001
        payload["error"] = {
            "kind": "llm_dir_stat_failed",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return _finalize(payload, output_abs, passed=False)

    # AE hash.
    try:
        ae_stat = _stat_record(args.vae_path)
        ae_sha = _sha256_file(args.vae_path)
        payload["vae_safetensors"] = {**ae_stat, "sha256": ae_sha}
        payload["finished_items"].append("vae_hashed")
        print(f"[base_provenance] ae.safetensors sha256 = {ae_sha}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        payload["error"] = {
            "kind": "vae_hash_failed",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return _finalize(payload, output_abs, passed=False)

    # EMA hash.
    try:
        print("[base_provenance] hashing ema.safetensors (27 GiB, will take minutes)", file=sys.stderr)
        ema_stat = _stat_record(args.ema_path)
        ema_sha = _sha256_file(args.ema_path)
        payload["ema_safetensors"] = {**ema_stat, "sha256": ema_sha}
        payload["finished_items"].append("ema_hashed")
        print(f"[base_provenance] ema.safetensors sha256 = {ema_sha}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        payload["error"] = {
            "kind": "ema_hash_failed",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        return _finalize(payload, output_abs, passed=False)

    # EMA reference-check.
    ema_matches_reference: Optional[bool] = None
    if args.expected_ema_sha256:
        ema_matches_reference = ema_sha == args.expected_ema_sha256
        payload["ema_safetensors"]["sha256_matches_reference"] = ema_matches_reference
        if not ema_matches_reference:
            payload["error"] = {
                "kind": "ema_sha_mismatch",
                "message": f"EMA SHA256 mismatch: got {ema_sha}, expected {args.expected_ema_sha256}",
                "expected": args.expected_ema_sha256,
                "got": ema_sha,
                "note": "Refusing to declare provenance passed. Do NOT start training.",
            }
            return _finalize(payload, output_abs, passed=False)
    else:
        payload["ema_safetensors"]["sha256_matches_reference"] = None

    payload["wall_seconds"] = time.time() - t0
    return _finalize(payload, output_abs, passed=True)


if __name__ == "__main__":
    sys.exit(main())
