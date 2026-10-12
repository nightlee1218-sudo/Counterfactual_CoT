"""Gold40 v3 overfit entrypoint (backend delegates to gold40_overfit.main()).

This module is the APPROVED "Option C" implementation.

Design contract (do not modify without explicit user approval):

  * ``gold40_overfit.py`` MUST stay byte-for-byte identical (SHA256 =
    dcb0f8a798b5b151ad712287a282d1cf0e1b5388c3c552e8de1ca2f5a19d5d3b).
  * This entrypoint MUST NOT copy or re-implement the training loop, the
    dataloader, the optimizer, the checkpoint IO, the evaluation logic, or
    the visualization logic. All of those come from
    ``gold40_overfit.main`` unchanged.
  * The ONLY behavioural difference between this entrypoint and running
    ``python -m experiments.counterfactual_cot.training.gold40_overfit``
    directly is: the ``FIXED_GOLD40_MANIFEST`` module-level constant is
    temporarily bound to the frozen Gold40 v3 manifest path during
    ``main()`` execution, and restored in a ``finally`` block.
  * The v3 manifest path is hard-coded and its SHA256 is verified before
    ANY model or dataset construction. A mismatch of either the resolved
    path or the SHA256 aborts before ``gold40_overfit.main()`` is called.
  * Only the ``--manifest`` command-line default is affected (indirectly
    through the constant swap); users may still pass ``--manifest`` on
    the command line and it will be validated against the swapped v3
    constant by the original ``_validate_fixed_manifest``.

Nothing else is patched. No monkey-patching of loader, forward, loss,
scheduler, evaluator, or any core BAGEL / qwen2_navit / siglip_navit
code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path("/group/40016/niightli/BAGEL").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# NOTE: we import the *module*, not just symbols, because we need to swap
# ``gold40_overfit.FIXED_GOLD40_MANIFEST`` in place.
from experiments.counterfactual_cot.training import gold40_overfit as _gold40_overfit  # noqa: E402


# ------------------------------- Constants -------------------------------

# The single approved Gold40 v3 manifest. Locked here; do NOT parametrize.
GOLD40_V3_MANIFEST: Path = (
    REPO_ROOT
    / "experiments"
    / "counterfactual_cot"
    / "reruns"
    / "nanobanana_v4_absolute_region"
    / "manifests"
    / "intervention_pairs_gold40_human_verified_final_normalized_v3_schema_consistent.jsonl"
)

# The pre-registered SHA256 of the Gold40 v3 manifest. Any mismatch aborts.
GOLD40_V3_MANIFEST_SHA256_EXPECTED: str = (
    "5960e1eebb4d7a394a7b7cba2b2c17c686d184756b637110ef8b43397e172061"
)

# The pre-registered SHA256 of the ORIGINAL gold40_overfit.py at the time this
# entrypoint was approved. If the on-disk file has drifted, refuse to run so
# that the delegation contract stays intact.
GOLD40_OVERFIT_SHA256_EXPECTED: str = (
    "dcb0f8a798b5b151ad712287a282d1cf0e1b5388c3c552e8de1ca2f5a19d5d3b"
)


# ------------------------------- Helpers ---------------------------------


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


def _fatal_stderr(msg: str, code: int = 2) -> "int":  # noqa: F821
    print(f"[gold40_v3_overfit] FATAL: {msg}", file=sys.stderr)
    return code


def _verify_v3_manifest_or_abort() -> None:
    """Pre-flight: verify the frozen v3 manifest path and SHA256 BEFORE any
    model / VAE / tokenizer / dataloader is constructed. Raises if invalid.
    """
    manifest = GOLD40_V3_MANIFEST
    if not manifest.exists():
        raise FileNotFoundError(
            f"Gold40 v3 manifest missing on disk: {manifest}"
        )
    resolved = manifest.resolve()
    sha = _sha256_file(resolved)
    if sha != GOLD40_V3_MANIFEST_SHA256_EXPECTED:
        raise RuntimeError(
            "Gold40 v3 manifest SHA256 mismatch. "
            f"Expected {GOLD40_V3_MANIFEST_SHA256_EXPECTED} at {resolved}, "
            f"got {sha}. Refusing to launch training."
        )


def _verify_original_backend_or_abort() -> str:
    """Pre-flight: verify the on-disk ``gold40_overfit.py`` still has the
    approved SHA256. Returns the observed SHA on success; raises otherwise.
    """
    module_file = Path(_gold40_overfit.__file__).resolve()
    observed = _sha256_file(module_file)
    if observed != GOLD40_OVERFIT_SHA256_EXPECTED:
        raise RuntimeError(
            "Backend gold40_overfit.py SHA256 mismatch. "
            f"Expected {GOLD40_OVERFIT_SHA256_EXPECTED} at {module_file}, "
            f"got {observed}. This entrypoint is a strict delegation over the "
            "original file; refusing to launch training when the backend has "
            "drifted."
        )
    return observed


def _emit_provenance(
    output_dir: Optional[Path],
    original_fixed_manifest: Path,
    temporary_fixed_manifest: Path,
    original_backend_sha256: str,
) -> Optional[Dict[str, Any]]:
    """Best-effort provenance: write ``entrypoint_provenance.json`` under
    ``--output-dir`` if we can find it. This is diagnostic-only; failure to
    write does not abort delivery of the training run.
    """
    provenance = {
        "entrypoint": "gold40_v3_overfit",
        "implementation_backend": "gold40_overfit.main",
        "manifest_contract": "gold40_v3_path_and_sha256_locked",
        "original_fixed_manifest": str(original_fixed_manifest),
        "temporary_fixed_manifest": str(temporary_fixed_manifest),
        "gold40_v3_manifest_sha256_expected": GOLD40_V3_MANIFEST_SHA256_EXPECTED,
        "gold40_v3_manifest_sha256_observed": _sha256_file(temporary_fixed_manifest),
        "backend_gold40_overfit_sha256_expected": GOLD40_OVERFIT_SHA256_EXPECTED,
        "backend_gold40_overfit_sha256_observed": original_backend_sha256,
        "backend_gold40_overfit_path": str(Path(_gold40_overfit.__file__).resolve()),
        "created_at_iso": _now_iso(),
        "env": {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", ""),
        },
        "argv": list(sys.argv),
        "delegation_semantics": {
            "training_loop_source": "gold40_overfit.run (unchanged)",
            "training_loop_copied_or_reimplemented": False,
            "manifest_lock_before": "gold40_overfit.FIXED_GOLD40_MANIFEST (v1)",
            "manifest_lock_after_swap": "GOLD40_V3_MANIFEST (v3, sha256 verified)",
            "swap_mechanism": "monkey-patch gold40_overfit.FIXED_GOLD40_MANIFEST inside try/finally",
            "swap_restored_on_exit": True,
        },
    }
    if output_dir is None:
        return provenance
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "entrypoint_provenance.json", "w") as fh:
            json.dump(provenance, fh, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception as exc:  # pragma: no cover - diagnostic best effort
        print(
            f"[gold40_v3_overfit] warning: failed to write entrypoint_provenance.json: {exc}",
            file=sys.stderr,
        )
    return provenance


def _sniff_output_dir_from_argv() -> Optional[Path]:
    """Diagnostic-only: peek at argv for --output-dir so we can drop a
    provenance file next to the training artifacts. Does NOT change any
    argparse behaviour in gold40_overfit.parse_args.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, default=None)
    try:
        known, _ = parser.parse_known_args(sys.argv[1:])
    except SystemExit:
        return None
    return known.output_dir


# ------------------------------- Entrypoint -------------------------------


def main() -> None:
    """Strict-delegation entrypoint. See module docstring for the contract."""
    # 1) Pre-flight: verify backend file and v3 manifest BEFORE anything else.
    original_backend_sha256 = _verify_original_backend_or_abort()
    _verify_v3_manifest_or_abort()

    # 2) Save and swap the module-level manifest lock. We monkey-patch the
    #    module attribute rather than editing gold40_overfit.py on disk.
    original_fixed_manifest: Path = _gold40_overfit.FIXED_GOLD40_MANIFEST
    temporary_fixed_manifest: Path = GOLD40_V3_MANIFEST

    _emit_provenance(
        output_dir=_sniff_output_dir_from_argv(),
        original_fixed_manifest=original_fixed_manifest,
        temporary_fixed_manifest=temporary_fixed_manifest,
        original_backend_sha256=original_backend_sha256,
    )

    _gold40_overfit.FIXED_GOLD40_MANIFEST = temporary_fixed_manifest
    swap_restored = False
    try:
        # 3) Delegate everything to the ORIGINAL main(). Do NOT re-implement
        #    parse_args, run, dataloader, forward, loss, or checkpointing.
        _gold40_overfit.main()
    finally:
        # 4) Restore the original module-level constant. This restoration is
        #    unconditional: even on KeyboardInterrupt / SystemExit / any raised
        #    exception, the swap is undone before this frame returns.
        _gold40_overfit.FIXED_GOLD40_MANIFEST = original_fixed_manifest
        swap_restored = (
            _gold40_overfit.FIXED_GOLD40_MANIFEST is original_fixed_manifest
        )
        # Diagnostic hint on stderr only; do not raise from within finally.
        if not swap_restored:  # pragma: no cover - defensive
            print(
                "[gold40_v3_overfit] warning: FIXED_GOLD40_MANIFEST restoration "
                "check failed; module attribute is not the original object.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
