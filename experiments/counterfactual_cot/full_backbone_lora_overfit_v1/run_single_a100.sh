#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Redirecting to the explicit A100 80GB launcher." >&2
exec bash "$SCRIPT_DIR/run_single_a100_80gb.sh" "$@"
