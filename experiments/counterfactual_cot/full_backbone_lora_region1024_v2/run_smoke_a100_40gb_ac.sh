#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Redirecting to the round-2 1024 region-loss smoke launcher." >&2
exec bash "$SCRIPT_DIR/run_smoke_a100_40gb.sh" "$@"
