#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
python -m experiments.counterfactual_cot.inference.generate_with_cot "$@"
