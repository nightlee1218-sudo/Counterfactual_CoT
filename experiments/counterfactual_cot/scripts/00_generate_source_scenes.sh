#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_env.sh"
python -m experiments.counterfactual_cot.data.generate_source_scenes "$@"
