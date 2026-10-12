#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
if [[ $# -lt 2 ]]; then
  echo "usage: $0 INPUT_JSONL OUTPUT_DIR" >&2
  exit 2
fi
INPUT="$1"
OUTPUT_DIR="$2"
TMP_SPLIT="${OUTPUT_DIR}/accepted_split.jsonl"
mkdir -p "${OUTPUT_DIR}"
python -m experiments.counterfactual_cot.data.filter_pairs --input "${INPUT}" --output "${OUTPUT_DIR}/accepted.jsonl"
python -m experiments.counterfactual_cot.data.split_dataset --input "${OUTPUT_DIR}/accepted.jsonl" --output "${TMP_SPLIT}"
python -m experiments.counterfactual_cot.data.freeze_manifest --input "${TMP_SPLIT}" --output-dir "${OUTPUT_DIR}"
