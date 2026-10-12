#!/usr/bin/env bash
# Shared launcher guard for counterfactual CoT scripts.

_cfcot_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_cfcot_resolved_root="$(cd "${_cfcot_script_dir}/../../.." && pwd)"

if [[ -z "${BAGEL_ROOT:-}" ]]; then
  export BAGEL_ROOT="${_cfcot_resolved_root}"
fi

if [[ ! -d "${BAGEL_ROOT}" || ! -f "${BAGEL_ROOT}/train/pretrain_unified_navit.py" ]]; then
  echo "ERROR: BAGEL_ROOT is not a valid BAGEL checkout: ${BAGEL_ROOT}" >&2
  exit 2
fi

export BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-/dockerdata/niightli/BAGEL/models/BAGEL-7B-MoT}"
if [[ ! -d "${BAGEL_MODEL_PATH}" || ! -f "${BAGEL_MODEL_PATH}/ema.safetensors" || ! -f "${BAGEL_MODEL_PATH}/ae.safetensors" ]]; then
  echo "ERROR: BAGEL_MODEL_PATH is not a valid BAGEL weight directory: ${BAGEL_MODEL_PATH}" >&2
  exit 2
fi

case ":${PYTHONPATH:-}:" in
  *":${BAGEL_ROOT}:"*) ;;
  *) export PYTHONPATH="${BAGEL_ROOT}:${PYTHONPATH:-}" ;;
esac

python - <<'PY'
import importlib
importlib.import_module("experiments.counterfactual_cot")
PY
