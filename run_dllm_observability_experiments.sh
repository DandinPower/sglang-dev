#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

SCRIPT="launch_dllm_observability_experiment_on_modal.py"
REQUEST_COUNTS=(1 8 16)

run_one() {
  local patch_target="$1"

  echo "Running ${patch_target}: ${REQUEST_COUNTS[*]}"
  modal run "${SCRIPT}" \
    --patch-target "${patch_target}" \
    --request-count_configurations "${REQUEST_COUNTS[@]}"
}

run_one "sglang"
run_one "sglang-baseline"
