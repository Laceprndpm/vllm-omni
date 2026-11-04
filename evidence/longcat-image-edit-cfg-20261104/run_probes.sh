#!/usr/bin/env bash
set -euo pipefail
STUDY_ROOT=/root/autodl-tmp/omni-9826/ssim-matrix
TASK_ROOT=/root/autodl-tmp/omni-9826
trap 'PROBE_EXIT_CODE=$?; printf "%s\n" "$PROBE_EXIT_CODE" > "$STUDY_ROOT/probe-exit-code.txt"' EXIT
source "$TASK_ROOT/env.sh"
cd "$STUDY_ROOT"
test -f GENERATIONS_COMPLETE
for VARIANT in reference baseline fixed; do
  SOURCE_ROOT="$STUDY_ROOT/baseline-src"
  if [[ "$VARIANT" == fixed ]]; then
    SOURCE_ROOT="$TASK_ROOT/vllm-omni"
  fi
  PYTHONPATH="$SOURCE_ROOT:$STUDY_ROOT:$TASK_ROOT" python run_probe.py --variant "$VARIANT" \
    > "logs/probe-$VARIANT.log" 2>&1
done
touch PROBES_COMPLETE
