#!/usr/bin/env bash
set -euo pipefail
STUDY_ROOT=/root/autodl-tmp/omni-9826/ssim-matrix
TASK_ROOT=/root/autodl-tmp/omni-9826
trap 'STUDY_EXIT_CODE=$?; printf "%s\n" "$STUDY_EXIT_CODE" > "$STUDY_ROOT/exit-code.txt"' EXIT
source "$TASK_ROOT/env.sh"
cd "$STUDY_ROOT"
for VARIANT in reference baseline fixed; do
  SOURCE_ROOT="$STUDY_ROOT/baseline-src"
  if [[ "$VARIANT" == fixed ]]; then
    SOURCE_ROOT="$TASK_ROOT/vllm-omni"
  fi
  PYTHONPATH="$SOURCE_ROOT:$STUDY_ROOT:$TASK_ROOT" python run_matrix.py --variant "$VARIANT" --preflight \
    > "logs/preflight-$VARIANT.log" 2>&1
done
for VARIANT in reference baseline fixed; do
  SOURCE_ROOT="$STUDY_ROOT/baseline-src"
  if [[ "$VARIANT" == fixed ]]; then
    SOURCE_ROOT="$TASK_ROOT/vllm-omni"
  fi
  PYTHONPATH="$SOURCE_ROOT:$STUDY_ROOT:$TASK_ROOT" python run_matrix.py --variant "$VARIANT" \
    > "logs/$VARIANT.log" 2>&1
done
touch "$STUDY_ROOT/GENERATIONS_COMPLETE"
