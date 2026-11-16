#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
if [[ "$(uname -s)" != Linux ]]; then
  echo 'This launcher requires Linux. The bounded Python check can run on Windows.' >&2
  exit 2
fi
command="${1:-help}"
case "$command" in
  check)
    device="${2:-cpu}"
    "$PYTHON_BIN" -m pip check
    "$PYTHON_BIN" -B -m unittest checks.test_ch5_neural_modules checks.test_ch5_data checks.test_ch5_training checks.test_ch5_runtime checks.test_ch5_pipeline checks.test_ch5_experiments
    run="outputs/chapter5/linux_check_$(date +%Y%m%d_%H%M%S)_$$"
    "$PYTHON_BIN" -B cli.py ch5-experiment prepare --config configs/chapter5/check_experiment_v1.json --root "$run" --execute
    "$PYTHON_BIN" -B cli.py ch5-experiment suite --root "$run" --device "$device" --resume --execute
    echo "Linux acceptance artifacts: $run"
    ;;
  plan)
    profile="${2:-pilot}"
    "$PYTHON_BIN" -B cli.py ch5-experiment plan --config "configs/chapter5/${profile}_experiment_v1.json"
    ;;
  prepare)
    profile="${2:?Usage: prepare pilot|formal RUN}"
    run="${3:?Provide a fresh outputs/chapter5/RUN directory}"
    "$PYTHON_BIN" -B cli.py ch5-experiment prepare --config "configs/chapter5/${profile}_experiment_v1.json" --root "$run" --execute
    ;;
  train|evaluate|all)
    run="${2:?Usage: train|evaluate|all RUN [cpu|cuda|auto]}"
    device="${3:-cpu}"
    "$PYTHON_BIN" -B cli.py ch5-experiment suite --root "$run" --stage "$command" --device "$device" --resume --execute
    ;;
  report)
    run="${2:?Usage: report RUN}"
    "$PYTHON_BIN" -B cli.py ch5-experiment report --root "$run" --execute
    ;;
  *)
    echo 'Usage: bash scripts/ch5_linux.sh check [cpu|cuda|auto]'
    echo '       bash scripts/ch5_linux.sh plan [pilot|formal]'
    echo '       bash scripts/ch5_linux.sh prepare pilot|formal outputs/chapter5/RUN'
    echo '       bash scripts/ch5_linux.sh train|evaluate|all outputs/chapter5/RUN [cpu|cuda|auto]'
    echo '       bash scripts/ch5_linux.sh report outputs/chapter5/RUN'
    [[ "$command" == help ]] || exit 2
    ;;
esac
