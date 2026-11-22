#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/ch5_history_pair.sh short|batch PATH_TO_ORIGINAL_BATCH
cd "$(dirname "$0")/.."
mode="${1:?short or batch}"
batch="${2:?original d2_batch_r123_20261121_140145_3923224 directory}"
stamp="$(date +%Y%m%d_%H%M%S)"
run="outputs/chapter5/history_pair_${mode}_${stamp}"
mkdir -p outputs/chapter5
case "$mode" in
  short) cases=(9 3) ;;
  batch) cases=({0..29}) ;;
  *) echo 'mode must be short or batch' >&2; exit 2 ;;
esac
echo "Results: $run ; log: ${run}.log"
python -u -m chapter5_comm.d2_history_pair --batch "$batch" --output "$run" \
  --cases "${cases[@]}" 2>&1 | tee "${run}.log"
