#!/usr/bin/env bash
# Official Cosmos teacher-only matched-budget K=1/2/4 evaluation on 4 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Four formal shards map each visible GPU to one evaluator and one
# official-teacher raw worker. Keep full JSON/CSV results while retaining only
# episode seed 0 as a representative video for every task and matched solver
# budget.
export COSMOS_TEACHER_FORMAL_NUM_SHARDS=4
export S4_FORMAL_GPU_LAYOUT=colocated
export S4_VIDEO_SEEDS=0

exec bash "${SCRIPT_DIR}/run_cosmos_official_teacher_124_eval_8gpu.sh" "$@"
