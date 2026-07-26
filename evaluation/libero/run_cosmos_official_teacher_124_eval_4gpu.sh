#!/usr/bin/env bash
# Official Cosmos teacher-only matched-budget K=1/2/4 evaluation on 4 GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Two formal shards map the four visible GPUs to evaluator pairs (0,1) and
# (2,3). Keep full JSON/CSV results while retaining only episode seed 0 as a
# representative video for every task and matched solver budget.
export COSMOS_TEACHER_FORMAL_NUM_SHARDS=2
export S4_VIDEO_SEEDS=0

exec bash "${SCRIPT_DIR}/run_cosmos_official_teacher_124_eval_8gpu.sh" "$@"
