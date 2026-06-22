#!/bin/bash
# ============================================================
# LIBERO Evaluation v2 - Fixed streaming KV cache inference
#
# Fixes the original 0% success rate caused by:
#   1. _infer() clearing KV cache (destroying streaming history)
#   2. flowmap_inference() bypassing KV cache (no temporal context)
#
# The wan_va_server.py has been patched to use standard denoising
# with KV cache (matching original LingBot-VA pattern).
#
# Usage:
#   bash run_libero_eval_v2.sh
#
# Tasks: easy single-object (8, 9) + simple two-object (3, 7)
# ============================================================

set -e

# --------------- Configuration ---------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Python environment
PYTHON="/root/nas/junjie/conda_envs/any_wam/bin/python"

# Checkpoints
BASE_MODEL="/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-base"
TEACHER_MODEL="/root/nas/junjie/jj/Any_WAM/checkpoints/lingbot-va-posttrain-libero"
STUDENT_CKPT="/root/nas/junjie/jj/Any_WAM/distillation_flowmap/output_libero_onpolicy/checkpoints/step_1700/online_student/transformer"

# Server configuration
TEACHER_PORT=29538
STUDENT_PORT=29539
TEACHER_GPU=1
STUDENT_GPU=2

# Evaluation configuration
LIBERO_BENCHMARK="libero_10"
# Tasks: 3, 7, 8, 9 (easy single/two-object tasks)
TASKS=(3 7 8 9)
TEST_NUM=20  # episodes per task

# Inference steps matching original LIBERO config
NUM_INFERENCE_STEPS=20
ACTION_NUM_INFERENCE_STEPS=50

# Output directories
OUTPUT_DIR="${PROJECT_ROOT}/evaluation/outputs"
TEACHER_OUT="${OUTPUT_DIR}/teacher_v2"
STUDENT_OUT="${OUTPUT_DIR}/student_v2"

# MuJoCo rendering
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

echo "============================================"
echo "  LIBERO Evaluation v2 (Fixed KV Cache)"
echo "============================================"
echo "  Teacher model:  ${TEACHER_MODEL}"
echo "  Student ckpt:   ${STUDENT_CKPT}"
echo "  Teacher GPU:    ${TEACHER_GPU} (port ${TEACHER_PORT})"
echo "  Student GPU:    ${STUDENT_GPU} (port ${STUDENT_PORT})"
echo "  Tasks:          ${TASKS[*]}"
echo "  Episodes/task:  ${TEST_NUM}"
echo "  Video steps:    ${NUM_INFERENCE_STEPS}"
echo "  Action steps:   ${ACTION_NUM_INFERENCE_STEPS}"
echo "============================================"

# --------------- Verify checkpoints ---------------
check_exists() {
    if [ ! -e "$1" ]; then
        echo "ERROR: $1 not found!"
        exit 1
    fi
}
check_exists "${TEACHER_MODEL}/transformer/config.json"
check_exists "${TEACHER_MODEL}/vae"
check_exists "${TEACHER_MODEL}/text_encoder"
check_exists "${TEACHER_MODEL}/tokenizer"
check_exists "${BASE_MODEL}/vae"
check_exists "${STUDENT_CKPT}/config.json"

# --------------- Cleanup function ---------------
cleanup() {
    echo ""
    echo "[Cleanup] Stopping servers..."
    if [ -n "${TEACHER_PID}" ]; then
        kill ${TEACHER_PID} 2>/dev/null || true
        wait ${TEACHER_PID} 2>/dev/null || true
    fi
    if [ -n "${STUDENT_PID}" ]; then
        kill ${STUDENT_PID} 2>/dev/null || true
        wait ${STUDENT_PID} 2>/dev/null || true
    fi
    pkill -f "torch.distributed.run.*2953[89]" 2>/dev/null || true
    echo "[Cleanup] Done."
}
trap cleanup EXIT

# --------------- Start Teacher Server ---------------
# Uses original LingBot-VA server (standard denoising with KV cache)
echo ""
echo "=== Starting Teacher Server (GPU ${TEACHER_GPU}, port ${TEACHER_PORT}) ==="

CUDA_VISIBLE_DEVICES=${TEACHER_GPU} \
WAN22_PRETRAINED_PATH="${TEACHER_MODEL}" \
${PYTHON} -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $((TEACHER_PORT + 100)) \
    ${PROJECT_ROOT}/lingbot-va/wan_va/wan_va_server.py \
    --config-name libero \
    --port ${TEACHER_PORT} \
    --save_root "${TEACHER_OUT}" \
    > "${OUTPUT_DIR}/teacher_server.log" 2>&1 &

TEACHER_PID=$!
echo "[Teacher] PID: ${TEACHER_PID}"

# --------------- Start Student Server ---------------
# Uses FIXED Flash-WAM server (standard denoising with KV cache)
echo ""
echo "=== Starting Student Server (GPU ${STUDENT_GPU}, port ${STUDENT_PORT}) ==="

CUDA_VISIBLE_DEVICES=${STUDENT_GPU} \
WAN22_PRETRAINED_PATH="${BASE_MODEL}" \
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}" \
${PYTHON} -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $((STUDENT_PORT + 100)) \
    ${PROJECT_ROOT}/wan_va/wan_va_server.py \
    --config-name libero \
    --port ${STUDENT_PORT} \
    --checkpoint-path "${STUDENT_CKPT}" \
    --num-steps ${NUM_INFERENCE_STEPS} \
    --action-num-steps ${ACTION_NUM_INFERENCE_STEPS} \
    --save-root "${STUDENT_OUT}" \
    > "${OUTPUT_DIR}/student_server.log" 2>&1 &

STUDENT_PID=$!
echo "[Student] PID: ${STUDENT_PID}"

# --------------- Wait for servers ---------------
echo ""
echo "=== Waiting for servers to initialize (120s) ==="
sleep 120

# Check servers are alive
for pid_info in "TEACHER ${TEACHER_PID}" "STUDENT ${STUDENT_PID}"; do
    set -- ${pid_info}
    name=$1
    pid=$2
    if ! kill -0 ${pid} 2>/dev/null; then
        echo "ERROR: ${name} server died! Last 50 lines of log:"
        log_name=$(echo "${name}" | tr '[:upper:]' '[:lower:]')
        tail -50 "${OUTPUT_DIR}/${log_name}_server.log"
        exit 1
    fi
    echo "[${name}] Server running OK (PID ${pid})"
done

# --------------- Run evaluation ---------------
mkdir -p "${TEACHER_OUT}/results" "${STUDENT_OUT}/results"

run_eval() {
    local name="$1"
    local port="$2"
    local out_dir="$3"
    local task="$4"

    echo ""
    echo "========================================="
    echo "  ${name}: Task ${task}"
    echo "========================================="

    ${PYTHON} ${PROJECT_ROOT}/evaluation/libero/client.py \
        --libero-benchmark ${LIBERO_BENCHMARK} \
        --port ${port} \
        --test-num ${TEST_NUM} \
        --task-range ${task} $((task + 1)) \
        --out-dir "${out_dir}/results" 2>&1 | tee -a "${out_dir}/task_${task}.log"

    local result_file="${out_dir}/results/libero_10_${task}.json"
    if [ -f "${result_file}" ]; then
        echo "[${name}] Task ${task} result: $(cat ${result_file})"
    fi
}

# Run teacher and student sequentially for each task
for task in "${TASKS[@]}"; do
    echo ""
    echo "########################################################"
    echo "  TASK ${task}"
    echo "########################################################"

    # Run teacher
    run_eval "TEACHER" ${TEACHER_PORT} "${TEACHER_OUT}" ${task}

    # Run student
    run_eval "STUDENT" ${STUDENT_PORT} "${STUDENT_OUT}" ${task}
done

# --------------- Summary ---------------
echo ""
echo "============================================"
echo "  EVALUATION COMPLETE"
echo "============================================"

compute_summary() {
    local label="$1"
    local out_dir="$2"
    local total_succ=0
    local total_eps=0
    echo ""
    echo "${label} results:"
    for task in "${TASKS[@]}"; do
        local f="${out_dir}/results/libero_10_${task}.json"
        if [ -f "$f" ]; then
            cat "$f" | ${PYTHON} -c "
import sys,json
d=json.load(sys.stdin)
print(f'  Task {d.get(\"task\",\"?\")}: {d[\"succ_num\"]:.0f}/{d[\"total_num\"]:.0f} = {d[\"succ_rate\"]*100:.1f}%')
" 2>/dev/null || echo "  Task ${task}: $(cat $f)"
            # Accumulate
            local succ=$(python3 -c "import json; d=json.load(open('$f')); print(d['succ_num'])" 2>/dev/null || echo 0)
            local total=$(python3 -c "import json; d=json.load(open('$f')); print(d['total_num'])" 2>/dev/null || echo 0)
            total_succ=$(python3 -c "print(${total_succ} + ${succ})")
            total_eps=$(python3 -c "print(${total_eps} + ${total})")
        fi
    done
    if [ "${total_eps}" != "0" ]; then
        avg_rate=$(python3 -c "print(f'${total_succ}/${total_eps} = {${total_succ}/${total_eps}*100:.1f}%')")
        echo "  OVERALL: ${avg_rate}"
    fi
}

compute_summary "Teacher" "${TEACHER_OUT}"
compute_summary "Student" "${STUDENT_OUT}"

echo ""
echo "Server logs:"
echo "  Teacher: ${OUTPUT_DIR}/teacher_server.log"
echo "  Student: ${OUTPUT_DIR}/student_server.log"
echo ""
echo "Videos saved to:"
echo "  Teacher: ${TEACHER_OUT}/results/${LIBERO_BENCHMARK}/"
echo "  Student: ${STUDENT_OUT}/results/${LIBERO_BENCHMARK}/"
