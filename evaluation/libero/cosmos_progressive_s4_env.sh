#!/usr/bin/env bash
# Source this file after activating flashwam and before launching the S4 suite.
# It only exports defaults; any variable already supplied by the caller wins.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    printf '%s\n' "Source this file instead: source evaluation/libero/cosmos_progressive_s4_env.sh" >&2
    exit 2
fi

export COSMOS_WORKER_ENV_ROOT="${COSMOS_WORKER_ENV_ROOT:-/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310}"
export COSMOS_POLICY_PYTHON="${COSMOS_POLICY_PYTHON:-${COSMOS_WORKER_ENV_ROOT}/bin/python}"
export COSMOS_PREDICT2_REPO="${COSMOS_PREDICT2_REPO:-/kpfs-intern/jialongliu/projects/cosmos-predict2.5}"
# The official code remains in the separately managed Cosmos checkout.  The
# worker receives these package roots explicitly, so it does not rely on an
# editable install in the FlashWAM environment.
export COSMOS_POLICY_EXTRA_PYTHONPATH="${COSMOS_POLICY_EXTRA_PYTHONPATH:-${COSMOS_PREDICT2_REPO}/packages/cosmos-cuda:${COSMOS_PREDICT2_REPO}/packages/cosmos-oss}"
# NVIDIA's Python wheels bundle cuDNN/CUDA libraries beneath site-packages.
# Transformer Engine loads them with ctypes, so make those directories visible
# to the dynamic linker without changing the host flashwam environment.
export COSMOS_WORKER_SITE_PACKAGES="${COSMOS_WORKER_SITE_PACKAGES:-${COSMOS_WORKER_ENV_ROOT}/lib/python3.10/site-packages}"
export COSMOS_WORKER_CUDA_LIBRARY_PATH="${COSMOS_WORKER_CUDA_LIBRARY_PATH:-${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cublas/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_cupti/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cudnn/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cufft/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cufile/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/curand/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusolver/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusparse/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/cusparselt/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/nccl/lib:${COSMOS_WORKER_SITE_PACKAGES}/nvidia/nvjitlink/lib}"
export LD_LIBRARY_PATH="${COSMOS_WORKER_CUDA_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export COSMOS_POLICY_PATH="${COSMOS_POLICY_PATH:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B}"
export COSMOS_PREDICT25_LOCAL_MODEL_DIR="${COSMOS_PREDICT25_LOCAL_MODEL_DIR:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World}"
export HF_HOME="${HF_HOME:-/kpfs-intern/jialongliu/models/cosmos_predict2_5/hf_cache}"

export S4_CKPT_ROOT="${S4_CKPT_ROOT:-/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/progressive_stage2_full/s4/step_5000/online_student/transformer}"
export S4_DATASET_PATH="${S4_DATASET_PATH:-/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot}"
export S4_EMPTY_EMBEDDING="${S4_EMPTY_EMBEDDING:-${S4_DATASET_PATH}/empty_emb.pt}"
export S4_CONFIG="${S4_CONFIG:-distillation_flowmap.config_libero_cosmos_policy_stage2_progressive}"
export S4_LIBERO_BENCHMARK="${S4_LIBERO_BENCHMARK:-libero_10}"

export S4_RESULTS_ROOT="${S4_RESULTS_ROOT:-/kpfs-intern/jialongliu/results}"
export S4_RUN_ID="${S4_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export SUITE_ROOT="${SUITE_ROOT:-${S4_RESULTS_ROOT}/cosmos_s4_full_${S4_RUN_ID}}"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
