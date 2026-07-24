import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "evaluation" / "libero" / "cosmos_progressive_s4_env.sh"


def _source_environment(*, extra=None):
    env = os.environ.copy()
    env.update(
        {
            "S4_RESULTS_ROOT": "/tmp/cosmos progressive results",
            "S4_RUN_ID": "unit-test",
        }
    )
    if extra:
        env.update(extra)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -e; source "$1"; env -0',
            "bash",
            str(SCRIPT),
        ],
        env=env,
        text=False,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode()
    return {
        item.split(b"=", 1)[0].decode(): item.split(b"=", 1)[1].decode()
        for item in result.stdout.split(b"\0")
        if item
    }


def test_sourceable_env_file_exports_concrete_cosmos_s4_defaults():
    values = _source_environment()

    assert values["COSMOS_WORKER_ENV_ROOT"] == "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310"
    assert values["COSMOS_POLICY_PYTHON"] == (
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/bin/python"
    )
    assert values["COSMOS_PREDICT2_REPO"] == "/kpfs-intern/jialongliu/projects/cosmos-predict2.5"
    assert values["COSMOS_POLICY_EXTRA_PYTHONPATH"] == (
        "/kpfs-intern/jialongliu/projects/cosmos-predict2.5/packages/cosmos-cuda:"
        "/kpfs-intern/jialongliu/projects/cosmos-predict2.5/packages/cosmos-oss"
    )
    assert values["COSMOS_WORKER_SITE_PACKAGES"] == (
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages"
    )
    assert values["COSMOS_WORKER_CUDA_LIBRARY_PATH"] == (
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/cublas/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/cuda_cupti/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/cuda_runtime/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/cudnn/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/cufft/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/cufile/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/curand/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/cusolver/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/cusparse/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/cusparselt/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/nccl/lib:"
        "/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/"
        "lib/python3.10/site-packages/nvidia/nvjitlink/lib"
    )
    assert values["COSMOS_POLICY_PATH"] == (
        "/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/"
        "Cosmos-Policy-LIBERO-Predict2-2B"
    )
    assert values["COSMOS_PREDICT25_LOCAL_MODEL_DIR"] == (
        "/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/local_hf/"
        "Cosmos-Predict2-2B-Video2World"
    )
    assert values["S4_CKPT_ROOT"] == (
        "/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/"
        "progressive_stage2_full/s4/step_5000/online_student/transformer"
    )
    assert values["S4_DATASET_PATH"] == "/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot"
    assert values["S4_EMPTY_EMBEDDING"] == (
        "/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot/empty_emb.pt"
    )
    assert values["SUITE_ROOT"] == "/tmp/cosmos progressive results/cosmos_s4_full_unit-test"
    assert values["S4_VIDEO_SEEDS"] == "0,1"
    assert values["S4_FORMAL_NUM_SHARDS"] == "4"
    assert values["MATRIX_ROOT"] == (
        "/tmp/cosmos progressive results/cosmos_joint_124_unit-test"
    )
    assert values["HF_HUB_OFFLINE"] == "1"
    assert values["HF_DATASETS_OFFLINE"] == "1"
    assert values["TRANSFORMERS_OFFLINE"] == "1"


def test_sourceable_env_file_preserves_caller_overrides():
    values = _source_environment(
        extra={
            "COSMOS_POLICY_PATH": "/custom/policy",
            "SUITE_ROOT": "/custom/results",
            "PYTHON_BIN": "/custom/flashwam/bin/python",
            "COSMOS_POLICY_EXTRA_PYTHONPATH": "/custom/cosmos/site-packages",
            "COSMOS_WORKER_CUDA_LIBRARY_PATH": "/custom/cuda-libs",
            "LD_LIBRARY_PATH": "/caller/lib",
            "S4_VIDEO_SEEDS": "3",
            "S4_FORMAL_NUM_SHARDS": "2",
            "MATRIX_ROOT": "/custom/matrix",
        }
    )

    assert values["COSMOS_POLICY_PATH"] == "/custom/policy"
    assert values["SUITE_ROOT"] == "/custom/results"
    assert values["PYTHON_BIN"] == "/custom/flashwam/bin/python"
    assert values["COSMOS_POLICY_EXTRA_PYTHONPATH"] == "/custom/cosmos/site-packages"
    assert values["COSMOS_WORKER_CUDA_LIBRARY_PATH"] == "/custom/cuda-libs"
    assert values["LD_LIBRARY_PATH"] == "/custom/cuda-libs:/caller/lib"
    assert values["S4_VIDEO_SEEDS"] == "3"
    assert values["S4_FORMAL_NUM_SHARDS"] == "2"
    assert values["MATRIX_ROOT"] == "/custom/matrix"
