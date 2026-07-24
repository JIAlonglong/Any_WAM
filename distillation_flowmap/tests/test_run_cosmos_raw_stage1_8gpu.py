import json
import os
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "distillation_flowmap" / "run_cosmos_raw_stage1_8gpu.sh"
PYTHON = Path("/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python")
DEVICES = "0,1,2,3,4,5,6,7"
RAW_CONTRACT = {
    "contract_version": 2,
    "training_contract_stage": "raw_stage1",
    "action_packing_schema": "downsample_survivor_v2",
    "action_downsample_factor": 4,
    "action_chunk_shape": [4, 4],
}


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _transformer(root: Path, *, step: int | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payload = {} if step is None else {**RAW_CONTRACT, "checkpoint_step": step}
    (root / "config.json").write_text(json.dumps(payload) + "\n", encoding="utf-8")
    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"test")
    return root


def _layout(tmp_path: Path) -> dict[str, str]:
    clean_base = tmp_path / "clean-base"
    _transformer(clean_base / "transformer")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "empty_emb.pt").write_bytes(b"test")
    policy = tmp_path / "cosmos-policy"
    policy.mkdir()
    worker_repo = tmp_path / "cosmos-predict2.5"
    worker_repo.mkdir()
    local_model = tmp_path / "cosmos-local-model"
    local_model.mkdir()
    worker_python = tmp_path / "cosmos-python"
    _write_executable(worker_python, "exit 0\n")
    preflight = tmp_path / "fake-preflight"
    _write_executable(
        preflight,
        'printf "%s\\n" "$*" >> "$PREFLIGHT_LOG"\n'
        'exit "${FAKE_PREFLIGHT_EXIT:-0}"\n',
    )
    torchrun = tmp_path / "fake-torchrun"
    _write_executable(
        torchrun,
        'printf "%s\\n" "$*" > "$TORCHRUN_LOG"\n'
        "env | LC_ALL=C sort >> \"$TORCHRUN_LOG\"\n",
    )
    return {
        "clean_base": str(clean_base),
        "dataset": str(dataset),
        "policy": str(policy),
        "worker_repo": str(worker_repo),
        "local_model": str(local_model),
        "worker_python": str(worker_python),
        "preflight": str(preflight),
        "torchrun": str(torchrun),
    }


def _env(tmp_path: Path) -> dict[str, str]:
    paths = _layout(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "STAGE1_OUTPUT": str(tmp_path / "stage1-output"),
            "CLEAN_STUDENT_BASE_MODEL_PATH": paths["clean_base"],
            "DATASET_PATH": paths["dataset"],
            "COSMOS_POLICY_PATH": paths["policy"],
            "COSMOS_POLICY_PYTHON": paths["worker_python"],
            "COSMOS_PREDICT2_REPO": paths["worker_repo"],
            "COSMOS_PREDICT25_LOCAL_MODEL_DIR": paths["local_model"],
            "PYTHON_BIN": str(PYTHON),
            "PREFLIGHT_BIN": paths["preflight"],
            "PREFLIGHT_LOG": str(tmp_path / "preflight.log"),
            "TORCHRUN_BIN": paths["torchrun"],
            "TORCHRUN_LOG": str(tmp_path / "torchrun.log"),
            "CUDA_VISIBLE_DEVICES": DEVICES,
        }
    )
    return env


def _run(mode: str, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), mode, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _checkpoint(output: Path, step: int) -> Path:
    checkpoint = output / "checkpoints" / f"step_{step}"
    for variant in ("online_student", "target_student"):
        _transformer(checkpoint / variant / "transformer", step=step)
    (checkpoint / "optimizer.pt").write_bytes(b"test")
    (checkpoint / "lr_scheduler.pt").write_bytes(b"test")
    return checkpoint


def test_dry_run_prints_corrected_raw_contract_without_mutation(tmp_path):
    env = _env(tmp_path)

    result = _run("dry-run", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage1" in result.stdout
    assert "MAX_TRAIN_STEPS=5000" in result.stdout
    assert "SAVE_INTERVAL=1000" in result.stdout
    assert "TRAIN_SEED=42" in result.stdout
    assert "MASTER_PORT=29671" in result.stdout
    assert "--nproc_per_node=8" in result.stdout
    assert "--master_port=29671" in result.stdout
    assert f"STUDENT_BASE_MODEL_PATH={env['CLEAN_STUDENT_BASE_MODEL_PATH']}" in result.stdout
    assert "RESUME_FROM_PATH=" in result.stdout
    assert "raw_stage1_5000" not in result.stdout
    assert not Path(env["STAGE1_OUTPUT"]).exists()
    assert Path(env["PREFLIGHT_LOG"]).read_text(encoding="utf-8")
    assert not Path(env["TORCHRUN_LOG"]).exists()


def test_dry_run_accepts_sharded_clean_base_layout(tmp_path):
    env = _env(tmp_path)
    transformer = Path(env["CLEAN_STUDENT_BASE_MODEL_PATH"]) / "transformer"
    (transformer / "diffusion_pytorch_model.safetensors").unlink()
    shards = [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]
    for shard in shards:
        (transformer / shard).write_bytes(b"test")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}}) + "\n",
        encoding="utf-8",
    )

    result = _run("dry-run", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not Path(env["STAGE1_OUTPUT"]).exists()


def test_cli_overrides_and_valid_run_tag_are_resolved(tmp_path):
    env = _env(tmp_path)
    output = tmp_path / "custom"

    result = _run(
        "dry-run",
        "--steps",
        "20",
        "--save-interval",
        "10",
        "--master-port",
        "30001",
        "--output-dir",
        str(output),
        "--run-tag",
        "trial_1.a-b",
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MAX_TRAIN_STEPS=20" in result.stdout
    assert "SAVE_INTERVAL=10" in result.stdout
    assert "MASTER_PORT=30001" in result.stdout
    assert f"OUTPUT_DIR={output}" in result.stdout
    assert "RUN_TAG=trial_1.a-b" in result.stdout
    assert not output.exists()


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--steps", "0"), "positive integer"),
        (("--save-interval", "x"), "positive integer"),
        (("--master-port", "0"), "invalid master port"),
        (("--master-port", "65536"), "invalid master port"),
        (("--run-tag", "../bad"), "invalid run tag"),
        (("--wat", "1"), "unknown option"),
    ],
)
def test_rejects_invalid_cli_values(tmp_path, args, message):
    env = _env(tmp_path)

    result = _run("dry-run", *args, env=env)

    assert result.returncode != 0
    assert message in result.stderr.lower()
    assert not Path(env["STAGE1_OUTPUT"]).exists()


@pytest.mark.parametrize(
    ("devices", "message"),
    [
        ("0,1", "exactly 8"),
        ("0,1,2,3,4,5,6,6", "duplicate"),
        ("0,1,2,3,4,5,6,x", "non-numeric"),
    ],
)
def test_rejects_invalid_gpu_layout(tmp_path, devices, message):
    env = _env(tmp_path)
    env["CUDA_VISIBLE_DEVICES"] = devices

    result = _run("dry-run", env=env)

    assert result.returncode != 0
    assert message in result.stderr


@pytest.mark.parametrize("missing", ["clean_base", "dataset", "policy"])
def test_rejects_missing_required_input_paths(tmp_path, missing):
    env = _env(tmp_path)
    key = {
        "clean_base": "CLEAN_STUDENT_BASE_MODEL_PATH",
        "dataset": "DATASET_PATH",
        "policy": "COSMOS_POLICY_PATH",
    }[missing]
    env[key] = str(tmp_path / "missing")

    result = _run("dry-run", env=env)

    assert result.returncode != 0
    assert key in result.stderr


def test_fresh_mode_refuses_any_existing_output_and_never_auto_selects_checkpoint(tmp_path):
    env = _env(tmp_path)
    output = Path(env["STAGE1_OUTPUT"])
    output.mkdir()

    empty = _run("dry-run", env=env)
    assert empty.returncode != 0
    assert "Refusing fresh run" in empty.stderr

    _checkpoint(output, 3000)
    _checkpoint(output, 4000)
    populated = _run("run", env=env)
    assert populated.returncode != 0
    assert "Refusing fresh run" in populated.stderr
    assert "step_4000" not in populated.stdout
    assert not Path(env["TORCHRUN_LOG"]).exists()


def test_resume_requires_exact_step_and_complete_v2_raw_checkpoint(tmp_path):
    env = _env(tmp_path)
    checkpoint = _checkpoint(Path(env["STAGE1_OUTPUT"]), 3000)

    result = _run("dry-run", "--steps", "5000", "--resume-step", "3000", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"RESUME_FROM_PATH={checkpoint}" in result.stdout
    assert "RESUME_ONLINE_FROM_TARGET=0" in result.stdout
    assert "RESET_RESUME_STEP=0" in result.stdout
    assert "RESUME_OPTIMIZER_STATE=1" in result.stdout
    assert not Path(env["TORCHRUN_LOG"]).exists()


@pytest.mark.parametrize(
    ("variant", "field", "value", "message"),
    [
        ("online_student", "contract_version", 1, "contract_version"),
        ("target_student", "training_contract_stage", "progressive_stage2", "training_contract_stage"),
        ("target_student", "checkpoint_step", 2999, "checkpoint_step"),
    ],
)
def test_resume_rejects_wrong_metadata_on_both_variants(
    tmp_path, variant, field, value, message
):
    env = _env(tmp_path)
    checkpoint = _checkpoint(Path(env["STAGE1_OUTPUT"]), 3000)
    config = checkpoint / variant / "transformer" / "config.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload[field] = value
    config.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = _run("dry-run", "--resume-step", "3000", env=env)

    assert result.returncode != 0
    assert message in result.stderr
    assert not Path(env["TORCHRUN_LOG"]).exists()


@pytest.mark.parametrize("state_file", ["optimizer.pt", "lr_scheduler.pt"])
def test_resume_rejects_missing_training_state(tmp_path, state_file):
    env = _env(tmp_path)
    checkpoint = _checkpoint(Path(env["STAGE1_OUTPUT"]), 3000)
    (checkpoint / state_file).unlink()

    result = _run("dry-run", "--resume-step", "3000", env=env)

    assert result.returncode != 0
    assert state_file in result.stderr


def test_resume_step_must_precede_requested_final_step(tmp_path):
    env = _env(tmp_path)
    _checkpoint(Path(env["STAGE1_OUTPUT"]), 5000)

    result = _run("dry-run", "--resume-step", "5000", "--steps", "5000", env=env)

    assert result.returncode != 0
    assert "less than MAX_TRAIN_STEPS" in result.stderr


def test_preflight_failure_is_fail_closed_and_does_not_create_output(tmp_path):
    env = _env(tmp_path)
    env["FAKE_PREFLIGHT_EXIT"] = "9"

    result = _run("dry-run", env=env)

    assert result.returncode != 0
    assert "config preflight failed" in result.stderr
    assert not Path(env["STAGE1_OUTPUT"]).exists()


def test_run_creates_output_only_after_preflight_and_executes_fake_torchrun(tmp_path):
    env = _env(tmp_path)

    result = _run("run", "--steps", "20", "--save-interval", "10", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    output = Path(env["STAGE1_OUTPUT"])
    assert output.is_dir()
    log = Path(env["TORCHRUN_LOG"]).read_text(encoding="utf-8")
    assert "--nproc_per_node=8" in log
    assert "--master_port=29671" in log
    assert "CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage1" in log
    assert f"STUDENT_BASE_MODEL_PATH={env['CLEAN_STUDENT_BASE_MODEL_PATH']}" in log
