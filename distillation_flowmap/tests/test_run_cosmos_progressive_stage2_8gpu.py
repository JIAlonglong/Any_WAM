import os
import stat
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "distillation_flowmap" / "run_cosmos_progressive_stage2_8gpu.sh"
DEVICES = "0,1,2,3,4,5,6,7"


def _transformer(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"checkpoint_step": 1000}\n', encoding="utf-8")
    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"test")
    return root


def _layout(tmp_path: Path):
    stage1 = tmp_path / "stage1"
    _transformer(stage1 / "target_student" / "transformer")
    (stage1 / "online_student").symlink_to("target_student", target_is_directory=True)

    output = tmp_path / "output"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "empty_emb.pt").write_bytes(b"test")

    policy = tmp_path / "policy"
    policy.mkdir()
    repo = tmp_path / "cosmos-predict2.5"
    repo.mkdir()
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    worker_python = tmp_path / "cosmos-python"
    worker_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    worker_python.chmod(worker_python.stat().st_mode | stat.S_IXUSR)
    return stage1, output, dataset, policy, repo, local_model, worker_python


def _env(tmp_path: Path):
    stage1, output, dataset, policy, repo, local_model, worker_python = _layout(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "COSMOS_STAGE1_ROOT": str(stage1),
            "OUTPUT_ROOT": str(output),
            "DATASET_PATH": str(dataset),
            "COSMOS_POLICY_PATH": str(policy),
            "COSMOS_POLICY_PYTHON": str(worker_python),
            "COSMOS_PREDICT2_REPO": str(repo),
            "COSMOS_PREDICT25_LOCAL_MODEL_DIR": str(local_model),
            "CUDA_VISIBLE_DEVICES": DEVICES,
            "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES": DEVICES,
        }
    )
    return env, output


def _run(*args: str, env: dict[str, str]):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


@pytest.mark.parametrize(
    ("stage", "steps", "port"),
    [
        ("s1", "3000", "29663"),
        ("s2", "3000", "29662"),
        ("s4", "5000", "29661"),
        ("universal", "5000", "29664"),
    ],
)
def test_dry_run_prints_independent_stage_contract_without_creating_output(
    tmp_path, stage, steps, port
):
    env, output = _env(tmp_path)
    result = _run(stage, "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assert f"MAX_TRAIN_STEPS={steps}" in result.stdout
    assert "RESUME_FROM_PATH=" + env["COSMOS_STAGE1_ROOT"] in result.stdout
    assert (
        "STUDENT_BASE_MODEL_PATH="
        + str(Path(env["COSMOS_STAGE1_ROOT"]) / "target_student")
    ) in result.stdout
    assert "RESUME_ONLINE_FROM_TARGET=1" in result.stdout
    assert "RESET_RESUME_STEP=1" in result.stdout
    assert "RESUME_OPTIMIZER_STATE=0" in result.stdout
    assert f"--master_port={port}" in result.stdout
    assert not (output / stage).exists()


def test_s4_dry_run_rejects_missing_target_to_online_compatibility_link(tmp_path):
    env, _ = _env(tmp_path)
    Path(env["COSMOS_STAGE1_ROOT"]).joinpath("online_student").unlink()

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "online_student/transformer" in result.stderr


def test_dry_run_rejects_non_eight_gpu_layout(tmp_path):
    env, _ = _env(tmp_path)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "exactly 8" in result.stderr


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,6", "duplicate"),
        ("COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,6", "duplicate"),
        ("COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,8", "must equal"),
    ],
)
def test_dry_run_rejects_invalid_or_mismatched_worker_gpu_layout(
    tmp_path, variable, value, message
):
    env, _ = _env(tmp_path)
    env[variable] = value

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert message in result.stderr


def test_dry_run_rejects_existing_fresh_checkpoint_root(tmp_path):
    env, output = _env(tmp_path)
    (output / "s4" / "checkpoints").mkdir(parents=True)

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "Refusing fresh run" in result.stderr


def test_dry_run_rejects_invalid_port_and_unknown_stage(tmp_path):
    env, _ = _env(tmp_path)

    bad_port = _run("s4", "--master-port", "70000", "--dry-run", env=env)
    unknown_stage = _run("s8", "--dry-run", env=env)

    assert bad_port.returncode != 0
    assert "invalid master port" in bad_port.stderr
    assert unknown_stage.returncode != 0
    assert "stage must be one of" in unknown_stage.stderr


def test_resume_step_uses_own_online_checkpoint_and_optimizer(tmp_path):
    env, output = _env(tmp_path)
    resume = output / "s4" / "checkpoints" / "step_1000"
    _transformer(resume / "online_student" / "transformer")
    (resume / "optimizer.pt").write_bytes(b"test")

    result = _run("s4", "--resume-step", "1000", "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assert f"RESUME_FROM_PATH={resume}" in result.stdout
    assert "RESET_RESUME_STEP=0" in result.stdout
    assert "RESUME_OPTIMIZER_STATE=1" in result.stdout
    assert "RESUME_ONLINE_FROM_TARGET=0" in result.stdout


def test_resume_step_rejects_missing_optimizer(tmp_path):
    env, output = _env(tmp_path)
    resume = output / "s4" / "checkpoints" / "step_1000"
    _transformer(resume / "online_student" / "transformer")

    result = _run("s4", "--resume-step", "1000", "--dry-run", env=env)

    assert result.returncode != 0
    assert "resume_optimizer" in result.stderr
