import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from distillation_flowmap.cosmos_stage2_lineage import validate_stage1_parent


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "distillation_flowmap" / "run_cosmos_progressive_stage2_8gpu.sh"
DEVICES = "0,1,2,3,4,5,6,7"
RAW_CONTRACT = {
    "contract_version": 2,
    "training_contract_stage": "raw_stage1",
    "action_packing_schema": "downsample_survivor_v2",
    "action_downsample_factor": 4,
    "action_chunk_shape": [4, 4],
    "checkpoint_step": 5000,
    "student_backend": "wan_flowmap",
    "teacher_backend": "cosmos_policy",
}
STAGE2_CONTRACT = {
    "contract_version": 2,
    "training_contract_stage": "progressive_stage2",
    "action_packing_schema": "downsample_survivor_v2",
    "action_downsample_factor": 4,
    "action_chunk_shape": [4, 4],
    "deployment_timestep_start": 1000,
    "deployment_timestep_end": 0,
    "joint_student_steps": [1, 2, 4],
    "deployment_joint_rollout_interval": 4,
    "deployment_action_weight": 1.0,
    "raw_teacher_window_is_auxiliary": True,
    "student_backend": "wan_flowmap",
    "teacher_backend": "cosmos_policy",
}


def _transformer(root: Path, payload: dict | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(payload or RAW_CONTRACT) + "\n", encoding="utf-8"
    )
    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"test")
    return root


def _layout(tmp_path: Path, *, parent_step: int = 5000):
    raw_contract = {**RAW_CONTRACT, "checkpoint_step": parent_step}
    stage1 = tmp_path / "stage1"
    for variant in ("online_student", "target_student"):
        _transformer(stage1 / variant / "transformer", raw_contract)

    output = tmp_path / "output"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "empty_emb.pt").write_bytes(b"test")

    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "config.json").write_text(
        '{"model_type":"cosmos-policy"}\n', encoding="utf-8"
    )
    (policy / "libero_dataset_statistics.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (policy / "Cosmos-Policy-LIBERO-Predict2-2B.pt").write_bytes(b"policy")
    (policy / "libero_t5_embeddings.pkl").write_bytes(b"embeddings")
    stage1_payload = {
        **raw_contract,
        "_class_name": "WanTransformer3DModel",
        "student_base_model_path": str(stage1 / "target_student"),
        "teacher_model_path": str(policy),
    }
    for variant in ("online_student", "target_student"):
        _transformer(stage1 / variant / "transformer", stage1_payload)
    repo = tmp_path / "cosmos-predict2.5"
    repo.mkdir()
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    worker_python = tmp_path / "cosmos-python"
    worker_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    worker_python.chmod(worker_python.stat().st_mode | stat.S_IXUSR)
    return stage1, output, dataset, policy, repo, local_model, worker_python


def _env(tmp_path: Path, *, parent_step: int = 5000):
    stage1, output, dataset, policy, repo, local_model, worker_python = _layout(
        tmp_path, parent_step=parent_step
    )
    env = os.environ.copy()
    env.update(
        {
            "COSMOS_STAGE1_ROOT": str(stage1),
            "STUDENT_BASE_MODEL_PATH": str(stage1 / "target_student"),
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


def _resume_checkpoint(env, output: Path, stage: str, step: int) -> Path:
    checkpoint = output / stage / "checkpoints" / f"step_{step}"
    parent = validate_stage1_parent(Path(env["COSMOS_STAGE1_ROOT"]))
    payload = {
        **STAGE2_CONTRACT,
        "checkpoint_step": step,
        "parent_stage1_path": parent.canonical_path,
        "parent_stage1_contract_identity": parent.contract_identity,
        "student_base_model_path": str(
            Path(parent.canonical_path) / "target_student"
        ),
        "teacher_model_path": str(
            json.loads(
                (
                    Path(parent.canonical_path)
                    / "target_student"
                    / "transformer"
                    / "config.json"
                ).read_text(encoding="utf-8")
            )["teacher_model_path"]
        ),
    }
    for variant in ("online_student", "target_student"):
        _transformer(checkpoint / variant / "transformer", payload)
    (checkpoint / "optimizer.pt").write_bytes(b"test")
    (checkpoint / "lr_scheduler.pt").write_bytes(b"test")
    return checkpoint


def _run(*args: str, env: dict[str, str], cwd: Path | None = None):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        text=True,
        capture_output=True,
        env=env,
        cwd=cwd or ROOT,
        check=False,
    )


def _assignments(stdout: str) -> dict[str, str]:
    return {
        key: value
        for line in stdout.splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
        if key.isupper()
    }


def test_stage2_launcher_accepts_explicit_stage1_step_3000(tmp_path):
    env, _ = _env(tmp_path, parent_step=3000)
    env["COSMOS_STAGE1_EXPECTED_STEP"] = "3000"

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assert "COSMOS_STAGE1_EXPECTED_STEP=3000" in result.stdout


def test_stage2_launcher_defaults_parent_step_to_5000(tmp_path):
    env, _ = _env(tmp_path, parent_step=5000)
    env.pop("COSMOS_STAGE1_EXPECTED_STEP", None)

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assert "COSMOS_STAGE1_EXPECTED_STEP=5000" in result.stdout


def test_stage2_launcher_rejects_parent_metadata_that_disagrees(tmp_path):
    env, _ = _env(tmp_path, parent_step=5000)
    env["COSMOS_STAGE1_EXPECTED_STEP"] = "3000"

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "step" in result.stderr.lower()


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
    assert "MECHANISM_DIAGNOSTICS=1" in result.stdout
    assert "MECHANISM_DIAGNOSTIC_INTERVAL=100" in result.stdout
    assert "MECHANISM_DIAGNOSTIC_SEED=42" in result.stdout
    assert "MECHANISM_DIAGNOSTIC_R=500" in result.stdout
    assert "MECHANISM_DIAGNOSTIC_S=250" in result.stdout
    assert "MECHANISM_DIAGNOSTIC_TEACHER_STEPS=8" in result.stdout
    assert "MECHANISM_COSMOS_T_MIN=0.8" in result.stdout
    assert "MECHANISM_COSMOS_T_MAX=0.9876543209876543" in result.stdout
    assert "ENABLE_WANDB=1" in result.stdout
    assert "WANDB_MODE=offline" in result.stdout
    assert f"--master_port={port}" in result.stdout
    assert not (output / stage).exists()


def test_dual_universal_video_modes_are_a_paired_cosmos_loss_ablation(tmp_path):
    env, output = _env(tmp_path)
    resolved = {}
    for mode in ("universal-video", "universal-video-action"):
        result = _run(mode, "--dry-run", env=env)

        assert result.returncode == 0, result.stderr
        resolved[mode] = _assignments(result.stdout)

    video = resolved["universal-video"]
    video_action = resolved["universal-video-action"]
    assert video["COSMOS_PROGRESSIVE_STAGE"] == "universal"
    assert video_action["COSMOS_PROGRESSIVE_STAGE"] == "universal"
    assert video["MAX_TRAIN_STEPS"] == video_action["MAX_TRAIN_STEPS"] == "5000"
    assert video["TRAIN_SEED"] == video_action["TRAIN_SEED"] == "42"
    assert video["OUTPUT_DIR"] == str(output / "universal-video")
    assert video_action["OUTPUT_DIR"] == str(output / "universal-video-action")
    assert video["MASTER_PORT"] == "29665"
    assert video_action["MASTER_PORT"] == "29666"
    assert video["OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT"] == "0.0"
    assert video_action["OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT"] == "1.0"
    assert video["OPD_AUX_ACTION"] == video_action["OPD_AUX_ACTION"] == "0"
    assert (
        video["COSMOS_USE_TEACHER_ACTION_ANCHOR"]
        == video_action["COSMOS_USE_TEACHER_ACTION_ANCHOR"]
        == "1"
    )
    assert (
        video["OPD_JOINT_ACTION_ROLLOUT"]
        == video_action["OPD_JOINT_ACTION_ROLLOUT"]
        == "1"
    )
    assert not (output / "universal-video").exists()
    assert not (output / "universal-video-action").exists()

    paired_keys = set(video) | set(video_action)
    paired_keys -= {
        "OUTPUT_DIR",
        "MASTER_PORT",
        "OPD_DANCEOPD_ACTION_ENDPOINT_WEIGHT",
    }
    for key in paired_keys:
        assert video[key] == video_action[key]


def test_s4_dry_run_rejects_missing_online_student(tmp_path):
    env, _ = _env(tmp_path)
    shutil.rmtree(Path(env["COSMOS_STAGE1_ROOT"]) / "online_student")

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "online_student" in result.stderr


def test_launcher_rejects_missing_explicit_cosmos_base_config(tmp_path):
    env, _ = _env(tmp_path)
    (
        Path(env["STUDENT_BASE_MODEL_PATH"]) / "transformer" / "config.json"
    ).unlink()

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "config.json" in result.stderr


def test_launcher_rejects_non_cosmos_backend(tmp_path):
    env, _ = _env(tmp_path)
    for variant in ("online_student", "target_student"):
        config = (
            Path(env["COSMOS_STAGE1_ROOT"])
            / variant
            / "transformer"
            / "config.json"
        )
        payload = json.loads(config.read_text())
        payload["teacher_backend"] = "wanva"
        config.write_text(json.dumps(payload))

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "cosmos_policy" in result.stderr


def test_launcher_rejects_base_model_identity_mismatch(tmp_path):
    env, _ = _env(tmp_path)
    other = tmp_path / "other-base"
    _transformer(other / "transformer")
    env["STUDENT_BASE_MODEL_PATH"] = str(other)

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "STUDENT_BASE_MODEL_PATH" in result.stderr


@pytest.mark.parametrize("relation", ["equal", "nested", "contains", "symlink"])
def test_launcher_rejects_unsafe_output_relation(tmp_path, relation):
    env, _ = _env(tmp_path)
    stage1 = Path(env["COSMOS_STAGE1_ROOT"])
    if relation == "equal":
        env["OUTPUT_DIR"] = str(stage1)
    elif relation == "nested":
        env["OUTPUT_DIR"] = str(stage1 / "stage2")
    elif relation == "contains":
        env["OUTPUT_DIR"] = str(tmp_path)
    else:
        real = tmp_path / "real-output"
        real.mkdir()
        alias = tmp_path / "output-alias"
        alias.symlink_to(real, target_is_directory=True)
        env["OUTPUT_DIR"] = str(alias)

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "isolat" in result.stderr.lower() or "symlink" in result.stderr.lower()


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


def test_dry_run_rejects_existing_empty_fresh_output(tmp_path):
    env, output = _env(tmp_path)
    (output / "s4" / "checkpoints").mkdir(parents=True)

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "existing OUTPUT_DIR" in result.stderr


def test_dry_run_rejects_nonempty_fresh_checkpoint_root(tmp_path):
    env, output = _env(tmp_path)
    checkpoints = output / "s4" / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / "step_1000").mkdir()

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "Refusing fresh run" in result.stderr


def test_fresh_run_exports_student_base_to_torchrun(tmp_path):
    env, _ = _env(tmp_path)
    capture = tmp_path / "student-base.txt"
    fake_torchrun = tmp_path / "fake-torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\nprintf '%s' \"$STUDENT_BASE_MODEL_PATH\" > \"$STUDENT_BASE_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(fake_torchrun.stat().st_mode | stat.S_IXUSR)
    env["TORCHRUN_BIN"] = str(fake_torchrun)
    env["STUDENT_BASE_CAPTURE"] = str(capture)

    result = _run("s2", env=env)

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8") == str(
        Path(env["COSMOS_STAGE1_ROOT"]) / "target_student"
    )


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
    resume = _resume_checkpoint(env, output, "s4", 1000)

    result = _run("s4", "--resume-step", "1000", "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assert f"RESUME_FROM_PATH={resume}" in result.stdout
    assert "RESET_RESUME_STEP=0" in result.stdout
    assert "RESUME_OPTIMIZER_STATE=1" in result.stdout
    assert "RESUME_ONLINE_FROM_TARGET=0" in result.stdout


def test_resume_step_rejects_missing_optimizer(tmp_path):
    env, output = _env(tmp_path)
    resume = _resume_checkpoint(env, output, "s4", 1000)
    (resume / "optimizer.pt").unlink()

    result = _run("s4", "--resume-step", "1000", "--dry-run", env=env)

    assert result.returncode != 0
    assert "optimizer.pt" in result.stderr


@pytest.mark.parametrize("missing", ["target_student", "lr_scheduler.pt"])
def test_resume_rejects_missing_target_or_scheduler(tmp_path, missing):
    env, output = _env(tmp_path)
    resume = _resume_checkpoint(env, output, "s4", 1000)
    path = resume / missing
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

    result = _run("s4", "--resume-step", "1000", "--dry-run", env=env)

    assert result.returncode != 0
    assert missing in result.stderr


@pytest.mark.parametrize("mismatch", ["step", "parent"])
def test_resume_rejects_step_or_parent_identity_mismatch(tmp_path, mismatch):
    env, output = _env(tmp_path)
    resume = _resume_checkpoint(env, output, "s4", 1000)
    for variant in ("online_student", "target_student"):
        config = resume / variant / "transformer" / "config.json"
        payload = json.loads(config.read_text())
        if mismatch == "step":
            payload["checkpoint_step"] = 999
        else:
            payload["parent_stage1_contract_identity"] = "0" * 64
        config.write_text(json.dumps(payload))

    result = _run("s4", "--resume-step", "1000", "--dry-run", env=env)

    assert result.returncode != 0
    expected = "checkpoint_step" if mismatch == "step" else "parent"
    assert expected in result.stderr.lower()


def test_dry_run_emits_lineage_and_writes_nothing(tmp_path):
    env, output = _env(tmp_path)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assignments = _assignments(result.stdout)
    assert assignments["PARENT_STAGE1_PATH"] == str(
        Path(env["COSMOS_STAGE1_ROOT"]).resolve()
    )
    assert len(assignments["PARENT_STAGE1_CONTRACT_IDENTITY"]) == 64
    assert assignments["STAGE2_LINEAGE_JSON"].startswith("{")
    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert after == before


def test_dry_run_allows_explicit_wandb_opt_out_but_keeps_offline_mode(tmp_path):
    env, _ = _env(tmp_path)
    env["ENABLE_WANDB"] = "0"

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assignments = _assignments(result.stdout)
    assert assignments["ENABLE_WANDB"] == "0"
    assert assignments["WANDB_MODE"] == "offline"


def test_launcher_rejects_invalid_wandb_toggle(tmp_path):
    env, output = _env(tmp_path)
    env["ENABLE_WANDB"] = "sometimes"

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "ENABLE_WANDB must be 0 or 1" in result.stderr
    assert not output.exists()


def test_relative_output_is_canonical_for_preflight_claim_and_torchrun(tmp_path):
    env, _ = _env(tmp_path)
    caller = tmp_path / "caller"
    caller.mkdir()
    relative_output = "relative-stage2"
    expected = (caller / relative_output).resolve()
    capture = tmp_path / "torchrun-output.txt"
    fake_torchrun = tmp_path / "fake-torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n%s\\n' \"$PWD\" \"$OUTPUT_DIR\" > \"$TORCHRUN_CAPTURE\"\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(fake_torchrun.stat().st_mode | stat.S_IXUSR)
    env.update(
        {
            "OUTPUT_DIR": relative_output,
            "TORCHRUN_BIN": str(fake_torchrun),
            "TORCHRUN_CAPTURE": str(capture),
        }
    )

    result = _run("s4", env=env, cwd=caller)

    assert result.returncode == 0, result.stderr
    assignments = _assignments(result.stdout)
    assert assignments["OUTPUT_DIR"] == str(expected)
    assert expected.is_dir()
    capture_lines = capture.read_text().splitlines()
    assert capture_lines[0] == str(ROOT)
    assert capture_lines[1] == str(expected)


def test_config_preflight_failure_occurs_before_fresh_output_claim(tmp_path):
    env, output = _env(tmp_path)
    marker = tmp_path / "torchrun-called"
    fake_torchrun = tmp_path / "fake-torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "touch \"$TORCHRUN_MARKER\"\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(fake_torchrun.stat().st_mode | stat.S_IXUSR)
    env.update(
        {
            "TORCHRUN_BIN": str(fake_torchrun),
            "TORCHRUN_MARKER": str(marker),
            "OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE": "nan",
        }
    )

    result = _run("s4", env=env)

    assert result.returncode != 0
    assert "config preflight" in result.stderr
    assert not (output / "s4").exists()
    assert not marker.exists()
