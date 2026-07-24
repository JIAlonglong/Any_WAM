"""RED launcher contracts for the independent Cosmos LIBERO Task-7 arms.

Copy this file to ``distillation_flowmap/tests/test_run_cosmos_libero_train_8gpu.py``
before implementing ``run_cosmos_libero_train_8gpu.sh``.  The tests use only
small filesystem fixtures and dry/check-only invocations: no GPU process is
started and no existing experiment output is modified.
"""

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from distillation_flowmap.cosmos_libero_variants import canonical_variant_json, resolve_variant
from distillation_flowmap.cosmos_stage2_lineage import validate_stage1_parent


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "distillation_flowmap" / "run_cosmos_libero_train_8gpu.sh"
DEVICES = "0,1,2,3,4,5,6,7"
ARMS = ("s1", "s2", "s4", "universal", "universal-video-action", "stage1_only", "anchor_only", "field_only", "apm")
RAW_CONTRACT = {
    "contract_version": 2,
    "training_contract_stage": "raw_stage1",
    "action_packing_schema": "downsample_survivor_v2",
    "action_downsample_factor": 4,
    "action_chunk_shape": [4, 4],
    "checkpoint_step": 5000,
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
    "teacher_backend": "cosmos_policy",
}


def _transformer(root: Path, payload: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"test")
    return root


def _layout(tmp_path: Path):
    stage1 = tmp_path / "stage1"
    for variant in ("online_student", "target_student"):
        _transformer(stage1 / variant / "transformer", RAW_CONTRACT)
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
    return stage1, dataset, policy, repo, local_model, worker_python


def _env(tmp_path: Path):
    stage1, dataset, policy, repo, local_model, worker_python = _layout(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "COSMOS_STAGE1_ROOT": str(stage1),
            "STUDENT_BASE_MODEL_PATH": str(stage1 / "target_student"),
            "DATASET_PATH": str(dataset),
            "COSMOS_POLICY_PATH": str(policy),
            "COSMOS_POLICY_PYTHON": str(worker_python),
            "COSMOS_PREDICT2_REPO": str(repo),
            "COSMOS_PREDICT25_LOCAL_MODEL_DIR": str(local_model),
            "CUDA_VISIBLE_DEVICES": DEVICES,
            "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES": DEVICES,
        }
    )
    return env, stage1


def _run(*args: str, env: dict[str, str], cwd: Path | None = None):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        text=True,
        capture_output=True,
        cwd=cwd or ROOT,
        env=env,
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


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_has_one_complete_write_free_eight_gpu_dry_run(tmp_path, arm):
    env, _ = _env(tmp_path)
    output_root = tmp_path / "uncreated-output-root"
    result = _run(
        arm,
        "--steps", "123",
        "--save-interval", "37",
        "--master-port", "32123",
        "--output-root", str(output_root),
        "--run-tag", "contract-red",
        "--dry-run",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    values = _assignments(result.stdout)
    expected = json.loads(
        canonical_variant_json(
            resolve_variant(
                arm,
                output_root=output_root,
                run_tag="contract-red",
                steps=123,
                save_interval=37,
                master_port=32123,
            )
        )
    )
    assert values["OUTPUT_DIR"] == expected["output_dir"]
    assert values["MAX_TRAIN_STEPS"] == "123"
    assert values["SAVE_INTERVAL"] == "37"
    assert values["MASTER_PORT"] == "32123"
    assert json.loads(values["COSMOS_LIBERO_VARIANT_JSON"]) == expected
    assert values["ENABLE_TENSORBOARD"] == "1"
    assert values["ENABLE_WANDB"] == "0"
    assert values["WANDB_MODE"] == "offline"
    assert values["HF_DATASETS_OFFLINE"] == "1"
    assert values["TRANSFORMERS_OFFLINE"] == "1"
    assert values["HF_HUB_OFFLINE"] == "1"
    assert values["STUDENT_BASE_MODEL_PATH"] == env["STUDENT_BASE_MODEL_PATH"]
    assert values["RESUME_FROM_PATH"] == env["COSMOS_STAGE1_ROOT"]
    assert values["PARENT_STAGE1_PATH"] == str(Path(env["COSMOS_STAGE1_ROOT"]).resolve())
    assert result.stdout.count("--nproc_per_node=8") == 1
    assert "distillation_flowmap/train.py" in result.stdout
    assert not output_root.exists()


def test_launcher_supports_check_only_without_claiming_output_or_running_torchrun(tmp_path):
    env, _ = _env(tmp_path)
    output_root = tmp_path / "check-only-output"
    fake_torchrun = tmp_path / "torchrun"
    marker = tmp_path / "torchrun-was-called"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\ntouch '" + str(marker) + "'\nexit 99\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(fake_torchrun.stat().st_mode | stat.S_IXUSR)
    env["TORCHRUN_BIN"] = str(fake_torchrun)

    result = _run(
        "apm", "--output-root", str(output_root), "--run-tag", "check", "--check-only", env=env
    )

    assert result.returncode == 0, result.stderr
    assert "--nproc_per_node=8" in result.stdout
    assert not output_root.exists()
    assert not marker.exists()


def test_formal_fresh_run_claims_one_arm_and_persists_canonical_manifest_before_exec(
    tmp_path,
):
    env, _ = _env(tmp_path)
    output_root = tmp_path / "formal-output"
    marker = tmp_path / "torchrun-env.json"
    fake_torchrun = tmp_path / "torchrun"
    fake_torchrun.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$COSMOS_LIBERO_VARIANT_JSON\" > '" + str(marker) + "'\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_torchrun.chmod(fake_torchrun.stat().st_mode | stat.S_IXUSR)
    env["TORCHRUN_BIN"] = str(fake_torchrun)

    result = _run(
        "field_only",
        "--output-root",
        str(output_root),
        "--run-tag",
        "formal",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    output = output_root / "formal" / "field_only"
    expected = canonical_variant_json(
        resolve_variant(
            "field_only", output_root=output_root, run_tag="formal"
        )
    )
    assert marker.read_text(encoding="utf-8").strip() == expected
    assert (
        output / "cosmos_libero_variant.json"
    ).read_text(encoding="utf-8").strip() == expected
    assert not (output_root / "formal" / "apm").exists()


def test_default_save_interval_is_exactly_1000(tmp_path):
    env, _ = _env(tmp_path)
    result = _run(
        "universal", "--output-root", str(tmp_path / "out"), "--run-tag", "default-save", "--dry-run", env=env
    )
    assert result.returncode == 0, result.stderr
    assert _assignments(result.stdout)["SAVE_INTERVAL"] == "1000"


@pytest.mark.parametrize("switch", ("--dry-run", "--check-only"))
def test_dry_and_check_only_do_lineage_preflight_but_never_create_output_parent(
    tmp_path, switch
):
    env, _ = _env(tmp_path)
    output_root = tmp_path / "output-root"
    result = _run(
        "s4", "--output-root", str(output_root), "--run-tag", "no-write", switch, env=env
    )
    assert result.returncode == 0, result.stderr
    assert "PARENT_STAGE1_CONTRACT_IDENTITY=" in result.stdout
    assert not output_root.exists()


@pytest.mark.parametrize("arm", ("anchor_only", "apm"))
def test_formal_run_rejects_existing_or_symlinked_selected_arm_before_torchrun(tmp_path, arm):
    env, _ = _env(tmp_path)
    output_root = tmp_path / "out"
    output = output_root / "existing" / arm
    output.mkdir(parents=True)
    result = _run(arm, "--output-root", str(output_root), "--run-tag", "existing", env=env)
    assert result.returncode != 0
    assert "existing" in result.stderr.lower() or "claim" in result.stderr.lower()

    linked_root = tmp_path / "linked-output"
    (linked_root / "existing").mkdir(parents=True)
    real = tmp_path / "real-output"
    real.mkdir()
    (linked_root / "existing" / arm).symlink_to(real, target_is_directory=True)
    result = _run(arm, "--output-root", str(linked_root), "--run-tag", "existing", env=env)
    assert result.returncode != 0
    assert "symlink" in result.stderr.lower() or "existing" in result.stderr.lower()


def test_dry_run_rejects_a_symlinked_output_root_instead_of_canonicalizing_alias(
    tmp_path,
):
    env, _ = _env(tmp_path)
    real_root = tmp_path / "real-output-root"
    real_root.mkdir()
    alias_root = tmp_path / "output-alias"
    alias_root.symlink_to(real_root, target_is_directory=True)

    result = _run(
        "apm",
        "--output-root",
        str(alias_root),
        "--run-tag",
        "alias",
        "--dry-run",
        env=env,
    )

    assert result.returncode != 0
    assert "symlink" in result.stderr.lower()
    assert not (real_root / "alias").exists()


def _write_resume_checkpoint(output: Path, *, arm: str, step: int, parent, variant_json: str):
    checkpoint = output / "checkpoints" / f"step_{step}"
    payload = {
        **STAGE2_CONTRACT,
        "checkpoint_step": step,
        "parent_stage1_path": parent.canonical_path,
        "parent_stage1_contract_identity": parent.contract_identity,
        "cosmos_libero_variant_json": variant_json,
    }
    for student in ("online_student", "target_student"):
        _transformer(checkpoint / student / "transformer", payload)
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "lr_scheduler.pt").write_bytes(b"scheduler")
    return checkpoint


def test_resume_is_allowed_only_from_same_arm_with_same_parent_and_variant_record(tmp_path):
    env, stage1 = _env(tmp_path)
    output_root = tmp_path / "out"
    own = output_root / "resume" / "apm"
    parent = validate_stage1_parent(stage1)
    own_record = resolve_variant("apm", output_root=output_root, run_tag="resume")
    _write_resume_checkpoint(
        own,
        arm="apm",
        step=100,
        parent=parent,
        variant_json=canonical_variant_json(own_record),
    )
    result = _run(
        "apm", "--output-root", str(output_root), "--run-tag", "resume", "--resume-step", "100", "--dry-run", env=env
    )
    assert result.returncode == 0, result.stderr
    assert _assignments(result.stdout)["RESUME_FROM_PATH"] == str(own / "checkpoints" / "step_100")

    other_record = resolve_variant("anchor_only", output_root=output_root, run_tag="resume")
    for student in ("online_student", "target_student"):
        config = own / "checkpoints" / "step_100" / student / "transformer" / "config.json"
        payload = json.loads(config.read_text(encoding="utf-8"))
        payload["cosmos_libero_variant_json"] = canonical_variant_json(other_record)
        config.write_text(json.dumps(payload), encoding="utf-8")
    result = _run(
        "apm", "--output-root", str(output_root), "--run-tag", "resume", "--resume-step", "100", "--dry-run", env=env
    )
    assert result.returncode != 0
    assert "variant" in result.stderr.lower()


@pytest.mark.parametrize(
    "args",
    [
        ("unknown", "--dry-run"),
        ("apm", "--steps", "0", "--dry-run"),
        ("apm", "--save-interval", "0", "--dry-run"),
        ("apm", "--master-port", "65536", "--dry-run"),
        ("apm", "--run-tag", "../escape", "--dry-run"),
        ("apm", "--unknown", "--dry-run"),
    ],
)
def test_invalid_cli_input_fails_before_output_creation(tmp_path, args):
    env, _ = _env(tmp_path)
    output_root = tmp_path / "missing-output"
    result = _run(*args[:-1], "--output-root", str(output_root), args[-1], env=env)
    assert result.returncode != 0
    assert not output_root.exists()


def test_missing_explicit_stage1_or_base_path_has_no_fallback(tmp_path):
    env, _ = _env(tmp_path)
    env.pop("COSMOS_STAGE1_ROOT")
    result = _run("apm", "--dry-run", env=env)
    assert result.returncode != 0
    assert "COSMOS_STAGE1_ROOT" in result.stderr

    env, _ = _env(tmp_path / "second")
    env.pop("STUDENT_BASE_MODEL_PATH")
    result = _run("apm", "--dry-run", env=env)
    assert result.returncode != 0
    assert "STUDENT_BASE_MODEL_PATH" in result.stderr


def test_launcher_accepts_task4_validated_sharded_stage1_weights(tmp_path):
    env, stage1 = _env(tmp_path)
    for student in ("online_student", "target_student"):
        transformer = stage1 / student / "transformer"
        (transformer / "diffusion_pytorch_model.safetensors").unlink()
        shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
        (transformer / shard).write_bytes(b"sharded")
        (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": shard}}),
            encoding="utf-8",
        )

    result = _run(
        "universal",
        "--output-root",
        str(tmp_path / "out"),
        "--run-tag",
        "sharded",
        "--dry-run",
        env=env,
    )

    assert result.returncode == 0, result.stderr


def test_launcher_and_resolved_commands_have_no_lingbot_or_root_nas_fallback():
    source = SCRIPT.read_text(encoding="utf-8").lower()
    assert "lingbot" not in source
    assert "/root/nas" not in source
