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
CUDA_LIBRARY_RELATIVES = (
    "nvidia/cublas/lib",
    "nvidia/cuda_cupti/lib",
    "nvidia/cuda_nvrtc/lib",
    "nvidia/cuda_runtime/lib",
    "nvidia/cudnn/lib",
    "nvidia/cufft/lib",
    "nvidia/cufile/lib",
    "nvidia/curand/lib",
    "nvidia/cusolver/lib",
    "nvidia/cusparse/lib",
    "nvidia/nccl/lib",
    "nvidia/nvjitlink/lib",
    "nvidia/nvtx/lib",
)
RAW_CONTRACT = {
    "contract_version": 2,
    "training_contract_stage": "raw_stage1",
    "action_packing_schema": "downsample_survivor_v2",
    "action_downsample_factor": 4,
    "action_chunk_shape": [4, 4],
    "student_backend": "wan_flowmap",
    "teacher_backend": "cosmos_policy",
}


def _write_executable(path: Path, body: str) -> None:
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _transformer(root: Path, *, step: int | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payload = {"_class_name": "WanTransformer3DModel"}
    if step is not None:
        payload.update({**RAW_CONTRACT, "checkpoint_step": step})
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
    (policy / "config.json").write_text(
        json.dumps({"model_type": "cosmos-policy"}) + "\n",
        encoding="utf-8",
    )
    for name in (
        "Cosmos-Policy-LIBERO-Predict2-2B.pt",
        "libero_dataset_statistics.json",
        "libero_t5_embeddings.pkl",
    ):
        (policy / name).write_bytes(b"fixture")
    worker_repo = tmp_path / "cosmos-predict2.5"
    worker_repo.mkdir()
    for relative in ("packages/cosmos-cuda", "packages/cosmos-oss"):
        package = worker_repo / relative
        package.mkdir(parents=True)
        (package / ".fixture").write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(worker_repo)], check=True)
    subprocess.run(
        ["git", "-C", str(worker_repo), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(worker_repo), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(worker_repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(worker_repo), "commit", "-qm", "fixture"],
        check=True,
    )
    local_model = tmp_path / "cosmos-local-model"
    local_model.mkdir()
    worker_env = tmp_path / "cosmos-worker-env"
    worker_site_packages = worker_env / "lib/python3.10/site-packages"
    for relative in CUDA_LIBRARY_RELATIVES:
        (worker_site_packages / relative).mkdir(parents=True)
    worker_python = tmp_path / "cosmos-python"
    _write_executable(
        worker_python,
        'printf "worker-import-preflight\\n" >> "$CALLS_LOG"\n'
        'printf "%s\\n" "$*" > "$WORKER_PREFLIGHT_LOG"\n'
        'printf "PYTHONPATH=%s\\n" "${PYTHONPATH:-}" >> "$WORKER_PREFLIGHT_LOG"\n'
        'printf "LD_LIBRARY_PATH=%s\\n" "${LD_LIBRARY_PATH:-}" >> "$WORKER_PREFLIGHT_LOG"\n'
        'printf "PYTHONDONTWRITEBYTECODE=%s\\n" "${PYTHONDONTWRITEBYTECODE:-}" >> "$WORKER_PREFLIGHT_LOG"\n'
        '[[ "$*" == *"import cosmos_cuda, cosmos_predict2; from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import get_action"* ]] || exit 91\n'
        'exit "${FAKE_WORKER_IMPORT_EXIT:-0}"\n',
    )
    preflight = tmp_path / "fake-preflight"
    _write_executable(
        preflight,
        'printf "config-preflight\\n" >> "$CALLS_LOG"\n'
        'printf "%s\\n" "$*" >> "$PREFLIGHT_LOG"\n'
        'if [[ "${CLAIM_OUTPUT_DURING_PREFLIGHT:-0}" == 1 ]]; then\n'
        '    mkdir -p "$OUTPUT_DIR"\n'
        "fi\n"
        'exit "${FAKE_PREFLIGHT_EXIT:-0}"\n',
    )
    torchrun = tmp_path / "fake-torchrun"
    _write_executable(
        torchrun,
        'printf "torchrun\\n" >> "$CALLS_LOG"\n'
        'printf "%s\\n" "$*" > "$TORCHRUN_LOG"\n'
        "env | LC_ALL=C sort >> \"$TORCHRUN_LOG\"\n",
    )
    return {
        "clean_base": str(clean_base),
        "dataset": str(dataset),
        "policy": str(policy),
        "worker_repo": str(worker_repo),
        "local_model": str(local_model),
        "worker_env": str(worker_env),
        "worker_site_packages": str(worker_site_packages),
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
            "COSMOS_WORKER_ENV_ROOT": paths["worker_env"],
            "PYTHON_BIN": str(PYTHON),
            "PREFLIGHT_BIN": paths["preflight"],
            "PREFLIGHT_LOG": str(tmp_path / "preflight.log"),
            "WORKER_PREFLIGHT_LOG": str(tmp_path / "worker-preflight.log"),
            "TORCHRUN_BIN": paths["torchrun"],
            "TORCHRUN_LOG": str(tmp_path / "torchrun.log"),
            "CALLS_LOG": str(tmp_path / "calls.log"),
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


def _assignments(stdout: str) -> dict[str, str]:
    return {
        key: value
        for line in stdout.splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
        if key.isupper()
    }


def _import_stage1_config(
    **updates: str,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "RESUME_FROM_PATH": "",
            "RESUME_ONLINE_FROM_TARGET": "0",
            "RESET_RESUME_STEP": "0",
            "RESUME_OPTIMIZER_STATE": "0",
            "TRAIN_SEED": "42",
            "WAN_STUDENT_BASE_MODEL_PATH": "/explicit/wan-flowmap-base",
            **updates,
        }
    )
    code = """import json
from distillation_flowmap.config_libero_cosmos_policy_stage1 import cfg
print(json.dumps({
    "resume_from_path": cfg.resume_from_path,
    "resume_online_from_target": cfg.resume_online_from_target,
    "reset_resume_step": cfg.reset_resume_step,
    "resume_optimizer_state": cfg.resume_optimizer_state,
    "seed": cfg.seed,
    "student_backend": cfg.student_backend,
    "student_base_model_path": cfg.student_base_model_path,
}))
"""
    return subprocess.run(
        [str(PYTHON), "-c", code],
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


def test_stage1_config_explicitly_parses_fresh_resume_controls_and_seed():
    result = _import_stage1_config(TRAIN_SEED="17")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "resume_from_path": None,
        "resume_online_from_target": False,
        "reset_resume_step": False,
        "resume_optimizer_state": False,
        "seed": 17,
        "student_backend": "wan_flowmap",
        "student_base_model_path": "/explicit/wan-flowmap-base",
    }


def test_stage1_config_explicitly_parses_resume_controls_and_seed(tmp_path):
    checkpoint = tmp_path / "output" / "checkpoints" / "step_10"

    result = _import_stage1_config(
        RESUME_FROM_PATH=str(checkpoint),
        RESUME_ONLINE_FROM_TARGET="false",
        RESET_RESUME_STEP="no",
        RESUME_OPTIMIZER_STATE="true",
        TRAIN_SEED="23",
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "resume_from_path": str(checkpoint),
        "resume_online_from_target": False,
        "reset_resume_step": False,
        "resume_optimizer_state": True,
        "seed": 23,
        "student_backend": "wan_flowmap",
        "student_base_model_path": "/explicit/wan-flowmap-base",
    }


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("RESUME_ONLINE_FROM_TARGET", "maybe"),
        ("RESET_RESUME_STEP", "2"),
        ("RESUME_OPTIMIZER_STATE", ""),
        ("TRAIN_SEED", "not-an-int"),
    ],
)
def test_stage1_config_rejects_invalid_typed_resume_or_seed_env(variable, value):
    result = _import_stage1_config(**{variable: value})

    assert result.returncode != 0
    assert variable in result.stderr


def _make_sharded(transformer: Path, shard_names: list[str]) -> Path:
    (transformer / "diffusion_pytorch_model.safetensors").unlink()
    weight_map = {}
    for index, shard_name in enumerate(shard_names):
        shard = transformer / shard_name
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(b"test")
        weight_map[f"weight_{index}"] = shard_name
    index_path = transformer / "diffusion_pytorch_model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": weight_map}) + "\n", encoding="utf-8")
    return index_path


def test_dry_run_prints_corrected_raw_contract_without_mutation(tmp_path):
    env = _env(tmp_path)

    result = _run("dry-run", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    values = _assignments(result.stdout)
    repo = Path(env["COSMOS_PREDICT2_REPO"])
    worker_site_packages = (
        Path(env["COSMOS_WORKER_ENV_ROOT"]) / "lib/python3.10/site-packages"
    )
    assert values["COSMOS_POLICY_EXTRA_PYTHONPATH"] == (
        f"{repo}/packages/cosmos-cuda:{repo}/packages/cosmos-oss"
    )
    assert values["COSMOS_WORKER_SITE_PACKAGES"] == str(worker_site_packages)
    assert "nvidia/cudnn/lib" in values["COSMOS_WORKER_CUDA_LIBRARY_PATH"]
    assert "CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage1" in result.stdout
    assert "MAX_TRAIN_STEPS=5000" in result.stdout
    assert "SAVE_INTERVAL=1000" in result.stdout
    assert "TRAIN_SEED=42" in result.stdout
    assert "MASTER_PORT=29671" in result.stdout
    assert "--nproc_per_node=8" in result.stdout
    assert "--master_port=29671" in result.stdout
    assert f"STUDENT_BASE_MODEL_PATH={env['CLEAN_STUDENT_BASE_MODEL_PATH']}" in result.stdout
    assert "STUDENT_BACKEND=wan_flowmap" in result.stdout
    assert (
        f"WAN_STUDENT_BASE_MODEL_PATH={env['CLEAN_STUDENT_BASE_MODEL_PATH']}"
        in result.stdout
    )
    assert "RESUME_FROM_PATH=" in result.stdout
    assert "raw_stage1_5000" not in result.stdout
    assert not Path(env["STAGE1_OUTPUT"]).exists()
    assert Path(env["PREFLIGHT_LOG"]).read_text(encoding="utf-8")
    worker_log = Path(env["WORKER_PREFLIGHT_LOG"]).read_text(encoding="utf-8")
    assert (
        f"PYTHONPATH={repo}:{repo}/packages/cosmos-cuda:"
        f"{repo}/packages/cosmos-oss"
    ) in worker_log
    assert f"LD_LIBRARY_PATH={values['COSMOS_WORKER_CUDA_LIBRARY_PATH']}" in worker_log
    assert "PYTHONDONTWRITEBYTECODE=1" in worker_log
    assert not Path(env["TORCHRUN_LOG"]).exists()


@pytest.mark.parametrize("package", ("cosmos-cuda", "cosmos-oss"))
def test_rejects_missing_cosmos_worker_package_roots(tmp_path, package):
    env = _env(tmp_path)
    package_root = Path(env["COSMOS_PREDICT2_REPO"]) / "packages" / package
    (package_root / ".fixture").unlink()
    package_root.rmdir()

    result = _run("dry-run", env=env)

    assert result.returncode != 0
    assert package in result.stderr
    assert not Path(env["WORKER_PREFLIGHT_LOG"]).exists()


def test_rejects_dirty_cosmos_worker_repository(tmp_path):
    env = _env(tmp_path)
    (Path(env["COSMOS_PREDICT2_REPO"]) / "untracked.py").write_text(
        "DIRTY = True\n",
        encoding="utf-8",
    )

    result = _run("dry-run", env=env)

    assert result.returncode != 0
    assert "COSMOS_PREDICT2_REPO must be clean" in result.stderr
    assert not Path(env["WORKER_PREFLIGHT_LOG"]).exists()


def test_rejects_missing_declared_worker_cuda_library_directory(tmp_path):
    env = _env(tmp_path)
    missing = (
        Path(env["COSMOS_WORKER_ENV_ROOT"])
        / "lib/python3.10/site-packages/nvidia/cudnn/lib"
    )
    missing.rmdir()

    result = _run("dry-run", env=env)

    assert result.returncode != 0
    assert str(missing) in result.stderr
    assert not Path(env["WORKER_PREFLIGHT_LOG"]).exists()


def test_primary_wan_path_is_explicit_and_conflicting_legacy_alias_is_rejected(
    tmp_path,
):
    env = _env(tmp_path)
    wan_base = env.pop("CLEAN_STUDENT_BASE_MODEL_PATH")
    env["WAN_STUDENT_BASE_MODEL_PATH"] = wan_base

    accepted = _run("dry-run", env=env)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert f"WAN_STUDENT_BASE_MODEL_PATH={wan_base}" in accepted.stdout
    assert "deprecated" not in accepted.stderr

    env["CLEAN_STUDENT_BASE_MODEL_PATH"] = str(tmp_path / "different")
    rejected = _run("dry-run", env=env)
    assert rejected.returncode != 0
    assert "disagree" in rejected.stderr


def test_rejects_cosmos_teacher_root_as_wan_student(tmp_path):
    env = _env(tmp_path)
    env["WAN_STUDENT_BASE_MODEL_PATH"] = env["COSMOS_POLICY_PATH"]
    env.pop("CLEAN_STUDENT_BASE_MODEL_PATH")

    result = _run("dry-run", env=env)

    assert result.returncode != 0
    assert "wan_flowmap" in result.stderr
    assert "hybrid Student/Teacher backend validation failed" in result.stderr


def test_full_cosmos_utils_worker_import_failure_is_fail_closed(tmp_path):
    env = _env(tmp_path)
    env["FAKE_WORKER_IMPORT_EXIT"] = "9"

    result = _run("dry-run", env=env)

    assert result.returncode != 0
    assert "worker import preflight failed" in result.stderr
    worker_call = Path(env["WORKER_PREFLIGHT_LOG"]).read_text(encoding="utf-8")
    assert "cosmos_utils import get_action" in worker_call
    assert not Path(env["STAGE1_OUTPUT"]).exists()
    assert not Path(env["TORCHRUN_LOG"]).exists()


def test_dry_run_accepts_sharded_clean_base_layout(tmp_path):
    env = _env(tmp_path)
    transformer = Path(env["CLEAN_STUDENT_BASE_MODEL_PATH"]) / "transformer"
    shards = [
        "diffusion_pytorch_model-00001-of-00002.safetensors",
        "diffusion_pytorch_model-00002-of-00002.safetensors",
    ]
    _make_sharded(transformer, shards)

    result = _run("dry-run", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert not Path(env["STAGE1_OUTPUT"]).exists()


@pytest.mark.parametrize(
    "protected_key",
    [
        "CLEAN_STUDENT_BASE_MODEL_PATH",
        "DATASET_PATH",
        "COSMOS_POLICY_PATH",
        "COSMOS_PREDICT2_REPO",
        "COSMOS_PREDICT25_LOCAL_MODEL_DIR",
    ],
)
def test_fresh_output_cannot_equal_or_nest_beneath_protected_inputs(
    tmp_path, protected_key
):
    env = _env(tmp_path)
    protected = Path(env[protected_key])

    equal = _run("dry-run", "--output-dir", str(protected), env=env)
    assert equal.returncode != 0
    assert "protected" in equal.stderr

    nested = _run("dry-run", "--output-dir", str(protected / "new-output"), env=env)
    assert nested.returncode != 0
    assert "protected" in nested.stderr


def test_fresh_mode_rejects_dangling_output_symlink(tmp_path):
    env = _env(tmp_path)
    output = Path(env["STAGE1_OUTPUT"])
    output.symlink_to(tmp_path / "missing-output", target_is_directory=True)

    result = _run("dry-run", env=env)

    assert result.returncode != 0
    assert "symlink" in result.stderr


def test_fresh_output_cannot_escape_under_protected_input_through_parent_symlink(
    tmp_path,
):
    env = _env(tmp_path)
    alias = tmp_path / "clean-base-alias"
    alias.symlink_to(env["CLEAN_STUDENT_BASE_MODEL_PATH"], target_is_directory=True)

    result = _run("dry-run", "--output-dir", str(alias / "new-output"), env=env)

    assert result.returncode != 0
    assert "protected" in result.stderr


@pytest.mark.parametrize(
    ("kind", "message"),
    [
        ("missing", "missing declared"),
        ("empty", "nonempty weight_map"),
        ("malformed", "nonempty weight_map"),
        ("nested", "invalid shard name"),
        ("absolute", "invalid shard name"),
        ("dotdot", "invalid shard name"),
        ("symlink", "symlink"),
        ("index_symlink", "symlink"),
    ],
)
def test_rejects_invalid_sharded_transformer_layout(tmp_path, kind, message):
    env = _env(tmp_path)
    transformer = Path(env["CLEAN_STUDENT_BASE_MODEL_PATH"]) / "transformer"
    (transformer / "diffusion_pytorch_model.safetensors").unlink()
    index_path = transformer / "diffusion_pytorch_model.safetensors.index.json"

    if kind == "missing":
        payload = {"weight_map": {"a": "missing.safetensors"}}
    elif kind == "empty":
        payload = {"weight_map": {}}
    elif kind == "malformed":
        payload = {"weight_map": []}
    elif kind == "nested":
        payload = {"weight_map": {"a": "nested/shard.safetensors"}}
    elif kind == "absolute":
        payload = {"weight_map": {"a": str(tmp_path / "outside.safetensors")}}
    elif kind == "dotdot":
        payload = {"weight_map": {"a": ".."}}
    else:
        payload = {"weight_map": {"a": "shard.safetensors"}}
        target = tmp_path / "outside-shard.safetensors"
        target.write_bytes(b"test")
        (transformer / "shard.safetensors").symlink_to(target)

    real_index = tmp_path / "real-index.json" if kind == "index_symlink" else index_path
    real_index.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    if kind == "index_symlink":
        index_path.symlink_to(real_index)

    result = _run("dry-run", env=env)

    assert result.returncode != 0
    assert message in result.stderr


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
        ("0,00,1,2,3,4,5,6", "canonical"),
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


def test_real_config_preflight_accepts_fresh_and_resume_modes(tmp_path):
    env = _env(tmp_path)
    env["PREFLIGHT_BIN"] = str(PYTHON)

    fresh = _run("dry-run", "--steps", "20", env=env)
    assert fresh.returncode == 0, fresh.stdout + fresh.stderr
    assert not Path(env["STAGE1_OUTPUT"]).exists()

    _checkpoint(Path(env["STAGE1_OUTPUT"]), 10)
    resume = _run(
        "dry-run",
        "--steps",
        "20",
        "--resume-step",
        "10",
        env=env,
    )
    assert resume.returncode == 0, resume.stdout + resume.stderr


@pytest.mark.parametrize(
    "component",
    [
        "output",
        "checkpoints",
        "checkpoint",
        "transformer",
        "config",
        "optimizer",
        "scheduler",
        "weights",
    ],
)
def test_resume_rejects_symlinked_path_components_and_files(tmp_path, component):
    env = _env(tmp_path)
    output = Path(env["STAGE1_OUTPUT"])
    checkpoint = _checkpoint(output, 3000)

    if component == "output":
        real_output = tmp_path / "real-output"
        output.rename(real_output)
        output.symlink_to(real_output, target_is_directory=True)
    elif component == "checkpoints":
        real_checkpoints = tmp_path / "real-checkpoints"
        (output / "checkpoints").rename(real_checkpoints)
        (output / "checkpoints").symlink_to(real_checkpoints, target_is_directory=True)
    elif component == "checkpoint":
        real_checkpoint = tmp_path / "real-checkpoint"
        checkpoint.rename(real_checkpoint)
        checkpoint.symlink_to(real_checkpoint, target_is_directory=True)
    elif component == "transformer":
        transformer = checkpoint / "online_student" / "transformer"
        real_transformer = tmp_path / "real-transformer"
        transformer.rename(real_transformer)
        transformer.symlink_to(real_transformer, target_is_directory=True)
    else:
        paths = {
            "config": checkpoint / "online_student" / "transformer" / "config.json",
            "optimizer": checkpoint / "optimizer.pt",
            "scheduler": checkpoint / "lr_scheduler.pt",
            "weights": (
                checkpoint
                / "target_student"
                / "transformer"
                / "diffusion_pytorch_model.safetensors"
            ),
        }
        path = paths[component]
        target = tmp_path / f"real-{component}"
        path.rename(target)
        path.symlink_to(target)

    result = _run("dry-run", "--resume-step", "3000", env=env)

    assert result.returncode != 0
    assert "symlink" in result.stderr
    assert not Path(env["TORCHRUN_LOG"]).exists()


@pytest.mark.parametrize(
    ("variant", "field", "value", "message"),
    [
        ("online_student", "contract_version", 1, "contract_version"),
        ("target_student", "training_contract_stage", "progressive_stage2", "training_contract_stage"),
        ("target_student", "checkpoint_step", 2999, "checkpoint_step"),
        ("online_student", "student_backend", "cosmos_policy", "student_backend"),
        ("target_student", "teacher_backend", "wanva", "teacher_backend"),
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


def test_optimized_python_cannot_bypass_invalid_config_preflight(tmp_path):
    env = _env(tmp_path)
    module = tmp_path / "invalid_stage1_config.py"
    module.write_text(
        "from distillation_flowmap.config_libero_cosmos_policy_stage1 import cfg\n"
        'cfg.teacher_backend = "invalid-backend"\n',
        encoding="utf-8",
    )
    wrapper = tmp_path / "optimized-preflight"
    _write_executable(
        wrapper,
        'export CONFIG_FILE="invalid_stage1_config"\n'
        'exec "$REAL_PREFLIGHT_PYTHON" "$@"\n',
    )
    env.update(
        {
            "PREFLIGHT_BIN": str(wrapper),
            "REAL_PREFLIGHT_PYTHON": str(PYTHON),
            "PYTHONOPTIMIZE": "2",
            "PYTHONPATH": str(tmp_path),
        }
    )

    result = _run("dry-run", env=env)

    assert result.returncode != 0
    assert "teacher_backend" in result.stderr
    assert "config preflight failed" in result.stderr
    assert not Path(env["STAGE1_OUTPUT"]).exists()


def test_run_loses_atomic_output_claim_without_starting_torchrun(tmp_path):
    env = _env(tmp_path)
    env["CLAIM_OUTPUT_DURING_PREFLIGHT"] = "1"

    result = _run("run", env=env)

    assert result.returncode != 0
    assert "claim OUTPUT_DIR" in result.stderr
    assert Path(env["STAGE1_OUTPUT"]).is_dir()
    assert not Path(env["TORCHRUN_LOG"]).exists()


def test_run_creates_output_only_after_preflight_and_executes_fake_torchrun(tmp_path):
    env = _env(tmp_path)

    result = _run("run", "--steps", "20", "--save-interval", "10", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    calls = Path(env["CALLS_LOG"]).read_text(encoding="utf-8").splitlines()
    assert calls.index("worker-import-preflight") < calls.index("torchrun")
    output = Path(env["STAGE1_OUTPUT"])
    assert output.is_dir()
    log = Path(env["TORCHRUN_LOG"]).read_text(encoding="utf-8")
    assert "--nproc_per_node=8" in log
    assert "--master_port=29671" in log
    assert "CONFIG_FILE=distillation_flowmap.config_libero_cosmos_policy_stage1" in log
    assert f"STUDENT_BASE_MODEL_PATH={env['CLEAN_STUDENT_BASE_MODEL_PATH']}" in log
    assert "RESUME_ONLINE_FROM_TARGET=0" in log
    assert "RESET_RESUME_STEP=0" in log
    assert "RESUME_OPTIMIZER_STATE=0" in log
    assert "COSMOS_POLICY_EXTRA_PYTHONPATH=" in log
    assert "COSMOS_WORKER_SITE_PACKAGES=" in log
    assert "COSMOS_WORKER_CUDA_LIBRARY_PATH=" in log
    assert "--resume-from-path" not in log


def test_run_exports_validated_cuda_libraries_to_torchrun_and_preserves_suffix(
    tmp_path,
):
    env = _env(tmp_path)
    inherited = "/caller/cuda/lib:/caller/vendor/lib"
    env["LD_LIBRARY_PATH"] = inherited

    result = _run("run", "--steps", "20", "--save-interval", "10", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    values = _assignments(result.stdout)
    torchrun_env = {
        key: value
        for line in Path(env["TORCHRUN_LOG"]).read_text(encoding="utf-8").splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }
    assert torchrun_env["LD_LIBRARY_PATH"] == (
        f"{values['COSMOS_WORKER_CUDA_LIBRARY_PATH']}:{inherited}"
    )


def test_resume_run_executes_fake_torchrun_with_exact_restore_contract(tmp_path):
    env = _env(tmp_path)
    checkpoint = _checkpoint(Path(env["STAGE1_OUTPUT"]), 10)

    result = _run("run", "--steps", "20", "--resume-step", "10", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    log = Path(env["TORCHRUN_LOG"]).read_text(encoding="utf-8")
    assert f"--resume-from-path {checkpoint}" in log
    assert f"RESUME_FROM_PATH={checkpoint}" in log
    assert "RESUME_ONLINE_FROM_TARGET=0" in log
    assert "RESET_RESUME_STEP=0" in log
    assert "RESUME_OPTIMIZER_STATE=1" in log


def test_resume_run_atomically_replaces_launch_artifact_symlinks(tmp_path):
    env = _env(tmp_path)
    output = Path(env["STAGE1_OUTPUT"])
    _checkpoint(output, 10)
    env_sentinel = tmp_path / "outside-env-sentinel"
    command_sentinel = tmp_path / "outside-command-sentinel"
    env_sentinel.write_text("outside env stays unchanged\n", encoding="utf-8")
    command_sentinel.write_text("outside command stays unchanged\n", encoding="utf-8")
    (output / "launch_env.txt").symlink_to(env_sentinel)
    (output / "launch_command.txt").symlink_to(command_sentinel)

    result = _run("run", "--steps", "20", "--resume-step", "10", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    assert env_sentinel.read_text(encoding="utf-8") == "outside env stays unchanged\n"
    assert (
        command_sentinel.read_text(encoding="utf-8")
        == "outside command stays unchanged\n"
    )
    for name in ("launch_env.txt", "launch_command.txt"):
        artifact = output / name
        assert artifact.is_file()
        assert not artifact.is_symlink()
    assert Path(env["TORCHRUN_LOG"]).is_file()
