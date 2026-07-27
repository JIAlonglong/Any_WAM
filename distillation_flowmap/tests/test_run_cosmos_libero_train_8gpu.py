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
import sys
from pathlib import Path

import pytest

from distillation_flowmap.cosmos_libero_variants import (
    canonical_variant_json,
    resolve_variant,
    validate_aligned_opd_arm_contract,
)
from distillation_flowmap.cosmos_libero_provenance import (
    build_artifact_lock,
    canonical_json,
    resolve_formal_provenance,
)
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


def _transformer(root: Path, payload: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "diffusion_pytorch_model.safetensors").write_bytes(b"test")
    return root


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_has_one_exact_aligned_opd_contract(arm, tmp_path):
    record = resolve_variant(arm, output_root=tmp_path)
    validate_aligned_opd_arm_contract(
        arm,
        rollout_steps=record["danceopd_rollout_steps"],
        endpoint_weight=record["video_endpoint_weight"],
        velocity_weight=record["video_velocity_weight"],
    )


@pytest.mark.parametrize(
    ("arm", "rollout_steps", "endpoint_weight", "velocity_weight"),
    [
        ("s1", (2,), 1.0, 0.0),
        ("s2", (2,), 1.0, 0.0),
        ("stage1_only", (2, 4), 1.0, 0.0),
        ("anchor_only", (2, 4), 1.0, 1.0),
        ("field_only", (2, 4), 1.0, 1.0),
    ],
)
def test_aligned_opd_contract_rejects_cross_arm_settings(
    arm, rollout_steps, endpoint_weight, velocity_weight
):
    with pytest.raises(ValueError, match="does not match variant"):
        validate_aligned_opd_arm_contract(
            arm,
            rollout_steps=rollout_steps,
            endpoint_weight=endpoint_weight,
            velocity_weight=velocity_weight,
        )


def _layout(tmp_path: Path, *, parent_step: int = 5000):
    raw_contract = {**RAW_CONTRACT, "checkpoint_step": parent_step}
    stage1 = tmp_path / "stage1"
    for variant in ("online_student", "target_student"):
        _transformer(stage1 / variant / "transformer", raw_contract)
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    for name in (
        "info.json",
        "tasks.jsonl",
        "episodes.jsonl",
        "episodes_ori.jsonl",
        "episodes_stats.jsonl",
    ):
        (dataset / "meta" / name).write_text(
            json.dumps({"path": name, "revision": 1}) + "\n",
            encoding="utf-8",
        )
    (dataset / "data" / "chunk-000").mkdir(parents=True)
    (dataset / "data" / "chunk-000" / "episode_000000.parquet").write_bytes(
        b"parquet"
    )
    (dataset / "empty_emb.pt").write_bytes(b"test")
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "config.json").write_text(
        '{"model_type":"cosmos-policy","revision":1}', encoding="utf-8"
    )
    (policy / "libero_dataset_statistics.json").write_text(
        '{"revision":1}', encoding="utf-8"
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
    (repo / "packages" / "cosmos-cuda").mkdir(parents=True)
    (repo / "packages" / "cosmos-oss").mkdir(parents=True)
    (repo / "source.py").write_text("REVISION = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Fixture"], check=True
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    local_model = tmp_path / "local-model"
    for relative in (
        "config.json",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json",
    ):
        path = local_model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"revision":1}', encoding="utf-8")
    (local_model / "model-480p-16fps.pt").write_bytes(b"video-model")
    (local_model / "tokenizer" / "tokenizer.pth").write_bytes(b"tokenizer")
    worker_python = tmp_path / "cosmos-python"
    worker_python.write_text(
        "#!"
        + sys.executable
        + "\n"
        + "import json, os, sys\n"
        + "print(json.dumps({'executable': os.path.realpath(sys.argv[0]),"
        + "'python_version':'fixture','implementation':'CPython',"
        + "'prefix':'/fixture','base_prefix':'/fixture',"
        + "'torch':{'version':'2.7.0','cuda':'12.8','git_version':'fixture'}},"
        + "sort_keys=True,separators=(',',':')))\n",
        encoding="utf-8",
    )
    worker_python.chmod(worker_python.stat().st_mode | stat.S_IXUSR)
    site_packages = tmp_path / "site-packages"
    metadata = site_packages / "torch-2.7.0.dist-info" / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Name: torch\nVersion: 2.7.0\n", encoding="utf-8")

    flashwam_repo = tmp_path / "flashwam-source"
    flashwam_repo.mkdir()
    (flashwam_repo / "source.py").write_text("REVISION = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(flashwam_repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(flashwam_repo),
            "config",
            "user.email",
            "fixture@example.com",
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(flashwam_repo), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(flashwam_repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(flashwam_repo), "commit", "-qm", "fixture"], check=True
    )
    (flashwam_repo / "distillation_flowmap").symlink_to(
        ROOT / "distillation_flowmap",
        target_is_directory=True,
    )
    (flashwam_repo / "wan_va").mkdir()
    subprocess.run(
        ["git", "-C", str(flashwam_repo), "add", "distillation_flowmap"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(flashwam_repo), "commit", "-qm", "test launcher seam"],
        check=True,
    )

    lock_root = tmp_path / "provenance-locks"
    lock_root.mkdir()
    lock_specs = {
        "dataset": (
            dataset,
            (
                "meta/info.json",
                "meta/tasks.jsonl",
                "meta/episodes.jsonl",
                "meta/episodes_ori.jsonl",
                "meta/episodes_stats.jsonl",
                "empty_emb.pt",
            ),
            ("data/chunk-000/episode_000000.parquet",),
        ),
        "teacher": (
            policy,
            ("config.json", "libero_dataset_statistics.json"),
            (
                "Cosmos-Policy-LIBERO-Predict2-2B.pt",
                "libero_t5_embeddings.pkl",
            ),
        ),
        "video_vae": (
            stage1 / "target_student",
            ("transformer/config.json",),
            ("transformer/diffusion_pytorch_model.safetensors",),
        ),
        "local_model": (
            local_model,
            (
                "config.json",
                "model_index.json",
                "scheduler/scheduler_config.json",
                "tokenizer/tokenizer_config.json",
            ),
            ("model-480p-16fps.pt", "tokenizer/tokenizer.pth"),
        ),
    }
    for name, (root, compact, large) in lock_specs.items():
        payload = build_artifact_lock(
            root,
            compact_paths=compact,
            large_paths=large,
            immutable_store=False,
        )
        (lock_root / f"{name}.lock.json").write_text(
            canonical_json(payload) + "\n", encoding="utf-8"
        )
    return (
        stage1,
        dataset,
        policy,
        repo,
        local_model,
        worker_python,
        site_packages,
        flashwam_repo,
        lock_root,
    )


def _env(tmp_path: Path, *, parent_step: int = 5000):
    (
        stage1,
        dataset,
        policy,
        repo,
        local_model,
        worker_python,
        site_packages,
        flashwam_repo,
        lock_root,
    ) = _layout(tmp_path, parent_step=parent_step)
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
            "COSMOS_WORKER_SITE_PACKAGES": str(site_packages),
            "COSMOS_PROVENANCE_LOCK_ROOT": str(lock_root),
            "VERIFY_LARGE_ARTIFACT_DIGESTS": "1",
            "CUDA_VISIBLE_DEVICES": DEVICES,
            "COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES": DEVICES,
            "_TEST_COSMOS_LAUNCHER_SCRIPT": str(
                flashwam_repo
                / "distillation_flowmap"
                / "run_cosmos_libero_train_8gpu.sh"
            ),
            "_TEST_COSMOS_PROJECT_ROOT": str(flashwam_repo),
        }
    )
    return env, stage1


def _run(*args: str, env: dict[str, str], cwd: Path | None = None):
    launch_env = dict(env)
    launcher = launch_env.pop("_TEST_COSMOS_LAUNCHER_SCRIPT", str(SCRIPT))
    launch_env.pop("_TEST_COSMOS_PROJECT_ROOT", None)
    return subprocess.run(
        ["bash", launcher, *args],
        text=True,
        capture_output=True,
        cwd=cwd or ROOT,
        env=launch_env,
        check=False,
    )


def _resolve_for_launcher(env, name, *, output_root, run_tag, **overrides):
    repo = Path(env["COSMOS_PREDICT2_REPO"])
    lock_root = Path(env["COSMOS_PROVENANCE_LOCK_ROOT"])
    provenance = resolve_formal_provenance(
        dataset_root=env["DATASET_PATH"],
        dataset_lock=lock_root / "dataset.lock.json",
        teacher_root=env["COSMOS_POLICY_PATH"],
        teacher_lock=lock_root / "teacher.lock.json",
        video_vae_root=env["STUDENT_BASE_MODEL_PATH"],
        video_vae_lock=lock_root / "video_vae.lock.json",
        local_model_root=env["COSMOS_PREDICT25_LOCAL_MODEL_DIR"],
        local_model_lock=lock_root / "local_model.lock.json",
        flashwam_repo=env["_TEST_COSMOS_PROJECT_ROOT"],
        cosmos_repo=repo,
        preflight_python=Path(sys.executable),
        preflight_import_roots=(
            env["_TEST_COSMOS_PROJECT_ROOT"],
            Path(env["_TEST_COSMOS_PROJECT_ROOT"]) / "wan_va",
            Path(sys.prefix) / "lib/python3.10/site-packages",
        ),
        worker_python=env["COSMOS_POLICY_PYTHON"],
        worker_site_packages=env["COSMOS_WORKER_SITE_PACKAGES"],
        extra_pythonpath=(
            repo / "packages/cosmos-cuda",
            repo / "packages/cosmos-oss",
        ),
        verify_large_artifact_digests=True,
    )
    return resolve_variant(
        name,
        output_root=output_root,
        run_tag=run_tag,
        dataset_path=env["DATASET_PATH"],
        teacher_model_path=env["COSMOS_POLICY_PATH"],
        cosmos_video_vae_model_path=env["STUDENT_BASE_MODEL_PATH"],
        cosmos_policy_repo=env["COSMOS_PREDICT2_REPO"],
        cosmos_policy_python=env["COSMOS_POLICY_PYTHON"],
        cosmos_policy_extra_pythonpath=(
            f"{repo}/packages/cosmos-cuda:{repo}/packages/cosmos-oss"
        ),
        cosmos_policy_local_model_dir=env["COSMOS_PREDICT25_LOCAL_MODEL_DIR"],
        attention_mode="flex",
        provenance=provenance,
        **overrides,
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


def test_stage2_launcher_preserves_explicit_cosmos_extra_pythonpath(tmp_path):
    env, _ = _env(tmp_path)
    custom_cuda = tmp_path / "custom-cosmos-cuda"
    custom_oss = tmp_path / "custom-cosmos-oss"
    custom_cuda.mkdir()
    custom_oss.mkdir()
    expected = f"{custom_cuda}:{custom_oss}"
    env["COSMOS_POLICY_EXTRA_PYTHONPATH"] = expected

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assert _assignments(result.stdout)["COSMOS_POLICY_EXTRA_PYTHONPATH"] == expected


def test_stage2_launcher_replaces_ambient_ld_library_path_with_worker_contract(
    tmp_path,
):
    env, _ = _env(tmp_path)
    env["LD_LIBRARY_PATH"] = "/ambient/poison"

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assignments = _assignments(result.stdout)
    assert assignments["LD_LIBRARY_PATH"] == assignments[
        "COSMOS_WORKER_CUDA_LIBRARY_PATH"
    ]
    assert "/ambient/poison" not in assignments["LD_LIBRARY_PATH"]


def test_dry_run_embeds_read_only_provenance_in_canonical_identity(tmp_path):
    env, _ = _env(tmp_path)
    output_root = tmp_path / "out"
    result = _run(
        "apm",
        "--output-root",
        str(output_root),
        "--run-tag",
        "provenance",
        "--dry-run",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    values = _assignments(result.stdout)
    provenance = json.loads(values["PROVENANCE_IDENTITY_JSON"])
    variant = json.loads(values["COSMOS_LIBERO_VARIANT_JSON"])
    actions = json.loads(values["ENV_CONTRACT_ACTIONS_JSON"])
    assert provenance["schema"] == "flashwam_cosmos_provenance_v1"
    assert variant["provenance"] == provenance
    assert actions["DATASET_PATH"]["action"] == "set_canonical"
    assert actions["USE_FSDP1"] == {
        "action": "set_pinned",
        "value": "1",
    }
    assert actions["DISTILL_MODE"] == {"action": "unset_legacy"}
    assert not output_root.exists()


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
            _resolve_for_launcher(
                env,
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
    assert values["RUN_TAG"] == "contract-red"
    assert json.loads(values["COSMOS_LIBERO_VARIANT_JSON"]) == expected
    assert json.loads(values["CONFIG_IDENTITY_JSON"]) == expected
    assert values["ENABLE_TENSORBOARD"] == "1"
    assert values["ENABLE_WANDB"] == "0"
    assert values["WANDB_MODE"] == "offline"
    assert values["HF_DATASETS_OFFLINE"] == "1"
    assert values["TRANSFORMERS_OFFLINE"] == "1"
    assert values["HF_HUB_OFFLINE"] == "1"
    assert values["MECHANISM_DIAGNOSTICS"] == "1"
    assert values["MECHANISM_DIAGNOSTIC_INTERVAL"] == "100"
    assert values["MECHANISM_DIAGNOSTIC_SEED"] == "42"
    assert values["MECHANISM_DIAGNOSTIC_R"] == "500"
    assert values["MECHANISM_DIAGNOSTIC_S"] == "250"
    assert values["MECHANISM_DIAGNOSTIC_TEACHER_STEPS"] == "8"
    assert values["MECHANISM_COSMOS_T_MIN"] == "0.8"
    assert values["MECHANISM_COSMOS_T_MAX"] == "0.9876543209876543"
    assert values["CONFIG_FILE"] == (
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    assert values["COSMOS_POLICY_PATH"] == env["COSMOS_POLICY_PATH"]
    assert values["DATASET_PATH"] == env["DATASET_PATH"]
    assert values["COSMOS_POLICY_PYTHON"] == env["COSMOS_POLICY_PYTHON"]
    assert values["COSMOS_PREDICT2_REPO"] == env["COSMOS_PREDICT2_REPO"]
    assert values["COSMOS_PREDICT25_LOCAL_MODEL_DIR"] == env[
        "COSMOS_PREDICT25_LOCAL_MODEL_DIR"
    ]
    assert values["ATTN_MODE"] == "flex"
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


def test_launcher_resolves_torchrun_without_conda_or_path_initialization(tmp_path):
    env, _ = _env(tmp_path)
    env.pop("TORCHRUN_BIN", None)
    env["PATH"] = "/usr/bin:/bin"

    result = _run("apm", "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    command = _assignments(result.stdout)["COMMAND"]
    assert command.startswith(
        "/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/torchrun "
    )


def test_launcher_rejects_non_file_torchrun_before_claiming_output(tmp_path):
    env, _ = _env(tmp_path)
    output_root = tmp_path / "must-not-exist"
    env["TORCHRUN_BIN"] = str(tmp_path)

    result = _run(
        "apm",
        "--output-root",
        str(output_root),
        "--run-tag",
        "invalid-torchrun",
        env=env,
    )

    assert result.returncode != 0
    assert "TORCHRUN_BIN is not an executable file" in result.stderr
    assert not output_root.exists()


def test_launcher_reports_missing_bare_torchrun_override_by_name(tmp_path):
    env, _ = _env(tmp_path)
    env["TORCHRUN_BIN"] = "missing-torchrun-for-test"
    env["PATH"] = "/usr/bin:/bin"

    result = _run("apm", "--dry-run", env=env)

    assert result.returncode != 0
    assert "TORCHRUN_BIN is not available: missing-torchrun-for-test" in result.stderr


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
        _resolve_for_launcher(
            env,
            "field_only", output_root=output_root, run_tag="formal"
        )
    )
    assert marker.read_text(encoding="utf-8").strip() == expected
    assert (
        output / "cosmos_libero_variant.json"
    ).read_text(encoding="utf-8").strip() == expected
    variant_payload = json.loads(expected)
    assert json.loads(
        (output / "cosmos_libero_provenance.json").read_text(encoding="utf-8")
    ) == variant_payload["provenance"]
    assert not (output_root / "formal" / "apm").exists()


def test_default_save_interval_is_exactly_1000(tmp_path):
    env, _ = _env(tmp_path)
    result = _run(
        "universal", "--output-root", str(tmp_path / "out"), "--run-tag", "default-save", "--dry-run", env=env
    )
    assert result.returncode == 0, result.stderr
    assert _assignments(result.stdout)["SAVE_INTERVAL"] == "1000"


def test_hostile_ambient_scientific_values_cannot_change_canonical_experiment(
    tmp_path,
):
    env, _ = _env(tmp_path)
    env.update(
        {
            "LEARNING_RATE": "9.0",
            "OPD_AUX_WEIGHT": "9.0",
            "OPD_AUX_WARMUP_STEPS": "999",
            "OPD_AUX_PROB": "0.125",
            "OPD_ROLLOUT_GRAD_MODE": "suffix",
            "OPD_ROLLOUT_GRAD_STEPS": "99",
            "OPD_ENDPOINT_FOCUS_PROB": "0.125",
            "OPD_DANCEOPD_QUERY_ALPHA": "1.0",
            "OPD_DANCEOPD_QUERY_BETA": "1.0",
        }
    )
    output_root = tmp_path / "hostile"
    result = _run(
        "universal",
        "--output-root",
        str(output_root),
        "--run-tag",
        "canonical",
        "--dry-run",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    values = _assignments(result.stdout)
    expected = json.loads(
        canonical_variant_json(
            _resolve_for_launcher(
                env,
                "universal",
                output_root=output_root,
                run_tag="canonical",
            )
        )
    )
    assert json.loads(values["COSMOS_LIBERO_VARIANT_JSON"]) == expected
    assert json.loads(values["CONFIG_IDENTITY_JSON"]) == expected
    assert values["LEARNING_RATE"] == "2e-07"
    assert values["OPD_AUX_WEIGHT"] == "0.1"
    assert values["OPD_AUX_WARMUP_STEPS"] == "8"
    assert values["OPD_AUX_PROB"] == "1.0"
    assert values["OPD_ROLLOUT_GRAD_MODE"] == "last_step"
    assert values["OPD_ROLLOUT_GRAD_STEPS"] == "1"
    assert values["OPD_ENDPOINT_FOCUS_PROB"] == "0.85"
    assert values["OPD_DANCEOPD_QUERY_ALPHA"] == "5.0"
    assert values["OPD_DANCEOPD_QUERY_BETA"] == "2.0"


def test_stage1_only_imported_config_has_no_opd_schedule_or_action_opd(tmp_path):
    env, _ = _env(tmp_path)
    output_root = tmp_path / "stage1-only"
    result = _run(
        "stage1_only",
        "--output-root",
        str(output_root),
        "--run-tag",
        "isolation",
        "--dry-run",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    values = _assignments(result.stdout)
    imported = json.loads(values["CONFIG_IDENTITY_JSON"])
    assert values["USE_OPD_AUX"] == "0"
    assert values["OPD_AUX_STANDALONE_STEP"] == "0"
    assert values["OPD_AUX_ACTION"] == "0"
    assert imported["use_opd_aux"] is False
    assert imported["opd_aux_standalone_step"] is False
    assert imported["action_opd_enabled"] is False


def test_entire_inherited_config_environment_is_sealed_against_hostile_ambient(
    tmp_path,
):
    env, _ = _env(tmp_path)
    hostile = {
        "BETA1": "0.1",
        "BETA2": "0.2",
        "EMA_DECAY": "0.5",
        "EMA_WARMUP_STEPS": "999",
        "DROP_TEXT_RATIO": "0.9",
        "FUSE_GUIDANCE_SCALE": "99",
        "CFG_MIN": "98",
        "CFG_MAX": "99",
        "MAX_GRAD_NORM": "99",
        "WARMUP_STEPS": "999",
        "NUM_DDIM_TIMESTEPS_ACTION": "99",
        "DIFFUSION_RATIO": "0.1",
        "CONSISTENCY_RATIO": "0.1",
        "FLOWMAP_RATIO": "0.8",
        "VIDEO_LOSS_WEIGHT": "99",
        "ACTION_LOSS_WEIGHT": "99",
        "ACTION_BLOCK_WEIGHT": "99",
        "COSMOS_POLICY_USE_RAW_INFERENCE": "0",
        "SKIP_TARGET_STUDENT_FOR_COSMOS_LATENT": "0",
        "COSMOS_LATENT_CDIFF_LOSS_WEIGHT": "99",
        "COSMOS_LATENT_ENDPOINT_LOSS_WEIGHT": "99",
        "COSMOS_LATENT_EPSILON": "0.5",
        "COSMOS_LATENT_T_MIN": "0.1",
        "COSMOS_LATENT_T_MAX": "0.2",
        "COSMOS_LATENT_CHANNELS": "99",
        "COSMOS_LATENT_FRAMES": "99",
        "COSMOS_LATENT_HEIGHT": "99",
        "COSMOS_LATENT_WIDTH": "99",
        "COSMOS_LATENT_CENTER_VELOCITY_MODE": "hostile",
        "COSMOS_LATENT_TARGET_MODE": "hostile",
        "COSMOS_LATENT_CDIFF_INTERVAL": "99",
        "OPD_ACTION_ROLLOUT_GRAD_MODE": "hostile",
        "OPD_AUX_INTERVAL": "99",
        "OPD_DANCEOPD_VERIFY_TERMINAL_PRIOR": "0",
        "OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE": "1",
        "OPD_DANCEOPD_TERMINAL_PRIOR_WARN_FACTOR": "0",
        "OPD_COSMOS_SPATIAL_CROP_SIZE": "1",
        "COSMOS_USE_TEACHER_ACTION_ANCHOR": "0",
        "OPD_JOINT_ACTION_ROLLOUT": "0",
        "COSMOS_POLICY_CONFIG_NAME": "hostile",
        "COSMOS_POLICY_CONFIG_FILE": "hostile",
        "COSMOS_POLICY_NUM_DENOISING_STEPS_ACTION": "99",
        "COSMOS_POLICY_SEED": "99",
        "COSMOS_POLICY_PRIMARY_IMAGE_KEY": "hostile",
        "COSMOS_POLICY_WRIST_IMAGE_KEY": "hostile",
        "MECHANISM_DIAGNOSTICS": "0",
        "MECHANISM_DIAGNOSTIC_INTERVAL": "999",
        "MECHANISM_DIAGNOSTIC_SEED": "999",
        "MECHANISM_DIAGNOSTIC_R": "900",
        "MECHANISM_DIAGNOSTIC_S": "800",
        "MECHANISM_DIAGNOSTIC_TEACHER_STEPS": "4",
        "MECHANISM_COSMOS_T_MIN": "0.1",
        "MECHANISM_COSMOS_T_MAX": "0.2",
        "GRADIENT_CHECKPOINTING": "0",
        "USE_FSDP1": "0",
        "OPD_AUX_GRADIENT_CHECKPOINTING": "0",
        "OPD_SERIAL_STUDENT_CFG": "0",
        "OPD_AUX_EMPTY_CACHE": "0",
        "COSMOS_TRAIN_STEP_PROFILE": "1",
        "OPD_PROFILE": "1",
        "SKIP_TEACHER_COMPILE": "0",
        "CACHE_DATASET_IN_MEMORY": "1",
        "COSMOS_POLICY_VALIDATE_WEIGHTS": "0",
        "ENABLE_LIGHT_EVAL": "1",
        "ENABLE_ROLLOUT_EVAL": "1",
        "ENABLE_STAGE1_START_EVAL": "1",
        "ENABLE_STAGE1_START_EVAL_BASELINE": "1",
        "STOP_AFTER_STEP": "7",
        "ATTN_MODE": "hostile",
        "DATASET_SAMPLE_MANIFEST": "/hostile/manifest.json",
        "STAGE1_CKPT_NAME": "hostile",
        "DISTILL_MODE": "hostile",
        "TEACHER_PATH": "/hostile/teacher",
        "COSMOS_PROGRESSIVE_RUN_ID": "hostile",
        "OPD_AUX_VARIANT": "hostile",
        "OPD_TEACHER_TARGET_MODE": "hostile",
        "ROLLOUT_STEP_PAIRS": "99,99",
        "VIDEO_TRANSITION_PARAM": "hostile",
        "VIDEO_TRANSITION_WEIGHT": "99",
        "OPD_ENDPOINT_AUX_WEIGHT": "99",
        "LOCAL_FM_WEIGHT": "99",
        "OPD_TRANSITION_GROUP_WEIGHT": "99",
        "OPD_ANCHOR_CAP_RATIO": "99",
        "OPD_AUX_USE_NOFSDP_ROLLOUT": "1",
        "ACTION_AWARE_WEIGHT": "99",
        "GT_REGRESSION_WEIGHT": "99",
        "ACTION_TRANSITION_PARAM": "hostile",
        "ACTION_LOCAL_FM_WEIGHT": "99",
        "ACTION_TRANSITION_BLOCK_WEIGHT": "99",
        "ACTION_LOCAL_FM_BLOCK_WEIGHT": "99",
        "LIGHT_EVAL_INTERVAL": "999",
        "LIGHT_EVAL_NUM_BATCHES": "999",
        "LIGHT_EVAL_SEED": "999",
        "LIGHT_EVAL_START_INDEX": "999",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/hostile/hf",
    }
    env.update(hostile)
    output_root = tmp_path / "sealed"
    result = _run(
        "universal",
        "--output-root",
        str(output_root),
        "--run-tag",
        "sealed",
        "--dry-run",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    values = _assignments(result.stdout)
    expected = json.loads(
        canonical_variant_json(
            _resolve_for_launcher(
                env,
                "universal",
                output_root=output_root,
                run_tag="sealed",
            )
        )
    )
    assert json.loads(values["COSMOS_LIBERO_VARIANT_JSON"]) == expected
    assert json.loads(values["CONFIG_IDENTITY_JSON"]) == expected
    for key, canonical in {
        "BETA1": "0.9",
        "BETA2": "0.95",
        "EMA_DECAY": "0.999",
        "EMA_WARMUP_STEPS": "100",
        "DROP_TEXT_RATIO": "0.1",
        "FUSE_GUIDANCE_SCALE": "3.0",
        "CFG_MIN": "3.0",
        "CFG_MAX": "3.0",
        "MAX_GRAD_NORM": "0.3",
        "WARMUP_STEPS": "100",
        "NUM_DDIM_TIMESTEPS_ACTION": "1",
        "DIFFUSION_RATIO": "0.5",
        "CONSISTENCY_RATIO": "0.25",
        "FLOWMAP_RATIO": "0.25",
        "VIDEO_LOSS_WEIGHT": "1.0",
        "ACTION_LOSS_WEIGHT": "1.0",
        "ACTION_BLOCK_WEIGHT": "4.0",
        "COSMOS_POLICY_USE_RAW_INFERENCE": "1",
        "SKIP_TARGET_STUDENT_FOR_COSMOS_LATENT": "1",
        "COSMOS_LATENT_CDIFF_LOSS_WEIGHT": "1.0",
        "COSMOS_LATENT_ENDPOINT_LOSS_WEIGHT": "0.0",
        "COSMOS_LATENT_EPSILON": "0.001",
        "COSMOS_LATENT_T_MIN": "0.8",
        "COSMOS_LATENT_T_MAX": "0.9876543209876543",
        "COSMOS_LATENT_CHANNELS": "16",
        "COSMOS_LATENT_FRAMES": "9",
        "COSMOS_LATENT_HEIGHT": "28",
        "COSMOS_LATENT_WIDTH": "28",
        "COSMOS_LATENT_CENTER_VELOCITY_MODE": "symmetric_average",
        "COSMOS_LATENT_TARGET_MODE": "hybrid_cdiff",
        "COSMOS_LATENT_CDIFF_INTERVAL": "4",
        "OPD_ACTION_ROLLOUT_GRAD_MODE": "last_step",
            "OPD_AUX_INTERVAL": "4",
        "OPD_DANCEOPD_VERIFY_TERMINAL_PRIOR": "1",
        "OPD_DANCEOPD_TERMINAL_PRIOR_TOLERANCE": "2e-06",
        "OPD_DANCEOPD_TERMINAL_PRIOR_WARN_FACTOR": "0.5",
        "OPD_COSMOS_SPATIAL_CROP_SIZE": "28",
        "COSMOS_USE_TEACHER_ACTION_ANCHOR": "1",
        "OPD_JOINT_ACTION_ROLLOUT": "1",
        "MECHANISM_DIAGNOSTICS": "1",
        "MECHANISM_DIAGNOSTIC_INTERVAL": "100",
        "MECHANISM_DIAGNOSTIC_SEED": "42",
        "MECHANISM_DIAGNOSTIC_R": "500",
        "MECHANISM_DIAGNOSTIC_S": "250",
        "MECHANISM_DIAGNOSTIC_TEACHER_STEPS": "8",
        "MECHANISM_COSMOS_T_MIN": "0.8",
        "MECHANISM_COSMOS_T_MAX": "0.9876543209876543",
        "GRADIENT_CHECKPOINTING": "1",
        "USE_FSDP1": "1",
        "OPD_AUX_GRADIENT_CHECKPOINTING": "1",
        "OPD_SERIAL_STUDENT_CFG": "1",
        "OPD_AUX_EMPTY_CACHE": "1",
        "COSMOS_TRAIN_STEP_PROFILE": "0",
        "OPD_PROFILE": "0",
        "SKIP_TEACHER_COMPILE": "1",
        "CACHE_DATASET_IN_MEMORY": "0",
        "COSMOS_POLICY_VALIDATE_WEIGHTS": "1",
        "ENABLE_LIGHT_EVAL": "0",
        "ENABLE_ROLLOUT_EVAL": "0",
        "ENABLE_STAGE1_START_EVAL": "0",
        "ENABLE_STAGE1_START_EVAL_BASELINE": "0",
        "STOP_AFTER_STEP": "0",
        "ATTN_MODE": "flex",
        "DATASET_SAMPLE_MANIFEST": "<unset>",
        "STAGE1_CKPT_NAME": "<unset>",
        "DISTILL_MODE": "<unset>",
        "TEACHER_PATH": "<unset>",
        "COSMOS_PROGRESSIVE_RUN_ID": "<unset>",
        "OPD_AUX_VARIANT": "<unset>",
        "OPD_TEACHER_TARGET_MODE": "<unset>",
        "ROLLOUT_STEP_PAIRS": "<unset>",
        "VIDEO_TRANSITION_PARAM": "<unset>",
        "VIDEO_TRANSITION_WEIGHT": "<unset>",
        "OPD_ENDPOINT_AUX_WEIGHT": "<unset>",
        "LOCAL_FM_WEIGHT": "<unset>",
        "OPD_TRANSITION_GROUP_WEIGHT": "<unset>",
        "OPD_ANCHOR_CAP_RATIO": "<unset>",
        "OPD_AUX_USE_NOFSDP_ROLLOUT": "<unset>",
        "ACTION_AWARE_WEIGHT": "<unset>",
        "GT_REGRESSION_WEIGHT": "<unset>",
        "ACTION_TRANSITION_PARAM": "<unset>",
        "ACTION_LOCAL_FM_WEIGHT": "<unset>",
        "ACTION_TRANSITION_BLOCK_WEIGHT": "<unset>",
        "ACTION_LOCAL_FM_BLOCK_WEIGHT": "<unset>",
        "LIGHT_EVAL_INTERVAL": "1000",
        "LIGHT_EVAL_NUM_BATCHES": "1",
        "LIGHT_EVAL_SEED": "42",
        "LIGHT_EVAL_START_INDEX": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": "/kpfs-intern/jialongliu/models/cosmos_predict2_5/hf_cache",
    }.items():
        assert values[key] == canonical


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


def test_launcher_allows_a_strict_allocator_override_without_changing_default(tmp_path):
    env, _ = _env(tmp_path)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assert _assignments(result.stdout)["PYTORCH_CUDA_ALLOC_CONF"] == (
        "max_split_size_mb:128"
    )


@pytest.mark.parametrize(
    "value", ("max_split_size_mb:0", "max_split_size_mb:128,garbage")
)
def test_launcher_rejects_an_invalid_allocator_override(tmp_path, value):
    env, _ = _env(tmp_path)
    env["PYTORCH_CUDA_ALLOC_CONF"] = value

    result = _run("s4", "--dry-run", env=env)

    assert result.returncode != 0
    assert "PYTORCH_CUDA_ALLOC_CONF" in result.stderr


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


def _write_resume_checkpoint(
    output: Path,
    *,
    arm: str,
    step: int,
    parent,
    variant_json: str,
    write_manifests: bool = True,
):
    checkpoint = output / "checkpoints" / f"step_{step}"
    provenance_json = canonical_json(json.loads(variant_json)["provenance"])
    payload = {
        **STAGE2_CONTRACT,
        "checkpoint_step": step,
        "parent_stage1_path": parent.canonical_path,
        "parent_stage1_contract_identity": parent.contract_identity,
        "student_base_model_path": str(
            Path(parent.canonical_path) / "target_student"
        ),
        "teacher_model_path": json.loads(
            (
                Path(parent.canonical_path)
                / "target_student"
                / "transformer"
                / "config.json"
            ).read_text(encoding="utf-8")
        )["teacher_model_path"],
        "cosmos_libero_variant_json": variant_json,
        "cosmos_libero_provenance_json": provenance_json,
    }
    for student in ("online_student", "target_student"):
        _transformer(checkpoint / student / "transformer", payload)
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "lr_scheduler.pt").write_bytes(b"scheduler")
    if write_manifests:
        output.mkdir(parents=True, exist_ok=True)
        (output / "cosmos_libero_variant.json").write_text(
            variant_json + "\n", encoding="utf-8"
        )
        (output / "cosmos_libero_provenance.json").write_text(
            provenance_json + "\n", encoding="utf-8"
        )
    return checkpoint


def _refresh_dataset_lock(env):
    root = Path(env["DATASET_PATH"])
    payload = build_artifact_lock(
        root,
        compact_paths=(
            "meta/info.json",
            "meta/tasks.jsonl",
            "meta/episodes.jsonl",
            "meta/episodes_ori.jsonl",
            "meta/episodes_stats.jsonl",
            "empty_emb.pt",
        ),
        large_paths=("data/chunk-000/episode_000000.parquet",),
        immutable_store=False,
    )
    (Path(env["COSMOS_PROVENANCE_LOCK_ROOT"]) / "dataset.lock.json").write_text(
        canonical_json(payload) + "\n", encoding="utf-8"
    )


def test_resume_recomputes_provenance_and_rejects_same_path_content_change(
    tmp_path,
):
    env, stage1 = _env(tmp_path)
    output_root = tmp_path / "out"
    own = output_root / "resume" / "apm"
    parent = validate_stage1_parent(stage1)
    record = _resolve_for_launcher(
        env, "apm", output_root=output_root, run_tag="resume"
    )
    _write_resume_checkpoint(
        own,
        arm="apm",
        step=100,
        parent=parent,
        variant_json=canonical_variant_json(record),
    )
    (own / "cosmos_libero_variant.json").write_text(
        canonical_variant_json(record) + "\n", encoding="utf-8"
    )
    (own / "cosmos_libero_provenance.json").write_text(
        canonical_json(
            json.loads(canonical_variant_json(record))["provenance"]
        )
        + "\n",
        encoding="utf-8",
    )

    metadata = Path(env["DATASET_PATH"]) / "meta" / "info.json"
    metadata.write_text('{"revision":2}\n', encoding="utf-8")
    _refresh_dataset_lock(env)
    result = _run(
        "apm",
        "--output-root",
        str(output_root),
        "--run-tag",
        "resume",
        "--resume-step",
        "100",
        "--dry-run",
        env=env,
    )
    assert result.returncode != 0
    assert "provenance" in result.stderr.lower() or "variant" in result.stderr.lower()


def test_resume_is_allowed_only_from_same_arm_with_same_parent_and_variant_record(tmp_path):
    env, stage1 = _env(tmp_path)
    output_root = tmp_path / "out"
    own = output_root / "resume" / "apm"
    parent = validate_stage1_parent(stage1)
    own_record = _resolve_for_launcher(
        env, "apm", output_root=output_root, run_tag="resume"
    )
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

    other_record = _resolve_for_launcher(
        env, "anchor_only", output_root=output_root, run_tag="resume"
    )
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
    ("manifest_name", "corruption"),
    (
        ("cosmos_libero_variant.json", "missing"),
        ("cosmos_libero_variant.json", "symlink"),
        ("cosmos_libero_variant.json", "malformed"),
        ("cosmos_libero_variant.json", "mismatch"),
        ("cosmos_libero_provenance.json", "missing"),
        ("cosmos_libero_provenance.json", "symlink"),
        ("cosmos_libero_provenance.json", "malformed"),
        ("cosmos_libero_provenance.json", "mismatch"),
    ),
)
def test_resume_requires_both_exact_plain_canonical_manifests(
    tmp_path, manifest_name, corruption
):
    env, stage1 = _env(tmp_path)
    output_root = tmp_path / "out"
    own = output_root / "resume" / "apm"
    parent = validate_stage1_parent(stage1)
    record = _resolve_for_launcher(
        env, "apm", output_root=output_root, run_tag="resume"
    )
    _write_resume_checkpoint(
        own,
        arm="apm",
        step=100,
        parent=parent,
        variant_json=canonical_variant_json(record),
    )
    manifest = own / manifest_name
    if corruption == "missing":
        manifest.unlink()
    elif corruption == "symlink":
        target = own / ("target-" + manifest_name)
        target.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
        manifest.unlink()
        manifest.symlink_to(target)
    elif corruption == "malformed":
        manifest.write_text("{not-json}\n", encoding="utf-8")
    else:
        manifest.write_text("{}\n", encoding="utf-8")

    result = _run(
        "apm",
        "--output-root",
        str(output_root),
        "--run-tag",
        "resume",
        "--resume-step",
        "100",
        "--dry-run",
        env=env,
    )

    assert result.returncode != 0
    assert "manifest" in result.stderr.lower()


def test_resume_rejects_a_changed_supported_scientific_identity(tmp_path):
    env, stage1 = _env(tmp_path)
    output_root = tmp_path / "out"
    own = output_root / "resume" / "apm"
    parent = validate_stage1_parent(stage1)
    record = _resolve_for_launcher(
        env, "apm", output_root=output_root, run_tag="resume"
    )
    payload = json.loads(canonical_variant_json(record))
    payload["opd_aux_weight"] = 0.25
    changed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    _write_resume_checkpoint(
        own,
        arm="apm",
        step=100,
        parent=parent,
        variant_json=changed,
    )

    result = _run(
        "apm",
        "--output-root",
        str(output_root),
        "--run-tag",
        "resume",
        "--resume-step",
        "100",
        "--dry-run",
        env=env,
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


def test_forged_pytest_fixture_cannot_reach_formal_launcher_or_torchrun(
    tmp_path,
):
    env, _ = _env(tmp_path)
    fake_repo = env["_TEST_COSMOS_PROJECT_ROOT"]
    env.pop("_TEST_COSMOS_LAUNCHER_SCRIPT")
    env.pop("_TEST_COSMOS_PROJECT_ROOT")
    env["PYTEST_CURRENT_TEST"] = "forged"
    env["COSMOS_PROVENANCE_TEST_FIXTURE"] = "1"
    env["COSMOS_PROVENANCE_TEST_FLASHWAM_REPO"] = fake_repo
    marker = tmp_path / "torchrun-called"
    torchrun = tmp_path / "torchrun"
    torchrun.write_text(
        "#!/bin/sh\nprintf called > \"$TORCHRUN_MARKER\"\n",
        encoding="utf-8",
    )
    torchrun.chmod(torchrun.stat().st_mode | stat.S_IXUSR)
    env["TORCHRUN_BIN"] = str(torchrun)
    env["TORCHRUN_MARKER"] = str(marker)
    output_root = tmp_path / "formal-output"
    result = _run(
        "apm",
        "--steps",
        "1",
        "--output-root",
        str(output_root),
        "--run-tag",
        "forged",
        env=env,
    )

    assert result.returncode != 0
    assert not marker.exists()
    assert not output_root.exists()


def test_production_launcher_has_no_pytest_fixture_override_surface():
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in (
        "COSMOS_PROVENANCE_TEST_FIXTURE",
        "COSMOS_PROVENANCE_TEST_FLASHWAM_REPO",
        "PYTEST_CURRENT_TEST",
    ):
        assert forbidden not in source


def test_ambient_preflight_binary_cannot_reach_output_claim_or_torchrun(
    tmp_path,
):
    env, _ = _env(tmp_path)
    preflight = tmp_path / "forged-preflight"
    preflight_marker = tmp_path / "preflight-called"
    preflight.write_text(
        "#!/bin/sh\nprintf called > \"$PREFLIGHT_MARKER\"\nexec "
        + repr(sys.executable)
        + " \"$@\"\n",
        encoding="utf-8",
    )
    preflight.chmod(preflight.stat().st_mode | stat.S_IXUSR)
    env["PREFLIGHT_BIN"] = str(preflight)
    env["PREFLIGHT_MARKER"] = str(preflight_marker)
    marker = tmp_path / "torchrun-called"
    torchrun = tmp_path / "torchrun"
    torchrun.write_text(
        "#!/bin/sh\nprintf called > \"$TORCHRUN_MARKER\"\n",
        encoding="utf-8",
    )
    torchrun.chmod(torchrun.stat().st_mode | stat.S_IXUSR)
    env["TORCHRUN_BIN"] = str(torchrun)
    env["TORCHRUN_MARKER"] = str(marker)
    output_root = tmp_path / "formal-output"

    result = _run(
        "apm",
        "--steps",
        "1",
        "--output-root",
        str(output_root),
        "--run-tag",
        "forged-preflight",
        env=env,
    )

    assert result.returncode != 0
    assert not preflight_marker.exists()
    assert not marker.exists()
    assert not output_root.exists()


def test_production_launcher_does_not_accept_preflight_bin_override():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'PREFLIGHT_BIN="${PREFLIGHT_BIN:-' not in source


@pytest.mark.parametrize("injection", ("pythonpath", "cwd"))
def test_formal_preflight_does_not_execute_ambient_sitecustomize(
    tmp_path, injection
):
    env, _ = _env(tmp_path)
    hostile = tmp_path / "hostile-python"
    hostile.mkdir()
    marker = tmp_path / "sitecustomize-called"
    (hostile / "sitecustomize.py").write_text(
        "import os\n"
        "open(os.environ['SITECUSTOMIZE_MARKER'], 'w').write('called')\n",
        encoding="utf-8",
    )
    env["SITECUSTOMIZE_MARKER"] = str(marker)
    if injection == "pythonpath":
        env["PYTHONPATH"] = str(hostile)
        cwd = ROOT
    else:
        env.pop("PYTHONPATH", None)
        cwd = hostile

    result = _run(
        "apm",
        "--output-root",
        str(tmp_path / "out"),
        "--run-tag",
        "isolated",
        "--dry-run",
        env=env,
        cwd=cwd,
    )

    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    assert not (tmp_path / "out").exists()


def test_production_preflights_never_append_ambient_pythonpath():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "${PYTHONPATH:-}" not in source
    assert ' -I -E -s -S -B -c "$FORMAL_PREFLIGHT_DRIVER"' in source
    assert "PYTHONDONTWRITEBYTECODE" not in source
    assert "PYTHONHASHSEED" not in source


def test_repeated_dry_and_check_preflights_do_not_write_bytecode(tmp_path):
    env, _ = _env(tmp_path)

    def snapshot():
        return {
            str(path): (
                path.read_bytes(),
                path.stat().st_mtime_ns,
                path.stat().st_ctime_ns,
            )
            for path in ROOT.joinpath("distillation_flowmap").rglob("*")
            if path.name == "__pycache__" or path.suffix == ".pyc"
            if path.is_file()
        }

    before = snapshot()
    for mode in ("--dry-run", "--check-only", "--dry-run", "--check-only"):
        result = _run(
            "apm",
            "--output-root",
            str(tmp_path / "out"),
            "--run-tag",
            "bytecode-read-only",
            mode,
            env=env,
        )
        assert result.returncode == 0, result.stderr
    assert snapshot() == before
    assert not (tmp_path / "out").exists()


def test_formal_default_hashes_same_size_large_artifact_mutation(tmp_path):
    env, _ = _env(tmp_path)
    env.pop("VERIFY_LARGE_ARTIFACT_DIGESTS")
    lock_root = Path(env["COSMOS_PROVENANCE_LOCK_ROOT"])
    for lock_path in lock_root.glob("*.lock.json"):
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        payload["immutable_store"] = True
        lock_path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    large = (
        Path(env["DATASET_PATH"])
        / "data"
        / "chunk-000"
        / "episode_000000.parquet"
    )
    original = large.read_bytes()
    large.write_bytes(bytes((value ^ 0xFF) for value in original))

    result = _run(
        "apm",
        "--output-root",
        str(tmp_path / "out"),
        "--run-tag",
        "large-mutation",
        "--dry-run",
        env=env,
    )

    assert result.returncode != 0
    assert "digest" in result.stderr.lower()


@pytest.mark.parametrize("mode", ("--dry-run", "--check-only"))
def test_launcher_preflight_preserves_git_index_bytes_and_metadata(
    tmp_path, mode
):
    env, _ = _env(tmp_path)
    project = Path(env["_TEST_COSMOS_PROJECT_ROOT"])
    git_dir = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "--git-dir"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_dir = Path(git_dir)
    if not git_dir.is_absolute():
        git_dir = project / git_dir
    index = git_dir.resolve() / "index"
    before = (
        index.read_bytes(),
        index.stat().st_size,
        index.stat().st_mtime_ns,
        index.stat().st_ctime_ns,
    )

    result = _run(
        "apm",
        "--output-root",
        str(tmp_path / "out"),
        "--run-tag",
        "read-only",
        mode,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    after_stat = index.stat()
    assert index.read_bytes() == before[0]
    assert after_stat.st_size == before[1]
    assert after_stat.st_mtime_ns == before[2]
    assert after_stat.st_ctime_ns == before[3]
    assert not index.with_name("index.lock").exists()
    assert not (tmp_path / "out").exists()


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
    target = stage1 / "target_student"
    video_vae_lock = build_artifact_lock(
        target,
        compact_paths=(
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
        ),
        large_paths=(
            "transformer/diffusion_pytorch_model-00001-of-00001.safetensors",
        ),
        immutable_store=False,
    )
    (
        Path(env["COSMOS_PROVENANCE_LOCK_ROOT"]) / "video_vae.lock.json"
    ).write_text(canonical_json(video_vae_lock) + "\n", encoding="utf-8")

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
