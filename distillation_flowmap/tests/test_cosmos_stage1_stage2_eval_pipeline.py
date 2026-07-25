import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from distillation_flowmap.cosmos_libero_provenance import (
    canonical_json,
    verify_artifact_lock,
)
from distillation_flowmap.cosmos_libero_variants import resolve_variant


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = (
    ROOT / "distillation_flowmap" / "run_cosmos_stage1_stage2_eval_8gpu.sh"
)
LOCK_CLI = ROOT / "distillation_flowmap" / "prepare_cosmos_libero_provenance_locks.py"


def _plain_file(path: Path, contents: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return path


def _artifact_layout(tmp_path: Path) -> dict[str, Path]:
    dataset = tmp_path / "dataset"
    for name in (
        "info.json",
        "tasks.jsonl",
        "episodes.jsonl",
        "episodes_ori.jsonl",
        "episodes_stats.jsonl",
    ):
        _plain_file(dataset / "meta" / name, name.encode())
    _plain_file(dataset / "empty_emb.pt")
    _plain_file(dataset / "data" / "chunk-000" / "episode_000000.parquet", b"a")
    _plain_file(dataset / "data" / "chunk-001" / "episode_000001.parquet", b"b")

    teacher = tmp_path / "teacher"
    for relative in ("config.json", "libero_dataset_statistics.json"):
        _plain_file(teacher / relative, relative.encode())
    for relative in (
        "Cosmos-Policy-LIBERO-Predict2-2B.pt",
        "libero_t5_embeddings.pkl",
    ):
        _plain_file(teacher / relative, relative.encode())

    local_model = tmp_path / "local-model"
    for relative in (
        "config.json",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer_config.json",
    ):
        _plain_file(local_model / relative, relative.encode())
    for relative in ("model-480p-16fps.pt", "tokenizer/tokenizer.pth"):
        _plain_file(local_model / relative, relative.encode())

    stage1_target = tmp_path / "stage1-target"
    _plain_file(stage1_target / "transformer" / "config.json", b"{}")
    _plain_file(
        stage1_target
        / "transformer"
        / "diffusion_pytorch_model-00001-of-00002.safetensors",
        b"one",
    )
    _plain_file(
        stage1_target
        / "transformer"
        / "diffusion_pytorch_model-00002-of-00002.safetensors",
        b"two",
    )
    _plain_file(
        stage1_target
        / "transformer"
        / "diffusion_pytorch_model.safetensors.index.json",
        b'{"weight_map":{"a":"diffusion_pytorch_model-00001-of-00002.safetensors","b":"diffusion_pytorch_model-00002-of-00002.safetensors"}}',
    )
    return {
        "dataset": dataset,
        "teacher": teacher,
        "local_model": local_model,
        "stage1_target": stage1_target,
    }


def _lock_command(layout: dict[str, Path], output: Path, *extra: str) -> list[str]:
    return [
        sys.executable,
        str(LOCK_CLI),
        "--output-root",
        str(output),
        "--dataset-root",
        str(layout["dataset"]),
        "--teacher-root",
        str(layout["teacher"]),
        "--local-model-root",
        str(layout["local_model"]),
        "--stage1-target-root",
        str(layout["stage1_target"]),
        *extra,
    ]


def test_lock_cli_is_canonical_complete_atomic_and_no_overwrite(tmp_path):
    layout = _artifact_layout(tmp_path)
    output = tmp_path / "locks"
    result = subprocess.run(
        _lock_command(layout, output),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in output.iterdir()) == [
        "dataset.lock.json",
        "local_model.lock.json",
        "teacher.lock.json",
        "video_vae.lock.json",
    ]
    for name, root in (
        ("dataset", layout["dataset"]),
        ("teacher", layout["teacher"]),
        ("local_model", layout["local_model"]),
        ("video_vae", layout["stage1_target"]),
    ):
        path = output / f"{name}.lock.json"
        raw = path.read_text(encoding="utf-8")
        assert raw.endswith("\n")
        assert canonical_json(json.loads(raw)) + "\n" == raw
        verify_artifact_lock(
            root,
            path,
            verify_large_artifact_digests=True,
        )
    dataset_entries = {
        item["path"]
        for item in json.loads(
            (output / "dataset.lock.json").read_text(encoding="utf-8")
        )["entries"]
    }
    assert dataset_entries == {
        "empty_emb.pt",
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_ori.jsonl",
        "meta/episodes_stats.jsonl",
        "data/chunk-000/episode_000000.parquet",
        "data/chunk-001/episode_000001.parquet",
    }
    second = subprocess.run(
        _lock_command(layout, output),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert second.returncode != 0
    assert "exists" in second.stderr.lower()


def test_lock_cli_dry_run_is_write_free(tmp_path):
    layout = _artifact_layout(tmp_path)
    output = tmp_path / "locks"
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    result = subprocess.run(
        _lock_command(layout, output, "--dry-run"),
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert result.returncode == 0, result.stderr
    assert before == after
    assert not output.exists()
    assert "dataset.lock.json=" in result.stdout
    assert "video_vae.lock.json=" in result.stdout


def test_universal_video_action_retains_target_required_by_action_opd(tmp_path):
    common = {
        "output_root": tmp_path,
        "run_tag": "contract",
        "steps": 123,
        "save_interval": 17,
        "master_port": 30123,
    }
    video = dict(resolve_variant("universal", **common))
    action = dict(resolve_variant("universal-video-action", **common))
    assert video["action_opd_enabled"] is False
    assert action["action_opd_enabled"] is True
    assert video["skip_target_student_for_cosmos_latent"] is True
    assert action["skip_target_student_for_cosmos_latent"] is False
    for key in (
        "name",
        "output_dir",
        "action_opd_enabled",
        "skip_target_student_for_cosmos_latent",
    ):
        video.pop(key)
        action.pop(key)
    assert video == action


def _write_executable(path: Path, body: str) -> Path:
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + body,
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _pipeline_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    calls = tmp_path / "calls.log"
    stage1 = _write_executable(
        tmp_path / "stage1.sh",
        'printf "stage1:%s\\n" "$*" >> "$CALLS_LOG"\n'
        'if [[ "$1" == run ]]; then\n'
        '  output=""; steps=""; while (( $# )); do\n'
        '    case "$1" in --output-dir) output="$2"; shift 2;;'
        ' --steps) steps="$2"; shift 2;; *) shift;; esac\n'
        "  done\n"
        '  for variant in online_student target_student; do\n'
        '    root="$output/checkpoints/step_$steps/$variant/transformer"\n'
        '    mkdir -p "$root"\n'
        "    printf "
        '\'{\"contract_version\":2,\"training_contract_stage\":\"raw_stage1\",'
        '\"action_packing_schema\":\"downsample_survivor_v2\",'
        '\"action_downsample_factor\":4,\"action_chunk_shape\":[4,4],'
        '\"checkpoint_step\":%s,\"student_backend\":\"wan_flowmap\",'
        '\"teacher_backend\":\"cosmos_policy\",'
        '\"student_base_model_path\":\"%s\",\"teacher_model_path\":\"%s\"}\\n\' '
        '"$steps" "$WAN_STUDENT_BASE_MODEL_PATH" "$COSMOS_POLICY_PATH" '
        '> "$root/config.json"\n'
        '    printf x > "$root/diffusion_pytorch_model.safetensors"\n'
        '  done\n'
        "fi\n",
    )
    stage2 = _write_executable(
        tmp_path / "stage2.sh",
        'printf "stage2:%s\\n" "$*" >> "$CALLS_LOG"\n'
        'if [[ ! " $* " =~ " --dry-run " && ! " $* " =~ " --check-only " ]]; then\n'
        '  root="$PIPELINE_RUN_ROOT/universal-video-action/checkpoints/step_$PIPELINE_STAGE2_STEPS/target_student/transformer"\n'
        '  mkdir -p "$root"; printf "{}\\n" > "$root/config.json"\n'
        '  printf x > "$root/diffusion_pytorch_model.safetensors"\n'
        "fi\n",
    )
    eval_script = _write_executable(
        tmp_path / "eval.sh",
        'printf "eval:%s:prompt=%s\\n" "$*" "$S4_PROMPT_TABLE" >> "$CALLS_LOG"\n',
    )
    lock_script = _write_executable(
        tmp_path / "locks.sh",
        'printf "locks:%s\\n" "$*" >> "$CALLS_LOG"\n'
        'output=""; while (( $# )); do\n'
        '  case "$1" in --output-root) output="$2"; shift 2;; *) shift;; esac\n'
        "done\n"
        'mkdir -p "$output"; for name in dataset teacher local_model video_vae; do\n'
        '  printf "{}\\n" > "$output/$name.lock.json"\n'
        "done\n",
    )
    prompt_builder = _write_executable(
        tmp_path / "prompt-builder.sh",
        'output=""; validate=0; read_only=0\n'
        'while (( $# )); do\n'
        '  case "$1" in\n'
        '    --output) output="$2"; shift 2;;\n'
        '    --validate-only) validate=1; shift;;\n'
        '    --dry-run) read_only=1; shift;;\n'
        '    *) shift;;\n'
        '  esac\n'
        'done\n'
        'if (( read_only )); then exit 0; fi\n'
        'printf "prompt:%s:%s\\n" "$validate" "$output" >> "$CALLS_LOG"\n'
        'if (( validate )); then\n'
        '  [[ -f "$output" ]] || { printf "missing prompt table\\n" >&2; exit 3; }\n'
        'else\n'
        '  [[ ! -e "$output" ]] || { printf "refusing overwrite\\n" >&2; exit 4; }\n'
        '  mkdir -p "$(dirname "$output")"; printf table > "$output"\n'
        'fi\n',
    )
    init = tmp_path / "lingbotva-init"
    _plain_file(
        init / "transformer" / "config.json",
        b'{"_class_name":"WanTransformer3DModel"}\n',
    )
    _plain_file(init / "transformer" / "diffusion_pytorch_model.safetensors")
    dataset = tmp_path / "dataset"
    _plain_file(dataset / "empty_emb.pt")
    prompt_table = _plain_file(dataset / "all_40_prompt_embeddings.pt")
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    _plain_file(teacher / "config.json", b'{"model_type":"cosmos-policy"}\n')
    for name in (
        "Cosmos-Policy-LIBERO-Predict2-2B.pt",
        "libero_dataset_statistics.json",
        "libero_t5_embeddings.pkl",
    ):
        _plain_file(teacher / name)
    local_model = tmp_path / "local-model"
    local_model.mkdir()
    worker_python = _write_executable(
        tmp_path / "cosmos-worker-python",
        "exit 0\n",
    )
    cosmos_cuda = tmp_path / "cosmos-worker-pythonpath" / "cosmos-cuda"
    cosmos_oss = tmp_path / "cosmos-worker-pythonpath" / "cosmos-oss"
    cosmos_cuda.mkdir(parents=True)
    cosmos_oss.mkdir(parents=True)
    repo = tmp_path / "clean-cosmos-repo"
    audited_source = Path(
        os.environ.get(
            "COSMOS_AUDITED_REPO_SOURCE",
            "/kpfs-intern/jialongliu/projects/cosmos-predict2.5-formal-441b897",
        )
    )
    subprocess.run(
        ["git", "clone", "-q", "--shared", "--no-checkout", str(audited_source), str(repo)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "sparse-checkout", "init", "--no-cone"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "sparse-checkout",
            "set",
            "cosmos_predict2/_src/predict2/cosmos_policy/experiments/robot/cosmos_utils.py",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "checkout",
            "-q",
            "--detach",
            "1eb8457072b4a1adfe1f83c3076e4aa5452cbab2",
        ],
        check=True,
    )
    output_root = tmp_path / "outputs"
    env = os.environ.copy()
    env.update(
        {
            "WAN_STUDENT_BASE_MODEL_PATH": str(init),
            "COSMOS_PREDICT2_REPO": str(repo),
            "DATASET_PATH": str(dataset),
            "COSMOS_POLICY_PATH": str(teacher),
            "S4_PROMPT_TABLE": str(prompt_table),
            "COSMOS_PREDICT25_LOCAL_MODEL_DIR": str(local_model),
            "COSMOS_POLICY_PYTHON": str(worker_python),
            "COSMOS_POLICY_EXTRA_PYTHONPATH": f"{cosmos_cuda}:{cosmos_oss}",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "CALLS_LOG": str(calls),
            "COSMOS_RAW_STAGE1_LAUNCHER": str(stage1),
            "COSMOS_STAGE2_LAUNCHER": str(stage2),
            "COSMOS_JOINT124_EVAL_LAUNCHER": str(eval_script),
            "COSMOS_LOCK_PREPARER": str(lock_script),
            "COSMOS_WAN_PROMPT_TABLE_BUILDER": str(prompt_builder),
        }
    )
    return env, output_root


def _pipeline(
    env: dict[str, str],
    output_root: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--phase",
            "all",
            "--output-root",
            str(output_root),
            "--run-tag",
            "serial",
            *extra,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_pipeline_formal_calls_stage1_locks_stage2_eval_in_order(tmp_path):
    env, output_root = _pipeline_env(tmp_path)
    result = _pipeline(env, output_root)
    assert result.returncode == 0, result.stderr
    lines = Path(env["CALLS_LOG"]).read_text(encoding="utf-8").splitlines()
    assert [line.split(":", 1)[0] for line in lines] == [
        "prompt",
        "stage1",
        "locks",
        "stage2",
        "eval",
    ]
    assert lines[0].startswith("prompt:1:")
    assert "run --steps 5000 --save-interval 1000 --master-port 29671" in lines[1]
    assert "universal-video-action --steps 10000 --save-interval 1000" in lines[3]
    assert "--master-port 29672" in lines[3]
    assert "stage2_target" in result.stdout
    assert "official_teacher" in result.stdout
    assert "40 tasks" in result.stdout
    assert "2000 episodes per role/K" in result.stdout
    eval_call = lines[4]
    assert "run" in eval_call


@pytest.mark.parametrize("mode", ("--dry-run", "--check-only"))
def test_pipeline_read_only_modes_write_nothing(tmp_path, mode):
    env, output_root = _pipeline_env(tmp_path)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    result = _pipeline(env, output_root, mode)
    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert result.returncode == 0, result.stderr
    assert before == after
    assert not Path(env["CALLS_LOG"]).exists()
    assert "universal-video-action" in result.stdout
    assert "stage2_target" in result.stdout
    for name in (
        "RESUME_FROM_PATH",
        "PARENT_STAGE1_PATH",
        "PARENT_STAGE1_CONTRACT_IDENTITY",
        "STAGE2_LINEAGE_JSON",
    ):
        assert result.stdout.count(f"{name}=") >= 2
    assert "PARENT_STAGE1_CONTRACT_IDENTITY=__DERIVED_AFTER_STAGE1__" in result.stdout
    assert "official_teacher" in result.stdout
    assert "COSMOS_POLICY_PATH=" in result.stdout
    assert "COSMOS_POLICY_TEACHER_LOCK=" in result.stdout
    assert (
        "COSMOS_PREDICT2_REPO_COMMIT="
        "1eb8457072b4a1adfe1f83c3076e4aa5452cbab2"
    ) in result.stdout
    assert "S4_PROMPT_TABLE=" in result.stdout
    evaluation = next(
        line for line in result.stdout.splitlines()
        if line.startswith("EVAL_COMMAND=")
    )
    assert f"COSMOS_POLICY_PYTHON={env['COSMOS_POLICY_PYTHON']}" in evaluation
    assert (
        f"COSMOS_POLICY_EXTRA_PYTHONPATH="
        f"{env['COSMOS_POLICY_EXTRA_PYTHONPATH']}"
    ) in evaluation


@pytest.mark.parametrize(
    "missing_name",
    ("COSMOS_POLICY_PYTHON", "COSMOS_POLICY_EXTRA_PYTHONPATH"),
)
def test_pipeline_requires_explicit_cosmos_worker_environment(
    tmp_path, missing_name
):
    env, output_root = _pipeline_env(tmp_path)
    env.pop(missing_name)
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    result = _pipeline(env, output_root, "--dry-run")

    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert result.returncode != 0
    assert missing_name in result.stderr
    assert before == after
    assert not output_root.exists()
    assert not Path(env["CALLS_LOG"]).exists()


@pytest.mark.parametrize("invalid_kind", ("python", "pythonpath"))
def test_pipeline_rejects_invalid_cosmos_worker_environment_without_writes(
    tmp_path, invalid_kind
):
    env, output_root = _pipeline_env(tmp_path)
    if invalid_kind == "python":
        invalid = tmp_path / "not-executable-python"
        invalid.write_text("#!/bin/sh\n", encoding="utf-8")
        env["COSMOS_POLICY_PYTHON"] = str(invalid)
        expected = "COSMOS_POLICY_PYTHON"
    else:
        env["COSMOS_POLICY_EXTRA_PYTHONPATH"] += f":{tmp_path / 'missing'}"
        expected = "COSMOS_POLICY_EXTRA_PYTHONPATH"
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    result = _pipeline(env, output_root, "--dry-run")

    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert result.returncode != 0
    assert expected in result.stderr
    assert before == after
    assert not output_root.exists()
    assert not Path(env["CALLS_LOG"]).exists()


def test_pipeline_rejects_cosmos_repo_not_at_audited_commit(tmp_path):
    env, output_root = _pipeline_env(tmp_path)
    repo = Path(env["COSMOS_PREDICT2_REPO"])
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "-q", "--detach", "HEAD^"],
        check=True,
    )

    result = _pipeline(env, output_root, "--dry-run")

    assert result.returncode != 0
    assert "audited commit" in result.stderr


def test_pipeline_builds_missing_all_40_prompt_table_after_stage2_before_eval(tmp_path):
    env, output_root = _pipeline_env(tmp_path)
    env.pop("S4_PROMPT_TABLE")

    result = _pipeline(env, output_root)

    assert result.returncode == 0, result.stderr
    lines = Path(env["CALLS_LOG"]).read_text(encoding="utf-8").splitlines()
    assert [line.split(":", 1)[0] for line in lines] == [
        "stage1",
        "locks",
        "stage2",
        "prompt",
        "eval",
    ]
    assert lines[3].startswith("prompt:0:")
    generated = output_root / "serial" / "libero_wan_prompt_embeddings_all40.pt"
    assert generated.is_file()
    assert f"prompt={generated}" in lines[4]


def test_pipeline_missing_prompt_table_dry_run_prints_build_without_writes(tmp_path):
    env, output_root = _pipeline_env(tmp_path)
    env.pop("S4_PROMPT_TABLE")
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))

    result = _pipeline(env, output_root, "--dry-run")

    after = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    assert result.returncode == 0, result.stderr
    assert before == after
    assert "PROMPT_TABLE_MODE=build" in result.stdout
    assert "PROMPT_TABLE_COMMAND=" in result.stdout
    assert "--dry-run" in result.stdout
    assert not Path(env["CALLS_LOG"]).exists()


def test_pipeline_python_prompt_builder_receives_project_pythonpath(tmp_path):
    env, output_root = _pipeline_env(tmp_path)
    builder = tmp_path / "python-prompt-builder.py"
    builder.write_text(
        "import distillation_flowmap.cosmos_wan_prompt_table\n",
        encoding="utf-8",
    )
    builder.chmod(builder.stat().st_mode | stat.S_IXUSR)
    env["COSMOS_WAN_PROMPT_TABLE_BUILDER"] = str(builder)
    env.pop("PYTHONPATH", None)

    result = _pipeline(env, output_root, "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "PROMPT_TABLE_COMMAND=" in result.stdout


def test_eval_only_reuses_generated_table_by_validation(tmp_path):
    env, output_root = _pipeline_env(tmp_path)
    env.pop("S4_PROMPT_TABLE")
    run_root = output_root / "serial"
    generated = _plain_file(run_root / "libero_wan_prompt_embeddings_all40.pt")
    transformer = (
        run_root
        / "universal-video-action/checkpoints/step_10000/target_student/transformer"
    )
    _plain_file(transformer / "config.json", b"{}")
    _plain_file(transformer / "diffusion_pytorch_model.safetensors")
    config = {
        "contract_version": 2,
        "training_contract_stage": "raw_stage1",
        "action_packing_schema": "downsample_survivor_v2",
        "action_downsample_factor": 4,
        "action_chunk_shape": [4, 4],
        "checkpoint_step": 5000,
        "student_backend": "wan_flowmap",
        "teacher_backend": "cosmos_policy",
        "student_base_model_path": env["WAN_STUDENT_BASE_MODEL_PATH"],
        "teacher_model_path": env["COSMOS_POLICY_PATH"],
    }
    for variant in ("online_student", "target_student"):
        stage1_transformer = (
            run_root / f"stage1/checkpoints/step_5000/{variant}/transformer"
        )
        _plain_file(
            stage1_transformer / "config.json",
            json.dumps(config, sort_keys=True).encode(),
        )
        _plain_file(stage1_transformer / "diffusion_pytorch_model.safetensors")
    for name in ("teacher.lock.json",):
        _plain_file(run_root / "provenance-locks" / name, b"{}")

    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--phase",
            "eval",
            "--output-root",
            str(output_root),
            "--run-tag",
            "serial",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    lines = Path(env["CALLS_LOG"]).read_text(encoding="utf-8").splitlines()
    assert lines[0] == f"prompt:1:{generated}"
    assert lines[1].startswith("eval:")


def test_pipeline_requires_explicit_student_init_and_clean_cosmos_repo(tmp_path):
    env, output_root = _pipeline_env(tmp_path)
    env.pop("WAN_STUDENT_BASE_MODEL_PATH")
    missing_init = _pipeline(env, output_root, "--dry-run")
    assert missing_init.returncode != 0
    assert "WAN_STUDENT_BASE_MODEL_PATH" in missing_init.stderr

    env, output_root = _pipeline_env(tmp_path / "second")
    env.pop("COSMOS_PREDICT2_REPO")
    missing_repo = _pipeline(env, output_root, "--dry-run")
    assert missing_repo.returncode != 0
    assert "COSMOS_PREDICT2_REPO" in missing_repo.stderr


def test_pipeline_rejects_overwrite_and_supports_exact_stage_resume(tmp_path):
    env, output_root = _pipeline_env(tmp_path)
    existing = output_root / "serial"
    existing.mkdir(parents=True)
    overwrite = _pipeline(env, output_root)
    assert overwrite.returncode != 0
    assert "exists" in overwrite.stderr.lower()

    stage1_output = existing / "stage1"
    checkpoint = stage1_output / "checkpoints" / "step_1000"
    checkpoint.mkdir(parents=True)
    result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--phase",
            "stage1",
            "--output-root",
            str(output_root),
            "--run-tag",
            "serial",
            "--resume-stage",
            "stage1",
            "--resume-step",
            "1000",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    call = Path(env["CALLS_LOG"]).read_text(encoding="utf-8")
    assert "--resume-step 1000" in call
