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
        '  root="$output/checkpoints/step_$steps/target_student/transformer"\n'
        '  mkdir -p "$root"; printf "{}\\n" > "$root/config.json"\n'
        '  printf x > "$root/diffusion_pytorch_model.safetensors"\n'
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
        'printf "eval:%s\\n" "$*" >> "$CALLS_LOG"\n',
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
    init = tmp_path / "lingbotva-init"
    _plain_file(
        init / "transformer" / "config.json",
        b'{"_class_name":"WanTransformer3DModel"}\n',
    )
    _plain_file(init / "transformer" / "diffusion_pytorch_model.safetensors")
    dataset = tmp_path / "dataset"
    _plain_file(dataset / "empty_emb.pt")
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
    repo = tmp_path / "clean-cosmos-repo"
    repo.mkdir()
    _plain_file(repo / "source.py", b"revision = 1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "fixture@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    output_root = tmp_path / "outputs"
    env = os.environ.copy()
    env.update(
        {
            "WAN_STUDENT_BASE_MODEL_PATH": str(init),
            "COSMOS_PREDICT2_REPO": str(repo),
            "DATASET_PATH": str(dataset),
            "COSMOS_POLICY_PATH": str(teacher),
            "COSMOS_PREDICT25_LOCAL_MODEL_DIR": str(local_model),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "CALLS_LOG": str(calls),
            "COSMOS_RAW_STAGE1_LAUNCHER": str(stage1),
            "COSMOS_STAGE2_LAUNCHER": str(stage2),
            "COSMOS_JOINT124_EVAL_LAUNCHER": str(eval_script),
            "COSMOS_LOCK_PREPARER": str(lock_script),
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
        "stage1",
        "locks",
        "stage2",
        "eval",
    ]
    assert "run --steps 5000 --save-interval 1000 --master-port 29671" in lines[0]
    assert "universal-video-action --steps 10000 --save-interval 1000" in lines[2]
    assert "--master-port 29672" in lines[2]
    assert "stage2_target" in result.stdout
    assert "LIBERO-10" in result.stdout
    assert "500" in result.stdout


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
