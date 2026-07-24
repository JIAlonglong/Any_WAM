import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "distillation_flowmap" / "run_libero_video_only_opd_stage2_8gpu.sh"


def _write_executable(path: Path):
    path.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "${FAKE_PYTHON_LOG:-/dev/null}"\nexit 0\n',
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _layout(tmp_path: Path):
    stage1 = tmp_path / "stage1" / "step_2000"
    for variant in ("online_student", "target_student"):
        transformer = stage1 / variant / "transformer"
        transformer.mkdir(parents=True)
        (transformer / "config.json").write_text(
            '{"checkpoint_step": 2000}\n', encoding="utf-8"
        )
        (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    teacher = tmp_path / "teacher"
    (teacher / "transformer").mkdir(parents=True)
    (teacher / "transformer" / "config.json").write_text("{}\n", encoding="utf-8")
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "meta" / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
    empty = dataset / "empty_emb.pt"
    empty.write_bytes(b"x")
    python = tmp_path / "python"
    _write_executable(python)
    return stage1, teacher, dataset, empty, python


def _env(tmp_path: Path):
    stage1, teacher, dataset, empty, python = _layout(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "STAGE1_CKPT": str(stage1),
            "TEACHER_MODEL_PATH": str(teacher),
            "DATASET_PATH": str(dataset),
            "EMPTY_EMB_PATH": str(empty),
            "PYTHON": str(python),
            "OUTPUT_DIR": str(tmp_path / "output"),
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        }
    )
    return env


def _run(*args, env):
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_prints_full_eight_gpu_video_only_contract(tmp_path):
    env = _env(tmp_path)
    log = tmp_path / "python.log"
    env["FAKE_PYTHON_LOG"] = str(log)
    result = _run("--dry-run", env=env)

    assert result.returncode == 0, result.stdout + result.stderr
    expected = (
        "CONFIG_FILE=distillation_flowmap."
        "config_libero_fullfinetune_stage2_video_only_opd"
    )
    assert expected in result.stdout
    assert "MAX_TRAIN_STEPS=10000" in result.stdout
    assert "SAVE_INTERVAL=1000" in result.stdout
    assert "OPD_DANCEOPD_ROLLOUT_STEPS=2,4" in result.stdout
    assert "OPD_DANCEOPD_ACTION_VELOCITY_WEIGHT=0.0" in result.stdout
    assert "OPD_AUX_ACTION=0" in result.stdout
    assert "OPD_JOINT_ACTION_ROLLOUT=0" in result.stdout
    assert "MECHANISM_DIAGNOSTIC_INTERVAL=50" in result.stdout
    assert "--nproc_per_node=8" in result.stdout
    assert "--master_port=29659" in result.stdout
    assert 'import_module(os.environ["CONFIG_FILE"])' in log.read_text()
    assert not Path(env["OUTPUT_DIR"]).exists()


def test_dry_run_rejects_non_eight_or_duplicate_gpu_layout(tmp_path):
    env = _env(tmp_path)
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    short = _run("--dry-run", env=env)
    assert short.returncode != 0
    assert "exactly 8" in short.stderr

    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,6"
    duplicate = _run("--dry-run", env=env)
    assert duplicate.returncode != 0
    assert "duplicate" in duplicate.stderr


def test_dry_run_rejects_existing_output_and_missing_stage1_weight(tmp_path):
    env = _env(tmp_path)
    Path(env["OUTPUT_DIR"]).mkdir()
    existing = _run("--dry-run", env=env)
    assert existing.returncode != 0
    assert "Refusing to reuse" in existing.stderr

    Path(env["OUTPUT_DIR"]).rmdir()
    weight = (
        Path(env["STAGE1_CKPT"])
        / "target_student"
        / "transformer"
        / "diffusion_pytorch_model.safetensors"
    )
    weight.unlink()
    missing = _run("--dry-run", env=env)
    assert missing.returncode != 0
    assert "target_student diffusion weights" in missing.stderr


def test_cli_overrides_steps_save_interval_and_port(tmp_path):
    env = _env(tmp_path)
    result = _run(
        "--dry-run",
        "--steps",
        "20",
        "--save-interval",
        "10",
        "--master-port",
        "30001",
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MAX_TRAIN_STEPS=20" in result.stdout
    assert "SAVE_INTERVAL=10" in result.stdout
    assert "--master_port=30001" in result.stdout
