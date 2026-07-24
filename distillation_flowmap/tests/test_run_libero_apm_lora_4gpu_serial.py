import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "distillation_flowmap"
    / "ablation"
    / "run_libero_apm_lora_4gpu_serial.sh"
)


def run_dry(tmp_path, *extra):
    command = [
        "bash",
        str(SCRIPT),
        "--phase",
        "all",
        "--gpu-ids",
        "0,1,2,3",
        "--output-root",
        str(tmp_path / "runs"),
        "--dry-run",
        *extra,
    ]
    return subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHON": "/usr/bin/python3"},
        text=True,
        capture_output=True,
    )


def test_dry_run_prints_four_serial_training_commands(tmp_path):
    result = run_dry(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--nproc_per_node=4") == 4
    for arm in ("stage1_only", "anchor_only", "field_only", "apm"):
        assert f"TRAIN arm={arm}" in result.stdout
    assert "step_2000" in result.stdout
    assert "train_manifest.json" in result.stdout
    assert not (tmp_path / "runs").exists()


def test_dry_run_prints_heldout_offline_jobs(tmp_path):
    result = run_dry(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("OFFLINE_EVAL arm=") == 4
    assert result.stdout.count("--student-steps 1 2 4") == 4
    assert result.stdout.count("--teacher-steps 1 2 4") == 4
    assert result.stdout.count("heldout_manifest.json") >= 4


def test_rejects_non_four_gpu_selection(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--phase",
            "train",
            "--gpu-ids",
            "0,1",
            "--output-root",
            str(tmp_path / "runs"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "exactly four" in result.stderr


def test_arm_subset_is_validated_and_preserves_order(tmp_path):
    result = run_dry(tmp_path, "--arms", "anchor_only,apm")

    assert result.returncode == 0, result.stderr
    train_lines = [
        line for line in result.stdout.splitlines() if line.startswith("TRAIN arm=")
    ]
    assert train_lines[0].startswith("TRAIN arm=anchor_only")
    assert train_lines[1].startswith("TRAIN arm=apm")
    assert len(train_lines) == 2
