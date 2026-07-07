import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "distillation_flowmap" / "ablation" / "launch_robotwin_stepwam_ablation.py"


def run_dry_run(tmp_path, variant):
    args = [
        sys.executable,
        str(LAUNCHER),
        "--variant",
        variant,
        "--seed",
        "0",
        "--root",
        str(tmp_path / "ablation_root"),
        "--teacher-model-path",
        "/tmp/teacher",
        "--dataset-path",
        "/tmp/dataset",
        "--stage1-steps",
        "5",
        "--stage2-steps",
        "7",
        "--master-port",
        "29990",
        "--dry-run",
    ]
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_full_variant_dry_run_contains_hybrid_opd_env(tmp_path):
    result = run_dry_run(tmp_path, "full_stepwam")

    assert "full_stepwam/seed_0" in result.stdout
    assert "OPD_SAME_STATE_VELOCITY_WEIGHT=0.1" in result.stdout
    assert "OPD_ENDPOINT_AUX_WEIGHT=0.1" in result.stdout
    assert "USE_OPD_AUX=1" in result.stdout
    assert not (tmp_path / "ablation_root").exists()


def test_without_opd_dry_run_disables_opd(tmp_path):
    result = run_dry_run(tmp_path, "w_o_opd")

    assert "w_o_opd/seed_0" in result.stdout
    assert "USE_OPD_AUX=0" in result.stdout


def test_local_variant_dry_run_contains_adjacent_grid_env(tmp_path):
    result = run_dry_run(tmp_path, "local_adjacent_only")

    assert "local_adjacent_only/seed_0" in result.stdout
    assert "FLOWMAP_PAIR_MODE=adjacent_grid" in result.stdout
    assert "OPD_PAIR_MODE=adjacent_grid" in result.stdout
    assert "FLOWMAP_ADJACENT_GRID=1000,750,500,250,0" in result.stdout


def test_default_paths_use_robotwin_teacher_dataset_and_empty_emb(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--variant",
            "full_stepwam",
            "--seed",
            "0",
            "--root",
            str(tmp_path / "ablation_root"),
            "--stage1-steps",
            "5",
            "--stage2-steps",
            "7",
            "--master-port",
            "29990",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "lingbot-va-posttrain-robotwin" in result.stdout
    assert "lerobot_robotwin_eef_aug_500" in result.stdout
    assert "EMPTY_EMB_PATH=" in result.stdout
    assert "/root/nas/junjie/conda_envs/any_wam/bin/torchrun" in result.stdout
