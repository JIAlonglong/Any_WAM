import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "distillation_flowmap" / "ablation" / "launch_robotwin_stepwam_ablation.py"


def parse_dry_run_manifest(stdout):
    return json.loads(stdout[stdout.index("{"):])


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
    assert "ENABLE_GRAD_BRANCH_DIAGNOSTICS=1" in result.stdout
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


def test_default_dry_run_filters_to_representative_robotwin_subset(tmp_path):
    result = run_dry_run(tmp_path, "full_stepwam")

    assert "DATASET_TASK_FILTER=" in result.stdout
    assert "place_a2b_right" in result.stdout
    assert "rotate_qrcode" in result.stdout
    assert "place_burger_fries" in result.stdout


def test_dry_run_accepts_single_task_smoke_limits(tmp_path):
    args = [
        sys.executable,
        str(LAUNCHER),
        "--variant",
        "full_stepwam",
        "--seed",
        "0",
        "--root",
        str(tmp_path / "ablation_root"),
        "--teacher-model-path",
        "/tmp/teacher",
        "--dataset-path",
        "/tmp/dataset",
        "--task-filter",
        "place_a2b_right",
        "--max-episodes-per-task",
        "5",
        "--max-samples-per-task",
        "5",
        "--stage1-steps",
        "5",
        "--stage2-steps",
        "7",
        "--master-port",
        "29990",
        "--dry-run",
    ]
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert "DATASET_TASK_FILTER=place_a2b_right" in result.stdout
    assert "DATASET_MAX_EPISODES_PER_TASK=5" in result.stdout
    assert "DATASET_MAX_SAMPLES_PER_TASK=5" in result.stdout


def test_core4_task_preset_limits_dry_run_to_protocol_tasks(tmp_path):
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
            "--teacher-model-path",
            "/tmp/teacher",
            "--dataset-path",
            "/tmp/dataset",
            "--task-preset",
            "core4",
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

    manifest = parse_dry_run_manifest(result.stdout)
    assert manifest["task_preset"] == "core4"
    assert manifest["selected_task_filter"] == [
        "place_a2b_right",
        "stack_bowls_three",
        "open_microwave",
        "pick_dual_bottles",
    ]
    assert (
        "DATASET_TASK_FILTER=place_a2b_right,stack_bowls_three,open_microwave,pick_dual_bottles"
        in result.stdout
    )
    assert "rotate_qrcode" not in manifest["selected_task_filter"]


def test_stage2_can_resume_from_shared_stage1_checkpoint(tmp_path):
    args = [
        sys.executable,
        str(LAUNCHER),
        "--variant",
        "endpoint_only_opd",
        "--seed",
        "1",
        "--root",
        str(tmp_path / "ablation_root"),
        "--teacher-model-path",
        "/tmp/teacher",
        "--dataset-path",
        "/tmp/dataset",
        "--task-preset",
        "core4",
        "--stage1-steps",
        "5",
        "--stage2-steps",
        "7",
        "--master-port",
        "29990",
        "--stage",
        "stage2",
        "--use-shared-stage1",
        "--dry-run",
    ]
    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    expected = (
        f"RESUME_FROM_PATH={tmp_path}/ablation_root/shared_stage1/core4/seed_1/"
        "stage1/checkpoints/step_5"
    )
    assert expected in result.stdout
    assert "shared_stage1_ckpt" in result.stdout
    assert "endpoint_only_opd/seed_1/stage2" in result.stdout


def test_protocol_manifest_paths_are_recorded_in_dry_run(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "--variant",
            "velocity_only_opd",
            "--seed",
            "2",
            "--root",
            str(tmp_path / "ablation_root"),
            "--teacher-model-path",
            "/tmp/teacher",
            "--dataset-path",
            "/tmp/dataset",
            "--task-preset",
            "core4",
            "--protocol-seed",
            "9",
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

    assert "train_manifest_path" in result.stdout
    assert "heldout_eval_manifest_path" in result.stdout
    assert "eval_pairs_path" in result.stdout
    assert "protocol_seed" in result.stdout
    assert "protocol/manifests/core4_protocol_seed_9" in result.stdout
    assert "DATASET_SAMPLE_MANIFEST=" in result.stdout
    assert "train_manifest.json" in result.stdout
