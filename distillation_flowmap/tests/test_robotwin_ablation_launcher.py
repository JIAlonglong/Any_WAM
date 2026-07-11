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


def run_calibration_dry_run(tmp_path, variant):
    args = [
        sys.executable,
        str(LAUNCHER),
        "--variant",
        variant,
        "--seed",
        "0",
        "--root",
        str(tmp_path / "calibration_root"),
        "--teacher-model-path",
        "/tmp/teacher",
        "--dataset-path",
        "/tmp/dataset",
        "--task-preset",
        "core2",
        "--train-samples-per-task",
        "20",
        "--heldout-samples-per-task",
        "10",
        "--stage2-steps",
        "750",
        "--stage1-ckpt",
        "/tmp/shared-stage1",
        "--stage",
        "stage2",
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
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in result.stdout
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


def test_calibration_full_grad_uses_explicit_small_protocol(tmp_path):
    result = run_calibration_dry_run(tmp_path, "calib_full_full_grad")
    manifest = parse_dry_run_manifest(result.stdout)
    env = manifest["stage2_env"]

    assert manifest["selected_task_filter"] == [
        "place_a2b_right",
        "open_microwave",
    ]
    assert manifest["stage1_ckpt"] == "/tmp/shared-stage1"
    assert manifest["train_samples_per_task"] == 20
    assert manifest["heldout_samples_per_task"] == 10
    assert env["OPD_LOSS_COMPOSITION"] == "explicit_hybrid"
    assert env["OPD_ROLLOUT_GRAD_MODE"] == "full"
    assert env["OPD_ROLLOUT_STEP_PAIRS"] == "8,4"
    assert env["OPD_AUX_ACTION"] == "0"
    assert env["VIDEO_TRANSITION_WEIGHT"] == "1.0"
    assert env["OPD_SAME_STATE_VELOCITY_WEIGHT"] == "1.0"
    assert env["OPD_ENDPOINT_AUX_WEIGHT"] == "0.0"
    assert env["LOCAL_FM_WEIGHT"] == "0.0"
    assert env["ACTION_LOCAL_FM_WEIGHT"] == "0.0"


def test_calibration_variants_cleanly_select_endpoint_and_velocity(tmp_path):
    endpoint = parse_dry_run_manifest(
        run_calibration_dry_run(tmp_path, "calib_endpoint_only").stdout
    )["stage2_env"]
    velocity = parse_dry_run_manifest(
        run_calibration_dry_run(tmp_path, "calib_velocity_only").stdout
    )["stage2_env"]
    last_step = parse_dry_run_manifest(
        run_calibration_dry_run(tmp_path, "calib_full_last_step").stdout
    )["stage2_env"]
    baseline = parse_dry_run_manifest(
        run_calibration_dry_run(tmp_path, "calib_w_o_opd").stdout
    )["stage2_env"]
    suffix = parse_dry_run_manifest(
        run_calibration_dry_run(tmp_path, "calib_full_suffix_grad").stdout
    )["stage2_env"]

    assert endpoint["VIDEO_TRANSITION_WEIGHT"] == "1.0"
    assert endpoint["OPD_SAME_STATE_VELOCITY_WEIGHT"] == "0.0"
    assert velocity["VIDEO_TRANSITION_WEIGHT"] == "0.0"
    assert velocity["OPD_SAME_STATE_VELOCITY_WEIGHT"] == "1.0"
    assert velocity["OPD_ANCHOR_CAP_RATIO"] == "-1.0"
    assert last_step["OPD_ROLLOUT_GRAD_MODE"] == "last_step"
    assert suffix["OPD_ROLLOUT_GRAD_MODE"] == "suffix"
    assert suffix["OPD_ROLLOUT_GRAD_STEPS"] == "2"
    assert suffix["OPD_ROLLOUT_STEP_PAIRS"] == "8,4"
    assert baseline["USE_OPD_AUX"] == "0"


def test_calibration_danceopd_variants_use_isolated_on_policy_velocity(tmp_path):
    interval_1 = parse_dry_run_manifest(
        run_calibration_dry_run(tmp_path, "calib_danceopd_i1").stdout
    )["stage2_env"]
    interval_4 = parse_dry_run_manifest(
        run_calibration_dry_run(tmp_path, "calib_danceopd_i4").stdout
    )["stage2_env"]

    for env in (interval_1, interval_4):
        assert env["USE_OPD_AUX"] == "1"
        assert env["OPD_QUERY_MODE"] == "danceopd"
        assert env["OPD_DANCEOPD_ROLLOUT_STEPS"] == "16"
        assert env["OPD_DANCEOPD_QUERY_ALPHA"] == "5.0"
        assert env["OPD_DANCEOPD_QUERY_BETA"] == "2.0"
        assert env["OPD_DANCEOPD_VERIFY_TERMINAL_PRIOR"] == "1"
        assert env["OPD_DANCEOPD_DIAGNOSTIC_INTERVAL"] == "50"
        assert env["OPD_AUX_WEIGHT"] == "1.0"
        assert env["OPD_AUX_ACTION"] == "0"
        assert env["OPD_LOSS_COMPOSITION"] == "explicit_hybrid"
        assert env["VIDEO_TRANSITION_WEIGHT"] == "0.0"
        assert env["OPD_ENDPOINT_AUX_WEIGHT"] == "0.0"
        assert env["OPD_SAME_STATE_VELOCITY_WEIGHT"] == "0.0"
        assert env["LOCAL_FM_WEIGHT"] == "0.0"
        assert env["ACTION_LOCAL_FM_WEIGHT"] == "0.0"

    assert interval_1["OPD_AUX_INTERVAL"] == "1"
    assert interval_4["OPD_AUX_INTERVAL"] == "4"
