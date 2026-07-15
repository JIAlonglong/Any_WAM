import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = REPO_ROOT / "distillation_flowmap" / "ablation"
LAUNCHER = ABLATION_DIR / "launch_robotwin_stepwam_ablation.py"
FINAL_ABLATION_RUNNER = ABLATION_DIR / "run_final_danceopd_ablation.sh"
FINAL_VARIANTS = [
    "final_w_o_opd",
    "final_endpoint_only_danceopd",
    "final_danceopd_velocity_only",
    "final_stepwam_danceopd",
    "final_local_adjacent_only",
    "final_action_only",
]


def _run_launcher_dry_run(tmp_path, variant):
    output_root = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(LAUNCHER),
            "--variant",
            variant,
            "--seed",
            "0",
            "--root",
            str(output_root),
            "--teacher-model-path",
            str(tmp_path / "missing-teacher"),
            "--dataset-path",
            str(tmp_path / "missing-data"),
            "--empty-emb-path",
            str(tmp_path / "missing-empty-emb.pt"),
            "--torchrun",
            str(tmp_path / "missing-torchrun"),
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
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=True,
    )
    return output_root, json.loads(completed.stdout[completed.stdout.index("{") :])


@pytest.mark.parametrize("variant", FINAL_VARIANTS)
def test_final_launcher_dry_run_preserves_frozen_variant_protocol(tmp_path, variant):
    output_root, manifest = _run_launcher_dry_run(tmp_path, variant)

    assert manifest["variant"] == variant
    assert manifest["seed"] == 0
    assert manifest["selected_task_filter"] == [
        "place_a2b_right",
        "stack_bowls_three",
        "open_microwave",
        "pick_dual_bottles",
    ]
    assert manifest["stage1_ckpt"].endswith("step_5")
    assert manifest["stage2_ckpt"].endswith("step_7")
    assert manifest["stage1_env"]["TRAIN_SEED"] == "0"
    assert manifest["stage2_env"]["TRAIN_SEED"] == "0"
    assert manifest["stage2_env"]["ENABLE_GRAD_BRANCH_DIAGNOSTICS"] == "1"
    assert not output_root.exists()

    stage1_env = manifest["stage1_env"]
    stage2_env = manifest["stage2_env"]
    if variant == "final_w_o_opd":
        assert stage1_env["DISTILL_MODE"] == "flashwam"
        assert stage2_env["DISTILL_MODE"] == "flashwam"
        assert stage2_env["USE_OPD_AUX"] == "0"
    elif variant == "final_action_only":
        assert stage1_env["DISTILL_MODE"] == "action"
        assert stage2_env["DISTILL_MODE"] == "action"
        assert stage2_env["USE_OPD_AUX"] == "0"
    else:
        expected = {
            "DISTILL_MODE": "flashwam",
            "USE_OPD_AUX": "1",
            "OPD_QUERY_MODE": "danceopd",
            "OPD_DANCEOPD_ROLLOUT_STEPS": "16",
            "OPD_DANCEOPD_QUERY_ALPHA": "5.0",
            "OPD_DANCEOPD_QUERY_BETA": "2.0",
            "OPD_DANCEOPD_VERIFY_TERMINAL_PRIOR": "1",
            "OPD_AUX_INTERVAL": "1",
            "OPD_AUX_ACTION": "0",
            "OPD_LOSS_COMPOSITION": "explicit_hybrid",
        }
        for key, value in expected.items():
            assert stage2_env[key] == value
        if variant == "final_endpoint_only_danceopd":
            assert stage2_env["OPD_DANCEOPD_VELOCITY_WEIGHT"] == "0.0"
            assert stage2_env["OPD_DANCEOPD_ENDPOINT_WEIGHT"] == "1.0"
            assert stage2_env["OPD_TEACHER_TARGET_MODE"] == "endpoint"
            assert stage2_env["OPD_ROLLOUT_STEP_PAIRS"] == "8,4"
            assert stage2_env["OPD_ROLLOUT_GRAD_MODE"] == "last_step"
        elif variant == "final_danceopd_velocity_only":
            assert stage2_env["OPD_DANCEOPD_VELOCITY_WEIGHT"] == "1.0"
            assert stage2_env["OPD_DANCEOPD_ENDPOINT_WEIGHT"] == "0.0"
            assert "OPD_TEACHER_TARGET_MODE" not in stage2_env
        elif variant == "final_stepwam_danceopd":
            assert stage2_env["OPD_DANCEOPD_VELOCITY_WEIGHT"] == "1.0"
            assert stage2_env["OPD_DANCEOPD_ENDPOINT_WEIGHT"] == "1.0"
            assert stage2_env["OPD_TEACHER_TARGET_MODE"] == "endpoint"
        else:
            assert variant == "final_local_adjacent_only"
            assert stage1_env["FLOWMAP_PAIR_MODE"] == "adjacent_grid"
            assert stage2_env["FLOWMAP_PAIR_MODE"] == "adjacent_grid"
            assert stage2_env["OPD_PAIR_MODE"] == "adjacent_grid"
            assert stage2_env["FLOWMAP_ADJACENT_GRID"] == "1000,750,500,250,0"


def test_final_ablation_wrapper_dry_run_schedules_only_final_matrix(tmp_path):
    output_root = tmp_path / "output"
    shared_stage1 = tmp_path / "missing-shared-stage1"
    completed = subprocess.run(
        [
            "bash",
            str(FINAL_ABLATION_RUNNER),
            "--dry-run",
            "--root",
            str(output_root),
            "--shared-stage1-ckpt",
            str(shared_stage1),
            "--stage1-steps",
            "5",
            "--stage2-steps",
            "7",
            "--protocol-seed",
            "9",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHON": sys.executable,
            "PYTHONDONTWRITEBYTECODE": "1",
            "TEACHER_MODEL_PATH": str(tmp_path / "missing-teacher"),
            "DATASET_PATH": str(tmp_path / "missing-data"),
            "EMPTY_EMB_PATH": str(tmp_path / "missing-empty-emb.pt"),
        },
        text=True,
        capture_output=True,
        check=True,
    )

    lines = [line for line in completed.stdout.splitlines() if line.startswith("DRY_RUN [")]
    assert len(lines) == len(FINAL_VARIANTS)
    for index, (variant, line) in enumerate(zip(FINAL_VARIANTS, lines)):
        assert f"DRY_RUN [{variant}]" in line
        assert f"CUDA_VISIBLE_DEVICES={index}" in line
        assert f"--master-port {38000 + index * 4}" in line
        if index < 4:
            assert "--stage stage2" in line
            assert f"--stage1-ckpt {shared_stage1}" in line
        else:
            assert "--stage both" in line
            assert "--stage1-ckpt" not in line
    assert not output_root.exists()
    assert "calib_" not in completed.stdout
    assert "full_stepwam" not in completed.stdout
    assert "run_final_eval.sh" not in FINAL_ABLATION_RUNNER.read_text(encoding="utf-8")
