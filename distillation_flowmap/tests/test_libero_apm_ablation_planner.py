import argparse
import json
from pathlib import Path

import pytest

from distillation_flowmap.ablation.launch_libero_apm_ablation import (
    build_run_plan,
    write_run_files,
)


def args(tmp_path, arm):
    stage1 = tmp_path / "stage1" / "checkpoints" / "step_2000"
    teacher = tmp_path / "teacher"
    dataset = tmp_path / "dataset"
    empty = dataset / "empty_emb.pt"
    for path in (stage1, teacher, dataset):
        path.mkdir(parents=True, exist_ok=True)
    empty.write_bytes(b"x")
    return argparse.Namespace(
        variant=arm,
        output_root=tmp_path / "runs",
        teacher_model_path=teacher,
        dataset_path=dataset,
        empty_emb_path=empty,
        stage1_ckpt=stage1,
        torchrun=Path("/opt/torchrun"),
        steps=500,
        save_interval=100,
        master_port=29671,
        gpu_ids="0,1,2,3",
        dry_run=True,
        execute=False,
    )


@pytest.mark.parametrize(
    ("arm", "endpoint", "velocity"),
    [
        ("stage1_only", "0.0", "0.0"),
        ("anchor_only", "1.0", "0.0"),
        ("field_only", "0.0", "1.0"),
        ("apm", "1.0", "1.0"),
    ],
)
def test_arm_changes_only_video_opd_terms(
    tmp_path, arm, endpoint, velocity
):
    plan = build_run_plan(args(tmp_path, arm))
    env = plan["stage2"]["env"]

    assert env["OPD_DANCEOPD_ENDPOINT_WEIGHT"] == endpoint
    assert env["OPD_DANCEOPD_VELOCITY_WEIGHT"] == velocity
    assert env["OPD_AUX_ACTION"] == "0"
    assert env["OPD_JOINT_ACTION_ROLLOUT"] == "0"
    assert env["OPD_DANCEOPD_ACTION_VELOCITY_WEIGHT"] == "0.0"
    assert env["DATASET_SAMPLE_MANIFEST"].endswith("train_manifest.json")


def test_plan_uses_four_processes_shared_stage1_and_seed42(tmp_path):
    ns = args(tmp_path, "apm")
    plan = build_run_plan(ns)

    assert "--nproc_per_node=4" in plan["stage2"]["argv"]
    assert "--master_port=29671" in plan["stage2"]["argv"]
    assert plan["stage2"]["env"]["RESUME_FROM_PATH"] == str(ns.stage1_ckpt)
    assert plan["stage2"]["env"]["TRAIN_SEED"] == "42"
    assert plan["stage2"]["env"]["ATTN_MODE"] == "torch"
    assert plan["checkpoint"].as_posix().endswith(
        "/apm/seed_42/stage2/checkpoints/step_500"
    )


def test_write_run_files_records_protocol_and_exact_command(tmp_path):
    ns = args(tmp_path, "anchor_only")
    ns.dry_run = False
    plan = build_run_plan(ns)

    manifest_path = write_run_files(plan)
    manifest = json.loads(manifest_path.read_text())

    assert manifest["variant"] == "anchor_only"
    assert manifest["protocol"]["train_indices"] == list(range(40))
    assert manifest["protocol"]["heldout_indices"] == list(range(40, 50))
    assert manifest["stage1_ckpt"] == str(ns.stage1_ckpt)
    assert "DATASET_SAMPLE_MANIFEST=" in manifest["command"]
    assert manifest_path == plan["run_dir"] / "run_manifest.json"


def test_unknown_arm_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown variant"):
        build_run_plan(args(tmp_path, "unknown"))


@pytest.mark.parametrize(("field", "value"), [("steps", 0), ("save_interval", 0)])
def test_nonpositive_training_values_are_rejected(tmp_path, field, value):
    ns = args(tmp_path, "apm")
    setattr(ns, field, value)

    with pytest.raises(ValueError, match=field.replace("_", " ")):
        build_run_plan(ns)


def test_real_plan_refuses_existing_run_directory(tmp_path):
    ns = args(tmp_path, "apm")
    ns.dry_run = False
    plan = build_run_plan(ns)
    plan["run_dir"].mkdir(parents=True)

    with pytest.raises(FileExistsError, match="run directory"):
        write_run_files(plan)
