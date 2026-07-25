import json
import os
import stat
import types
from pathlib import Path

import pytest
import torch

from distillation_flowmap.eval_libero_figure4_mechanism import (
    build_manifest,
    build_parser,
    configure_eval_config,
    probe_metadata,
    render_reproduce_script,
    select_checkpoints,
    validate_factor1_checkpoint,
)
from distillation_flowmap.mechanism_checkpoint_sweep import CheckpointSpec


def _spec(tmp_path: Path, step: int) -> CheckpointSpec:
    root = tmp_path / "checkpoints" / f"step_{step}"
    transformer = root / "target_student" / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text(
        json.dumps(
            {
                "checkpoint_step": step,
                "action_downsample_factor": 1,
                "action_grid_protocol": "continuous_action_v1",
            }
        )
    )
    (transformer / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    return CheckpointSpec(step=step, root=root, transformer=transformer)


def test_cli_defaults_to_target_student_factor1_protocol():
    args = build_parser().parse_args(
        [
            "--run-root",
            "/run",
            "--teacher-model-path",
            "/teacher",
            "--dataset-path",
            "/dataset",
        ]
    )

    assert args.student_role == "target_student"
    assert args.seed == 42
    assert args.r == 500
    assert args.s == 250
    assert args.teacher_steps == 8


def test_checkpoint_filter_preserves_real_sorted_steps(tmp_path):
    checkpoints = [_spec(tmp_path, 3000), _spec(tmp_path, 1000), _spec(tmp_path, 2000)]

    selected = select_checkpoints(checkpoints, requested_steps=[3000, 1000])

    assert [item.step for item in selected] == [1000, 3000]
    with pytest.raises(ValueError, match="not found"):
        select_checkpoints(checkpoints, requested_steps=[4000])


def test_factor1_validation_rejects_legacy_checkpoint(tmp_path):
    checkpoint = _spec(tmp_path, 1000)
    config_path = checkpoint.transformer / "config.json"
    config = json.loads(config_path.read_text())
    config["action_downsample_factor"] = 4
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="action_downsample_factor=1"):
        validate_factor1_checkpoint(checkpoint)


def test_eval_config_disables_training_only_state(tmp_path):
    base = types.SimpleNamespace(
        enable_wandb=True,
        use_torch_compile=True,
        gradient_checkpointing=True,
        opd_aux_gradient_checkpointing=True,
        load_worker=8,
    )
    checkpoint = _spec(tmp_path, 1000)

    cfg = configure_eval_config(
        base,
        checkpoint=checkpoint,
        output_dir=tmp_path / "outputs",
        teacher_model_path=tmp_path / "teacher",
        dataset_path=tmp_path / "dataset",
        seed=42,
        r=500,
        s=250,
        teacher_steps=8,
    )

    assert cfg is not base
    assert cfg.rank == 0 and cfg.local_rank == 0 and cfg.world_size == 1
    assert cfg.enable_wandb is False
    assert cfg.use_torch_compile is False
    assert cfg.gradient_checkpointing is False
    assert cfg.opd_aux_gradient_checkpointing is False
    assert cfg.offline_eval_skip_target_student is True
    assert cfg.resume_optimizer_state is False
    assert cfg.load_worker == 0
    assert cfg.resume_from_path == str(checkpoint.root)
    assert cfg.resume_online_from_target is True
    assert cfg.action_downsample_factor == 1


def test_probe_metadata_records_shapes_mask_and_fixed_inputs():
    batch = {
        "latents": torch.zeros(1, 2, 3, 4, 5),
        "actions": torch.zeros(1, 1, 7, 3, 1),
        "actions_mask": torch.ones(1, 1, 7, 1, 1),
        "_mechanism_diagnostic_batch_index": 0,
    }
    metadata = probe_metadata(
        batch,
        dataset_index=42,
        seed=42,
        r=500,
        s=250,
        teacher_steps=8,
    )

    assert metadata["dataset_index"] == 42
    assert metadata["seed"] == 42
    assert metadata["r"] == 500 and metadata["s"] == 250
    assert metadata["teacher_integration_steps"] == 8
    assert metadata["tensor_shapes"]["latents"] == [1, 2, 3, 4, 5]
    assert metadata["tensor_shapes"]["actions"] == [1, 1, 7, 3, 1]
    assert metadata["action_mask_rule"] == "broadcast valid action tokens only"


def test_manifest_labels_counterfactual_video_swap(tmp_path):
    checkpoints = [_spec(tmp_path, 1000), _spec(tmp_path, 2000)]
    manifest = build_manifest(
        git_commit="abc123",
        teacher_checkpoint=tmp_path / "teacher",
        checkpoints=checkpoints,
        probe={"seed": 42},
        dtype="torch.bfloat16",
        gpu="NVIDIA A800-SXM4-80GB",
        peak_memory_bytes=123,
        runtime_seconds=45.0,
    )

    assert manifest["student_role"] == "target_student"
    assert manifest["checkpoint_steps"] == [1000, 2000]
    assert manifest["action_contexts"]["e_video"]["trajectory_synchronized"] is False
    assert "counterfactual" in manifest["action_contexts"]["e_video"]["description"]
    assert manifest["metric_reduction"]["g_anchor_l2"] == "per-sample squared L2 mean"


def test_reproduce_script_is_executable_and_quotes_paths(tmp_path):
    script = render_reproduce_script(
        python=Path("/env with spaces/bin/python"),
        repo_root=Path("/repo with spaces"),
        run_root=Path("/run with spaces"),
        teacher_model_path=Path("/teacher"),
        dataset_path=Path("/dataset"),
        output_dir=Path("/output with spaces"),
        seed=42,
        r=500,
        s=250,
        teacher_steps=8,
        gpu_id="0",
    )
    path = tmp_path / "reproduce.sh"
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)

    assert "target_student" in script
    assert "'/repo with spaces'" in script
    assert "'/run with spaces'" in script
    assert "--teacher-steps 8" in script
    assert os.access(path, os.X_OK)
