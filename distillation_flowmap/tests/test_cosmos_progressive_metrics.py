from pathlib import Path

import torch

from distillation_flowmap.cosmos_progressive_metrics import (
    build_paired_eval_timesteps,
    denoised_endpoint,
    macro_average_by_task,
)


def test_denoised_endpoint_uses_the_terminal_sigma_per_frame():
    state = torch.tensor([[[[[3.0]], [[5.0]]]]])
    velocity = torch.tensor([[[[[2.0]], [[4.0]]]]])
    sigma = torch.tensor([[0.25, 0.5]])

    endpoint = denoised_endpoint(state, velocity, sigma)

    expected = torch.tensor([[[[[2.5]], [[3.0]]]]])
    assert torch.allclose(endpoint, expected)


def test_macro_average_by_task_does_not_weight_large_tasks_more_heavily():
    per_task = {
        "easy": {"endpoint_mse": 1.0, "velocity_mse": 2.0},
        "hard": {"endpoint_mse": 5.0, "velocity_mse": 8.0},
    }

    assert macro_average_by_task(per_task) == {
        "endpoint_mse": 3.0,
        "velocity_mse": 5.0,
    }


def test_paired_eval_timesteps_keep_video_and_action_horizons_separate():
    video_t, video_r, action_t, action_r = build_paired_eval_timesteps(
        batch_size=1,
        video_frames=9,
        action_frames=16,
        t=1000,
        r=0,
        device="cpu",
    )

    assert tuple(video_t.shape) == (1, 9)
    assert tuple(video_r.shape) == (1, 9)
    assert tuple(action_t.shape) == (1, 16)
    assert tuple(action_r.shape) == (1, 16)
    assert torch.equal(video_t, torch.full((1, 9), 1000.0))
    assert torch.equal(action_r, torch.zeros((1, 16)))


def test_trainer_reduces_all_deployment_metrics_in_a_fixed_position():
    from distillation_flowmap.cosmos_progressive_opd import (
        DEPLOYMENT_METRIC_SPECS,
    )

    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    metric_block = source.split("metric_tensors = [", 1)[1].split(
        "kto_main_enabled =", 1
    )[0]
    expected = [spec.accumulator for spec in DEPLOYMENT_METRIC_SPECS]

    append_block = source.split("# 累积损失值", 1)[1].split(
        "step_in_acc += 1", 1
    )[0]
    unpack_block = source.split("base_metric_count = 47", 1)[1].split(
        "metric_cursor =", 1
    )[0]
    reset_block = source.split("# 重置累积器", 1)[1].split(
        "step_in_acc = 0", 1
    )[0]
    logging_block = source.split(
        'if scheduled_kind == "aligned_video_opd":', 1
    )[1].split(
        "if self.distill_video:", 1
    )[0]
    stages = (
        (append_block, [spec.result_key for spec in DEPLOYMENT_METRIC_SPECS]),
        (metric_block, expected),
        (unpack_block, [spec.average_name for spec in DEPLOYMENT_METRIC_SPECS]),
        (reset_block, expected),
        (logging_block, [spec.log_name for spec in DEPLOYMENT_METRIC_SPECS]),
    )
    for block, names in stages:
        positions = [block.index(name) for name in names]
        assert positions == sorted(positions)
    assert len(DEPLOYMENT_METRIC_SPECS) == 6
    assert "base_metric_count = 47" in source
