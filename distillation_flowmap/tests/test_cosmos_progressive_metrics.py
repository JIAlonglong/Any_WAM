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
