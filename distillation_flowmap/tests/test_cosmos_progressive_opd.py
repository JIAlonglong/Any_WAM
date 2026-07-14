import pytest
import torch

from distillation_flowmap.cosmos_progressive_opd import (
    apply_full_endpoint_focus,
    broadcast_joint_action_timesteps,
    center_spatial_crop_slices,
    rollout_velocity_field,
    should_stop_training_at_step,
    should_run_standalone_opd,
)


def test_teacher_rollout_queries_each_visited_state_and_terminal_field():
    calls = []

    def velocity_field(x, t):
        calls.append((x.detach().clone(), t.detach().clone()))
        return 1.0 + x

    x_t = torch.zeros(1, 1, 1, 1, 1)
    t = torch.ones(1, 1)
    r = torch.zeros(1, 1)

    x_r, v_r, path = rollout_velocity_field(
        x_t, t, r, num_steps=2, velocity_field=velocity_field
    )

    assert torch.allclose(path[:, 0, 0], torch.tensor([1.0, 0.5, 0.0]))
    assert len(calls) == 3
    assert torch.allclose(calls[0][0], torch.zeros_like(x_t))
    assert torch.allclose(calls[1][0], torch.full_like(x_t, -0.5))
    assert torch.allclose(calls[2][0], torch.full_like(x_t, -0.75))
    assert torch.allclose(x_r, torch.full_like(x_t, -0.75))
    assert torch.allclose(v_r, torch.full_like(x_t, 0.25))


def test_full_endpoint_focus_uses_the_deployment_endpoints():
    t = torch.full((3, 2), 840.0)
    r = torch.full((3, 2), 120.0)

    focused_t, focused_r, focus_mask = apply_full_endpoint_focus(
        t, r, probability=1.0, num_train_timesteps=1000
    )

    assert bool(focus_mask.all())
    assert torch.equal(focused_t, torch.full_like(t, 1000.0))
    assert torch.equal(focused_r, torch.zeros_like(r))


def test_zero_focus_leaves_random_endpoint_pairs_unchanged():
    t = torch.tensor([[900.0, 900.0], [700.0, 700.0]])
    r = torch.tensor([[300.0, 300.0], [100.0, 100.0]])

    focused_t, focused_r, focus_mask = apply_full_endpoint_focus(
        t, r, probability=0.0, num_train_timesteps=1000
    )

    assert not bool(focus_mask.any())
    assert torch.equal(focused_t, t)
    assert torch.equal(focused_r, r)


def test_standalone_opd_schedule_respects_warmup_and_interval():
    assert not should_run_standalone_opd(step=0, warmup_steps=8, interval=8)
    assert not should_run_standalone_opd(step=7, warmup_steps=8, interval=8)
    assert should_run_standalone_opd(step=8, warmup_steps=8, interval=8)
    assert not should_run_standalone_opd(step=9, warmup_steps=8, interval=8)
    assert should_run_standalone_opd(step=16, warmup_steps=8, interval=8)


def test_stop_after_step_only_caps_explicit_chunk_boundaries():
    assert not should_stop_training_at_step(step=249, stop_after_step=250)
    assert should_stop_training_at_step(step=250, stop_after_step=250)
    assert not should_stop_training_at_step(step=250, stop_after_step=None)
    assert not should_stop_training_at_step(step=250, stop_after_step=0)


def test_center_spatial_crop_uses_a_symmetric_window():
    h_slice, w_slice = center_spatial_crop_slices(28, 28, crop_size=24)

    assert (h_slice.start, h_slice.stop) == (2, 26)
    assert (w_slice.start, w_slice.stop) == (2, 26)


def test_crop_size_at_or_above_input_keeps_the_full_latent():
    h_slice, w_slice = center_spatial_crop_slices(28, 30, crop_size=32)

    assert (h_slice.start, h_slice.stop) == (0, 28)
    assert (w_slice.start, w_slice.stop) == (0, 30)


def test_joint_action_timesteps_share_the_endpoint_pair_across_action_tokens():
    video_t = torch.tensor([[1000.0] * 9, [750.0] * 9])
    video_r = torch.tensor([[0.0] * 9, [500.0] * 9])

    action_t, action_r = broadcast_joint_action_timesteps(
        video_t, video_r, action_frames=16
    )

    assert tuple(action_t.shape) == (2, 16)
    assert tuple(action_r.shape) == (2, 16)
    assert torch.equal(action_t[0], torch.full((16,), 1000.0))
    assert torch.equal(action_r[1], torch.full((16,), 500.0))


def test_joint_action_timesteps_reject_per_frame_video_pairs_without_a_mapping_rule():
    video_t = torch.tensor([[1000.0, 750.0]])
    video_r = torch.tensor([[750.0, 0.0]])

    with pytest.raises(ValueError, match="constant across video frames"):
        broadcast_joint_action_timesteps(video_t, video_r, action_frames=4)
