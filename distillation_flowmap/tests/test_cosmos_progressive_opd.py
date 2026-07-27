from pathlib import Path

import pytest
import torch

from distillation_flowmap.cosmos_progressive_opd import (
    apply_full_endpoint_focus,
    build_cosmos_teacher_window_path,
    broadcast_joint_action_timesteps,
    center_spatial_crop_slices,
    compose_cosmos_endpoint_loss,
    constrain_cosmos_teacher_timestep_pair,
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


@pytest.mark.parametrize("num_steps", [1, 2, 4])
def test_cosmos_teacher_window_path_never_leaves_raw_teacher_support(num_steps):
    t_min = 4.0 / 5.0
    t_max = 80.0 / 81.0

    path = build_cosmos_teacher_window_path(
        batch_size=2,
        num_frames=3,
        num_steps=num_steps,
        t_min=t_min,
        t_max=t_max,
        num_train_timesteps=1000,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert path.shape == (num_steps + 1, 2, 3)
    assert torch.allclose(path[0], torch.full((2, 3), t_max * 1000))
    assert torch.allclose(path[-1], torch.full((2, 3), t_min * 1000))
    assert bool((path >= t_min * 1000).all())
    assert bool((path <= t_max * 1000).all())


def test_constrain_cosmos_teacher_pair_clamps_invalid_raw_velocity_endpoints():
    timesteps = torch.tensor([[1000.0, 900.0], [700.0, 850.0]])
    target_timesteps = torch.tensor([[0.0, 1000.0], [100.0, 820.0]])

    t, r = constrain_cosmos_teacher_timestep_pair(
        timesteps,
        target_timesteps,
        t_min=0.8,
        t_max=80.0 / 81.0,
        num_train_timesteps=1000,
    )

    assert bool((t >= 800.0).all())
    assert bool((t <= (80.0 / 81.0) * 1000).all())
    assert bool((r >= 800.0).all())
    assert bool((r <= t).all())


def test_full_endpoint_focus_can_use_the_raw_cosmos_teacher_window():
    t = torch.full((2, 3), 840.0)
    r = torch.full((2, 3), 820.0)

    focused_t, focused_r, _ = apply_full_endpoint_focus(
        t,
        r,
        probability=1.0,
        num_train_timesteps=1000,
        focus_timestep=(80.0 / 81.0) * 1000,
        focus_target_timestep=800.0,
    )

    assert torch.allclose(focused_t, torch.full_like(t, (80.0 / 81.0) * 1000))
    assert torch.allclose(focused_r, torch.full_like(r, 800.0))


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


def test_zero_weight_action_endpoint_cannot_poison_video_only_loss():
    video_loss = torch.tensor(1.25, requires_grad=True)
    action_loss = torch.tensor(float("nan"), requires_grad=True)

    loss = compose_cosmos_endpoint_loss(
        video_loss,
        action_loss,
        action_endpoint_weight=0.0,
    )

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(1.25)
    loss.backward()
    assert video_loss.grad.item() == pytest.approx(1.0)
    assert action_loss.grad is None


def test_video_only_cosmos_opd_skips_action_endpoint_loss_graph():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_step.py"
    ).read_text(encoding="utf-8")
    cosmos_full_block = source.split(
        "def _cosmos_latent_full_opd_aux_transition_step("
    )[1].split("def _cosmos_latent_opd_aux_transition_step(")[0]

    assert "if joint_action_rollout and action_endpoint_weight > 0.0:" in cosmos_full_block
    assert "endpoint_loss = compose_cosmos_endpoint_loss(" in cosmos_full_block
