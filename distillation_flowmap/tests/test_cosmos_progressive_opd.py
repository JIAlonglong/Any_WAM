import torch

from distillation_flowmap.cosmos_progressive_opd import (
    apply_full_endpoint_focus,
    rollout_velocity_field,
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
