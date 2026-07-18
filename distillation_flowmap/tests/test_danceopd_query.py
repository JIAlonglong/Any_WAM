import unittest
from pathlib import Path

import pytest
import torch
from wan_va.utils.scheduler import FlowMatchScheduler

from distillation_flowmap.danceopd_query import (
    append_post_update_trajectory_state,
    denoised_endpoint_mse,
    direct_velocity_mse,
    interpolate_raw_flow_segment,
    legacy_danceopd_query_floor,
    require_query_timestep_floor,
    sample_low_noise_query_indices,
    select_per_sample_trajectory_state,
)


def test_appending_post_update_state_preserves_explicit_terminal_metadata():
    states = [torch.full((1,), 10.0)]
    timesteps = [torch.full((1,), 1000.0)]
    endpoint = torch.tensor([0.5], requires_grad=True)
    safe_t = torch.full((1,), 62.5)

    append_post_update_trajectory_state(
        states, timesteps, state=endpoint, timestep=safe_t
    )

    assert len(states) == len(timesteps) == 2
    assert torch.equal(states[-1], torch.tensor([0.5]))
    assert not states[-1].requires_grad
    assert torch.equal(timesteps[-1], safe_t)


def test_mixed_dance_terminal_floor_matches_the_legacy_16_step_minimum():
    floor = legacy_danceopd_query_floor(num_train_timesteps=1000)

    assert floor == 62.5
    assert 0.0 < floor < 1000.0


def test_positive_terminal_query_stays_on_the_final_raw_euler_segment():
    segment_start = torch.tensor([[[[[2.0]], [[4.0]]]]])
    velocity = torch.tensor([[[[[8.0]], [[16.0]]]]])
    start_timestep = torch.tensor([[1000.0, 1000.0]])
    target_timestep = torch.tensor([[62.5, 125.0]])

    query_state = interpolate_raw_flow_segment(
        segment_start,
        velocity,
        start_timestep,
        target_timestep,
        num_train_timesteps=1000,
        t_dim=2,
    )

    expected = torch.tensor([[[[[-5.5]], [[-10.0]]]]])
    assert torch.allclose(query_state, expected)


def test_query_floor_guard_rejects_an_exact_zero_teacher_timestamp():
    safe_floor = legacy_danceopd_query_floor(num_train_timesteps=1000)

    require_query_timestep_floor(
        torch.tensor([[safe_floor, 1000.0]]), safe_floor=safe_floor
    )
    with pytest.raises(RuntimeError, match="below the safe floor"):
        require_query_timestep_floor(
            torch.tensor([[0.0, safe_floor]]), safe_floor=safe_floor
        )


def test_mixed_cosmos_danceopd_never_appends_an_exact_zero_teacher_query():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_step.py"
    ).read_text(encoding="utf-8")
    velocity_loss_block = source.split(
        "def _cosmos_danceopd_velocity_loss("
    )[1].split("def _cosmos_latent_full_opd_aux_transition_step(")[0]
    post_update_block = velocity_loss_block.split(
        "if append_post_update_terminal:"
    )[1].split("query_indices =")[0]

    assert "legacy_danceopd_query_floor" in post_update_block
    assert "interpolate_raw_flow_segment" in post_update_block
    assert "timestep=terminal_query_video_t" in post_update_block
    assert "timestep=zero_video_t" not in post_update_block


class DanceOPDQueryTest(unittest.TestCase):
    def test_beta_query_indices_bias_toward_late_trajectory_states(self):
        torch.manual_seed(17)

        indices = sample_low_noise_query_indices(
            n_states=16,
            batch_size=4096,
            alpha=5.0,
            beta=2.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(indices.shape, (4096,))
        self.assertGreaterEqual(int(indices.min()), 0)
        self.assertLess(int(indices.max()), 16)
        self.assertGreater(float(indices.float().mean()), 9.0)

    def test_selects_one_trajectory_state_for_each_sample(self):
        trajectory = torch.stack(
            [torch.full((3, 2), float(step)) for step in range(4)], dim=0
        )
        indices = torch.tensor([0, 2, 3])

        selected = select_per_sample_trajectory_state(trajectory, indices)

        expected = torch.tensor([[0.0, 0.0], [2.0, 2.0], [3.0, 3.0]])
        self.assertTrue(torch.equal(selected, expected))

    def test_direct_velocity_mse_detaches_teacher_target(self):
        student = torch.tensor([0.0, 1.0], requires_grad=True)
        teacher = torch.tensor([1.0, 3.0], requires_grad=True)

        loss = direct_velocity_mse(student, teacher)
        loss.backward()

        self.assertTrue(torch.allclose(loss, torch.tensor(2.5)))
        self.assertTrue(torch.allclose(student.grad, torch.tensor([-1.0, -2.0])))
        self.assertIsNone(teacher.grad)

    def test_denoised_endpoint_mse_detaches_teacher_endpoint(self):
        student_x = torch.tensor([2.0], requires_grad=True)
        student_v = torch.tensor([1.0], requires_grad=True)
        teacher_x = torch.tensor([2.0], requires_grad=True)
        teacher_v = torch.tensor([0.5], requires_grad=True)

        loss = denoised_endpoint_mse(
            student_x,
            student_v,
            teacher_x,
            teacher_v,
            torch.tensor([0.5]),
        )
        loss.backward()

        self.assertTrue(torch.allclose(loss, torch.tensor(0.0625)))
        self.assertIsNotNone(student_x.grad)
        self.assertIsNotNone(student_v.grad)
        self.assertIsNone(teacher_x.grad)
        self.assertIsNone(teacher_v.grad)

    def test_flowmatch_terminal_timestep_is_exactly_pure_noise(self):
        scheduler = FlowMatchScheduler(
            num_inference_steps=1000,
            num_train_timesteps=1000,
            shift=5.0,
        )
        scheduler.set_timesteps(1000, training=True)
        clean = torch.randn(2, 3, 4, 1, 1)
        noise = torch.randn_like(clean)
        terminal_t = torch.full((2, 4), 1000.0)

        terminal_state = scheduler.add_noise(clean, noise, terminal_t, t_dim=2)

        self.assertTrue(torch.equal(terminal_state, noise))

    def test_flowmap_step_has_an_opt_in_danceopd_aux_dispatch(self):
        source = (
            Path(__file__).resolve().parents[1] / "flowmap_step.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def _danceopd_aux_transition_step(", source)
        self.assertIn("opd_query_mode", source)
        self.assertIn("direct_velocity_mse(", source)
        self.assertIn("danceopd_terminal_prior_max_error", source)

    def test_danceopd_can_add_an_independent_endpoint_target_once(self):
        source = (
            Path(__file__).resolve().parents[1] / "flowmap_step.py"
        ).read_text(encoding="utf-8")
        danceopd_block = source.split(
            "def _danceopd_aux_transition_step("
        )[1].split("def _opd_aux_transition_step(")[0]

        self.assertIn("def _danceopd_independent_endpoint_loss(", source)
        self.assertIn("opd_danceopd_endpoint_weight", danceopd_block)
        self.assertIn("_danceopd_independent_endpoint_loss(", danceopd_block)
        self.assertEqual(danceopd_block.count("loss.backward()"), 1)

        config_source = (
            Path(__file__).resolve().parents[1]
            / "config_robotwin_fullfinetune_stage2_anyflow.py"
        ).read_text(encoding="utf-8")
        self.assertIn("OPD_DANCEOPD_ENDPOINT_WEIGHT", config_source)
        self.assertIn("OPD_DANCEOPD_VELOCITY_WEIGHT", config_source)


if __name__ == "__main__":
    unittest.main()
