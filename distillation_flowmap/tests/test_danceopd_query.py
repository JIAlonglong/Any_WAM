import unittest
from pathlib import Path

import torch
import distillation_flowmap.danceopd_query as danceopd_query
from wan_va.utils.scheduler import FlowMatchScheduler

from distillation_flowmap.danceopd_query import (
    denoised_endpoint_mse,
    direct_velocity_mse,
    sample_low_noise_query_indices,
    sample_semantic_query_indices,
    select_per_sample_trajectory_state,
)


class DanceOPDQueryTest(unittest.TestCase):
    def test_endpoint_sigma_uses_clean_region_beta_formula(self):
        sampler = getattr(danceopd_query, "sample_endpoint_sigmas", None)
        self.assertIsNotNone(
            sampler, "endpoint sigma sampler contract is missing"
        )
        original_sample = torch.distributions.Beta.sample
        try:
            torch.distributions.Beta.sample = lambda self, shape: torch.tensor(
                [0.0, 0.2, 0.75, 1.0], device=self.concentration1.device
            )
            sigma = sampler(
                batch_size=4,
                alpha=5.0,
                beta=2.0,
                max_sigma=0.25,
                device=torch.device("cpu"),
            )
        finally:
            torch.distributions.Beta.sample = original_sample

        self.assertTrue(torch.equal(
            sigma, torch.tensor([0.25, 0.20, 0.0625, 0.0])
        ))
        self.assertGreaterEqual(float(sigma.min()), 0.0)
        self.assertLessEqual(float(sigma.max()), 0.25)

    def test_pre_update_candidate_sigmas_match_compositional_grids(self):
        candidate_sigmas = getattr(
            danceopd_query, "pre_update_candidate_sigmas", None
        )
        self.assertIsNotNone(
            candidate_sigmas, "pre-update candidate sigma contract is missing"
        )
        expected = {
            2: (1.0, 0.5),
            4: (1.0, 0.75, 0.5, 0.25),
        }
        for rollout_steps, grid in expected.items():
            with self.subTest(rollout_steps=rollout_steps):
                actual = candidate_sigmas(
                    rollout_steps=rollout_steps,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                self.assertEqual(tuple(actual.tolist()), grid)

    def test_beta_query_indices_are_independent_per_batch_element(self):
        original_sample = torch.distributions.Beta.sample
        try:
            torch.distributions.Beta.sample = lambda self, shape: torch.tensor(
                [0.0, 0.24, 0.50, 0.999], device=self.concentration1.device
            )
            indices = sample_low_noise_query_indices(
                n_states=4,
                batch_size=4,
                alpha=5.0,
                beta=2.0,
                device=torch.device("cpu"),
            )
        finally:
            torch.distributions.Beta.sample = original_sample

        self.assertTrue(torch.equal(indices, torch.tensor([0, 0, 2, 3])))

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

    def test_semantic_query_excludes_initial_noise_and_keeps_one_step_terminal(self):
        torch.manual_seed(23)

        one_step = sample_semantic_query_indices(
            rollout_steps=1,
            batch_size=128,
            alpha=5.0,
            beta=2.0,
            device=torch.device("cpu"),
        )
        four_step = sample_semantic_query_indices(
            rollout_steps=4,
            batch_size=4096,
            alpha=5.0,
            beta=2.0,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(one_step, torch.ones_like(one_step)))
        self.assertGreaterEqual(int(four_step.min()), 1)
        self.assertLessEqual(int(four_step.max()), 4)
        self.assertGreater(float(four_step.float().mean()), 2.5)

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

    def test_shared_query_state_keeps_student_gradient_and_video_only_action_zero(self):
        video = torch.tensor([1.0, 2.0])
        action = torch.tensor([3.0, 4.0])
        sigma = torch.tensor([0.75, 0.25])
        calls = []

        def student_field(query_video, query_action, query_sigma):
            calls.append(("student", query_video, query_action, query_sigma))
            return (query_video + query_action + query_sigma).requires_grad_()

        def teacher_field(query_video, query_action, query_sigma):
            calls.append(("teacher", query_video, query_action, query_sigma))
            return query_video - query_action + query_sigma

        student_velocity = student_field(video, action, sigma)
        with torch.no_grad():
            teacher_velocity = teacher_field(video, action, sigma).detach()
        loss = direct_velocity_mse(student_velocity, teacher_velocity)
        action_velocity_contribution = torch.zeros_like(loss)

        self.assertIs(calls[0][1], calls[1][1])
        self.assertIs(calls[0][2], calls[1][2])
        self.assertIs(calls[0][3], calls[1][3])
        self.assertTrue(student_velocity.requires_grad)
        self.assertFalse(teacher_velocity.requires_grad)
        self.assertEqual(float(action_velocity_contribution), 0.0)

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

    def test_flowmatch_shifted_action_terminal_is_exactly_pure_noise(self):
        scheduler = FlowMatchScheduler(
            num_inference_steps=1000,
            num_train_timesteps=1000,
            shift=0.05,
            sigma_min=0.0,
            extra_one_step=True,
        )
        scheduler.set_timesteps(1000, training=True)
        clean = torch.full((2, 30, 64, 1, 1), 10.0)
        noise = torch.randn_like(clean)
        terminal_t = torch.full((2, 64), 1000.0)

        terminal_state = scheduler.add_noise(clean, noise, terminal_t, t_dim=2)

        self.assertTrue(torch.equal(scheduler.sigmas[0], torch.tensor(1.0)))
        self.assertTrue(torch.equal(terminal_state, noise))

    def test_flowmatch_full_forward_shifted_endpoints_are_exact_noise(self):
        clean = torch.full((2, 30, 64, 1, 1), 10.0)
        noise = torch.randn_like(clean)
        terminal_t = torch.full((2, 64), 1000.0)

        for shift in (0.001, 0.005, 0.02, 0.05, 5.0):
            with self.subTest(shift=shift):
                scheduler = FlowMatchScheduler(
                    num_inference_steps=1000,
                    num_train_timesteps=1000,
                    shift=shift,
                    sigma_min=0.0,
                    extra_one_step=True,
                )
                scheduler.set_timesteps(1000, training=True)

                terminal_state = scheduler.add_noise(
                    clean, noise, terminal_t, t_dim=2
                )

                self.assertTrue(torch.equal(scheduler.sigmas[0], torch.tensor(1.0)))
                self.assertTrue(torch.equal(terminal_state, noise))

    def test_flowmatch_rejects_nonpositive_linear_shift(self):
        with self.assertRaisesRegex(ValueError, "shift must be finite and positive"):
            FlowMatchScheduler(
                num_inference_steps=1000,
                num_train_timesteps=1000,
                shift=0.0,
                sigma_min=0.0,
                extra_one_step=True,
            )

    def test_flowmatch_rejects_invalid_linear_shift_override(self):
        scheduler = FlowMatchScheduler(
            num_inference_steps=1000,
            num_train_timesteps=1000,
            shift=0.05,
            sigma_min=0.0,
            extra_one_step=True,
        )

        with self.assertRaisesRegex(ValueError, "shift must be finite and positive"):
            scheduler.set_timesteps(1000, shift=0.0)

    def test_flowmatch_rejects_unrepresentable_linear_shift(self):
        for shift in (1e-46, 1e100):
            with self.subTest(shift=shift):
                with self.assertRaisesRegex(
                    ValueError, "shift must be finite and positive"
                ):
                    FlowMatchScheduler(
                        num_inference_steps=1000,
                        num_train_timesteps=1000,
                        shift=shift,
                        sigma_min=0.0,
                        extra_one_step=True,
                    )

    def test_flowmatch_rejects_nonfinite_full_forward_endpoint(self):
        with self.assertRaisesRegex(
            ValueError, "full-forward scheduler endpoint is non-finite"
        ):
            FlowMatchScheduler(
                num_inference_steps=1000,
                num_train_timesteps=1000,
                exponential_shift=True,
                exponential_shift_mu=float("inf"),
                sigma_min=0.0,
                extra_one_step=True,
            )

    def test_flowmatch_partial_schedule_is_not_snapped_to_pure_noise(self):
        scheduler = FlowMatchScheduler(
            num_inference_steps=1000,
            num_train_timesteps=1000,
            shift=0.05,
            sigma_min=0.0,
            extra_one_step=True,
        )
        scheduler.set_timesteps(
            1000, denoising_strength=0.9, training=True
        )
        clean = torch.full((2, 30, 64, 1, 1), 10.0)
        noise = torch.randn_like(clean)
        terminal_t = torch.full((2, 64), 1000.0)

        terminal_state = scheduler.add_noise(clean, noise, terminal_t, t_dim=2)

        self.assertLess(float(scheduler.sigmas[0]), 1.0)
        self.assertFalse(torch.equal(terminal_state, noise))

    def test_cosmos_danceopd_syncs_configured_rollout_choices(self):
        source = (
            Path(__file__).resolve().parents[1] / "flowmap_step.py"
        ).read_text(encoding="utf-8")
        cosmos_block = source.split(
            "def _cosmos_danceopd_velocity_loss("
        )[1].split("def _cosmos_latent_full_opd_aux_transition_step(")[0]

        self.assertIn("opd_danceopd_rollout_step_choices", cosmos_block)
        self.assertIn("choice_index = torch.randint(", cosmos_block)
        self.assertIn("dist.broadcast(choice_index, src=0)", cosmos_block)
        self.assertIn(
            "rollout_steps = rollout_step_choices[choice_index.item()]",
            cosmos_block,
        )

    def test_cosmos_danceopd_releases_rollout_trajectory_before_query(self):
        source = (
            Path(__file__).resolve().parents[1] / "flowmap_step.py"
        ).read_text(encoding="utf-8")
        cosmos_block = source.split(
            "def _cosmos_danceopd_velocity_loss("
        )[1].split("def _cosmos_latent_full_opd_aux_transition_step(")[0]

        release_position = cosmos_block.index(
            "del video_states, action_states, video_timesteps, action_timesteps"
        )
        query_position = cosmos_block.index("student_velocity = _student_joint_forward(")

        self.assertLess(release_position, query_position)
        self.assertIn("torch.cuda.empty_cache()", cosmos_block)

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
