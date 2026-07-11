import unittest
from pathlib import Path

import torch

from distillation_flowmap.danceopd_query import (
    direct_velocity_mse,
    sample_low_noise_query_indices,
    select_per_sample_trajectory_state,
)


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

    def test_flowmap_step_has_an_opt_in_danceopd_aux_dispatch(self):
        source = (
            Path(__file__).resolve().parents[1] / "flowmap_step.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def _danceopd_aux_transition_step(", source)
        self.assertIn("opd_query_mode", source)
        self.assertIn("direct_velocity_mse(", source)


if __name__ == "__main__":
    unittest.main()
