import unittest

from distillation_flowmap.rollout_eval_steps import normalize_rollout_steps


class RolloutEvalStepsTest(unittest.TestCase):
    def test_sorts_unique_positive_rollout_steps(self):
        self.assertEqual(normalize_rollout_steps([8, 1, 4, 2]), [1, 2, 4, 8])

    def test_rejects_empty_duplicate_and_nonpositive_rollout_steps(self):
        for values in ([], [1, 1], [0], [-1, 2]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    normalize_rollout_steps(values)


if __name__ == "__main__":
    unittest.main()
