import unittest
from pathlib import Path

from distillation_flowmap.rollout_eval_steps import normalize_rollout_steps


class RolloutEvalStepsTest(unittest.TestCase):
    def test_sorts_unique_positive_rollout_steps(self):
        self.assertEqual(normalize_rollout_steps([8, 1, 4, 2]), [1, 2, 4, 8])

    def test_rejects_empty_duplicate_and_nonpositive_rollout_steps(self):
        for values in ([], [1, 1], [0], [-1, 2]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    normalize_rollout_steps(values)

    def test_offline_evaluators_accept_normalized_teacher_step_lists(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ("rollout_eval_stage2.py", "rollout_eval_video_stage2.py"):
            with self.subTest(filename=filename):
                source = (root / filename).read_text(encoding="utf-8")
                self.assertIn('nargs="+"', source)
                self.assertIn(
                    "normalize_rollout_steps(args.teacher_steps)", source
                )

    def test_offline_evaluators_freeze_teacher_in_eval_mode(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ("rollout_eval_stage2.py", "rollout_eval_video_stage2.py"):
            with self.subTest(filename=filename):
                source = (root / filename).read_text(encoding="utf-8")
                self.assertIn("trainer.teacher.eval()", source)


if __name__ == "__main__":
    unittest.main()
