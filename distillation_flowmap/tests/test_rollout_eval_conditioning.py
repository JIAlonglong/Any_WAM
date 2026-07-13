import unittest
from pathlib import Path
from types import SimpleNamespace

from distillation_flowmap.rollout_eval_conditioning import (
    CACHE_CONDITIONING_METADATA_KEY,
    CACHE_CONDITIONING_SCHEMA,
    cache_metadata_matches_offline_conditioning,
    freeze_offline_eval_conditioning,
)


class _ChildDataset:
    def __init__(self, cfg_prob):
        self.cfg_prob = cfg_prob


class RolloutEvalConditioningTest(unittest.TestCase):
    def test_freeze_disables_all_eval_text_dropout(self):
        children = [_ChildDataset(0.1), _ChildDataset(0.25)]
        trainer = SimpleNamespace(
            config=SimpleNamespace(drop_text_ratio=0.1),
            train_loader=SimpleNamespace(
                dataset=SimpleNamespace(_datasets=children),
            ),
        )

        metadata = freeze_offline_eval_conditioning(trainer)

        self.assertEqual(trainer.config.drop_text_ratio, 0.0)
        self.assertEqual([child.cfg_prob for child in children], [0.0, 0.0])
        self.assertEqual(
            metadata,
            {
                "schema": CACHE_CONDITIONING_SCHEMA,
                "text_dropout": 0.0,
            },
        )

    def test_old_or_mismatched_cache_metadata_is_rejected(self):
        expected = {
            CACHE_CONDITIONING_METADATA_KEY: {
                "schema": CACHE_CONDITIONING_SCHEMA,
                "text_dropout": 0.0,
            }
        }

        self.assertTrue(cache_metadata_matches_offline_conditioning(expected))
        self.assertFalse(cache_metadata_matches_offline_conditioning({}))
        self.assertFalse(
            cache_metadata_matches_offline_conditioning(
                {
                    CACHE_CONDITIONING_METADATA_KEY: {
                        "schema": "offline_eval_conditioning_v0",
                        "text_dropout": 0.1,
                    }
                }
            )
        )

    def test_both_offline_evaluators_freeze_conditioning_before_records(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ("rollout_eval_stage2.py", "rollout_eval_video_stage2.py"):
            with self.subTest(filename=filename):
                source = (root / filename).read_text(encoding="utf-8")
                self.assertIn("freeze_offline_eval_conditioning(trainer)", source)


if __name__ == "__main__":
    unittest.main()
