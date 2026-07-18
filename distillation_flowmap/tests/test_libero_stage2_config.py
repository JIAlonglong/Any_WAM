import importlib
import os
import sys
import unittest


CONFIG_MODULE = "distillation_flowmap.config_libero_fullfinetune_stage2_anyflow"


def load_config(**env):
    old_env = {key: os.environ.get(key) for key in env}
    try:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sys.modules.pop(CONFIG_MODULE, None)
        return importlib.import_module(CONFIG_MODULE).cfg
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class LiberoStage2ConfigTest(unittest.TestCase):
    def test_defaults_match_rollout_eval_endpoint_objective(self):
        cfg = load_config(
            OPD_TEACHER_TARGET_MODE=None,
            OPD_ROLLOUT_GRAD_MODE=None,
            OPD_ACTION_ROLLOUT_GRAD_MODE=None,
            OPD_ROLLOUT_STEP_PAIRS=None,
            ROLLOUT_STEP_PAIRS=None,
            VIDEO_TRANSITION_PARAM=None,
            ACTION_TRANSITION_PARAM=None,
            OPD_SAME_STATE_VELOCITY_WEIGHT=None,
            OPD_AUX_GRADIENT_CHECKPOINTING=None,
            OPD_QUERY_MODE=None,
            OPD_DANCEOPD_ROLLOUT_STEPS=None,
        )

        self.assertEqual(cfg.opd_teacher_target_mode, "endpoint")
        self.assertEqual(cfg.video_transition_param, "x0")
        self.assertEqual(cfg.action_transition_param, "x0")
        self.assertEqual(cfg.opd_rollout_grad_mode, "last_step")
        self.assertEqual(cfg.opd_action_rollout_grad_mode, "last_step")
        self.assertEqual(cfg.opd_rollout_step_pairs, [[4, 1], [4, 2]])
        self.assertEqual(cfg.opd_same_state_velocity_weight, 0.0)
        self.assertFalse(cfg.opd_aux_gradient_checkpointing)
        self.assertEqual(cfg.opd_query_mode, "legacy")

    def test_env_same_state_velocity_weight_enables_hybrid_regularizer(self):
        cfg = load_config(OPD_SAME_STATE_VELOCITY_WEIGHT="0.1")
        self.assertEqual(cfg.opd_same_state_velocity_weight, 0.1)

    def test_env_rollout_step_pairs_override_endpoint_curriculum(self):
        cfg = load_config(OPD_ROLLOUT_STEP_PAIRS="4,1;4,2;4,4")
        self.assertEqual(cfg.opd_rollout_step_pairs, [[4, 1], [4, 2], [4, 4]])

    def test_opd_aux_gradient_checkpointing_can_be_enabled_explicitly(self):
        cfg = load_config(OPD_AUX_GRADIENT_CHECKPOINTING="1")
        self.assertTrue(cfg.opd_aux_gradient_checkpointing)

    def test_danceopd_s4_defaults_to_the_single_deployment_rollout(self):
        cfg = load_config(
            OPD_QUERY_MODE="danceopd",
            OPD_ROLLOUT_STEP_PAIRS="8,4",
            OPD_DANCEOPD_ROLLOUT_STEPS=None,
        )

        self.assertEqual(cfg.opd_query_mode, "danceopd")
        self.assertEqual(cfg.opd_rollout_step_pairs, [[8, 4]])
        self.assertEqual(cfg.opd_danceopd_rollout_steps, 4)

    def test_danceopd_mixed_student_budgets_require_an_explicit_query_grid(self):
        with self.assertRaisesRegex(ValueError, "OPD_DANCEOPD_ROLLOUT_STEPS"):
            load_config(
                OPD_QUERY_MODE="danceopd",
                OPD_ROLLOUT_STEP_PAIRS="4,1;4,2",
                OPD_DANCEOPD_ROLLOUT_STEPS=None,
            )

    def test_danceopd_allows_an_explicit_dense_query_grid_for_short_budgets(self):
        cfg = load_config(
            OPD_QUERY_MODE="danceopd",
            OPD_ROLLOUT_STEP_PAIRS="4,1;4,2",
            OPD_DANCEOPD_ROLLOUT_STEPS="16",
        )

        self.assertEqual(cfg.opd_danceopd_rollout_steps, 16)


if __name__ == "__main__":
    unittest.main()
