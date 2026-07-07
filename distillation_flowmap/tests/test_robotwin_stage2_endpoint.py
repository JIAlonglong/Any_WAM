import importlib
import logging
import os
import sys
import types
import unittest

import torch


CONFIG_MODULE = "distillation_flowmap.config_robotwin_fullfinetune_stage2_anyflow"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLOWMAP_DIR = os.path.join(REPO_ROOT, "distillation_flowmap")
WANVA_UTILS_DIR = os.path.join(REPO_ROOT, "wan_va", "utils")
for path in (FLOWMAP_DIR, WANVA_UTILS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


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


class RobotWinStage2EndpointConfigTest(unittest.TestCase):
    def test_defaults_use_endpoint_objective(self):
        cfg = load_config(
            OPD_TEACHER_TARGET_MODE=None,
            OPD_ROLLOUT_GRAD_MODE=None,
            OPD_ACTION_ROLLOUT_GRAD_MODE=None,
            OPD_ROLLOUT_STEP_PAIRS=None,
            VIDEO_TRANSITION_PARAM=None,
            ACTION_TRANSITION_PARAM=None,
            OPD_SAME_STATE_VELOCITY_WEIGHT=None,
            GRADIENT_CHECKPOINTING=None,
            OPD_AUX_GRADIENT_CHECKPOINTING=None,
        )

        self.assertEqual(cfg.opd_teacher_target_mode, "endpoint")
        self.assertEqual(cfg.video_transition_param, "x0")
        self.assertEqual(cfg.action_transition_param, "x0")
        self.assertEqual(cfg.opd_rollout_grad_mode, "last_step")
        self.assertEqual(cfg.opd_action_rollout_grad_mode, "last_step")
        self.assertEqual(cfg.opd_rollout_step_pairs, [[4, 1], [4, 2]])
        self.assertEqual(cfg.opd_same_state_velocity_weight, 0.0)
        self.assertTrue(cfg.gradient_checkpointing)
        self.assertFalse(cfg.opd_aux_gradient_checkpointing)

    def test_gradient_checkpointing_can_be_reenabled_for_non_opd_ablation(self):
        cfg = load_config(GRADIENT_CHECKPOINTING="1")
        self.assertTrue(cfg.gradient_checkpointing)

    def test_opd_aux_gradient_checkpointing_can_be_reenabled_explicitly(self):
        cfg = load_config(OPD_AUX_GRADIENT_CHECKPOINTING="1")
        self.assertTrue(cfg.opd_aux_gradient_checkpointing)

    def test_same_state_velocity_weight_env_override(self):
        cfg = load_config(OPD_SAME_STATE_VELOCITY_WEIGHT="0.25")
        self.assertEqual(cfg.opd_same_state_velocity_weight, 0.25)


class SameStateVelocityLossTest(unittest.TestCase):
    def test_weighted_mse_matches_per_sample_average(self):
        utils_stub = types.ModuleType("utils")
        utils_stub.data_seq_to_patch = lambda *args, **kwargs: None
        utils_stub.logger = logging.getLogger("test-flowmap-step")
        old_utils = sys.modules.get("utils")
        sys.modules["utils"] = utils_stub
        try:
            from distillation_flowmap.flowmap_step import _same_state_velocity_loss
        finally:
            if old_utils is None:
                sys.modules.pop("utils", None)
            else:
                sys.modules["utils"] = old_utils

        student_v = torch.tensor([
            [[[[1.0, 3.0]]]],
            [[[[2.0, 5.0]]]],
        ])
        teacher_v = torch.zeros_like(student_v)
        sample_weight = torch.tensor([0.25, 0.75])

        loss = _same_state_velocity_loss(
            student_v,
            teacher_v,
            sample_weight,
            transition_loss_type="mse",
            transition_huber_c=1e-3,
        )

        expected = torch.tensor([5.0, 14.5]).mul(sample_weight).mean()
        self.assertTrue(torch.allclose(loss, expected))


if __name__ == "__main__":
    unittest.main()
