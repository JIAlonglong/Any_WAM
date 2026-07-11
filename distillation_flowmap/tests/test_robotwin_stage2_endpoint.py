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
WANVA_ROOT = os.path.join(REPO_ROOT, "wan_va")
for path in (FLOWMAP_DIR, WANVA_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)


def import_flowmap_trainer_helper(name="_resolve_use_fsdp1"):
    stubs = {}

    def stub_module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        stubs[name] = sys.modules.get(name)
        sys.modules[name] = mod
        return mod

    class _DataMixin:
        pass

    class _FlowMapStepMixin:
        pass

    stub_module("distributed", __path__=[])
    stub_module("distributed.fsdp", shard_model=lambda model, **_: model, apply_ac=lambda model: model)
    stub_module(
        "distributed.util",
        _configure_model=lambda model, **_: model,
        dist_mean=lambda value: value,
    )
    stub_module("wan_va.distributed.fsdp", shard_model_fsdp1=lambda model, **_: model)
    stub_module(
        "modules.utils",
        WanVAEStreamingWrapper=object,
        load_transformer=lambda *args, **kwargs: None,
        load_vae=lambda *args, **kwargs: None,
    )
    stub_module("distillation.data", DataMixin=_DataMixin)
    stub_module("distillation.ema", update_ema=lambda *args, **kwargs: None)
    stub_module("flowmap_step", FlowMapStepMixin=_FlowMapStepMixin)
    stub_module(
        "model_flowmap",
        setup_flowmap_model=lambda model, **kwargs: model,
        patch_model_forward=lambda model: model,
    )

    sys.modules.pop("distillation_flowmap.flowmap_trainer", None)
    try:
        from distillation_flowmap import flowmap_trainer
        helper = getattr(flowmap_trainer, name)
    finally:
        for name, old in stubs.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return helper


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
            OPD_ROLLOUT_GRAD_STEPS=None,
            OPD_ROLLOUT_STEP_PAIRS=None,
            VIDEO_TRANSITION_PARAM=None,
            ACTION_TRANSITION_PARAM=None,
            OPD_SAME_STATE_VELOCITY_WEIGHT=None,
            OPD_SERIAL_STUDENT_CFG=None,
            OPD_AUX_EMPTY_CACHE=None,
            USE_8BIT_OPTIMIZER=None,
            GRADIENT_CHECKPOINTING=None,
            OPD_AUX_GRADIENT_CHECKPOINTING=None,
            OPD_LOSS_COMPOSITION=None,
        )

        self.assertEqual(cfg.opd_teacher_target_mode, "endpoint")
        self.assertEqual(cfg.video_transition_param, "x0")
        self.assertEqual(cfg.action_transition_param, "x0")
        self.assertEqual(cfg.opd_rollout_grad_mode, "last_step")
        self.assertEqual(cfg.opd_action_rollout_grad_mode, "last_step")
        self.assertEqual(cfg.opd_rollout_grad_steps, 1)
        self.assertEqual(cfg.opd_rollout_step_pairs, [[4, 1], [4, 2]])
        self.assertEqual(cfg.opd_same_state_velocity_weight, 0.0)
        self.assertTrue(cfg.gradient_checkpointing)
        self.assertTrue(cfg.opd_aux_gradient_checkpointing)
        self.assertTrue(cfg.opd_serial_student_cfg)
        self.assertTrue(cfg.opd_aux_empty_cache)
        self.assertTrue(cfg.use_8bit_optimizer)
        self.assertTrue(cfg.use_fsdp1)
        self.assertEqual(cfg.opd_loss_composition, "legacy")

    def test_explicit_hybrid_loss_composition_env_override(self):
        cfg = load_config(OPD_LOSS_COMPOSITION="explicit_hybrid")

        self.assertEqual(cfg.opd_loss_composition, "explicit_hybrid")

    def test_invalid_loss_composition_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "OPD_LOSS_COMPOSITION must be legacy or explicit_hybrid",
        ):
            load_config(OPD_LOSS_COMPOSITION="ambiguous")

    def test_suffix_rollout_gradient_window_env_override(self):
        cfg = load_config(
            OPD_ROLLOUT_GRAD_MODE="suffix",
            OPD_ROLLOUT_GRAD_STEPS="2",
        )

        self.assertEqual(cfg.opd_rollout_grad_mode, "suffix")
        self.assertEqual(cfg.opd_action_rollout_grad_mode, "suffix")
        self.assertEqual(cfg.opd_rollout_grad_steps, 2)

    def test_nonpositive_rollout_gradient_window_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "OPD_ROLLOUT_GRAD_STEPS must be positive",
        ):
            load_config(OPD_ROLLOUT_GRAD_STEPS="0")

    def test_use_fsdp1_env_override_is_available_for_non_checkpointed_debug(self):
        cfg = load_config(USE_FSDP1="0", GRADIENT_CHECKPOINTING="0")
        self.assertFalse(cfg.use_fsdp1)

    def test_opd_checkpointing_forces_fsdp1_even_if_env_disables_it(self):
        cfg = load_config(USE_FSDP1="0", USE_OPD_AUX="1", GRADIENT_CHECKPOINTING="1")

        _resolve_use_fsdp1 = import_flowmap_trainer_helper()

        self.assertTrue(_resolve_use_fsdp1(cfg))

    def test_opd_aux_checkpointing_switch_reaches_wrapped_module(self):
        _call_with_student_checkpointing = import_flowmap_trainer_helper(
            "_call_with_student_checkpointing")
        inner = types.SimpleNamespace(_flowmap_gradient_checkpointing=True)
        wrapper = types.SimpleNamespace(module=inner)
        observed = []

        def run_aux():
            observed.append((
                wrapper._flowmap_gradient_checkpointing,
                inner._flowmap_gradient_checkpointing,
            ))
            return "ok"

        result = _call_with_student_checkpointing(wrapper, False, run_aux)

        self.assertEqual(result, "ok")
        self.assertEqual(observed, [(False, False)])
        self.assertFalse(hasattr(wrapper, "_flowmap_gradient_checkpointing"))
        self.assertTrue(inner._flowmap_gradient_checkpointing)

    def test_gradient_checkpointing_can_be_reenabled_for_non_opd_ablation(self):
        cfg = load_config(GRADIENT_CHECKPOINTING="1")
        self.assertTrue(cfg.gradient_checkpointing)

    def test_opd_aux_gradient_checkpointing_can_be_disabled_explicitly(self):
        cfg = load_config(OPD_AUX_GRADIENT_CHECKPOINTING="0")
        self.assertFalse(cfg.opd_aux_gradient_checkpointing)

    def test_opd_serial_student_cfg_can_be_disabled_explicitly(self):
        cfg = load_config(OPD_SERIAL_STUDENT_CFG="0")
        self.assertFalse(cfg.opd_serial_student_cfg)

    def test_opd_empty_cache_can_be_disabled_explicitly(self):
        cfg = load_config(OPD_AUX_EMPTY_CACHE="0")
        self.assertFalse(cfg.opd_aux_empty_cache)

    def test_8bit_optimizer_can_be_disabled_explicitly(self):
        cfg = load_config(USE_8BIT_OPTIMIZER="0")
        self.assertFalse(cfg.use_8bit_optimizer)

    def test_same_state_velocity_weight_env_override(self):
        cfg = load_config(OPD_SAME_STATE_VELOCITY_WEIGHT="0.25")
        self.assertEqual(cfg.opd_same_state_velocity_weight, 0.25)

    def test_danceopd_query_mode_is_opt_in_and_configurable(self):
        default_cfg = load_config(OPD_QUERY_MODE=None)
        dance_cfg = load_config(
            OPD_QUERY_MODE="danceopd",
            OPD_DANCEOPD_ROLLOUT_STEPS="8",
            OPD_DANCEOPD_QUERY_ALPHA="5.0",
            OPD_DANCEOPD_QUERY_BETA="2.0",
        )

        self.assertEqual(default_cfg.opd_query_mode, "legacy")
        self.assertEqual(dance_cfg.opd_query_mode, "danceopd")
        self.assertEqual(dance_cfg.opd_danceopd_rollout_steps, 8)
        self.assertEqual(dance_cfg.opd_danceopd_query_alpha, 5.0)
        self.assertEqual(dance_cfg.opd_danceopd_query_beta, 2.0)

    def test_invalid_danceopd_query_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "OPD_QUERY_MODE"):
            load_config(OPD_QUERY_MODE="not-a-mode")


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
