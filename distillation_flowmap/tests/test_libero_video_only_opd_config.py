import importlib
import os
import sys


MODULE = "distillation_flowmap.config_libero_fullfinetune_stage2_video_only_opd"
BASE_MODULE = "distillation_flowmap.config_libero_fullfinetune_stage2_anyflow"


def load_config(**env):
    names = {
        "MAX_TRAIN_STEPS",
        "SAVE_INTERVAL",
        "OPD_DANCEOPD_ROLLOUT_STEPS",
        "OPD_QUERY_MODE",
        "OPD_ROLLOUT_STEP_PAIRS",
        "OPD_DANCEOPD_ACTION_VELOCITY_WEIGHT",
        "OPD_AUX_ACTION",
        "OPD_JOINT_ACTION_ROLLOUT",
        "MECHANISM_DIAGNOSTICS",
        "MECHANISM_DIAGNOSTIC_INTERVAL",
        *env,
    }
    old = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update({key: str(value) for key, value in env.items()})
        sys.modules.pop(MODULE, None)
        sys.modules.pop(BASE_MODULE, None)
        return importlib.import_module(MODULE).cfg
    finally:
        sys.modules.pop(MODULE, None)
        sys.modules.pop(BASE_MODULE, None)
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_defaults_define_full_video_only_universal_contract():
    cfg = load_config()

    assert cfg.opd_query_mode == "danceopd"
    assert cfg.opd_danceopd_rollout_step_choices == (2, 4)
    assert cfg.opd_rollout_step_pairs == [[8, 1], [8, 2], [8, 4]]
    assert cfg.opd_danceopd_endpoint_weight == 1.0
    assert cfg.opd_danceopd_velocity_weight == 1.0
    assert cfg.opd_aux_action is False
    assert cfg.opd_danceopd_action_velocity_weight == 0.0
    assert cfg.opd_joint_action_rollout is False
    assert cfg.action_loss_weight > 0
    assert cfg.gt_regression_weight > 0
    assert cfg.mechanism_diagnostics is True
    assert cfg.mechanism_diagnostic_interval == 50
    assert cfg.max_train_steps == 10000
    assert cfg.save_interval == 1000
    assert cfg.seed == 42


def test_explicit_mixed_query_grid_survives_fresh_base_import():
    cfg = load_config(
        OPD_QUERY_MODE="danceopd",
        OPD_ROLLOUT_STEP_PAIRS="8,1;8,2;8,4",
        OPD_DANCEOPD_ROLLOUT_STEPS="2,4",
    )

    assert cfg.opd_danceopd_rollout_step_choices == (2, 4)
    assert cfg.opd_rollout_step_pairs == [[8, 1], [8, 2], [8, 4]]


def test_action_opd_cannot_be_enabled_by_environment():
    cfg = load_config(
        OPD_AUX_ACTION=1,
        OPD_DANCEOPD_ACTION_VELOCITY_WEIGHT=3,
        OPD_JOINT_ACTION_ROLLOUT=1,
    )

    assert cfg.opd_aux_action is False
    assert cfg.opd_danceopd_action_velocity_weight == 0.0
    assert cfg.opd_joint_action_rollout is False


def test_rollout_choices_must_be_unique_positive_integers():
    for value in ("", "0,4", "2,2", "2,x"):
        try:
            load_config(OPD_DANCEOPD_ROLLOUT_STEPS=value)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid rollout choices to fail: {value!r}")
