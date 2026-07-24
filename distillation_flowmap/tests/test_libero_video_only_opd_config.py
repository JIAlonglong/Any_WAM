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
        "ATTN_MODE",
        "USE_FSDP1",
        "OPD_DANCEOPD_ACTION_VELOCITY_WEIGHT",
        "OPD_AUX_ACTION",
        "OPD_JOINT_ACTION_ROLLOUT",
        "MECHANISM_DIAGNOSTICS",
        "MECHANISM_DIAGNOSTIC_INTERVAL",
        "VIDEO_ACTION_BRIDGE",
        "VIDEO_ACTION_BRIDGE_WEIGHT",
        "VIDEO_ACTION_BRIDGE_WARMUP_END",
        "VIDEO_ACTION_BRIDGE_MID_END",
        "VIDEO_ACTION_BRIDGE_START_PROB",
        "VIDEO_ACTION_BRIDGE_MID_PROB",
        "VIDEO_ACTION_BRIDGE_FINAL_PROB",
        "ACTION_DOWNSAMPLE_FACTOR",
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
    assert cfg.action_downsample_factor == 1
    assert cfg.video_action_bridge is True
    assert cfg.video_action_bridge_weight == 1.0
    assert cfg.video_action_bridge_warmup_end == 500
    assert cfg.video_action_bridge_mid_end == 1500
    assert cfg.video_action_bridge_start_probability == 0.25
    assert cfg.video_action_bridge_mid_probability == 0.50
    assert cfg.video_action_bridge_final_probability == 0.75
    assert cfg.action_loss_weight > 0
    assert cfg.gt_regression_weight > 0
    assert cfg.mechanism_diagnostics is True
    assert cfg.mechanism_diagnostic_interval == 50
    assert cfg.max_train_steps == 10000
    assert cfg.save_interval == 1000
    assert cfg.seed == 42
    assert cfg.attn_mode == "torch"
    assert cfg.use_fsdp1 is True


def test_explicit_mixed_query_grid_survives_fresh_base_import():
    cfg = load_config(
        OPD_QUERY_MODE="danceopd",
        OPD_ROLLOUT_STEP_PAIRS="8,1;8,2;8,4",
        OPD_DANCEOPD_ROLLOUT_STEPS="2,4",
    )

    assert cfg.opd_danceopd_rollout_step_choices == (2, 4)
    assert cfg.opd_rollout_step_pairs == [[8, 1], [8, 2], [8, 4]]


def test_attention_backend_is_explicit_and_validated():
    assert load_config(ATTN_MODE="torch").attn_mode == "torch"
    assert load_config(ATTN_MODE="flex").attn_mode == "flex"
    try:
        load_config(ATTN_MODE="unknown")
    except ValueError as exc:
        assert "ATTN_MODE" in str(exc)
    else:
        raise AssertionError("invalid attention backend must fail")


def test_fsdp1_backend_is_enabled_for_real_training():
    assert load_config(USE_FSDP1=1).use_fsdp1 is True
    assert load_config(USE_FSDP1=0).use_fsdp1 is False


def test_action_opd_cannot_be_enabled_by_environment():
    cfg = load_config(
        OPD_AUX_ACTION=1,
        OPD_DANCEOPD_ACTION_VELOCITY_WEIGHT=3,
        OPD_JOINT_ACTION_ROLLOUT=1,
    )

    assert cfg.opd_aux_action is False
    assert cfg.opd_danceopd_action_velocity_weight == 0.0
    assert cfg.opd_joint_action_rollout is False


def test_action_downsampling_cannot_be_enabled_by_environment():
    cfg = load_config(ACTION_DOWNSAMPLE_FACTOR=4)

    assert cfg.action_downsample_factor == 1


def test_rollout_choices_must_be_unique_positive_integers():
    for value in ("", "0,4", "2,2", "2,x"):
        try:
            load_config(OPD_DANCEOPD_ROLLOUT_STEPS=value)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid rollout choices to fail: {value!r}")


def test_video_action_bridge_curriculum_is_configurable():
    cfg = load_config(
        VIDEO_ACTION_BRIDGE=0,
        VIDEO_ACTION_BRIDGE_WEIGHT=0.4,
        VIDEO_ACTION_BRIDGE_WARMUP_END=100,
        VIDEO_ACTION_BRIDGE_MID_END=300,
        VIDEO_ACTION_BRIDGE_START_PROB=0.1,
        VIDEO_ACTION_BRIDGE_MID_PROB=0.2,
        VIDEO_ACTION_BRIDGE_FINAL_PROB=0.3,
    )
    assert cfg.video_action_bridge is False
    assert cfg.video_action_bridge_weight == 0.4
    assert cfg.video_action_bridge_warmup_end == 100
    assert cfg.video_action_bridge_mid_end == 300
    assert cfg.video_action_bridge_start_probability == 0.1
    assert cfg.video_action_bridge_mid_probability == 0.2
    assert cfg.video_action_bridge_final_probability == 0.3
