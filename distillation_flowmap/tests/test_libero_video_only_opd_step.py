import inspect
import sys

import torch

from distillation_flowmap.tests.test_online_mechanism_diagnostics import _load_mixin


def _module():
    _load_mixin()
    return sys.modules["distillation_flowmap.flowmap_step"]


def test_disabled_joint_action_rollout_keeps_action_state_fixed():
    module = _module()
    action = torch.tensor([[[[[2.0]]]]])
    velocity = torch.tensor([[[[[5.0]]]]])
    t = torch.tensor([[1000.0]])
    r = torch.tensor([[500.0]])

    result = module.danceopd_action_euler_update(
        action,
        velocity,
        t,
        r,
        num_train_timesteps=1000,
        enabled=False,
    )

    assert torch.equal(result, action)


def test_enabled_joint_action_rollout_uses_flowmatch_euler_update():
    module = _module()
    action = torch.tensor([[[[[2.0]]]]])
    velocity = torch.tensor([[[[[5.0]]]]])
    t = torch.tensor([[1000.0]])
    r = torch.tensor([[500.0]])

    result = module.danceopd_action_euler_update(
        action,
        velocity,
        t,
        r,
        num_train_timesteps=1000,
        enabled=True,
    )

    assert torch.equal(result, torch.tensor([[[[[-0.5]]]]]))


def test_danceopd_reads_video_only_rollout_flag():
    mixin = _load_mixin()
    source = inspect.getsource(mixin._danceopd_aux_transition_step)

    assert "opd_joint_action_rollout" in source
    assert "danceopd_action_euler_update" in source


def test_main_action_loss_reads_detached_student_generated_video():
    mixin = _load_mixin()
    source = inspect.getsource(mixin._train_step)

    assert "action_condition_on_student_video" in source
    assert "student_video_condition_r = (" in source
    assert '"noisy_latents": student_video_condition_r' in source
    assert "if student_video_condition_r is not None" in source
