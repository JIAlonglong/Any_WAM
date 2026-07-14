import importlib
from types import SimpleNamespace

import torch


def test_progressive_evaluator_imports_the_same_flow_scheduler_as_training():
    module = importlib.import_module("distillation_flowmap.eval_cosmos_progressive_stage2")

    assert callable(module._student_input)


def test_offline_eval_configuration_keeps_endpoint_rollout_suffix_valid():
    module = importlib.import_module("distillation_flowmap.eval_cosmos_progressive_stage2")
    config = SimpleNamespace(opd_rollout_grad_steps=0)

    module.configure_offline_eval_config(config, skip_same_state_velocity=False)

    assert config.opd_rollout_grad_mode == "endpoint"
    assert config.opd_rollout_grad_steps == 1


def test_cached_teacher_anchor_replaces_joint_state_but_retains_dataset_actions():
    module = importlib.import_module("distillation_flowmap.eval_cosmos_progressive_stage2")
    dataset_actions = torch.full((1, 30, 16, 4, 1), 3.0)
    batch = {
        "latents": torch.zeros((1, 16, 16, 28, 28)),
        "actions": dataset_actions.clone(),
    }
    cache = {
        "video_x0": torch.full((1, 16, 9, 28, 28), 5.0),
        "teacher_action_x0": torch.full((1, 30, 16, 4, 1), 7.0),
    }

    retained_actions = module.install_cached_teacher_anchor(
        batch, cache, device=torch.device("cpu"), dtype=torch.float32
    )

    assert torch.equal(retained_actions, dataset_actions)
    assert torch.equal(batch["latents"], cache["video_x0"])
    assert torch.equal(batch["actions"], cache["teacher_action_x0"])
