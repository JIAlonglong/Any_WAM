import ast
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch
from torch import nn

from distillation_flowmap.cosmos_deployment_rollout import deployment_endpoint_losses
from distillation_flowmap.cosmos_progressive_opd import broadcast_joint_action_timesteps
from distillation_flowmap.opd_rollout_grad import (
    SUPPORTED_ROLLOUT_GRAD_MODES,
    rollout_step_requires_grad,
)
from distillation_flowmap.run_cosmos_progressive_stage2 import build_stage_chunk_plan
from distillation_flowmap.flowmap_step import _synchronized_nonfinite_decision


def _plan(tmp_path, **overrides):
    defaults = {
        "stage": "s4",
        "root": tmp_path / "progressive",
        "dataset_path": Path("/tmp/libero"),
        "train_manifest": Path("/tmp/train.json"),
        "selection_manifest": Path("/tmp/selection.json"),
        "eval_pairs": Path("/tmp/pairs.json"),
        "selection_cache_dir": Path("/tmp/selection_cache"),
        "stage1_checkpoint": Path("/tmp/stage1_step5000"),
        "current_step": 0,
        "chunk_size": 250,
        "master_port": 29761,
        "train_seed": 20260714,
        "torchrun": Path("/tmp/torchrun"),
    }
    defaults.update(overrides)
    return build_stage_chunk_plan(**defaults)


def test_initial_s4_chunk_starts_from_fixed_stage1_and_preserves_full_schedule(tmp_path):
    plan = _plan(tmp_path)

    assert plan["spec"] == {"teacher_steps": 8, "student_steps": 4, "max_steps": 5000}
    assert plan["checkpoint_dir"] == tmp_path / "progressive" / "s4" / "checkpoints" / "step_250"
    assert plan["train_env"]["RESUME_FROM_PATH"] == "/tmp/stage1_step5000"
    assert plan["train_env"]["RESET_RESUME_STEP"] == "1"
    assert plan["train_env"]["RESUME_OPTIMIZER_STATE"] == "0"
    assert plan["train_env"]["MAX_TRAIN_STEPS"] == "5000"
    assert plan["train_env"]["STOP_AFTER_STEP"] == "250"
    assert plan["train_env"]["DATASET_SAMPLE_MANIFEST"] == "/tmp/train.json"
    assert plan["train_env"]["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert plan["train_env"]["COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES"] == "6,7"
    assert "--nproc_per_node=2" in plan["train_argv"]
    gradient_arg = plan["train_argv"].index("--gradient-accumulation-steps")
    assert plan["train_argv"][gradient_arg + 1] == "1"
    assert plan["eval_env"]["CUDA_VISIBLE_DEVICES"] == "6"
    assert plan["eval_env"]["COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES"] == "7"
    assert "--student-steps" in plan["eval_argv"]
    assert "4" in plan["eval_argv"]


def test_resume_chunk_uses_the_previous_checkpoint_and_optimizer_state(tmp_path):
    plan = _plan(tmp_path, current_step=250)

    assert plan["checkpoint_dir"] == tmp_path / "progressive" / "s4" / "checkpoints" / "step_500"
    assert plan["train_env"]["RESUME_FROM_PATH"].endswith("s4/checkpoints/step_250")
    assert plan["train_env"]["RESET_RESUME_STEP"] == "0"
    assert plan["train_env"]["RESUME_OPTIMIZER_STATE"] == "1"


def test_later_stage_requires_an_explicit_selected_predecessor_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="selected predecessor"):
        _plan(tmp_path, stage="s2")

    plan = _plan(
        tmp_path,
        stage="s2",
        resume_from_path=Path("/tmp/s4_selected_step2500"),
    )
    assert plan["spec"] == {"teacher_steps": 4, "student_steps": 2, "max_steps": 3000}
    assert plan["train_env"]["RESUME_FROM_PATH"] == "/tmp/s4_selected_step2500"
    assert plan["train_env"]["RESET_RESUME_STEP"] == "1"


def test_standalone_opd_runner_rejects_gradient_accumulation_above_one(tmp_path):
    with pytest.raises(ValueError, match="gradient_accumulation_steps=1"):
        _plan(tmp_path, gradient_accumulation_steps=8)


def _flowmap_method(name, **overrides):
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_step.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    mixin = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowMapStepMixin"
    )
    method = next(
        node for node in mixin.body
        if isinstance(node, ast.FunctionDef)
        and node.name == name
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "torch": torch,
        "broadcast_joint_action_timesteps": broadcast_joint_action_timesteps,
        "deployment_endpoint_losses": deployment_endpoint_losses,
        "cosmos_actions_to_flowmap_x0": lambda actions, **kwargs: torch.full(
            kwargs["target_shape"],
            99.0,
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        ),
        "_downsample_action_grid_id": lambda grid_id, *_args: grid_id,
        "SUPPORTED_ROLLOUT_GRAD_MODES": SUPPORTED_ROLLOUT_GRAD_MODES,
        "rollout_step_requires_grad": rollout_step_requires_grad,
        "contextlib": contextlib,
        "_synchronized_nonfinite_decision": _synchronized_nonfinite_decision,
    }
    namespace.update(overrides)
    exec(compile(module, "<deployment-method>", "exec"), namespace)
    return namespace[name]


def _deployment_method():
    return _flowmap_method("_cosmos_deployment_joint_rollout_step")


def test_full_deployment_step_integrates_the_same_joint_1000_to_zero_path():
    class Teacher:
        raw_inference_enabled = True

        def __init__(self):
            self.calls = []

        def predict_raw_latent_target(self, batch, **kwargs):
            self.calls.append(kwargs)
            return {
                "cosmos_latent_x0": torch.full((1, 1, 9, 1, 1), 2.0),
                "actions": torch.full((1, 16, 1), 99.0),
            }

    class Scheduler:
        @staticmethod
        def training_target(clean, noise, timesteps):
            return noise - clean

    class Harness:
        _cosmos_deployment_joint_rollout_step = _deployment_method()

        def __init__(self):
            self.device = torch.device("cpu")
            self.gradient_accumulation_steps = 1
            self._action_teacher_model = Teacher()
            self.train_scheduler_latent = Scheduler()
            self.train_scheduler_action = Scheduler()
            self.empty_emb = torch.zeros(1, 1, 1)
            self.integrate_calls = []
            self.config = SimpleNamespace(
                cosmos_latent_channels=1,
                cosmos_latent_frames=9,
                cosmos_latent_height=1,
                cosmos_latent_width=1,
                cosmos_latent_t_min=0.8,
                cosmos_latent_t_max=80.0 / 81.0,
                cosmos_latent_epsilon=0.001,
                num_train_timesteps=1000,
                cfg_min=1.0,
                cfg_max=1.0,
                norm_stat={"q01": [0.0], "q99": [1.0]},
                inverse_used_action_channel_ids=[0],
                action_packing_schema="libero",
                action_downsample_factor=4,
                deployment_action_weight=1.0,
                opd_joint_action_rollout=True,
                opd_rollout_grad_mode="last_step",
            )

        @staticmethod
        def convert_input_format(batch):
            return batch

        @staticmethod
        def _prepare_base_dict(batch):
            return {
                "latent_dict": {
                    "latent": batch["latents"],
                    "cond_timesteps": torch.zeros(1, 9),
                    "text_emb": torch.zeros(1, 1, 1),
                },
                "action_dict": {
                    "latent": batch["actions"],
                    "cond_timesteps": torch.zeros(1, 16),
                    "text_emb": torch.zeros(1, 1, 1),
                    "grid_id": None,
                    "actions_mask": batch["actions_mask"],
                },
                "chunk_size": 1,
                "window_size": 1,
            }

        def _student_euler_integrate(self, **kwargs):
            self.integrate_calls.append(kwargs)
            video = torch.full_like(
                kwargs["noisy_latents"], 9.0, requires_grad=True
            )
            action = torch.tensor(
                [9.0, 99.0, 9.0, 99.0], requires_grad=True
            ).reshape(1, 1, 4, 1, 1)
            return video, torch.zeros_like(video), None, action

    harness = Harness()
    batch = {
        "latents": torch.zeros(1, 1, 9, 1, 1),
        "actions": torch.full((1, 1, 16, 1, 1), 3.0),
        "actions_mask": torch.tensor(
            [1.0] * 4 + [0.0] * 4 + [1.0] * 4 + [0.0] * 4
        ).reshape(1, 1, 16, 1, 1),
    }

    result = harness._cosmos_deployment_joint_rollout_step(
        batch, 0, student_steps=2
    )

    call = harness.integrate_calls[0]
    assert torch.equal(call["timesteps"], torch.full((1, 9), 1000.0))
    assert torch.equal(call["target_r"], torch.zeros(1, 9))
    assert torch.equal(call["action_target_r"], torch.zeros(1, 16))
    assert call["K_steps"] == 2
    assert call["return_final_action_state"] is True
    assert torch.equal(
        call["base_input_dict"]["action_dict"]["actions_mask"],
        batch["actions_mask"][:, :, ::4],
    )
    assert harness._action_teacher_model.calls[0]["include_cdiff"] is False
    assert result["deployment_video_endpoint_loss"].item() == pytest.approx(49.0)
    assert result["deployment_action_endpoint_loss"].item() == pytest.approx(36.0)
    assert result["deployment_total_loss"].item() == pytest.approx(85.0)
    assert result["deployment_t_start"].item() == 1000
    assert result["deployment_t_end"].item() == 0

    endpoint_harness = Harness()
    endpoint_harness.config.opd_rollout_grad_mode = "endpoint"
    with pytest.raises(ValueError, match="deployment.*endpoint"):
        endpoint_harness._cosmos_deployment_joint_rollout_step(
            batch, 0, student_steps=2
        )

    nonfinite_harness = Harness()

    def nonfinite_integrate(**kwargs):
        video = torch.full_like(
            kwargs["noisy_latents"], float("nan"), requires_grad=True
        )
        action = torch.full(
            (1, 1, 4, 1, 1), float("inf"), requires_grad=True
        )
        return video, torch.zeros_like(video), None, action

    nonfinite_harness._student_euler_integrate = nonfinite_integrate
    nonfinite_result = nonfinite_harness._cosmos_deployment_joint_rollout_step(
        batch, 0, student_steps=2
    )
    assert nonfinite_result["skip_step"] is True
    for name in (
        "loss",
        "deployment_video_endpoint_loss",
        "deployment_action_endpoint_loss",
        "deployment_total_loss",
        "deployment_student_steps",
        "deployment_t_start",
        "deployment_t_end",
    ):
        assert torch.isfinite(nonfinite_result[name])


def test_real_student_euler_final_states_backpropagate_to_student(monkeypatch):
    fake_model_module = ModuleType("modules.model")

    class FlexAttnFunc:
        @staticmethod
        def init_mask(*_args, **_kwargs):
            return None

    fake_model_module.FlexAttnFunc = FlexAttnFunc
    monkeypatch.setitem(sys.modules, "modules.model", fake_model_module)

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.blocks = nn.ModuleList()

    class Harness:
        _student_euler_integrate = _flowmap_method(
            "_student_euler_integrate"
        )

        def __init__(self):
            self.device = torch.device("cpu")
            self.patch_size = (1, 1, 1)
            self.student = Student()
            self.config = SimpleNamespace(
                action_downsample_factor=4,
                opd_joint_action_rollout=True,
                opd_rollout_grad_mode="last_step",
                opd_rollout_grad_steps=1,
                offline_eval_force_gradient_checkpointing=False,
                offline_eval_force_cfg=False,
                num_train_timesteps=1000,
            )
            self.distill_action = True
            self.action_aware = False

        @staticmethod
        def _build_timestep_path(t, r, steps):
            alpha = torch.linspace(0.0, 1.0, steps + 1)
            return t.unsqueeze(0) + (r - t).unsqueeze(0) * alpha[:, None, None]

        @staticmethod
        def _timestep_to_sigma_5d(t):
            return t[:, None, :, None, None] / 1000.0

        @staticmethod
        def _extract_action_v(action, _frames):
            return action

        @staticmethod
        def _student_cfg_forward(
            model, step_input, *_args, return_action=False, **_kwargs
        ):
            video = model.scale.expand_as(
                step_input["latent_dict"]["noisy_latents"]
            )
            if not return_action:
                return video
            action = model.scale.expand_as(
                step_input["action_dict"]["noisy_latents"]
            )
            return video, action

    harness = Harness()
    video_noise = torch.full((1, 1, 1, 1, 1), 10.0)
    action_noise = torch.full((1, 1, 1, 1, 1), 10.0)
    base_input = {
        "latent_dict": {"latent": torch.zeros_like(video_noise)},
        "action_dict": {
            "latent": torch.zeros_like(action_noise),
            "noisy_latents": action_noise,
            "timesteps": torch.full((1, 1), 1000.0),
        },
        "chunk_size": 1,
        "window_size": 1,
    }

    video_final, _, _, action_final = harness._student_euler_integrate(
        noisy_latents=video_noise,
        timesteps=torch.full((1, 1), 1000.0),
        target_r=torch.zeros(1, 1),
        base_input_dict=base_input,
        empty_emb=torch.zeros(1, 1, 1),
        cfg_scale=1.0,
        ref_shape=video_noise.shape,
        B=1,
        num_frames=1,
        K_steps=1,
        action_target_r=torch.zeros(1, 4),
        return_final_action=True,
        return_final_action_state=True,
    )

    assert video_final.requires_grad
    assert action_final.requires_grad
    (video_final.sum() + action_final.sum()).backward()
    assert harness.student.scale.grad is not None
    assert harness.student.scale.grad.abs().item() > 0
