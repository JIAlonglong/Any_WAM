import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from distillation_flowmap.cosmos_deployment_rollout import deployment_endpoint_losses
from distillation_flowmap.cosmos_progressive_opd import broadcast_joint_action_timesteps
from distillation_flowmap.run_cosmos_progressive_stage2 import build_stage_chunk_plan


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


def _deployment_method():
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
        and node.name == "_cosmos_deployment_joint_rollout_step"
    )
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {
        "torch": torch,
        "broadcast_joint_action_timesteps": broadcast_joint_action_timesteps,
        "deployment_endpoint_losses": deployment_endpoint_losses,
        "cosmos_actions_to_flowmap_x0": lambda actions, **kwargs: torch.zeros(
            kwargs["target_shape"],
            device=kwargs["device"],
            dtype=kwargs["dtype"],
        ),
        "_downsample_action_grid_id": lambda grid_id, *_args: grid_id,
    }
    exec(compile(module, "<deployment-method>", "exec"), namespace)
    return namespace["_cosmos_deployment_joint_rollout_step"]


def test_full_deployment_step_integrates_the_same_joint_1000_to_zero_path():
    class Teacher:
        raw_inference_enabled = True

        def __init__(self):
            self.calls = []

        def predict_raw_latent_target(self, batch, **kwargs):
            self.calls.append(kwargs)
            return {
                "cosmos_latent_x0": torch.zeros(1, 1, 9, 1, 1),
                "actions": torch.zeros(1, 16, 1),
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
                opd_rollout_grad_mode="full",
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
            video = torch.zeros_like(kwargs["noisy_latents"], requires_grad=True)
            action = torch.zeros(
                1, 1, 4, 1, 1, requires_grad=True
            )
            return video, torch.zeros_like(video), None, action

    harness = Harness()
    batch = {
        "latents": torch.zeros(1, 1, 9, 1, 1),
        "actions": torch.zeros(1, 1, 16, 1, 1),
        "actions_mask": torch.ones(1, 1, 16, 1, 1),
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
    assert result["deployment_t_start"].item() == 1000
    assert result["deployment_t_end"].item() == 0
