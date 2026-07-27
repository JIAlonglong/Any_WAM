import ast
import contextlib
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch
import torch.distributed as dist
from torch import nn

from distillation_flowmap.cosmos_deployment_rollout import deployment_endpoint_losses
from distillation_flowmap.cosmos_policy_adapter import unpack_flowmap_action_query
from distillation_flowmap.cosmos_progressive_opd import broadcast_joint_action_timesteps
from distillation_flowmap.cosmos_training_contract import pack_actions_for_downsample
from distillation_flowmap.danceopd_query import (
    aligned_anchor_mse,
    build_shifted_terminal_path,
    masked_video_velocity_mse,
    sample_nonterminal_semantic_query_indices,
    select_per_sample_trajectory_state,
)
from distillation_flowmap.distributed_safety import (
    NONFINITE_ORIGINS,
    all_ranks_finite,
    reduce_nonfinite_origins,
)
import distillation_flowmap.flowmap_step as flowmap_step_module
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
        "aligned_anchor_mse": aligned_anchor_mse,
        "all_ranks_finite": all_ranks_finite,
        "build_shifted_terminal_path": build_shifted_terminal_path,
        "dist": dist,
        "masked_video_velocity_mse": masked_video_velocity_mse,
        "pack_actions_for_downsample": pack_actions_for_downsample,
        "sample_nonterminal_semantic_query_indices": sample_nonterminal_semantic_query_indices,
        "select_per_sample_trajectory_state": select_per_sample_trajectory_state,
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


def test_aligned_cosmos_video_opd_reuses_one_generated_canonical_joint_state():
    class Student(nn.Module):
        def __init__(self, perturb=0.0):
            super().__init__()
            self.rollout_scale = nn.Parameter(torch.tensor(1.0))
            self.field_scale = nn.Parameter(torch.tensor(0.5))
            self.anchor_scale = nn.Parameter(torch.tensor(-0.5))
            self.perturb = perturb
            self.calls = []

    class Teacher:
        raw_inference_enabled = True

        def __init__(self):
            self.endpoint_call = None

        def predict_raw_joint_latent_velocity(
            self, batch, query_latent, query_action, t
        ):
            del batch, query_action, t
            return {
                "cosmos_joint_query": query_latent,
                "cosmos_latent_velocity": torch.full_like(query_latent, 2.0),
                "cosmos_video_frame_mask": torch.ones(
                    query_latent.shape[0], query_latent.shape[2], dtype=torch.bool
                ),
            }

        def predict_raw_same_prior_endpoint(
            self, batch, *, video_prior, action_prior, teacher_steps
        ):
            del batch
            self.endpoint_call = SimpleNamespace(
                video_prior=video_prior,
                action_prior=action_prior,
                teacher_steps=teacher_steps,
            )
            return {
                "endpoint_video": torch.full_like(video_prior, -2.0),
                "video_frame_mask": torch.ones(
                    video_prior.shape[0], video_prior.shape[2], dtype=torch.bool
                ),
                "effective_teacher_steps": 8,
                "video_prior_sha256": "verified",
                "action_prior_sha256": "verified",
            }

    class Harness:
        def __init__(self, perturb=0.0):
            self.device = torch.device("cpu")
            self.student = Student(perturb)
            self._action_teacher_model = Teacher()
            self.empty_emb = torch.zeros(1, 1, 1)
            self.config = SimpleNamespace(
                rank=0,
                num_train_timesteps=1000,
                cosmos_latent_channels=1,
                cosmos_latent_frames=3,
                cosmos_latent_height=1,
                cosmos_latent_width=1,
                action_downsample_factor=4,
                action_packing_schema="downsample_survivor_v2",
                used_action_channel_ids=list(range(7)),
                snr_shift=5.0,
                action_snr_shift=0.05,
                opd_danceopd_rollout_steps=(2, 4),
                cfg_min=1.0,
                cfg_max=1.0,
            )

        @staticmethod
        def convert_input_format(batch):
            return batch

        @staticmethod
        def _prepare_base_dict(batch):
            return {
                "latent_dict": {
                    "latent": batch["latents"],
                    "cond_timesteps": torch.zeros(1, 3),
                    "text_emb": torch.zeros(1, 1, 1),
                    "grid_id": None,
                },
                "action_dict": {
                    "latent": batch["actions"],
                    "cond_timesteps": torch.zeros(1, 16),
                    "text_emb": torch.zeros(1, 1, 1),
                    "grid_id": None,
                    "actions_mask": torch.ones_like(batch["actions"][:, :1]),
                },
                "chunk_size": 1,
                "window_size": 1,
            }

        @staticmethod
        def _mechanism_joint_input(video, action, video_t, action_t, context):
            return {
                "latent_dict": {
                    **context["video_base"],
                    "noisy_latents": video,
                    "timesteps": video_t,
                },
                "action_dict": {
                    "noisy_latents": action,
                    "latent": context["action_latent"],
                    "timesteps": action_t,
                    "cond_timesteps": context["action_cond_t"],
                    "text_emb": context["action_text"],
                },
                "chunk_size": 1,
                "window_size": 1,
            }

        @staticmethod
        def _init_joint_mask(_joint_input):
            return None

        @staticmethod
        def _extract_action_v(action, _frames):
            return action

        @staticmethod
        def _timestep_to_sigma_5d(t):
            return t[:, None, :, None, None] / 1000.0

        def _student_joint_forward(
            self,
            model,
            joint_input,
            _empty,
            video_r,
            action_r,
            *,
            require_action,
            **_kwargs,
        ):
            del action_r
            video = joint_input["latent_dict"]["noisy_latents"]
            action = joint_input["action_dict"]["noisy_latents"]
            assert joint_input["latent_dict"]["latent"].data_ptr() == video.data_ptr()
            assert joint_input["action_dict"]["latent"].data_ptr() == action.data_ptr()
            kind = (
                "rollout"
                if require_action
                else ("anchor" if bool((video_r == 0).all()) else "field")
            )
            model.calls.append(SimpleNamespace(kind=kind, video=video, action=action))
            if require_action:
                video_v = torch.ones_like(video) * (
                    model.rollout_scale + model.perturb
                )
                return video_v, torch.ones_like(action) * model.rollout_scale
            scale = model.anchor_scale if kind == "anchor" else model.field_scale
            return torch.ones_like(video) * scale

        def _joint_euler_update(
            self, video, action, video_v, action_v, vt, vr, at, ar
        ):
            return (
                video
                + video_v
                * (
                    self._timestep_to_sigma_5d(vr)
                    - self._timestep_to_sigma_5d(vt)
                ),
                action
                + action_v
                * (
                    self._timestep_to_sigma_5d(ar)
                    - self._timestep_to_sigma_5d(at)
                ),
            )

    Harness._build_cosmos_shifted_shared_query = _flowmap_method(
        "_build_cosmos_shifted_shared_query"
    )
    Harness._cosmos_aligned_video_opd_step = _flowmap_method(
        "_cosmos_aligned_video_opd_step"
    )
    batch = {
        "gt_video": torch.full((1, 1, 3, 1, 1), 91.0),
        "gt_action": torch.full((1, 7, 16, 4, 1), 73.0),
    }
    batch["latents"] = batch["gt_video"]
    batch["actions"] = batch["gt_action"]

    def run(perturb):
        torch.manual_seed(17)
        harness = Harness(perturb)
        result = harness._cosmos_aligned_video_opd_step(batch.copy(), 0)
        return harness, result

    harness, result = run(0.0)
    rollout_calls = [
        call for call in harness.student.calls if call.kind == "rollout"
    ]
    field_call = next(call for call in harness.student.calls if call.kind == "field")
    anchor_call = next(
        call for call in harness.student.calls if call.kind == "anchor"
    )

    assert len(rollout_calls) in (2, 4)
    assert torch.equal(field_call.video, anchor_call.video)
    assert torch.equal(field_call.action, anchor_call.action)
    assert field_call.video.data_ptr() == anchor_call.video.data_ptr()
    assert field_call.action.data_ptr() == anchor_call.action.data_ptr()
    assert not field_call.video.requires_grad
    assert not field_call.action.requires_grad
    assert not torch.equal(field_call.video, batch["gt_video"])
    assert not torch.equal(field_call.action, batch["gt_action"][:, :, ::4])
    for previous, current in zip(rollout_calls, rollout_calls[1:]):
        assert not torch.equal(previous.video, current.video)

    endpoint_call = harness._action_teacher_model.endpoint_call
    assert endpoint_call.teacher_steps == 8
    assert endpoint_call.video_prior.data_ptr() == rollout_calls[0].video.data_ptr()
    unpacked_prior = unpack_flowmap_action_query(
        rollout_calls[0].action,
        used_action_channel_ids=list(range(7)),
        packing_schema="downsample_survivor_v2",
        downsample_factor=4,
    )
    assert torch.equal(unpacked_prior, endpoint_call.action_prior)

    perturbed, _ = run(0.75)
    perturbed_field = next(
        call for call in perturbed.student.calls if call.kind == "field"
    )
    perturbed_anchor = next(
        call for call in perturbed.student.calls if call.kind == "anchor"
    )
    assert not torch.equal(field_call.video, perturbed_field.video)
    assert not torch.equal(anchor_call.video, perturbed_anchor.video)

    assert harness.student.rollout_scale.grad is None
    assert harness.student.field_scale.grad is not None
    assert harness.student.field_scale.grad.abs().item() > 0
    assert harness.student.anchor_scale.grad is not None
    assert harness.student.anchor_scale.grad.abs().item() > 0
    assert result["skip_step"] is False
    assert result["opd_teacher_steps"].item() == 8
    assert result["opd_same_prior_verified"].item() == 1
    assert result["opd_canonical_state_verified"].item() == 1
    assert "opd_action_endpoint_loss" not in result
    assert "opd_action_field_loss" not in result


class _AlignedRecorderStudent(nn.Module):
    def __init__(self, events=None):
        super().__init__()
        self.rollout_scale = nn.Parameter(torch.tensor(1.0))
        self.field_scale = nn.Parameter(torch.tensor(0.5))
        self.anchor_scale = nn.Parameter(torch.tensor(-0.5))
        self.calls = []
        self.events = events


class _AlignedRecorderTeacher:
    raw_inference_enabled = True

    def __init__(
        self,
        *,
        events=None,
        canonical_mismatch=False,
        endpoint_mask_mismatch=False,
    ):
        self.events = events
        self.canonical_mismatch = canonical_mismatch
        self.endpoint_mask_mismatch = endpoint_mask_mismatch
        self.endpoint_call = None

    def predict_raw_joint_latent_velocity(
        self, batch, query_latent, query_action, t
    ):
        del batch
        if self.events is not None:
            self.events.append("teacher:field")
        self.field_call = SimpleNamespace(
            video=query_latent,
            action=query_action,
            t=t,
        )
        canonical = query_latent
        if self.canonical_mismatch:
            canonical = query_latent.clone()
            canonical[0, :, 0] += 1.0
        mask = torch.ones(
            query_latent.shape[0],
            query_latent.shape[2],
            dtype=torch.bool,
            device=query_latent.device,
        )
        return {
            "cosmos_joint_query": canonical,
            "cosmos_latent_velocity": torch.full_like(query_latent, 2.0),
            "cosmos_video_frame_mask": mask,
        }

    def predict_raw_same_prior_endpoint(
        self, batch, *, video_prior, action_prior, teacher_steps
    ):
        del batch
        if self.events is not None:
            self.events.append("teacher:endpoint")
        self.endpoint_call = SimpleNamespace(
            video_prior=video_prior,
            action_prior=action_prior,
            teacher_steps=teacher_steps,
        )
        mask = torch.ones(
            video_prior.shape[0],
            video_prior.shape[2],
            dtype=torch.bool,
            device=video_prior.device,
        )
        if self.endpoint_mask_mismatch:
            mask = mask.clone()
            mask[0, 0] = False
        return {
            "endpoint_video": torch.full_like(video_prior, -2.0),
            "video_frame_mask": mask,
            "effective_teacher_steps": 8,
            "video_prior_sha256": "provider-verified",
            "action_prior_sha256": "provider-verified",
        }


class _AlignedRecorderHarness:
    def __init__(self, batch_size, teacher, events=None):
        self.device = torch.device("cpu")
        self.student = _AlignedRecorderStudent(events)
        self._action_teacher_model = teacher
        self.empty_emb = torch.zeros(1, 1, 1)
        self.config = SimpleNamespace(
            rank=0,
            num_train_timesteps=1000,
            cosmos_latent_channels=1,
            cosmos_latent_frames=3,
            cosmos_latent_height=1,
            cosmos_latent_width=1,
            action_downsample_factor=4,
            action_packing_schema="downsample_survivor_v2",
            used_action_channel_ids=list(range(7)),
            snr_shift=5.0,
            action_snr_shift=0.05,
            opd_danceopd_rollout_steps=(2, 4),
            cfg_min=1.0,
            cfg_max=1.0,
        )
        self.batch_size = batch_size

    @staticmethod
    def convert_input_format(batch):
        return batch

    @staticmethod
    def _prepare_base_dict(batch):
        batch_size = batch["latents"].shape[0]
        return {
            "latent_dict": {
                "latent": batch["latents"],
                "cond_timesteps": torch.zeros(batch_size, 3),
                "text_emb": torch.zeros(batch_size, 1, 1),
                "grid_id": None,
            },
            "action_dict": {
                "latent": batch["actions"],
                "cond_timesteps": torch.zeros(batch_size, 16),
                "text_emb": torch.zeros(batch_size, 1, 1),
                "grid_id": None,
                "actions_mask": torch.ones_like(batch["actions"][:, :1]),
            },
            "chunk_size": 1,
            "window_size": 1,
        }

    @staticmethod
    def _mechanism_joint_input(video, action, video_t, action_t, context):
        return {
            "latent_dict": {
                **context["video_base"],
                "noisy_latents": video,
                "timesteps": video_t,
            },
            "action_dict": {
                "noisy_latents": action,
                "latent": context["action_latent"],
                "timesteps": action_t,
                "cond_timesteps": context["action_cond_t"],
                "text_emb": context["action_text"],
            },
            "chunk_size": 1,
            "window_size": 1,
        }

    @staticmethod
    def _init_joint_mask(_joint_input):
        return None

    @staticmethod
    def _extract_action_v(action, _frames):
        return action

    @staticmethod
    def _timestep_to_sigma_5d(t):
        return t[:, None, :, None, None] / 1000.0

    def _student_joint_forward(
        self,
        model,
        joint_input,
        _empty,
        video_r,
        action_r,
        *,
        require_action,
        **_kwargs,
    ):
        video = joint_input["latent_dict"]["noisy_latents"]
        action = joint_input["action_dict"]["noisy_latents"]
        kind = (
            "rollout"
            if require_action
            else ("anchor" if bool((video_r == 0).all()) else "field")
        )
        model.calls.append(
            SimpleNamespace(
                kind=kind,
                video=video,
                action=action,
                video_t=joint_input["latent_dict"]["timesteps"],
                video_cond_t=joint_input["latent_dict"]["cond_timesteps"],
                action_t=joint_input["action_dict"]["timesteps"],
                video_r=video_r,
                action_r=action_r,
            )
        )
        if require_action:
            return (
                torch.ones_like(video) * model.rollout_scale,
                torch.ones_like(action) * model.rollout_scale,
            )
        scale = model.anchor_scale if kind == "anchor" else model.field_scale
        return torch.ones_like(video) * scale

    def _joint_euler_update(
        self, video, action, video_v, action_v, vt, vr, at, ar
    ):
        return (
            video
            + video_v
            * (self._timestep_to_sigma_5d(vr) - self._timestep_to_sigma_5d(vt)),
            action
            + action_v
            * (self._timestep_to_sigma_5d(ar) - self._timestep_to_sigma_5d(at)),
        )


class _ScriptedAllRanksFinite:
    def __init__(self, outcomes=None, events=None):
        self.outcomes = list(outcomes or ())
        self.events = events
        self.calls = []

    def __call__(self, local_finite, *, device):
        del device
        call_index = len(self.calls)
        local_finite = bool(local_finite)
        self.calls.append(local_finite)
        if self.events is not None:
            self.events.append(f"gate:{call_index}")
        if self.outcomes:
            return bool(self.outcomes.pop(0))
        return local_finite


class _ScriptedOriginDist:
    class ReduceOp:
        MAX = "max"

    def __init__(self, *, remote_origin=None, events=None):
        self.remote_origin = remote_origin
        self.events = events
        self.call_index = 0

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_initialized():
        return True

    def all_reduce(self, value, op):
        assert op == self.ReduceOp.MAX
        origin = NONFINITE_ORIGINS[self.call_index]
        if self.events is not None:
            self.events.append(f"origin:{origin}")
        if origin == self.remote_origin:
            value.fill_(1)
        self.call_index += 1


def _aligned_recording_case(
    *,
    batch_size=1,
    query_indices=None,
    all_rank_outcomes=None,
    events=None,
    canonical_mismatch=False,
    endpoint_mask_mismatch=False,
    rollout_steps=(2, 4),
    endpoint_weight=1.0,
    velocity_weight=1.0,
):
    teacher = _AlignedRecorderTeacher(
        events=events,
        canonical_mismatch=canonical_mismatch,
        endpoint_mask_mismatch=endpoint_mask_mismatch,
    )
    gate = _ScriptedAllRanksFinite(all_rank_outcomes, events)
    overrides = {"all_ranks_finite": gate}
    if query_indices is not None:
        forced = torch.as_tensor(query_indices, dtype=torch.long)
        overrides["sample_nonterminal_semantic_query_indices"] = (
            lambda _sigmas, batch_size: forced.clone()
        )

    class Harness(_AlignedRecorderHarness):
        pass

    Harness._build_cosmos_shifted_shared_query = _flowmap_method(
        "_build_cosmos_shifted_shared_query",
        **overrides,
    )
    Harness._cosmos_aligned_video_opd_step = _flowmap_method(
        "_cosmos_aligned_video_opd_step",
        **overrides,
    )
    harness = Harness(batch_size, teacher, events)
    harness.config.opd_danceopd_rollout_steps = rollout_steps
    harness.config.opd_danceopd_endpoint_weight = endpoint_weight
    harness.config.opd_danceopd_velocity_weight = velocity_weight
    batch = {
        "latents": torch.full((batch_size, 1, 3, 1, 1), 91.0),
        "actions": torch.full((batch_size, 7, 16, 4, 1), 73.0),
    }
    return harness, batch, gate


@pytest.mark.parametrize(
    ("arm", "rollout_steps", "endpoint_weight", "velocity_weight"),
    [
        ("s1", (1,), 1.0, 0.0),
        ("s2", (2,), 1.0, 1.0),
        ("s4", (4,), 1.0, 1.0),
        ("universal", (2, 4), 1.0, 1.0),
        ("universal-video-action", (2, 4), 1.0, 1.0),
        ("stage1_only", (2, 4), 0.0, 0.0),
        ("anchor_only", (2, 4), 1.0, 0.0),
        ("field_only", (2, 4), 0.0, 1.0),
        ("apm", (2, 4), 1.0, 1.0),
    ],
)
def test_aligned_runtime_executes_every_canonical_arm_without_cross_arm_defaults(
    arm, rollout_steps, endpoint_weight, velocity_weight
):
    harness, batch, _ = _aligned_recording_case(
        rollout_steps=rollout_steps,
        endpoint_weight=endpoint_weight,
        velocity_weight=velocity_weight,
    )

    result = harness._cosmos_aligned_video_opd_step(batch, 0)
    call_kinds = [call.kind for call in harness.student.calls]

    assert result["skip_step"] is False
    assert result["opd_endpoint_contrib"].item() == pytest.approx(
        result["opd_endpoint_loss"].item() * endpoint_weight
    )
    assert result["opd_same_state_velocity_contrib"].item() == pytest.approx(
        result["opd_same_state_velocity_loss"].item() * velocity_weight
    )
    if arm == "stage1_only":
        assert call_kinds == []
        assert harness._action_teacher_model.endpoint_call is None
        assert not hasattr(harness._action_teacher_model, "field_call")
        assert result["loss"].item() == 0.0
    elif arm == "s1":
        assert call_kinds == ["anchor"]
        assert harness._action_teacher_model.endpoint_call is not None
        assert not hasattr(harness._action_teacher_model, "field_call")
    else:
        assert ("anchor" in call_kinds) is bool(endpoint_weight)
        assert ("field" in call_kinds) is bool(velocity_weight)
        assert (
            harness._action_teacher_model.endpoint_call is not None
        ) is bool(endpoint_weight)
        assert hasattr(
            harness._action_teacher_model, "field_call"
        ) is bool(velocity_weight)


@pytest.mark.parametrize("rollout_steps", [(3,), (1, 2), (4, 2), ()])
def test_aligned_runtime_rejects_noncanonical_rollout_choice_sets(rollout_steps):
    harness, batch, _ = _aligned_recording_case(
        rollout_steps=rollout_steps,
    )

    with pytest.raises(ValueError, match="canonical"):
        harness._cosmos_aligned_video_opd_step(batch, 0)


def test_aligned_runtime_rejects_field_loss_for_one_step_budget():
    harness, batch, _ = _aligned_recording_case(
        rollout_steps=(1,),
        endpoint_weight=1.0,
        velocity_weight=1.0,
    )

    with pytest.raises(ValueError, match="K=1.*field"):
        harness._cosmos_aligned_video_opd_step(batch, 0)

    assert harness.student.calls == []
    assert not hasattr(harness._action_teacher_model, "field_call")
    assert harness._action_teacher_model.endpoint_call is None


def test_aligned_runtime_applies_configured_weights_to_exact_contributions():
    harness, batch, _ = _aligned_recording_case(
        rollout_steps=(2,),
        endpoint_weight=0.25,
        velocity_weight=3.0,
    )

    result = harness._cosmos_aligned_video_opd_step(batch, 0)

    torch.testing.assert_close(
        result["opd_endpoint_contrib"],
        result["opd_endpoint_loss"] * 0.25,
    )
    torch.testing.assert_close(
        result["opd_same_state_velocity_contrib"],
        result["opd_same_state_velocity_loss"] * 3.0,
    )
    torch.testing.assert_close(
        result["loss"],
        result["opd_endpoint_contrib"]
        + result["opd_same_state_velocity_contrib"],
    )


def _install_scripted_origin_reducer(monkeypatch, *, remote_origin, events):
    scripted_dist = _ScriptedOriginDist(
        remote_origin=remote_origin,
        events=events,
    )

    def scripted_reduce(flags, *, device):
        return reduce_nonfinite_origins(
            flags,
            device=device,
            dist_module=scripted_dist,
        )

    monkeypatch.setattr(
        flowmap_step_module,
        "reduce_nonfinite_origins",
        scripted_reduce,
    )
    return scripted_dist


def test_aligned_cosmos_gathers_distinct_per_sample_joint_states_and_shifted_times(
    monkeypatch,
):
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: torch.tensor([1], device=kwargs.get("device")),
    )
    harness, batch, _ = _aligned_recording_case(
        batch_size=2,
        query_indices=[1, 2],
    )

    result = harness._cosmos_aligned_video_opd_step(batch, 0)

    rollout = [call for call in harness.student.calls if call.kind == "rollout"]
    field = next(call for call in harness.student.calls if call.kind == "field")
    anchor = next(call for call in harness.student.calls if call.kind == "anchor")
    video_sigmas = build_shifted_terminal_path(
        steps=4,
        shift=5.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    action_sigmas = build_shifted_terminal_path(
        steps=4,
        shift=0.05,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    expected_video_t = torch.stack(
        [
            video_sigmas[1].expand(3),
            video_sigmas[2].expand(3),
        ]
    ) * 1000
    expected_action_t = torch.stack(
        [
            action_sigmas[1].expand(4),
            action_sigmas[2].expand(4),
        ]
    ) * 1000

    assert len(rollout) == 4
    assert torch.equal(field.video[0], rollout[1].video[0])
    assert torch.equal(field.video[1], rollout[2].video[1])
    assert torch.equal(field.action[0], rollout[1].action[0])
    assert torch.equal(field.action[1], rollout[2].action[1])
    torch.testing.assert_close(field.video_t, expected_video_t)
    torch.testing.assert_close(field.video_r, expected_video_t)
    torch.testing.assert_close(field.action_t, expected_action_t)
    torch.testing.assert_close(field.action_r, expected_action_t)
    torch.testing.assert_close(anchor.video_t, expected_video_t)
    torch.testing.assert_close(anchor.action_t, expected_action_t)
    assert torch.equal(anchor.video_r, torch.zeros_like(expected_video_t))
    assert torch.equal(anchor.action_r, torch.zeros_like(expected_action_t))
    assert result["opd_query_index"].item() == pytest.approx(1.5)


def test_aligned_cosmos_rebuilds_clean_condition_clock_for_generated_video_state():
    harness, batch, _ = _aligned_recording_case(batch_size=1)

    def mismatched_base_dict(_batch):
        return {
            "latent_dict": {
                "latent": torch.zeros(1, 1, 16, 1, 1),
                "cond_timesteps": torch.zeros(1, 16),
                "text_emb": torch.zeros(1, 1, 1),
                "grid_id": None,
            },
            "action_dict": {
                "latent": batch["actions"],
                "cond_timesteps": torch.zeros(1, 16),
                "text_emb": torch.zeros(1, 1, 1),
                "grid_id": None,
                "actions_mask": torch.ones_like(batch["actions"][:, :1]),
            },
            "chunk_size": 1,
            "window_size": 1,
        }

    harness._prepare_base_dict = mismatched_base_dict
    harness._build_cosmos_shifted_shared_query(
        batch,
        mismatched_base_dict(batch),
        student_steps=2,
    )

    assert harness.student.calls
    for call in harness.student.calls:
        assert call.video.shape[2] == 3
        assert call.video_cond_t.shape == (1, 3)
        assert torch.count_nonzero(call.video_cond_t).item() == 0


def test_aligned_cosmos_rejects_canonical_valid_video_mismatch_before_queries():
    harness, batch, _ = _aligned_recording_case(canonical_mismatch=True)

    with pytest.raises(RuntimeError, match="canonical valid-video frames differ"):
        harness._cosmos_aligned_video_opd_step(batch, 0)

    assert {call.kind for call in harness.student.calls} == {"rollout"}
    assert harness.student.field_scale.grad is None
    assert harness.student.anchor_scale.grad is None


def test_aligned_cosmos_rejects_endpoint_field_mask_disagreement_before_queries():
    harness, batch, _ = _aligned_recording_case(endpoint_mask_mismatch=True)

    with pytest.raises(RuntimeError, match="shape, mask, or step contract"):
        harness._cosmos_aligned_video_opd_step(batch, 0)

    assert {call.kind for call in harness.student.calls} == {"rollout"}
    assert harness.student.field_scale.grad is None
    assert harness.student.anchor_scale.grad is None


def test_aligned_cosmos_remote_teacher_failure_raises_before_student_queries():
    events = []
    harness, batch, gate = _aligned_recording_case(
        all_rank_outcomes=[False],
        events=events,
    )

    with pytest.raises(RuntimeError, match="same-state Teacher query failed"):
        harness._cosmos_aligned_video_opd_step(batch, 0)

    assert gate.calls == [True]
    assert events[-2:] == ["teacher:field", "gate:0"]
    assert harness._action_teacher_model.endpoint_call is None
    assert {call.kind for call in harness.student.calls} == {"rollout"}


def test_aligned_cosmos_remote_contract_failure_raises_before_backward():
    events = []
    harness, batch, gate = _aligned_recording_case(
        all_rank_outcomes=[True, True, False],
        events=events,
    )

    with pytest.raises(RuntimeError, match="response contract mismatch"):
        harness._cosmos_aligned_video_opd_step(batch, 0)

    assert gate.calls == [True, True, True]
    assert events[-1] == "gate:2"
    assert {call.kind for call in harness.student.calls} == {"rollout"}
    assert harness.student.field_scale.grad is None
    assert harness.student.anchor_scale.grad is None


def test_aligned_cosmos_remote_nonfinite_skips_backward_and_reduces_origins(
    monkeypatch,
):
    events = []
    scripted_dist = _install_scripted_origin_reducer(
        monkeypatch,
        remote_origin="opd_compositional",
        events=events,
    )
    harness, batch, gate = _aligned_recording_case(
        all_rank_outcomes=[True] * 5,
        events=events,
    )

    result = harness._cosmos_aligned_video_opd_step(batch, 0)

    assert gate.calls == [True] * 5
    assert scripted_dist.call_index == len(NONFINITE_ORIGINS)
    assert result["skip_step"] is True
    assert result["nonfinite_origins"]["opd_compositional"] is True
    assert harness.student.field_scale.grad is None
    assert harness.student.anchor_scale.grad is None
    assert "backward" not in events
    assert events[:7] == [
        "teacher:field",
        "gate:0",
        "teacher:endpoint",
        "gate:1",
        "gate:2",
        "gate:3",
        "gate:4",
    ]
    assert [
        event for event in events if event.startswith("origin:")
    ] == [f"origin:{origin}" for origin in NONFINITE_ORIGINS]


def test_aligned_cosmos_collective_order_precedes_successful_backward(monkeypatch):
    events = []
    scripted_dist = _install_scripted_origin_reducer(
        monkeypatch,
        remote_origin=None,
        events=events,
    )
    harness, batch, gate = _aligned_recording_case(
        all_rank_outcomes=[True] * 5,
        events=events,
    )
    harness.student.field_scale.register_hook(
        lambda grad: events.append("backward") or grad
    )

    result = harness._cosmos_aligned_video_opd_step(batch, 0)

    assert result["skip_step"] is False
    assert gate.calls == [True] * 5
    assert scripted_dist.call_index == len(NONFINITE_ORIGINS)
    assert [event for event in events if event.startswith("gate:")] == [
        f"gate:{index}" for index in range(5)
    ]
    assert events[:7] == [
        "teacher:field",
        "gate:0",
        "teacher:endpoint",
        "gate:1",
        "gate:2",
        "gate:3",
        "gate:4",
    ]
    origin_events = [event for event in events if event.startswith("origin:")]
    assert origin_events == [
        f"origin:{origin}" for origin in NONFINITE_ORIGINS
    ]
    assert events.index("backward") > max(
        events.index(event) for event in origin_events
    )
