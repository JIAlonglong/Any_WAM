from types import SimpleNamespace
import sys
from pathlib import Path

import pytest

from distillation_flowmap.cosmos_teacher_roles import resolve_teacher_roles


def test_action_only_cosmos_does_not_require_video_teacher():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=False,
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.action_backend == "cosmos_policy"
    assert roles.action_model_path == "/ckpts/cosmos"
    assert roles.video_backend is None
    assert roles.video_model_path is None
    assert roles.uses_separate_video_teacher is False


def test_dual_teacher_requires_wanva_video_model_path():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=True,
    )

    with pytest.raises(ValueError, match="video_teacher_model_path"):
        resolve_teacher_roles(cfg)


def test_all_cosmos_video_target_does_not_require_wanva_video_teacher():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=True,
        cosmos_video_target=True,
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.action_backend == "cosmos_policy"
    assert roles.action_model_path == "/ckpts/cosmos"
    assert roles.video_backend == "cosmos_future"
    assert roles.video_model_path == "/ckpts/cosmos"
    assert roles.uses_separate_video_teacher is False


def test_cosmos_latent_target_does_not_require_wanva_video_teacher():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=True,
        cosmos_latent_target=True,
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.action_backend == "cosmos_policy"
    assert roles.action_model_path == "/ckpts/cosmos"
    assert roles.video_backend == "cosmos_latent"
    assert roles.video_model_path == "/ckpts/cosmos"
    assert roles.uses_separate_video_teacher is False


def test_dual_teacher_defaults_video_backend_to_wanva():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=True,
        video_teacher_model_path="/ckpts/wanva",
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.action_backend == "cosmos_policy"
    assert roles.action_model_path == "/ckpts/cosmos"
    assert roles.video_backend == "wanva"
    assert roles.video_model_path == "/ckpts/wanva"
    assert roles.uses_separate_video_teacher is True


def test_rejects_cosmos_as_video_teacher_for_latent_flowmap():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        distill_video=True,
        video_teacher_backend="cosmos_policy",
        video_teacher_model_path="/ckpts/cosmos",
    )

    with pytest.raises(
        ValueError,
        match="Cosmos Policy cannot be used as the FlowMap video teacher",
    ):
        resolve_teacher_roles(cfg)


def test_normal_wanva_config_keeps_single_teacher():
    cfg = SimpleNamespace(
        teacher_backend="wanva",
        teacher_model_path="/ckpts/wanva",
        distill_video=True,
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.action_backend == "wanva"
    assert roles.action_model_path == "/ckpts/wanva"
    assert roles.video_backend == "wanva"
    assert roles.video_model_path == "/ckpts/wanva"
    assert roles.uses_separate_video_teacher is False


def test_dual_teacher_uses_student_base_as_default_video_teacher_when_requested():
    cfg = SimpleNamespace(
        teacher_backend="cosmos_policy",
        teacher_model_path="/ckpts/cosmos",
        student_base_model_path="/ckpts/wanva",
        distill_video=True,
        use_student_base_as_video_teacher=True,
    )

    roles = resolve_teacher_roles(cfg)

    assert roles.video_backend == "wanva"
    assert roles.video_model_path == "/ckpts/wanva"
    assert roles.uses_separate_video_teacher is True


def test_flowmap_step_accessors_split_action_and_video_teachers():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "wan_va"))

    from distillation_flowmap.flowmap_step import FlowMapStepMixin

    class DummyDistiller(FlowMapStepMixin):
        pass

    distiller = DummyDistiller()
    distiller.teacher = "fsdp_action_teacher"
    distiller._teacher_nofsdp = "action_teacher"
    distiller._video_teacher_nofsdp = "video_teacher"

    assert distiller._teacher_model == "action_teacher"
    assert distiller._action_teacher_model == "action_teacher"
    assert distiller._video_teacher_model == "video_teacher"


def test_flowmap_step_video_accessor_falls_back_to_default_teacher():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "wan_va"))

    from distillation_flowmap.flowmap_step import FlowMapStepMixin

    class DummyDistiller(FlowMapStepMixin):
        pass

    distiller = DummyDistiller()
    distiller.teacher = "fsdp_teacher"
    distiller._teacher_nofsdp = "default_teacher"

    assert distiller._video_teacher_model == "default_teacher"


def test_flowmap_step_cosmos_action_x0_target_downsamples_raw_teacher_output():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "wan_va"))

    import torch
    from distillation_flowmap.flowmap_step import FlowMapStepMixin

    class Teacher:
        raw_inference_enabled = True

        def __init__(self):
            self.calls = []

        def action_target_x0(self, action_dict, raw_batch=None):
            self.calls.append((action_dict, raw_batch))
            return torch.arange(1 * 2 * 6 * 1 * 1).reshape(1, 2, 6, 1, 1).float()

    class DummyDistiller(FlowMapStepMixin):
        pass

    teacher = Teacher()
    distiller = DummyDistiller()
    distiller.is_cosmos_policy_teacher = True
    distiller.teacher = teacher
    distiller._teacher_nofsdp = teacher
    action_dict = {"latent": torch.zeros(1, 2, 6, 1, 1)}
    raw_batch = {"raw_task": ["open drawer"]}

    target = distiller._cosmos_action_x0_target(
        action_dict,
        raw_batch=raw_batch,
        downsample_factor=2,
    )

    assert torch.equal(target, torch.arange(12).reshape(1, 2, 6, 1, 1).float()[:, :, ::2])
    assert teacher.calls == [(action_dict, raw_batch)]


def test_flowmap_step_cosmos_action_x0_target_returns_none_when_raw_disabled():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "wan_va"))

    import torch
    from distillation_flowmap.flowmap_step import FlowMapStepMixin

    class Teacher:
        raw_inference_enabled = False

        def action_target_x0(self, action_dict, raw_batch=None):
            raise AssertionError("raw-disabled teacher should not be called")

    class DummyDistiller(FlowMapStepMixin):
        pass

    distiller = DummyDistiller()
    distiller.is_cosmos_policy_teacher = True
    distiller.teacher = Teacher()
    distiller._teacher_nofsdp = distiller.teacher

    assert distiller._cosmos_action_x0_target(
        {"latent": torch.zeros(1, 2, 6, 1, 1)},
        raw_batch={},
        downsample_factor=2,
    ) is None


def test_flowmap_step_cosmos_latent_opd_queries_velocity_at_student_state():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "wan_va"))

    import torch
    from distillation_flowmap.flowmap_step import FlowMapStepMixin

    class Scheduler:
        def training_target(self, latents, noise, _timesteps):
            return noise - latents

    class Teacher:
        raw_inference_enabled = True

        def __init__(self):
            self.velocity_query = None

        def predict_raw_latent_target(self, raw_batch, noise, t, r, epsilon, include_cdiff=True):
            assert raw_batch["raw_task"] == ["open drawer"]
            assert include_cdiff is False
            assert noise.shape == (1, 2, 3, 4, 4)
            assert t.shape == (1, 3)
            assert r.shape == (1, 3)
            assert bool((t >= 0.8).all())
            assert bool((t <= 80.0 / 81.0).all())
            assert bool((r >= 0.8).all())
            assert bool((r <= t).all())
            assert epsilon == 0.001
            return {
                "actions": torch.zeros(1, 16, 7),
                "cosmos_latent_x0": torch.zeros(1, 2, 3, 4, 4),
            }

        def predict_raw_latent_velocity(self, raw_batch, query_latent, t):
            assert raw_batch["raw_task"] == ["open drawer"]
            assert bool((t >= 0.8).all())
            assert bool((t <= 80.0 / 81.0).all())
            self.velocity_query = (query_latent.detach().clone(), t.detach().clone())
            return {
                "actions": torch.zeros(1, 16, 7),
                "cosmos_latent_velocity": torch.full_like(query_latent, 0.25),
            }

    class Student:
        def set_requires_gradient_sync(self, should_sync):
            self.should_sync = should_sync

    class DummyDistiller(FlowMapStepMixin):
        def convert_input_format(self, batch):
            return batch

        def sample_cosmos_latent_timestep_mixed(self, batch_size, num_frames, dtype, device):
            t_norm = torch.full((batch_size, num_frames), 0.8, dtype=dtype, device=device)
            r_norm = torch.full((batch_size, num_frames), 0.4, dtype=dtype, device=device)
            return t_norm * 1000, r_norm * 1000, t_norm, r_norm, torch.zeros(batch_size, dtype=torch.bool)

        def _apply_opd_low_noise_query_bias(self, query_t, query_r):
            return query_r

        def _prepare_base_dict(self, batch):
            return {
                "latent_dict": {
                    "text_emb": torch.zeros(batch["actions"].shape[0], 1, 2),
                },
                "action_dict": {},
                "chunk_size": 1,
                "window_size": 1,
            }

        def _student_euler_integrate(
            self,
            noisy_latents,
            timesteps,
            target_r,
            base_input_dict,
            empty_emb,
            cfg_scale,
            ref_shape,
            B,
            num_frames,
            K_steps=1,
            action_target_r=None,
            return_last_step_start=False,
        ):
            self.student_velocity = torch.full_like(noisy_latents, 0.5, requires_grad=True)
            return noisy_latents.detach(), self.student_velocity

        def _get_timestep_weight(self, timestep, weight_type):
            return torch.ones(timestep.shape[0], device=timestep.device)

    teacher = Teacher()
    distiller = DummyDistiller()
    distiller.config = SimpleNamespace(
        rank=0,
        num_train_timesteps=1000,
        cosmos_latent_channels=2,
        cosmos_latent_frames=3,
        cosmos_latent_height=4,
            cosmos_latent_width=4,
            cosmos_latent_epsilon=0.001,
            cosmos_latent_t_min=0.8,
            cosmos_latent_t_max=80.0 / 81.0,
        opd_rollout_step_pairs=[[1, 1]],
        cfg_min=1.0,
        cfg_max=1.0,
        transition_loss_type="l2",
        weight_type="uniform",
        video_transition_param="velocity",
        video_transition_weight=1.0,
        opd_endpoint_aux_weight=0.1,
        local_fm_weight=0.0,
        opd_transition_group_weight=1.0,
        opd_anchor_cap_ratio=-1.0,
        opd_aux_weight=1.0,
    )
    distiller.device = torch.device("cpu")
    distiller.gradient_accumulation_steps = 1
    distiller.distill_action = False
    distiller.action_aware = False
    distiller.empty_emb = torch.zeros(1, 1, 2)
    distiller.teacher = teacher
    distiller._teacher_nofsdp = teacher
    distiller.student = Student()
    distiller.train_scheduler_latent = Scheduler()
    batch = {
        "actions": torch.zeros(1, 1, 1, 1, 1),
        "raw_task": ["open drawer"],
    }

    result = distiller._cosmos_latent_opd_aux_transition_step(batch, batch_idx=0)

    assert result["skip_step"] is False
    assert result["teacher_steps"] == 1
    assert result["rollout_steps"] == 1
    assert teacher.velocity_query is not None
    query_latent, query_t = teacher.velocity_query
    assert query_latent.shape == (1, 2, 3, 4, 4)
    assert torch.allclose(query_t, torch.full((1, 3), 0.8))
    assert distiller.student_velocity.grad is not None


def test_full_cosmos_endpoint_focus_never_queries_raw_teacher_outside_supported_window():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "wan_va"))

    import torch
    from distillation_flowmap.flowmap_step import FlowMapStepMixin

    class Scheduler:
        def training_target(self, latents, noise, _timesteps):
            return noise - latents

    class Teacher:
        raw_inference_enabled = True

        def __init__(self):
            self.target_pairs = []
            self.velocity_timesteps = []

        def predict_raw_latent_target(self, raw_batch, noise, t, r, epsilon, include_cdiff=True):
            assert raw_batch["raw_task"] == ["open drawer"]
            assert include_cdiff is False
            assert epsilon == 0.001
            self.target_pairs.append((t.detach().clone(), r.detach().clone()))
            return {"cosmos_latent_x0": torch.zeros_like(noise)}

        def predict_raw_latent_velocity(self, raw_batch, query_latent, t):
            assert raw_batch["raw_task"] == ["open drawer"]
            self.velocity_timesteps.append(t.detach().clone())
            return {"cosmos_latent_velocity": torch.zeros_like(query_latent)}

        def predict_raw_joint_latent_velocity(
            self, raw_batch, query_latent, query_action, t
        ):
            del query_action
            result = self.predict_raw_latent_velocity(
                raw_batch, query_latent, t
            )
            return {
                **result,
                "cosmos_joint_query": query_latent,
                "cosmos_video_frame_mask": torch.ones(
                    query_latent.shape[0],
                    query_latent.shape[2],
                    dtype=torch.bool,
                    device=query_latent.device,
                ),
            }

    class DummyDistiller(FlowMapStepMixin):
        def convert_input_format(self, batch):
            return batch

        def sample_cosmos_latent_timestep_mixed(self, batch_size, num_frames, dtype, device):
            t_norm = torch.full((batch_size, num_frames), 0.9, dtype=dtype, device=device)
            r_norm = torch.zeros((batch_size, num_frames), dtype=dtype, device=device)
            return t_norm * 1000, r_norm * 1000, t_norm, r_norm, torch.zeros(batch_size, dtype=torch.bool)

        def _prepare_base_dict(self, batch):
            return {
                "latent_dict": {"text_emb": torch.zeros(batch["actions"].shape[0], 1, 2)},
                "action_dict": {},
                "chunk_size": 1,
                "window_size": 1,
            }

        def _student_euler_integrate(
            self,
            noisy_latents,
            timesteps,
            target_r,
            base_input_dict,
            empty_emb,
            cfg_scale,
            ref_shape,
            B,
            num_frames,
            K_steps=1,
            action_target_r=None,
            return_final_action=False,
            return_final_action_state=False,
        ):
            self.student_velocity = torch.zeros_like(noisy_latents, requires_grad=True)
            return noisy_latents.detach(), self.student_velocity

    teacher = Teacher()
    distiller = DummyDistiller()
    distiller.config = SimpleNamespace(
        rank=0,
        num_train_timesteps=1000,
        cosmos_latent_channels=2,
        cosmos_latent_frames=3,
        cosmos_latent_height=4,
        cosmos_latent_width=4,
        cosmos_latent_epsilon=0.001,
        cosmos_latent_t_min=0.8,
        cosmos_latent_t_max=80.0 / 81.0,
        cosmos_use_teacher_action_anchor=False,
        opd_endpoint_focus_prob=1.0,
        opd_rollout_step_pairs=[[1, 1]],
        opd_joint_action_rollout=False,
        opd_danceopd_action_endpoint_weight=0.0,
        opd_danceopd_endpoint_weight=1.0,
        opd_danceopd_velocity_weight=0.0,
        opd_aux_weight=1.0,
        cfg_min=1.0,
        cfg_max=1.0,
        opd_cosmos_spatial_crop_size=0,
    )
    distiller.device = torch.device("cpu")
    distiller.distill_action = False
    distiller.action_aware = False
    distiller.empty_emb = torch.zeros(1, 1, 2)
    distiller.teacher = teacher
    distiller._teacher_nofsdp = teacher
    distiller.train_scheduler_latent = Scheduler()
    batch = {
        "actions": torch.zeros(1, 1, 1, 1, 1),
        "raw_task": ["open drawer"],
    }

    result = distiller._cosmos_latent_full_opd_aux_transition_step(batch, batch_idx=0)

    assert result["skip_step"] is False
    assert teacher.target_pairs
    assert not teacher.velocity_timesteps
    for t, r in teacher.target_pairs:
        assert bool((t >= 0.8).all())
        assert bool((t <= 80.0 / 81.0).all())
        assert bool((r >= 0.8).all())
        assert bool((r <= t).all())
    for t in teacher.velocity_timesteps:
        assert bool((t >= 0.8).all())
        assert bool((t <= 80.0 / 81.0).all())
    assert distiller.student_velocity.grad is not None
