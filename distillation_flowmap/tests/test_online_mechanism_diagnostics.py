import logging
import os
import sys
import types

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FLOWMAP_DIR = os.path.join(REPO_ROOT, "distillation_flowmap")
for path in (REPO_ROOT, FLOWMAP_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


class _Scheduler:
    def add_noise(self, clean, noise, timesteps, t_dim=2):
        del clean, timesteps, t_dim
        return noise


class _DiagnosticHarness:
    def __init__(self, mixin):
        self.config = types.SimpleNamespace(
            mechanism_diagnostic_seed=42,
            mechanism_diagnostic_r=500,
            mechanism_diagnostic_s=250,
            mechanism_diagnostic_teacher_steps=8,
            num_train_timesteps=1000,
            rank=3,
            action_downsample_factor=1,
            cfg_min=2.0,
            cfg_max=4.0,
        )
        self.device = torch.device("cpu")
        self.train_scheduler_latent = _Scheduler()
        self.train_scheduler_action = _Scheduler()
        self.parameter = torch.nn.Parameter(torch.tensor(2.0))
        self.student_calls = []
        self.teacher_calls = []
        self.mixin = mixin

    def _prepare_mechanism_diagnostic_context(self, batch):
        return {
            "action_mask": batch.get("action_mask"),
            "batch_size": batch["latents"].shape[0],
        }

    def _diagnostic_student_joint_map(
        self,
        video_x,
        action_x,
        video_t,
        action_t,
        video_r,
        action_r,
        *,
        context,
    ):
        del context
        torch.rand(1)
        self.student_calls.append(
            {
                "video": video_x.clone(),
                "action": action_x.clone(),
                "from": float(video_t[0, 0]),
                "to": float(video_r[0, 0]),
            }
        )
        scale = (video_t[:, None, :, None, None] - video_r[:, None, :, None, None]) / 1000
        action_scale = (
            action_t[:, None, :, None, None]
            - action_r[:, None, :, None, None]
        ) / 1000
        video_velocity = torch.ones_like(video_x) * self.parameter
        action_velocity = torch.ones_like(action_x) * (self.parameter + 1)
        return (
            video_x - scale * video_velocity,
            action_x - action_scale * action_velocity,
            video_velocity,
            action_velocity,
        )

    def _diagnostic_teacher_joint_rollout(
        self,
        video_x,
        action_x,
        video_t,
        action_t,
        video_r,
        action_r,
        *,
        num_steps,
        context,
    ):
        del context
        torch.rand(1)
        self.teacher_calls.append(
            {
                "video": video_x.clone(),
                "action": action_x.clone(),
                "from": float(video_t[0, 0]),
                "to": float(video_r[0, 0]),
                "steps": num_steps,
            }
        )
        scale = (video_t[:, None, :, None, None] - video_r[:, None, :, None, None]) / 1000
        action_scale = (
            action_t[:, None, :, None, None]
            - action_r[:, None, :, None, None]
        ) / 1000
        return video_x - 4 * scale, action_x - 5 * action_scale

    def _diagnostic_joint_fields(
        self, video_x, action_x, video_t, action_t, *, context
    ):
        del action_x, video_t, action_t, context
        return torch.ones_like(video_x) * self.parameter, torch.ones_like(video_x) * 4


def _load_mixin():
    old_modules = sys.modules.get("modules")
    old_modules_model = sys.modules.get("modules.model")
    old_utils = sys.modules.get("utils")
    modules_stub = types.ModuleType("modules")
    modules_model_stub = types.ModuleType("modules.model")
    modules_model_stub.FlexAttnFunc = type("_FlexAttnFunc", (), {})
    sys.modules["modules"] = modules_stub
    sys.modules["modules.model"] = modules_model_stub
    utils_stub = types.ModuleType("utils")
    utils_stub.data_seq_to_patch = lambda *args, **kwargs: None
    utils_stub.logger = logging.getLogger("test-online-mechanism-diagnostics")
    sys.modules["utils"] = utils_stub
    sys.modules.pop("distillation_flowmap.flowmap_step", None)
    from distillation_flowmap.flowmap_step import FlowMapStepMixin

    if old_modules is None:
        sys.modules.pop("modules", None)
    else:
        sys.modules["modules"] = old_modules
    if old_modules_model is None:
        sys.modules.pop("modules.model", None)
    else:
        sys.modules["modules.model"] = old_modules_model
    if old_utils is None:
        sys.modules.pop("utils", None)
    else:
        sys.modules["utils"] = old_utils
    return FlowMapStepMixin


def _host():
    mixin = _load_mixin()
    host = _DiagnosticHarness(mixin)
    for name in (
        "_compute_mechanism_diagnostic_stats",
        "_compute_mechanism_diagnostic_stats_impl",
    ):
        setattr(host, name, types.MethodType(getattr(mixin, name), host))
    return host


def _batch():
    return {
        "latents": torch.zeros(2, 1, 2, 1, 1),
        "actions": torch.zeros(2, 1, 2, 1, 1),
        "action_mask": torch.ones(2, 1, 2, 1, 1),
    }


def test_diagnostic_prior_is_repeatable_and_does_not_change_global_rng():
    batch = _batch()
    before = torch.random.get_rng_state().clone()
    first = _host()
    first._compute_mechanism_diagnostic_stats(batch, diagnostic_index=7)
    after = torch.random.get_rng_state()

    second = _host()
    second._compute_mechanism_diagnostic_stats(batch, diagnostic_index=7)

    assert torch.equal(before, after)
    assert torch.equal(
        first.student_calls[0]["video"], second.student_calls[0]["video"]
    )
    assert torch.equal(
        first.student_calls[0]["action"], second.student_calls[0]["action"]
    )


def test_action_interventions_replace_only_the_requested_joint_state():
    host = _host()
    host._compute_mechanism_diagnostic_stats(_batch(), diagnostic_index=0)

    student_context, teacher_video_context, teacher_joint_context = host.student_calls[-3:]
    assert torch.equal(student_context["action"], teacher_video_context["action"])
    assert not torch.equal(student_context["video"], teacher_video_context["video"])
    assert torch.equal(
        teacher_video_context["video"], teacher_joint_context["video"]
    )
    assert not torch.equal(
        teacher_video_context["action"], teacher_joint_context["action"]
    )


def test_routes_are_faithful_and_statistics_are_no_grad():
    host = _host()
    stats = host._compute_mechanism_diagnostic_stats(_batch(), diagnostic_index=2)

    student_routes = [(call["from"], call["to"]) for call in host.student_calls]
    teacher_routes = [
        (call["from"], call["to"], call["steps"]) for call in host.teacher_calls
    ]
    assert (1000.0, 500.0) in student_routes
    assert (500.0, 0.0) in student_routes
    assert (500.0, 250.0) in student_routes
    assert (250.0, 0.0) in student_routes
    assert (1000.0, 500.0, 8) in teacher_routes
    assert (500.0, 0.0, 8) in teacher_routes
    assert all(not value.requires_grad for value in stats.values())
    assert host.parameter.grad is None
