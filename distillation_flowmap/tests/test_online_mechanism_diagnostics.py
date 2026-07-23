import ast
import json
import logging
import os
import sys
import time
import types
from pathlib import Path
from unittest import mock

import torch

from distillation_flowmap.mechanism_diagnostics import (
    diagnostic_due,
    means_from_reduced_stats,
)


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


def _load_trainer_harness():
    """Compile the scoped trainer methods without importing its GPU stack."""
    trainer_path = os.path.join(FLOWMAP_DIR, "flowmap_trainer.py")
    tree = ast.parse(Path(trainer_path).read_text())
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "FlowMapDistiller"
    )
    wanted = {
        "_get_mechanism_diagnostic_batches",
        "_run_mechanism_diagnostics",
        "_maybe_run_mechanism_diagnostics",
    }
    methods = [
        node
        for node in trainer.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    harness_ast = ast.Module(
        body=[
            ast.ClassDef(
                name="_TrainerHarness",
                bases=[],
                keywords=[],
                body=methods or [ast.Pass()],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(harness_ast)
    namespace = {
        "HAS_WANDB": False,
        "Path": Path,
        "diagnostic_due": diagnostic_due,
        "dist": torch.distributed,
        "json": json,
        "means_from_reduced_stats": means_from_reduced_stats,
        "time": time,
        "torch": torch,
        "wandb": None,
    }
    exec(compile(harness_ast, trainer_path, "exec"), namespace)
    return namespace["_TrainerHarness"]


def _trainer_host(tmp_path, *, rank=0, enabled=True):
    cls = _load_trainer_harness()
    host = cls()
    host.config = types.SimpleNamespace(
        mechanism_diagnostics=enabled,
        mechanism_diagnostic_interval=100,
        mechanism_diagnostic_seed=42,
        mechanism_diagnostic_r=500,
        mechanism_diagnostic_s=250,
        mechanism_diagnostic_teacher_steps=8,
        mechanism_diagnostic_num_batches=1,
        output_dir=str(tmp_path),
        enable_wandb=False,
        rank=rank,
    )
    host.device = torch.device("cpu")
    host.step = 0
    host._mechanism_pending = False
    host._last_mechanism_diagnostic_step = None
    host._mechanism_diagnostic_batches = None
    host.tb_writer = None
    return host


def test_fixed_diagnostic_batches_are_cached_without_consuming_train_iterator(
    tmp_path,
):
    host = _trainer_host(tmp_path, rank=3)
    host.config.mechanism_diagnostic_num_batches = 2

    class Dataset:
        def __init__(self):
            self.requested = []

        def __len__(self):
            return 11

        def __getitem__(self, index):
            self.requested.append(index)
            return {"latents": torch.tensor([float(index)])}

    dataset = Dataset()
    host.train_loader = types.SimpleNamespace(dataset=dataset)

    class ForbiddenIterator:
        def __next__(self):
            raise AssertionError("diagnostics consumed the train iterator")

    host.train_loader_iter = ForbiddenIterator()
    batches = host._get_mechanism_diagnostic_batches()
    again = host._get_mechanism_diagnostic_batches()

    assert batches is again
    assert dataset.requested == [1, 2]  # (seed 42 + rank 3 + offset) % 11
    assert [batch["_mechanism_diagnostic_batch_index"] for batch in batches] == [
        0,
        1,
    ]
    assert all(batch["latents"].shape[0] == 1 for batch in batches)


def test_diagnostic_schedule_disabled_boundaries_and_duplicate_steps(tmp_path):
    host = _trainer_host(tmp_path, enabled=False)
    calls = []
    host._run_mechanism_diagnostics = lambda: calls.append(host.step) or {}

    for step in (0, 99, 100):
        host.step = step
        host._maybe_run_mechanism_diagnostics(skipped_optimizer_step=False)
    assert calls == []

    host.config.mechanism_diagnostics = True
    host._mechanism_pending = False
    for step in (99, 100, 100, 101, 200, 200, 201):
        host.step = step
        host._maybe_run_mechanism_diagnostics(skipped_optimizer_step=False)
    assert calls == [100, 200]


def test_skipped_boundary_remains_pending_until_next_success(tmp_path):
    host = _trainer_host(tmp_path)
    calls = []
    host._run_mechanism_diagnostics = lambda: calls.append(host.step) or {}

    host.step = 100
    host._maybe_run_mechanism_diagnostics(skipped_optimizer_step=True)
    assert host._mechanism_pending
    assert calls == []

    host.step = 101
    host._maybe_run_mechanism_diagnostics(skipped_optimizer_step=False)
    host._maybe_run_mechanism_diagnostics(skipped_optimizer_step=False)
    assert calls == [101]
    assert not host._mechanism_pending


def test_runner_uses_one_packed_sum_reduction_and_global_counts(tmp_path):
    host = _trainer_host(tmp_path)
    host._mechanism_diagnostic_batches = [{"latents": torch.zeros(1)}]
    host._compute_mechanism_diagnostic_stats = lambda batch, diagnostic_index: {
        "mechanism/g_anchor_sum": torch.tensor(10.0),
        "mechanism/g_anchor_count": torch.tensor(2.0),
        "diagnostic_valid": torch.tensor(1.0),
    }
    reduce_calls = []

    def fake_all_reduce(packed, op):
        reduce_calls.append((packed.clone(), op))
        # Rank 1 has mean 10 from three samples, and one invalid diagnostic.
        packed.add_(torch.tensor([0.0, 3.0, 30.0]))

    with (
        mock.patch.object(torch.distributed, "is_initialized", return_value=True),
        mock.patch.object(torch.distributed, "get_world_size", return_value=2),
        mock.patch.object(torch.distributed, "all_reduce", side_effect=fake_all_reduce),
    ):
        result = host._run_mechanism_diagnostics()

    assert len(reduce_calls) == 1
    assert reduce_calls[0][1] == torch.distributed.ReduceOp.SUM
    assert result["mechanism/g_anchor"] == 8.0
    record = json.loads(
        (tmp_path / "diagnostics" / "mechanism_metrics.jsonl").read_text()
    )
    assert record["valid_count"] == 1.0
    assert record["diagnostic_valid"] is False


def test_rank_zero_logs_jsonl_wandb_tensorboard_and_preserves_state(tmp_path):
    host = _trainer_host(tmp_path)
    host.step = 100
    host.student = torch.nn.Linear(1, 1)
    host.student.train()
    host._mechanism_diagnostic_batches = [{"latents": torch.zeros(1)}]
    before_rng = torch.random.get_rng_state().clone()

    def compute(batch, diagnostic_index):
        del batch
        assert diagnostic_index == 1
        assert not host.student.training
        torch.rand(3)
        return {
            "mechanism/g_anchor_sum": torch.tensor(12.0),
            "mechanism/g_anchor_count": torch.tensor(3.0),
            "diagnostic_valid": torch.tensor(1.0),
        }

    host._compute_mechanism_diagnostic_stats = compute
    tb_calls = []
    host.tb_writer = types.SimpleNamespace(
        add_scalar=lambda key, value, step: tb_calls.append((key, value, step)),
        flush=mock.Mock(),
    )
    wandb_log = mock.Mock()
    globals_dict = host._maybe_run_mechanism_diagnostics.__globals__
    globals_dict["HAS_WANDB"] = True
    globals_dict["wandb"] = types.SimpleNamespace(log=wandb_log)
    host.config.enable_wandb = True

    result = host._maybe_run_mechanism_diagnostics(
        skipped_optimizer_step=False
    )

    assert result == {"mechanism/g_anchor": 4.0}
    wandb_log.assert_called_once_with(result, step=100)
    assert tb_calls == [("mechanism/g_anchor", 4.0, 100)]
    host.tb_writer.flush.assert_called_once_with()
    assert host.student.training
    assert torch.equal(before_rng, torch.random.get_rng_state())

    record = json.loads(
        (tmp_path / "diagnostics" / "mechanism_metrics.jsonl").read_text()
    )
    assert record["step"] == 100
    assert record["seed"] == 42
    assert record["diagnostic_index"] == 1
    assert record["r"] == 500
    assert record["s"] == 250
    assert record["teacher_steps"] == 8
    assert record["valid_count"] == 1.0
    assert record["diagnostic_valid"] is True
    assert record["mechanism/g_anchor"] == 4.0
    assert record["elapsed_seconds"] >= 0.0


def test_nonzero_rank_does_not_log_or_create_diagnostics_directory(tmp_path):
    host = _trainer_host(tmp_path, rank=1)
    host.step = 100
    host._mechanism_diagnostic_batches = [{"latents": torch.zeros(1)}]
    host._compute_mechanism_diagnostic_stats = lambda batch, diagnostic_index: {
        "mechanism/g_anchor_sum": torch.tensor(2.0),
        "mechanism/g_anchor_count": torch.tensor(1.0),
        "diagnostic_valid": torch.tensor(1.0),
    }

    result = host._maybe_run_mechanism_diagnostics(
        skipped_optimizer_step=False
    )

    assert result == {}
    assert not (tmp_path / "diagnostics").exists()


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
