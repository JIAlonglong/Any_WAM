from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _path in (
    os.path.join(REPO_ROOT, "distillation_flowmap"),
    os.path.join(REPO_ROOT, "wan_va"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from distillation_flowmap.cosmos_policy_adapter import unpack_flowmap_action_query
from distillation_flowmap.danceopd_query import build_shifted_terminal_path
from distillation_flowmap.flowmap_step import FlowMapStepMixin
from distillation_flowmap.mechanism_diagnostics import (
    capture_diagnostic_snapshot_if_due,
    diagnostic_runtime,
    compute_mechanism_metric_samples,
    fatal_mechanism_process_exit,
    materialize_diagnostic_snapshot,
    means_from_reduced_stats,
    pack_finite_metric_stats,
)


class _BuilderHarness(FlowMapStepMixin):
    patch_size = (1, 1, 1)
    device = torch.device("cpu")

    def __init__(self):
        self.config = SimpleNamespace(
            num_train_timesteps=1000,
            attn_mode="flex",
        )
        self.student = object()
        self._student_blocks_compiled = False
        self.captured = []

    def _student_joint_forward(
        self,
        model,
        joint_input,
        model_empty_emb,
        video_r_t,
        action_r_t,
        *,
        cfg_scale,
        batch_size,
        ref_shape,
        require_action,
    ):
        del model, model_empty_emb, video_r_t, action_r_t
        del cfg_scale, batch_size, ref_shape
        self.captured.append(
            (
                joint_input["latent_dict"]["noisy_latents"].clone(),
                joint_input["action_dict"]["noisy_latents"].clone(),
                torch.is_grad_enabled(),
            )
        )
        video_velocity = torch.zeros_like(
            joint_input["latent_dict"]["noisy_latents"]
        )
        if not require_action:
            return video_velocity
        action = joint_input["action_dict"]["noisy_latents"]
        action_tokens = action.squeeze(-1).permute(0, 2, 3, 1).flatten(1, 2)
        return video_velocity, action_tokens


def _joint_context():
    return {
        "video_base": {
            "latent": torch.zeros(1, 1, 1, 1, 1),
            "text_emb": torch.zeros(1, 1, 1),
        },
        "action_latent": torch.zeros(1, 1, 1, 1, 1),
        "action_cond_t": torch.zeros(1, 1),
        "action_text": torch.zeros(1, 1, 1),
        "action_grid": torch.arange(4).reshape(1, 4, 1),
        "action_mask": torch.ones(1, 1, 1, dtype=torch.bool),
        "chunk_size": 1,
        "window_size": 1,
        "student_model": object(),
        "empty_emb": torch.zeros(1, 1, 1),
        "cfg_scale": 1.0,
        "batch_size": 1,
        "ref_shape": (1, 1, 1, 1, 1),
        "action_frames": 1,
    }


def test_shared_cosmos_joint_builder_replaces_video_and_holds_action_state():
    harness = _BuilderHarness()
    context = _joint_context()
    common_action = torch.full((1, 1, 1, 1, 1), 7.0)
    video_t = torch.zeros(1, 1)
    action_t = torch.full((1, 1), 500.0)

    for value in (1.0, 2.0, 3.0):
        joint = harness._build_joint_input(
            torch.full((1, 1, 1, 1, 1), value),
            video_t,
            common_action,
            action_t,
            video_base=context["video_base"],
            action_latent=context["action_latent"],
            action_cond_t=context["action_cond_t"],
            action_text=context["action_text"],
            action_grid=context["action_grid"],
            action_valid_mask=context["action_mask"],
            chunk_size=context["chunk_size"],
            window_size=context["window_size"],
        )
        harness._student_joint_forward(
            context["student_model"],
            joint,
            context["empty_emb"],
            video_t,
            torch.zeros_like(action_t),
            cfg_scale=1.0,
            batch_size=1,
            ref_shape=context["ref_shape"],
            require_action=True,
        )

    assert [capture[0].item() for capture in harness.captured] == [1.0, 2.0, 3.0]
    assert all(torch.equal(capture[1], common_action) for capture in harness.captured)
    assert joint["action_dict"]["grid_id"] is context["action_grid"]
    assert joint["action_dict"]["actions_mask"] is context["action_mask"]


def test_mechanism_preflight_rejects_teacher_without_joint_continuation():
    harness = _BuilderHarness()

    class LegacyTeacher:
        raw_inference_enabled = True

        @staticmethod
        def predict_raw_latent_target(*args, **kwargs):
            raise AssertionError("preflight must not run inference")

        @staticmethod
        def predict_raw_joint_latent_velocity(*args, **kwargs):
            raise AssertionError("preflight must not run inference")

        @staticmethod
        def predict_raw_same_prior_endpoint(*args, **kwargs):
            raise AssertionError("preflight must not run inference")

    harness.teacher = LegacyTeacher()
    batch = {
        "latents": torch.zeros(1, 1, 1),
        "actions": torch.zeros(1, 1, 1),
    }
    with pytest.raises(RuntimeError, match="continuation"):
        harness._validate_cosmos_mechanism_preflight(
            batch, teacher_steps=8, student_steps=2
        )
    with pytest.raises(ValueError, match="2 or 4"):
        harness._validate_cosmos_mechanism_preflight(
            batch, teacher_steps=8, student_steps=1
        )


def test_context_action_routes_use_actual_replacement_before_no_grad_forward():
    harness = _BuilderHarness()
    context = _joint_context()
    videos = {
        "gt": torch.full((1, 1, 1, 1, 1), 11.0),
        "student": torch.full((1, 1, 1, 1, 1), 22.0),
        "teacher_video": torch.full((1, 1, 1, 1, 1), 33.0),
    }
    common_action = torch.full((1, 1, 1, 1, 1), 5.0)

    predictions = harness._cosmos_action_context_predictions(
        videos,
        common_action=common_action,
        action_t=torch.full((1, 1), 500.0),
        context=context,
    )

    assert set(predictions) == set(videos)
    assert [capture[0].item() for capture in harness.captured] == [11.0, 22.0, 33.0]
    assert all(torch.equal(capture[1], common_action) for capture in harness.captured)
    assert all(capture[2] is False for capture in harness.captured)
    assert all(not value.requires_grad for value in predictions.values())


def test_g_metrics_are_route_geometry_and_not_opd_weight_aliases():
    common = dict(
        teacher_continuation_video=torch.tensor([[[2.0, 4.0]]]),
        same_prior_teacher_endpoint_video=torch.tensor([[[1.0, 1.0]]]),
        direct_route_video=torch.tensor([[[5.0, 7.0]]]),
        composed_route_video=torch.tensor([[[4.0, 5.0]]]),
        student_field_video=torch.tensor([[[9.0, 11.0]]]),
        teacher_field_video=torch.tensor([[[8.0, 8.0]]]),
        video_frame_mask=torch.tensor([[True, False]]),
        teacher_endpoint_action=torch.zeros(1, 1, 1, 1, 1),
        action_student_context=torch.ones(1, 1, 1, 1, 1),
        action_teacher_video_context=torch.full((1, 1, 1, 1, 1), 0.5),
        action_teacher_joint_context=None,
        action_mask=None,
        teacher_joint_available=False,
    )
    samples = compute_mechanism_metric_samples(**common)

    # Every video metric honors the worker frame mask, not only field error.
    assert samples["mechanism/g_anchor"].item() == 1.0
    assert samples["mechanism/g_anchor_mse"].item() == 1.0
    assert samples["mechanism/g_comp"].item() == 1.0
    assert samples["mechanism/g_comp_mse"].item() == 1.0
    assert samples["mechanism/video_endpoint_error"].item() == 16.0
    # The mask keeps only the first temporal element: (9 - 8)^2.
    assert samples["mechanism/video_field_match_error"].item() == 1.0

    stats = pack_finite_metric_stats(samples)
    assert stats["mechanism/action_error_teacher_joint_context_count"].item() == 0
    assert stats["mechanism/video_to_action_full_joint_gain_count"].item() == 0
    assert stats["mechanism/video_to_action_residual_action_gap_count"].item() == 0
    assert stats["mechanism/teacher_joint_available_sum"].item() == 0
    assert stats["mechanism/teacher_joint_available_count"].item() == 1
    assert means_from_reduced_stats(stats)["mechanism/diagnostic_valid"] == 1.0


def test_all_video_metrics_ignore_worker_excluded_frames():
    mask = torch.tensor([[True, False]])
    common = dict(
        teacher_continuation_video=torch.tensor([[[[2.0]], [[9999.0]]]]).permute(0, 2, 1, 3),
        same_prior_teacher_endpoint_video=torch.tensor([[[[1.0]], [[-9999.0]]]]).permute(0, 2, 1, 3),
        direct_route_video=torch.tensor([[[[3.0]], [[7777.0]]]]).permute(0, 2, 1, 3),
        composed_route_video=torch.tensor([[[[1.0]], [[-7777.0]]]]).permute(0, 2, 1, 3),
        student_field_video=torch.tensor([[[[4.0]], [[5555.0]]]]).permute(0, 2, 1, 3),
        teacher_field_video=torch.tensor([[[[1.0]], [[-5555.0]]]]).permute(0, 2, 1, 3),
        video_frame_mask=mask,
        teacher_endpoint_action=torch.zeros(1, 1, 1, 1, 1),
        action_gt_context=torch.zeros(1, 1, 1, 1, 1),
        action_student_context=torch.zeros(1, 1, 1, 1, 1),
        action_teacher_video_context=torch.zeros(1, 1, 1, 1, 1),
        action_teacher_joint_context=None,
        action_mask=None,
        teacher_joint_available=False,
    )
    samples = compute_mechanism_metric_samples(**common)
    assert samples["mechanism/g_anchor"].item() == 1.0
    assert samples["mechanism/g_anchor_mse"].item() == 1.0
    assert samples["mechanism/g_comp"].item() == 4.0
    assert samples["mechanism/g_comp_mse"].item() == 4.0
    assert samples["mechanism/video_endpoint_error"].item() == 4.0
    assert samples["mechanism/video_field_match_error"].item() == 9.0


def test_calibrated_noisy_state_uses_teacher_endpoint_not_dataset_gt():
    harness = _BuilderHarness()
    dataset_gt = torch.full((1, 1, 1, 1, 1), -100.0)
    teacher_x0 = torch.full_like(dataset_gt, 4.0)
    noise = torch.full_like(dataset_gt, 10.0)
    state = harness._cosmos_calibrated_noisy_state(
        teacher_x0=teacher_x0, noise=noise, normalized_t=0.8
    )
    assert torch.allclose(state, torch.full_like(state, 8.8))
    assert not torch.equal(state, (1 - 0.8) * dataset_gt + 0.8 * noise)


def test_diagnostic_snapshot_is_due_only_and_survives_training_mutation():
    from distillation_flowmap.mechanism_diagnostics import MechanismDiagnosticScheduler

    scheduler = MechanismDiagnosticScheduler(interval=100)
    original = {
        "latents": torch.tensor([1.0], device="cpu"),
        "actions": torch.tensor([2.0], device="cpu"),
        "nested": {"mask": torch.tensor([True])},
        "task": ["pick"],
    }
    assert capture_diagnostic_snapshot_if_due(
        scheduler, original, completed_step=99
    ) is None
    snapshot = capture_diagnostic_snapshot_if_due(
        scheduler, original, completed_step=100
    )
    assert snapshot is not None
    original["latents"].fill_(101)
    original["actions"].fill_(202)
    original["nested"]["mask"].fill_(False)
    assert snapshot["latents"].item() == 1
    assert snapshot["actions"].item() == 2
    assert snapshot["nested"]["mask"].item() is True
    assert snapshot["latents"].device.type == "cpu"


def test_mutated_training_batch_cannot_change_gt_action_context_forward():
    from distillation_flowmap.mechanism_diagnostics import MechanismDiagnosticScheduler

    harness = _BuilderHarness()
    scheduler = MechanismDiagnosticScheduler(interval=1)
    training_batch = {
        "latents": torch.full((1, 1, 1, 1, 1), 3.0),
        "actions": torch.zeros(1, 1, 1, 1, 1),
    }
    snapshot = capture_diagnostic_snapshot_if_due(
        scheduler, training_batch, completed_step=1
    )
    # Mirror the in-place teacher-target replacement in the main Cosmos step.
    training_batch["latents"].fill_(99.0)
    training_batch["actions"].fill_(88.0)
    diagnostic_batch = materialize_diagnostic_snapshot(
        snapshot, device=torch.device("cpu")
    )
    harness._cosmos_action_context_predictions(
        {"gt": diagnostic_batch["latents"]},
        common_action=diagnostic_batch["actions"],
        action_t=torch.full((1, 1), 500.0),
        context=_joint_context(),
    )
    assert harness.captured[0][0].item() == 3.0
    assert harness.captured[0][1].item() == 0.0


def test_materialize_moves_model_tensors_but_keeps_raw_teacher_payload_on_cpu():
    snapshot = {
        "latents": torch.ones(1),
        "actions": torch.ones(1),
        "raw_images": torch.ones(1),
        "raw_observation": {
            "pixels": torch.ones(1),
            "nested": [torch.ones(1)],
        },
    }
    materialized = materialize_diagnostic_snapshot(
        snapshot, device=torch.device("meta")
    )
    assert materialized["latents"].device.type == "meta"
    assert materialized["actions"].device.type == "meta"
    assert materialized["raw_images"].device.type == "cpu"
    assert materialized["raw_observation"]["pixels"].device.type == "cpu"
    assert materialized["raw_observation"]["nested"][0].device.type == "cpu"

    class RawTeacher:
        def predict(self, batch):
            assert batch["raw_images"].device.type == "cpu"
            assert batch["raw_observation"]["pixels"].device.type == "cpu"
            return True

    assert RawTeacher().predict(materialized) is True


def test_three_context_action_errors_and_finite_counts_reach_log_mapping():
    samples = compute_mechanism_metric_samples(
        teacher_continuation_video=torch.zeros(1, 1, 1),
        same_prior_teacher_endpoint_video=torch.zeros(1, 1, 1),
        direct_route_video=torch.zeros(1, 1, 1),
        composed_route_video=torch.zeros(1, 1, 1),
        teacher_endpoint_action=torch.zeros(1, 1, 1, 1, 1),
        action_gt_context=torch.full((1, 1, 1, 1, 1), 0.25),
        action_student_context=torch.ones(1, 1, 1, 1, 1),
        action_teacher_video_context=torch.full((1, 1, 1, 1, 1), 0.5),
        action_teacher_joint_context=None,
        action_mask=None,
        teacher_joint_available=False,
    )
    logged = means_from_reduced_stats(pack_finite_metric_stats(samples))
    assert logged["mechanism/action_error_gt_context"] == pytest.approx(0.0625)
    assert logged["mechanism/action_error_student_context"] == pytest.approx(1.0)
    assert logged["mechanism/action_error_teacher_video_context"] == pytest.approx(0.25)
    assert logged["mechanism/action_error_gt_context_finite_count"] == 1.0
    assert logged["mechanism/action_error_student_context_finite_count"] == 1.0
    assert logged["mechanism/action_error_teacher_joint_context_finite_count"] == 0.0
    assert logged["mechanism/action_error_teacher_joint_context_available"] == 0.0


def test_force_no_grad_routes_stay_eval_and_restore_modes(monkeypatch):
    import types

    class FakeFlexAttn:
        attention_mask = None
        cross_attention_mask = None

        @staticmethod
        def init_mask(*args, **kwargs):
            return None

    fake_model_module = types.ModuleType("modules.model")
    fake_model_module.FlexAttnFunc = FakeFlexAttn
    monkeypatch.setitem(sys.modules, "modules.model", fake_model_module)

    class ModeModel(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name
            self.anchor = torch.nn.Parameter(torch.zeros(()))

    class Harness(FlowMapStepMixin):
        patch_size = (1, 1, 1)
        device = torch.device("cpu")
        distill_action = True
        action_aware = False

        def __init__(self):
            self.config = SimpleNamespace(
                action_downsample_factor=1,
                opd_joint_action_rollout=True,
                opd_rollout_grad_mode="endpoint",
                opd_rollout_grad_steps=1,
                offline_eval_force_gradient_checkpointing=False,
                offline_eval_force_cfg=False,
                num_train_timesteps=1000,
                attn_mode="flex",
            )
            self.student = ModeModel("student")
            self._student_nofsdp = ModeModel("nofsdp")
            self._student_blocks_compiled = False
            self._nofsdp_synced = True
            self.mode_observations = []

        def _timestep_to_sigma_5d(self, timesteps):
            return timesteps[:, None, :, None, None] / 1000

        def _student_cfg_forward(
            self,
            model,
            input_dict,
            empty_emb,
            cfg_scale,
            B,
            ref_shape,
            r_timestep,
            action_r_timestep,
            force_cfg=False,
            return_action=False,
        ):
            del empty_emb, cfg_scale, r_timestep, action_r_timestep, force_cfg
            self.mode_observations.append((model.name, model.training))
            video = torch.zeros(
                ref_shape,
                dtype=input_dict["latent_dict"]["noisy_latents"].dtype,
            )
            if not return_action:
                return video
            action = input_dict["action_dict"]["noisy_latents"]
            tokens = action.squeeze(-1).permute(0, 2, 3, 1).flatten(1, 2)
            return video, torch.zeros_like(tokens)

    harness = Harness()
    video = torch.zeros(1, 1, 1, 1, 1)
    action = torch.zeros(1, 1, 1, 1, 1)
    video_t = torch.full((1, 1), 1000.0)
    action_t = torch.full((1, 1), 1000.0)
    zero_t = torch.zeros_like(video_t)
    base = {
        "latent_dict": {
            "latent": video,
            "noisy_latents": video,
            "timesteps": video_t,
            "text_emb": torch.zeros(1, 1, 1),
        },
        "action_dict": {
            "latent": action,
            "noisy_latents": action,
            "timesteps": action_t,
            "cond_timesteps": torch.zeros(1, 1),
            "text_emb": torch.zeros(1, 1, 1),
        },
        "chunk_size": 1,
        "window_size": 1,
    }
    context = {
        "video_base": base["latent_dict"],
        "action_latent": action,
        "action_cond_t": torch.zeros(1, 1),
        "action_text": torch.zeros(1, 1, 1),
        "action_grid": None,
        "action_mask": None,
        "chunk_size": 1,
        "window_size": 1,
        "student_model": harness.student,
        "empty_emb": torch.zeros(1, 1, 1),
        "cfg_scale": 1.0,
        "batch_size": 1,
        "ref_shape": tuple(video.shape),
        "action_frames": 1,
    }

    assert harness.student.training and harness._student_nofsdp.training
    with diagnostic_runtime(
        seed=42, models=(harness.student, harness._student_nofsdp)
    ):
        harness._student_euler_integrate(
            noisy_latents=video,
            timesteps=video_t,
            target_r=zero_t,
            base_input_dict=base,
            empty_emb=torch.zeros(1, 1, 1),
            cfg_scale=1.0,
            ref_shape=tuple(video.shape),
            B=1,
            num_frames=1,
            K_steps=1,
            action_target_r=zero_t,
            return_final_action=True,
            return_final_action_state=True,
            force_no_grad=True,
        )
        harness._cosmos_student_joint_map(
            video,
            action,
            video_t,
            action_t,
            zero_t,
            zero_t,
            context=context,
        )
        harness._cosmos_action_context_predictions(
            {"gt": video},
            common_action=action,
            action_t=action_t,
            context=context,
        )
    assert harness.mode_observations
    assert all(training is False for _, training in harness.mode_observations)
    assert harness.student.training and harness._student_nofsdp.training


def test_torchrun_supervisor_terminates_peer_on_asymmetric_probe_failure(tmp_path):
    worker = tmp_path / "mechanism_failure_worker.py"
    marker = tmp_path / "rank1_reached_collective"
    worker.write_text(
        "\n".join(
            [
                "import os, time",
                "import torch.distributed as dist",
                "from distillation_flowmap.mechanism_diagnostics import fatal_mechanism_process_exit",
                "dist.init_process_group('gloo')",
                "rank = dist.get_rank()",
                "if rank == 0:",
                "    fatal_mechanism_process_exit(RuntimeError('asymmetric raw worker failure'), distributed=True)",
                "time.sleep(60)",
                f"open({str(marker)!r}, 'w').write('reached')",
                "dist.barrier()",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT
    started = time.monotonic()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(worker),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    elapsed = time.monotonic() - started
    assert result.returncode != 0
    assert elapsed < 20
    assert "MECHANISM_FATAL_EXIT" in result.stderr
    assert not marker.exists()


def test_teacher_band_rejects_out_of_contract_values_and_never_uses_r_s():
    harness = _BuilderHarness()
    assert harness._validate_cosmos_mechanism_teacher_band(
        4.0 / 5.0, 80.0 / 81.0
    ) == pytest.approx((4.0 / 5.0, 80.0 / 81.0))
    with pytest.raises(ValueError, match="calibrated"):
        harness._validate_cosmos_mechanism_teacher_band(0.5, 80.0 / 81.0)
    with pytest.raises(ValueError, match="calibrated"):
        harness._validate_cosmos_mechanism_teacher_band(4.0 / 5.0, 1.0)


def test_teacher_field_probe_uses_calibrated_joint_payload_and_post_injection_state():
    class Teacher:
        def __init__(self):
            self.calls = []

        def predict_raw_joint_latent_velocity(
            self, batch, *, query_latent, query_action, t
        ):
            self.calls.append((batch, query_latent.clone(), query_action.clone(), t.clone()))
            assert bool(((t >= 4.0 / 5.0) & (t <= 80.0 / 81.0)).all())
            return {
                "cosmos_joint_query": torch.full_like(query_latent, 19.0),
                "cosmos_latent_velocity": torch.full_like(query_latent, 3.0),
                "cosmos_video_frame_mask": torch.ones(
                    query_latent.shape[0],
                    query_latent.shape[2],
                    dtype=torch.bool,
                ),
            }

    class Harness(_BuilderHarness):
        def _cosmos_student_joint_map(
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
            del action_x, video_t, action_t, video_r, action_r, context
            self.student_field_input = video_x.clone()
            return video_x + 1, None, torch.full_like(video_x, 4.0), None

    harness = Harness()
    teacher = Teacher()
    query_latent = torch.full((1, 1, 2, 1, 1), 7.0)
    query_action = torch.full((1, 1, 1, 1, 1), 5.0)
    t_norm = torch.full((1, 2), 80.0 / 81.0)
    raw_video_t = t_norm * 1000
    raw_video_r = torch.full_like(raw_video_t, 800.0)
    raw_action_t = torch.full((1, 1), 1000 * 80.0 / 81.0)
    raw_action_r = torch.full_like(raw_action_t, 800.0)

    result = harness._cosmos_teacher_field_probe(
        teacher=teacher,
        batch={"raw": "batch"},
        query_latent=query_latent,
        query_action=query_action,
        t_norm=t_norm,
        raw_video_t=raw_video_t,
        raw_video_r=raw_video_r,
        raw_action_t=raw_action_t,
        raw_action_r=raw_action_r,
        context=_joint_context(),
    )

    assert torch.equal(teacher.calls[0][2], query_action)
    assert torch.equal(harness.student_field_input, torch.full_like(query_latent, 19.0))
    assert torch.equal(result["student_query_video"], harness.student_field_input)
    assert torch.equal(result["video_frame_mask"], torch.ones(1, 2, dtype=torch.bool))


def test_teacher_continuation_uses_eight_in_band_video_steps_and_held_action():
    class Teacher:
        def __init__(self):
            self.times = []
            self.actions = []

        def predict_raw_joint_latent_velocity(
            self, batch, *, query_latent, query_action, t
        ):
            del batch
            self.times.append(t.clone())
            self.actions.append(query_action.clone())
            return {
                "cosmos_joint_query": query_latent.clone(),
                "cosmos_latent_velocity": torch.ones_like(query_latent),
                "cosmos_video_frame_mask": torch.ones(
                    query_latent.shape[0],
                    query_latent.shape[2],
                    dtype=torch.bool,
                ),
            }

    harness = _BuilderHarness()
    teacher = Teacher()
    start = torch.zeros(1, 1, 2, 1, 1)
    action = torch.full((1, 1, 1, 1, 1), 6.0)
    result = harness._cosmos_teacher_video_continuation(
        teacher=teacher,
        batch={},
        start_video=start,
        held_action=action,
        t_min=4.0 / 5.0,
        t_max=80.0 / 81.0,
        teacher_steps=8,
    )

    assert len(teacher.times) == 8
    assert all(
        bool(((time >= 4.0 / 5.0) & (time <= 80.0 / 81.0)).all())
        for time in teacher.times
    )
    assert all(torch.equal(captured, action) for captured in teacher.actions)
    assert torch.allclose(
        result,
        torch.full_like(start, (4.0 / 5.0) - (80.0 / 81.0)),
    )


def test_teacher_direct_and_composed_routes_share_start_and_endpoint_time():
    class ConstantTeacher:
        raw_inference_enabled = True

        def predict_raw_joint_latent_velocity(
            self, batch, *, query_latent, query_action, t
        ):
            del batch, query_action, t
            return {
                "cosmos_joint_query": query_latent,
                "cosmos_latent_velocity": torch.ones_like(query_latent),
            }

    harness = _BuilderHarness()
    start = torch.full((1, 1, 2, 1, 1), 7.0)
    field = torch.ones_like(start)
    t_min, t_max = 4.0 / 5.0, 80.0 / 81.0
    direct = harness._cosmos_teacher_direct_endpoint(
        start_video=start,
        teacher_field_video=field,
        t_min=t_min,
        t_max=t_max,
    )
    composed = harness._cosmos_teacher_video_continuation(
        teacher=ConstantTeacher(),
        batch={},
        start_video=start,
        held_action=torch.zeros(1, 1, 1, 1, 1),
        t_min=t_min,
        t_max=t_max,
        teacher_steps=8,
    )
    assert torch.allclose(direct, composed)
    mismatched_endpoint = harness._cosmos_teacher_direct_endpoint(
        start_video=start,
        teacher_field_video=field,
        t_min=t_min + 0.01,
        t_max=t_max,
    )
    assert not torch.allclose(mismatched_endpoint, composed)


def test_shared_query_builder_uses_k4_shifted_paths_and_distinct_legal_states(
    monkeypatch,
):
    import distillation_flowmap.flowmap_step as flowmap_step_module

    class Harness(FlowMapStepMixin):
        device = torch.device("cpu")

        def __init__(self):
            self.student = object()
            self.empty_emb = torch.zeros(1, 1, 1)
            self.config = SimpleNamespace(
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
            self.calls = []

        @staticmethod
        def _mechanism_joint_input(video, action, video_t, action_t, context):
            return {
                "latent_dict": {
                    "noisy_latents": video,
                    "timesteps": video_t,
                },
                "action_dict": {
                    "noisy_latents": action,
                    "timesteps": action_t,
                },
            }

        @staticmethod
        def _init_joint_mask(_joint_input):
            return None

        def _student_joint_forward(
            self,
            model,
            joint_input,
            empty_emb,
            video_r,
            action_r,
            **kwargs,
        ):
            del model, empty_emb, kwargs
            self.calls.append(
                SimpleNamespace(
                    video=joint_input["latent_dict"]["noisy_latents"],
                    action=joint_input["action_dict"]["noisy_latents"],
                    video_t=joint_input["latent_dict"]["timesteps"],
                    action_t=joint_input["action_dict"]["timesteps"],
                    video_r=video_r,
                    action_r=action_r,
                )
            )
            return (
                torch.ones_like(joint_input["latent_dict"]["noisy_latents"]),
                torch.ones_like(joint_input["action_dict"]["noisy_latents"]),
            )

        @staticmethod
        def _extract_action_v(action, _frames):
            return action

        @staticmethod
        def _joint_euler_update(
            video, action, video_v, action_v, video_t, video_r, action_t, action_r
        ):
            def view(time):
                return time[:, None, :, None, None] / 1000.0

            return (
                video + video_v * (view(video_r) - view(video_t)),
                action + action_v * (view(action_r) - view(action_t)),
            )

    batch = {
        "latents": torch.zeros(2, 1, 3, 1, 1),
        "actions": torch.zeros(2, 7, 16, 4, 1),
    }
    input_dict = {
        "latent_dict": {
            "latent": batch["latents"],
            "cond_timesteps": torch.zeros(2, 3),
            "text_emb": torch.zeros(2, 1, 1),
            "grid_id": None,
        },
        "action_dict": {
            "latent": batch["actions"],
            "cond_timesteps": torch.zeros(2, 16),
            "text_emb": torch.zeros(2, 1, 1),
            "grid_id": None,
            "actions_mask": torch.ones_like(batch["actions"][:, :1]),
        },
        "chunk_size": 1,
        "window_size": 1,
    }
    monkeypatch.setattr(
        flowmap_step_module,
        "sample_nonterminal_semantic_query_indices",
        lambda sigmas, batch_size: torch.tensor([1, 2]),
    )
    harness = Harness()
    shared = harness._build_cosmos_shifted_shared_query(
        batch, input_dict, student_steps=4
    )

    video_sigmas = build_shifted_terminal_path(
        steps=4, shift=5.0, device=torch.device("cpu"), dtype=torch.float32
    )
    action_sigmas = build_shifted_terminal_path(
        steps=4, shift=0.05, device=torch.device("cpu"), dtype=torch.float32
    )
    torch.testing.assert_close(
        shared["query_video_t"],
        torch.stack((video_sigmas[1].expand(3), video_sigmas[2].expand(3)))
        * 1000,
    )
    torch.testing.assert_close(
        shared["query_action_t"],
        torch.stack((action_sigmas[1].expand(4), action_sigmas[2].expand(4)))
        * 1000,
    )
    unpacked = unpack_flowmap_action_query(
        harness.calls[0].action,
        used_action_channel_ids=list(range(7)),
        packing_schema="downsample_survivor_v2",
        downsample_factor=4,
    )
    torch.testing.assert_close(unpacked, shared["native_action_prior"])
    assert shared["student_steps"] == 4
    assert shared["query_indices"].tolist() == [1, 2]


class _AlignedMechanismTeacher:
    raw_inference_enabled = True

    def __init__(self, *, legacy_value=0.0, mask_mode="valid", steps=8):
        self.legacy_value = legacy_value
        self.mask_mode = mask_mode
        self.steps = steps
        self.field_call = None
        self.same_prior_call = None
        self.continuation_call = None

    def _mask(self, reference, *, endpoint=False):
        mask = torch.zeros(
            reference.shape[0], reference.shape[2], dtype=torch.bool
        )
        if self.mask_mode != "empty":
            mask[:, 6:8] = True
        if self.mask_mode == "mismatch" and endpoint:
            mask[0, 6] = False
        return mask

    def predict_raw_joint_latent_velocity(
        self, batch, query_latent, query_action, t
    ):
        del batch
        self.field_call = SimpleNamespace(
            video=query_latent, action=query_action, t=t
        )
        return {
            "cosmos_joint_query": query_latent,
            "cosmos_latent_velocity": torch.zeros_like(query_latent),
            "cosmos_video_frame_mask": self._mask(query_latent),
        }

    def predict_raw_same_prior_endpoint(
        self, batch, *, video_prior, action_prior, teacher_steps
    ):
        del batch
        self.same_prior_call = SimpleNamespace(
            video_prior=video_prior,
            action_prior=action_prior,
            teacher_steps=teacher_steps,
        )
        return {
            "endpoint_video": torch.full_like(video_prior, 9.0),
            "video_frame_mask": self._mask(video_prior, endpoint=True),
            "effective_teacher_steps": self.steps,
            "video_prior_sha256": "verified",
            "action_prior_sha256": "verified",
        }

    def predict_raw_joint_continuation_endpoint(
        self,
        batch,
        *,
        canonical_joint_state,
        normalized_t,
        teacher_steps,
    ):
        del batch
        self.continuation_call = SimpleNamespace(
            state=canonical_joint_state,
            normalized_t=normalized_t,
            teacher_steps=teacher_steps,
        )
        batch_size = canonical_joint_state.shape[0]
        return {
            "endpoint_video": torch.full_like(canonical_joint_state, 7.0),
            "video_frame_mask": self._mask(
                canonical_joint_state, endpoint=True
            ),
            "effective_teacher_steps": self.steps,
            "joint_state_sha256": "verified",
            "normalized_t_sha256": "verified",
            "normalized_t": normalized_t[:, 0],
            "edm_sigma": normalized_t[:, 0] / (1.0 - normalized_t[:, 0]),
        }

    def predict_raw_latent_target(self, batch, **kwargs):
        del batch, kwargs
        return {
            "cosmos_latent_x0": torch.full(
                (2, 16, 9, 28, 28), self.legacy_value
            ),
            "actions": torch.full((2, 16, 7), self.legacy_value),
        }


class _AlignedMechanismHarness(FlowMapStepMixin):
    device = torch.device("cpu")

    def __init__(self, teacher, *, action_snr_shift=0.05):
        self.teacher = teacher
        self.student = object()
        self.empty_emb = torch.zeros(1, 1, 1)
        self.calls = []
        self.config = SimpleNamespace(
            num_train_timesteps=1000,
            cosmos_latent_channels=16,
            cosmos_latent_frames=9,
            cosmos_latent_height=28,
            cosmos_latent_width=28,
            action_downsample_factor=4,
            action_packing_schema="downsample_survivor_v2",
            used_action_channel_ids=list(range(7)),
            inverse_used_action_channel_ids=list(range(7)),
            snr_shift=5.0,
            action_snr_shift=action_snr_shift,
            opd_danceopd_rollout_steps=(2, 4),
            cfg_min=1.0,
            cfg_max=1.0,
            cosmos_latent_epsilon=1e-3,
            norm_stat={"q01": 0.0, "q99": 1.0},
            mechanism_diagnostic_r=500.0,
            mechanism_diagnostic_s=250.0,
            mechanism_cosmos_t_min=4.0 / 5.0,
            mechanism_cosmos_t_max=80.0 / 81.0,
        )

    @staticmethod
    def _prepare_base_dict(batch):
        batch_size = batch["latents"].shape[0]
        return {
            "latent_dict": {
                "latent": batch["latents"],
                "cond_timesteps": torch.zeros(batch_size, 9),
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

    def _prepare_cosmos_mechanism_context(self, batch):
        input_dict = self._prepare_base_dict(batch)
        action = batch["actions"][:, :, ::4]
        return {
            "batch": batch,
            "input_dict": input_dict,
            "batch_size": batch["latents"].shape[0],
            "ref_shape": tuple(batch["latents"].shape),
            "action_clean": action,
            "action_frames": action.shape[2],
            "action_mask": input_dict["action_dict"]["actions_mask"][:, :, ::4],
            "student_model": self.student,
            "video_base": input_dict["latent_dict"],
            "action_latent": action,
            "action_cond_t": input_dict["action_dict"]["cond_timesteps"][:, ::4],
            "action_text": input_dict["action_dict"]["text_emb"],
            "action_grid": None,
            "empty_emb": self.empty_emb.expand(batch["latents"].shape[0], -1, -1),
            "chunk_size": 1,
            "window_size": 1,
            "cfg_scale": 1.0,
        }

    @staticmethod
    def _mechanism_joint_input(video, action, video_t, action_t, context):
        return {
            "latent_dict": {
                "noisy_latents": video,
                "timesteps": video_t,
            },
            "action_dict": {
                "noisy_latents": action,
                "timesteps": action_t,
            },
        }

    @staticmethod
    def _init_joint_mask(_joint_input):
        return None

    def _student_joint_forward(
        self,
        model,
        joint_input,
        empty_emb,
        video_r,
        action_r,
        *,
        require_action,
        **kwargs,
    ):
        del model, empty_emb, kwargs
        call = SimpleNamespace(
            video=joint_input["latent_dict"]["noisy_latents"],
            action=joint_input["action_dict"]["noisy_latents"],
            video_t=joint_input["latent_dict"]["timesteps"],
            action_t=joint_input["action_dict"]["timesteps"],
            video_r=video_r,
            action_r=action_r,
            require_action=require_action,
        )
        self.calls.append(call)
        video_velocity = torch.ones_like(call.video)
        if not require_action:
            return video_velocity
        return video_velocity, torch.ones_like(call.action)

    @staticmethod
    def _extract_action_v(action, _frames):
        return action

    @staticmethod
    def _joint_euler_update(
        video, action, video_v, action_v, video_t, video_r, action_t, action_r
    ):
        def view(time):
            return time[:, None, :, None, None] / 1000.0

        return (
            video + video_v * (view(video_r) - view(video_t)),
            action + action_v * (view(action_r) - view(action_t)),
        )

    def _cosmos_action_context_predictions(self, *args, **kwargs):
        del args, kwargs
        zero = torch.zeros(2, 7, 4, 4, 1)
        return {"gt": zero, "student": zero, "teacher_video": zero}


def _run_aligned_mechanism_case(
    monkeypatch,
    *,
    legacy_value=0.0,
    mask_mode="valid",
    steps=8,
    equal_clocks=False,
):
    import distillation_flowmap.flowmap_step as flowmap_step_module

    captured_metrics = {}
    monkeypatch.setattr(
        flowmap_step_module,
        "sample_nonterminal_semantic_query_indices",
        lambda sigmas, batch_size: torch.tensor([1, 2]),
    )
    monkeypatch.setattr(
        flowmap_step_module,
        "cosmos_actions_to_flowmap_x0",
        lambda actions, **kwargs: torch.zeros(
            kwargs["target_shape"], dtype=kwargs["dtype"]
        ),
    )

    def capture_samples(**kwargs):
        captured_metrics.update(kwargs)
        return compute_mechanism_metric_samples(**kwargs)

    monkeypatch.setattr(
        flowmap_step_module, "compute_mechanism_metric_samples", capture_samples
    )
    teacher = _AlignedMechanismTeacher(
        legacy_value=legacy_value, mask_mode=mask_mode, steps=steps
    )
    harness = _AlignedMechanismHarness(
        teacher, action_snr_shift=5.0 if equal_clocks else 0.05
    )
    batch = {
        "latents": torch.zeros(2, 16, 9, 28, 28),
        "actions": torch.zeros(2, 7, 16, 4, 1),
    }
    stats = harness._run_cosmos_mechanism_probe(
        batch, seed=17, teacher_steps=8, student_steps=4
    )
    return harness, teacher, captured_metrics, stats


def test_mixed_clock_probe_skips_continuation_but_keeps_composition_metrics(
    monkeypatch,
):
    harness, teacher, metrics, stats = _run_aligned_mechanism_case(monkeypatch)

    rollout = harness.calls[:4]
    field = harness.calls[4]
    direct, first_edge, second_edge = harness.calls[5:8]
    video_sigmas = build_shifted_terminal_path(
        steps=4, shift=5.0, device=torch.device("cpu"), dtype=torch.float32
    )
    expected_r = torch.stack(
        (video_sigmas[1].expand(9), video_sigmas[2].expand(9))
    ) * 1000
    torch.testing.assert_close(field.video_t, expected_r)
    assert field.video.data_ptr() == direct.video.data_ptr()
    assert teacher.continuation_call is None
    torch.testing.assert_close(direct.video_t, expected_r)
    assert torch.count_nonzero(direct.video_r) == 0
    torch.testing.assert_close(first_edge.video_t, expected_r)
    assert torch.all(first_edge.video_r == 250)
    assert torch.all(second_edge.video_t == 250)
    assert torch.count_nonzero(second_edge.video_r) == 0
    assert torch.all(first_edge.action_r < first_edge.action_t)
    assert torch.all(second_edge.action_t > second_edge.action_r)
    assert metrics["teacher_continuation_video"] is None
    torch.testing.assert_close(
        metrics["same_prior_teacher_endpoint_video"],
        torch.full_like(metrics["same_prior_teacher_endpoint_video"], 9.0),
    )
    assert metrics["shared_state_verified"] is True
    assert metrics["same_prior_verified"] is True
    assert metrics["continuation_verified"] is False
    assert metrics["effective_teacher_steps"] is None
    assert stats["mechanism/g_anchor_count"].item() == 0
    assert stats["mechanism/g_anchor_mse_count"].item() == 0
    assert stats["mechanism/g_anchor_available_sum"].item() == 0
    assert stats["mechanism/g_anchor_available_count"].item() == 2
    assert (
        stats["mechanism/g_anchor_unavailable_mixed_clock_sum"].item()
        == 2
    )
    assert stats["mechanism/g_comp_count"].item() == 2
    assert stats["mechanism/student_steps_sum"].item() == 8
    assert len(rollout) == 4


def test_equal_clock_probe_dispatches_continuation_and_computes_anchor(
    monkeypatch,
):
    harness, teacher, metrics, stats = _run_aligned_mechanism_case(
        monkeypatch, equal_clocks=True
    )

    field = harness.calls[4]
    assert teacher.continuation_call.state.data_ptr() == field.video.data_ptr()
    torch.testing.assert_close(
        teacher.continuation_call.normalized_t,
        field.video_t / 1000.0,
    )
    assert metrics["continuation_verified"] is True
    assert metrics["effective_teacher_steps"] == 8
    assert stats["mechanism/g_anchor_count"].item() == 2
    assert stats["mechanism/g_anchor_mse_count"].item() == 2
    assert stats["mechanism/g_anchor_available_sum"].item() == 2
    assert (
        stats["mechanism/g_anchor_unavailable_mixed_clock_sum"].item()
        == 0
    )


def test_legacy_target_changes_cannot_change_g_inputs(monkeypatch):
    _, _, first, _ = _run_aligned_mechanism_case(
        monkeypatch, legacy_value=-100.0, equal_clocks=True
    )
    _, _, second, _ = _run_aligned_mechanism_case(
        monkeypatch, legacy_value=100.0, equal_clocks=True
    )
    for key in (
        "teacher_continuation_video",
        "same_prior_teacher_endpoint_video",
        "direct_route_video",
        "composed_route_video",
    ):
        torch.testing.assert_close(first[key], second[key])


@pytest.mark.parametrize(
    "mask_mode,steps,match",
    [
        ("empty", 8, "nonempty mask"),
        ("mismatch", 8, "mask"),
        ("valid", 7, "eight-step"),
    ],
)
def test_live_probe_fails_closed_on_mask_or_step_provenance(
    monkeypatch, mask_mode, steps, match
):
    with pytest.raises(RuntimeError, match=match):
        _run_aligned_mechanism_case(
            monkeypatch,
            mask_mode=mask_mode,
            steps=steps,
            equal_clocks=True,
        )


def test_probe_interface_is_observation_only_and_capability_gates_teacher_joint():
    assert hasattr(FlowMapStepMixin, "_run_cosmos_mechanism_probe")
    assert getattr(
        FlowMapStepMixin._run_cosmos_mechanism_probe, "__wrapped__", None
    ) is not None


def test_trainer_installs_all_rank_post_success_probe_with_failure_abort():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    assert "capture_diagnostic_snapshot_if_due(" in source
    assert "completed_step=self.step + 1" in source
    assert "self._mechanism_diagnostic_snapshot = None" in source
    assert "def _sync_cosmos_mechanism_preflight(" in source
    assert "dist.all_reduce(status, op=dist.ReduceOp.MIN)" in source
    assert "self._validate_cosmos_mechanism_preflight(" in source
    assert "fatal_mechanism_process_exit(" in source
    assert "dist.destroy_process_group()" not in source
    assert "reduced = reduce_metric_stats(local_stats)" in source
    runner_source = source[
        source.index("def _run_cosmos_mechanism_diagnostics("):
        source.index("def _maybe_run_mechanism_diagnostics(")
    ]
    assert (
        runner_source.index("try:")
        < runner_source.index("materialize_diagnostic_snapshot(")
        < runner_source.index("self._validate_cosmos_mechanism_preflight(")
        < runner_source.rindex("self._sync_cosmos_mechanism_preflight(")
    )
