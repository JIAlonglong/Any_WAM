from __future__ import annotations

import os
import sys
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

from distillation_flowmap.flowmap_step import FlowMapStepMixin
from distillation_flowmap.mechanism_diagnostics import (
    compute_mechanism_metric_samples,
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
        teacher_cont_video=torch.tensor([[[2.0, 4.0]]]),
        teacher_endpoint_video=torch.tensor([[[1.0, 1.0]]]),
        student_direct_video=torch.tensor([[[5.0, 7.0]]]),
        student_composed_video=torch.tensor([[[4.0, 5.0]]]),
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

    assert samples["mechanism/g_anchor"].item() == 10.0
    assert samples["mechanism/g_anchor_mse"].item() == 5.0
    assert samples["mechanism/g_comp"].item() == 5.0
    assert samples["mechanism/g_comp_mse"].item() == 2.5
    assert samples["mechanism/video_endpoint_error"].item() == 52.0
    # The mask keeps only the first temporal element: (9 - 8)^2.
    assert samples["mechanism/video_field_match_error"].item() == 1.0

    stats = pack_finite_metric_stats(samples)
    assert stats["mechanism/action_error_teacher_joint_context_count"].item() == 0
    assert stats["mechanism/video_to_action_full_joint_gain_count"].item() == 0
    assert stats["mechanism/video_to_action_residual_action_gap_count"].item() == 0
    assert stats["mechanism/teacher_joint_available_sum"].item() == 0
    assert stats["mechanism/teacher_joint_available_count"].item() == 1
    assert means_from_reduced_stats(stats)["mechanism/diagnostic_valid"] == 1.0


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


def test_probe_interface_is_observation_only_and_capability_gates_teacher_joint():
    assert hasattr(FlowMapStepMixin, "_run_cosmos_mechanism_probe")
    assert getattr(
        FlowMapStepMixin._run_cosmos_mechanism_probe, "__wrapped__", None
    ) is not None


def test_trainer_installs_all_rank_post_success_probe_with_failure_abort():
    source = (
        Path(__file__).resolve().parents[1] / "flowmap_trainer.py"
    ).read_text(encoding="utf-8")
    assert "self._mechanism_diagnostic_batch = batch" in source
    assert "if not skipped_optimizer_step:" in source
    assert "def _sync_cosmos_mechanism_preflight(" in source
    assert "dist.all_reduce(status, op=dist.ReduceOp.MIN)" in source
    assert "def _abort_cosmos_mechanism_forward_failure(" in source
    assert "dist.destroy_process_group()" in source
    assert "reduced = reduce_metric_stats(local_stats)" in source
