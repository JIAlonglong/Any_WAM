import importlib
import json
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for path in (
    REPO_ROOT,
    os.path.join(REPO_ROOT, "distillation_flowmap"),
    os.path.join(REPO_ROOT, "wan_va"),
):
    if path not in sys.path:
        sys.path.insert(0, path)


from distillation_flowmap.cosmos_mixed_step_policy import (
    CosmosMixedStepPolicySpec,
    append_selection_jsonl,
    build_selection_record,
    get_mixed_danceopd_schedule,
    get_mixed_step_policy_spec,
    record_selection,
    select_rank_synchronized_pair,
)
import distillation_flowmap.flowmap_step as flowmap_step
from distillation_flowmap.flowmap_step import (
    FlowMapStepMixin,
)


@pytest.mark.parametrize(
    ("name", "expected_weights"),
    [
        ("universe", (0.50, 0.30, 0.20)),
        ("s2", (0.20, 0.60, 0.20)),
        ("s1", (0.70, 0.20, 0.10)),
    ],
)
def test_mixed_policy_specs_have_the_agreed_pair_order_and_weights(name, expected_weights):
    spec = get_mixed_step_policy_spec(name)

    assert spec.name == name
    assert spec.rollout_step_pairs == ((4, 1), (4, 2), (8, 4))
    assert spec.weights == pytest.approx(expected_weights)
    assert math.fsum(spec.weights) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "forced_index, expected", [(0, (1, 0.25)), (1, (2, 0.50)), (2, (4, 1.00))]
)
def test_mixed_dance_schedule_matches_the_selected_endpoint_pair(forced_index, expected):
    selection = select_rank_synchronized_pair(
        get_mixed_step_policy_spec("universe"), forced_indices=(forced_index,)
    )

    schedule = get_mixed_danceopd_schedule(selection)

    assert (schedule.rollout_steps, schedule.velocity_weight) == pytest.approx(expected)
    assert schedule.rollout_steps == selection.student_steps


def test_policy_spec_normalizes_positive_finite_weights():
    spec = CosmosMixedStepPolicySpec(
        name="test",
        rollout_step_pairs=((4, 1), (8, 4)),
        weights=(2.0, 3.0),
    )

    assert spec.weights == pytest.approx((0.4, 0.6))


def test_unknown_policy_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown Cosmos mixed-step policy"):
        get_mixed_step_policy_spec("s3")


class _NoRandomDraw:
    def random(self):
        raise AssertionError("nonzero rank must consume the rank-zero broadcast")


def test_rank_zero_weighted_pair_is_broadcast_without_nonzero_resampling():
    spec = get_mixed_step_policy_spec("universe")
    wire = {}

    rank_zero = select_rank_synchronized_pair(
        spec,
        rank=0,
        seed=17,
        broadcast_index=lambda index: wire.setdefault("index", index),
    )
    rank_one = select_rank_synchronized_pair(
        spec,
        rank=1,
        generator=_NoRandomDraw(),
        broadcast_index=lambda _unused: wire["index"],
    )

    assert rank_one.index == rank_zero.index
    assert rank_one.rollout_step_pair == rank_zero.rollout_step_pair
    assert rank_one.label == rank_zero.label


class _EndpointTeacher:
    raw_inference_enabled = True

    def predict_raw_latent_target(self, _batch, *, noise, **_kwargs):
        return {"cosmos_latent_x0": torch.zeros_like(noise)}

    def predict_raw_latent_velocity(self, _batch, *, query_latent, **_kwargs):
        return {"cosmos_latent_velocity": torch.zeros_like(query_latent)}


class _EndpointLatentScheduler:
    @staticmethod
    def training_target(latents, _noise, _timesteps):
        return torch.zeros_like(latents)


class _EndpointWireProbe(FlowMapStepMixin):
    """Minimal CPU-only probe that executes the real full-OPD endpoint method."""

    def __init__(self, metrics_path, *, cosmos_mixed_step_policy="universe"):
        self.config = SimpleNamespace(
            cosmos_mixed_step_policy=cosmos_mixed_step_policy,
            # Generic OPD paths remain at the legacy S4 singleton pair.
            opd_rollout_step_pairs=[[8, 4]],
            rollout_step_pairs=[[8, 4]],
            opd_rollout_step_pair_weights=None,
            # Only the full-OPD endpoint selector owns the mixed distribution.
            opd_mixed_endpoint_rollout_step_pairs=[[4, 1], [4, 2], [8, 4]],
            opd_mixed_endpoint_rollout_step_pair_weights=[0.50, 0.30, 0.20],
            opd_rollout_step_forced_indices=(1,),
            opd_rollout_selection_seed=23,
            opd_rollout_selection_metrics_path=str(metrics_path),
            opd_selected_rollout_pair_context=None,
            cosmos_latent_channels=1,
            cosmos_latent_frames=1,
            cosmos_latent_height=1,
            cosmos_latent_width=1,
            cosmos_latent_epsilon=0.001,
            cosmos_use_teacher_action_anchor=False,
            opd_endpoint_focus_prob=0.0,
            opd_cosmos_spatial_crop_size=0,
            num_train_timesteps=10,
            cfg_min=1.0,
            cfg_max=1.0,
            opd_joint_action_rollout=False,
            opd_danceopd_action_endpoint_weight=0.0,
            opd_danceopd_endpoint_weight=1.0,
            opd_danceopd_velocity_weight=1.0,
            opd_aux_weight=1.0,
            opd_danceopd_rollout_steps=16,
            rank=0,
        )
        self._teacher_nofsdp = _EndpointTeacher()
        self.train_scheduler_latent = _EndpointLatentScheduler()
        self.device = torch.device("cpu")
        self.step = 3
        self.distill_action = False
        self.action_aware = False
        self.empty_emb = torch.zeros(1, 1, 1)
        self.student_k_steps = []
        self.dance_calls = []

    @staticmethod
    def convert_input_format(batch):
        return batch

    def sample_cosmos_latent_timestep_mixed(self, batch_size, num_frames, *, dtype, device):
        video_t = torch.full((batch_size, num_frames), 2.0, dtype=dtype, device=device)
        video_r = torch.full((batch_size, num_frames), 1.0, dtype=dtype, device=device)
        return video_t, video_r, video_t / 10.0, video_r / 10.0, None

    @staticmethod
    def _prepare_base_dict(_batch):
        return {
            "latent_dict": {},
            "action_dict": {},
            "chunk_size": None,
            "window_size": None,
        }

    def _student_euler_integrate(self, **kwargs):
        self.student_k_steps.append(kwargs["K_steps"])
        state = torch.ones_like(kwargs["noisy_latents"], requires_grad=True)
        velocity = torch.zeros_like(state, requires_grad=True)
        return state, velocity

    def _cosmos_danceopd_velocity_loss(self, *_args, **kwargs):
        self.dance_calls.append({
            "rollout_steps": kwargs.get("rollout_steps"),
            "append_post_update_terminal": kwargs.get("append_post_update_terminal"),
        })
        zero = torch.zeros((), device=self.device, requires_grad=True)
        effective_rollout_steps = kwargs.get("rollout_steps")
        if effective_rollout_steps is None:
            effective_rollout_steps = self.config.opd_danceopd_rollout_steps
        return zero, {
            "query_index_mean": zero.detach(),
            "query_sigma_mean": zero.detach(),
            "terminal_query_sigma": zero.detach(),
            "terminal_query_timestep": zero.detach(),
            "query_t_min": zero.detach(),
            "query_t_max": zero.detach(),
            "terminal_prior_max_error": zero.detach(),
            "rollout_steps": effective_rollout_steps,
            "state_count": effective_rollout_steps + int(
                bool(kwargs.get("append_post_update_terminal"))
            ),
        }


def test_real_full_opd_endpoint_wiring_uses_mixed_pair_without_touching_generic_paths(
    monkeypatch, tmp_path
):
    teacher_rollout_steps = []
    monkeypatch.setattr(
        flowmap_step,
        "apply_full_endpoint_focus",
        lambda video_t, video_r, **_kwargs: (
            video_t,
            video_r,
            torch.zeros_like(video_t, dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(
        flowmap_step,
        "center_spatial_crop_slices",
        lambda height, width, **_kwargs: (slice(0, height), slice(0, width)),
    )

    def fake_rollout_velocity_field(initial_state, *_args, num_steps, **_kwargs):
        teacher_rollout_steps.append(num_steps)
        zero = torch.zeros_like(initial_state)
        return zero, zero, None

    monkeypatch.setattr(flowmap_step, "rollout_velocity_field", fake_rollout_velocity_field)
    monkeypatch.setattr(
        flowmap_step,
        "denoised_endpoint_mse",
        lambda student_x, *_args: student_x.sum(),
    )

    probe = _EndpointWireProbe(tmp_path / "mixed_endpoint.jsonl")
    result = probe._cosmos_latent_full_opd_aux_transition_step(
        {"actions": torch.zeros(1, 1, 1, 1, 1)},
        batch_idx=0,
    )

    assert probe.student_k_steps == [2]
    assert teacher_rollout_steps == [4]
    assert result["cosmos_mixed_step_pair_index"] == 1
    assert result["cosmos_mixed_step_pair_label"] == "s2"
    assert probe.config.opd_rollout_step_pairs == [[8, 4]]
    assert probe.config.rollout_step_pairs == [[8, 4]]
    assert probe.config.opd_danceopd_rollout_steps == 16
    assert probe.config.opd_selected_rollout_pair_context["pair_label"] == "s2"
    assert probe.dance_calls == [
        {"rollout_steps": 2, "append_post_update_terminal": True}
    ]
    assert result["danceopd_velocity_rollout_steps"] == 2
    assert result["danceopd_velocity_state_count"] == 3
    assert result["danceopd_velocity_weight"] == pytest.approx(0.50)
    assert "danceopd_terminal_query_sigma" in result


def test_real_full_opd_endpoint_wiring_retains_legacy_dance_behavior(monkeypatch, tmp_path):
    monkeypatch.setattr(
        flowmap_step,
        "apply_full_endpoint_focus",
        lambda video_t, video_r, **_kwargs: (
            video_t,
            video_r,
            torch.zeros_like(video_t, dtype=torch.bool),
        ),
    )
    monkeypatch.setattr(
        flowmap_step,
        "center_spatial_crop_slices",
        lambda height, width, **_kwargs: (slice(0, height), slice(0, width)),
    )
    monkeypatch.setattr(
        flowmap_step,
        "rollout_velocity_field",
        lambda initial_state, *_args, **_kwargs: (
            torch.zeros_like(initial_state),
            torch.zeros_like(initial_state),
            None,
        ),
    )
    monkeypatch.setattr(
        flowmap_step,
        "denoised_endpoint_mse",
        lambda student_x, *_args: student_x.sum(),
    )

    probe = _EndpointWireProbe(
        tmp_path / "legacy_endpoint.jsonl", cosmos_mixed_step_policy=""
    )
    result = probe._cosmos_latent_full_opd_aux_transition_step(
        {"actions": torch.zeros(1, 1, 1, 1, 1)},
        batch_idx=0,
    )

    assert probe.dance_calls == [
        {"rollout_steps": None, "append_post_update_terminal": False}
    ]
    assert result["danceopd_velocity_rollout_steps"] == 16
    assert result["danceopd_velocity_state_count"] == 16
    assert result["danceopd_velocity_weight"] == pytest.approx(1.0)
    assert "danceopd_terminal_query_sigma" not in result


def test_selection_record_updates_pair_label_histogram_and_persists_provenance(tmp_path):
    spec = get_mixed_step_policy_spec("s1")
    selection = select_rank_synchronized_pair(
        spec,
        forced_indices=(2,),
        selection_ordinal=0,
    )
    histogram = {}

    record = build_selection_record(
        spec=spec,
        selection=selection,
        histogram=record_selection(histogram, selection),
        seed=31,
        global_step=12,
        selection_ordinal=4,
    )
    path = tmp_path / "cosmos_mixed_step_opd.jsonl"
    append_selection_jsonl(path, record)

    assert histogram == {"s4": 1}
    assert record["policy_name"] == "s1"
    assert record["rollout_step_pairs"] == [[4, 1], [4, 2], [8, 4]]
    assert record["weights"] == pytest.approx([0.70, 0.20, 0.10])
    assert record["pair_index"] == 2
    assert record["pair_label"] == "s4"
    assert record["teacher_steps"] == 8
    assert record["student_steps"] == 4
    assert record["global_step"] == 12
    assert record["histogram"] == {"s4": 1}
    assert json.loads(path.read_text().strip()) == record


def test_progressive_config_isolates_mixed_endpoint_pairs_from_generic_opd(monkeypatch):
    monkeypatch.setenv("COSMOS_MIXED_STEP_POLICY", "universe")
    monkeypatch.setenv("COSMOS_MIXED_STEP_FORCE_SEQUENCE", "s1,s2,s4")
    monkeypatch.setenv("COSMOS_MIXED_STEP_SELECTOR_SEED", "31")
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "s4")
    monkeypatch.setenv("OPD_AUX_WARMUP_STEPS", "0")
    monkeypatch.setenv("OPD_AUX_INTERVAL", "1")
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.cosmos_mixed_step_policy == "universe"
    assert module.cfg.opd_rollout_step_pairs == [[8, 4]]
    assert module.cfg.rollout_step_pairs == [[8, 4]]
    assert module.cfg.opd_rollout_step_pair_weights is None
    assert module.cfg.opd_mixed_endpoint_rollout_step_pairs == [
        [4, 1],
        [4, 2],
        [8, 4],
    ]
    assert module.cfg.opd_mixed_endpoint_rollout_step_pair_weights == pytest.approx(
        [0.50, 0.30, 0.20]
    )
    assert module.cfg.opd_rollout_step_forced_indices == (0, 1, 2)
    assert module.cfg.opd_rollout_selection_seed == 31
    assert module.cfg.opd_aux_warmup_steps == 0
    assert module.cfg.opd_aux_interval == 1
