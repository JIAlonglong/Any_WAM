import importlib
import json
import math
import os
import sys

import pytest


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
    get_mixed_step_policy_spec,
    record_selection,
    select_rank_synchronized_pair,
)
from distillation_flowmap.flowmap_step import (
    _select_cosmos_mixed_step_endpoint_rollout,
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


class _MixedPolicyConfig:
    cosmos_mixed_step_policy = "universe"
    opd_rollout_step_pairs = [[4, 1], [4, 2], [8, 4]]
    opd_rollout_step_pair_weights = [0.50, 0.30, 0.20]
    opd_rollout_step_forced_indices = (1,)
    opd_rollout_selection_seed = 123
    opd_danceopd_rollout_steps = 16


def test_one_endpoint_selection_controls_both_rollout_branches_but_not_danceopd():
    config = _MixedPolicyConfig()
    selection = _select_cosmos_mixed_step_endpoint_rollout(
        config,
        device=None,
        global_step=7,
        selection_ordinal=0,
        rank=0,
    )

    student_euler_call = {"K_steps": selection.student_steps}
    teacher_velocity_call = {"num_steps": selection.teacher_steps}

    assert selection.rollout_step_pair == (4, 2)
    assert student_euler_call["K_steps"] == 2
    assert teacher_velocity_call["num_steps"] == 4
    assert config.opd_danceopd_rollout_steps == 16


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


def test_progressive_config_exposes_policy_weights_and_forced_preflight_sequence(monkeypatch):
    monkeypatch.setenv("COSMOS_MIXED_STEP_POLICY", "universe")
    monkeypatch.setenv("COSMOS_MIXED_STEP_FORCE_SEQUENCE", "s1,s2,s4")
    monkeypatch.setenv("COSMOS_MIXED_STEP_SELECTOR_SEED", "31")
    module = importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    )
    module = importlib.reload(module)

    assert module.cfg.cosmos_mixed_step_policy == "universe"
    assert module.cfg.opd_rollout_step_pairs == [[4, 1], [4, 2], [8, 4]]
    assert module.cfg.opd_rollout_step_pair_weights == pytest.approx([0.50, 0.30, 0.20])
    assert module.cfg.opd_rollout_step_forced_indices == (0, 1, 2)
    assert module.cfg.opd_rollout_selection_seed == 31
