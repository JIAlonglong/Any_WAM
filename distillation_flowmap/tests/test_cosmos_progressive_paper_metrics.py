import pytest
import torch

from distillation_flowmap.cosmos_progressive_paper_metrics import (
    DEPLOYMENT_GRID,
    composition_pairs,
    internal_rollout_nodes,
    mean_video_mse,
    merge_task_records,
    paper_metric_template,
)


def test_s4_paper_grid_uses_only_deployed_internal_states():
    assert DEPLOYMENT_GRID == (1000, 750, 500, 250, 0)
    assert internal_rollout_nodes() == (750, 500, 250)
    assert composition_pairs() == ((750, 500), (500, 250))


def test_mean_video_mse_ignores_action_channels():
    video_left = torch.zeros(1, 16, 1, 1, 1)
    video_right = torch.ones_like(video_left)
    action_left = torch.zeros(1, 30, 1, 1, 1)
    action_right = torch.full_like(action_left, 999.0)

    assert mean_video_mse(video_left, video_right) == 1.0


def test_paper_metric_template_has_every_table_three_slot():
    assert paper_metric_template() == {
        "g_anchor": {
            "node_750": None,
            "node_500": None,
            "node_250": None,
        },
        "g_comp": {
            "pair_750_500": None,
            "pair_500_250": None,
        },
        "video_ep": {
            "node_750": None,
            "node_500": None,
            "node_250": None,
        },
        "field_match": {
            "node_750": None,
            "node_500": None,
            "node_250": None,
        },
    }


def test_merge_task_records_macro_averages_task_means_and_preserves_leaf_names():
    records = [
        {
            "task": "easy",
            "record_index": 0,
            "pair_id": "pair-a",
            "metrics": {
                "g_anchor": {"node_750": 1.0},
                "g_comp/pair_750_500": 2.0,
            },
        },
        {
            "task": "easy",
            "record_index": 1,
            "pair_id": "pair-a",
            "metrics": {
                "g_anchor": {"node_750": 3.0},
                "g_comp/pair_750_500": 4.0,
            },
        },
        {
            "task": "hard",
            "record_index": 0,
            "pair_id": "pair-a",
            "metrics": {
                "g_anchor": {"node_750": 9.0},
                "g_comp/pair_750_500": 8.0,
            },
        },
    ]

    assert merge_task_records(records) == {
        "num_records": 3,
        "per_task": {
            "easy": {
                "g_anchor/node_750": 2.0,
                "g_comp/pair_750_500": 3.0,
            },
            "hard": {
                "g_anchor/node_750": 9.0,
                "g_comp/pair_750_500": 8.0,
            },
        },
        "metrics": {
            "g_anchor/node_750": 5.5,
            "g_comp/pair_750_500": 5.5,
        },
    }


def test_merge_task_records_rejects_non_string_task_before_key_coercion():
    numeric_task_record = {
        "task": 1,
        "record_index": 0,
        "pair_id": "pair-a",
        "metrics": {"g_anchor/node_750": 1.0},
    }
    string_task_record = dict(numeric_task_record)
    string_task_record["task"] = "1"

    with pytest.raises(ValueError, match="task.*non-empty string"):
        merge_task_records([numeric_task_record, string_task_record])


@pytest.mark.parametrize("identity_field", ("task", "record_index", "pair_id"))
def test_merge_task_records_rejects_missing_identity(identity_field):
    record = {
        "task": "easy",
        "record_index": 0,
        "pair_id": "pair-a",
        "metrics": {"g_anchor/node_750": 1.0},
    }
    record.pop(identity_field)

    with pytest.raises(ValueError, match=identity_field):
        merge_task_records([record])


def test_merge_task_records_rejects_duplicate_identity():
    record = {
        "task": "easy",
        "record_index": 0,
        "pair_id": "pair-a",
        "metrics": {"g_anchor/node_750": 1.0},
    }

    with pytest.raises(ValueError, match="duplicate"):
        merge_task_records([record, dict(record)])


def test_merge_task_records_rejects_equivalent_numeric_identity():
    record = {
        "task": "easy",
        "record_index": 1,
        "pair_id": "pair-a",
        "metrics": {"g_anchor/node_750": 1.0},
    }
    equivalent_record = dict(record)
    equivalent_record["record_index"] = 1.0

    with pytest.raises(ValueError, match="duplicate"):
        merge_task_records([record, equivalent_record])
