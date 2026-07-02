from collections import Counter
from types import SimpleNamespace

from torch.utils.data import DistributedSampler

from distillation_flowmap.samplers import (
    DistributedGroupBalancedSampler,
    build_stage2_sampler,
    normalize_group_ids,
)


def test_normalize_group_ids_rejects_length_mismatch():
    try:
        normalize_group_ids(["a"], expected_length=2)
    except ValueError as exc:
        assert "expected 2 group ids" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_group_balanced_sampler_balances_imbalanced_groups_on_one_rank():
    group_ids = ["easy"] * 8 + ["hard"] * 2
    sampler = DistributedGroupBalancedSampler(
        group_ids,
        num_replicas=1,
        rank=0,
        shuffle=False,
        seed=123,
        samples_per_group=4,
    )

    sampled_groups = [group_ids[i] for i in list(iter(sampler))]

    assert len(sampled_groups) == 8
    assert Counter(sampled_groups) == {"easy": 4, "hard": 4}


def test_group_balanced_sampler_splits_work_across_ranks_without_overlap():
    group_ids = ["a"] * 4 + ["b"] * 4 + ["c"] * 4
    rank0 = DistributedGroupBalancedSampler(
        group_ids,
        num_replicas=2,
        rank=0,
        shuffle=False,
        seed=7,
        samples_per_group=2,
    )
    rank1 = DistributedGroupBalancedSampler(
        group_ids,
        num_replicas=2,
        rank=1,
        shuffle=False,
        seed=7,
        samples_per_group=2,
    )

    ids0 = list(iter(rank0))
    ids1 = list(iter(rank1))

    assert len(ids0) == len(ids1) == 3
    assert set(ids0).isdisjoint(ids1)
    assert Counter(group_ids[i] for i in ids0 + ids1) == {"a": 2, "b": 2, "c": 2}


class TinyGroupedDataset:
    def __init__(self):
        self.group_by = None

    def __len__(self):
        return 4

    def get_group_ids(self, group_by):
        self.group_by = group_by
        return ["a", "a", "b", "b"]


def test_build_stage2_sampler_keeps_default_distributed_sampler():
    dataset = TinyGroupedDataset()
    config = SimpleNamespace(
        stage2_sampler="default",
        world_size=2,
        rank=1,
        seed=99,
    )

    sampler = build_stage2_sampler(dataset, config)

    assert isinstance(sampler, DistributedSampler)
    assert dataset.group_by is None


def test_build_stage2_sampler_uses_group_balanced_when_enabled():
    dataset = TinyGroupedDataset()
    config = SimpleNamespace(
        stage2_sampler="group_balanced",
        stage2_group_by="task",
        stage2_samples_per_group=1,
        world_size=2,
        rank=0,
        seed=99,
    )

    sampler = build_stage2_sampler(dataset, config)

    assert isinstance(sampler, DistributedGroupBalancedSampler)
    assert dataset.group_by == "task"
    assert sampler.samples_per_group == 1
