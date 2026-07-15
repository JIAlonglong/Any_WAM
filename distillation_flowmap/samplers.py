import math
from collections import defaultdict

import torch
from torch.utils.data import DistributedSampler, Sampler


def normalize_group_ids(group_ids, expected_length):
    out = [str(group_id) for group_id in group_ids]
    if len(out) != expected_length:
        raise ValueError(f"expected {expected_length} group ids, got {len(out)}")
    if not out:
        raise ValueError("group ids must be non-empty")
    return out


def build_stage2_sampler(train_dataset, config):
    sampler_name = str(getattr(config, "stage2_sampler", "default")).lower()
    world_size = int(getattr(config, "world_size", 1))
    rank = int(getattr(config, "rank", 0))
    seed = int(getattr(config, "seed", 0))

    if sampler_name in ("default", "", "distributed"):
        if world_size <= 1:
            return None
        return DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )

    if sampler_name in ("group_balanced", "group-balanced", "balanced"):
        if not hasattr(train_dataset, "get_group_ids"):
            raise ValueError(
                "stage2_sampler='group_balanced' requires dataset.get_group_ids()"
            )
        group_by = str(getattr(config, "stage2_group_by", "task")).lower()
        group_ids = train_dataset.get_group_ids(group_by)
        samples_per_group = getattr(config, "stage2_samples_per_group", None)
        return DistributedGroupBalancedSampler(
            group_ids,
            num_replicas=max(1, world_size),
            rank=rank,
            shuffle=True,
            seed=seed,
            samples_per_group=samples_per_group,
        )

    raise ValueError(
        "stage2_sampler must be 'default' or 'group_balanced', "
        f"got {sampler_name!r}"
    )


class DistributedGroupBalancedSampler(Sampler):
    """Distributed sampler that draws an equal number of samples per group."""

    def __init__(
        self,
        group_ids,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=0,
        samples_per_group=None,
    ):
        self.group_ids = normalize_group_ids(group_ids, len(group_ids))
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        groups = defaultdict(list)
        for idx, group_id in enumerate(self.group_ids):
            groups[group_id].append(idx)
        self.groups = dict(sorted(groups.items()))
        if samples_per_group is None:
            self.samples_per_group = max(len(indices) for indices in self.groups.values())
        else:
            self.samples_per_group = int(samples_per_group)
        if self.samples_per_group <= 0:
            raise ValueError("samples_per_group must be positive")

        self.total_size = len(self.groups) * self.samples_per_group
        self.num_samples = int(math.ceil(self.total_size / self.num_replicas))

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        indices = []
        for _, group_indices in self.groups.items():
            local_indices = list(group_indices)
            if self.shuffle:
                order = torch.randperm(len(local_indices), generator=generator).tolist()
                local_indices = [local_indices[i] for i in order]

            repeats = int(math.ceil(self.samples_per_group / len(local_indices)))
            indices.extend((local_indices * repeats)[: self.samples_per_group])

        if self.shuffle:
            order = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in order]

        padded_size = self.num_samples * self.num_replicas
        padding = padded_size - len(indices)
        if padding > 0:
            indices.extend(indices[:padding])

        rank_indices = indices[self.rank:padded_size:self.num_replicas]
        return iter(rank_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
