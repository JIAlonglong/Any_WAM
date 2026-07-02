# Stage2 Group-Balanced Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in distributed group-balanced sampler for Stage 2 so later KTO/priority experiments can rebalance task or episode coverage without changing the default Stage 2 path.

**Architecture:** Keep sampling logic in a new focused `distillation_flowmap/samplers.py` module. Add metadata accessors to the safe LeRobot wrapper so the trainer can build group ids from task or episode metadata. Wire the sampler through config flags only; default remains the existing `DistributedSampler`.

**Tech Stack:** Python, PyTorch `torch.utils.data.Sampler`, pytest, existing FlowMap Stage 2 trainer/configs.

---

### Task 1: Add Sampler Unit Tests

**Files:**
- Create: `distillation_flowmap/tests/test_group_balanced_sampler.py`
- Later create: `distillation_flowmap/samplers.py`

- [ ] **Step 1: Write the failing tests**

```python
from collections import Counter

from distillation_flowmap.samplers import (
    DistributedGroupBalancedSampler,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_group_balanced_sampler.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'distillation_flowmap.samplers'`.

- [ ] **Step 3: Implement minimal sampler module**

Create `distillation_flowmap/samplers.py` with:

```python
import math
from collections import defaultdict

import torch
from torch.utils.data import Sampler


def normalize_group_ids(group_ids, expected_length):
    out = [str(group_id) for group_id in group_ids]
    if len(out) != expected_length:
        raise ValueError(f"expected {expected_length} group ids, got {len(out)}")
    if not out:
        raise ValueError("group ids must be non-empty")
    return out


class DistributedGroupBalancedSampler(Sampler):
    def __init__(self, group_ids, num_replicas=1, rank=0, shuffle=True, seed=0, samples_per_group=None):
        self.group_ids = normalize_group_ids(group_ids, len(group_ids))
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        groups = defaultdict(list)
        for idx, group_id in enumerate(self.group_ids):
            groups[group_id].append(idx)
        self.groups = dict(sorted(groups.items()))
        self.samples_per_group = int(samples_per_group or max(len(v) for v in self.groups.values()))
        self.total_size = len(self.groups) * self.samples_per_group
        self.num_samples = int(math.ceil(self.total_size / self.num_replicas))

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = []
        for group_id, group_indices in self.groups.items():
            local = list(group_indices)
            if self.shuffle:
                perm = torch.randperm(len(local), generator=generator).tolist()
                local = [local[i] for i in perm]
            repeat = int(math.ceil(self.samples_per_group / len(local)))
            indices.extend((local * repeat)[: self.samples_per_group])
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=generator).tolist()
            indices = [indices[i] for i in perm]
        padding = self.num_samples * self.num_replicas - len(indices)
        if padding > 0:
            indices.extend(indices[:padding])
        return iter(indices[self.rank : self.num_samples * self.num_replicas : self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_group_balanced_sampler.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/samplers.py distillation_flowmap/tests/test_group_balanced_sampler.py
git commit -m "feat: add group-balanced distributed sampler"
```

### Task 2: Expose Dataset Group Metadata

**Files:**
- Modify: `distillation/patches.py`
- Test: `distillation_flowmap/tests/test_dataset_group_metadata.py`

- [ ] **Step 1: Write the failing test**

```python
from distillation.patches import SafeMultiLatentLeRobotDataset


class TinySubDataset:
    repo_id = "/data/task_alpha"
    new_metas = [
        {"episode_index": 10, "tasks": ["alpha"], "start_frame": 0, "end_frame": 8},
        {"episode_index": 11, "tasks": ["beta"], "start_frame": 0, "end_frame": 8},
    ]

    def __len__(self):
        return len(self.new_metas)


def test_get_group_ids_uses_task_metadata_without_loading_samples():
    dataset = object.__new__(SafeMultiLatentLeRobotDataset)
    dataset._datasets = [TinySubDataset()]
    dataset.item_id_to_dataset_id = {0: 0, 1: 0}
    dataset.acc_dset_num = {0: 0}

    assert dataset.get_group_ids("task") == ["alpha", "beta"]
    assert dataset.get_group_ids("episode") == ["episode:10", "episode:11"]
    assert dataset.get_group_ids("dataset") == ["/data/task_alpha", "/data/task_alpha"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_dataset_group_metadata.py -q`

Expected: FAIL because `get_group_ids` does not exist.

- [ ] **Step 3: Implement metadata accessors**

Add `get_sample_meta(idx)` and `get_group_ids(group_by)` to `SafeMultiLatentLeRobotDataset`. The method must read `new_metas` directly and must not call `__getitem__`.

- [ ] **Step 4: Run metadata tests**

Run: `/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_dataset_group_metadata.py -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add distillation/patches.py distillation_flowmap/tests/test_dataset_group_metadata.py
git commit -m "feat: expose latent dataset group metadata"
```

### Task 3: Wire Sampler Into Stage 2 Trainer

**Files:**
- Modify: `distillation_flowmap/flowmap_trainer.py`
- Modify: `distillation_flowmap/config_libero_fullfinetune_stage2_kto_paopd.py`
- Test: `distillation_flowmap/tests/test_group_balanced_sampler.py`

- [ ] **Step 1: Add failing construction test**

Extend `test_group_balanced_sampler.py` with a helper test for `build_stage2_sampler(dataset, config)` once the helper is introduced.

- [ ] **Step 2: Implement opt-in trainer helper**

Add `build_stage2_sampler(train_dataset, config)` in `flowmap_trainer.py`. It should return the existing `DistributedSampler` unless `config.stage2_sampler == "group_balanced"`. For the group-balanced path, call `train_dataset.get_group_ids(config.stage2_group_by)` and create `DistributedGroupBalancedSampler`.

- [ ] **Step 3: Add config flags**

In the KTO config only:

```python
cfg.stage2_sampler = os.environ.get("STAGE2_SAMPLER", "default").lower()
cfg.stage2_group_by = os.environ.get("STAGE2_GROUP_BY", "task").lower()
_samples_per_group = os.environ.get("STAGE2_SAMPLES_PER_GROUP")
cfg.stage2_samples_per_group = int(_samples_per_group) if _samples_per_group else None
```

- [ ] **Step 4: Run targeted tests and compile**

Run:
`/root/nas/junjie/conda_envs/any_wam/bin/python -m pytest distillation_flowmap/tests/test_group_balanced_sampler.py distillation_flowmap/tests/test_dataset_group_metadata.py distillation_flowmap/tests/test_kto_reweighting.py -q`

Run:
`/root/nas/junjie/conda_envs/any_wam/bin/python -m py_compile distillation_flowmap/samplers.py distillation_flowmap/flowmap_trainer.py distillation_flowmap/config_libero_fullfinetune_stage2_kto_paopd.py distillation/patches.py`

Expected: tests pass and py_compile exits 0.

- [ ] **Step 5: Commit**

```bash
git add distillation_flowmap/flowmap_trainer.py distillation_flowmap/config_libero_fullfinetune_stage2_kto_paopd.py distillation_flowmap/tests/test_group_balanced_sampler.py
git commit -m "feat: wire opt-in group-balanced stage2 sampler"
```

### Task 4: Short Two-GPU Smoke Experiment

**Files:**
- No code changes expected.

- [ ] **Step 1: Run default sampler short smoke**

Run 10-20 Stage 2 steps with the existing KTO variant and default sampler.

- [ ] **Step 2: Run group-balanced sampler short smoke**

Run the same command with `STAGE2_SAMPLER=group_balanced STAGE2_GROUP_BY=task`.

- [ ] **Step 3: Compare**

Compare step time, loss stability, light eval if enabled, and confirm no sampler crash under `torchrun` with 2 GPUs.

- [ ] **Step 4: Commit experiment notes if useful**

If notes are written, commit only the notes file. Do not commit checkpoints or logs.
