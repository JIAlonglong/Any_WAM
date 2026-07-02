from distillation.patches import SafeMultiLatentLeRobotDataset


class TinySubDataset:
    repo_id = "/data/task_alpha"
    new_metas = [
        {"episode_index": 10, "tasks": ["alpha"], "start_frame": 0, "end_frame": 8},
        {"episode_index": 11, "tasks": ["beta"], "start_frame": 0, "end_frame": 8},
    ]

    def __len__(self):
        return len(self.new_metas)

    def __getitem__(self, idx):
        raise AssertionError("metadata lookup must not load samples")


def test_get_group_ids_uses_task_metadata_without_loading_samples():
    dataset = object.__new__(SafeMultiLatentLeRobotDataset)
    dataset._datasets = [TinySubDataset()]
    dataset.item_id_to_dataset_id = {0: 0, 1: 0}
    dataset.acc_dset_num = {0: 0}

    assert dataset.get_group_ids("task") == ["alpha", "beta"]
    assert dataset.get_group_ids("episode") == ["episode:10", "episode:11"]
    assert dataset.get_group_ids("dataset") == ["/data/task_alpha", "/data/task_alpha"]
