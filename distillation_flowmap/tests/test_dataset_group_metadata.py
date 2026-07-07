from distillation import patches
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


def test_repo_task_filter_matches_robotwin_task_prefixes():
    assert hasattr(patches, "_filter_repo_list_by_tasks")
    repos = [
        "/data/place_a2b_right-aloha-agilex_randomized_500-1000",
        "/data/place_a2b_left-aloha-agilex_randomized_500-1000",
        "/data/blocks_ranking_size",
    ]

    assert patches._filter_repo_list_by_tasks(repos, ["place_a2b_right", "blocks_ranking_size"]) == [
        repos[0],
        repos[2],
    ]


def test_limit_dataset_metas_keeps_first_episodes_and_samples():
    assert hasattr(patches, "_limit_dataset_metas")
    dataset = object.__new__(TinySubDataset)
    dataset.repo_id = "/data/task_alpha"
    dataset.new_metas = [
        {"episode_index": 10, "tasks": ["alpha"], "start_frame": 0, "end_frame": 8},
        {"episode_index": 10, "tasks": ["alpha"], "start_frame": 8, "end_frame": 16},
        {"episode_index": 11, "tasks": ["alpha"], "start_frame": 0, "end_frame": 8},
        {"episode_index": 12, "tasks": ["alpha"], "start_frame": 0, "end_frame": 8},
    ]

    patches._limit_dataset_metas(dataset, max_episodes=2, max_samples=2)

    assert [m["episode_index"] for m in dataset.new_metas] == [10, 10]
