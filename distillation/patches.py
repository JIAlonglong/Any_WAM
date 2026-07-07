"""
LCM 蒸馏的兼容性补丁。

包含两个关键组件：
  1. install_flash_attn_stub(): 在导入 wan_va 模块之前调用
     安装假的 flash_attn 模块，防止 model.py 导入崩溃

  2. SafeMultiLatentLeRobotDataset: MultiLatentLeRobotDataset 的安全替代
     跳过加载失败的子数据集，而不是直接崩溃

为什么需要这些补丁：
  - flash_attn 可能未安装在蒸馏环境中，但 model.py 会尝试导入它
  - 数据集可能不完整（下载中断、文件缺失），需要优雅处理
"""

import importlib.machinery
import os
import sys
import types


def _split_config_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.replace(";", ",").split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def _positive_int_or_none(value):
    if value in (None, "", 0, "0"):
        return None
    value = int(value)
    return value if value > 0 else None


def _repo_task_name(repo_id):
    name = os.path.basename(str(repo_id).rstrip("/"))
    return name.split("-", 1)[0]


def _repo_matches_task(repo_id, task_names):
    basename = os.path.basename(str(repo_id).rstrip("/")).lower()
    canonical = basename.split("-", 1)[0]
    for task in task_names:
        task = str(task).strip().lower()
        if not task:
            continue
        if task == basename or task == canonical or basename.startswith(f"{task}-"):
            return True
    return False


def _filter_repo_list_by_tasks(repo_list, task_filter):
    task_names = _split_config_list(task_filter)
    if not task_names:
        return list(repo_list)
    return [repo_id for repo_id in repo_list if _repo_matches_task(repo_id, task_names)]


def _limit_dataset_metas(dataset, max_episodes=None, max_samples=None):
    metas = getattr(dataset, "new_metas", None)
    if metas is None:
        return 0, 0

    max_episodes = _positive_int_or_none(max_episodes)
    max_samples = _positive_int_or_none(max_samples)
    if max_episodes is None and max_samples is None:
        return len(metas), len(metas)

    selected = []
    seen_episodes = []
    seen_set = set()
    for meta in metas:
        episode_index = meta.get("episode_index")
        if max_episodes is not None and episode_index not in seen_set:
            if len(seen_episodes) >= max_episodes:
                continue
            seen_episodes.append(episode_index)
            seen_set.add(episode_index)
        selected.append(meta)
        if max_samples is not None and len(selected) >= max_samples:
            break

    dataset.new_metas = selected
    return len(metas), len(selected)


def install_flash_attn_stub():
    """
    安装 flash_attn 的存根模块（如果真实模块不可用）。

    工作原理：
      - 检查 flash_attn_interface 和 flash_attn 是否已导入
      - 如果未导入，尝试导入真实模块
      - 如果导入失败，创建一个假模块并放入 sys.modules
      - 假模块包含必要的属性（flash_attn_func 等），但值为 None
      - 这样 model.py 的导入不会崩溃，但会回退到 torch SDPA

    为什么需要这个：
      - model.py 中有 try/except 导入 flash_attn
      - 但如果 flash_attn 完全未安装，导入会失败
      - 这个存根确保导入总是成功，只是功能降级
    """
    for mod_name in ("flash_attn_interface", "flash_attn"):
        if mod_name in sys.modules:
            continue
        try:
            __import__(mod_name)
        except ImportError:
            # 创建存根模块
            stub = types.ModuleType(mod_name)
            stub.__spec__ = importlib.machinery.ModuleSpec(mod_name, None)
            stub.__version__ = "0.0.0"
            stub.flash_attn_func = None
            stub.flash_attn_varlen_func = None
            sys.modules[mod_name] = stub
            pass  # stub installed silently; torch SDPA will be used


class SafeMultiLatentLeRobotDataset:
    """
    MultiLatentLeRobotDataset 的安全版本。

    与原始版本的区别：
      - 原始版本：任何子数据集加载失败都会崩溃
      - 安全版本：跳过失败的子数据集，打印警告，继续加载其他

    适用场景：
      - 数据集下载不完整
      - 部分 parquet 文件损坏
      - 磁盘空间不足导致部分文件缺失

    数据集结构：
      - 每个子数据集是一个独立的 LeRobot 数据集
      - 通过 info.json 文件识别子数据集
      - 所有子数据集共享相同的配置
    """

    def __init__(self, config, num_init_worker=128):
        """
        初始化安全数据集。

        参数:
            config: 配置对象，包含 dataset_path 等参数
            num_init_worker: 初始化时的 worker 数量（未使用，保留接口兼容）

        初始化流程：
          1. 递归查找所有 info.json 文件
          2. 提取每个子数据集的路径
          3. 尝试加载每个子数据集，失败则跳过
          4. 构建全局索引映射
        """
        from pathlib import Path
        from dataset.lerobot_latent_dataset import (
            recursive_find_file,
            LatentLeRobotDataset,
        )

        # 递归查找所有 info.json 文件，确定子数据集路径
        repo_list = recursive_find_file(config.dataset_path, "info.json")
        repo_list = [v.split("/meta/info.json")[0] for v in repo_list]
        total_discovered = len(repo_list)

        task_filter = _split_config_list(getattr(config, "dataset_task_filter", None))
        repo_list = _filter_repo_list_by_tasks(repo_list, task_filter)
        if task_filter:
            matched_names = ", ".join(_repo_task_name(v) for v in repo_list)
            print(
                "Dataset task filter matched "
                f"{len(repo_list)}/{total_discovered} sub-datasets: {matched_names}"
            )
        if not repo_list:
            raise RuntimeError(
                "No sub-datasets matched dataset_task_filter="
                f"{','.join(task_filter)} under {config.dataset_path}"
            )

        max_episodes = _positive_int_or_none(
            getattr(config, "dataset_max_episodes_per_task", None)
        )
        max_samples = _positive_int_or_none(
            getattr(config, "dataset_max_samples_per_task", None)
        )
        defer_cache = bool(getattr(config, "cache_dataset_in_memory", False)) and (
            max_episodes is not None or max_samples is not None
        )

        self._datasets = []
        skipped = []
        for repo_id in repo_list:
            old_cache = getattr(config, "cache_dataset_in_memory", False)
            try:
                if defer_cache:
                    config.cache_dataset_in_memory = False
                ds = LatentLeRobotDataset(repo_id=repo_id, config=config)
                before, after = _limit_dataset_metas(ds, max_episodes, max_samples)
                if after == 0:
                    raise RuntimeError(
                        "Dataset filter left no samples in "
                        f"{os.path.basename(repo_id)}"
                    )
                if after != before:
                    print(
                        "Limited "
                        f"{os.path.basename(repo_id)} samples: {before} -> {after}"
                    )
                if defer_cache:
                    ds.cache_in_memory = True
                    ds._memory_cache = {}
                    ds._preload_dataset()
                self._datasets.append(ds)
            except Exception as e:
                skipped.append((repo_id, e))
            finally:
                if defer_cache:
                    config.cache_dataset_in_memory = old_cache

        total = len(repo_list)
        loaded = len(self._datasets)
        print(f"Loaded {loaded}/{total} sub-datasets successfully")
        if skipped:
            allow_partial = bool(getattr(config, "allow_partial_datasets", False))
            details = "\n".join(
                f"  - {os.path.basename(repo_id)}: {err}" for repo_id, err in skipped
            )
            if not allow_partial:
                raise RuntimeError(
                    "Incomplete sub-datasets detected.\n"
                    "Set config.allow_partial_datasets=True to ignore them.\n"
                    f"{details}"
                )
            print("WARNING: allow_partial_datasets=True, skipping incomplete datasets:\n" + details)
        if loaded == 0:
            raise RuntimeError(
                "No valid sub-datasets found. Dataset download may be incomplete."
            )

        # 构建全局索引到子数据集的映射
        self.item_id_to_dataset_id, self.acc_dset_num = self._get_item_id_to_dataset_id()

    def __len__(self):
        """返回所有子数据集的总样本数。"""
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        """
        构建全局索引到子数据集 ID 的映射。

        返回:
            item_id_to_dataset_id: {全局索引: 子数据集 ID}
            acc_dset_num: {子数据集 ID: 累积样本数（用于计算局部索引）}

        映射原理：
          - 全局索引是连续的整数 [0, 1, 2, ...]
          - 每个子数据集占据一段连续的索引范围
          - 通过累积样本数确定每个子数据集的起始索引
        """
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def _resolve_dataset_index(self, idx):
        assert idx < len(self)
        dset_id = self.item_id_to_dataset_id[idx]
        local_idx = idx - self.acc_dset_num[dset_id]
        return dset_id, self._datasets[dset_id], local_idx

    def get_sample_meta(self, idx):
        dset_id, cur_dset, local_idx = self._resolve_dataset_index(idx)
        metas = getattr(cur_dset, "new_metas", None)
        if metas is None:
            meta = {}
        else:
            meta = dict(metas[local_idx])
        meta["dataset_id"] = dset_id
        meta["local_index"] = local_idx
        meta["repo_id"] = str(getattr(cur_dset, "repo_id", dset_id))
        return meta

    def get_group_ids(self, group_by="task"):
        group_by = str(group_by).lower()
        out = []
        for idx in range(len(self)):
            meta = self.get_sample_meta(idx)
            if group_by in ("task", "tasks"):
                tasks = meta.get("tasks") or meta.get("task") or meta.get("action_text")
                if isinstance(tasks, (list, tuple)):
                    group_id = tasks[0] if tasks else "task:unknown"
                else:
                    group_id = tasks or "task:unknown"
            elif group_by == "episode":
                group_id = f"episode:{meta.get('episode_index', 'unknown')}"
            elif group_by == "dataset":
                group_id = meta.get("repo_id", f"dataset:{meta.get('dataset_id', 'unknown')}")
            else:
                raise ValueError(
                    "group_by must be one of 'task', 'episode', or 'dataset', "
                    f"got {group_by!r}"
                )
            out.append(str(group_id))
        return out

    def __getitem__(self, idx):
        """
        通过全局索引获取样本。

        参数:
            idx: 全局样本索引

        返回:
            对应的样本数据

        查找过程：
          1. 通过 item_id_to_dataset_id 找到子数据集 ID
            2. 计算局部索引 = 全局索引 - 子数据集起始索引
            3. 从子数据集中获取样本
        """
        _, cur_dset, local_idx = self._resolve_dataset_index(idx)
        return cur_dset[local_idx]
