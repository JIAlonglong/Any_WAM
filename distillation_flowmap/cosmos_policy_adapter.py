"""Cosmos Policy compatibility layer for FlowMap action distillation.

Cosmos Policy is not a Wan/Diffusers transformer directory, so the WanVA teacher
path cannot be reused directly. This module keeps checkpoint validation,
action-only fallback targets, and optional raw-observation official inference in
one narrow adapter.
"""

import atexit
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


COSMOS_POLICY_WEIGHT_NAMES = (
    "Cosmos-Policy-LIBERO-Predict2-2B.pt",
    "Cosmos-Policy-RoboCasa-Predict2-2B.pt",
)


def resolve_cosmos_policy_assets(path):
    """Validate a local Cosmos Policy checkpoint directory and return assets."""
    if path is None:
        raise FileNotFoundError("cosmos_policy teacher_model_path is required.")

    root = Path(os.path.abspath(os.path.expanduser(path)))
    if not root.is_dir():
        raise FileNotFoundError(f"Cosmos Policy checkpoint directory not found: {root}")

    weight_path = None
    for name in COSMOS_POLICY_WEIGHT_NAMES:
        candidate = root / name
        if candidate.is_file():
            weight_path = candidate
            break
    if weight_path is None:
        pt_files = sorted(root.glob("Cosmos-Policy-*.pt"))
        if pt_files:
            weight_path = pt_files[0]
    if weight_path is None:
        raise FileNotFoundError(
            f"Invalid Cosmos Policy checkpoint: expected one of {COSMOS_POLICY_WEIGHT_NAMES} in {root}"
        )

    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Invalid Cosmos Policy checkpoint: missing {config_path}")

    stats_candidates = sorted(root.glob("*_dataset_statistics.json"))
    if not stats_candidates:
        raise FileNotFoundError(
            f"Invalid Cosmos Policy checkpoint: missing *_dataset_statistics.json in {root}"
        )

    embedding_candidates = sorted(root.glob("*_t5_embeddings.pkl"))
    if not embedding_candidates:
        raise FileNotFoundError(
            f"Invalid Cosmos Policy checkpoint: missing *_t5_embeddings.pkl in {root}"
        )

    return {
        "root": str(root),
        "weight_path": str(weight_path),
        "config_path": str(config_path),
        "dataset_stats_path": str(stats_candidates[0]),
        "t5_embeddings_path": str(embedding_candidates[0]),
    }


def _as_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_task_list(value, batch_size):
    if isinstance(value, str):
        return [value] * batch_size
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if len(value) != batch_size:
            raise ValueError(f"raw_task has {len(value)} entries, expected batch size {batch_size}")
        return [str(v) for v in value]
    raise TypeError(f"raw_task must be a string or list of strings, got {type(value)!r}")


def libero_state_to_cosmos_proprio(state):
    """Convert LeRobot LIBERO state [xyz, rpy, gripper2] to official 9D proprio.

    Official LIBERO inference expects:
    concat(robot0_gripper_qpos[2], robot0_eef_pos[3], robot0_eef_quat[4]).
    """
    is_tensor = torch.is_tensor(state)
    arr = _as_numpy(state).astype(np.float32)
    if arr.shape[-1] != 8:
        raise ValueError(f"Expected LIBERO state last dim 8, got shape {arr.shape}")

    xyz = arr[..., :3]
    rpy = arr[..., 3:6]
    gripper = arr[..., 6:8]

    roll = rpy[..., 0] * 0.5
    pitch = rpy[..., 1] * 0.5
    yaw = rpy[..., 2] * 0.5
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    quat = np.stack(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        axis=-1,
    ).astype(np.float32)

    proprio = np.concatenate([gripper, xyz, quat], axis=-1).astype(np.float32)
    if is_tensor:
        return torch.from_numpy(proprio).to(device=state.device, dtype=state.dtype)
    return proprio


def cosmos_actions_to_flowmap_x0(
    actions,
    target_shape,
    q01,
    q99,
    inverse_used_action_channel_ids,
    device,
    dtype,
):
    """Map official Cosmos action chunks to FlowMap's normalized action x0."""
    if len(target_shape) != 5:
        raise ValueError(f"Expected target shape [B,C,F,N,1], got {target_shape}")
    batch_size, channels, frames, per_frame, width = target_shape
    if width != 1:
        raise ValueError(f"Expected action width 1, got {width}")

    actions = torch.as_tensor(actions, device=device, dtype=torch.float32)
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    if actions.ndim != 3:
        raise ValueError(f"Expected actions [B,T,D], got {tuple(actions.shape)}")
    if actions.shape[0] != batch_size:
        raise ValueError(f"Action batch {actions.shape[0]} does not match target batch {batch_size}")

    action_dim = actions.shape[-1]
    pad_col = torch.zeros(*actions.shape[:-1], 1, device=device, dtype=actions.dtype)
    padded = torch.cat([actions, pad_col], dim=-1)
    inverse = torch.as_tensor(inverse_used_action_channel_ids, device=device, dtype=torch.long)
    if inverse.numel() != channels:
        raise ValueError(f"inverse_used_action_channel_ids has {inverse.numel()} entries, expected {channels}")
    if int(inverse.max().item()) > action_dim:
        raise ValueError(
            f"inverse_used_action_channel_ids references {int(inverse.max().item())}, "
            f"but raw action dim is {action_dim}"
        )

    aligned = padded.index_select(dim=-1, index=inverse)
    q01 = torch.as_tensor(q01, device=device, dtype=actions.dtype).view(1, 1, channels)
    q99 = torch.as_tensor(q99, device=device, dtype=actions.dtype).view(1, 1, channels)
    aligned = (aligned - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    valid_channel_mask = (inverse < action_dim).to(dtype=aligned.dtype).view(1, 1, channels)
    aligned = aligned * valid_channel_mask

    flat = torch.zeros(batch_size, frames * per_frame, channels, device=device, dtype=actions.dtype)
    num_tokens = min(aligned.shape[1], flat.shape[1])
    flat[:, :num_tokens] = aligned[:, :num_tokens]
    x0 = flat.reshape(batch_size, frames, per_frame, channels).permute(0, 3, 1, 2).unsqueeze(-1)
    return x0.to(dtype=dtype)


def compute_masked_action_stats(teacher_x0, target_x0, mask=None):
    """Return masked action x0 diagnostics normalized over valid tokens/channels."""
    teacher = teacher_x0.detach().float()
    target = target_x0.detach().float()
    if teacher.shape != target.shape:
        raise ValueError(
            f"teacher_x0 shape {tuple(teacher.shape)} does not match target_x0 {tuple(target.shape)}"
        )

    if mask is None:
        mask = torch.ones_like(target[:, :1])
    else:
        mask = mask.detach().to(device=target.device, dtype=torch.float32)
    denom = (mask.sum() * teacher.shape[1]).clamp(min=1)
    diff = (teacher - target) * mask
    return {
        "mse": (diff.square().sum() / denom).to(device=teacher_x0.device),
        "l1": (diff.abs().sum() / denom).to(device=teacher_x0.device),
        "teacher_abs_mean": ((teacher.abs() * mask).sum() / denom).to(device=teacher_x0.device),
        "target_abs_mean": ((target.abs() * mask).sum() / denom).to(device=teacher_x0.device),
    }


class CosmosPolicyActionTeacher:
    """Action teacher with latent fallback and optional official raw inference."""

    def __init__(self, checkpoint_dir, dtype=torch.bfloat16, device="cpu", config=None):
        self.assets = resolve_cosmos_policy_assets(checkpoint_dir)
        self.dtype = dtype
        self.device = torch.device(device)
        self.config = config
        with open(self.assets["config_path"], "r") as f:
            self.policy_config = json.load(f)
        with open(self.assets["dataset_stats_path"], "r") as f:
            self.dataset_stats = json.load(f)
        self._state_metadata = None

        self.raw_inference_enabled = bool(
            getattr(config, "cosmos_policy_use_raw_inference", False)
        ) if config is not None else False
        self.raw_inference_mode = str(
            getattr(config, "cosmos_policy_inference_mode", "subprocess")
        ).lower() if config is not None else "subprocess"
        self.cosmos_repo_path = getattr(
            config,
            "cosmos_policy_repo",
            os.environ.get("COSMOS_PREDICT2_REPO", "/root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5"),
        ) if config is not None else os.environ.get(
            "COSMOS_PREDICT2_REPO", "/root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5"
        )
        self.cosmos_python = getattr(
            config,
            "cosmos_policy_python",
            os.environ.get("COSMOS_POLICY_PYTHON", "/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/python"),
        ) if config is not None else os.environ.get(
            "COSMOS_POLICY_PYTHON", "/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/python"
        )
        self.cosmos_config_name = getattr(
            config, "cosmos_policy_config_name", "cosmos_predict2_2b_480p_libero__inference_only"
        ) if config is not None else "cosmos_predict2_2b_480p_libero__inference_only"
        default_config_file = "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py"
        self.cosmos_config_file = getattr(
            config, "cosmos_policy_config_file", default_config_file
        ) if config is not None else default_config_file
        if os.path.isabs(self.cosmos_config_file) and self.cosmos_repo_path:
            try:
                self.cosmos_config_file = os.path.relpath(self.cosmos_config_file, self.cosmos_repo_path)
            except ValueError:
                pass
        self.cosmos_extra_pythonpath = getattr(
            config,
            "cosmos_policy_extra_pythonpath",
            os.environ.get(
                "COSMOS_POLICY_EXTRA_PYTHONPATH",
                "/root/nas/junjie/conda_envs/any_wam/lib/python3.10/site-packages",
            ),
        ) if config is not None else os.environ.get(
            "COSMOS_POLICY_EXTRA_PYTHONPATH",
            "/root/nas/junjie/conda_envs/any_wam/lib/python3.10/site-packages",
        )
        self.cosmos_local_model_dir = getattr(
            config,
            "cosmos_policy_local_model_dir",
            os.environ.get(
                "COSMOS_PREDICT25_LOCAL_MODEL_DIR",
                "/root/nas/junjie/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World",
            ),
        ) if config is not None else os.environ.get(
            "COSMOS_PREDICT25_LOCAL_MODEL_DIR",
            "/root/nas/junjie/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World",
        )
        self.num_denoising_steps_action = int(
            getattr(config, "cosmos_policy_num_denoising_steps_action", 5)
        ) if config is not None else 5
        self.raw_seed = int(getattr(config, "cosmos_policy_seed", 1)) if config is not None else 1

        self._official_model = None
        self._official_dataset_stats = None
        self._official_get_action = None
        self._official_cfg = None
        self._raw_action_provider = None
        self._raw_worker = None
        self._raw_worker_tmpdir = None

    @property
    def weight_path(self):
        return self.assets["weight_path"]

    def load_state_metadata(self):
        """Load the checkpoint state dict once to prove the file is readable."""
        if self._state_metadata is None:
            state = torch.load(self.weight_path, map_location="cpu")
            if not isinstance(state, dict):
                raise ValueError(f"Unexpected Cosmos Policy checkpoint type: {type(state)!r}")
            keys = state.get("model", state)
            if not isinstance(keys, dict):
                raise ValueError("Cosmos Policy checkpoint does not contain a state dict.")
            sample_keys = list(keys.keys())[:8]
            has_net_keys = any(key.startswith("net.") for key in keys.keys())
            if not has_net_keys:
                raise ValueError("Cosmos Policy checkpoint does not contain expected 'net.' keys.")
            self._state_metadata = {
                "num_keys": len(keys),
                "sample_keys": sample_keys,
            }
            del state
        return self._state_metadata

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def requires_grad_(self, requires_grad=False):
        return self

    def parameters(self):
        return iter(())

    def close(self):
        if self._raw_worker is not None:
            try:
                self._raw_worker.stdin.close()
            except Exception:
                pass
            try:
                self._raw_worker.terminate()
            except Exception:
                pass
            self._raw_worker = None

    def action_target_tokens(self, action_dict):
        """Return fallback action teacher tokens with WanVA action sequence shape."""
        target = action_dict.get("targets")
        if target is None:
            target = action_dict["latent"]
        tokens = target.permute(0, 2, 3, 4, 1).reshape(
            target.shape[0],
            target.shape[2] * target.shape[3] * target.shape[4],
            target.shape[1],
        )
        return tokens.to(dtype=self.dtype, device=target.device)

    def _raw_batch_to_numpy(self, raw_batch):
        required = ("raw_primary_image", "raw_wrist_image", "raw_proprio", "raw_task")
        missing = [key for key in required if key not in raw_batch]
        if missing:
            raise KeyError(
                "Raw Cosmos Policy inference requires batch fields "
                f"{required}; missing {missing}. Set cfg.return_raw_observation=True."
            )
        primary = _as_numpy(raw_batch["raw_primary_image"])
        wrist = _as_numpy(raw_batch["raw_wrist_image"])
        proprio = _as_numpy(raw_batch["raw_proprio"]).astype(np.float32)
        if primary.ndim == 3:
            primary = primary[None]
        if wrist.ndim == 3:
            wrist = wrist[None]
        if proprio.ndim == 1:
            proprio = proprio[None]
        if primary.shape[0] != wrist.shape[0] or primary.shape[0] != proprio.shape[0]:
            raise ValueError(
                "Raw image/proprio batch sizes differ: "
                f"primary={primary.shape}, wrist={wrist.shape}, proprio={proprio.shape}"
            )
        if primary.shape[-1] != 3 or wrist.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB images, got primary={primary.shape}, wrist={wrist.shape}")
        if primary.dtype != np.uint8:
            primary = primary.astype(np.uint8)
        if wrist.dtype != np.uint8:
            wrist = wrist.astype(np.uint8)
        tasks = _as_task_list(raw_batch["raw_task"], primary.shape[0])
        return primary, wrist, proprio, tasks

    def _make_official_cfg(self):
        return SimpleNamespace(
            suite="libero",
            model_family="cosmos",
            config=self.cosmos_config_name,
            ckpt_path=self.assets["weight_path"],
            planning_model_config_name="",
            planning_model_ckpt_path="",
            config_file=self.cosmos_config_file,
            use_third_person_image=True,
            num_third_person_images=1,
            use_wrist_image=True,
            num_wrist_images=1,
            use_proprio=True,
            flip_images=bool(getattr(self.config, "cosmos_policy_flip_images", True))
            if self.config is not None else True,
            use_variance_scale=bool(getattr(self.config, "cosmos_policy_use_variance_scale", False))
            if self.config is not None else False,
            use_jpeg_compression=bool(getattr(self.config, "cosmos_policy_use_jpeg_compression", True))
            if self.config is not None else True,
            ar_future_prediction=False,
            ar_value_prediction=False,
            ar_qvalue_prediction=False,
            num_denoising_steps_action=self.num_denoising_steps_action,
            num_denoising_steps_future_state=1,
            num_denoising_steps_value=1,
            shift=int(getattr(self.config, "cosmos_policy_shift", 5)) if self.config is not None else 5,
            unnormalize_actions=True,
            normalize_proprio=True,
            dataset_stats_path=self.assets["dataset_stats_path"],
            t5_text_embeddings_path=self.assets["t5_embeddings_path"],
            text_embeddings_kind="t5",
            trained_with_image_aug=True,
            chunk_size=int(getattr(self.config, "cosmos_policy_chunk_size", 16)) if self.config is not None else 16,
            num_open_loop_steps=16,
            deterministic=True,
            randomize_seed=False,
            seed=self.raw_seed,
        )

    def _ensure_official_inprocess(self):
        if self._official_model is not None:
            return
        if self.cosmos_repo_path and self.cosmos_repo_path not in sys.path:
            sys.path.insert(0, self.cosmos_repo_path)
        try:
            from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
                get_action,
                get_model,
                init_t5_text_embeddings_cache,
                load_dataset_stats,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to import official Cosmos Policy inference in this Python environment. "
                "Use COSMOS_POLICY_INFERENCE_MODE=subprocess with COSMOS_POLICY_PYTHON pointing "
                "to the Cosmos env, or install the Cosmos CUDA extra into the training env."
            ) from exc
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        cfg = self._make_official_cfg()
        init_t5_text_embeddings_cache(
            cfg.t5_text_embeddings_path,
            worker_id=self.device.index or 0,
            embeddings_kind=cfg.text_embeddings_kind,
        )
        self._official_dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
        self._official_model, _ = get_model(cfg)
        self._official_get_action = get_action
        self._official_cfg = cfg

    def _predict_raw_actions_inprocess(self, raw_batch):
        return self._predict_raw_action_result_inprocess(raw_batch, include_future=False)["actions"]

    def _coerce_raw_action_result(self, result):
        if isinstance(result, dict):
            if "actions" not in result:
                raise KeyError("Raw action result dictionary must contain an 'actions' entry.")
            coerced = dict(result)
            coerced["actions"] = torch.as_tensor(coerced["actions"], dtype=torch.float32)
            return coerced
        return {"actions": torch.as_tensor(result, dtype=torch.float32)}

    @staticmethod
    def _future_predictions_from_official_result(result):
        predictions = result.get("future_image_predictions")
        if predictions is None:
            return None
        return {
            key: np.asarray(value, dtype=np.uint8)
            for key, value in predictions.items()
            if value is not None
        }

    def _predict_raw_action_result_inprocess(self, raw_batch, include_future=False):
        primary, wrist, proprio, tasks = self._raw_batch_to_numpy(raw_batch)
        self._ensure_official_inprocess()
        actions = []
        future_predictions = []
        value_predictions = []
        with torch.no_grad():
            for idx, task in enumerate(tasks):
                obs = {
                    "primary_image": primary[idx],
                    "wrist_image": wrist[idx],
                    "proprio": proprio[idx],
                }
                result = self._official_get_action(
                    self._official_cfg,
                    self._official_model,
                    self._official_dataset_stats,
                    obs,
                    task,
                    seed=self.raw_seed,
                    randomize_seed=False,
                    num_denoising_steps_action=self.num_denoising_steps_action,
                    generate_future_state_and_value_in_parallel=bool(include_future),
                    worker_id=self.device.index or 0,
                    batch_size=1,
                )
                actions.append(np.asarray(result["actions"], dtype=np.float32))
                if include_future:
                    future_predictions.append(self._future_predictions_from_official_result(result))
                    value_predictions.append(result.get("value_prediction"))
        output = {"actions": torch.from_numpy(np.stack(actions, axis=0))}
        if include_future:
            output["future_image_predictions"] = future_predictions
            output["value_prediction"] = value_predictions
        return output

    def _ensure_raw_worker(self):
        if self._raw_worker is not None and self._raw_worker.poll() is None:
            return
        worker_script = Path(__file__).with_name("cosmos_policy_raw_worker.py")
        if not worker_script.is_file():
            raise FileNotFoundError(f"Missing Cosmos raw worker script: {worker_script}")
        if not os.path.isfile(self.cosmos_python):
            raise FileNotFoundError(
                f"Cosmos Policy python not found: {self.cosmos_python}. "
                "Set COSMOS_POLICY_PYTHON to the official Cosmos env python."
            )
        env = os.environ.copy()
        if self.cosmos_repo_path:
            env["PYTHONPATH"] = self.cosmos_repo_path + os.pathsep + env.get("PYTHONPATH", "")
        if self.device.type == "cuda" and self.device.index is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.device.index)
        if self.cosmos_local_model_dir:
            env["COSMOS_PREDICT25_LOCAL_MODEL_DIR"] = self.cosmos_local_model_dir
        cmd = [
            self.cosmos_python,
            str(worker_script),
            "--checkpoint-dir", self.assets["weight_path"],
            "--repo", self.cosmos_repo_path,
            "--config-name", self.cosmos_config_name,
            "--config-file", self.cosmos_config_file,
            "--dataset-stats-path", self.assets["dataset_stats_path"],
            "--t5-embeddings-path", self.assets["t5_embeddings_path"],
            "--num-denoising-steps-action", str(self.num_denoising_steps_action),
            "--seed", str(self.raw_seed),
        ]
        if self.cosmos_extra_pythonpath:
            cmd.extend(["--extra-pythonpath", self.cosmos_extra_pythonpath])
        self._raw_worker_tmpdir = tempfile.mkdtemp(prefix="cosmos_policy_raw_")
        self._raw_worker = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        atexit.register(self.close)

    def _predict_raw_actions_subprocess(self, raw_batch):
        return self._predict_raw_action_result_subprocess(raw_batch, include_future=False)["actions"]

    def _predict_raw_action_result_subprocess(self, raw_batch, include_future=False):
        primary, wrist, proprio, tasks = self._raw_batch_to_numpy(raw_batch)
        self._ensure_raw_worker()
        req_dir = self._raw_worker_tmpdir or tempfile.mkdtemp(prefix="cosmos_policy_raw_")
        fd, npz_path = tempfile.mkstemp(prefix="request_", suffix=".npz", dir=req_dir)
        os.close(fd)
        actions_path = npz_path.replace("request_", "actions_")
        np.savez_compressed(npz_path, primary_image=primary, wrist_image=wrist, proprio=proprio)
        payload = {
            "npz_path": npz_path,
            "actions_path": actions_path,
            "tasks": tasks,
            "seed": self.raw_seed,
            "include_future_predictions": bool(include_future),
        }
        try:
            self._raw_worker.stdin.write(json.dumps(payload) + "\n")
            self._raw_worker.stdin.flush()
            line = self._raw_worker.stdout.readline()
        except BrokenPipeError as exc:
            raise RuntimeError("Cosmos Policy raw worker exited before responding.") from exc
        if not line:
            try:
                code = self._raw_worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                code = self._raw_worker.poll()
            raise RuntimeError(f"Cosmos Policy raw worker produced no response; exit code={code}")
        response = json.loads(line)
        if not response.get("ok", False):
            raise RuntimeError("Cosmos Policy raw worker failed:\n" + response.get("error", "unknown error"))
        with np.load(response["actions_path"]) as data:
            actions = torch.from_numpy(data["actions"].astype(np.float32))
            result = {"actions": actions}
            if include_future and "future_prediction_keys" in data.files:
                keys = [str(key) for key in data["future_prediction_keys"].tolist()]
                futures = [dict() for _ in range(actions.shape[0])]
                for key in keys:
                    values = data[f"future_{key}"].astype(np.uint8)
                    for batch_idx in range(values.shape[0]):
                        futures[batch_idx][key] = values[batch_idx]
                result["future_image_predictions"] = futures
                if "value_prediction" in data.files:
                    result["value_prediction"] = [
                        float(value) for value in data["value_prediction"].astype(np.float32).tolist()
                    ]
        for path in (npz_path, response["actions_path"]):
            try:
                os.remove(path)
            except OSError:
                pass
        return result

    def predict_raw_actions(self, raw_batch):
        return self.predict_raw_action_result(raw_batch, include_future=False)["actions"]

    def predict_raw_action_result(self, raw_batch, include_future=False):
        if self._raw_action_provider is not None:
            return self._coerce_raw_action_result(self._raw_action_provider(raw_batch))
        if self.raw_inference_mode == "inprocess":
            return self._predict_raw_action_result_inprocess(raw_batch, include_future=include_future)
        if self.raw_inference_mode == "subprocess":
            return self._predict_raw_action_result_subprocess(raw_batch, include_future=include_future)
        raise ValueError(
            f"Unsupported cosmos_policy_inference_mode={self.raw_inference_mode!r}; "
            "expected 'subprocess' or 'inprocess'."
        )

    def action_target_x0(self, action_dict, raw_batch=None):
        """Return FlowMap-normalized action x0 from official raw Cosmos inference."""
        if not self.raw_inference_enabled:
            return None
        if raw_batch is None:
            raise ValueError("raw_batch is required when cosmos_policy_use_raw_inference=True")
        target = action_dict["latent"]
        raw_actions = self.predict_raw_actions(raw_batch)
        if self.config is None:
            raise ValueError("config is required to map raw Cosmos actions to FlowMap action channels")
        return cosmos_actions_to_flowmap_x0(
            raw_actions,
            target_shape=tuple(target.shape),
            q01=self.config.norm_stat["q01"],
            q99=self.config.norm_stat["q99"],
            inverse_used_action_channel_ids=self.config.inverse_used_action_channel_ids,
            device=target.device,
            dtype=target.dtype,
        )

    def __call__(self, input_dict, train_mode=True, **_kwargs):
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        video_target = latent_dict.get("targets")
        if video_target is None:
            video_target = latent_dict["noisy_latents"]
        patch_t, patch_h, patch_w = (1, 2, 2)
        bsz, channels, frames, height, width = video_target.shape
        video_tokens = video_target.reshape(
            bsz,
            channels,
            frames // patch_t,
            patch_t,
            height // patch_h,
            patch_h,
            width // patch_w,
            patch_w,
        )
        video_tokens = video_tokens.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
            bsz,
            (frames // patch_t) * (height // patch_h) * (width // patch_w),
            channels * patch_t * patch_h * patch_w,
        )
        return video_tokens.to(dtype=self.dtype), self.action_target_tokens(action_dict)
