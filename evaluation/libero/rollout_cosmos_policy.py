#!/usr/bin/env python3
"""Roll out official Cosmos Policy in LIBERO and save environment videos."""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "wan_va"))

from distillation_flowmap.cosmos_policy_adapter import (  # noqa: E402
    CosmosPolicyActionTeacher,
    resolve_cosmos_policy_assets,
)

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def save_video(real_obs_list, save_path, fps=15):
    if not real_obs_list:
        raise RuntimeError("No observation frames were collected; cannot save video.")

    names = ["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
    first = real_obs_list[0]
    base_h, base_w = first[names[0]].shape[:2]
    target_size = (base_w, base_h)
    frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in names]).astype(np.uint8)
        for obs in real_obs_list
    ]
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(save_path), frames, fps=fps)


def official_metainfo_path(cosmos_repo, libero_benchmark):
    return str(
        Path(cosmos_repo)
        / "cosmos_predict2/_src/predict2/cosmos_policy/experiments/robot/libero"
        / f"{libero_benchmark}_metainfo.json"
    )


def apply_eval_defaults(args):
    official = bool(args.official_libero_eval)
    if args.seed is None:
        args.seed = 195 if official else 1
    if args.env_seed is None and official:
        args.env_seed = 0
    if args.warmup_steps is None:
        args.warmup_steps = 10 if official else 5
    if args.warmup_gripper is None:
        args.warmup_gripper = -1.0 if official else 0.0
    if args.max_env_steps is None:
        args.max_env_steps = (
            TASK_MAX_STEPS.get(args.libero_benchmark, 800) + args.warmup_steps
            if official
            else 800
        )
    if official and args.initial_states_json is None:
        args.initial_states_json = official_metainfo_path(args.cosmos_repo, args.libero_benchmark)
    return args


def load_initial_state_from_metainfo(path, prompt, episode_idx):
    task_key = prompt.replace(" ", "_")
    episode_key = f"demo_{episode_idx}"
    with open(path, "r") as f:
        metainfo = json.load(f)
    if task_key not in metainfo:
        raise KeyError(f"Task key {task_key!r} not found in initial states JSON: {path}")
    if episode_key not in metainfo[task_key]:
        raise KeyError(
            f"Episode key {episode_key!r} not found for task {task_key!r} in {path}"
        )

    source = f"{path}:{task_key}/{episode_key}"
    entry = metainfo[task_key][episode_key]
    if not bool(entry.get("success", False)):
        return None, source, True
    return np.asarray(entry["initial_state"], dtype=np.float64), source, False


def resolve_initial_state(benchmark_instance, task_idx, episode_idx, prompt, args):
    if args.initial_states_json:
        return load_initial_state_from_metainfo(args.initial_states_json, prompt, episode_idx)

    init_states = benchmark_instance.get_task_init_states(task_idx)
    source = f"benchmark_init_states:task_{task_idx}/episode_{episode_idx % init_states.shape[0]}"
    return init_states[episode_idx % init_states.shape[0]], source, False


def construct_single_env(env_args, env_seed=None):
    last_error = None
    for _ in range(5):
        try:
            env = OffScreenRenderEnv(**env_args)
            if env_seed is not None:
                env.seed(int(env_seed))
            return env
        except Exception as exc:
            last_error = exc
            print(f"construct env failed: {exc}", flush=True)
            time.sleep(5)
    raise RuntimeError(f"Failed to construct LIBERO env after retries: {last_error}")


def extract_video_obs(obs):
    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(obs["agentview_image"][::-1]),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1]
        ),
    }


def extract_raw_batch(obs, prompt):
    primary = np.ascontiguousarray(obs["agentview_image"][::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    proprio = np.concatenate(
        [
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    if proprio.shape[-1] != 9:
        raise ValueError(f"Expected 9D Cosmos LIBERO proprio, got shape {proprio.shape}")
    return {
        "raw_primary_image": torch.from_numpy(primary),
        "raw_wrist_image": torch.from_numpy(wrist),
        "raw_proprio": torch.from_numpy(proprio),
        "raw_task": prompt,
    }


def init_single_env(env, init_state, warmup_steps=5, warmup_gripper=0.0):
    env.reset()
    obs = env.set_init_state(init_state)
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(warmup_gripper)]
    for _ in range(int(warmup_steps)):
        obs, _, _, _ = env.step(dummy_action)
    if obs is None:
        raise RuntimeError("LIBERO env did not return an observation during reset warmup.")
    return obs


def configure_cosmos(args):
    cfg = importlib.import_module(args.config).cfg
    cfg.teacher_backend = "cosmos_policy"
    cfg.teacher_model_path = resolve_cosmos_policy_assets(args.cosmos_policy_path)["root"]
    cfg.cosmos_policy_use_raw_inference = True
    cfg.return_raw_observation = True
    cfg.cosmos_policy_inference_mode = args.inference_mode
    cfg.cosmos_policy_num_denoising_steps_action = args.num_denoising_steps_action
    cfg.cosmos_policy_seed = args.seed
    cfg.cosmos_policy_repo = args.cosmos_repo
    cfg.cosmos_policy_python = args.cosmos_python
    cfg.cosmos_policy_extra_pythonpath = args.extra_pythonpath
    cfg.cosmos_policy_local_model_dir = args.local_model_dir
    cfg.cosmos_policy_config_name = args.cosmos_config_name
    cfg.cosmos_policy_config_file = args.cosmos_config_file
    return cfg


def rollout_one(teacher, libero_benchmark, task_idx, episode_idx, out_dir, args):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
        "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
        "camera_heights": args.camera_size,
        "camera_widths": args.camera_size,
    }
    initial_state, initial_state_source, skipped = resolve_initial_state(
        benchmark_instance, task_idx, episode_idx, prompt, args
    )
    if skipped:
        return {
            "task_idx": task_idx,
            "episode_idx": episode_idx,
            "prompt": prompt,
            "done": False,
            "skipped": True,
            "skip_reason": "official initial-state metainfo marks this demo as unsuccessful",
            "initial_state_source": initial_state_source,
            "env_steps": 0,
            "num_chunks": 0,
            "video_path": None,
            "actions_path": None,
        }

    env = construct_single_env(env_args, env_seed=args.env_seed)
    obs = init_single_env(
        env,
        initial_state,
        warmup_steps=args.warmup_steps,
        warmup_gripper=args.warmup_gripper,
    )
    frames = []
    action_chunks = []
    done = False
    chunk_idx = 0

    try:
        while env.env.timestep < args.max_env_steps and not done:
            raw_batch = extract_raw_batch(obs, prompt)
            actions = teacher.predict_raw_actions(raw_batch).detach().cpu().numpy()
            if actions.ndim == 3:
                actions = actions[0]
            if actions.ndim != 2 or actions.shape[-1] != 7:
                raise ValueError(f"Expected Cosmos actions [T,7], got {actions.shape}")
            action_chunks.append(actions.astype(np.float32))

            start_idx = 1 if args.skip_first_action and chunk_idx == 0 else 0
            for action in actions[start_idx:]:
                obs, _, done, _ = env.step(action.astype(np.float32))
                frames.append(extract_video_obs(obs))
                if done or env.env.timestep >= args.max_env_steps:
                    break
            chunk_idx += 1

        task_name = prompt.replace(" ", "_").replace("/", "_")
        video_path = (
            Path(out_dir)
            / libero_benchmark
            / f"task_{task_idx}_{task_name}"
            / f"episode_{episode_idx}_done_{done}.mp4"
        )
        save_video(frames, video_path, fps=args.fps)

        actions_path = video_path.with_suffix(".actions.pt")
        torch.save(
            {
                "action_chunks": [torch.from_numpy(a) for a in action_chunks],
                "prompt": prompt,
                "task_idx": task_idx,
                "episode_idx": episode_idx,
                "done": bool(done),
                "env_steps": int(env.env.timestep),
                "initial_state_source": initial_state_source,
                "official_libero_eval": bool(args.official_libero_eval),
                "env_seed": args.env_seed,
                "warmup_steps": args.warmup_steps,
                "warmup_gripper": args.warmup_gripper,
            },
            actions_path,
        )
        return {
            "task_idx": task_idx,
            "episode_idx": episode_idx,
            "prompt": prompt,
            "done": bool(done),
            "skipped": False,
            "env_steps": int(env.env.timestep),
            "num_chunks": len(action_chunks),
            "initial_state_source": initial_state_source,
            "video_path": str(video_path),
            "actions_path": str(actions_path),
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-benchmark", default="libero_10")
    parser.add_argument("--task-range", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--config", default="distillation_flowmap.config_libero_cosmos_policy_stage1")
    parser.add_argument(
        "--cosmos-policy-path",
        default="/root/nas/junjie/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    )
    parser.add_argument("--cosmos-repo", default="/root/nas/junjie/cosmos_predict2_5/repos/cosmos-predict2.5")
    parser.add_argument(
        "--cosmos-python",
        default="/root/nas/junjie/cosmos_predict2_5/envs/predict2_py310/bin/python",
    )
    parser.add_argument(
        "--extra-pythonpath",
        default="/root/nas/junjie/conda_envs/any_wam/lib/python3.10/site-packages",
    )
    parser.add_argument(
        "--local-model-dir",
        default="/root/nas/junjie/cosmos_predict2_5/checkpoints/local_hf/Cosmos-Predict2-2B-Video2World",
    )
    parser.add_argument("--cosmos-config-name", default="cosmos_predict2_2b_480p_libero__inference_only")
    parser.add_argument(
        "--cosmos-config-file",
        default="cosmos_predict2/_src/predict2/cosmos_policy/config/config.py",
    )
    parser.add_argument("--inference-mode", choices=["subprocess", "inprocess"], default="subprocess")
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--camera-size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--skip-first-action", action="store_true")
    parser.add_argument(
        "--official-libero-eval",
        action="store_true",
        help="Use NVIDIA Cosmos LIBERO eval defaults: seed 195, env.seed(0), 10 dummy steps, official initial-state metainfo, and suite max steps.",
    )
    parser.add_argument("--env-seed", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup-gripper", type=float, default=None)
    parser.add_argument("--initial-states-json", default=None)
    args = parser.parse_args()
    apply_eval_defaults(args)

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    cfg = configure_cosmos(args)
    teacher = CosmosPolicyActionTeacher(
        cfg.teacher_model_path,
        dtype=torch.bfloat16,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        config=cfg,
    )

    results = []
    try:
        for task_idx in tqdm(range(args.task_range[0], args.task_range[1]), desc="tasks"):
            for episode_idx in tqdm(range(args.test_num), desc=f"task {task_idx}", leave=False):
                results.append(
                    rollout_one(teacher, args.libero_benchmark, task_idx, episode_idx, args.out_dir, args)
                )
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    finally:
        teacher.close()

    attempted = [item for item in results if not item.get("skipped", False)]
    success_count = int(sum(1 for item in attempted if item["done"]))
    summary = {
        "libero_benchmark": args.libero_benchmark,
        "task_range": args.task_range,
        "test_num": args.test_num,
        "num_denoising_steps_action": args.num_denoising_steps_action,
        "official_libero_eval": bool(args.official_libero_eval),
        "seed": args.seed,
        "env_seed": args.env_seed,
        "warmup_steps": args.warmup_steps,
        "warmup_gripper": args.warmup_gripper,
        "max_env_steps": args.max_env_steps,
        "initial_states_json": args.initial_states_json,
        "success_count": success_count,
        "total": len(attempted),
        "skipped_count": len(results) - len(attempted),
        "success_rate": float(success_count / max(1, len(attempted))),
        "results": results,
    }
    summary_json = args.summary_json or str(Path(args.out_dir) / "summary.json")
    Path(summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
