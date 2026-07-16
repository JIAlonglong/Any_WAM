"""LIBERO-owned rollout client for the dedicated Cosmos Progressive S4 service.

The client owns reset, shared initial states, frame collection, action execution,
and durable trial records.  The policy service owns only a single fresh-anchor
decision; a failed service request is therefore an unsuccessful trial, never a
successful no-op.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from evaluation.libero.cosmos_progressive_s4_server import S4_ACTION_DIM, S4_ACTION_STEPS


def _video_observation(obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Mirror rollout_cosmos_policy.extract_video_obs for pure-client tests."""
    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(obs["agentview_image"][::-1]),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(
            obs["robot0_eye_in_hand_image"][::-1]
        ),
    }


def _init_env_like_official_loop(env: Any, initial_state: Any, *, warmup_steps: int, warmup_gripper: float):
    """Exact reset/set-init/warm-up sequence from rollout_cosmos_policy.py."""
    env.reset()
    obs = env.set_init_state(initial_state)
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(warmup_gripper)]
    for _ in range(int(warmup_steps)):
        obs, _, _, _ = env.step(dummy_action)
    if obs is None:
        raise RuntimeError("LIBERO env did not return an observation during reset warmup.")
    return obs


class CosmosProgressiveS4Client:
    """Execute dedicated S4 chunks in a client-owned LIBERO environment."""

    def __init__(
        self,
        service: Any,
        *,
        output_dir: str | Path,
        warmup_steps: int = 5,
        warmup_gripper: float = 0.0,
        skip_first_action: bool = False,
    ) -> None:
        if not hasattr(service, "infer"):
            raise TypeError("service must provide infer(request)")
        self.service = service
        self.output_dir = Path(output_dir)
        self.warmup_steps = int(warmup_steps)
        self.warmup_gripper = float(warmup_gripper)
        self.skip_first_action = bool(skip_first_action)

    def _record_path(self, task_idx: int, episode_idx: int) -> Path:
        return self.output_dir / "records" / f"task_{int(task_idx)}_episode_{int(episode_idx)}.json"

    def _write_record(self, record: Mapping[str, Any]) -> Path:
        path = self._record_path(int(record["task_idx"]), int(record["episode_idx"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        # A single replacement makes every requested pair either absent before
        # rollout or a complete JSON record afterwards.
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(dict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    @staticmethod
    def _validate_service_action(response: Mapping[str, Any]) -> np.ndarray:
        if not isinstance(response, Mapping):
            raise TypeError(f"S4 service response must be a mapping, got {type(response)!r}")
        action = response.get("action", response.get("actions"))
        if action is None:
            raise KeyError("S4 service response is missing action")
        action = np.ascontiguousarray(np.asarray(action), dtype=np.float32)
        if action.shape != (S4_ACTION_STEPS, S4_ACTION_DIM):
            raise ValueError(
                f"Dedicated S4 service must return float32 [{S4_ACTION_STEPS}, {S4_ACTION_DIM}], "
                f"got {action.shape}"
            )
        return action

    def _server_failure_record(
        self,
        *,
        task_idx: int,
        episode_idx: int,
        prompt: str,
        env_steps: int,
        chunks: int,
        exc: Exception,
    ) -> dict[str, Any]:
        record = {
            "task_idx": int(task_idx),
            "episode_idx": int(episode_idx),
            "prompt": str(prompt),
            "done": False,
            "success": False,
            "server_failure": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "env_steps": int(env_steps),
            "num_chunks": int(chunks),
            "video_path": None,
        }
        self._write_record(record)
        return record

    def run_with_env(
        self,
        *,
        env: Any,
        initial_state: Any,
        task_idx: int,
        episode_idx: int,
        prompt: str,
        max_env_steps: int,
        init_env_fn: Callable[..., Any] | None = None,
        extract_video_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        save_video_fn: Callable[..., Any] | None = None,
        video_path: str | Path | None = None,
        close_env: bool = True,
    ) -> dict[str, Any]:
        """Run one requested pair and always write exactly one trial record.

        ``init_env_fn`` and ``extract_video_fn`` are injection seams for pure
        tests.  The CLI path passes the official functions from
        ``rollout_cosmos_policy.py`` so its reset, warm-up, flip, and video
        behavior stays identical to the established LIBERO evaluator.
        """
        init_env_fn = init_env_fn or _init_env_like_official_loop
        extract_video_fn = extract_video_fn or _video_observation
        frames: list[Mapping[str, Any]] = []
        chunks = 0
        done = False
        service_metadata: dict[str, Any] = {}
        env_steps = 0
        try:
            obs = init_env_fn(
                env,
                initial_state,
                warmup_steps=self.warmup_steps,
                warmup_gripper=self.warmup_gripper,
            )
            # Preserve the legacy reset/infer request shape.  A reset has no
            # action and does not permit a cached anchor to carry into a trial.
            # It is a service operation, so a failed connection here is a
            # failed trial rather than an environment setup failure.
            try:
                self.service.infer({"reset": True, "prompt": prompt})
            except Exception as exc:
                return self._server_failure_record(
                    task_idx=task_idx,
                    episode_idx=episode_idx,
                    prompt=prompt,
                    env_steps=env_steps,
                    chunks=chunks,
                    exc=exc,
                )
            while int(env.env.timestep) < int(max_env_steps) and not done:
                try:
                    response = self.service.infer({"obs": obs, "prompt": prompt})
                    actions = self._validate_service_action(response)
                except Exception as exc:
                    return self._server_failure_record(
                        task_idx=task_idx,
                        episode_idx=episode_idx,
                        prompt=prompt,
                        env_steps=env_steps,
                        chunks=chunks,
                        exc=exc,
                    )
                service_metadata = {
                    key: response[key]
                    for key in ("s4_checkpoint", "decision_duration_s", "raw_anchor_record")
                    if key in response
                }
                start_idx = 1 if self.skip_first_action and chunks == 0 else 0
                for action in actions[start_idx:]:
                    obs, _, done, _ = env.step(action.astype(np.float32, copy=False))
                    env_steps += 1
                    frames.append(extract_video_fn(obs))
                    if done or int(env.env.timestep) >= int(max_env_steps):
                        break
                chunks += 1
            if video_path is not None and frames and save_video_fn is not None:
                save_video_fn(frames, video_path)
            record = {
                "task_idx": int(task_idx),
                "episode_idx": int(episode_idx),
                "prompt": str(prompt),
                "done": bool(done),
                "success": bool(done),
                "server_failure": False,
                "env_steps": int(env_steps),
                "num_chunks": int(chunks),
                "video_path": str(video_path) if video_path is not None and frames else None,
                **service_metadata,
            }
            self._write_record(record)
            return record
        except Exception as exc:
            # Setup/reset failures still need a record for the requested pair;
            # distinguish them from a connection/action service failure.
            record = {
                "task_idx": int(task_idx),
                "episode_idx": int(episode_idx),
                "prompt": str(prompt),
                "done": False,
                "success": False,
                "server_failure": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "env_steps": int(env_steps),
                "num_chunks": int(chunks),
                "video_path": None,
            }
            self._write_record(record)
            return record
        finally:
            if close_env:
                try:
                    env.close()
                except Exception:
                    pass

    def run_libero_task(
        self,
        *,
        libero_benchmark: str,
        task_idx: int,
        episode_idx: int,
        camera_size: int,
        max_env_steps: int,
        env_seed: int | None = None,
        initial_states_json: str | None = None,
        video_fps: int = 15,
    ) -> dict[str, Any]:
        """Construct a real LIBERO task while reusing the official rollout helpers."""
        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
            from evaluation.libero.rollout_cosmos_policy import (
                TASK_MAX_STEPS,
                construct_single_env,
                extract_video_obs,
                resolve_initial_state,
                save_video,
            )
        except Exception as exc:
            raise CosmosProgressiveS4ClientPrerequisiteError(
                "LIBERO environment imports are unavailable. Install/use the existing LIBERO evaluation "
                "environment for rollout; CPU pure service tests do not require it."
            ) from exc

        benchmark_instance = benchmark.get_benchmark_dict()[libero_benchmark]()
        if not 0 <= int(task_idx) < int(benchmark_instance.get_num_tasks()):
            raise IndexError(f"task_idx={task_idx} is outside benchmark {libero_benchmark}")
        prompt = benchmark_instance.get_task(int(task_idx)).language
        # Reuse resolve_initial_state exactly, including official shared state files.
        argument_view = type("RolloutArgs", (), {"initial_states_json": initial_states_json})()
        initial_state, state_source, skipped = resolve_initial_state(
            benchmark_instance, int(task_idx), int(episode_idx), prompt, argument_view
        )
        if skipped:
            record = {
                "task_idx": int(task_idx),
                "episode_idx": int(episode_idx),
                "prompt": prompt,
                "done": False,
                "success": False,
                "server_failure": False,
                "skipped": True,
                "skip_reason": "official initial-state metainfo marks this demo as unsuccessful",
                "initial_state_source": state_source,
                "env_steps": 0,
                "num_chunks": 0,
                "video_path": None,
            }
            self._write_record(record)
            return record
        env_args = {
            "bddl_file_name": benchmark_instance.get_task_bddl_file_path(int(task_idx)),
            "camera_heights": int(camera_size),
            "camera_widths": int(camera_size),
        }
        env = construct_single_env(env_args, env_seed=env_seed)
        task_name = prompt.replace(" ", "_").replace("/", "_")
        video_path = (
            self.output_dir
            / libero_benchmark
            / f"task_{int(task_idx)}_{task_name}"
            / f"episode_{int(episode_idx)}_done.mp4"
        )
        # TASK_MAX_STEPS import prevents a caller accidentally using a generic
        # 48-channel server default; its value is only a sane upper bound here.
        bounded_steps = min(int(max_env_steps), int(TASK_MAX_STEPS.get(libero_benchmark, max_env_steps)))
        record = self.run_with_env(
            env=env,
            initial_state=initial_state,
            task_idx=task_idx,
            episode_idx=episode_idx,
            prompt=prompt,
            max_env_steps=bounded_steps,
            init_env_fn=lambda current_env, state, *, warmup_steps, warmup_gripper: __import__(
                "evaluation.libero.rollout_cosmos_policy", fromlist=["init_single_env"]
            ).init_single_env(
                current_env,
                state,
                warmup_steps=warmup_steps,
                warmup_gripper=warmup_gripper,
            ),
            extract_video_fn=extract_video_obs,
            save_video_fn=lambda frames, path: save_video(frames, path, fps=video_fps),
            video_path=video_path,
            close_env=True,
        )
        record["initial_state_source"] = state_source
        self._write_record(record)
        return record


class CosmosProgressiveS4ClientPrerequisiteError(RuntimeError):
    """Actionable error emitted only when a real LIBERO rollout is requested."""
