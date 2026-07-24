import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from evaluation.libero.cosmos_progressive_s4_client import CosmosProgressiveS4Client
from evaluation.libero.cosmos_progressive_s4_server import (
    ActionDecodingTemplate,
    CosmosProgressiveS4Engine,
    CosmosProgressiveS4Service,
    PromptEmbeddingTable,
    build_cosmos_raw_request,
    decode_student_action,
)


OBS = {
    "agentview_image": np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3),
    "robot0_eye_in_hand_image": np.full((4, 5, 3), 7, dtype=np.uint8),
    "robot0_gripper_qpos": np.asarray([0.25, -0.5], dtype=np.float32),
    "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
    "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}


def _template():
    return ActionDecodingTemplate(
        q01=np.full(7, -1.0, dtype=np.float32),
        q99=np.full(7, 1.0, dtype=np.float32),
        inverse_used_action_channel_ids=tuple(range(7)),
        action_dim=7,
    )


def test_live_s4_request_uses_cosmos_raw_observation_contract():
    payload = build_cosmos_raw_request(libero_obs=OBS, prompt="open the drawer")

    assert payload["raw_primary_image"].shape[-1] == 3
    assert payload["raw_wrist_image"].shape[-1] == 3
    assert payload["raw_proprio"].shape == (9,)
    np.testing.assert_array_equal(payload["raw_primary_image"], OBS["agentview_image"][::-1])
    np.testing.assert_allclose(
        payload["raw_proprio"],
        np.concatenate(
            [
                OBS["robot0_gripper_qpos"],
                OBS["robot0_eef_pos"],
                OBS["robot0_eef_quat"],
            ]
        ),
    )


def test_live_s4_request_converts_eight_dim_libero_state_to_official_nine_dim_proprio():
    state_obs = {
        "agentview_image": OBS["agentview_image"],
        "robot0_eye_in_hand_image": OBS["robot0_eye_in_hand_image"],
        "state": np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.25, -0.5], dtype=np.float32),
    }

    payload = build_cosmos_raw_request(libero_obs=state_obs, prompt="open the drawer")

    np.testing.assert_allclose(
        payload["raw_proprio"],
        np.asarray([0.25, -0.5, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )


def test_live_s4_action_decoder_returns_only_16_valid_seven_dof_actions():
    flowmap_action_tensor = np.zeros((1, 7, 4, 4, 1), dtype=np.float32)

    decoded = decode_student_action(flowmap_action_tensor, _template())

    assert decoded.shape == (16, 7)
    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, 0.0)


def test_prompt_table_rejects_a_missing_libero_task_embedding():
    with pytest.raises(KeyError, match="missing prompt embedding"):
        PromptEmbeddingTable({}).get("unseen task")


def test_prompt_table_only_returns_training_shape_embedding():
    embedding = np.zeros((1, 512, 4096), dtype=np.float32)
    table = PromptEmbeddingTable({"open the drawer": embedding})

    assert table.get("open the drawer") is embedding
    with pytest.raises(ValueError, match="\\[1, 512, 4096\\]"):
        PromptEmbeddingTable({"bad": np.zeros((512, 4096), dtype=np.float32)}).get("bad")


class _RecordingTeacher:
    def __init__(self):
        self.calls = []

    def predict_raw_latent_target(self, raw_batch, *, noise, t, r, epsilon, include_cdiff):
        self.calls.append(
            {
                "raw_batch": raw_batch,
                "noise": noise.copy(),
                "t": t.copy(),
                "r": r.copy(),
                "include_cdiff": include_cdiff,
            }
        )
        return {
            "cosmos_latent_x0": np.zeros((1, 16, 9, 28, 28), dtype=np.float32),
            "actions": np.zeros((1, 16, 7), dtype=np.float32),
        }


def test_every_infer_request_queries_a_fresh_cosmos_raw_anchor(tmp_path):
    teacher = _RecordingTeacher()
    template = _template()
    joint_calls = []

    def run_joint_student(video_x0, action_x0, text_emb, **kwargs):
        joint_calls.append(kwargs)
        return np.zeros((1, 7, 4, 4, 1), dtype=np.float32)

    engine = CosmosProgressiveS4Engine(
        cosmos_teacher=teacher,
        prompt_table=PromptEmbeddingTable(
            {"open the drawer": np.zeros((1, 512, 4096), dtype=np.float32)}
        ),
        action_template=template,
        action_encoder=lambda raw_actions: raw_actions,
        joint_s4_runner=run_joint_student,
        anchor_noise_factory=lambda: np.ones((1, 16, 9, 28, 28), dtype=np.float32),
        student_steps=2,
    )
    service = CosmosProgressiveS4Service(
        engine=engine,
        checkpoint_identifier="s4-checkpoint",
        anchor_record_dir=tmp_path / "anchors",
    )

    first = service.infer({"obs": OBS, "prompt": "open the drawer"})
    second = service.infer({"obs": OBS, "prompt": "open the drawer"})

    assert len(teacher.calls) == 2
    assert [call["k_steps"] for call in joint_calls] == [2, 2]
    assert all(call["include_cdiff"] is False for call in teacher.calls)
    assert first["action"].shape == (16, 7)
    assert first["action"].dtype == np.float32
    assert first["raw_anchor_record"] != second["raw_anchor_record"]
    assert (tmp_path / "anchors" / first["raw_anchor_record"]).is_file()
    assert first["s4_checkpoint"] == "s4-checkpoint"
    assert first["student_steps"] == 2
    assert first["decision_duration_s"] >= 0.0


class _FailingService:
    def infer(self, _request):
        raise RuntimeError("anchor worker unavailable")


class _WarmupEnv:
    def __init__(self):
        self.env = type("_Inner", (), {"timestep": 0})()

    def reset(self):
        return None

    def set_init_state(self, _state):
        return OBS

    def step(self, _action):
        return OBS, 0.0, False, {}

    def close(self):
        return None


def test_client_records_server_failure_as_unsuccessful_trial(tmp_path):
    client = CosmosProgressiveS4Client(
        _FailingService(), output_dir=tmp_path, student_steps=2, warmup_steps=1
    )

    record = client.run_with_env(
        env=_WarmupEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=2,
        episode_idx=5,
        prompt="open the drawer",
        max_env_steps=2,
        rollout_seed=17,
    )

    assert record["success"] is False
    assert record["done"] is False
    assert record["server_failure"] is True
    assert record["seed"] == 17
    assert record["student_steps"] == 2
    record_path = tmp_path / "records" / "task_2_episode_5.json"
    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert persisted["success"] is False
    assert persisted["seed"] == 17
    assert persisted["student_steps"] == 2


class _SuccessfulService:
    def infer(self, request):
        if request.get("reset"):
            return {"ok": True, "s4_checkpoint": "s4-checkpoint", "student_steps": 2}
        return {
            "action": np.zeros((16, 7), dtype=np.float32),
            "s4_checkpoint": "s4-checkpoint",
            "student_steps": 2,
        }


class _DoneEnv:
    def __init__(self):
        self.env = type("_Inner", (), {"timestep": 0})()

    def step(self, _action):
        self.env.timestep += 1
        return OBS, 0.0, True, {}

    def close(self):
        return None


def test_client_records_rollout_seed_on_success(tmp_path):
    client = CosmosProgressiveS4Client(
        _SuccessfulService(), output_dir=tmp_path, student_steps=2, warmup_steps=0
    )

    record = client.run_with_env(
        env=_DoneEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=3,
        episode_idx=0,
        prompt="open the drawer",
        max_env_steps=1,
        init_env_fn=lambda *_args, **_kwargs: OBS,
        extract_video_fn=lambda _obs: {},
        rollout_seed=23,
    )

    assert record["success"] is True
    assert record["server_failure"] is False
    assert record["seed"] == 23
    assert record["student_steps"] == 2
    persisted = json.loads(
        (tmp_path / "records" / "task_3_episode_0.json").read_text(encoding="utf-8")
    )
    assert persisted["seed"] == 23
    assert persisted["student_steps"] == 2


def test_client_records_rollout_seed_on_setup_failure(tmp_path):
    client = CosmosProgressiveS4Client(
        _SuccessfulService(), output_dir=tmp_path, student_steps=2, warmup_steps=0
    )

    def fail_init(*_args, **_kwargs):
        raise RuntimeError("fixture setup failure")

    record = client.run_with_env(
        env=_DoneEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=4,
        episode_idx=0,
        prompt="open the drawer",
        max_env_steps=1,
        init_env_fn=fail_init,
        rollout_seed=29,
    )

    assert record["success"] is False
    assert record["server_failure"] is False
    assert record["seed"] == 29
    assert record["student_steps"] == 2
    persisted = json.loads(
        (tmp_path / "records" / "task_4_episode_0.json").read_text(encoding="utf-8")
    )
    assert persisted["seed"] == 29
    assert persisted["student_steps"] == 2


@pytest.mark.parametrize("steps", [0, 3, 5])
def test_client_rejects_unsupported_student_steps(tmp_path, steps):
    with pytest.raises(ValueError, match="student_steps must be one of"):
        CosmosProgressiveS4Client(
            _SuccessfulService(), output_dir=tmp_path, student_steps=steps
        )


class _MismatchedInferStepsService:
    def infer(self, request):
        if request.get("reset"):
            return {"ok": True, "student_steps": 2}
        return {
            "action": np.zeros((16, 7), dtype=np.float32),
            "student_steps": 4,
        }


def test_client_rejects_infer_student_steps_mismatch_without_overwriting_record(tmp_path):
    client = CosmosProgressiveS4Client(
        _MismatchedInferStepsService(),
        output_dir=tmp_path,
        student_steps=2,
        warmup_steps=0,
    )

    record = client.run_with_env(
        env=_DoneEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=5,
        episode_idx=0,
        prompt="open the drawer",
        max_env_steps=1,
        init_env_fn=lambda *_args, **_kwargs: OBS,
        extract_video_fn=lambda _obs: {},
        rollout_seed=30,
    )

    assert record["server_failure"] is True
    assert record["student_steps"] == 2
    assert "student_steps mismatch" in record["error"]
    persisted = json.loads(
        (tmp_path / "records" / "task_5_episode_0.json").read_text(encoding="utf-8")
    )
    assert persisted["student_steps"] == 2


class _MismatchedResetStepsService:
    def __init__(self):
        self.calls = 0

    def infer(self, request):
        self.calls += 1
        if request.get("reset"):
            return {"ok": True, "student_steps": 4}
        return {
            "action": np.zeros((16, 7), dtype=np.float32),
            "student_steps": 2,
        }


def test_client_rejects_reset_student_steps_mismatch_before_infer(tmp_path):
    service = _MismatchedResetStepsService()
    client = CosmosProgressiveS4Client(
        service,
        output_dir=tmp_path,
        student_steps=2,
        warmup_steps=0,
    )

    record = client.run_with_env(
        env=_DoneEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=5,
        episode_idx=1,
        prompt="open the drawer",
        max_env_steps=1,
        init_env_fn=lambda *_args, **_kwargs: OBS,
        extract_video_fn=lambda _obs: {},
        rollout_seed=30,
    )

    assert service.calls == 1
    assert record["server_failure"] is True
    assert record["student_steps"] == 2
    assert "student_steps mismatch" in record["error"]
    persisted = json.loads(
        (tmp_path / "records" / "task_5_episode_1.json").read_text(encoding="utf-8")
    )
    assert persisted["student_steps"] == 2


def _install_fake_libero(monkeypatch, *, skipped):
    libero_package = ModuleType("libero")
    libero_package.__path__ = []
    libero_module = ModuleType("libero.libero")
    libero_module.__path__ = []
    benchmark_module = ModuleType("libero.libero.benchmark")
    envs_module = ModuleType("libero.libero.envs")
    envs_module.OffScreenRenderEnv = object

    class _Benchmark:
        def get_num_tasks(self):
            return 10

        def get_task(self, _task_idx):
            return SimpleNamespace(language="open the drawer")

        def get_task_bddl_file_path(self, _task_idx):
            return "fixture.bddl"

    benchmark_module.get_benchmark_dict = lambda: {"libero_10": _Benchmark}
    libero_module.benchmark = benchmark_module

    rollout_module = ModuleType("evaluation.libero.rollout_cosmos_policy")
    rollout_module.TASK_MAX_STEPS = {"libero_10": 1}
    rollout_module.construct_single_env = lambda *_args, **_kwargs: _DoneEnv()
    rollout_module.extract_video_obs = lambda _obs: {}
    rollout_module.init_single_env = lambda *_args, **_kwargs: OBS
    rollout_module.resolve_initial_state = lambda *_args, **_kwargs: (
        np.zeros(1, dtype=np.float32),
        "fixture initial state",
        skipped,
    )
    rollout_module.save_video = lambda *_args, **_kwargs: None

    monkeypatch.setitem(sys.modules, "libero", libero_package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_module)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark_module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs_module)
    monkeypatch.setitem(sys.modules, "evaluation.libero.rollout_cosmos_policy", rollout_module)


def test_run_libero_task_persists_env_seed_for_skipped_record_without_libero(tmp_path, monkeypatch):
    _install_fake_libero(monkeypatch, skipped=True)
    client = CosmosProgressiveS4Client(
        _SuccessfulService(), output_dir=tmp_path, student_steps=2
    )

    record = client.run_libero_task(
        libero_benchmark="libero_10",
        task_idx=6,
        episode_idx=0,
        camera_size=128,
        max_env_steps=1,
        env_seed=31,
    )

    assert record["skipped"] is True
    assert record["seed"] == 31
    assert record["student_steps"] == 2
    persisted = json.loads(
        (tmp_path / "records" / "task_6_episode_0.json").read_text(encoding="utf-8")
    )
    assert persisted["seed"] == 31
    assert persisted["student_steps"] == 2


def test_run_libero_task_forwards_env_seed_to_run_with_env_without_libero(tmp_path, monkeypatch):
    _install_fake_libero(monkeypatch, skipped=False)
    client = CosmosProgressiveS4Client(
        _SuccessfulService(), output_dir=tmp_path, student_steps=2
    )
    captured = {}

    def fake_run_with_env(**kwargs):
        captured.update(kwargs)
        return {
            "task_idx": kwargs["task_idx"],
            "episode_idx": kwargs["episode_idx"],
            "seed": kwargs["rollout_seed"],
        }

    monkeypatch.setattr(client, "run_with_env", fake_run_with_env)

    record = client.run_libero_task(
        libero_benchmark="libero_10",
        task_idx=7,
        episode_idx=0,
        camera_size=128,
        max_env_steps=1,
        env_seed=37,
    )

    assert captured["rollout_seed"] == 37
    assert record["seed"] == 37


def test_live_runtime_preflight_rejects_cpu_with_actionable_error(tmp_path):
    from evaluation.libero.rollout_cosmos_progressive_s4 import require_live_s4_prerequisites

    with pytest.raises(RuntimeError, match="CPU-only"):
        require_live_s4_prerequisites(
            device="cpu",
            checkpoint_transformer=tmp_path / "s4-checkpoint",
        )


def _cuda_available_torch(*, device_count=1):
    return SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: int(device_count),
        )
    )


def test_live_runtime_preflight_rejects_incompatible_host_driver(tmp_path, monkeypatch):
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    checkpoint = tmp_path / "s4-checkpoint"
    checkpoint.mkdir()
    monkeypatch.setitem(sys.modules, "torch", _cuda_available_torch())
    monkeypatch.setattr(rollout, "_nvidia_driver_version", lambda: "550.90.07")

    with pytest.raises(RuntimeError, match="NVIDIA driver.*570.124.06"):
        rollout.require_live_s4_prerequisites(
            device="cuda:0",
            checkpoint_transformer=checkpoint,
        )


def test_live_runtime_preflight_rejects_non_cu128_cosmos_python(tmp_path, monkeypatch):
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    checkpoint = tmp_path / "s4-checkpoint"
    checkpoint.mkdir()
    cosmos_python = tmp_path / "cosmos-python"
    cosmos_python.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "torch", _cuda_available_torch())
    monkeypatch.setattr(rollout, "_nvidia_driver_version", lambda: "570.124.06")
    monkeypatch.setattr(rollout, "_cosmos_python_cuda_version", lambda _path: "12.4")
    monkeypatch.setenv("COSMOS_POLICY_PYTHON", str(cosmos_python))

    with pytest.raises(RuntimeError, match="Cosmos worker CUDA runtime.*12.8"):
        rollout.require_live_s4_prerequisites(
            device="cuda:0",
            checkpoint_transformer=checkpoint,
        )


def test_live_cli_requires_seed_before_constructing_service(monkeypatch):
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    def unexpected_service_construction(_args):
        raise AssertionError("live service construction must not run without --env-seed")

    monkeypatch.setattr(rollout, "build_live_service", unexpected_service_construction)

    with pytest.raises(ValueError, match="--env-seed is required"):
        rollout.main(
            [
                "--checkpoint-transformer",
                "/tmp/s4-transformer",
                "--prompt-table",
                "/tmp/training-prompt-table.pt",
            ]
        )


def test_live_rollout_constructs_client_with_requested_student_steps(monkeypatch):
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    captured = {}

    class RecordingClient:
        def __init__(self, _service, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        rollout,
        "build_live_service",
        lambda _args: (object(), SimpleNamespace(close=lambda: None)),
    )
    monkeypatch.setattr(rollout, "CosmosProgressiveS4Client", RecordingClient)

    result = rollout.main(
        [
            "--checkpoint-transformer",
            "/tmp/s4-transformer",
            "--prompt-table",
            "/tmp/training-prompt-table.pt",
            "--student-steps",
            "2",
            "--env-seed",
            "41",
            "--task-range",
            "0",
            "0",
        ]
    )

    assert result == 0
    assert captured["student_steps"] == 2


def test_cli_defaults_joint_student_steps_to_four():
    from evaluation.libero.rollout_cosmos_progressive_s4 import parse_args

    args = parse_args(
        [
            "--checkpoint-transformer",
            "/tmp/s4-transformer",
            "--prompt-table",
            "/tmp/training-prompt-table.pt",
        ]
    )

    assert args.student_steps == 4


@pytest.mark.parametrize("steps", [0, 3, 5])
def test_cli_rejects_unsupported_joint_steps(steps):
    from evaluation.libero.rollout_cosmos_progressive_s4 import parse_args

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--checkpoint-transformer",
                "/tmp/s4-transformer",
                "--prompt-table",
                "/tmp/training-prompt-table.pt",
                "--student-steps",
                str(steps),
            ]
        )


def test_s4_action_encoder_preserves_sixteen_decodable_actions_after_downsample():
    from evaluation.libero.rollout_cosmos_progressive_s4 import FlowMapActionAnchorEncoder

    captured = {}

    def converter(_actions, **kwargs):
        captured.update(kwargs)
        return np.zeros(kwargs["target_shape"], dtype=np.float32)

    config = SimpleNamespace(
        inverse_used_action_channel_ids=list(range(7)),
        action_per_frame=4,
        norm_stat={"q01": [-1.0] * 7, "q99": [1.0] * 7},
    )
    encoder = FlowMapActionAnchorEncoder(
        config=config,
        device="cpu",
        torch_module=SimpleNamespace(bfloat16=np.float32),
        converter=converter,
    )

    encoder(np.zeros((1, 16, 7), dtype=np.float32))

    assert captured["target_shape"] == (1, 7, 16, 4, 1)


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_joint_runner_uses_requested_steps_for_video_and_action(steps):
    import torch

    from evaluation.libero.rollout_cosmos_progressive_s4 import FlowMapJointS4Runner

    class RecordingHarness:
        def __init__(self):
            self.calls = []

        def _student_euler_integrate(self, **kwargs):
            self.calls.append(kwargs)
            return (
                kwargs["noisy_latents"],
                torch.zeros_like(kwargs["noisy_latents"]),
                None,
                kwargs["base_input_dict"]["action_dict"]["noisy_latents"].clone(),
            )

    def prepare_base_dict(_batch, _config, _device):
        return {
            "latent_dict": {},
            "action_dict": {},
            "chunk_size": 1,
            "window_size": 1,
        }

    def build_paired_eval_timesteps(
        *, batch_size, video_frames, action_frames, t, r, device
    ):
        return (
            torch.full((batch_size, video_frames), float(t), device=device),
            torch.full((batch_size, video_frames), float(r), device=device),
            torch.full((batch_size, action_frames), float(t), device=device),
            torch.full((batch_size, action_frames), float(r), device=device),
        )

    def student_input(
        batch,
        base,
        video_x0,
        video_noise,
        video_t,
        action_noise,
        action_t,
        action_downsample,
    ):
        video_scale = video_t[:, None, :, None, None].to(video_x0)
        video_noisy = (1.0 - video_scale) * video_x0 + video_scale * video_noise.to(video_x0)
        return (
            {
                "latent_dict": {
                    "noisy_latents": video_noisy,
                    "timesteps": video_t,
                },
                "action_dict": {
                    "noisy_latents": batch["actions"][:, :, ::action_downsample],
                },
                "chunk_size": base["input"]["chunk_size"],
                "window_size": base["input"]["window_size"],
            },
            action_noise,
        )

    harness = RecordingHarness()
    runner = FlowMapJointS4Runner(
        harness=harness,
        config=SimpleNamespace(action_downsample_factor=4),
        device="cpu",
        torch_module=torch,
        prepare_base_dict=prepare_base_dict,
        student_input=student_input,
        build_paired_eval_timesteps=build_paired_eval_timesteps,
        empty_embedding=None,
        cfg_scale=3.0,
    )
    video_x0 = torch.zeros((1, 16, 9, 28, 28), dtype=torch.bfloat16)
    video_noise = torch.full_like(video_x0, 0.25)
    action_x0 = torch.zeros((1, 7, 16, 4, 1), dtype=torch.bfloat16)
    text_emb = torch.zeros((1, 512, 4096), dtype=torch.bfloat16)

    final_action = runner(
        video_x0,
        action_x0,
        text_emb,
        noise=video_noise,
        t1000=np.ones((1, 9), dtype=np.float32),
        t0=np.zeros((1, 9), dtype=np.float32),
        k_steps=steps,
    )

    assert len(harness.calls) == 1
    call = harness.calls[0]
    torch.testing.assert_close(call["noisy_latents"], video_noise, rtol=0, atol=0)
    torch.testing.assert_close(
        call["timesteps"],
        torch.full((1, 9), 1000.0),
        rtol=0,
        atol=0,
    )
    assert [
        {
            "K_steps": call["K_steps"],
            "return_final_action": call["return_final_action"],
            "return_final_action_state": call["return_final_action_state"],
        }
    ] == [
        {
            "K_steps": steps,
            "return_final_action": True,
            "return_final_action_state": True,
        }
    ]
    decoded = decode_student_action(final_action, _template())
    assert decoded.shape == (16, 7)
    assert decoded.dtype == np.float32
