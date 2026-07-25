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
    assert [(call["video_steps"], call["action_steps"]) for call in joint_calls] == [
        (2, 2),
        (2, 2),
    ]
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
        _FailingService(),
        output_dir=tmp_path,
        student_steps=2,
        expected_s4_checkpoint="/resolved/s4-checkpoint",
        warmup_steps=1,
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
    assert record["s4_checkpoint"] == "/resolved/s4-checkpoint"
    record_path = tmp_path / "records" / "task_2_episode_5.json"
    persisted = json.loads(record_path.read_text(encoding="utf-8"))
    assert persisted["success"] is False
    assert persisted["seed"] == 17
    assert persisted["student_steps"] == 2
    assert persisted["s4_checkpoint"] == "/resolved/s4-checkpoint"


def test_client_rejects_reset_and_infer_checkpoint_mismatches(tmp_path):
    class MismatchService:
        checkpoint_identifier = "/expected/checkpoint"

        def __init__(self, *, mismatch_on_reset):
            self.mismatch_on_reset = mismatch_on_reset

        def infer(self, request):
            mismatch = bool(request.get("reset")) is self.mismatch_on_reset
            return {
                "ok": True,
                "action": np.zeros((16, 7), dtype=np.float32),
                "student_steps": 2,
                "s4_checkpoint": (
                    "/wrong/checkpoint" if mismatch else "/expected/checkpoint"
                ),
            }

    for mismatch_on_reset in (True, False):
        client = CosmosProgressiveS4Client(
            MismatchService(mismatch_on_reset=mismatch_on_reset),
            output_dir=tmp_path / str(mismatch_on_reset),
            student_steps=2,
        )
        record = client.run_with_env(
            env=_DoneEnv(),
            initial_state=np.zeros(1, dtype=np.float32),
            task_idx=0,
            episode_idx=int(mismatch_on_reset),
            prompt="open the drawer",
            max_env_steps=1,
            init_env_fn=lambda *_args, **_kwargs: OBS,
            rollout_seed=7,
        )
        assert record["success"] is False
        assert record["server_failure"] is True
        assert record["s4_checkpoint"] == "/wrong/checkpoint"
        assert record["expected_s4_checkpoint"] == "/expected/checkpoint"
        assert "checkpoint mismatch" in record["error"]


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


def test_client_omitted_student_steps_defaults_and_persists_four(tmp_path):
    class LegacySuccessfulService:
        def infer(self, request):
            if request.get("reset"):
                return {"ok": True}
            return {"action": np.zeros((16, 7), dtype=np.float32)}

    client = CosmosProgressiveS4Client(
        LegacySuccessfulService(), output_dir=tmp_path, warmup_steps=0
    )

    record = client.run_with_env(
        env=_DoneEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=1,
        episode_idx=0,
        prompt="open the drawer",
        max_env_steps=1,
        init_env_fn=lambda *_args, **_kwargs: OBS,
        extract_video_fn=lambda _obs: {},
        rollout_seed=19,
    )

    assert client.student_steps == 4
    assert record["student_steps"] == 4
    persisted = json.loads(
        (tmp_path / "records" / "task_1_episode_0.json").read_text(encoding="utf-8")
    )
    assert persisted["student_steps"] == 4


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


def test_real_service_metadata_is_persisted_in_client_episode_record(tmp_path):
    """Dropping service provenance must make the formal merger reject real rollouts."""
    teacher = _RecordingTeacher()
    engine = CosmosProgressiveS4Engine(
        cosmos_teacher=teacher,
        prompt_table=PromptEmbeddingTable(
            {"open the drawer": np.zeros((1, 512, 4096), dtype=np.float32)}
        ),
        action_template=_template(),
        action_encoder=lambda raw_actions: raw_actions,
        joint_s4_runner=lambda *_args, **_kwargs: np.zeros(
            (1, 7, 4, 4, 1), dtype=np.float32
        ),
        anchor_noise_factory=lambda: np.ones(
            (1, 16, 9, 28, 28), dtype=np.float32
        ),
        model_role="stage2_target",
        video_steps=2,
        action_steps=2,
        checkpoint_contract_identity="real-service-contract",
    )
    service = CosmosProgressiveS4Service(
        engine=engine,
        checkpoint_identifier="s4-checkpoint",
        checkpoint_contract_identity="real-service-contract",
    )
    client = CosmosProgressiveS4Client(
        service,
        output_dir=tmp_path,
        student_steps=2,
        libero_benchmark="libero_spatial",
        expected_s4_checkpoint="s4-checkpoint",
        expected_checkpoint_contract_identity="real-service-contract",
        warmup_steps=0,
    )

    record = client.run_with_env(
        env=_DoneEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=3,
        episode_idx=7,
        prompt="open the drawer",
        max_env_steps=1,
        init_env_fn=lambda *_args, **_kwargs: OBS,
        extract_video_fn=lambda _obs: {},
        rollout_seed=7,
    )

    persisted = json.loads(
        (tmp_path / "records" / "task_3_episode_7.json").read_text(encoding="utf-8")
    )
    expected = {
        "model_role": "stage2_target",
        "video_steps": 2,
        "action_steps": 2,
        "libero_benchmark": "libero_spatial",
    }
    assert {key: record[key] for key in expected} == expected
    assert {key: persisted[key] for key in expected} == expected


def test_non_video_episode_does_not_extract_or_save_frames(tmp_path):
    client = CosmosProgressiveS4Client(
        _SuccessfulService(), output_dir=tmp_path, student_steps=2, warmup_steps=0
    )
    extracted = []
    saved = []

    record = client.run_with_env(
        env=_DoneEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=3,
        episode_idx=1,
        prompt="open the drawer",
        max_env_steps=1,
        init_env_fn=lambda *_args, **_kwargs: OBS,
        extract_video_fn=lambda obs: extracted.append(obs),
        save_video_fn=lambda frames, path: saved.append((frames, path)),
        video_path=None,
        rollout_seed=23,
    )

    assert extracted == []
    assert saved == []
    assert record["video_path"] is None


def test_enabled_capture_with_no_frames_does_not_save_or_report_video(tmp_path):
    client = CosmosProgressiveS4Client(
        _SuccessfulService(), output_dir=tmp_path, student_steps=2, warmup_steps=0
    )
    extracted = []
    saved = []

    record = client.run_with_env(
        env=_DoneEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=3,
        episode_idx=2,
        prompt="open the drawer",
        max_env_steps=0,
        init_env_fn=lambda *_args, **_kwargs: OBS,
        extract_video_fn=lambda obs: extracted.append(obs),
        save_video_fn=lambda frames, path: saved.append((frames, path)),
        video_path=tmp_path / "empty.mp4",
        rollout_seed=23,
    )

    assert extracted == []
    assert saved == []
    assert record["video_path"] is None


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


def test_engine_uses_explicit_matched_video_and_action_budgets_and_reports_them():
    teacher = _RecordingTeacher()
    calls = []

    def run_joint_student(_video_x0, _action_x0, _text_emb, **kwargs):
        calls.append(kwargs)
        return np.zeros((1, 7, 4, 4, 1), dtype=np.float32)

    engine = CosmosProgressiveS4Engine(
        cosmos_teacher=teacher,
        prompt_table=PromptEmbeddingTable(
            {"open the drawer": np.zeros((1, 512, 4096), dtype=np.float32)}
        ),
        action_template=_template(),
        action_encoder=lambda raw_actions: raw_actions,
        joint_s4_runner=run_joint_student,
        anchor_noise_factory=lambda: np.ones((1, 16, 9, 28, 28), dtype=np.float32),
        video_steps=2,
        action_steps=2,
    )
    service = CosmosProgressiveS4Service(
        engine=engine,
        checkpoint_identifier="s4-checkpoint",
    )

    response = service.infer({"obs": OBS, "prompt": "open the drawer"})

    assert calls[0]["video_steps"] == 2
    assert calls[0]["action_steps"] == 2
    assert "k_steps" not in calls[0]
    assert response["video_steps"] == 2
    assert response["action_steps"] == 2
    assert response["student_steps"] == 2


def test_engine_rejects_mismatched_or_ambiguous_legacy_step_arguments():
    common = dict(
        cosmos_teacher=_RecordingTeacher(),
        prompt_table=PromptEmbeddingTable(
            {"open the drawer": np.zeros((1, 512, 4096), dtype=np.float32)}
        ),
        action_template=_template(),
        action_encoder=lambda raw_actions: raw_actions,
        joint_s4_runner=lambda *_args, **_kwargs: np.zeros((1, 7, 4, 4, 1), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="video_steps.*action_steps"):
        CosmosProgressiveS4Engine(**common, video_steps=1, action_steps=2)
    with pytest.raises(ValueError, match="student_steps"):
        CosmosProgressiveS4Engine(
            **common, video_steps=2, action_steps=2, student_steps=2
        )


class _MissingResetStepsService:
    def __init__(self):
        self.calls = 0

    def infer(self, request):
        self.calls += 1
        if request.get("reset"):
            return {"ok": True}
        return {
            "action": np.zeros((16, 7), dtype=np.float32),
            "student_steps": 2,
        }


def test_k2_client_rejects_reset_response_missing_student_steps(tmp_path):
    service = _MissingResetStepsService()
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
        episode_idx=2,
        prompt="open the drawer",
        max_env_steps=1,
        init_env_fn=lambda *_args, **_kwargs: OBS,
        extract_video_fn=lambda _obs: {},
        rollout_seed=30,
    )

    assert service.calls == 1
    assert record["server_failure"] is True
    assert record["student_steps"] == 2
    assert "missing student_steps" in record["error"]
    persisted = json.loads(
        (tmp_path / "records" / "task_5_episode_2.json").read_text(encoding="utf-8")
    )
    assert persisted["student_steps"] == 2


class _MissingInferStepsService:
    def infer(self, request):
        if request.get("reset"):
            return {"ok": True, "student_steps": 2}
        return {"action": np.zeros((16, 7), dtype=np.float32)}


def test_k2_client_rejects_infer_response_missing_student_steps(tmp_path):
    client = CosmosProgressiveS4Client(
        _MissingInferStepsService(),
        output_dir=tmp_path,
        student_steps=2,
        warmup_steps=0,
    )

    record = client.run_with_env(
        env=_DoneEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=5,
        episode_idx=3,
        prompt="open the drawer",
        max_env_steps=1,
        init_env_fn=lambda *_args, **_kwargs: OBS,
        extract_video_fn=lambda _obs: {},
        rollout_seed=30,
    )

    assert record["server_failure"] is True
    assert record["student_steps"] == 2
    assert "missing student_steps" in record["error"]
    persisted = json.loads(
        (tmp_path / "records" / "task_5_episode_3.json").read_text(encoding="utf-8")
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
    return rollout_module


def test_run_libero_task_persists_env_seed_for_skipped_record_without_libero(tmp_path, monkeypatch):
    _install_fake_libero(monkeypatch, skipped=True)
    client = CosmosProgressiveS4Client(
        _SuccessfulService(),
        output_dir=tmp_path,
        student_steps=2,
        expected_s4_checkpoint="s4-checkpoint",
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
    assert record["s4_checkpoint"] == "s4-checkpoint"
    persisted = json.loads(
        (tmp_path / "records" / "task_6_episode_0.json").read_text(encoding="utf-8")
    )
    assert persisted["seed"] == 31
    assert persisted["student_steps"] == 2
    assert persisted["s4_checkpoint"] == "s4-checkpoint"


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


@pytest.mark.parametrize(("save_video", "has_path"), [(False, False), (True, True)])
def test_run_libero_task_controls_video_capture(
    tmp_path, monkeypatch, save_video, has_path
):
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
            "video_path": (
                str(kwargs["video_path"]) if kwargs["video_path"] is not None else None
            ),
        }

    monkeypatch.setattr(client, "run_with_env", fake_run_with_env)

    record = client.run_libero_task(
        libero_benchmark="libero_10",
        task_idx=7,
        episode_idx=1,
        camera_size=128,
        max_env_steps=1,
        env_seed=37,
        save_video=save_video,
    )

    assert (record["video_path"] is not None) is has_path
    assert (captured["video_path"] is not None) is has_path


def test_run_libero_task_enabled_capture_extracts_and_saves_planned_video(
    tmp_path, monkeypatch
):
    rollout_module = _install_fake_libero(monkeypatch, skipped=False)
    client = CosmosProgressiveS4Client(
        _SuccessfulService(), output_dir=tmp_path, student_steps=2, warmup_steps=0
    )
    extracted = []
    saved = []
    frame = {"frame": "sentinel"}

    def extract_video(obs):
        extracted.append(obs)
        return frame

    rollout_module.extract_video_obs = extract_video
    rollout_module.save_video = lambda frames, path, *, fps: saved.append(
        (list(frames), path, fps)
    )

    record = client.run_libero_task(
        libero_benchmark="libero_10",
        task_idx=7,
        episode_idx=2,
        camera_size=128,
        max_env_steps=1,
        env_seed=37,
        save_video=True,
    )

    expected_path = (
        tmp_path
        / "libero_10"
        / "task_7_open_the_drawer"
        / "episode_2_done.mp4"
    )
    assert extracted == [OBS]
    assert saved == [([frame], expected_path, 15)]
    assert record["video_path"] == str(expected_path)


def test_run_libero_task_omitted_save_video_defaults_to_enabled(tmp_path, monkeypatch):
    rollout_module = _install_fake_libero(monkeypatch, skipped=False)
    client = CosmosProgressiveS4Client(
        _SuccessfulService(), output_dir=tmp_path, student_steps=2, warmup_steps=0
    )
    saved = []
    rollout_module.extract_video_obs = lambda _obs: {"frame": "default"}
    rollout_module.save_video = lambda frames, path, *, fps: saved.append(
        (list(frames), path, fps)
    )

    record = client.run_libero_task(
        libero_benchmark="libero_10",
        task_idx=7,
        episode_idx=3,
        camera_size=128,
        max_env_steps=1,
        env_seed=37,
    )

    assert len(saved) == 1
    assert record["video_path"] == str(saved[0][1])


def test_live_runtime_preflight_rejects_cpu_with_actionable_error(tmp_path):
    from evaluation.libero.rollout_cosmos_progressive_s4 import require_live_s4_prerequisites

    with pytest.raises(RuntimeError, match="CPU-only"):
        require_live_s4_prerequisites(
            device="cpu",
            checkpoint_transformer=tmp_path / "s4-checkpoint",
        )


def test_cosmos_policy_python_has_no_machine_local_fallback(monkeypatch):
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    monkeypatch.delenv("COSMOS_POLICY_PYTHON", raising=False)
    with pytest.raises(RuntimeError, match="COSMOS_POLICY_PYTHON must explicitly"):
        rollout.resolve_cosmos_policy_python()


def test_live_service_resolves_checkpoint_role_before_cuda_preflight(monkeypatch):
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    events = []
    args = SimpleNamespace(
        model_role="stage2_target",
        video_steps=2,
        action_steps=2,
        student_steps=None,
        checkpoint_transformer="/untrusted/checkpoint",
        device="cuda:0",
    )

    resolved = SimpleNamespace(transformer_path="/validated/target/transformer")
    monkeypatch.setattr(
        rollout,
        "resolve_cosmos_inference_checkpoint",
        lambda **kwargs: events.append(("resolve", kwargs)) or resolved,
    )

    def stop_at_preflight(**kwargs):
        events.append(("preflight", kwargs))
        raise RuntimeError("stop before model construction")

    monkeypatch.setattr(rollout, "require_live_s4_prerequisites", stop_at_preflight)
    with pytest.raises(RuntimeError, match="stop before model construction"):
        rollout.build_live_service(args)

    assert [name for name, _kwargs in events] == ["resolve", "preflight"]
    assert events[1][1]["checkpoint_transformer"] == "/validated/target/transformer"


def test_live_student_service_binds_validated_lineage_before_config_or_load():
    import inspect
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    source = inspect.getsource(rollout.build_live_service)
    bind_at = source.index("bind_stage2_inference_runtime(")
    config_at = source.index("_configure_live_config(", bind_at)
    student_load_at = source.index('dependencies["load_stage1_model"](', bind_at)

    assert bind_at < config_at < student_load_at
    assert (
        "args.teacher_model_path = "
        "resolved_checkpoint.cosmos_teacher_model_path"
    ) in source


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


@pytest.mark.parametrize(
    ("video_args", "expected_save_video"),
    [([], False), (["--save-video"], True)],
)
def test_live_rollout_forwards_cli_video_choice_to_every_task_call(
    monkeypatch, video_args, expected_save_video
):
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    calls = []

    class RecordingClient:
        def __init__(self, _service, **_kwargs):
            pass

        def run_libero_task(self, **kwargs):
            calls.append(kwargs)
            return {
                "task_idx": kwargs["task_idx"],
                "episode_idx": kwargs["episode_idx"],
            }

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
            "--env-seed",
            "41",
            "--task-range",
            "0",
            "2",
            "--episodes",
            "2",
            *video_args,
        ]
    )

    assert result == 0
    assert len(calls) == 4
    assert all(call["save_video"] is expected_save_video for call in calls)


def test_live_rollout_applies_episode_index_offset(monkeypatch):
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    calls = []

    class RecordingClient:
        def __init__(self, _service, **_kwargs):
            pass

        def run_libero_task(self, **kwargs):
            calls.append(kwargs)
            return {
                "task_idx": kwargs["task_idx"],
                "episode_idx": kwargs["episode_idx"],
            }

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
            "--env-seed",
            "17",
            "--episode-index-offset",
            "17",
            "--task-range",
            "0",
            "2",
            "--episodes",
            "1",
        ]
    )

    assert result == 0
    assert [call["episode_idx"] for call in calls] == [17, 17]


def test_live_rollout_seed_helper_pairs_python_numpy_and_torch_noise_across_k():
    import random

    import torch

    from evaluation.libero.rollout_cosmos_progressive_s4 import seed_live_rollout

    draws = []
    for _student_steps in (1, 2, 4):
        seed_live_rollout(23)
        draws.append(
            (
                random.random(),
                np.random.standard_normal(4),
                torch.randn(4),
            )
        )

    for candidate in draws[1:]:
        assert candidate[0] == draws[0][0]
        np.testing.assert_array_equal(candidate[1], draws[0][1])
        torch.testing.assert_close(candidate[2], draws[0][2], rtol=0, atol=0)


def test_live_main_reseeds_each_episode_after_k_dependent_prior_consumption(monkeypatch):
    import random

    import torch

    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    draws_by_k = {}

    def run_for_k(student_steps):
        draws = []

        class RecordingClient:
            def __init__(self, _service, **_kwargs):
                pass

            def run_libero_task(self, **kwargs):
                draws.append(
                    (
                        kwargs["task_idx"],
                        random.random(),
                        np.random.standard_normal(),
                        torch.randn(()).item(),
                    )
                )
                if kwargs["task_idx"] == 0:
                    for _ in range(student_steps * 7):
                        random.random()
                        np.random.standard_normal()
                        torch.randn(())
                return {"task_idx": kwargs["task_idx"]}

        monkeypatch.setattr(
            rollout,
            "build_live_service",
            lambda _args: (object(), SimpleNamespace(close=lambda: None)),
        )
        monkeypatch.setattr(rollout, "CosmosProgressiveS4Client", RecordingClient)
        rollout.main(
            [
                "--checkpoint-transformer",
                "/tmp/s4-transformer",
                "--prompt-table",
                "/tmp/training-prompt-table.pt",
                "--student-steps",
                str(student_steps),
                "--env-seed",
                "41",
                "--task-range",
                "0",
                "2",
            ]
        )
        draws_by_k[student_steps] = draws

    for student_steps in (1, 2, 4):
        run_for_k(student_steps)

    next_task_draws = [draws_by_k[k][1][1:] for k in (1, 2, 4)]
    assert next_task_draws[0] == next_task_draws[1] == next_task_draws[2]


def test_live_main_seeds_before_building_service(monkeypatch):
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    events = []

    class RecordingClient:
        def __init__(self, _service, **_kwargs):
            pass

    monkeypatch.setattr(
        rollout,
        "seed_live_rollout",
        lambda seed: events.append(("seed", seed)),
    )
    monkeypatch.setattr(
        rollout,
        "build_live_service",
        lambda _args: (
            events.append(("build", None)) or object(),
            SimpleNamespace(close=lambda: None),
        ),
    )
    monkeypatch.setattr(rollout, "CosmosProgressiveS4Client", RecordingClient)

    result = rollout.main(
        [
            "--checkpoint-transformer",
            "/tmp/s4-transformer",
            "--prompt-table",
            "/tmp/training-prompt-table.pt",
            "--env-seed",
            "41",
            "--task-range",
            "0",
            "0",
        ]
    )

    assert result == 0
    assert events == [("seed", 41), ("build", None)]


def test_preflight_neither_seeds_nor_builds_live_service(monkeypatch, capsys):
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    events = []
    monkeypatch.setattr(
        rollout,
        "require_live_s4_prerequisites",
        lambda **_kwargs: events.append(("preflight", None)),
    )
    monkeypatch.setattr(
        rollout,
        "seed_live_rollout",
        lambda seed: events.append(("seed", seed)),
    )
    monkeypatch.setattr(
        rollout,
        "build_live_service",
        lambda _args: events.append(("build", None)),
    )

    result = rollout.main(
        [
            "--checkpoint-transformer",
            "/tmp/s4-transformer",
            "--prompt-table",
            "/tmp/training-prompt-table.pt",
            "--preflight",
        ]
    )

    assert result == 0
    assert events == [("preflight", None)]
    note = capsys.readouterr().out
    assert "validated the configured Cosmos worker CUDA runtime" in note
    assert "did not start a Cosmos worker" in note
    assert (
        "did not start a Cosmos worker or prove Cosmos CUDA-extra compatibility"
        not in note
    )


def test_stdio_builds_service_without_seeding(monkeypatch):
    import evaluation.libero.rollout_cosmos_progressive_s4 as rollout

    events = []
    service = object()
    monkeypatch.setattr(
        rollout,
        "seed_live_rollout",
        lambda seed: events.append(("seed", seed)),
    )
    monkeypatch.setattr(
        rollout,
        "build_live_service",
        lambda _args: (
            events.append(("build", None)) or service,
            SimpleNamespace(close=lambda: events.append(("close", None))),
        ),
    )
    monkeypatch.setattr(
        rollout,
        "serve_json_lines",
        lambda received, **_kwargs: events.append(("serve", received)),
    )

    result = rollout.main(
        [
            "--checkpoint-transformer",
            "/tmp/s4-transformer",
            "--prompt-table",
            "/tmp/training-prompt-table.pt",
            "--serve-stdio",
        ]
    )

    assert result == 0
    assert events == [
        ("build", None),
        ("serve", service),
        ("close", None),
    ]


def test_cli_exposes_explicit_video_and_action_budgets_without_implicit_default():
    from evaluation.libero.rollout_cosmos_progressive_s4 import parse_args

    args = parse_args(
        [
            "--checkpoint-transformer",
            "/tmp/s4-transformer",
            "--prompt-table",
            "/tmp/training-prompt-table.pt",
        ]
    )

    assert args.student_steps is None
    assert args.video_steps is None
    assert args.action_steps is None
    assert args.save_video is False
    assert args.episode_index_offset == 0

    explicit = parse_args(
        [
            "--checkpoint-transformer", "/tmp/s4-transformer",
            "--prompt-table", "/tmp/training-prompt-table.pt",
            "--model-role", "stage2_target",
            "--video-steps", "4",
            "--action-steps", "4",
        ]
    )
    assert (explicit.model_role, explicit.video_steps, explicit.action_steps) == (
        "stage2_target", 4, 4
    )


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
        action_packing_schema="downsample_survivor_v2",
        action_downsample_factor=4,
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
    assert captured["packing_schema"] == "downsample_survivor_v2"
    assert captured["downsample_factor"] == 4


def test_action_packing_and_service_decode_round_trip_all_sixteen_actions():
    """The 4x4 carrier must preserve distinct raw actions, not merely its shape."""
    import torch

    from evaluation.libero.rollout_cosmos_progressive_s4 import FlowMapActionAnchorEncoder

    raw_actions = torch.linspace(-0.9, 0.9, 16 * 7, dtype=torch.float32).reshape(1, 16, 7)
    seen = {}

    def converter(actions, *, target_shape, **kwargs):
        seen.update(kwargs)
        seen["target_shape"] = target_shape
        seen["actions"] = actions.clone()
        packed = torch.zeros(target_shape, dtype=actions.dtype)
        packed[:, :, ::4, :, 0] = actions.reshape(1, 4, 4, 7).permute(0, 3, 1, 2)
        return packed

    config = SimpleNamespace(
        inverse_used_action_channel_ids=list(range(7)),
        action_per_frame=4,
        action_packing_schema="downsample_survivor_v2",
        action_downsample_factor=4,
        norm_stat={"q01": [-1.0] * 7, "q99": [1.0] * 7},
    )
    encoder = FlowMapActionAnchorEncoder(
        config=config, device="cpu", torch_module=torch, converter=converter
    )
    packed = encoder(raw_actions)
    decoded = decode_student_action(packed[:, :, ::4], _template())

    assert seen["actions"].shape == (1, 16, 7)
    assert seen["target_shape"] == (1, 7, 16, 4, 1)
    assert seen["downsample_factor"] == 4
    assert config.action_per_frame == 4
    np.testing.assert_allclose(decoded, raw_actions[0].numpy(), rtol=0, atol=1e-6)


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
        video_steps=steps,
        action_steps=steps,
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


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_joint_runner_cpu_nonlinear_trajectory_matches_training_integrator(steps):
    """Service must expose the exact states produced by the shared integrator."""
    import torch

    from evaluation.libero.rollout_cosmos_progressive_s4 import FlowMapJointS4Runner

    class NonlinearHarness:
        def __init__(self):
            self.calls = []
            self.action_trajectories = []

        def _student_euler_integrate(self, **kwargs):
            self.calls.append(kwargs)
            state = kwargs["noisy_latents"].float()
            trajectory = []
            action_state = kwargs["base_input_dict"]["action_dict"]["noisy_latents"].float()
            action_trajectory = []
            for index in range(kwargs["K_steps"]):
                state = state + torch.tanh(state + float(index + 1)) / float(index + 1)
                trajectory.append(state.clone())
                action_state = action_state + torch.sin(action_state + float(index + 1)) / float(index + 2)
                action_trajectory.append(action_state.clone())
            self.action_trajectories.append(tuple(action_trajectory))
            return state, torch.zeros_like(state), None, action_state, tuple(trajectory)

    def prepare_base_dict(_batch, _config, _device):
        return {"latent_dict": {}, "action_dict": {}, "chunk_size": 1, "window_size": 1}

    def paired(**kwargs):
        return (
            torch.full((1, kwargs["video_frames"]), float(kwargs["t"])),
            torch.full((1, kwargs["video_frames"]), float(kwargs["r"])),
            torch.full((1, kwargs["action_frames"]), float(kwargs["t"])),
            torch.full((1, kwargs["action_frames"]), float(kwargs["r"])),
        )

    def student_input(batch, base, video_x0, video_noise, video_t, action_noise, action_t, action_downsample):
        return ({"latent_dict": {"noisy_latents": video_noise, "timesteps": video_t},
                 "action_dict": {"noisy_latents": batch["actions"][:, :, ::action_downsample]},
                 "chunk_size": 1, "window_size": 1}, action_noise)

    harness = NonlinearHarness()
    runner = FlowMapJointS4Runner(
        harness=harness, config=SimpleNamespace(action_downsample_factor=4), device="cpu",
        torch_module=torch, prepare_base_dict=prepare_base_dict, student_input=student_input,
        build_paired_eval_timesteps=paired, empty_embedding=None, cfg_scale=1.0,
    )
    video = torch.full((1, 16, 9, 28, 28), 0.2)
    action = torch.zeros((1, 7, 16, 4, 1))
    text = torch.zeros((1, 512, 4096))
    service_action, service_trajectory = runner(
        video, action, text, noise=video, t1000=np.ones((1, 9)), t0=np.zeros((1, 9)),
        video_steps=steps, action_steps=steps, return_trajectory=True,
    )
    _final, _field, _sequence, training_action, training_trajectory = (
        harness._student_euler_integrate(**harness.calls[0])
    )
    assert len(service_trajectory) == steps
    assert len(training_trajectory) == steps
    for service_state, training_state in zip(service_trajectory, training_trajectory):
        torch.testing.assert_close(service_state, training_state, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(service_action, training_action, rtol=1e-4, atol=1e-4)
    for service_action_state, training_action_state in zip(
        harness.action_trajectories[0], harness.action_trajectories[1]
    ):
        torch.testing.assert_close(
            service_action_state, training_action_state, rtol=1e-4, atol=1e-4
        )
    service_call = harness.calls[0]
    training_call = harness.calls[1]
    torch.testing.assert_close(service_call["timesteps"], training_call["timesteps"])
    torch.testing.assert_close(
        service_call["action_target_r"], training_call["action_target_r"]
    )
    assert service_call["base_input_dict"]["action_dict"]["noisy_latents"].shape == (
        1, 7, 4, 4, 1
    )
