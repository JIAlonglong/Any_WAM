import json
from types import SimpleNamespace

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
    engine = CosmosProgressiveS4Engine(
        cosmos_teacher=teacher,
        prompt_table=PromptEmbeddingTable(
            {"open the drawer": np.zeros((1, 512, 4096), dtype=np.float32)}
        ),
        action_template=template,
        action_encoder=lambda raw_actions: raw_actions,
        joint_s4_runner=lambda video_x0, action_x0, text_emb, **_: np.zeros(
            (1, 7, 4, 4, 1), dtype=np.float32
        ),
        anchor_noise_factory=lambda: np.ones((1, 16, 9, 28, 28), dtype=np.float32),
    )
    service = CosmosProgressiveS4Service(
        engine=engine,
        checkpoint_identifier="s4-checkpoint",
        anchor_record_dir=tmp_path / "anchors",
    )

    first = service.infer({"obs": OBS, "prompt": "open the drawer"})
    second = service.infer({"obs": OBS, "prompt": "open the drawer"})

    assert len(teacher.calls) == 2
    assert all(call["include_cdiff"] is False for call in teacher.calls)
    assert first["action"].shape == (16, 7)
    assert first["action"].dtype == np.float32
    assert first["raw_anchor_record"] != second["raw_anchor_record"]
    assert (tmp_path / "anchors" / first["raw_anchor_record"]).is_file()
    assert first["s4_checkpoint"] == "s4-checkpoint"
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
    client = CosmosProgressiveS4Client(_FailingService(), output_dir=tmp_path, warmup_steps=1)

    record = client.run_with_env(
        env=_WarmupEnv(),
        initial_state=np.zeros(1, dtype=np.float32),
        task_idx=2,
        episode_idx=5,
        prompt="open the drawer",
        max_env_steps=2,
    )

    assert record["success"] is False
    assert record["done"] is False
    assert record["server_failure"] is True
    record_path = tmp_path / "records" / "task_2_episode_5.json"
    assert json.loads(record_path.read_text(encoding="utf-8"))["success"] is False


def test_live_runtime_preflight_rejects_cpu_with_actionable_error(tmp_path):
    from evaluation.libero.rollout_cosmos_progressive_s4 import require_live_s4_prerequisites

    with pytest.raises(RuntimeError, match="CPU-only"):
        require_live_s4_prerequisites(
            device="cpu",
            checkpoint_transformer=tmp_path / "s4-checkpoint",
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
