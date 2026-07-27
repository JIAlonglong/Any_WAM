import json
from types import SimpleNamespace

import pytest
import torch

import distillation_flowmap.eval_cosmos_progressive_s4_paper as paper_eval

from distillation_flowmap.eval_cosmos_progressive_s4_paper import (
    DEPLOYMENT_GRID,
    PAPER_METRIC_NAMES,
    S4_STUDENT_STEPS,
    S4_TEACHER_STEPS,
    cache_smoke_summary,
    compute_paper_metrics,
    grid_time,
    _load_test_manifest,
    parse_args,
    rollout_student_with_joint_action_trajectory,
    validate_s4_contract,
    write_paper_outputs,
)


def test_paper_evaluator_requires_full_s4_grid_and_test_manifest(tmp_path):
    args = parse_args([
        "--checkpoint-transformer", "/ckpt",
        "--dataset-path", "/data",
        "--manifest", "/proto/test_manifest.json",
        "--pairs", "/proto/eval_pairs.json",
        "--cache-dir", "/cache",
        "--output-dir", str(tmp_path),
    ])

    assert args.student_steps == 4
    assert args.teacher_steps == 8


def test_paper_metric_names_are_the_table_three_names():
    assert PAPER_METRIC_NAMES == ("g_anchor", "g_comp", "video_ep", "field_match")


def _latent(value):
    return torch.full((1, 16, 1, 1, 1), float(value))


def test_s4_contract_rejects_non_s4_steps_and_non_same_prior_pair():
    pair = {"pair_id": "same-prior", "t": 1000.0, "r": 0.0, "pair_seed": 7}

    assert DEPLOYMENT_GRID == (1000, 750, 500, 250, 0)
    assert validate_s4_contract(
        student_steps=S4_STUDENT_STEPS,
        teacher_steps=S4_TEACHER_STEPS,
        pairs=[pair],
    ) is None

    with pytest.raises(ValueError, match="student_steps=4"):
        validate_s4_contract(student_steps=3, teacher_steps=8, pairs=[pair])
    with pytest.raises(ValueError, match="same-prior"):
        validate_s4_contract(
            student_steps=4,
            teacher_steps=8,
            pairs=[{**pair, "r": 250.0}],
        )


def test_grid_time_converts_student_grid_to_normalized_cosmos_time():
    assert torch.equal(
        grid_time(750, batch_size=2, frames=3, device="cpu"),
        torch.full((2, 3), 0.75),
    )


def test_paper_metrics_use_joint_action_composition_and_dynamic_teacher_calls():
    joint_states = {
        750: (_latent(1), _latent(1)),
        500: (_latent(2), _latent(2)),
        250: (_latent(3), _latent(3)),
    }
    student_fields = {
        750: _latent(2),
        500: _latent(3),
        250: _latent(4),
    }
    continuation_calls = []

    def finite_student_map(video, action, *, r, target):
        if target == 0:
            return video, action
        return video + action, action + 1

    def teacher_continuation(video, *, r, target, steps):
        continuation_calls.append((r, target, steps))
        return video + 1

    def teacher_field(video, *, r):
        return video + 2

    metrics = compute_paper_metrics(
        y0=_latent(0),
        joint_states=joint_states,
        student_fields=student_fields,
        finite_student_map=finite_student_map,
        teacher_continuation=teacher_continuation,
        teacher_field=teacher_field,
    )

    assert metrics == {
        "g_anchor": {"node_750": 4.0, "node_500": 9.0, "node_250": 16.0},
        "g_comp": {"pair_750_500": 1.0, "pair_500_250": 4.0},
        "video_ep": {"node_750": 1.0, "node_500": 4.0, "node_250": 9.0},
        "field_match": {"node_750": 1.0, "node_500": 1.0, "node_250": 1.0},
    }
    assert continuation_calls == [(750, 0, 6), (500, 0, 4), (250, 0, 2)]


def test_paper_outputs_write_jsonl_and_task_macro_summary(tmp_path):
    metrics = {
        "g_anchor": {"node_750": 1.0},
        "g_comp": {"pair_750_500": 2.0},
        "video_ep": {"node_750": 3.0},
        "field_match": {"node_750": 4.0},
    }
    records = [
        {
            "task": "easy",
            "record_index": 0,
            "pair_id": "p0",
            "metrics": metrics,
        },
        {
            "task": "hard",
            "record_index": 1,
            "pair_id": "p0",
            "metrics": metrics,
        },
    ]

    summary = write_paper_outputs(
        tmp_path,
        records,
        metadata={"checkpoint_transformer": "/ckpt", "student_steps": 4},
    )

    assert [json.loads(line)["task"] for line in (
        tmp_path / "records.jsonl"
    ).read_text(encoding="utf-8").splitlines()] == ["easy", "hard"]
    assert summary["schema"] == "cosmos_progressive_s4_paper_metrics_v1"
    assert summary["is_paper_metric"] is True
    assert summary["per_task"]["easy"]["g_anchor/node_750"] == 1.0
    assert summary["metrics"]["field_match/node_750"] == 4.0
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8")) == summary


def test_cache_smoke_summary_is_explicitly_not_a_paper_metric():
    summary = cache_smoke_summary(
        checkpoint_transformer="/ckpt",
        validated_video_shapes={"video_noise": [1, 16, 9, 28, 28]},
    )

    assert summary["schema"] == "cosmos_progressive_s4_cache_smoke_v1"
    assert summary["is_paper_metric"] is False
    assert "metrics" not in summary
    assert "per_task" not in summary


def test_joint_trajectory_rebuilds_the_full_action_horizon_from_downsampled_state():
    class Config:
        action_downsample_factor = 4

    class Harness:
        config = Config()

        def __init__(self):
            self.action_target_frames = []

        def _student_euler_integrate(self, **kwargs):
            self.action_target_frames.append(kwargs["action_target_r"].shape[1])
            action_state = kwargs["base_input_dict"]["action_dict"]["noisy_latents"]
            video = kwargs["noisy_latents"]
            return video, torch.zeros_like(video), None, action_state

    video = _latent(0)
    action = torch.zeros(1, 2, 4, 1, 1)
    base_input = {
        "latent_dict": {"latent": video},
        "action_dict": {"latent": action, "noisy_latents": action},
        "chunk_size": 1,
        "window_size": 1,
    }
    harness = Harness()

    states, fields = rollout_student_with_joint_action_trajectory(
        harness,
        base_input=base_input,
        initial_video=video,
        initial_action=action,
        empty_emb=torch.empty(1),
        cfg_scale=3.0,
    )

    assert tuple(states[250][1].shape) == tuple(action.shape)
    assert set(fields) == {750, 500, 250, 0}
    assert harness.action_target_frames == [16, 16, 16, 16]


def test_test_manifest_rejects_non_string_task_identity(tmp_path):
    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "split": "test",
                "records": [{"index": 0, "task": None}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="task.*non-empty string"):
        _load_test_manifest(manifest_path)


def test_cache_only_main_uses_injected_runtime_before_any_cosmos_access(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "data"
    data_path.mkdir()
    torch.save(torch.zeros(1, 1, 1), data_path / "empty_emb.pt")
    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "split": "test",
                "records": [{"index": 0, "task": "task-a"}],
            }
        ),
        encoding="utf-8",
    )
    pairs_path = tmp_path / "eval_pairs.json"
    pairs_path.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": "same-prior",
                        "t": 1000,
                        "r": 0,
                        "pair_seed": 7,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    video = torch.zeros(1, 16, 1, 1, 1, dtype=torch.bfloat16)
    actions = torch.zeros(1, 2, 4, 1, 1, dtype=torch.bfloat16)
    cache = {
        "video_x0": video.clone(),
        "video_noise": video.clone(),
        "teacher_x_r": video.clone(),
        "teacher_v_r": video.clone(),
        "teacher_action_x0": actions.clone(),
    }
    cfg = SimpleNamespace(
        action_downsample_factor=4,
        teacher_model_path="/must-not-resolve-cosmos",
    )
    cosmos_calls = []
    validated = []
    harnesses = []

    class FakeDataset:
        def __init__(self, config):
            assert config is cfg

        def __getitem__(self, index):
            assert index == 0
            return {
                "latents": video.clone(),
                "actions": actions.clone(),
                "text_emb": torch.zeros(1, 1, 1, dtype=torch.bfloat16),
            }

    class FakeHarness:
        def __init__(self, student, config, device):
            assert config is cfg
            self.k_steps = []
            self.input_shapes = []
            harnesses.append(self)

        def _student_euler_integrate(self, **kwargs):
            self.k_steps.append(kwargs["K_steps"])
            self.input_shapes.append(tuple(kwargs["noisy_latents"].shape))
            return (
                kwargs["noisy_latents"],
                torch.zeros_like(kwargs["noisy_latents"]),
                None,
                kwargs["base_input_dict"]["action_dict"]["noisy_latents"],
            )

    def forbidden_cosmos(*args, **kwargs):
        cosmos_calls.append((args, kwargs))
        raise AssertionError("cache-only smoke touched Cosmos")

    def configure_offline_eval_config(config, *, skip_same_state_velocity):
        assert skip_same_state_velocity is True
        config.cosmos_policy_use_raw_inference = False
        config.return_raw_observation = False

    def prepare_base_dict(batch, config, device):
        return {
            "latent_dict": {
                "latent": batch["latents"],
                "cond_timesteps": torch.zeros(1, 1),
                "text_emb": batch["text_emb"],
            },
            "action_dict": {
                "latent": batch["actions"],
                "cond_timesteps": torch.zeros(1, 4),
                "text_emb": batch["text_emb"],
            },
            "chunk_size": 1,
            "window_size": 1,
        }

    def student_input(
        batch, base, video_x0, video_noise, video_t, action_noise, action_t, action_ds
    ):
        base_input = base["input"]
        return (
            {
                "latent_dict": {
                    **base_input["latent_dict"],
                    "noisy_latents": video_noise,
                    "timesteps": video_t,
                },
                "action_dict": {
                    **base_input["action_dict"],
                    "noisy_latents": batch["actions"][:, :, ::action_ds],
                    "timesteps": action_t[:, ::action_ds],
                },
                "chunk_size": base_input["chunk_size"],
                "window_size": base_input["window_size"],
            },
            action_noise,
        )

    def install_cached_teacher_anchor(batch, payload, *, device, dtype):
        retained = batch["actions"].clone()
        batch["latents"] = payload["video_x0"].to(device=device, dtype=dtype)
        batch["actions"] = payload["teacher_action_x0"].to(
            device=device, dtype=dtype
        )
        return retained

    runtime = {
        "CosmosPolicyActionTeacher": forbidden_cosmos,
        "SafeMultiLatentLeRobotDataset": FakeDataset,
        "StudentRolloutHarness": FakeHarness,
        "cache_payload": lambda *args, **kwargs: cache,
        "configure_offline_eval_config": configure_offline_eval_config,
        "default_collate": lambda items: items[0],
        "install_cached_teacher_anchor": install_cached_teacher_anchor,
        "load_stage1_model": lambda *args, **kwargs: object(),
        "move_batch": lambda batch, device: batch,
        "prepare_base_dict": prepare_base_dict,
        "resolve_cosmos_policy_assets": forbidden_cosmos,
        "student_input": student_input,
    }
    original_require_video = paper_eval._require_video_latent

    def spy_require_video(name, tensor, *, reference_shape=None):
        validated.append((name, tuple(tensor.shape)))
        return original_require_video(name, tensor, reference_shape=reference_shape)

    monkeypatch.setattr(
        paper_eval.importlib,
        "import_module",
        lambda _name: SimpleNamespace(cfg=cfg),
    )
    monkeypatch.setattr(paper_eval, "_runtime_dependencies", forbidden_cosmos)
    monkeypatch.setattr(paper_eval, "_require_video_latent", spy_require_video)

    output_dir = tmp_path / "out"
    summary = paper_eval.main(
        [
            "--checkpoint-transformer",
            str(tmp_path / "checkpoint"),
            "--dataset-path",
            str(data_path),
            "--manifest",
            str(manifest_path),
            "--pairs",
            str(pairs_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-dir",
            str(output_dir),
            "--cache-only-smoke",
            "--skip-same-state-velocity",
            "--device",
            "cpu",
        ],
        runtime=runtime,
    )

    assert cosmos_calls == []
    assert len(harnesses) == 1
    assert harnesses[0].k_steps == [4]
    assert harnesses[0].input_shapes == [(1, 16, 1, 1, 1)]
    assert {name for name, _shape in validated} >= {
        "video_noise",
        "cache smoke student video",
        "cache smoke student field",
    }
    assert summary["is_paper_metric"] is False
    assert "metrics" not in summary
    assert "per_task" not in summary
    assert summary["validated_video_shapes"]["student_video"] == [1, 16, 1, 1, 1]
    assert not (output_dir / "records.jsonl").exists()
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8")) == summary


@pytest.mark.parametrize(
    ("record", "pair", "message"),
    [
        ({"index": 0, "task": None}, {"pair_id": "same-prior"}, "task"),
        ({"index": 0, "task": "task-a"}, {"pair_id": None}, "pair_id"),
    ],
)
def test_validate_record_and_pair_requires_nonempty_string_identities(
    record, pair, message
):
    with pytest.raises(ValueError, match=message):
        paper_eval._validate_record_and_pair(record, pair)
