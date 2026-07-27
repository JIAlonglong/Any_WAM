import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


def _runtime_model(*, nfe=8, endpoint_dtype=torch.float32, sigma_min=4, sigma_max=80):
    captured = {}

    def get_x0_fn_from_batch(self, data_batch, guidance, **kwargs):
        return lambda value, sigma: value

    def generate_samples_from_batch(
        self,
        data_batch,
        *,
        x_sigma_max=None,
        num_steps=None,
        solver_option=None,
        sigma_max=None,
        guidance=None,
        use_variance_scale=None,
        seed=None,
    ):
        captured.update(
            data_batch=data_batch,
            x_sigma_max=x_sigma_max,
            num_steps=num_steps,
            solver_option=solver_option,
            sigma_max=sigma_max,
            guidance=guidance,
            use_variance_scale=use_variance_scale,
            seed=seed,
        )
        captured.setdefault("calls", []).append(
            {
                "data_batch": data_batch,
                "x_sigma_max": x_sigma_max,
                "sigma_max": sigma_max,
            }
        )
        x0_fn = self.get_x0_fn_from_batch(data_batch, guidance)
        sigma = torch.ones(x_sigma_max.shape[0])
        for _ in range(self.requested_nfe):
            x0_fn(x_sigma_max, sigma)
        captured["joint_prior"] = x_sigma_max.clone()
        return (x_sigma_max + 10).to(endpoint_dtype)

    model_type = type(
        "CosmosPolicyVideo2WorldModel",
        (torch.nn.Module,),
        {
            "__module__": (
                "cosmos_predict2._src.predict2.cosmos_policy.models."
                "policy_video2world_model"
            ),
            "get_x0_fn_from_batch": get_x0_fn_from_batch,
            "generate_samples_from_batch": generate_samples_from_batch,
        },
    )
    sampler_type = type(
        "CosmosPolicySampler",
        (torch.nn.Module,),
        {
            "__module__": (
                "cosmos_predict2._src.predict2.cosmos_policy.modules.cosmos_sampler"
            )
        },
    )
    sde_type = type(
        "HybridEDMSDE",
        (),
        {
            "__module__": (
                "cosmos_predict2._src.predict2.cosmos_policy.modules.hybrid_edm_sde"
            )
        },
    )

    class Tokenizer:
        spatial_compression_factor = 8

        @staticmethod
        def get_latent_num_frames(num_frames):
            return (num_frames - 1) // 4 + 1

    model = model_type()
    model.requested_nfe = nfe
    model.sampler = sampler_type()
    model.sde = sde_type()
    model.sde.sigma_min = sigma_min
    model.sde.sigma_max = sigma_max
    model.config = SimpleNamespace(
        use_flowunipc_scheduler=False,
        state_ch=16,
        state_t=9,
        min_num_conditional_frames=4,
    )
    model.tokenizer = Tokenizer()
    model.tensor_kwargs = {"device": "cpu"}
    return model, captured


def _libero_cfg(**overrides):
    fields = dict(
        suite="libero",
        use_proprio=True,
        normalize_proprio=True,
        use_wrist_image=True,
        num_wrist_images=1,
        use_third_person_image=True,
        num_third_person_images=1,
        use_jpeg_compression=True,
        trained_with_image_aug=True,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _data_batch(action_index=4):
    return {
        "video": torch.zeros(1, 3, 33, 224, 224, dtype=torch.uint8),
        "action_latent_idx": torch.tensor([action_index]),
    }


def _latent_indices():
    return {
        "future_wrist_image_latent_idx": 6,
        "future_wrist_image2_latent_idx": -1,
        "future_image_latent_idx": 7,
        "future_image2_latent_idx": -1,
    }


def _cosmos_layout_module(repo):
    source = (
        repo
        / "cosmos_predict2"
        / "_src"
        / "predict2"
        / "cosmos_policy"
        / "experiments"
        / "robot"
        / "cosmos_utils.py"
    )
    source.parent.mkdir(parents=True)
    source.write_text("# audited layout source fixture\n")
    return SimpleNamespace(__file__=str(source))


def _install_git_provenance(
    monkeypatch,
    worker,
    repo,
    *,
    head="583ba1d85b51148e898bdd4232cfeee73f6fce1e",
    status="",
    metadata_available=True,
):
    def fake_run(command, **kwargs):
        assert command[:3] == ["/usr/bin/git", "-C", str(repo.resolve())]
        operation = tuple(command[3:])
        if operation == ("rev-parse", "--show-toplevel"):
            if not metadata_available:
                return SimpleNamespace(
                    returncode=128,
                    stdout="",
                    stderr="fatal: not a git repository",
                )
            return SimpleNamespace(
                returncode=0,
                stdout=str(repo.resolve()) + "\n",
                stderr="",
            )
        if operation == ("rev-parse", "HEAD"):
            return SimpleNamespace(returncode=0, stdout=head + "\n", stderr="")
        if operation == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return SimpleNamespace(returncode=0, stdout=status, stderr="")
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(worker.subprocess, "run", fake_run)


def test_same_prior_model_load_accepts_only_audited_clean_layout_commit(
    monkeypatch, tmp_path
):
    import distillation_flowmap.cosmos_policy_raw_worker as worker

    repo = tmp_path / "cosmos"
    cosmos_utils = _cosmos_layout_module(repo)
    _install_git_provenance(monkeypatch, worker, repo)
    sentinel = object()
    loader_calls = []

    def get_model(cfg):
        loader_calls.append(cfg)
        return sentinel, None

    cfg = SimpleNamespace()
    loaded = worker._model_for_request(
        None,
        mode="same_prior_endpoint",
        repo=str(repo),
        cosmos_utils=cosmos_utils,
        cfg=cfg,
        get_model=get_model,
    )

    assert loaded is sentinel
    assert loader_calls == [cfg]


def test_raw_actions_model_load_uses_same_audited_clean_repository_gate(
    monkeypatch, tmp_path
):
    import distillation_flowmap.cosmos_policy_raw_worker as worker

    repo = tmp_path / "cosmos"
    cosmos_utils = _cosmos_layout_module(repo)
    _install_git_provenance(
        monkeypatch,
        worker,
        repo,
        head="0000000000000000000000000000000000000000",
    )
    loader_calls = []

    with pytest.raises(RuntimeError, match="audited commit"):
        worker._model_for_request(
            None,
            mode="actions",
            repo=str(repo),
            cosmos_utils=cosmos_utils,
            cfg=SimpleNamespace(),
            get_model=lambda cfg: loader_calls.append(cfg),
        )

    assert loader_calls == []


def test_audited_cosmos_source_identity_records_commit_and_source_digest(
    monkeypatch, tmp_path
):
    import distillation_flowmap.cosmos_policy_raw_worker as worker

    repo = tmp_path / "cosmos"
    cosmos_utils = _cosmos_layout_module(repo)
    _install_git_provenance(monkeypatch, worker, repo)

    identity = worker._validate_same_prior_layout_source(repo, cosmos_utils)

    assert identity == {
        "cosmos_repo_commit": "583ba1d85b51148e898bdd4232cfeee73f6fce1e",
        "cosmos_source_sha256": hashlib.sha256(
            Path(cosmos_utils.__file__).read_bytes()
        ).hexdigest(),
    }


def test_joint_continuation_model_load_uses_same_audited_layout_gate(
    monkeypatch, tmp_path
):
    import distillation_flowmap.cosmos_policy_raw_worker as worker

    repo = tmp_path / "cosmos"
    cosmos_utils = _cosmos_layout_module(repo)
    _install_git_provenance(
        monkeypatch,
        worker,
        repo,
        head="0000000000000000000000000000000000000000",
    )
    loader_calls = []

    with pytest.raises(RuntimeError, match="audited commit"):
        worker._model_for_request(
            None,
            mode="joint_continuation_endpoint",
            repo=str(repo),
            cosmos_utils=cosmos_utils,
            cfg=SimpleNamespace(),
            get_model=lambda cfg: loader_calls.append(cfg),
        )

    assert loader_calls == []


@pytest.mark.parametrize(
    ("git_fields", "match"),
    [
        (
            {"head": "0000000000000000000000000000000000000000"},
            "audited commit",
        ),
        (
            {
                "status": (
                    " M cosmos_predict2/_src/predict2/cosmos_policy/"
                    "experiments/robot/cosmos_utils.py\n"
                )
            },
            "must be clean",
        ),
        ({"metadata_available": False}, "Git metadata"),
    ],
)
def test_same_prior_provenance_rejection_precedes_model_invocation(
    monkeypatch, tmp_path, git_fields, match
):
    import distillation_flowmap.cosmos_policy_raw_worker as worker

    repo = tmp_path / "cosmos"
    cosmos_utils = _cosmos_layout_module(repo)
    _install_git_provenance(monkeypatch, worker, repo, **git_fields)
    loader_calls = []

    def get_model(cfg):
        loader_calls.append(cfg)
        return object(), None

    with pytest.raises(RuntimeError, match=match):
        worker._model_for_request(
            None,
            mode="same_prior_endpoint",
            repo=str(repo),
            cosmos_utils=cosmos_utils,
            cfg=SimpleNamespace(),
            get_model=get_model,
        )

    assert loader_calls == []


def test_same_prior_rejects_layout_module_imported_outside_selected_repo(
    monkeypatch, tmp_path
):
    import distillation_flowmap.cosmos_policy_raw_worker as worker

    repo = tmp_path / "selected-cosmos"
    _cosmos_layout_module(repo)
    outside_module = _cosmos_layout_module(tmp_path / "other-cosmos")
    loader_calls = []

    with pytest.raises(RuntimeError, match="imported from selected repository"):
        worker._model_for_request(
            None,
            mode="same_prior_endpoint",
            repo=str(repo),
            cosmos_utils=outside_module,
            cfg=SimpleNamespace(),
            get_model=lambda cfg: loader_calls.append(cfg),
        )

    assert loader_calls == []


def test_explicit_prior_sampler_proves_runtime_and_observes_exact_edm_budget():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _generate_from_explicit_prior,
    )

    model, captured = _runtime_model()
    prior = torch.randn(1, 16, 9, 28, 28)
    batch = _data_batch()
    endpoint, effective_steps = _generate_from_explicit_prior(
        model, batch, prior, teacher_steps=8
    )

    assert captured["data_batch"] is batch
    assert captured["x_sigma_max"] is prior
    assert captured["num_steps"] == 8
    assert captured["solver_option"] == "2ab"
    assert captured["sigma_max"] == 80
    assert captured["guidance"] == 0
    assert captured["use_variance_scale"] is False
    assert captured["seed"] is None
    assert effective_steps == 8
    assert torch.equal(endpoint, prior + 10)


def test_same_signature_wrong_runtime_class_fails_before_invocation():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _generate_from_explicit_prior,
    )

    class CompatibleLookingModel:
        calls = 0

        def generate_samples_from_batch(self, data_batch, *, x_sigma_max, **kwargs):
            self.calls += 1
            return x_sigma_max

    model = CompatibleLookingModel()
    with pytest.raises(RuntimeError, match="runtime model"):
        _generate_from_explicit_prior(
            model, {}, torch.zeros(1, 16, 9, 28, 28), teacher_steps=8
        )
    assert model.calls == 0


@pytest.mark.parametrize(
    "model_kwargs,match",
    [
        ({"sigma_min": 3}, "sigma_min"),
        ({"sigma_max": 79}, "sigma_max"),
    ],
)
def test_explicit_prior_sampler_rejects_sde_drift(model_kwargs, match):
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _generate_from_explicit_prior,
    )

    model, _ = _runtime_model(**model_kwargs)
    with pytest.raises(RuntimeError, match=match):
        _generate_from_explicit_prior(
            model, _data_batch(), torch.zeros(1, 16, 9, 28, 28), teacher_steps=8
        )


def test_explicit_prior_sampler_rejects_observed_non_eight_nfe():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _generate_from_explicit_prior,
    )

    model, _ = _runtime_model(nfe=7)
    with pytest.raises(RuntimeError, match="observed 7"):
        _generate_from_explicit_prior(
            model, _data_batch(), torch.zeros(1, 16, 9, 28, 28), teacher_steps=8
        )


@pytest.mark.parametrize("teacher_steps", [8.5, "8", True])
def test_worker_rejects_non_integer_teacher_budget(teacher_steps):
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _generate_from_explicit_prior,
    )

    model, _ = _runtime_model()
    with pytest.raises((TypeError, ValueError), match="integer"):
        _generate_from_explicit_prior(
            model,
            _data_batch(),
            torch.zeros(1, 16, 9, 28, 28),
            teacher_steps=teacher_steps,
        )


@pytest.mark.parametrize(
    "cfg_overrides,match",
    [
        ({"suite": "robocasa"}, "suite"),
        ({"use_proprio": False}, "use_proprio"),
        ({"normalize_proprio": False}, "normalize_proprio"),
        ({"num_wrist_images": 2}, "num_wrist_images"),
        ({"num_third_person_images": 2}, "num_third_person_images"),
        ({"trained_with_image_aug": False}, "trained_with_image_aug"),
    ],
)
def test_official_libero_geometry_rejects_config_drift(cfg_overrides, match):
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _validate_official_libero_geometry,
    )

    model, _ = _runtime_model()
    with pytest.raises(RuntimeError, match=match):
        _validate_official_libero_geometry(_libero_cfg(**cfg_overrides), model)


def test_official_libero_geometry_rejects_model_or_tokenizer_drift():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _validate_official_libero_geometry,
    )

    model, _ = _runtime_model()
    model.config.state_t = 8
    with pytest.raises(RuntimeError, match="state_t"):
        _validate_official_libero_geometry(_libero_cfg(), model)

    model, _ = _runtime_model()
    model.tokenizer.spatial_compression_factor = 4
    with pytest.raises(RuntimeError, match="spatial"):
        _validate_official_libero_geometry(_libero_cfg(), model)


def test_local_libero_batch_builder_matches_pinned_official_layout():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _build_libero_data_batch,
    )

    model, _ = _runtime_model()

    class CosmosUtils:
        COSMOS_IMAGE_SIZE = 224
        COSMOS_TEMPORAL_COMPRESSION_FACTOR = 4

        @staticmethod
        def get_t5_embedding_from_cache(task):
            return torch.zeros(1, 2, 3)

        @staticmethod
        def prepare_images_for_model(images, cfg):
            return [
                np.full((224, 224, 3), 2, dtype=np.uint8),
                np.full((224, 224, 3), 3, dtype=np.uint8),
            ]

        @staticmethod
        def rescale_proprio(value, *args, **kwargs):
            return value

        @staticmethod
        def duplicate_array(value, total_num_copies):
            return np.repeat(value[None], total_num_copies, axis=0)

    data_batch, indices = _build_libero_data_batch(
        _libero_cfg(),
        model,
        {},
        {
            "wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "primary_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "proprio": np.zeros(9, dtype=np.float32),
        },
        "task",
        CosmosUtils,
    )

    video = data_batch["video"]
    assert video.shape == (1, 3, 33, 224, 224)
    assert torch.count_nonzero(video[:, :, 0:5]) == 0
    assert torch.all(video[:, :, 5:9] == 2)
    assert torch.all(video[:, :, 9:13] == 3)
    assert torch.count_nonzero(video[:, :, 13:21]) == 0
    assert torch.all(video[:, :, 21:25] == 2)
    assert torch.all(video[:, :, 25:29] == 3)
    assert torch.count_nonzero(video[:, :, 29:33]) == 0
    assert indices["action_latent_idx"] == 4
    assert indices["future_wrist_image_latent_idx"] == 6
    assert indices["future_image_latent_idx"] == 7
    assert data_batch["action_latent_idx"].tolist() == [4]


def test_same_prior_worker_requires_exact_joint_geometry_and_preserves_video():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _run_same_prior_endpoint,
    )

    model, captured = _runtime_model()
    video_prior = torch.arange(
        1 * 16 * 9 * 28 * 28, dtype=torch.float32
    ).reshape(1, 16, 9, 28, 28)
    action_prior = torch.arange(1 * 16 * 7, dtype=torch.float32).reshape(1, 16, 7)
    result = _run_same_prior_endpoint(
        model,
        _data_batch(),
        _latent_indices(),
        video_prior,
        action_prior,
        teacher_steps=8,
    )

    edm_joint = captured["joint_prior"]
    packed = action_prior.flatten().repeat(112)[: 16 * 28 * 28].reshape(16, 28, 28)
    assert torch.equal(edm_joint[0, :, 4], packed * 80.0)
    assert torch.equal(edm_joint[0, :, 6], video_prior[0, :, 6] * 80.0)
    assert torch.equal(edm_joint[0, :, 7], video_prior[0, :, 7] * 80.0)
    assert result["video_frame_mask"].tolist() == [
        [False, False, False, False, False, False, True, True, False]
    ]
    assert torch.equal(result["endpoint_video"], edm_joint + 10)
    assert result["effective_teacher_steps"] == 8

    with pytest.raises(ValueError, match=r"\[B,16,9,28,28\]"):
        _run_same_prior_endpoint(
            model,
            _data_batch(),
            _latent_indices(),
            torch.zeros(1, 16, 9, 14, 14),
            action_prior,
            teacher_steps=8,
        )


def test_joint_continuation_converts_each_normalized_state_to_its_own_edm_sigma():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _run_joint_continuation_endpoint,
    )

    model, captured = _runtime_model()
    canonical = torch.stack(
        (
            torch.full((16, 9, 28, 28), 2.0),
            torch.full((16, 9, 28, 28), 3.0),
        )
    )
    normalized_t = torch.tensor(
        [[5.0 / 6.0] * 9, [15.0 / 16.0] * 9],
        dtype=torch.float32,
    )
    data_batch = {
        "video": torch.zeros(2, 3, 33, 224, 224, dtype=torch.uint8),
        "action_latent_idx": torch.tensor([4, 4]),
    }

    result = _run_joint_continuation_endpoint(
        model,
        data_batch,
        _latent_indices(),
        canonical,
        normalized_t,
        teacher_steps=8,
    )

    assert len(captured["calls"]) == 2
    assert captured["calls"][0]["sigma_max"] == pytest.approx(5.0)
    assert captured["calls"][1]["sigma_max"] == pytest.approx(15.0)
    assert isinstance(captured["calls"][0]["sigma_max"], float)
    torch.testing.assert_close(
        captured["calls"][0]["x_sigma_max"],
        canonical[0:1] * 6.0,
    )
    torch.testing.assert_close(
        captured["calls"][1]["x_sigma_max"],
        canonical[1:2] * 16.0,
    )
    assert result["effective_teacher_steps"] == 8
    torch.testing.assert_close(
        result["edm_sigma"], torch.tensor([5.0, 15.0])
    )


def test_joint_continuation_rejects_per_frame_time_disagreement_before_sampler():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _run_joint_continuation_endpoint,
    )

    model, captured = _runtime_model()
    normalized_t = torch.full((1, 9), 5.0 / 6.0, dtype=torch.float32)
    normalized_t[0, 3] = 15.0 / 16.0

    with pytest.raises(ValueError, match="constant across frames"):
        _run_joint_continuation_endpoint(
            model,
            _data_batch(),
            _latent_indices(),
            torch.zeros(1, 16, 9, 28, 28),
            normalized_t,
            teacher_steps=8,
        )

    assert captured.get("calls", []) == []


def test_joint_continuation_rejects_empty_video_mask_before_sampler():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _run_joint_continuation_endpoint,
    )

    model, captured = _runtime_model()
    empty_indices = {key: -1 for key in _latent_indices()}
    with pytest.raises(RuntimeError, match="nonempty"):
        _run_joint_continuation_endpoint(
            model,
            _data_batch(),
            empty_indices,
            torch.zeros(1, 16, 9, 28, 28),
            torch.full((1,), 5.0 / 6.0),
            teacher_steps=8,
        )

    assert captured.get("calls", []) == []


def test_worker_main_dispatches_joint_continuation_with_audited_response(
    monkeypatch, tmp_path
):
    import distillation_flowmap.cosmos_policy_raw_worker as worker

    canonical = np.stack(
        (
            np.full((16, 9, 28, 28), 2.0, dtype=np.float32),
            np.full((16, 9, 28, 28), 3.0, dtype=np.float32),
        )
    )
    normalized_t = np.asarray(
        [[5.0 / 6.0] * 9, [15.0 / 16.0] * 9],
        dtype=np.float32,
    )
    request_path = tmp_path / "request.npz"
    response_path = tmp_path / "response.npz"
    np.savez_compressed(
        request_path,
        primary_image=np.zeros((2, 2, 2, 3), dtype=np.uint8),
        wrist_image=np.zeros((2, 2, 2, 3), dtype=np.uint8),
        proprio=np.zeros((2, 9), dtype=np.float32),
        canonical_joint_state=canonical,
        normalized_t=normalized_t,
    )
    request = {
        "npz_path": str(request_path),
        "actions_path": str(response_path),
        "tasks": ["task-a", "task-b"],
        "mode": "joint_continuation_endpoint",
        "teacher_steps": 8,
    }
    monkeypatch.setattr(
        worker,
        "_parse_args",
        lambda: SimpleNamespace(
            checkpoint_dir="checkpoint",
            repo="/audited/cosmos",
            config_name="config",
            config_file="config.py",
            dataset_stats_path="stats.json",
            t5_embeddings_path="t5.pkl",
            extra_pythonpath="",
            num_denoising_steps_action=5,
            seed=1,
        ),
    )

    robot_name = (
        "cosmos_predict2._src.predict2.cosmos_policy.experiments.robot"
    )
    cosmos_utils = ModuleType(f"{robot_name}.cosmos_utils")
    cosmos_utils.get_action = lambda *args, **kwargs: None
    cosmos_utils.get_model = lambda cfg: (object(), None)
    cosmos_utils.init_t5_text_embeddings_cache = lambda *args, **kwargs: None
    cosmos_utils.load_dataset_stats = lambda path: {}
    robot_module = ModuleType(robot_name)
    robot_module.cosmos_utils = cosmos_utils
    for name in (
        "cosmos_predict2",
        "cosmos_predict2._src",
        "cosmos_predict2._src.predict2",
        "cosmos_predict2._src.predict2.cosmos_policy",
        "cosmos_predict2._src.predict2.cosmos_policy.experiments",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    robot_module.__path__ = []
    monkeypatch.setitem(sys.modules, robot_name, robot_module)
    monkeypatch.setitem(sys.modules, f"{robot_name}.cosmos_utils", cosmos_utils)

    model = object()
    gates = []

    def model_for_request(current, *, mode, **kwargs):
        del current, kwargs
        gates.append(("layout", mode))
        return model

    monkeypatch.setattr(worker, "_model_for_request", model_for_request)
    monkeypatch.setattr(
        worker,
        "_validate_explicit_prior_runtime",
        lambda received: gates.append(("runtime", received)),
    )
    monkeypatch.setattr(
        worker,
        "_validate_official_libero_geometry",
        lambda cfg, received: gates.append(("geometry", received)),
    )
    monkeypatch.setattr(
        worker,
        "_build_libero_data_batch",
        lambda cfg, received, stats, obs, task, utils: (
            {"video": torch.zeros(1, 3, 33, 224, 224)},
            _latent_indices(),
        ),
    )
    calls = []

    def run_continuation(
        received_model,
        data_batch,
        latent_indices,
        state,
        time_value,
        *,
        teacher_steps,
    ):
        del data_batch, latent_indices
        calls.append(
            (
                received_model,
                state.detach().cpu().clone(),
                time_value.detach().cpu().clone(),
                teacher_steps,
            )
        )
        per_sample_t = time_value[:, 0]
        return {
            "endpoint_video": state + 1,
            "video_frame_mask": torch.ones(1, 9, dtype=torch.bool),
            "effective_teacher_steps": 8,
            "normalized_t": per_sample_t,
            "edm_sigma": per_sample_t / (1.0 - per_sample_t),
        }

    monkeypatch.setattr(
        worker, "_run_joint_continuation_endpoint", run_continuation
    )
    response_stream = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    monkeypatch.setattr(sys, "stdout", response_stream)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    worker.main()

    payload = json.loads(response_stream.getvalue())
    assert payload["ok"] is True
    assert gates == [
        ("layout", "joint_continuation_endpoint"),
        ("runtime", model),
        ("geometry", model),
    ]
    assert len(calls) == 2
    assert [call[3] for call in calls] == [8, 8]
    torch.testing.assert_close(calls[0][1], torch.from_numpy(canonical[0:1]))
    torch.testing.assert_close(
        calls[1][2], torch.from_numpy(normalized_t[1:2])
    )
    with np.load(response_path) as response:
        np.testing.assert_array_equal(
            response["endpoint_video"], canonical + 1
        )
        assert response["video_frame_mask"].dtype == np.bool_
        assert int(response["effective_teacher_steps"]) == 8
        assert str(response["joint_state_sha256"]) == worker._array_sha256(
            canonical
        )
        assert str(response["normalized_t_sha256"]) == worker._array_sha256(
            normalized_t
        )
        np.testing.assert_allclose(
            response["edm_sigma"], np.asarray([5.0, 15.0]), rtol=1e-6
        )


def test_same_prior_npz_loader_rejects_dtype_before_fingerprinting():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _load_same_prior_arrays,
    )

    float64_video = np.zeros((1, 16, 9, 28, 28), dtype=np.float64)
    with pytest.raises(ValueError, match="video_prior.*float32"):
        _load_same_prior_arrays(
            {
                "video_prior": float64_video,
                "action_prior": np.zeros((1, 16, 7), dtype=np.float32),
            }
        )

    video = np.arange(16 * 9 * 28 * 28, dtype=np.float32).reshape(
        1, 16, 9, 28, 28
    )
    action = np.arange(16 * 7, dtype=np.float32).reshape(1, 16, 7)
    loaded_video, loaded_action, video_sha, action_sha = _load_same_prior_arrays(
        {"video_prior": video, "action_prior": action}
    )
    assert loaded_video.dtype == np.float32
    assert loaded_action.dtype == np.float32
    assert video_sha == hashlib.sha256(
        np.ascontiguousarray(video).view(np.uint8)
    ).hexdigest()
    assert action_sha == hashlib.sha256(
        np.ascontiguousarray(action).view(np.uint8)
    ).hexdigest()


@pytest.mark.parametrize(
    "endpoint_dtype,mask_dtype,match",
    [
        (torch.bfloat16, torch.bool, "endpoint.*float32"),
        (torch.float32, torch.uint8, "mask.*boolean"),
    ],
)
def test_same_prior_response_serialization_rejects_backend_dtype_drift(
    endpoint_dtype, mask_dtype, match
):
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _same_prior_response_fields,
    )

    with pytest.raises(RuntimeError, match=match):
        _same_prior_response_fields(
            {
                "endpoint_video": torch.zeros(
                    1, 16, 9, 28, 28, dtype=endpoint_dtype
                ),
                "video_frame_mask": torch.zeros(1, 9, dtype=mask_dtype),
                "effective_teacher_steps": 8,
            },
            video_prior_sha256="v",
            action_prior_sha256="a",
        )


def test_worker_main_closes_loaded_npz_when_request_fails(monkeypatch):
    import distillation_flowmap.cosmos_policy_raw_worker as worker

    fake_data = {
        "primary_image": np.zeros((0, 2, 2, 3), dtype=np.uint8),
        "wrist_image": np.zeros((0, 2, 2, 3), dtype=np.uint8),
        "proprio": np.zeros((0, 9), dtype=np.float32),
    }

    class LoadedNpz:
        closed = False

        def __getitem__(self, key):
            return fake_data[key]

        def close(self):
            self.closed = True

    loaded = LoadedNpz()
    monkeypatch.setattr(worker.np, "load", lambda path: loaded)
    monkeypatch.setattr(
        worker,
        "_parse_args",
        lambda: SimpleNamespace(
            checkpoint_dir="checkpoint",
            repo="",
            config_name="config",
            config_file="config.py",
            dataset_stats_path="stats.json",
            t5_embeddings_path="t5.pkl",
            extra_pythonpath="",
            num_denoising_steps_action=5,
            seed=1,
        ),
    )

    robot_name = (
        "cosmos_predict2._src.predict2.cosmos_policy.experiments.robot"
    )
    cosmos_utils = ModuleType(f"{robot_name}.cosmos_utils")
    cosmos_utils.get_action = lambda *args, **kwargs: None
    cosmos_utils.get_model = lambda cfg: (object(), None)
    cosmos_utils.init_t5_text_embeddings_cache = lambda *args, **kwargs: None
    cosmos_utils.load_dataset_stats = lambda path: {}
    robot_module = ModuleType(robot_name)
    robot_module.cosmos_utils = cosmos_utils
    for name in (
        "cosmos_predict2",
        "cosmos_predict2._src",
        "cosmos_predict2._src.predict2",
        "cosmos_predict2._src.predict2.cosmos_policy",
        "cosmos_predict2._src.predict2.cosmos_policy.experiments",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    robot_module.__path__ = []
    monkeypatch.setitem(sys.modules, robot_name, robot_module)
    monkeypatch.setitem(sys.modules, f"{robot_name}.cosmos_utils", cosmos_utils)

    request = {
        "npz_path": "request.npz",
        "actions_path": "response.npz",
        "tasks": [],
        "mode": "unsupported",
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request) + "\n"))
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    worker.main()

    assert loaded.closed is True
