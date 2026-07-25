import pytest
import torch


def test_explicit_prior_sampler_uses_exact_edm_protocol():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _generate_from_explicit_prior,
    )

    captured = {}

    class ExplicitPriorModel:
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
            return x_sigma_max + 1

    prior = torch.randn(1, 3, 5, 2, 2)
    batch = {"action_latent_idx": torch.tensor([1])}
    endpoint = _generate_from_explicit_prior(
        ExplicitPriorModel(), batch, prior, teacher_steps=8
    )

    assert captured["data_batch"] is batch
    assert captured["x_sigma_max"] is prior
    assert captured["num_steps"] == 8
    assert captured["solver_option"] == "2ab"
    assert captured["sigma_max"] == 80
    assert captured["guidance"] == 0
    assert captured["use_variance_scale"] is False
    assert captured["seed"] is None
    assert torch.equal(endpoint, prior + 1)


def test_seed_only_or_kwargs_backend_fails_closed_without_invocation():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _generate_from_explicit_prior,
    )

    class SeedOnlyModel:
        calls = 0

        def generate_samples_from_batch(self, data_batch, *, seed=1, **kwargs):
            self.calls += 1
            return torch.zeros(1)

    model = SeedOnlyModel()
    with pytest.raises(RuntimeError, match="cannot honor explicit same-prior"):
        _generate_from_explicit_prior(
            model, {}, torch.zeros(1, 3, 5, 2, 2), teacher_steps=8
        )
    assert model.calls == 0


def test_same_prior_worker_packs_action_without_changing_video_frames():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _run_same_prior_endpoint,
    )

    video_prior = torch.arange(1 * 16 * 5 * 2 * 2, dtype=torch.float32).reshape(
        1, 16, 5, 2, 2
    )
    action_prior = torch.arange(1 * 16 * 7, dtype=torch.float32).reshape(1, 16, 7)
    data_batch = {"action_latent_idx": torch.tensor([1])}
    latent_indices = {
        "future_wrist_image_latent_idx": 3,
        "future_wrist_image2_latent_idx": -1,
        "future_image_latent_idx": 4,
        "future_image2_latent_idx": -1,
    }
    captured = {}

    class ExplicitPriorModel:
        def generate_samples_from_batch(
            self,
            data_batch,
            *,
            x_sigma_max,
            num_steps,
            solver_option,
            sigma_max,
            guidance,
            use_variance_scale,
        ):
            captured["joint_prior"] = x_sigma_max.clone()
            return x_sigma_max + 10

    result = _run_same_prior_endpoint(
        ExplicitPriorModel(),
        data_batch,
        latent_indices,
        video_prior,
        action_prior,
        teacher_steps=8,
    )

    joint = captured["joint_prior"]
    expected_action = action_prior.flatten()[:64].reshape(16, 2, 2)
    assert torch.equal(joint[0, :, 1], expected_action)
    assert torch.equal(joint[0, :, 3], video_prior[0, :, 3])
    assert torch.equal(joint[0, :, 4], video_prior[0, :, 4])
    assert result["video_frame_mask"].tolist() == [[False, False, False, True, True]]
    assert torch.equal(result["endpoint_video"], joint + 10)
    assert result["effective_teacher_steps"] == 8


def test_same_prior_worker_rejects_nonfinite_or_wrong_action_shape():
    from distillation_flowmap.cosmos_policy_raw_worker import (
        _run_same_prior_endpoint,
    )

    class ExplicitPriorModel:
        def generate_samples_from_batch(self, data_batch, *, x_sigma_max, **kwargs):
            return x_sigma_max

    data_batch = {"action_latent_idx": torch.tensor([1])}
    video_prior = torch.zeros(1, 16, 5, 2, 2)

    with pytest.raises(ValueError, match="finite"):
        bad_video = video_prior.clone()
        bad_video[0, 0, 0, 0, 0] = float("nan")
        _run_same_prior_endpoint(
            ExplicitPriorModel(),
            data_batch,
            {},
            bad_video,
            torch.zeros(1, 16, 7),
            teacher_steps=8,
        )
    with pytest.raises(ValueError, match=r"\[B,16,7\]"):
        _run_same_prior_endpoint(
            ExplicitPriorModel(),
            data_batch,
            {},
            video_prior,
            torch.zeros(1, 4),
            teacher_steps=8,
        )
