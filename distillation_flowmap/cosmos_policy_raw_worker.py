"""Line-oriented worker for official Cosmos Policy raw inference.

The Any_WAM training env currently does not import the official Cosmos package
with CUDA extras. This worker is launched under the Cosmos env, loads the policy
once, and serves action chunks over stdin/stdout JSON messages.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch


_AUDITED_COSMOS_LAYOUT_COMMIT = "1eb8457072b4a1adfe1f83c3076e4aa5452cbab2"
_COSMOS_LAYOUT_SOURCE = Path(
    "cosmos_predict2/_src/predict2/cosmos_policy/experiments/robot/"
    "cosmos_utils.py"
)


def _official_matched_budget_response_fields(
    request,
    *,
    configured_action_steps,
    configured_future_steps,
    include_future,
    observed_joint_nfe,
):
    supported = (1, 2, 4)
    missing = [
        name
        for name in ("requested_video_steps", "requested_action_steps")
        if name not in request
    ]
    if missing:
        raise ValueError(
            f"official matched-budget request is missing requested fields: {missing}"
        )
    video_steps = request["requested_video_steps"]
    action_steps = request["requested_action_steps"]
    if (
        type(video_steps) is not int
        or type(action_steps) is not int
        or video_steps not in supported
        or action_steps not in supported
    ):
        raise ValueError(
            "official requested video/action steps must be one of 1, 2, 4"
        )
    if video_steps != action_steps:
        raise ValueError(
            "official matched-budget request requires equal video/action steps"
        )
    if not include_future:
        raise ValueError(
            "official matched video/action budget requires future-video generation"
        )
    if configured_action_steps != action_steps:
        raise RuntimeError(
            "official worker configured action steps "
            f"{configured_action_steps!r} do not match requested {action_steps}"
        )
    effective_video_steps = configured_future_steps
    if effective_video_steps != video_steps:
        raise RuntimeError(
            "official worker configured future-video steps "
            f"{configured_future_steps!r} do not match requested {video_steps}"
        )
    if type(observed_joint_nfe) is not int or observed_joint_nfe != action_steps:
        raise RuntimeError(
            "official worker observed joint sampler NFE "
            f"{observed_joint_nfe!r}, expected exactly {action_steps}"
        )
    return {
        "requested_video_steps": np.int64(video_steps),
        "requested_action_steps": np.int64(action_steps),
        "effective_video_steps": np.int64(observed_joint_nfe),
        "effective_action_steps": np.int64(observed_joint_nfe),
        "matched_budget_verified": np.bool_(True),
        "observed_joint_nfe": np.int64(observed_joint_nfe),
    }


def _call_get_action_with_observed_joint_nfe(
    get_action,
    *,
    model,
    get_action_kwargs,
):
    """Count actual official joint denoiser calls, restoring the model exactly."""

    observed = 0
    restorations = []
    for method_name in ("get_x0_fn_from_batch", "get_velocity_fn_from_batch"):
        original = getattr(model, method_name, None)
        if original is None:
            continue
        marker = object()
        previous_instance = model.__dict__.get(method_name, marker)

        def instrumented(*args, __original=original, **kwargs):
            result = __original(*args, **kwargs)
            if isinstance(result, tuple):
                denoiser, *rest = result
            else:
                denoiser, rest = result, None

            def counted(*fn_args, **fn_kwargs):
                nonlocal observed
                observed += 1
                return denoiser(*fn_args, **fn_kwargs)

            return counted if rest is None else (counted, *rest)

        setattr(model, method_name, instrumented)
        restorations.append((method_name, previous_instance, marker))
    if not restorations:
        raise RuntimeError(
            "official Cosmos model exposes no auditable denoiser factory"
        )
    try:
        result = get_action(model=model, **get_action_kwargs)
    finally:
        for method_name, previous_instance, marker in restorations:
            if previous_instance is marker:
                delattr(model, method_name)
            else:
                setattr(model, method_name, previous_instance)
    return result, observed


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dataset-stats-path", required=True)
    parser.add_argument("--t5-embeddings-path", required=True)
    parser.add_argument("--extra-pythonpath", default="")
    parser.add_argument("--num-denoising-steps-action", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _make_cfg(args):
    return SimpleNamespace(
        suite="libero",
        model_family="cosmos",
        config=args.config_name,
        ckpt_path=args.checkpoint_dir,
        planning_model_config_name="",
        planning_model_ckpt_path="",
        config_file=args.config_file,
        use_third_person_image=True,
        num_third_person_images=1,
        use_wrist_image=True,
        num_wrist_images=1,
        use_proprio=True,
        flip_images=True,
        use_variance_scale=False,
        use_jpeg_compression=True,
        ar_future_prediction=False,
        ar_value_prediction=False,
        ar_qvalue_prediction=False,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_denoising_steps_future_state=args.num_denoising_steps_action,
        num_denoising_steps_value=1,
        shift=5,
        unnormalize_actions=True,
        normalize_proprio=True,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_embeddings_path,
        text_embeddings_kind="t5",
        trained_with_image_aug=True,
        chunk_size=16,
        num_open_loop_steps=16,
        deterministic=True,
        randomize_seed=False,
        seed=args.seed,
    )


def _time_view(tensor):
    return tensor[:, None, :, None, None]


def _profile_enabled():
    return os.environ.get("COSMOS_POLICY_WORKER_PROFILE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _inject_normalized_action_chunk(query_latent, action_chunk, action_indices):
    """Use Cosmos Policy's repeated action-frame layout on a cloned query."""
    joint = query_latent.clone()
    action = action_chunk.to(device=joint.device, dtype=joint.dtype)
    if action.ndim != 3:
        raise ValueError("action_chunk must have shape [B,16,7]")
    indices = torch.as_tensor(action_indices, device=joint.device, dtype=torch.long)
    if indices.shape != (joint.shape[0],):
        raise ValueError("action_indices must have shape [B]")
    flat = action.reshape(action.shape[0], -1)
    latent_elements = joint.shape[1] * joint.shape[3] * joint.shape[4]
    repeats = (latent_elements + flat.shape[1] - 1) // flat.shape[1]
    packed = flat.repeat(1, repeats)[:, :latent_elements].reshape(
        joint.shape[0], joint.shape[1], joint.shape[3], joint.shape[4]
    )
    batch_indices = torch.arange(joint.shape[0], device=joint.device)
    joint[batch_indices, :, indices] = packed
    return joint


def _array_sha256(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _require_exact_teacher_steps(value, *, error_cls=ValueError, source="request"):
    if type(value) is not int:
        raise error_cls(
            f"same-prior {source} teacher steps must be an integer equal to 8"
        )
    if value != 8:
        raise error_cls("same-prior endpoint requires exactly 8 teacher steps")
    return value


def _qualified_type_name(value):
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__name__}"


def _run_layout_git(repo, *arguments):
    git_env = os.environ.copy()
    git_env.update(
        GIT_OPTIONAL_LOCKS="0",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_SYSTEM="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
    )
    try:
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(repo), *arguments],
            capture_output=True,
            text=True,
            timeout=10,
            env=git_env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "same-prior Cosmos repository Git metadata is unverifiable"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git failed"
        raise RuntimeError(
            "same-prior Cosmos repository Git metadata is unverifiable: "
            f"{detail}"
        )
    return result.stdout


def _validate_same_prior_layout_source(repo, cosmos_utils):
    try:
        repo_path = Path(repo).resolve(strict=True)
        expected_source = (repo_path / _COSMOS_LAYOUT_SOURCE).resolve(strict=True)
        imported_source = Path(cosmos_utils.__file__).resolve(strict=True)
    except (AttributeError, OSError, TypeError) as exc:
        raise RuntimeError(
            "same-prior Cosmos layout source is missing or unverifiable"
        ) from exc
    if imported_source != expected_source:
        raise RuntimeError(
            "same-prior cosmos_utils must be imported from selected repository"
        )

    git_root_text = _run_layout_git(repo_path, "rev-parse", "--show-toplevel")
    try:
        git_root = Path(git_root_text.strip()).resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "same-prior Cosmos repository Git metadata is unverifiable"
        ) from exc
    if git_root != repo_path:
        raise RuntimeError(
            "same-prior selected Cosmos path must be the Git repository root"
        )

    head = _run_layout_git(repo_path, "rev-parse", "HEAD").strip()
    if head != _AUDITED_COSMOS_LAYOUT_COMMIT:
        raise RuntimeError(
            "same-prior Cosmos repository must be at audited commit "
            f"{_AUDITED_COSMOS_LAYOUT_COMMIT}, got {head!r}"
        )
    status = _run_layout_git(
        repo_path,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError(
            "same-prior Cosmos repository must be clean before model invocation"
        )


def _model_for_request(
    model,
    *,
    mode,
    repo,
    cosmos_utils,
    cfg,
    get_model,
):
    if mode in ("same_prior_endpoint", "joint_continuation_endpoint"):
        _validate_same_prior_layout_source(repo, cosmos_utils)
    if model is None:
        model, _ = get_model(cfg)
    return model


def _validate_explicit_prior_runtime(model):
    expected_model = (
        "cosmos_predict2._src.predict2.cosmos_policy.models."
        "policy_video2world_model.CosmosPolicyVideo2WorldModel"
    )
    expected_sampler = (
        "cosmos_predict2._src.predict2.cosmos_policy.modules."
        "cosmos_sampler.CosmosPolicySampler"
    )
    expected_sde = (
        "cosmos_predict2._src.predict2.cosmos_policy.modules."
        "hybrid_edm_sde.HybridEDMSDE"
    )
    actual_model = _qualified_type_name(model)
    if actual_model != expected_model:
        raise RuntimeError(
            f"unsupported Cosmos runtime model {actual_model!r}; "
            f"expected {expected_model!r}"
        )
    sampler = getattr(model, "sampler", None)
    actual_sampler = _qualified_type_name(sampler)
    if actual_sampler != expected_sampler:
        raise RuntimeError(
            f"unsupported Cosmos runtime sampler {actual_sampler!r}; "
            f"expected {expected_sampler!r}"
        )
    sde = getattr(model, "sde", None)
    actual_sde = _qualified_type_name(sde)
    if actual_sde != expected_sde:
        raise RuntimeError(
            f"unsupported Cosmos runtime SDE {actual_sde!r}; "
            f"expected {expected_sde!r}"
        )
    if getattr(sde, "sigma_min", None) != 4:
        raise RuntimeError(
            f"Cosmos HybridEDMSDE sigma_min must equal 4, got "
            f"{getattr(sde, 'sigma_min', None)!r}"
        )
    if getattr(sde, "sigma_max", None) != 80:
        raise RuntimeError(
            f"Cosmos HybridEDMSDE sigma_max must equal 80, got "
            f"{getattr(sde, 'sigma_max', None)!r}"
        )
    if getattr(getattr(model, "config", None), "use_flowunipc_scheduler", None) is not False:
        raise RuntimeError(
            "Cosmos backend is not using the required named EDM sampler path"
        )


def _generate_from_explicit_prior(
    model,
    data_batch,
    joint_prior,
    *,
    teacher_steps,
    sigma_max=80.0,
):
    """Run the audited EDM endpoint and return its observed denoiser count."""
    _require_exact_teacher_steps(teacher_steps)
    _validate_explicit_prior_runtime(model)
    if (
        isinstance(sigma_max, bool)
        or not isinstance(sigma_max, (int, float))
        or not math.isfinite(float(sigma_max))
        or not 4.0 <= float(sigma_max) <= 80.0
    ):
        raise ValueError("explicit-prior sigma_max must be a scalar in [4,80]")
    sigma_max = float(sigma_max)
    generate = model.generate_samples_from_batch
    signature = inspect.signature(generate)
    if "x_sigma_max" not in signature.parameters:
        raise RuntimeError(
            "Cosmos backend cannot honor explicit same-prior sampling"
        )

    original_get_x0 = model.get_x0_fn_from_batch
    instance_marker = object()
    previous_instance_value = model.__dict__.get(
        "get_x0_fn_from_batch", instance_marker
    )
    observed_nfe = 0

    def instrumented_get_x0(*args, **kwargs):
        result = original_get_x0(*args, **kwargs)
        if isinstance(result, tuple):
            x0_fn, *rest = result
        else:
            x0_fn, rest = result, None

        def counted_x0(*fn_args, **fn_kwargs):
            nonlocal observed_nfe
            observed_nfe += 1
            return x0_fn(*fn_args, **fn_kwargs)

        if rest is None:
            return counted_x0
        return (counted_x0, *rest)

    model.get_x0_fn_from_batch = instrumented_get_x0
    try:
        endpoint = generate(
            data_batch,
            x_sigma_max=joint_prior,
            num_steps=teacher_steps,
            solver_option="2ab",
            sigma_max=sigma_max,
            guidance=0,
            use_variance_scale=False,
        )
    finally:
        if previous_instance_value is instance_marker:
            delattr(model, "get_x0_fn_from_batch")
        else:
            model.get_x0_fn_from_batch = previous_instance_value
    if observed_nfe != 8:
        raise RuntimeError(
            "Cosmos explicit-prior sampler did not execute exactly 8 denoiser "
            f"evaluations; observed {observed_nfe}"
        )
    return endpoint, observed_nfe


def _slice_batch_value(value, index, batch_size):
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index : index + 1]
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == batch_size:
        return value[index : index + 1]
    if isinstance(value, dict):
        return {
            key: _slice_batch_value(item, index, batch_size)
            for key, item in value.items()
        }
    if isinstance(value, list) and len(value) == batch_size:
        return value[index : index + 1]
    if isinstance(value, tuple) and len(value) == batch_size:
        return value[index : index + 1]
    return value


def _run_joint_continuation_endpoint(
    model,
    data_batch,
    latent_indices,
    canonical_joint_state,
    normalized_t,
    *,
    teacher_steps,
):
    """Continue canonical normalized joint states with exact per-sample EDM clocks."""
    _require_exact_teacher_steps(teacher_steps)
    canonical = torch.as_tensor(canonical_joint_state)
    times = torch.as_tensor(normalized_t, device=canonical.device)
    if canonical.dtype != torch.float32:
        raise ValueError("canonical_joint_state must use float32 without conversion")
    if canonical.ndim != 5 or tuple(canonical.shape[1:]) != (16, 9, 28, 28):
        raise ValueError(
            "canonical_joint_state must have Cosmos shape [B,16,9,28,28]"
        )
    if times.dtype != torch.float32:
        raise ValueError("normalized_t must use float32 without conversion")
    batch_size = canonical.shape[0]
    if times.ndim == 1:
        if times.shape != (batch_size,):
            raise ValueError("normalized_t must have shape [B] or [B,F]")
        per_sample_t = times
    elif times.ndim == 2:
        if times.shape[0] != batch_size:
            raise ValueError("normalized_t batch size must match canonical state")
        if not torch.equal(times, times[:, :1].expand_as(times)):
            raise ValueError("normalized_t must be constant across frames per sample")
        per_sample_t = times[:, 0]
    else:
        raise ValueError("normalized_t must have shape [B] or [B,F]")
    if not bool(torch.isfinite(canonical).all()):
        raise ValueError("canonical_joint_state must contain only finite values")
    if not bool(torch.isfinite(per_sample_t).all()):
        raise ValueError("normalized_t must contain only finite values")
    if not bool(
        ((per_sample_t >= 4.0 / 5.0) & (per_sample_t <= 80.0 / 81.0)).all()
    ):
        raise ValueError("normalized_t must remain in the calibrated Cosmos band")

    mask = _video_frame_mask(canonical, latent_indices)
    if not bool(mask.any(dim=1).all()):
        raise RuntimeError("joint continuation requires a nonempty video mask")

    endpoints = []
    observed_steps = []
    edm_sigmas = per_sample_t / (1.0 - per_sample_t)
    for index in range(batch_size):
        sigma = float(edm_sigmas[index].item())
        edm_state = canonical[index : index + 1] / (
            1.0 - per_sample_t[index]
        )
        endpoint, effective_steps = _generate_from_explicit_prior(
            model,
            _slice_batch_value(data_batch, index, batch_size),
            edm_state,
            teacher_steps=teacher_steps,
            sigma_max=sigma,
        )
        if (
            not isinstance(endpoint, torch.Tensor)
            or endpoint.shape != canonical[index : index + 1].shape
            or endpoint.dtype != torch.float32
            or not bool(torch.isfinite(endpoint).all())
        ):
            raise RuntimeError(
                "Cosmos joint continuation returned an invalid clean endpoint"
            )
        endpoints.append(endpoint)
        observed_steps.append(effective_steps)
    if not observed_steps or any(value != 8 for value in observed_steps):
        raise RuntimeError(
            "Cosmos joint continuation did not observe eight steps per sample"
        )
    return {
        "endpoint_video": torch.cat(endpoints, dim=0),
        "video_frame_mask": mask,
        "effective_teacher_steps": observed_steps[0],
        "normalized_t": per_sample_t,
        "edm_sigma": edm_sigmas,
    }


def _validate_official_libero_geometry(cfg, model):
    required_cfg = {
        "suite": "libero",
        "use_proprio": True,
        "normalize_proprio": True,
        "use_wrist_image": True,
        "num_wrist_images": 1,
        "use_third_person_image": True,
        "num_third_person_images": 1,
        "use_jpeg_compression": True,
        "trained_with_image_aug": True,
    }
    for name, expected in required_cfg.items():
        actual = getattr(cfg, name, None)
        if actual != expected or type(actual) is not type(expected):
            raise RuntimeError(
                f"official LIBERO {name} must equal {expected!r}, got {actual!r}"
            )

    required_model = {
        "state_ch": 16,
        "state_t": 9,
        "min_num_conditional_frames": 4,
    }
    config = getattr(model, "config", None)
    for name, expected in required_model.items():
        actual = getattr(config, name, None)
        if actual != expected:
            raise RuntimeError(
                f"official LIBERO model {name} must equal {expected}, got {actual!r}"
            )
    tokenizer = getattr(model, "tokenizer", None)
    spatial_factor = getattr(tokenizer, "spatial_compression_factor", None)
    if spatial_factor != 8:
        raise RuntimeError(
            "official LIBERO tokenizer spatial compression factor must equal 8, "
            f"got {spatial_factor!r}"
        )
    latent_frames = tokenizer.get_latent_num_frames(33)
    if latent_frames != 9:
        raise RuntimeError(
            "official LIBERO tokenizer must map 33 input frames to 9 latent "
            f"frames, got {latent_frames!r}"
        )
    return (16, 9, 28, 28)


def _load_same_prior_arrays(data):
    video_raw = np.asarray(data["video_prior"])
    action_raw = np.asarray(data["action_prior"])
    if video_raw.dtype != np.float32:
        raise ValueError(
            f"same-prior video_prior NPZ dtype must be float32, got {video_raw.dtype}"
        )
    if action_raw.dtype != np.float32:
        raise ValueError(
            f"same-prior action_prior NPZ dtype must be float32, got {action_raw.dtype}"
        )
    video_prior = np.ascontiguousarray(video_raw)
    action_prior = np.ascontiguousarray(action_raw)
    if not np.isfinite(video_prior).all():
        raise ValueError("same-prior video_prior must contain only finite values")
    if not np.isfinite(action_prior).all():
        raise ValueError("same-prior action_prior must contain only finite values")
    return (
        video_prior,
        action_prior,
        _array_sha256(video_prior),
        _array_sha256(action_prior),
    )


def _load_joint_continuation_arrays(data):
    state_raw = np.asarray(data["canonical_joint_state"])
    time_raw = np.asarray(data["normalized_t"])
    if state_raw.dtype != np.float32:
        raise ValueError(
            "joint continuation canonical_joint_state NPZ dtype must be "
            f"float32, got {state_raw.dtype}"
        )
    if time_raw.dtype != np.float32:
        raise ValueError(
            "joint continuation normalized_t NPZ dtype must be "
            f"float32, got {time_raw.dtype}"
        )
    state = np.ascontiguousarray(state_raw)
    normalized_t = np.ascontiguousarray(time_raw)
    if not np.isfinite(state).all():
        raise ValueError(
            "joint continuation canonical_joint_state must contain only "
            "finite values"
        )
    if not np.isfinite(normalized_t).all():
        raise ValueError(
            "joint continuation normalized_t must contain only finite values"
        )
    return (
        state,
        normalized_t,
        _array_sha256(state),
        _array_sha256(normalized_t),
    )


def _joint_continuation_response_fields(
    result, *, joint_state_sha256, normalized_t_sha256
):
    endpoint = result["endpoint_video"]
    frame_mask = result["video_frame_mask"]
    effective_steps = result["effective_teacher_steps"]
    normalized_t = result["normalized_t"]
    edm_sigma = result["edm_sigma"]
    _require_exact_teacher_steps(
        effective_steps,
        error_cls=RuntimeError,
        source="observed effective",
    )
    if not isinstance(endpoint, torch.Tensor) or endpoint.dtype != torch.float32:
        raise RuntimeError(
            "joint continuation backend endpoint dtype must be float32"
        )
    if not isinstance(frame_mask, torch.Tensor) or frame_mask.dtype != torch.bool:
        raise RuntimeError(
            "joint continuation backend video frame mask must be boolean"
        )
    if frame_mask.shape != (endpoint.shape[0], endpoint.shape[2]):
        raise RuntimeError(
            "joint continuation backend video frame mask shape is invalid"
        )
    if not bool(frame_mask.any(dim=1).all()):
        raise RuntimeError(
            "joint continuation backend video frame mask must be nonempty"
        )
    if (
        not isinstance(normalized_t, torch.Tensor)
        or normalized_t.dtype != torch.float32
        or normalized_t.shape != (endpoint.shape[0],)
        or not bool(torch.isfinite(normalized_t).all())
    ):
        raise RuntimeError(
            "joint continuation backend normalized time is invalid"
        )
    if (
        not isinstance(edm_sigma, torch.Tensor)
        or edm_sigma.dtype != torch.float32
        or edm_sigma.shape != (endpoint.shape[0],)
        or not bool(torch.isfinite(edm_sigma).all())
    ):
        raise RuntimeError("joint continuation backend EDM sigma is invalid")
    return {
        "endpoint_video": endpoint.detach().cpu().numpy(),
        "video_frame_mask": frame_mask.detach().cpu().numpy(),
        "effective_teacher_steps": np.asarray(effective_steps, dtype=np.int64),
        "joint_state_sha256": np.asarray(joint_state_sha256),
        "normalized_t_sha256": np.asarray(normalized_t_sha256),
        "normalized_t": normalized_t.detach().cpu().numpy(),
        "edm_sigma": edm_sigma.detach().cpu().numpy(),
    }


def _same_prior_response_fields(
    result, *, video_prior_sha256, action_prior_sha256
):
    endpoint = result["endpoint_video"]
    frame_mask = result["video_frame_mask"]
    effective_steps = result["effective_teacher_steps"]
    _require_exact_teacher_steps(
        effective_steps,
        error_cls=RuntimeError,
        source="observed effective",
    )
    if not isinstance(endpoint, torch.Tensor) or endpoint.dtype != torch.float32:
        raise RuntimeError("same-prior backend endpoint dtype must be float32")
    if not isinstance(frame_mask, torch.Tensor) or frame_mask.dtype != torch.bool:
        raise RuntimeError("same-prior backend video frame mask must be boolean")
    if frame_mask.shape != (endpoint.shape[0], endpoint.shape[2]):
        raise RuntimeError("same-prior backend video frame mask shape is invalid")
    return {
        "endpoint_video": endpoint.detach().cpu().numpy(),
        "video_frame_mask": frame_mask.detach().cpu().numpy(),
        "effective_teacher_steps": np.asarray(effective_steps, dtype=np.int64),
        "video_prior_sha256": np.asarray(video_prior_sha256),
        "action_prior_sha256": np.asarray(action_prior_sha256),
    }


def _video_frame_mask(video_prior, latent_indices):
    mask = torch.zeros(
        (video_prior.shape[0], video_prior.shape[2]),
        dtype=torch.bool,
        device=video_prior.device,
    )
    for key in (
        "future_wrist_image_latent_idx",
        "future_wrist_image2_latent_idx",
        "future_image_latent_idx",
        "future_image2_latent_idx",
    ):
        frame_index = int(latent_indices.get(key, -1))
        if frame_index >= 0:
            if frame_index >= video_prior.shape[2]:
                raise ValueError(
                    f"{key}={frame_index} is outside the supplied video prior"
                )
            mask[:, frame_index] = True
    return mask


def _run_same_prior_endpoint(
    model,
    data_batch,
    latent_indices,
    video_prior,
    action_prior,
    *,
    teacher_steps,
):
    """Pack the split caller priors and run the exact eight-step EDM endpoint."""
    video_prior = torch.as_tensor(video_prior)
    action_prior = torch.as_tensor(action_prior, device=video_prior.device)
    if video_prior.dtype != torch.float32:
        raise ValueError("video_prior must use float32 without conversion")
    if action_prior.dtype != torch.float32:
        raise ValueError("action_prior must use float32 without conversion")
    if video_prior.ndim != 5 or tuple(video_prior.shape[1:]) != (16, 9, 28, 28):
        raise ValueError(
            "video_prior must have Cosmos joint latent shape [B,16,9,28,28]"
        )
    if action_prior.ndim != 3 or action_prior.shape[1:] != (16, 7):
        raise ValueError(
            "action_prior must have native Cosmos-normalized shape [B,16,7]"
        )
    if action_prior.shape[0] != video_prior.shape[0]:
        raise ValueError("video_prior and action_prior batch sizes must match")
    if not bool(torch.isfinite(video_prior).all()):
        raise ValueError("video_prior must contain only finite values")
    if not bool(torch.isfinite(action_prior).all()):
        raise ValueError("action_prior must contain only finite values")

    data_video = data_batch.get("video")
    if not isinstance(data_video, torch.Tensor) or data_video.ndim != 5:
        raise ValueError("Cosmos data batch is missing a rank-5 video tensor")
    tokenizer = getattr(model, "tokenizer", None)
    spatial_factor = getattr(tokenizer, "spatial_compression_factor", None)
    if spatial_factor != 8:
        raise RuntimeError("Cosmos tokenizer spatial compression drift detected")
    expected_shape = (
        video_prior.shape[0],
        getattr(getattr(model, "config", None), "state_ch", None),
        tokenizer.get_latent_num_frames(data_video.shape[-3]),
        data_video.shape[-2] // spatial_factor,
        data_video.shape[-1] // spatial_factor,
    )
    if expected_shape != (video_prior.shape[0], 16, 9, 28, 28):
        raise RuntimeError(
            f"Cosmos conditioning batch derives unsupported joint shape {expected_shape}"
        )
    if tuple(video_prior.shape) != expected_shape:
        raise ValueError(
            "video_prior must match the official derived "
            f"[B,16,9,28,28] shape, got {tuple(video_prior.shape)}"
        )
    action_indices = data_batch.get("action_latent_idx")
    if action_indices is None:
        raise ValueError("Cosmos data batch is missing action_latent_idx")
    joint_prior = _inject_normalized_action_chunk(
        video_prior, action_prior, action_indices
    )
    mask = _video_frame_mask(video_prior, latent_indices)
    if not torch.equal(joint_prior[mask[:, None, :, None, None].expand_as(joint_prior)],
                       video_prior[mask[:, None, :, None, None].expand_as(video_prior)]):
        raise RuntimeError("action packing changed a valid video frame")
    edm_joint_prior = joint_prior * 80.0
    endpoint, effective_steps = _generate_from_explicit_prior(
        model,
        data_batch,
        edm_joint_prior,
        teacher_steps=teacher_steps,
    )
    if not isinstance(endpoint, torch.Tensor):
        raise RuntimeError("Cosmos same-prior sampler returned a non-tensor endpoint")
    if endpoint.shape != joint_prior.shape:
        raise RuntimeError("Cosmos same-prior sampler returned an invalid endpoint shape")
    if endpoint.dtype != torch.float32:
        raise RuntimeError("Cosmos same-prior sampler endpoint dtype must be float32")
    if not bool(torch.isfinite(endpoint).all()):
        raise RuntimeError("Cosmos same-prior sampler returned non-finite values")
    if mask.dtype != torch.bool:
        raise RuntimeError("Cosmos same-prior video frame mask must be boolean")
    return {
        "endpoint_video": endpoint,
        "video_frame_mask": mask,
        "effective_teacher_steps": effective_steps,
    }


def _build_libero_data_batch(cfg, model, dataset_stats, obs, task, cosmos_utils):
    """Reproduce the official LIBERO pre-sampling batch without calling get_action."""
    _validate_explicit_prior_runtime(model)
    expected_state_shape = _validate_official_libero_geometry(cfg, model)
    if cosmos_utils.COSMOS_IMAGE_SIZE != 224:
        raise RuntimeError(
            "official LIBERO COSMOS_IMAGE_SIZE drifted from pinned value 224"
        )
    if cosmos_utils.COSMOS_TEMPORAL_COMPRESSION_FACTOR != 4:
        raise RuntimeError(
            "official LIBERO temporal compression factor drifted from pinned value 4"
        )
    device = torch.device(model.tensor_kwargs["device"])
    text_embedding = cosmos_utils.get_t5_embedding_from_cache(task)
    wrist_image, primary_image = cosmos_utils.prepare_images_for_model(
        [obs["wrist_image"], obs["primary_image"]], cfg
    )
    proprio = obs["proprio"]
    if cfg.normalize_proprio:
        proprio = cosmos_utils.rescale_proprio(
            proprio,
            dataset_stats,
            non_negative_only=False,
            scale_multiplier=1.0,
        )

    blank_image = np.zeros_like(primary_image)
    blank_image_duplicated = cosmos_utils.duplicate_array(
        blank_image.copy(),
        total_num_copies=cosmos_utils.COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    )
    wrist_image_duplicated = cosmos_utils.duplicate_array(
        wrist_image,
        total_num_copies=cosmos_utils.COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    )
    primary_image_duplicated = cosmos_utils.duplicate_array(
        primary_image,
        total_num_copies=cosmos_utils.COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    )
    image_sequence = [
        np.expand_dims(blank_image, axis=0),
        blank_image_duplicated.copy(),
        wrist_image_duplicated,
        primary_image_duplicated,
        blank_image_duplicated.copy(),
        blank_image_duplicated.copy(),
        wrist_image_duplicated.copy(),
        primary_image_duplicated.copy(),
        blank_image_duplicated.copy(),
    ]
    latent_indices = {
        "current_proprio_latent_idx": 1,
        "current_wrist_image_latent_idx": 2,
        "current_wrist_image2_latent_idx": -1,
        "current_image_latent_idx": 3,
        "current_image2_latent_idx": -1,
        "action_latent_idx": 4,
        "future_proprio_latent_idx": 5,
        "future_wrist_image_latent_idx": 6,
        "future_wrist_image2_latent_idx": -1,
        "future_image_latent_idx": 7,
        "future_image2_latent_idx": -1,
        "value_latent_idx": 8,
    }

    raw_image_sequence = np.concatenate(image_sequence, axis=0)[None]
    raw_image_sequence = np.transpose(raw_image_sequence, (0, 4, 1, 2, 3))
    if raw_image_sequence.shape != (1, 3, 33, 224, 224):
        raise RuntimeError(
            "official LIBERO conditioning layout drifted; expected "
            f"[1,3,33,224,224], got {raw_image_sequence.shape}"
        )
    raw_image_sequence = torch.from_numpy(raw_image_sequence).to(
        device=device, dtype=torch.uint8
    )
    proprio_tensor = torch.from_numpy(np.asarray(proprio)).reshape(1, -1).to(
        device=device, dtype=torch.bfloat16
    )
    text_embedding = torch.as_tensor(text_embedding)
    if text_embedding.dim() == 2:
        text_embedding = text_embedding.unsqueeze(0)
    text_embedding = text_embedding.to(device=device, dtype=torch.bfloat16)

    data_batch = {
        "dataset_name": "video_data",
        "video": raw_image_sequence,
        "t5_text_embeddings": text_embedding,
        "fps": torch.tensor([16], dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros(
            (1, 1, cosmos_utils.COSMOS_IMAGE_SIZE, cosmos_utils.COSMOS_IMAGE_SIZE),
            dtype=torch.bfloat16,
            device=device,
        ),
        "num_conditional_frames": model.config.min_num_conditional_frames,
        "proprio": proprio_tensor,
    }
    for key, value in latent_indices.items():
        data_batch[key] = torch.tensor([value], dtype=torch.int64, device=device)
    derived_state_shape = (
        model.config.state_ch,
        model.tokenizer.get_latent_num_frames(raw_image_sequence.shape[-3]),
        raw_image_sequence.shape[-2] // model.tokenizer.spatial_compression_factor,
        raw_image_sequence.shape[-1] // model.tokenizer.spatial_compression_factor,
    )
    if derived_state_shape != expected_state_shape:
        raise RuntimeError(
            "official LIBERO conditioning batch and model tokenizer geometry differ"
        )
    return data_batch, latent_indices


def _velocity_from_x0_fn(model, x0_fn, x_t, t):
    t = t.to(device=x_t.device, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    sigma = t / (1.0 - t)
    sigma_view = _time_view(sigma).to(x_t)
    x_sigma = x_t * (1.0 + sigma_view)
    x0_pred = x0_fn(x_sigma, sigma)
    eps_pred = (x_sigma - x0_pred) / sigma_view
    return eps_pred - x0_pred


def _compute_latent_cdiff(
    model,
    data_batch,
    x0_anchor,
    noise,
    t,
    r,
    epsilon,
    center_velocity_mode="exact",
):
    eps = float(epsilon)
    if eps <= 0:
        raise ValueError(f"cosmos latent central-diff epsilon must be positive, got {eps}")
    x0_anchor = x0_anchor.detach().float()
    noise = torch.as_tensor(noise, device=x0_anchor.device, dtype=torch.float32)
    t = torch.as_tensor(t, device=x0_anchor.device, dtype=torch.float32)
    r = torch.as_tensor(r, device=x0_anchor.device, dtype=torch.float32)
    if noise.shape != x0_anchor.shape:
        raise ValueError(f"noise shape {tuple(noise.shape)} does not match x0 {tuple(x0_anchor.shape)}")
    if t.shape != x0_anchor.shape[:1] + x0_anchor.shape[2:3]:
        raise ValueError(f"t shape {tuple(t.shape)} does not match latent [B,T] {tuple((x0_anchor.shape[0], x0_anchor.shape[2]))}")
    if r.shape != t.shape:
        raise ValueError(f"r shape {tuple(r.shape)} does not match t {tuple(t.shape)}")

    t_center = t.clamp(min=eps + 1e-6, max=1.0 - eps - 1e-6)
    r = torch.minimum(r.clamp(min=0.0), t_center)
    t_plus = t_center + eps
    t_minus = t_center - eps

    oracle_v = noise - x0_anchor
    x_t = (1.0 - _time_view(t_center).to(x0_anchor)) * x0_anchor + _time_view(t_center).to(x0_anchor) * noise
    x_plus = x_t + oracle_v * eps
    x_minus = x_t - oracle_v * eps

    x0_fn = model.get_x0_fn_from_batch(
        data_batch,
        guidance=0,
        is_negative_prompt=False,
    )
    v_plus = _velocity_from_x0_fn(model, x0_fn, x_plus, t_plus)
    v_minus = _velocity_from_x0_fn(model, x0_fn, x_minus, t_minus)
    mode = str(center_velocity_mode).lower()
    if mode in ("symmetric_average", "symmetric-avg", "avg", "fast"):
        v_center = 0.5 * (v_plus + v_minus)
    elif mode == "exact":
        v_center = _velocity_from_x0_fn(model, x0_fn, x_t, t_center)
    else:
        raise ValueError(
            "cosmos latent center velocity mode must be 'exact' or "
            f"'symmetric_average', got {center_velocity_mode!r}"
        )
    dF_dt = (v_plus - v_minus) / (2.0 * eps)
    target = v_center - _time_view(t_center - r).to(v_center) * dF_dt
    return {
        "x0": x0_anchor.detach().cpu().numpy().astype(np.float32),
        "target": target.detach().cpu().numpy().astype(np.float32),
        "velocity": v_center.detach().cpu().numpy().astype(np.float32),
    }


def main():
    response_out = sys.stdout
    sys.stdout = sys.stderr
    args = _parse_args()
    if args.repo:
        sys.path.insert(0, args.repo)
    if os.path.isabs(args.config_file) and args.repo:
        args.config_file = os.path.relpath(args.config_file, args.repo)
    for extra_path in [p for p in args.extra_pythonpath.split(os.pathsep) if p]:
        if os.path.isdir(extra_path) and extra_path not in sys.path:
            sys.path.append(extra_path)

    from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot import (
        cosmos_utils,
    )
    from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )

    cfg = _make_cfg(args)
    init_t5_text_embeddings_cache(
        cfg.t5_text_embeddings_path,
        worker_id=0,
        embeddings_kind=cfg.text_embeddings_kind,
    )
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model = None

    for line in sys.stdin:
        data = None
        try:
            request = json.loads(line)
            data = np.load(request["npz_path"])
            primary = data["primary_image"]
            wrist = data["wrist_image"]
            proprio = data["proprio"].astype(np.float32)
            tasks = request["tasks"]
            mode = request.get("mode", "actions")
            model = _model_for_request(
                model,
                mode=mode,
                repo=args.repo,
                cosmos_utils=cosmos_utils,
                cfg=cfg,
                get_model=get_model,
            )
            if mode == "joint_continuation_endpoint":
                teacher_steps = request.get("teacher_steps")
                _require_exact_teacher_steps(teacher_steps)
                (
                    canonical_state_np,
                    normalized_t_np,
                    joint_state_sha256,
                    normalized_t_sha256,
                ) = _load_joint_continuation_arrays(data)
                if tuple(canonical_state_np.shape[1:]) != (16, 9, 28, 28):
                    raise ValueError(
                        "joint continuation canonical_joint_state NPZ shape "
                        "must be [B,16,9,28,28], got "
                        f"{canonical_state_np.shape}"
                    )
                batch_size = canonical_state_np.shape[0]
                if normalized_t_np.ndim not in (1, 2):
                    raise ValueError(
                        "joint continuation normalized_t NPZ shape must be "
                        "[B] or [B,F]"
                    )
                if normalized_t_np.shape[0] != batch_size:
                    raise ValueError(
                        "joint continuation state/time batch sizes differ"
                    )
                _validate_explicit_prior_runtime(model)
                _validate_official_libero_geometry(cfg, model)
                if len(tasks) != batch_size:
                    raise ValueError(
                        "joint continuation task count does not match state batch"
                    )
                if (
                    primary.shape[0] != len(tasks)
                    or wrist.shape[0] != len(tasks)
                    or proprio.shape[0] != len(tasks)
                ):
                    raise ValueError(
                        "joint continuation observation batch does not match "
                        "task count"
                    )
                endpoint_tensors = []
                mask_tensors = []
                returned_times = []
                edm_sigmas = []
                observed_steps = []
                with torch.no_grad():
                    for idx, task in enumerate(tasks):
                        obs = {
                            "primary_image": primary[idx],
                            "wrist_image": wrist[idx],
                            "proprio": proprio[idx],
                        }
                        data_batch, latent_indices = _build_libero_data_batch(
                            cfg,
                            model,
                            dataset_stats,
                            obs,
                            task,
                            cosmos_utils,
                        )
                        device = data_batch["video"].device
                        continuation_result = _run_joint_continuation_endpoint(
                            model,
                            data_batch,
                            latent_indices,
                            torch.from_numpy(
                                canonical_state_np[idx : idx + 1]
                            ).to(device),
                            torch.from_numpy(
                                normalized_t_np[idx : idx + 1]
                            ).to(device),
                            teacher_steps=teacher_steps,
                        )
                        endpoint_tensors.append(
                            continuation_result["endpoint_video"].detach().cpu()
                        )
                        mask_tensors.append(
                            continuation_result["video_frame_mask"].detach().cpu()
                        )
                        returned_times.append(
                            continuation_result["normalized_t"].detach().cpu()
                        )
                        edm_sigmas.append(
                            continuation_result["edm_sigma"].detach().cpu()
                        )
                        observed_steps.append(
                            continuation_result["effective_teacher_steps"]
                        )
                if not observed_steps or any(
                    value != 8 for value in observed_steps
                ):
                    raise RuntimeError(
                        "joint continuation batch did not observe exactly eight "
                        "teacher steps per sample"
                    )
                fields = _joint_continuation_response_fields(
                    {
                        "endpoint_video": torch.cat(endpoint_tensors, dim=0),
                        "video_frame_mask": torch.cat(mask_tensors, dim=0),
                        "effective_teacher_steps": observed_steps[0],
                        "normalized_t": torch.cat(returned_times, dim=0),
                        "edm_sigma": torch.cat(edm_sigmas, dim=0),
                    },
                    joint_state_sha256=joint_state_sha256,
                    normalized_t_sha256=normalized_t_sha256,
                )
                response_path = request["actions_path"]
                np.savez_compressed(response_path, **fields)
                print(
                    json.dumps({"ok": True, "actions_path": response_path}),
                    file=response_out,
                    flush=True,
                )
                continue
            if mode == "same_prior_endpoint":
                teacher_steps = request.get("teacher_steps")
                _require_exact_teacher_steps(teacher_steps)
                (
                    video_prior_np,
                    action_prior_np,
                    video_prior_sha256,
                    action_prior_sha256,
                ) = _load_same_prior_arrays(data)
                if tuple(video_prior_np.shape[1:]) != (16, 9, 28, 28):
                    raise ValueError(
                        "same-prior video_prior NPZ shape must be "
                        f"[B,16,9,28,28], got {video_prior_np.shape}"
                    )
                if tuple(action_prior_np.shape[1:]) != (16, 7):
                    raise ValueError(
                        "same-prior action_prior NPZ shape must be [B,16,7], "
                        f"got {action_prior_np.shape}"
                    )
                if action_prior_np.shape[0] != video_prior_np.shape[0]:
                    raise ValueError(
                        "same-prior action/video prior batch sizes differ"
                    )
                _validate_explicit_prior_runtime(model)
                _validate_official_libero_geometry(cfg, model)
                if len(tasks) != video_prior_np.shape[0]:
                    raise ValueError(
                        "same-prior task count does not match video prior batch"
                    )
                if (
                    primary.shape[0] != len(tasks)
                    or wrist.shape[0] != len(tasks)
                    or proprio.shape[0] != len(tasks)
                ):
                    raise ValueError(
                        "same-prior observation batch does not match task count"
                    )
                endpoint_tensors = []
                mask_tensors = []
                observed_steps = []
                with torch.no_grad():
                    for idx, task in enumerate(tasks):
                        obs = {
                            "primary_image": primary[idx],
                            "wrist_image": wrist[idx],
                            "proprio": proprio[idx],
                        }
                        data_batch, latent_indices = _build_libero_data_batch(
                            cfg,
                            model,
                            dataset_stats,
                            obs,
                            task,
                            cosmos_utils,
                        )
                        device = data_batch["video"].device
                        endpoint_result = _run_same_prior_endpoint(
                            model,
                            data_batch,
                            latent_indices,
                            torch.from_numpy(video_prior_np[idx: idx + 1]).to(device),
                            torch.from_numpy(action_prior_np[idx: idx + 1]).to(device),
                            teacher_steps=teacher_steps,
                        )
                        endpoint_tensors.append(
                            endpoint_result["endpoint_video"].detach().cpu()
                        )
                        mask_tensors.append(
                            endpoint_result["video_frame_mask"].detach().cpu()
                        )
                        observed_steps.append(
                            endpoint_result["effective_teacher_steps"]
                        )
                if not observed_steps or any(
                    value != observed_steps[0] for value in observed_steps
                ):
                    raise RuntimeError(
                        "same-prior batch returned inconsistent observed teacher steps"
                    )
                fields = _same_prior_response_fields(
                    {
                        "endpoint_video": torch.cat(endpoint_tensors, dim=0),
                        "video_frame_mask": torch.cat(mask_tensors, dim=0),
                        "effective_teacher_steps": observed_steps[0],
                    },
                    video_prior_sha256=video_prior_sha256,
                    action_prior_sha256=action_prior_sha256,
                )
                response_path = request["actions_path"]
                np.savez_compressed(response_path, **fields)
                print(
                    json.dumps({"ok": True, "actions_path": response_path}),
                    file=response_out,
                    flush=True,
                )
                continue
            if mode != "actions":
                raise ValueError(f"unsupported Cosmos raw worker mode: {mode!r}")
            actions = []
            include_future = bool(request.get("include_future_predictions", False))
            has_budget_request = any(
                name in request
                for name in ("requested_video_steps", "requested_action_steps")
            )
            matched_budget_fields = {}
            include_latent_x0 = bool(request.get("include_latent_x0", False))
            include_latent_cdiff = bool(request.get("include_latent_cdiff", False))
            include_latent_velocity_query = bool(request.get("include_latent_velocity_query", False))
            include_joint_action_query = bool(request.get("include_joint_action_query", False))
            latent_noise = data["cosmos_latent_noise"] if include_latent_cdiff else None
            latent_t = data["cosmos_latent_t"] if include_latent_cdiff else None
            latent_r = data["cosmos_latent_r"] if include_latent_cdiff else None
            latent_query_x = data["cosmos_latent_query_x"] if include_latent_velocity_query else None
            latent_query_t = data["cosmos_latent_query_t"] if include_latent_velocity_query else None
            action_query_x = data["cosmos_action_query_x"] if include_joint_action_query else None
            latent_epsilon = float(request.get("cosmos_latent_epsilon", 0.001))
            latent_center_velocity_mode = request.get(
                "cosmos_latent_center_velocity_mode", "exact")
            profile = _profile_enabled()
            request_t0 = time.perf_counter() if profile else None
            future_predictions = []
            value_predictions = []
            latent_x0 = []
            latent_cdiff_targets = []
            latent_velocities = []
            latent_query_velocities = []
            joint_queries = []
            video_frame_masks = []
            observed_joint_nfes = []
            with torch.no_grad():
                for idx, task in enumerate(tasks):
                    sample_t0 = time.perf_counter() if profile else None
                    obs = {
                        "primary_image": primary[idx],
                        "wrist_image": wrist[idx],
                        "proprio": proprio[idx],
                    }
                    action_t0 = time.perf_counter() if profile else None
                    result, observed_joint_nfe = _call_get_action_with_observed_joint_nfe(
                        get_action,
                        model=model,
                        get_action_kwargs={
                            "cfg": cfg,
                            "dataset_stats": dataset_stats,
                            "obs": obs,
                            "task_label_or_embedding": task,
                            "seed": int(request.get("seed", args.seed)),
                            "randomize_seed": False,
                            "num_denoising_steps_action": args.num_denoising_steps_action,
                            "generate_future_state_and_value_in_parallel": include_future,
                            "worker_id": 0,
                            "batch_size": 1,
                        },
                    )
                    observed_joint_nfes.append(observed_joint_nfe)
                    if profile:
                        print(
                            "[cosmos_worker_profile] "
                            f"sample={idx} get_action_s={time.perf_counter() - action_t0:.3f}",
                            file=sys.stderr,
                            flush=True,
                        )
                    actions.append(np.asarray(result["actions"], dtype=np.float32))
                    if include_latent_x0 and not include_latent_cdiff:
                        latent_x0.append(
                            result["generated_latent"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                    if include_latent_cdiff:
                        cdiff_t0 = time.perf_counter() if profile else None
                        cdiff = _compute_latent_cdiff(
                            model,
                            result["data_batch"],
                            result["generated_latent"],
                            latent_noise[idx: idx + 1],
                            latent_t[idx: idx + 1],
                            latent_r[idx: idx + 1],
                            latent_epsilon,
                            latent_center_velocity_mode,
                        )
                        if profile:
                            print(
                                "[cosmos_worker_profile] "
                                f"sample={idx} latent_cdiff_s={time.perf_counter() - cdiff_t0:.3f} "
                                f"center_mode={latent_center_velocity_mode}",
                                file=sys.stderr,
                                flush=True,
                            )
                        latent_x0.append(cdiff["x0"])
                        latent_cdiff_targets.append(cdiff["target"])
                        latent_velocities.append(cdiff["velocity"])
                    if include_latent_velocity_query:
                        query_t0 = time.perf_counter() if profile else None
                        x0_fn = model.get_x0_fn_from_batch(
                            result["data_batch"],
                            guidance=0,
                            is_negative_prompt=False,
                        )
                        query_x = torch.as_tensor(
                            latent_query_x[idx: idx + 1],
                            device=result["generated_latent"].device,
                            dtype=torch.float32,
                        )
                        query_t = torch.as_tensor(
                            latent_query_t[idx: idx + 1],
                            device=result["generated_latent"].device,
                            dtype=torch.float32,
                        )
                        if include_joint_action_query:
                            query_action = torch.as_tensor(
                                action_query_x[idx: idx + 1],
                                device=result["generated_latent"].device,
                                dtype=torch.float32,
                            )
                            action_index = result["data_batch"]["action_latent_idx"]
                            query_x = _inject_normalized_action_chunk(
                                query_x, query_action, action_index
                            )
                            mask = torch.zeros(
                                (1, query_x.shape[2]), dtype=torch.bool
                            )
                            for key in (
                                "future_wrist_image_latent_idx",
                                "future_wrist_image2_latent_idx",
                                "future_image_latent_idx",
                                "future_image2_latent_idx",
                            ):
                                frame_index = result["latent_indices"].get(key, -1)
                                if int(frame_index) >= 0:
                                    mask[0, int(frame_index)] = True
                            joint_queries.append(
                                query_x.detach().cpu().numpy().astype(np.float32)
                            )
                            video_frame_masks.append(mask.numpy())
                        query_v = _velocity_from_x0_fn(model, x0_fn, query_x, query_t)
                        if profile:
                            print(
                                "[cosmos_worker_profile] "
                                f"sample={idx} latent_velocity_query_s={time.perf_counter() - query_t0:.3f}",
                                file=sys.stderr,
                                flush=True,
                            )
                        latent_query_velocities.append(
                            query_v.detach().cpu().numpy().astype(np.float32)
                        )
                    if include_future:
                        future_predictions.append(
                            {
                                key: np.asarray(value, dtype=np.uint8)
                                for key, value in result.get("future_image_predictions", {}).items()
                                if value is not None
                            }
                        )
                        value_predictions.append(float(result.get("value_prediction", 0.0)))
                    if profile:
                        print(
                            "[cosmos_worker_profile] "
                            f"sample={idx} total_s={time.perf_counter() - sample_t0:.3f}",
                            file=sys.stderr,
                            flush=True,
                        )

            actions = np.stack(actions, axis=0)
            if has_budget_request:
                if not observed_joint_nfes or len(set(observed_joint_nfes)) != 1:
                    raise RuntimeError(
                        "official worker observed inconsistent joint sampler NFE "
                        f"across batch: {observed_joint_nfes!r}"
                    )
                matched_budget_fields = _official_matched_budget_response_fields(
                    request,
                    configured_action_steps=args.num_denoising_steps_action,
                    configured_future_steps=cfg.num_denoising_steps_future_state,
                    include_future=include_future,
                    observed_joint_nfe=observed_joint_nfes[0],
                )
            actions_path = request["actions_path"]
            fields = {"actions": actions, **matched_budget_fields}
            if include_future:
                keys = sorted(
                    {
                        key
                        for prediction in future_predictions
                        for key, value in prediction.items()
                        if value is not None
                    }
                )
                fields["future_prediction_keys"] = np.asarray(keys)
                fields["value_prediction"] = np.asarray(value_predictions, dtype=np.float32)
                for key in keys:
                    fields[f"future_{key}"] = np.stack(
                        [prediction[key] for prediction in future_predictions],
                        axis=0,
                    )
            if include_latent_cdiff:
                fields["cosmos_latent_x0"] = np.concatenate(latent_x0, axis=0)
                fields["cosmos_latent_cdiff_target"] = np.concatenate(latent_cdiff_targets, axis=0)
                fields["cosmos_latent_velocity"] = np.concatenate(latent_velocities, axis=0)
            elif include_latent_x0:
                fields["cosmos_latent_x0"] = np.concatenate(latent_x0, axis=0)
            if include_latent_velocity_query:
                fields["cosmos_latent_velocity"] = np.concatenate(latent_query_velocities, axis=0)
            if include_joint_action_query:
                fields["cosmos_joint_query"] = np.concatenate(joint_queries, axis=0)
                fields["cosmos_video_frame_mask"] = np.concatenate(
                    video_frame_masks, axis=0
                )
            np.savez_compressed(actions_path, **fields)
            if profile:
                print(
                    "[cosmos_worker_profile] "
                    f"request_total_s={time.perf_counter() - request_t0:.3f} "
                    f"batch={len(tasks)} latent_cdiff={include_latent_cdiff} "
                    f"latent_velocity_query={include_latent_velocity_query}",
                    file=sys.stderr,
                    flush=True,
                )
            print(json.dumps({"ok": True, "actions_path": actions_path}), file=response_out, flush=True)
        except Exception:
            print(
                json.dumps({"ok": False, "error": traceback.format_exc()}),
                file=response_out,
                flush=True,
            )
        finally:
            if data is not None:
                data.close()


if __name__ == "__main__":
    main()
