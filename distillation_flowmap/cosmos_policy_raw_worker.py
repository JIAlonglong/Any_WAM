"""Line-oriented worker for official Cosmos Policy raw inference.

The Any_WAM training env currently does not import the official Cosmos package
with CUDA extras. This worker is launched under the Cosmos env, loads the policy
once, and serves action chunks over stdin/stdout JSON messages.
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch


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
        num_denoising_steps_future_state=1,
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


def _generate_from_explicit_prior(
    model, data_batch, joint_prior, *, teacher_steps
):
    """Call only a sampler whose public signature proves explicit prior use."""
    if int(teacher_steps) != 8:
        raise ValueError("same-prior endpoint requires exactly 8 teacher steps")
    generate = model.generate_samples_from_batch
    signature = inspect.signature(generate)
    if "x_sigma_max" not in signature.parameters:
        raise RuntimeError(
            "Cosmos backend cannot honor explicit same-prior sampling"
        )
    if bool(getattr(getattr(model, "config", None), "use_flowunipc_scheduler", False)):
        raise RuntimeError(
            "Cosmos backend is not using the required named EDM sampler path"
        )
    return generate(
        data_batch,
        x_sigma_max=joint_prior,
        num_steps=8,
        solver_option="2ab",
        sigma_max=80,
        guidance=0,
        use_variance_scale=False,
    )


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
    video_prior = torch.as_tensor(video_prior, dtype=torch.float32)
    action_prior = torch.as_tensor(
        action_prior, device=video_prior.device, dtype=torch.float32
    )
    if video_prior.ndim != 5 or video_prior.shape[1] != 16:
        raise ValueError(
            "video_prior must have Cosmos joint latent shape [B,16,T,H,W]"
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
    endpoint = _generate_from_explicit_prior(
        model,
        data_batch,
        joint_prior,
        teacher_steps=teacher_steps,
    )
    if not isinstance(endpoint, torch.Tensor):
        raise RuntimeError("Cosmos same-prior sampler returned a non-tensor endpoint")
    if endpoint.shape != joint_prior.shape:
        raise RuntimeError("Cosmos same-prior sampler returned an invalid endpoint shape")
    if not bool(torch.isfinite(endpoint).all()):
        raise RuntimeError("Cosmos same-prior sampler returned non-finite values")
    return {
        "endpoint_video": endpoint,
        "video_frame_mask": mask,
        "effective_teacher_steps": 8,
    }


def _build_libero_data_batch(cfg, model, dataset_stats, obs, task, cosmos_utils):
    """Reproduce the official LIBERO pre-sampling batch without calling get_action."""
    if cfg.suite != "libero":
        raise RuntimeError("same-prior endpoint currently supports only LIBERO")
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
    model, _ = get_model(cfg)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            data = np.load(request["npz_path"])
            primary = data["primary_image"]
            wrist = data["wrist_image"]
            proprio = data["proprio"].astype(np.float32)
            tasks = request["tasks"]
            mode = request.get("mode", "actions")
            if mode == "same_prior_endpoint":
                teacher_steps = int(request.get("teacher_steps", -1))
                video_prior_np = np.ascontiguousarray(
                    data["video_prior"], dtype=np.float32
                )
                action_prior_np = np.ascontiguousarray(
                    data["action_prior"], dtype=np.float32
                )
                if len(tasks) != video_prior_np.shape[0]:
                    raise ValueError(
                        "same-prior task count does not match video prior batch"
                    )
                if primary.shape[0] != len(tasks) or wrist.shape[0] != len(tasks):
                    raise ValueError(
                        "same-prior observation batch does not match task count"
                    )
                endpoints = []
                masks = []
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
                        endpoints.append(
                            endpoint_result["endpoint_video"]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        masks.append(
                            endpoint_result["video_frame_mask"].detach().cpu().numpy()
                        )
                response_path = request["actions_path"]
                np.savez_compressed(
                    response_path,
                    endpoint_video=np.concatenate(endpoints, axis=0),
                    video_frame_mask=np.concatenate(masks, axis=0).astype(bool),
                    effective_teacher_steps=np.asarray(8, dtype=np.int64),
                    video_prior_sha256=np.asarray(_array_sha256(video_prior_np)),
                    action_prior_sha256=np.asarray(_array_sha256(action_prior_np)),
                )
                data.close()
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
            with torch.no_grad():
                for idx, task in enumerate(tasks):
                    sample_t0 = time.perf_counter() if profile else None
                    obs = {
                        "primary_image": primary[idx],
                        "wrist_image": wrist[idx],
                        "proprio": proprio[idx],
                    }
                    action_t0 = time.perf_counter() if profile else None
                    result = get_action(
                        cfg,
                        model,
                        dataset_stats,
                        obs,
                        task,
                        seed=int(request.get("seed", args.seed)),
                        randomize_seed=False,
                        num_denoising_steps_action=args.num_denoising_steps_action,
                        generate_future_state_and_value_in_parallel=include_future,
                        worker_id=0,
                        batch_size=1,
                    )
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
            actions_path = request["actions_path"]
            fields = {"actions": actions}
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


if __name__ == "__main__":
    main()
