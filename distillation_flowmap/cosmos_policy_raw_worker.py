"""Line-oriented worker for official Cosmos Policy raw inference.

The Any_WAM training env currently does not import the official Cosmos package
with CUDA extras. This worker is launched under the Cosmos env, loads the policy
once, and serves action chunks over stdin/stdout JSON messages.
"""

import argparse
import json
import os
import sys
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


def _velocity_from_x0_fn(model, x0_fn, x_t, t):
    t = t.to(device=x_t.device, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    sigma = t / (1.0 - t)
    sigma_view = _time_view(sigma).to(x_t)
    x_sigma = x_t * (1.0 + sigma_view)
    x0_pred = x0_fn(x_sigma, sigma)
    eps_pred = (x_sigma - x0_pred) / sigma_view
    return eps_pred - x0_pred


def _compute_latent_cdiff(model, data_batch, x0_anchor, noise, t, r, epsilon):
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
    v_center = _velocity_from_x0_fn(model, x0_fn, x_t, t_center)
    v_plus = _velocity_from_x0_fn(model, x0_fn, x_plus, t_plus)
    v_minus = _velocity_from_x0_fn(model, x0_fn, x_minus, t_minus)
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
            actions = []
            include_future = bool(request.get("include_future_predictions", False))
            include_latent_cdiff = bool(request.get("include_latent_cdiff", False))
            latent_noise = data["cosmos_latent_noise"] if include_latent_cdiff else None
            latent_t = data["cosmos_latent_t"] if include_latent_cdiff else None
            latent_r = data["cosmos_latent_r"] if include_latent_cdiff else None
            latent_epsilon = float(request.get("cosmos_latent_epsilon", 0.001))
            future_predictions = []
            value_predictions = []
            latent_x0 = []
            latent_cdiff_targets = []
            latent_velocities = []
            with torch.no_grad():
                for idx, task in enumerate(tasks):
                    obs = {
                        "primary_image": primary[idx],
                        "wrist_image": wrist[idx],
                        "proprio": proprio[idx],
                    }
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
                    actions.append(np.asarray(result["actions"], dtype=np.float32))
                    if include_latent_cdiff:
                        cdiff = _compute_latent_cdiff(
                            model,
                            result["data_batch"],
                            result["generated_latent"],
                            latent_noise[idx: idx + 1],
                            latent_t[idx: idx + 1],
                            latent_r[idx: idx + 1],
                            latent_epsilon,
                        )
                        latent_x0.append(cdiff["x0"])
                        latent_cdiff_targets.append(cdiff["target"])
                        latent_velocities.append(cdiff["velocity"])
                    if include_future:
                        future_predictions.append(
                            {
                                key: np.asarray(value, dtype=np.uint8)
                                for key, value in result.get("future_image_predictions", {}).items()
                                if value is not None
                            }
                        )
                        value_predictions.append(float(result.get("value_prediction", 0.0)))

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
            np.savez_compressed(actions_path, **fields)
            print(json.dumps({"ok": True, "actions_path": actions_path}), file=response_out, flush=True)
        except Exception:
            print(
                json.dumps({"ok": False, "error": traceback.format_exc()}),
                file=response_out,
                flush=True,
            )


if __name__ == "__main__":
    main()
