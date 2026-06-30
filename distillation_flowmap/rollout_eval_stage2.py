import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.append(str(REPO_ROOT / "wan_va"))
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "distillation_flowmap"))

from distillation.patches import install_flash_attn_stub
install_flash_attn_stub()

from distributed.util import init_distributed, dist_mean
from utils import init_logger, logger
from distillation_flowmap.flowmap_trainer import FlowMapDistiller
from distillation_flowmap.flowmap_step import _downsample_action_grid_id


def parse_pair(text):
    t, r = text.split(",")
    return float(t), float(r)


def init_mask(trainer, input_dict):
    from modules.model import FlexAttnFunc

    ld = input_dict["latent_dict"]
    ad = input_dict["action_dict"]
    total_length = (
        ld["noisy_latents"].flatten(0, 1).shape[0] * 2
        + ad["noisy_latents"].flatten(0, 1).shape[0] * 2
    )
    padded_length = (128 - total_length % 128) % 128
    FlexAttnFunc.init_mask(
        ld["noisy_latents"].shape,
        ad["noisy_latents"].shape,
        padded_length,
        input_dict["chunk_size"],
        window_size=input_dict["window_size"],
        patch_size=trainer.patch_size,
        device=trainer.device,
    )


def masked_mse_l1(diff, mask):
    if mask is None:
        return (diff.float().pow(2).mean(), diff.float().abs().mean())
    mask = mask.float()
    denom = (mask.sum() * diff.shape[1]).clamp(min=1)
    return ((diff.float() * mask).pow(2).sum() / denom,
            (diff.float().abs() * mask).sum() / denom)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="distillation_flowmap.config_libero_fullfinetune_stage2_anyflow")
    parser.add_argument("--teacher-model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from-path", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--pairs", nargs="+", default=["1000,0", "1000,500", "750,250"])
    parser.add_argument("--student-steps", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--teacher-steps", type=int, default=4)
    args = parser.parse_args()

    init_logger()
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)

    cfg = importlib.import_module(args.config).cfg
    cfg.rank = rank
    cfg.local_rank = local_rank
    cfg.world_size = world_size
    cfg.teacher_model_path = args.teacher_model_path
    cfg.dataset_path = args.dataset_path
    cfg.empty_emb_path = os.path.join(args.dataset_path, "empty_emb.pt")
    cfg.output_dir = args.output_dir
    cfg.resume_from_path = args.resume_from_path
    cfg.reset_resume_step = False
    cfg.enable_wandb = False
    cfg.enable_light_eval = False
    cfg.enable_stage1_start_eval = False
    cfg.enable_stage1_start_eval_baseline = False
    cfg.skip_teacher_compile = True
    cfg.light_eval_num_batches = args.num_batches
    cfg.light_eval_seed = args.seed
    cfg.light_eval_start_index = 0
    # Keep memory closer to training but avoid the full no-FSDP student copy for this offline pass.
    cfg.opd_aux_use_nofsdp_rollout = False

    trainer = FlowMapDistiller(cfg)
    trainer.student.eval()
    trainer.target_student.eval()
    if getattr(trainer, "_student_nofsdp", None) is not None:
        trainer._student_nofsdp.eval()

    action_ds = getattr(trainer.config, "action_downsample_factor", 4)
    pairs = [parse_pair(p) for p in args.pairs]
    metrics = {}
    counts = {}

    def add(name, value):
        value = value.detach().float()
        metrics[name] = metrics.get(name, torch.zeros((), device=trainer.device)) + value
        counts[name] = counts.get(name, 0) + 1

    batches = trainer._get_light_eval_batches()
    for batch_idx, batch in enumerate(batches[:args.num_batches]):
        batch = trainer._move_eval_batch_to_device(batch)
        base_input = trainer._prepare_base_dict(batch)
        B = batch["latents"].shape[0]
        ref_shape = batch["latents"].shape
        num_frames = ref_shape[2]
        empty_emb = trainer.empty_emb.expand(base_input["latent_dict"]["text_emb"].shape[0], -1, -1)

        for pair_idx, (t_value, r_value) in enumerate(pairs):
            gen = torch.Generator(device=trainer.device)
            gen.manual_seed(args.seed + batch_idx * 1009 + pair_idx)
            video_t = torch.full((B, num_frames), t_value, device=trainer.device, dtype=torch.float32)
            video_r = torch.full((B, num_frames), r_value, device=trainer.device, dtype=torch.float32)
            action_t = video_t
            action_r = video_r

            video_noise = torch.randn(batch["latents"].shape, device=trainer.device,
                                      dtype=batch["latents"].dtype, generator=gen)
            action_noise = torch.randn(batch["actions"].shape, device=trainer.device,
                                       dtype=batch["actions"].dtype, generator=gen)
            video_noisy_t = trainer.train_scheduler_latent.add_noise(
                batch["latents"], video_noise, video_t, t_dim=2)
            video_noisy_r = trainer.train_scheduler_latent.add_noise(
                batch["latents"], video_noise, video_r, t_dim=2)
            video_v_target_r = trainer.train_scheduler_latent.training_target(
                batch["latents"], video_noise, video_r)

            action_noisy_t = trainer.train_scheduler_action.add_noise(
                batch["actions"], action_noise, action_t, t_dim=2)
            action_noisy_r = trainer.train_scheduler_action.add_noise(
                batch["actions"], action_noise, action_r, t_dim=2)
            action_v_target_t = trainer.train_scheduler_action.training_target(
                batch["actions"], action_noise, action_t)

            input_dict = {
                "latent_dict": {
                    **base_input["latent_dict"],
                    "noisy_latents": video_noisy_t,
                    "timesteps": video_t,
                    "targets": trainer.train_scheduler_latent.training_target(
                        batch["latents"], video_noise, video_t),
                },
                "action_dict": {
                    **base_input["action_dict"],
                    "noisy_latents": action_noisy_t,
                    "timesteps": action_t,
                    "targets": action_v_target_t,
                },
                "chunk_size": base_input["chunk_size"],
                "window_size": base_input["window_size"],
            }
            student_input = {
                "latent_dict": input_dict["latent_dict"],
                "action_dict": {
                    "noisy_latents": action_noisy_t[:, :, ::action_ds],
                    "latent": batch["actions"][:, :, ::action_ds],
                    "timesteps": action_t[:, ::action_ds],
                    "cond_timesteps": input_dict["action_dict"]["cond_timesteps"][:, ::action_ds],
                    "text_emb": input_dict["action_dict"]["text_emb"],
                    "grid_id": _downsample_action_grid_id(
                        input_dict["action_dict"].get("grid_id"), batch["actions"], action_ds),
                    "actions_mask": (
                        input_dict["action_dict"]["actions_mask"][:, :, ::action_ds]
                        if input_dict["action_dict"].get("actions_mask") is not None else None
                    ),
                },
                "chunk_size": input_dict["chunk_size"],
                "window_size": input_dict["window_size"],
            }

            pair_name = f"t{int(t_value)}_r{int(r_value)}"
            teacher_x_r, teacher_v_r = trainer._teacher_integrate_to_r(
                noisy_latents=video_noisy_t,
                timesteps=video_t,
                target_r=video_r,
                input_dict=input_dict,
                empty_emb=empty_emb,
                cfg_scale=args.cfg_scale,
                ref_shape=ref_shape,
                B=B,
                num_frames=num_frames,
                num_steps=args.teacher_steps,
            )

            for k_steps in args.student_steps:
                student_x_r, student_v_r = trainer._student_euler_integrate(
                    noisy_latents=video_noisy_t,
                    timesteps=video_t,
                    target_r=video_r,
                    base_input_dict=student_input,
                    empty_emb=empty_emb,
                    cfg_scale=args.cfg_scale,
                    ref_shape=ref_shape,
                    B=B,
                    num_frames=num_frames,
                    K_steps=k_steps,
                    action_target_r=action_r,
                )
                prefix = f"rollout_eval/{pair_name}/s{k_steps}_t{args.teacher_steps}"
                add(prefix + "/video_teacher_x_mse", (student_x_r.float() - teacher_x_r.float()).pow(2).mean())
                add(prefix + "/video_teacher_x_l1", (student_x_r.float() - teacher_x_r.float()).abs().mean())
                add(prefix + "/video_teacher_v_mse", (student_v_r.float() - teacher_v_r.float()).pow(2).mean())
                add(prefix + "/video_teacher_v_l1", (student_v_r.float() - teacher_v_r.float()).abs().mean())
                add(prefix + "/video_gt_x_mse", (student_x_r.float() - video_noisy_r.float()).pow(2).mean())
                add(prefix + "/video_gt_x_l1", (student_x_r.float() - video_noisy_r.float()).abs().mean())
                add(prefix + "/video_gt_v_mse", (student_v_r.float() - video_v_target_r.float()).pow(2).mean())
                add(prefix + "/video_latent_norm", student_x_r.float().pow(2).mean().sqrt())

                action_input = {
                    "latent_dict": {
                        **input_dict["latent_dict"],
                        "noisy_latents": student_x_r.detach(),
                        "timesteps": video_r,
                    },
                    "action_dict": student_input["action_dict"],
                    "chunk_size": input_dict["chunk_size"],
                    "window_size": input_dict["window_size"],
                }
                init_mask(trainer, action_input)
                _, student_action_seq = trainer.student(
                    action_input, train_mode=True,
                    r_timestep=video_r,
                    action_r_timestep=action_r[:, ::action_ds],
                )
                action_frames = batch["actions"].shape[2] // action_ds
                student_action_v = trainer._extract_action_v(student_action_seq, action_frames)
                action_noisy_t_ds = action_noisy_t[:, :, ::action_ds]
                action_noisy_r_ds = action_noisy_r[:, :, ::action_ds]
                action_v_target_ds = action_v_target_t[:, :, ::action_ds]
                action_sigma_t = action_t[:, None, ::action_ds, None, None] / trainer.config.num_train_timesteps
                action_sigma_r = action_r[:, None, ::action_ds, None, None] / trainer.config.num_train_timesteps
                action_pred_xr = action_noisy_t_ds + student_action_v * (
                    action_sigma_r.to(student_action_v) - action_sigma_t.to(student_action_v))
                mask = batch.get("actions_mask")
                mask_ds = mask[:, :, ::action_ds] if mask is not None else None
                mse, l1 = masked_mse_l1(action_pred_xr - action_noisy_r_ds, mask_ds)
                add(prefix + "/action_gt_xr_mse", mse)
                add(prefix + "/action_gt_xr_l1", l1)
                mse, l1 = masked_mse_l1(student_action_v - action_v_target_ds, mask_ds)
                add(prefix + "/action_gt_v_mse", mse)
                add(prefix + "/action_gt_v_l1", l1)
                if student_action_v.shape[2] > 1:
                    smooth = (student_action_v[:, :, 1:] - student_action_v[:, :, :-1]).float().abs().mean()
                    add(prefix + "/action_v_smoothness_l1", smooth)
                add(prefix + "/action_v_norm", student_action_v.float().pow(2).mean().sqrt())

    out = {}
    for name, total in metrics.items():
        avg = total / max(1, counts[name])
        if dist.is_initialized():
            avg = dist_mean(avg)
        out[name] = avg.item()

    if rank == 0:
        Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.result_json, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
        logger.info("Wrote rollout eval metrics to %s", args.result_json)
        print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
