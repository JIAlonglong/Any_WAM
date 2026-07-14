# RobotWin Final StepWAM Ablation Protocol

## Scope

- Teacher: LingBot-VA only.
- Benchmark: twelve representative RoboTwin tasks:
  - Easy: place_a2b_right, put_object_cabinet, stack_bowls_three, lift_pot,
    place_can_basket, handover_block.
  - Hard: open_microwave, open_laptop, pick_dual_bottles,
    blocks_ranking_size, place_burger_fries, rotate_qrcode.
- Training split: task-local indices 0-39, forty records per task.
- Held-out numerical split: task-local indices 40-49, ten records per task.
- Student budget: K=4 for the primary result; K=1/2/4 curves are reported.
- Teacher budget: equal-NFE T=1/2/4; T=8 is a high-quality reference only.
- Fixed evaluation pairs: 1000->0, 1000->500, and 750->250.
- Stage 1: one frozen representative Stage1 seed-0 checkpoint at step 5000.
- Stage 2: 5000 steps, fixed manifest, fixed teacher cache, fixed pair seeds,
  and fixed evaluation conditioning.

## Main OPD Table

All four variants initialize from exactly the same frozen Stage1 checkpoint and
use one Stage2 seed. No legacy fixed-state velocity objective is used.

| Variant | Endpoint x0 | DanceOPD local velocity | Stage1 |
| --- | --- | --- | --- |
| final_w_o_opd | no | no | shared joint video-action |
| final_endpoint_only_danceopd | yes | no | shared joint video-action |
| final_danceopd_velocity_only | no | yes | shared joint video-action |
| final_stepwam_danceopd | yes | yes | shared joint video-action |

DanceOPD uses a terminal-prior 16-step on-policy rollout, a Beta(5, 2)
low-noise query, and one query per optimizer update. The Stage2 auxiliary is
video-only; action remains jointly distilled in Stage1 and the Stage2 base
objective.

## Structural Controls

These runs are not placed in the main OPD table because they change Stage1.

| Variant | Change | Seeds | Primary report |
| --- | --- | --- | --- |
| final_local_adjacent_only | adjacent-grid transitions replace arbitrary t-to-r shortcuts in both stages | 1 | latent/video and closed-loop |
| final_action_only | action-only Stage1/Stage2 with no video branch or video OPD | 1 | action and closed-loop |

The no-video-loss variant is intentionally excluded.

## Metrics And Assets

- Latent: endpoint MSE/L1 to GT and equal-NFE teacher, same-state velocity
  MSE, action endpoint error, and rollout drift at intermediate states.
- Decoded video: pixel MSE/L1, LPIPS (AlexNet) as the primary perceptual metric,
  PSNR/SSIM as supporting metrics, and temporal difference error:
  MSE((I_s[t+1] - I_s[t]) - (I_gt[t+1] - I_gt[t])).
- FVD is supplementary only and requires a separate 50-clip-per-task
  completely unseen video-test split.
- Closed loop: success rate over 50 episodes per task, teacher retention,
  latency per chunk, Hz, NFE, and speedup.
- Aggregation: task-level metrics first, then macro average. Report the paired
  task delta versus final_w_o_opd and a bootstrap confidence interval over
  tasks; do not describe this as seed significance.
- Persist the run config, git hash, checkpoint, manifests, teacher-cache hash,
  JSON/CSV metrics, and representative teacher/baseline/full videos and
  contact sheets.

For LingBot-VA, decoded videos are produced by the native WanVAE decoding
path. The official future_image_predictions API is specific to Cosmos Policy;
it must not be claimed as a LingBot-VA decoding path.

## Decoded Metric Runtime

- The final numeric runner must use
  `--decoded-video-metrics --decoded-video-device cuda --decoded-video-lpips`.
  LPIPS is fail-fast rather than silently omitted.
- The verified environment is `lpips==0.1.4` with the package's v0.1 AlexNet
  perceptual head. Torchvision's ImageNet AlexNet checkpoint is cached as
  `alexnet-owt-7be5be79.pth`; its SHA256 is
  `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02`.
- CPU WanVAE decode was intentionally rejected for the final protocol after a
  one-record smoke took more than six minutes. CUDA decode completed the same
  smoke successfully and is required for the 120-record held-out evaluation.

## Reproducible Execution

- Run `distillation_flowmap/ablation/run_final_danceopd_eval.sh` after the
  requested step_5000 checkpoints exist. It builds one shared teacher cache
  with T=1/2/4 trajectory states, then replays every compatible checkpoint.
- Main numerical replay requires CUDA WanVAE/LPIPS and preserves the fixed
  held-out manifest, pair seeds, teacher cache, and K=1/2/4, T=1/2/4/8 grids.
- Its asset stage creates a separate one-held-out-record-per-task `1000->0`
  manifest and exports native WanVAE teacher/baseline/full contact sheets.
- The action-only structural control is evaluated and reported through action
  endpoint and closed-loop metrics only; it is excluded from video conclusions.
