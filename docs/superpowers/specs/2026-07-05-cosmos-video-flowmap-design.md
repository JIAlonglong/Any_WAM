# Cosmos Video FlowMap Design

## Goal

Make the Cosmos Policy LIBERO path comparable to the LingbotVA FlowMap baseline without changing the existing baseline configs or the current action-only Cosmos configs.

The work has two parallel variants:

1. Dual-teacher FlowMap: use WanVA/LingbotVA as the video teacher and Cosmos Policy as the action teacher.
2. Cosmos future-image auxiliary: use official `future_image_predictions` as an explicit auxiliary signal instead of treating them as WanVA latent velocity targets.

## Current State

The existing Cosmos configs are action-only:

- `distillation_flowmap/config_libero_cosmos_policy_stage1.py`
- `distillation_flowmap/config_libero_cosmos_policy_stage2.py`

They intentionally set:

- `teacher_backend = "cosmos_policy"`
- `distill_video = False`
- `distill_action = True`
- `action_use_flowmap = False`
- `diffusion_ratio = 1.0`
- `consistency_ratio = 0.0`
- `flowmap_ratio = 0.0`

This made the compatibility chain run, but it does not reproduce the LingbotVA FlowMap signal. Stage 1 and Stage 2 reached stable losses, but offline action metrics showed no meaningful improvement over the current baseline.

The official Cosmos Policy checkpoint can return `future_image_predictions`, but these are RGB future images/state predictions. They are not WanVA latent velocity predictions at arbitrary `(t, r)`. Directly enabling `distill_video=True` with `CosmosPolicyActionTeacher` would be mathematically wrong because the existing FlowMap video path expects a WanVA-like teacher forward returning latent-space v-predictions.

## Non-Goals

- Do not change `config_libero_fullfinetune_stage1_warmup.py` or `config_libero_fullfinetune_stage2_anyflow.py` behavior.
- Do not change the existing action-only Cosmos configs except for shared bug fixes required by tests.
- Do not make Cosmos future images replace the WanVA teacher in the latent FlowMap video objective.
- Do not run long official training until smoke tests and checkpoint-sweep scripts are working.

## Approach A: Dual-Teacher FlowMap

### Summary

Use two frozen teachers with explicit roles:

- Action teacher: `CosmosPolicyActionTeacher`
- Video teacher: WanVA/LingbotVA transformer loaded from `student_base_model_path` or `video_teacher_model_path`

The student remains a WanVA-compatible FlowMap transformer, so the existing video latent losses, central difference, teacher rollout, and OPD auxiliary can run against the WanVA video teacher. The action branch can still query Cosmos raw policy outputs where available.

### Config

Add new parallel configs:

- `distillation_flowmap/config_libero_cosmos_policy_stage1_dual_teacher.py`
- `distillation_flowmap/config_libero_cosmos_policy_stage2_dual_teacher.py`

Default stage settings:

- `teacher_backend = "cosmos_policy"`
- `action_teacher_backend = "cosmos_policy"`
- `video_teacher_backend = "wanva"`
- `video_teacher_model_path = $VIDEO_TEACHER_MODEL_PATH or cfg.student_base_model_path`
- `distill_mode = "flashwam"`
- `distill_video = True`
- `distill_action = True`
- `action_use_flowmap = True`
- `use_central_diff = True`
- `use_gt_regression = True`
- `diffusion_ratio = 0.5`
- `consistency_ratio = 0.25`
- `flowmap_ratio = 0.25`

Stage 2 should inherit the LingbotVA Stage 2 OPD defaults but keep the action teacher role as Cosmos. `OPD_AUX_ACTION` should remain opt-in until action teacher integration is verified in the OPD path.

### Code Boundary

Add a small role resolver instead of hard-coding conditions throughout training:

- `distillation_flowmap/cosmos_teacher_roles.py`
  - validates teacher-role settings
  - rejects `distill_video=True` with Cosmos as the only video teacher
  - provides normalized action/video backend names and model paths

Modify `flowmap_trainer.py` only at teacher loading:

- keep `self.teacher` and `self._teacher_nofsdp` as the action teacher when `teacher_backend="cosmos_policy"`
- load a separate `self.video_teacher` and `self._video_teacher_nofsdp` when `distill_video=True` and the action teacher is Cosmos
- freeze the video teacher and put the non-FSDP copy on the local GPU

Modify `flowmap_step.py` only through accessors:

- `_action_teacher_model` returns the Cosmos action teacher if present, otherwise the normal teacher
- `_video_teacher_model` returns the WanVA video teacher if present, otherwise the normal teacher
- all video teacher calls use `_video_teacher_model`
- action-only raw Cosmos calls use `_action_teacher_model`

This keeps the existing LingbotVA path unchanged because no extra video teacher is created for normal WanVA configs.

## Approach B: Cosmos Future-Image Auxiliary

### Summary

Use `future_image_predictions` as a separate auxiliary supervision source. It should never be fed into the existing WanVA v-prediction loss as if it were a velocity target.

Implementation will happen in two steps:

1. Diagnostic/eval path: collect and normalize official future images, then log image/action metrics beside the existing offline eval.
2. Optional training loss: encode Cosmos future images into the same WanVA video latent layout and add a small weighted latent x0 auxiliary.

### Target Representation

The official prediction can include multiple image streams. The auxiliary helper must:

- accept `future_image_predictions` from `CosmosPolicyActionTeacher.predict_raw_action_result(..., include_future=True)`
- select the primary and wrist streams by configured keys when available
- convert outputs to float tensors in `[0, 1]`
- resize/crop to the WanVA image size used by the LIBERO dataset
- preserve time order

For a training loss, the helper must build a WanVA-compatible latent target by using the same VAE/image layout as the existing dataset pipeline. The target encoder runs under `torch.no_grad()` and the loss is applied to the student-predicted clean latent, not to teacher velocity.

### Config

New optional fields:

- `cosmos_future_aux_weight`, default `0.0`
- `cosmos_future_aux_interval`, default `16`
- `cosmos_future_aux_num_frames`, default `8`
- `cosmos_future_aux_loss`, default `"latent_l1"`
- `cosmos_future_aux_primary_key`, default from raw observation config
- `cosmos_future_aux_wrist_key`, default from raw observation config

The first committed configs should keep `cosmos_future_aux_weight=0.0` so the path can be smoke-tested without changing training behavior. Experiments can enable it through env vars.

## Checkpoint Sweep and Evaluation

Add a reproducible sweep script that compares:

- current LingbotVA baseline checkpoints
- current Cosmos action-only Stage 1/Stage 2 checkpoints
- new dual-teacher checkpoints
- new future-aux checkpoints when trained

The sweep should emit JSONL/CSV with at least:

- checkpoint path
- config module
- step
- `student_gt/l1`, `student_gt/mse`
- `student_teacher/l1`, `student_teacher/mse`
- `student_v_gtv/l1`, `student_v_gtv/mse` where available
- optional env success rate when official rollout eval is requested
- video path for generated GT/teacher/student comparisons

For quick iteration, smoke runs should use a tiny batch/episode count. Official comparison runs should use fixed seeds, the same task subset, and the same `NUM_STEPS`/`ACTION_NUM_STEPS` as the LingbotVA baseline runs.

## Failure Protections

- If `distill_video=True` and the action teacher is Cosmos, require a valid WanVA `video_teacher_model_path`.
- If the video teacher cannot return latent v-predictions, fail before training starts.
- If `cosmos_future_aux_weight>0` and raw Cosmos inference is disabled, fail before training starts.
- If future image shapes do not map to the expected camera/frame layout, skip the auxiliary for that batch only when the weight is zero; otherwise raise a clear error.
- Keep the existing action-only Cosmos configs available as controls.

## Success Criteria

Before any long run:

- unit tests pass for role resolution and future-image normalization
- a dual-teacher 1-2 step train smoke runs without using Cosmos as the video teacher
- the checkpoint sweep script can evaluate at least one existing checkpoint and produce machine-readable output

For experiments:

- dual-teacher Stage 1 has video/action losses that move similarly to LingbotVA Stage 1
- dual-teacher Stage 2 does not regress the Stage 1 checkpoint on fixed offline action metrics
- generated teacher/student videos use true official future images for Cosmos diagnostics and WanVA latent decoding for student video rollout
- any future-aux improvement must beat the matched dual-teacher control, not only the action-only Cosmos baseline

## Rollout Order

1. Add role resolver and tests.
2. Add checkpoint sweep script and validate it on existing checkpoints.
3. Add dual-teacher loading/accessors and parallel configs.
4. Run smoke training for Stage 1 dual-teacher.
5. Run a short checkpoint comparison against the current action-only Cosmos Stage 1/Stage 2.
6. Add future-image auxiliary helper in eval/diagnostic mode.
7. Add opt-in latent auxiliary loss with default weight zero.
8. Run ablations with small weights and fixed seeds.
