# DanceOPD Component Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to execute this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish which OPD component has a held-out signal before promoting DanceOPD to the broader RoboTwin ablation.

**Architecture:** Re-evaluate the existing explicit endpoint/velocity calibration checkpoints with the corrected per-task evaluator, then complete a shared-Stage1 three-seed comparison of `calib_w_o_opd` and `calib_danceopd_i1`. Treat endpoint, fixed-state velocity, and DanceOPD on-policy velocity as distinct objectives; do not combine results from different step budgets as one comparison.

**Tech Stack:** Python 3.10, PyTorch, RobotWin manifests, existing single-GPU launcher/evaluator, JSON/CSV summaries.

## Global Constraints

- Teacher is LingBot-VA; do not modify or run Cosmos.
- Keep student inference at K=4 and teacher rollout at N=8.
- Reuse `/root/nas/junjie/jj/Any_WAM/.worktrees/danceopd-joint-query/distillation_flowmap/output_robotwin_stepwam_ablation/protocol_final12_5000/shared_stage1/representative/seed_0/stage1/checkpoints/step_5000` for every new Stage2 run.
- Use the fixed `core2` train/held-out manifests: 20 train and 10 held-out samples for each of `place_a2b_right` and `open_microwave`.
- Do not overwrite seed 0 checkpoints or manifests.
- Record only trends until a three-seed mean and standard deviation are available.

---

### Task 1: Re-evaluate Existing Explicit Component Checkpoints

**Files:**
- Runtime outputs only: `distillation_flowmap/output_robotwin_stepwam_ablation/protocol_opd_mechanism_calibration_v1/summary_component_recheck/`

**Interfaces:**
- Consumes: `calib_w_o_opd`, `calib_endpoint_only`, `calib_velocity_only`, `calib_full_last_step`, and `calib_full_suffix_grad` seed-0, step-750 checkpoints.
- Produces: held-out and train per-task JSON/CSV metrics under a single fixed evaluator revision.

- [ ] Run a manifest/checkpoint preflight for all five variants.
- [ ] Reuse one teacher target cache and evaluate five variants on GPUs 0-4 without video decoding.
- [ ] Assert every variant has exactly two per-task rows and that aggregate metrics reconcile to the task means.
- [ ] Report endpoint-only, fixed-state velocity-only, and full-objective deltas separately; do not conflate them with DanceOPD.

### Task 2: Preflight Shared-Stage1 Multi-Seed DanceOPD Confirmation

**Files:**
- Runtime outputs only: `distillation_flowmap/output_robotwin_stepwam_ablation/protocol_danceopd_query_gate_v1/{calib_w_o_opd,calib_danceopd_i1}/seed_{1,2}/`

**Interfaces:**
- Consumes: the existing core2 protocol manifests and the shared Stage1 checkpoint.
- Produces: dry-run manifests for seeds 1 and 2 with `MAX_TRAIN_STEPS=250`.

- [ ] Dry-run both `calib_w_o_opd` and `calib_danceopd_i1` for seeds 1 and 2.
- [ ] Verify that all four manifests reference the same Stage1 checkpoint, protocol seed 0, core2 task list, 40 train samples, 20 held-out samples, and step budget 250.
- [ ] Verify that the only Stage2 method difference is `USE_OPD_AUX` and the DanceOPD i1 auxiliary configuration.

### Task 3: Train and Evaluate the Multi-Seed Confirmation

**Files:**
- Runtime outputs only under `protocol_danceopd_query_gate_v1` and `summary_danceopd_i1_confirm/`.

**Interfaces:**
- Consumes: seed 0 results plus the four preflight-approved Stage2 jobs.
- Produces: three seeds per baseline/i1 variant, held-out per-task metrics, K=1/2/4 video metrics, assets, and mean +/- standard deviation.

- [ ] Launch baseline seed 1, i1 seed 1, baseline seed 2, and i1 seed 2 on separate free H100s; no DDP.
- [ ] Stop and inspect immediately on non-finite loss, missing checkpoint, or terminal-prior diagnostic failure.
- [ ] Reuse one held-out teacher cache per manifest and evaluate all six seed/variant combinations with fixed t/r pair seeds.
- [ ] Require a positive mean effect with a confidence interval that does not cross zero for the primary held-out K=4 endpoint metric before promotion.

### Task 4: Promotion Decision

**Files:**
- Modify: `docs/experiments/2026-07-12-danceopd-component-confirmation.md`

**Interfaces:**
- Consumes: component recheck and three-seed confirmation outputs.
- Produces: an explicit decision on the next ablation variant set.

- [ ] Promote only the stable objective to the core4/core6 screen.
- [ ] If endpoint-only is stable but fixed-state velocity is not, compare endpoint-only against DanceOPD i1 rather than using a generic endpoint-plus-velocity hybrid.
- [ ] Add action-only/decoupled-video only after the video-side mechanism has passed the multi-seed gate.
- [ ] Commit the experiment record with the run roots, hashes, metrics, and decision.
