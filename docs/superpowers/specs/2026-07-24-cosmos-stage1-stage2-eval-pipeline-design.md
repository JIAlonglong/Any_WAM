# Cosmos Stage-1 → Stage-2 → Evaluation Pipeline Design

Date: 2026-07-24
Status: Approved design
Target branch: `codex/cosmos-progressive-s4-eval`

## 1. Goal

Provide one resumable command that uses a single long-lived 8×A800 allocation
to run, in order:

1. executable training-contract verification;
2. corrected raw Cosmos action Stage-1 training for 5,000 steps;
3. corrected progressive `universal-video-action` Stage-2 training for 5,000
   steps;
4. full LIBERO closed-loop evaluation for joint video/action K=1, K=2, and
   K=4, with 500 episodes per K.

The pipeline must fail closed. It must never evaluate an old or unverified
checkpoint as a formal result, silently reuse incompatible state, overwrite a
prior run, or continue after a failed stage.

## 2. Scope

### In scope

- Correct the raw 16-action packing used by Cosmos action anchors.
- Add an explicit deployment-aligned joint K rollout objective covering the
  complete raw timestep interval 1000→0 for K=1/2/4.
- Keep the reliable raw Cosmos teacher window as a separate auxiliary
  objective.
- Add executable CPU contract verification and checkpoint contract metadata.
- Train a new raw Stage-1 from the clean LingBotVA LIBERO base.
- Train only the required progressive `universal-video-action` Stage-2 from
  the corrected Stage-1 EMA/target.
- Reuse the verified K1→K2→K4, four-shard evaluation implementation.
- Add one state-machine orchestrator with safe resume, status, dry-run,
  provenance, signal handling, and fail-stop behavior.

### Out of scope

- Reusing the contaminated `raw_stage1_5000` or any Stage-2 descendant.
- Retraining independent S1-only, S2-only, S4-only, or video-only arms.
- Changing the frozen official Cosmos Policy teacher.
- Starting a live GPU training or evaluation during implementation.
- Automatically deleting incomplete checkpoints or old output directories.
- Treating the current step-5000 checkpoint as a formal result.

## 3. Confirmed Existing Defects

### 3.1 Action packing

The current converter writes 16 raw actions into the first 16 flattened cells
of a `[F=16,N=4]` carrier. They occupy dense frames 0–3. The student then
selects `F::4`, retaining frames 0,4,8,12, so only raw actions 0–3 survive.
The remaining 12 executed slots are trained toward normalized-zero targets.

The original LingBotVA LIBERO policy instead represents the 16-action chunk as
`[F=4,N=4]` and executes all 16 actions.

### 3.2 Deployment trajectory

The latest K-specific raw Cosmos OPD/DanceOPD path is limited to the reliable
raw teacher interval `t∈[0.8,80/81]`. Live K-step inference integrates over
the complete 1000→0 interval. A high-noise auxiliary window cannot be used as
the deployment K trajectory contract.

### 3.3 Contaminated lineage

The published `raw_stage1_5000` was already trained through the faulty action
packing path. The current progressive step-5000 checkpoint inherits that
Stage-1 and continues using the same packing. Neither checkpoint nor their
optimizer/scheduler state is eligible for the corrected formal pipeline.

## 4. Architecture

Add one top-level state-machine orchestrator:

```text
distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh
```

It exposes exactly three modes:

```text
run       Execute or safely resume the pipeline.
dry-run   Print all decisions and commands without creating or changing files.
status    Read and validate existing state without launching child processes.
```

The state machine has four sequential states:

```text
contract preflight
  → raw Stage-1
  → progressive universal-video-action Stage-2
  → joint K1/K2/K4 evaluation
```

Only one state may be active. Every training stage uses all eight visible
GPUs. The evaluation stage uses the established four student/worker GPU pairs:

```text
student 0 / worker 1 / tasks [0,3)
student 2 / worker 3 / tasks [3,6)
student 4 / worker 5 / tasks [6,8)
student 6 / worker 7 / tasks [8,10)
```

## 5. Corrected Action Packing Contract

Define a versioned packing schema:

```text
downsample_survivor_v2
```

For a full carrier `[B,C,F=16,N=4,1]` with downsample factor four, raw action
index `i` is assigned to:

```text
survivor_frame = (i // 4) * 4
slot = i % 4
```

Thus actions 0–15 occupy frames 0,4,8,12 and slots 0–3. After the existing
`F::4`, the student receives the original LingBotVA semantic layout
`[F=4,N=4]` with all 16 actions in order.

The implementation must not infer this schema merely from tensor shape. The
converter receives or derives an explicit schema/version, and training config,
checkpoint config, manifests, and evaluation preflight record the same value.

The authoritative CPU contract test uses distinguishable, nonzero action
values and proves:

```text
raw actions
  → production converter
  → production F::4 downsample
  → production decoder
  → same 16 actions in the same order
```

Shape-only tests are insufficient.

## 6. Deployment-Aligned K Objective

Separate two objectives that previously shared ambiguous terminology.

### 6.1 Full deployment joint rollout

Stage-2 must train an explicit joint video/action rollout over the exact
deployment endpoints:

```text
raw timestep start = 1000
raw timestep end = 0
joint student steps = {1,2,4}
```

Video and action use the same K-loop and return the final integrated action
state.

The target is defined without querying the raw Cosmos velocity API outside its
supported interval:

1. obtain the clean Cosmos video endpoint `video_x0` and the corrected packed
   action endpoint `action_x0`;
2. sample one Gaussian `video_noise` and one Gaussian `action_noise`;
3. start the joint student at raw timestep 1000, where the state is the sampled
   noise;
4. integrate both branches to raw timestep 0 with the same sampled K;
5. compare the final video state directly with `video_x0`;
6. compare the final action state directly with the downsampled
   `[F=4,N=4]` `action_x0`, under the real action mask.

At timestep zero the integrated state is the denoised endpoint, so the loss
does not perform another `x - sigma*v` conversion. The loss is:

```text
L_deploy =
    mean((video_final - video_x0)^2)
  + action_weight * masked_mean((action_final - action_x0)^2)
```

with `action_weight=1.0`. Video and action endpoint terms are logged
separately. K is selected from `{1,2,4}` with a rank-synchronized balanced
cycle, not independently per rank.

The deployment rollout runs every four optimizer steps. Across 5,000 Stage-2
steps this gives 1,250 full-path joint rollout updates, balanced across the
three K values. The raw-window auxiliary retains interval eight with phase
offset two: after its warmup it runs where `step % 8 == 2`. Deployment updates
run where `step % 4 == 0`, so the two expensive rollout graphs never occupy the
same optimizer step.

This objective is the only path allowed to attest that the checkpoint supports
full K-step deployment.

### 6.2 Raw Cosmos auxiliary window

The raw teacher interval remains:

```text
t∈[0.8,80/81]
```

It is retained as a separate auxiliary anchor/OPD objective. Its configuration,
metrics, and names must say `auxiliary`; it must not supply the deployment
start/end metadata or alignment attestation.

## 7. Executable Contract Verification

Add:

```text
distillation_flowmap/verify_cosmos_joint_training_contract.py
```

The verifier imports and executes production helpers. It does not approve a
contract by grepping source text or trusting environment variables.

It validates:

- action packing identity for all 16 actions;
- post-downsample action shape `[B,C,4,4,1]`;
- decoder identity and ordering;
- accepted K set exactly `{1,2,4}`;
- deployment start 1000 and end 0;
- one joint video/action integration loop per K;
- raw teacher window is distinct from deployment rollout;
- config contract fields and production constants agree.

It writes an atomic attestation JSON containing:

```json
{
  "contract_version": 2,
  "action_packing_schema": "downsample_survivor_v2",
  "action_chunk_shape": [4, 4],
  "deployment_timestep_start": 1000,
  "deployment_timestep_end": 0,
  "joint_student_steps": [1, 2, 4],
  "raw_teacher_window_is_auxiliary": true
}
```

The attestation additionally records the git commit, verifier module identity,
and timestamp. It is necessary but not sufficient: each checkpoint must also
carry matching metadata.

## 8. Training Stages

### 8.1 Corrected raw Stage-1

- Source: clean LingBotVA LIBERO base:

  ```text
  /kpfs-intern/jialongliu/projects/lingbot-va/checkpoints/libero
  ```

- Frozen teacher: official Cosmos Policy LIBERO model.
- Mode: raw action Stage-1.
- GPUs: 8.
- Seed: 42.
- Steps: 5,000.
- Default torchrun port: 29671.
- Action packing: `downsample_survivor_v2`.
- Resume: latest complete checkpoint only.

The existing contaminated `raw_stage1_5000` cannot be selected as a base or
resume source.

### 8.2 Corrected progressive Stage-2

- Source: corrected Stage-1 `target_student`/EMA checkpoint.
- Mode: `universal-video-action`.
- GPUs: 8.
- Seed: 42.
- Steps: 5,000.
- Default torchrun port: 29672.
- Joint K values: 1,2,4.
- Full deployment objective: enabled.
- Raw teacher auxiliary window: enabled and separately identified.
- Resume: latest complete Stage-2 checkpoint only.

No other progressive arm is trained by this pipeline.

## 9. Checkpoint Contract and Lineage

Every completed Stage-1 and Stage-2 transformer config and lineage manifest
must include:

```text
contract_version=2
action_packing_schema=downsample_survivor_v2
action_chunk_shape=[4,4]
checkpoint_step
git_commit
parent_checkpoint
parent_checkpoint_identity
dataset_path
seed=42
```

Stage-2 additionally must include:

```text
deployment_timestep_start=1000
deployment_timestep_end=0
joint_student_steps=[1,2,4]
deployment_joint_rollout_interval=4
deployment_action_weight=1.0
raw_teacher_window_is_auxiliary=true
```

Stage-2 validates the Stage-1 contract before loading it. Evaluation validates
the Stage-2 config, lineage manifest, and executable attestation. The
orchestrator sets `S4_ALIGNMENT_VERIFIED=1` only after all three agree.

Old checkpoints with missing or different contract fields remain eligible only
for the existing explicit diagnostic override.

## 10. Output Layout

One caller-provided `RUN_ROOT` owns the entire pipeline:

```text
$RUN_ROOT/
├── manifests/
│   ├── run.json
│   ├── contract-attestation.json
│   ├── stage1.complete.json
│   ├── stage2.complete.json
│   └── evaluation.complete.json
├── state/
│   └── pipeline.json
├── logs/
│   ├── preflight.log
│   ├── stage1.log
│   ├── stage2.log
│   └── evaluation.log
├── stage1/
│   └── checkpoints/step_5000/
├── stage2/
│   └── universal-video-action/checkpoints/step_5000/
└── evaluation/
    ├── k1/formal_summary.json
    ├── k2/formal_summary.json
    ├── k4/formal_summary.json
    └── matrix_summary.json
```

Manifests and state transitions use sibling temporary files plus atomic
rename. The pipeline never overwrites a different run.

## 11. Resume and Idempotency

Re-running the same command against the same `RUN_ROOT`:

- verifies immutable run identity;
- skips a completed stage only after revalidating its checkpoint and manifest;
- resumes Stage-1 or Stage-2 from the highest complete checkpoint containing
  transformer, optimizer, scheduler, and matching `checkpoint_step`;
- preserves and reports incomplete checkpoints instead of deleting them;
- validates a completed evaluation K before skipping it;
- resumes evaluation at the first incomplete K.

Run identity includes:

```text
git commit
clean base path and identity
dataset path
teacher path
seed
contract version
packing schema
Stage-1/Stage-2 requested steps
evaluation protocol
```

If immutable identity differs, the orchestrator refuses to reuse `RUN_ROOT`.

## 12. Failure and Signal Handling

- `set -euo pipefail` and explicit child status aggregation are mandatory.
- A child failure prevents every later state from starting.
- Parallel evaluation shards are all waited and reaped before returning.
- `SIGINT` and `SIGTERM` traps terminate the current process group, wait for
  torchrun/Cosmos workers, and atomically mark the stage interrupted.
- No cleanup routine deletes checkpoints, logs, or result roots.
- A partial or corrupted completion marker is treated as a hard error.
- The final pipeline state is `complete` only after the matrix summary and all
  three child summaries pass verification.

## 13. Evaluation

The existing matrix evaluator is reused with the newly trained Stage-2 online
student:

```text
K order: 1 → 2 → 4
episodes per K: 500
tasks: 10
shared seeds/states per task: 50
formal shards: 4
GPUs: all 8 as four student/worker pairs
saved video seeds: 0,1
saved videos per K: 20
total saved videos: 60
```

Checkpoint provenance, K, task, seed, episode index, shard assignment,
classification, and formal status remain mandatory on every record and
summary.

## 14. Top-Level Interface

The intended launch is:

```bash
cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval &&
RUN_ROOT=/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/runs/cosmos_joint_fixed_20260724 \
bash distillation_flowmap/run_cosmos_stage1_stage2_eval_8gpu.sh run
```

The script uses absolute Python/torchrun paths by default and must not require
the caller to activate Conda. Callers may override documented paths and ports.

`dry-run` prints:

- contract checks;
- stage decisions: run, resume, or skip;
- exact Stage-1 and Stage-2 commands;
- exact checkpoint handoffs;
- exact K1/K2/K4 evaluation commands;
- output and manifest paths.

It creates no file or directory and starts no training/evaluation process.

`status` reads only existing files and starts no child process.

## 15. Tests and Acceptance Criteria

### Production contracts

- Distinguishable 16-action round trip through converter, downsample, decoder.
- K=1/2/4 exact deployment grid and joint video/action loop.
- Raw teacher auxiliary window remains bounded and separately named.
- Checkpoint metadata is saved and validated.

### Orchestrator

- Exact state order and eight-GPU commands.
- Stage-1 uses the clean base, not `raw_stage1_5000`.
- Stage-2 receives the corrected Stage-1 EMA path.
- Evaluation receives the corrected Stage-2 online path.
- Dry-run produces no filesystem mutations or child processes.
- Status is read-only.
- Existing incompatible `RUN_ROOT` is rejected.
- Completed stages are safely skipped.
- Valid complete checkpoints resume; incomplete checkpoints stop safely.
- Stage failure prevents the next stage.
- Signals terminate and reap sentinel child process groups.
- Evaluation resumes at the first incomplete K.

### End-to-end CPU/stub test

A sentinel implementation executes the full state machine without GPUs:

```text
contract → Stage-1 → Stage-2 → K1 → K2 → K4 → final complete
```

It creates realistic checkpoint/manifest fixtures and proves every handoff.

### Final verification

Implementation completion requires:

- all focused contract, launcher, service, and matrix tests passing;
- Python compilation and Bash syntax checks;
- exact real-path dry-run on the intended worktree;
- no created `RUN_ROOT`;
- no training/evaluation/GPU process;
- independent per-task and whole-branch review with no unresolved
  Critical, Important, or Minor finding.

## 16. Operational Estimate

The pipeline is intentionally a single long-lived allocation. Its walltime
must cover:

- corrected raw Stage-1: 5,000 steps;
- progressive Stage-2: 5,000 steps;
- 1,500 closed-loop LIBERO episodes.

The orchestrator does not assume a fixed duration. It reports elapsed time per
stage and supports safe resumption if the allocation ends.
