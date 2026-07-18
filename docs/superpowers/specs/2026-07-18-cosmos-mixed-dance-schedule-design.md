# Cosmos Mixed-Step Dance Schedule — Design

## Goal

Make the Cosmos `cosmos_latent_full` mixed-policy objective use the same
student NFE for its DanceOPD trajectory as the endpoint selected for that
scheduled OPD update, while preserving Dance's low-noise same-state velocity
supervision.

The new schedule is sample/update dynamic, not policy-global:

| Selected endpoint pair | Label | Dance rollout steps | Dance velocity weight |
| --- | --- | ---: | ---: |
| teacher 8, student 4 | S4 | 4 | 1.00 |
| teacher 4, student 2 | S2 | 2 | 0.50 |
| teacher 4, student 1 | S1 | 1 | 0.25 |

This applies to every selected pair in Universe, S2, and S1 policies.  In
particular, the minority regularization samples in the specialized policies
retain their own S1/S2/S4 schedule.

## Current problem

The mixed selector already chooses and rank-synchronizes one endpoint pair per
standalone full-OPD update.  Endpoint student integration and teacher rollout
use that pair, but `_cosmos_danceopd_velocity_loss` currently reads a fixed
global value (`opd_danceopd_rollout_steps=16`).  Thus mixed endpoint NFE and
Dance NFE disagree.

A literal `16 -> 1/2/4` substitution is unsafe.  The current trajectory saves
only states before Euler updates.  With one step its only query is terminal
pure noise, while Dance's Beta(5,2) selector is intended to favor low-noise
semantic-side states.

## Design

### Single source of truth

`_cosmos_latent_full_opd_aux_transition_step` will pass the already selected,
rank-broadcast `student_steps` explicitly to the Cosmos Dance helper.  The
helper will not resample a pair and will not rely on mutable configuration
context.  Legacy/non-mixed callers retain the configured fixed Dance rollout
and velocity weight.

### Low-noise trajectory preservation

For the dynamic Cosmos mixed path, Dance will retain the initial high-noise
state and every Euler pre-state, then append the post-update time-zero
endpoint.  Query sampling therefore sees `K + 1` ordered high-to-low noise
states for a K-step rollout:

- S1: pure noise and the post-update time-zero endpoint;
- S2: high-noise, intermediate, and time-zero endpoint states;
- S4: high-noise, three intermediate, and time-zero endpoint states.

The query, teacher field, and gradient-carrying student re-forward remain at
the same selected state/timestep.  No teacher rollout, endpoint loss, action
anchor, or primary Cdiff update changes.

### Weighting and diagnostics

The caller will use effective velocity weights `1.00`, `0.50`, and `0.25` for
S4/S2/S1, respectively.  It will log the effective Dance rollout step count,
state count, and velocity weight alongside the existing selected pair and
endpoint metrics, so `rollout_steps` cannot be mistaken for the Dance setting.

## Scope boundaries

- Change only the Cosmos `cosmos_latent_full` mixed-policy path.
- Keep generic DanceOPD, RobotWin, non-mixed Cosmos, existing S4 checkpoints,
  datasets, cache protocol, and output/ckpt directories unchanged.
- Do not start, stop, resume, or overwrite training while making the code
  change.  The paused preflight queue will be recreated only after tests pass
  and the change is committed.

## Validation

CPU-only tests will first prove the new schedule fails before implementation,
then verify:

1. forced S1/S2/S4 selections pass the selected `student_steps` to Dance;
2. the corresponding effective velocity weights are 0.25/0.50/1.00;
3. dynamic trajectories expose `K + 1` states and include the post-update
   time-zero terminal state, including S1;
4. legacy/non-mixed Cosmos continues to use configured fixed Dance settings;
5. the existing endpoint selector, rank synchronization, and generic OPD
   isolation regressions remain green.

The implementation will then run the focused mixed-policy and Dance query
tests, static compilation, and a diff/working-tree audit.  GPU preflight is a
separate post-merge gate; it will not begin until GPUs are actually free.
