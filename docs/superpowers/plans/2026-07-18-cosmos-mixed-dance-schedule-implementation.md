# Cosmos Mixed-Step Dance Schedule Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every mixed Cosmos full-OPD update use an S4/S2/S1-specific Dance rollout of 4/2/1 with velocity weights 1.00/0.50/0.25, while retaining a low-noise query state for S1.

**Architecture:** Keep immutable S4/S2/S1 schedule definitions next to the existing immutable mixed-pair policy definitions.  The full-OPD caller will pass the already rank-synchronized schedule explicitly into the Cosmos Dance helper.  That helper will append the post-update time-zero state only for this dynamic path; all existing callers keep their configured fixed rollout and trajectory behavior.

**Tech Stack:** Python 3.10, PyTorch, pytest, existing `distillation_flowmap` Cosmos/DanceOPD utilities.

## Global Constraints

- Touch only `cosmos_latent_full` mixed-policy behavior; do not change generic DanceOPD, RobotWin, non-mixed Cosmos, primary Cdiff, teacher endpoint rollout, action anchor, datasets, caches, checkpoint roots, or tmux/GPU jobs.
- The selected rank-broadcast `MixedStepSelection` is the only source of the dynamic schedule; never resample and never infer it from mutable config context.
- Exact schedule: `(8,4)/s4 -> (dance K=4, velocity=1.00)`, `(4,2)/s2 -> (K=2, velocity=0.50)`, `(4,1)/s1 -> (K=1, velocity=0.25)`.
- A dynamic K-step Dance trajectory exposes `K + 1` high-to-low noise states by appending the post-update zero-timestep state; legacy trajectories continue exposing their current pre-update states only.
- No training is launched as part of this code change.  GPU preflight remains a separate gate after tests and commit.

---

### Task 1: Add immutable mixed-Dance schedule and terminal-state utility

**Files:**
- Modify: `distillation_flowmap/cosmos_mixed_step_policy.py`
- Modify: `distillation_flowmap/danceopd_query.py`
- Modify: `distillation_flowmap/tests/test_cosmos_mixed_step_policy.py`
- Modify: `distillation_flowmap/tests/test_danceopd_query.py`

**Interfaces:**
- Produces `CosmosMixedDanceSchedule(rollout_steps: int, velocity_weight: float)` and `get_mixed_danceopd_schedule(selection: MixedStepSelection) -> CosmosMixedDanceSchedule`.
- Produces `append_post_update_trajectory_state(states: list[torch.Tensor], timesteps: list[torch.Tensor], *, state: torch.Tensor, timestep: torch.Tensor) -> None`.
- The schedule resolver accepts only the three approved endpoint pairs and raises `ValueError` for any unsupported pair.

- [ ] **Step 1: Write the failing schedule and terminal-state tests.**

  Add to `test_cosmos_mixed_step_policy.py`:

  ```python
  @pytest.mark.parametrize(
  "forced_index, expected", [(0, (1, 0.25)), (1, (2, 0.50)), (2, (4, 1.00))]
  )
  def test_mixed_dance_schedule_matches_the_selected_endpoint_pair(forced_index, expected):
      selection = select_rank_synchronized_pair(
          get_mixed_step_policy_spec("universe"), forced_indices=(forced_index,)
      )

      schedule = get_mixed_danceopd_schedule(selection)

      assert (schedule.rollout_steps, schedule.velocity_weight) == pytest.approx(expected)
      assert schedule.rollout_steps == selection.student_steps
  ```

  Add to `test_danceopd_query.py`:

  ```python
  def test_appending_post_update_state_preserves_a_time_zero_terminal_query():
      states = [torch.full((1,), 10.0)]
      timesteps = [torch.full((1,), 1000.0)]
      endpoint = torch.tensor([0.0], requires_grad=True)
      zero_t = torch.zeros(1)

      append_post_update_trajectory_state(
          states, timesteps, state=endpoint, timestep=zero_t
      )

      assert len(states) == len(timesteps) == 2
      assert torch.equal(states[-1], torch.tensor([0.0]))
      assert not states[-1].requires_grad
      assert torch.equal(timesteps[-1], zero_t)
  ```

- [ ] **Step 2: Run the focused tests and verify they fail because the new imports/functions do not exist.**

  Run:

  ```bash
  /root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
    distillation_flowmap/tests/test_cosmos_mixed_step_policy.py \
    distillation_flowmap/tests/test_danceopd_query.py
  ```

  Expected: collection/import failure for `get_mixed_danceopd_schedule` and `append_post_update_trajectory_state`.

- [ ] **Step 3: Implement the smallest pure interfaces.**

  After `MixedStepSelection` in `cosmos_mixed_step_policy.py`, add:

  ```python
  @dataclass(frozen=True)
  class CosmosMixedDanceSchedule:
      rollout_steps: int
      velocity_weight: float


  _MIXED_DANCEOPD_SCHEDULES = {
      (4, 1): CosmosMixedDanceSchedule(rollout_steps=1, velocity_weight=0.25),
      (4, 2): CosmosMixedDanceSchedule(rollout_steps=2, velocity_weight=0.50),
      (8, 4): CosmosMixedDanceSchedule(rollout_steps=4, velocity_weight=1.00),
  }


  def get_mixed_danceopd_schedule(selection: MixedStepSelection) -> CosmosMixedDanceSchedule:
      try:
          return _MIXED_DANCEOPD_SCHEDULES[selection.rollout_step_pair]
      except KeyError as exc:
          raise ValueError(
              "Unsupported Cosmos mixed DanceOPD pair "
              f"{selection.rollout_step_pair!r}"
          ) from exc
  ```

  Add the following utility to `danceopd_query.py`:

  ```python
  def append_post_update_trajectory_state(
      states: list[torch.Tensor],
      timesteps: list[torch.Tensor],
      *,
      state: torch.Tensor,
      timestep: torch.Tensor,
  ) -> None:
      if len(states) != len(timesteps):
          raise ValueError("trajectory states and timesteps must have the same length")
      states.append(state.detach().clone())
      timesteps.append(timestep.detach().clone())
  ```

- [ ] **Step 4: Re-run the focused tests and verify they pass.**

  Run the command from Step 2.

  Expected: PASS; the schedule is exact and the appended endpoint is detached at time zero.

- [ ] **Step 5: Commit the self-contained pure utilities and tests.**

  ```bash
  git add distillation_flowmap/cosmos_mixed_step_policy.py \
    distillation_flowmap/danceopd_query.py \
    distillation_flowmap/tests/test_cosmos_mixed_step_policy.py \
    distillation_flowmap/tests/test_danceopd_query.py
  git commit -m "feat(cosmos): define mixed Dance rollout schedule"
  ```

### Task 2: Wire the schedule through only the Cosmos full mixed objective

**Files:**
- Modify: `distillation_flowmap/flowmap_step.py`
- Modify: `distillation_flowmap/tests/test_cosmos_mixed_step_policy.py`

**Interfaces:**
- Extends `_cosmos_danceopd_velocity_loss(..., *, cfg_scale, teacher_crop_context=None, rollout_steps=None, append_post_update_terminal=False)`.
- `rollout_steps=None` retains `config.opd_danceopd_rollout_steps`; `append_post_update_terminal=False` retains the old state list.
- Full mixed Cosmos passes the selected schedule's `rollout_steps`, sets `append_post_update_terminal=True`, and uses its `velocity_weight` rather than the global config weight.
- Adds result metrics `danceopd_velocity_rollout_steps`, `danceopd_velocity_state_count`, and `danceopd_velocity_weight`.

- [ ] **Step 1: Write the failing endpoint-wiring regression test.**

  Extend `_EndpointWireProbe` in `test_cosmos_mixed_step_policy.py` so its `_cosmos_danceopd_velocity_loss` captures keyword arguments and returns diagnostics with `rollout_steps` and `state_count`:

  ```python
  def _cosmos_danceopd_velocity_loss(self, *_args, **kwargs):
      self.dance_calls.append({
          "rollout_steps": kwargs.get("rollout_steps"),
          "append_post_update_terminal": kwargs.get("append_post_update_terminal"),
      })
      zero = torch.zeros((), device=self.device, requires_grad=True)
      effective_rollout_steps = (
          kwargs["rollout_steps"]
          if kwargs["rollout_steps"] is not None
          else self.config.opd_danceopd_rollout_steps
      )
      return zero, {
          "query_index_mean": zero.detach(),
          "query_sigma_mean": zero.detach(),
          "terminal_prior_max_error": zero.detach(),
          "rollout_steps": effective_rollout_steps,
          "state_count": effective_rollout_steps + int(
              bool(kwargs["append_post_update_terminal"])
          ),
      }
  ```

  In the existing forced-S2 integration test, assert:

  ```python
  assert probe.dance_calls == [{"rollout_steps": 2, "append_post_update_terminal": True}]
  assert result["danceopd_velocity_rollout_steps"] == 2
  assert result["danceopd_velocity_state_count"] == 3
  assert result["danceopd_velocity_weight"] == pytest.approx(0.50)
  ```

  Add a legacy probe configuration with `cosmos_mixed_step_policy=""`; assert it calls Dance with `rollout_steps=None`, `append_post_update_terminal=False`, retains global weight `1.0`, and reports the configured 16-step behavior.

- [ ] **Step 2: Run the focused mixed-policy test and verify it fails at the new keyword/metric assertions.**

  Run:

  ```bash
  /root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
    distillation_flowmap/tests/test_cosmos_mixed_step_policy.py
  ```

  Expected: FAIL because the full-OPD caller still invokes Dance without the dynamic schedule and does not return the new diagnostics.

- [ ] **Step 3: Implement the minimal scoped wiring.**

  In `flowmap_step.py`:

  ```python
  from distillation_flowmap.danceopd_query import (
      append_post_update_trajectory_state,
      denoised_endpoint_mse,
      direct_velocity_mse,
      sample_low_noise_query_indices,
      select_per_sample_trajectory_state,
  )
  from distillation_flowmap.cosmos_mixed_step_policy import (
      append_selection_jsonl,
      build_selection_record,
      get_mixed_danceopd_schedule,
      get_mixed_step_policy_spec,
      record_selection,
      select_rank_synchronized_pair,
  )
  ```

  Resolve `effective_rollout_steps` from the optional helper argument, validate it is positive, and after the no-grad Euler loop append `current_video/current_action` with `zero_video_t/zero_action_t` only when `append_post_update_terminal` is true.  Sample with `n_states=len(video_states)` rather than the rollout count.  Return `rollout_steps` and `state_count` in Dance diagnostics.

  At the full-OPD call site, use:

  ```python
  dance_schedule = (
      get_mixed_danceopd_schedule(mixed_step_selection)
      if mixed_step_selection is not None else None
  )
  velocity_loss, dance_diagnostics = self._cosmos_danceopd_velocity_loss(
      batch, teacher, self._prepare_base_dict(batch), cfg_scale=cfg_scale,
      teacher_crop_context=teacher_crop_context,
      rollout_steps=(dance_schedule.rollout_steps if dance_schedule else None),
      append_post_update_terminal=(dance_schedule is not None),
  )
  velocity_weight = (
      dance_schedule.velocity_weight if dance_schedule is not None
      else float(getattr(self.config, "opd_danceopd_velocity_weight", 1.0))
  )
  ```

  Add the three explicit `danceopd_velocity_*` metrics from the schedule/diagnostics.  Do not mutate `self.config`, and do not alter the generic DanceOPD method later in the file.

- [ ] **Step 4: Re-run the mixed-policy test and verify it passes.**

  Run the command from Step 2.

  Expected: PASS; forced S2 routes endpoint K=2, Dance K=2, three query states, and velocity weight 0.50, while legacy behavior is unchanged.

- [ ] **Step 5: Run the complete scoped regression suite and static checks.**

  ```bash
  /root/nas/junjie/conda_envs/any_wam/bin/python -m pytest -q \
    distillation_flowmap/tests/test_cosmos_mixed_step_policy.py \
    distillation_flowmap/tests/test_danceopd_query.py \
    distillation_flowmap/tests/test_cosmos_progressive_config.py \
    distillation_flowmap/tests/test_cosmos_latent_checkpoint_config.py
  /root/nas/junjie/conda_envs/any_wam/bin/python -m py_compile \
    distillation_flowmap/flowmap_step.py \
    distillation_flowmap/cosmos_mixed_step_policy.py \
    distillation_flowmap/danceopd_query.py
  git diff --check
  ```

  Expected: all tests pass, compilation exits zero, and no whitespace errors.

- [ ] **Step 6: Commit the integration and tests.**

  ```bash
  git add distillation_flowmap/flowmap_step.py \
    distillation_flowmap/tests/test_cosmos_mixed_step_policy.py
  git commit -m "feat(cosmos): match Dance rollout to mixed endpoint"
  ```

### Task 3: Audit, publish, and leave training gated

**Files:**
- Modify: `docs/superpowers/specs/2026-07-18-cosmos-mixed-dance-schedule-design.md` only if validation reveals a factual mismatch.
- Modify: `docs/superpowers/plans/2026-07-18-cosmos-mixed-dance-schedule-implementation.md` only to check completed boxes after execution.

**Interfaces:**
- Produces a clean isolated worktree, recorded commit hashes, and a fast-forward push to `research/cosmos-stage2-progressive` only after the scoped regression suite passes.

- [ ] **Step 1: Inspect the final diff and verify scope.**

  Run:

  ```bash
  git status --short
  git diff HEAD~2..HEAD -- \
    distillation_flowmap/flowmap_step.py \
    distillation_flowmap/cosmos_mixed_step_policy.py \
    distillation_flowmap/danceopd_query.py \
    distillation_flowmap/tests
  ```

  Expected: only the approved schedule, terminal-state utility, wiring, metrics, and regression tests are present.

- [ ] **Step 2: Push the verified commits without changing the user's dirty checkout.**

  ```bash
  git push origin HEAD:refs/heads/research/cosmos-stage2-progressive
  ```

  Expected: fast-forward success.  Do not reset, clean, merge, or write in `/root/nas/junjie/jj/Any_WAM_cosmos_stage2_progressive`.

- [ ] **Step 3: Report the exact static evidence and preserve the training gate.**

  Report commit hashes, passed test count, static-check result, the unchanged external training state, and that no Cosmos preflight/training was launched.  Recreate only the paused preflight waiting queue after this report and only if GPUs remain occupied; never auto-launch Universe/S2/S1.
