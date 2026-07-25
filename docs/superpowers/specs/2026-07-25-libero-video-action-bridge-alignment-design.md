# LIBERO Video-Action Bridge Alignment Design

## Goal

Repair the LIBERO LingBot-VA Stage-2 video-only OPD bridge so that its action
supervision uses the same x0 semantics as the main action objective and its
action history matches autoregressive deployment.

## Confirmed mismatches

1. The bridge is documented as x0 supervision but currently compares predicted
   velocity with `noise - clean_action`.
2. The bridge and DanceOPD rollout use clean ground-truth action history while
   deployment uses the current generated action as both state and condition.
3. Existing mechanism diagnostics change action state and action condition
   together for `E_joint`, so they cannot isolate action-history exposure bias.

## Design

### Bridge objective

Add a focused helper that converts predicted velocity to x0:

```python
pred_x0 = noisy_action - sigma * predicted_velocity
loss = masked_mse(pred_x0, clean_action)
```

The helper validates broadcast-compatible sigma and preserves the existing
valid-token mask. The bridge weight remains configurable, but the default
loss is now on the same scale and in the same parameterization as the main
action x0 objective.

### Deployment-aligned action history

During the no-grad DanceOPD rollout, each query uses the current generated
action as `condition_action`. The bridge uses the selected generated
`query_action` as its causal action-history condition while retaining a noisy
ground-truth action state and clean x0 target. The causal attention mask
prevents the current clean target action from being visible, so this change
aligns past action history without introducing action OPD.

### Diagnostics

Retain the paper metrics and add deployment-aligned action-context metrics that
hold video and action state fixed while swapping only clean versus generated
action history. Existing JSON keys remain backward compatible.

## Constraints

- Do not modify the main AnyFlow objective.
- Do not enable action OPD.
- Keep student rollout video detached from the bridge action loss.
- Do not overwrite existing checkpoints or output directories.
- Add tests before production changes and run a single-GPU smoke test before
  recommending a new formal run.

## Success criteria

- A unit test proves bridge x0 loss is zero for an exact x0 reconstruction.
- A unit test proves velocity-space equality alone is not the public bridge API.
- Source-level rollout tests prove generated action history is used.
- Diagnostic tests prove action-history-only swaps are recorded independently.
- Existing focused Stage-2 tests remain green.
