# Two-replica-per-GPU LIBERO evaluation

## Goal

Increase formal native-teacher LIBERO evaluation throughput without changing
the checkpoint, task set, episode count, sampling budgets, action semantics,
or result schema. The active one-replica-per-GPU evaluation is left untouched.

## Constraint discovered

`WebsocketPolicyServer` is intentionally single-client because its teacher
instance owns mutable KV-cache and frame state. Multiple clients cannot share
one server safely: a newer connection closes the older one.

## Design

The dynamic launcher gains a replica count, initially used as `2` per physical
GPU. For each `(gpu, replica, budget)`, it starts an independent teacher server
and one independent LIBERO client lane.  Master/WebSocket ports and worker
directories are unique among the simultaneously active lanes of a budget.
Every lane uses the existing atomic claim directory for
the same 40-task budget queue, so each task is still evaluated exactly once.

Each replica owns its own model process and mutable cache. There is no
cross-client cache sharing or batching. This preserves the current rollout
semantics while overlapping CPU simulation/render time from two environments
on the same 80GB GPU.

## Capacity and safety gates

- Default remains one replica per GPU; the fast launcher explicitly requests
  two only after a capacity smoke test.
- Validate unique lane IDs and port ranges before launch.
- Keep all current checkpoint/output overlap and fresh-result-root guards.
- If any lane exits, terminate sibling lane/server process groups as today.
- The formal rerun uses a new output root; failed-output evidence is retained.

## Validation

1. Offline plan test: four GPUs with two replicas creates eight unique lanes
   per budget, distinct ports and worker roots, and retains one shared queue.
2. Fake-server lifecycle test: a failure in one replica still cleans up every
   sibling process group.
3. One-GPU, two-replica capacity smoke on the A800 before a full restart:
   both servers become healthy, each client completes a minimal task, and no
   CUDA OOM occurs.
4. Compare the resulting task/result coverage with the existing merger's
   40-task and episode-count contract.

## Expected outcome

The first practical target is two simultaneous rollouts per A800, reducing
wall time toward half when simulator work dominates. Actual speedup is
measured in the smoke test; no fixed multiplier is assumed.
