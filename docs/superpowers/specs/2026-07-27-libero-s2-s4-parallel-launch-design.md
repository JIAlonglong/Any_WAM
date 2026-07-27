# LIBERO S2/S4 Parallel Launch Design

## Goal

Provide one small shell entry point that starts the remaining native-teacher LIBERO evaluations concurrently in a newly allocated four-A800 job, while the existing S1 evaluation continues untouched.

## Scope

The wrapper starts exactly two child launchers:

- S2 on physical GPUs `0,1`, with two independent replicas per GPU.
- S4 on physical GPUs `2,3`, with two independent replicas per GPU.

It delegates worker, server, claim, artifact, and per-budget behavior to the already-verified `run_lingbotva_native_teacher_4gpu_2replica_formal.sh` launcher. It does not start S1, alter checkpoints, alter training, remove result data, or issue broad process termination.

## Interface and Defaults

The entry point is `evaluation/libero/run_lingbotva_native_teacher_s2_s4_parallel_4gpu.sh` and runs with:

```bash
bash evaluation/libero/run_lingbotva_native_teacher_s2_s4_parallel_4gpu.sh
```

All operational values remain environment-overridable. Defaults are:

| Workload | GPU IDs | Replicas/GPU | Budget | Result root | Master ports | WebSocket ports |
| --- | --- | --- | --- | --- | --- | --- |
| S2 | `0,1` | `2` | `2` | `/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_dynamic_s2_2gpu2replica_formal_20260727` | `34680-34683` | `34780-34783` |
| S4 | `2,3` | `2` | `4` | `/kpfs-intern/jialongliu/projects/Flash-WAM/evaluation/results/libero_teacher_native_dynamic_s4_2gpu2replica_formal_20260727` | `34880-34883` | `34980-34983` |

The roots are intentionally distinct from the active S1 result root. The two port ranges are mutually disjoint, including across server types.

## Execution and Failure Handling

The wrapper launches S2 and S4 in the background, captures only their two wrapper PIDs, and waits for both. It returns success only if both return success. On `INT` or `TERM`, it forwards the signal only to those captured child wrappers; their existing cleanup traps manage their own workers. It never targets arbitrary GPU processes or the running S1 job.

`CHECK_ONLY=1` passes through to both child launchers. It validates lane mapping, roots, and port allocation without creating result roots or starting evaluation workers.

## Validation

A focused pytest test invokes the new wrapper with `CHECK_ONLY=1`, temporary checkpoint and output locations, and asserts:

1. Four S2 workers map to GPU/replica pairs `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)` and use `steps_2`.
2. Four S4 workers map to the analogous pairs on GPUs 2 and 3 and use `steps_4`.
3. All eight master ports and all eight WebSocket ports are unique.
4. The check leaves both temporary output roots absent.

The existing launcher test suite and shell syntax checks remain green after the wrapper is added.
