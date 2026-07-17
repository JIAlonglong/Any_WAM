# RobotWin small-protocol offline teacher–student evaluation

This is an offline latent/action error comparison for the RobotWin ablation
weights. It is not a RobotWin simulator success-rate evaluation, and it must
not use the full RobotWin evaluation split.

## Fixed inputs

The guarded launcher accepts no arbitrary checkpoint, manifest, or pair-file
override. It selects only:

| Name | Checkpoint | Held-out protocol |
| --- | --- | --- |
| `danceopd` | final DANCE/OPD Stage-2 `step_5000` | `core2` archive subset diagnostic or its own final12 protocol |
| `full_stepwam` | final full StepWAM Stage-2 `step_5000` | `core2` archive subset diagnostic or its own final12 protocol |

`core2` is the 20-record fixed held-out diagnostic and is deliberately marked
as an archive subset in the plan; it is not represented as either checkpoint's
training split. `final12` is the fixed 120-record representative held-out
protocol. Both use only `lerobot_robotwin_eef_aug_500`, its held-out manifest,
and its recorded pair file.

A `final12` execution is additionally locked behind a completed `core2` gate
for every selected `(checkpoint, source, K)` tuple. The launcher verifies the
Core2 `plan.json` directly: it must be an executed Core2 run, contain the
matching tuples, preserve source provenance, and contain all fixed pair/action/
video teacher/student metric keys. A summary file alone is not accepted.

Each entry is an independent equal-NFE invocation:

- student source: `target` (primary) or `online` (supplementary);
- K: `1`, `2`, or `4`, with `--student-steps K --teacher-steps K`;
- source selection is set before trainer construction with
  `RESUME_ONLINE_FROM_TARGET=1` for target or `0` for online.

The command uses `rollout_eval_video_stage2.py` but intentionally omits
`--video-dir`; it therefore emits video latent error metrics without decoding
video. It always enables `--emit-teacher-action-gt` and never permits a
teacher cache, because legacy caches cannot yield teacher-action-to-GT values.

## Plan first

Choose a fresh output root outside code, datasets, checkpoint roots, and
existing experiment directories. The default invocation is side-effect free:
it checks the fixed input paths and prints a JSON plan without creating the
output directory or loading a model.

```bash
cd /root/nas/junjie/jj/Any_WAM_robotwin_offline_teacher_student

distillation_flowmap/run_robotwin_teacher_student_offline.sh \
  --split core2 \
  --checkpoint danceopd \
  --source target \
  --k 1 \
  --output-root /root/nas/junjie/jj/robotwin_offline_teacher_student_core2_gate \
  > /root/nas/junjie/jj/robotwin_offline_teacher_student_core2_gate.plan.json
```

Review the plan before execution. It records the selected checkpoint, source
subdirectory, input manifest/pair SHA-256 values, checkpoint inventory digest,
equal-NFE K, expected raw result path, and exact evaluator command. A plan
must contain `--emit-teacher-action-gt`, must not contain
`--teacher-cache-path` or `--video-dir`, and must state
`video_decode_enabled: false`.

## Explicit execution

Only after GPU availability and the printed plan have been checked, run the
same command with `--run`. `--run` creates the previously non-existent output
root, writes `plan.json`, runs the selected commands sequentially, and then
writes `summary/summary.csv` and `summary/summary.md`.

```bash
distillation_flowmap/run_robotwin_teacher_student_offline.sh \
  --run \
  --split core2 \
  --checkpoint danceopd \
  --source target \
  --k 1 \
  --cuda-visible-devices 6 \
  --output-root /root/nas/junjie/jj/robotwin_offline_teacher_student_core2_gate
```

Run the core2 target K=1 gate first, inspect source provenance plus all action
and video keys, then add online and K=2/4. Do not schedule `final12` until the
corresponding Core2 gate has completed and its raw results have passed the
summary check. For example, a final12 DANCE target K=1 run must point at the
matching completed Core2 plan:

```bash
distillation_flowmap/run_robotwin_teacher_student_offline.sh \
  --run \
  --split final12 \
  --checkpoint danceopd \
  --source target \
  --k 1 \
  --core2-gate /root/nas/junjie/jj/robotwin_offline_teacher_student_core2_gate/plan.json \
  --cuda-visible-devices 6 \
  --output-root /root/nas/junjie/jj/robotwin_offline_teacher_student_final12_dance_target_k1
```

Plan-only `final12` previews may omit `--core2-gate`; `--run` may not. The
final12 plan records the Core2 plan path and SHA-256 when one is supplied.

## Interpreting the summary

Every pair/K reports the following student-to-GT and teacher-to-GT metric
pairs:

- action: `xr` MSE/L1 and velocity MSE/L1;
- video: endpoint `x` MSE/L1 and velocity MSE.

The video evaluator's current contract does not emit video velocity L1, so the
summary requires only the real shared MSE key for that component. It refuses
to summarize a raw result that lacks any required action/video teacher/student
pair, lacks or adds a fixed protocol pair, has a pair-file SHA mismatch, whose
source provenance differs from the plan, or whose path is outside the
launcher's isolated output root.

`student_minus_teacher = student - teacher`. A negative value means the
student has lower error than the teacher on that same record/pair/K; it does
not establish RobotWin closed-loop success rate.
