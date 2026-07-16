# Cosmos Universe and Specialized Mixed-Step Policies — Design

## Goal

Train three new, independently deployable policies on eight H100 GPUs, while
retaining the existing S4 policy as a frozen reference:

1. **Universe**: one universal policy that supports 1-, 2-, and 4-step
   sampling, with a low-NFE preference.
2. **S2**: a policy specialized for 2-step inference.
3. **S1**: a policy specialized for 1-step inference.
4. **Existing S4**: the completed 4-step reference; it is not retrained.

The student is the LingBotVA/Wan transformer. Cosmos Policy remains a frozen
latent/action teacher. “Universe” denotes the sampling policy, not a separate
Cosmos checkpoint or world-model architecture.

## Common initialization and checkpoint semantics

All three new runs start independently from the same read-only parent:

```
/root/nas/junjie/jj/Any_WAM/distillation_flowmap/
output_libero_cosmos_policy_stage1_cosmos_latent_cdiff_8gpu_20260706_cosmos_latent_s1s2_8gpu/
checkpoints/step_5000/online_student/transformer
```

This is the exact latent Stage-1 parent used by the existing S4 run. It has
the compatible 16-channel latent interface. The raw action Stage-1 checkpoint
must not be used: it has a 48-channel interface and an action-only objective.

Each new policy resets optimizer state and global step to zero. No run resumes
from S4, S2, Universe, or another new policy; this is not a staged curriculum.

The new policies keep the existing `cosmos_latent_full` target-free checkpoint
semantics: `online_student/transformer` is the deployable checkpoint. No
`target_student` EMA is created, updated, or advertised.

## Fixed mixed-step policy matrix

Sampling is fixed for the full training run; it is not annealed over time.

| Policy | S1 probability | S2 probability | S4 probability | Deployment intent |
| --- | ---: | ---: | ---: | --- |
| Universe | 0.50 | 0.30 | 0.20 | One policy usable at 1, 2, or 4 steps; prioritize low NFE |
| S2 | 0.20 | 0.60 | 0.20 | Best 2-step policy with S1/S4 regularization |
| S1 | 0.70 | 0.20 | 0.10 | Best 1-step policy while retaining longer-path support |
| Existing S4 | 0.00 | 0.00 | 1.00 | Existing specialized 4-step reference |

For each sampled mode, the trainer uses the matching student rollout length
and the corresponding Cosmos endpoint/velocity supervision. The full S4 loss
family remains enabled: joint video/action endpoint, teacher action anchor,
DanceOPD same-state velocity, and OPD auxiliary update.

## Eight-GPU execution model

Each new policy owns all eight H100 GPUs and runs serially:

1. Universe
2. S2
3. S1

The current progressive runner hard-codes two ranks and devices 6,7. It must
be parameterized so that `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, master
port, output root, and policy sampling distribution are explicit launch
arguments. Eight FSDP ranks will use the existing rank-local Cosmos teacher
worker mechanism; a preflight is mandatory because eight concurrent teacher
workers have not been validated.

Assumption for this design: performance is preferred over strict compute
matching to the legacy two-GPU S4 run. Each new run receives 5,000 optimizer
steps. The existing S4 is therefore a legacy reference, not a compute-matched
ablation point.

## Safety and validation gates

Before any full run:

1. Unit-test deterministic mixed-step sampling and exact probability routing.
2. Run a 5–10 step, eight-rank Universe preflight from the common Stage-1
   parent.
3. Confirm no OOM, deadlock, teacher-worker port collision, divergent rank
   loss, or source-checkpoint writes.
4. Confirm that each sampled mode reaches the correct student-step branch and
   that only `online_student` is saved.

For every full policy run:

- save every 250 steps;
- emit the fixed selection proxy per checkpoint for S1, S2, and S4;
- run a final held-out offline proxy at all three inference budgets;
- retain configuration, sampled-step histogram, source parent, git hash, and
  checkpoint-selection provenance;
- do not claim RobotWin/LIBERO closed-loop success unless a separate valid
  rollout evaluation completes.

## Scope boundaries

- Do not modify, delete, or overwrite existing S4, raw Cosmos, Stage-1, or
  ModelScope checkpoint artifacts.
- Do not train a generic Cosmos Universe/world-model checkpoint; current code
  trains a LingBotVA/Wan student under a Cosmos Policy teacher.
- Do not silently substitute a copied online checkpoint for an EMA target.
