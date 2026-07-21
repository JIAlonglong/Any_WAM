# Cosmos independent four-way Stage-2 launcher

## Goal

Provide four separately launchable, serializable eight-GPU Cosmos Stage-2
experiments: `s1`, `s2`, `s4`, and `universal`.  Each fresh run starts from
the same verified public Cosmos Stage-1 artifact, rather than from another
new Stage-2 run, so their results are directly comparable.

## Fixed experiment definitions

| mode | train steps | endpoint rollout pairs | DanceOPD rollout choices | Dance velocity weight | port |
| --- | ---: | --- | --- | ---: | ---: |
| `s1` | 3000 | `4,1` | `1` | 0.0 | 29663 |
| `s2` | 3000 | `4,2` | `2` | 1.0 | 29662 |
| `s4` | 5000 | `8,4` | `4` | 1.0 | 29661 |
| `universal` | 5000 | `8,1;8,2;8,4` | `2,4` | 1.0 | 29664 |

`universal` deliberately retains the established LingbotVA definition: its
endpoint pairs are sampled from `8->1`, `8->2`, and `8->4`, while DanceOPD
independently samples its pre-existing `2` or `4` rollout choice.  It does
not introduce pair-coupled DanceOPD sampling and it never returns to a
16-step DanceOPD rollout.

## Launcher behavior

The existing Cosmos launcher becomes a four-mode independent launcher.  Fresh
runs all use:

`/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/raw_stage1_5000`

with `RESUME_ONLINE_FROM_TARGET=1`, reset step numbering, and no optimizer
state.  Outputs are isolated below:

`/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_independent_dance_4way_8gpu_20260721/s1`,
with corresponding `s2`, `s4`, and `universal` subdirectories.

The launcher continues to require exactly eight matching GPU ordinals, reject
existing checkpoint directories for fresh runs, save every 1000 steps, and
support explicit self-resume only through `--resume-step`.

## Config and runtime behavior

The Cosmos config gains the `universal` stage and parses the existing
comma-separated DanceOPD rollout choice convention.  It stores both the
choice tuple and the historical scalar first choice.  The DanceOPD trainer
samples a single synchronized choice across distributed ranks when more than
one choice is configured; fixed S1/S2/S4 behavior remains unchanged.

## Validation

Tests will cover all four mode definitions, direct Stage-1 initialization,
universal rollout choices, output and GPU safety, and self-resume behavior.
Validation will run shell syntax checking, focused pytest tests, all four
dry-runs, and confirm that dry-runs create no output directories or GPU jobs.
