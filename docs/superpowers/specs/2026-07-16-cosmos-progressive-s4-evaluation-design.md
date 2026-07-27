# Cosmos Progressive S4 Evaluation Design

## Goal

Evaluate only the published `progressive_stage2_full/s4/step_5000/online_student/transformer` checkpoint with (1) paper-faithful offline mechanism diagnostics and (2) real LIBERO closed-loop success evaluation.

## Scope and invariants

- Work from `origin/research/cosmos-stage2-progressive` in an isolated worktree; do not modify the user's dirty `main` checkout.
- The public checkpoint is a 16-channel Cosmos-latent FlowMap student. It must never be sent to the 48-channel generic WanVA server.
- Cosmos Policy is required at every live control cycle to create the 16-channel generated-latent and action anchor. LingBot/Wan text components, if loaded, are only student conditioning dependencies and must not be described as the teacher.
- Store all downloaded weights, protocol artifacts, caches, logs, videos, and results under a new S4-specific output root. Do not overwrite pre-existing checkpoint or experiment directories.
- Use the fixed K=4 deployment grid `1000 -> 750 -> 500 -> 250 -> 0`, the official Cosmos teacher with N=8 continuation, and video-latent metrics only for the paper Table-3 diagnostics.
- The first smoke uses one GPU and no mutation of source data: a 16-channel student load plus one cached, teacher-free K=4 offline forward. The full runtime is only launched after this gate passes.
- Formal closed-loop evaluation uses `libero_10`, shared initial-state episode seeds, 50 episodes per task (500 total), per-task and macro success rates, and a hierarchical bootstrap confidence interval. The initial gate is 5 episodes per task.
- Four A800s are partitioned into two replicas, each with one student GPU and one Cosmos raw-worker GPU. They are not four independent single-GPU replicas.

## Offline paper protocol

The evaluator freezes a held-out, task-stratified manifest, a fixed prior/noise seed for each record, and fixed teacher anchors. It writes per-record JSONL as well as per-task and macro JSON aggregates. No diagnostic becomes a loss.

For each sample, preserve the joint video/action S4 state at `z_750`, `z_500`, and `z_250`. At those states compute:

1. `G_anchor`: the video MSE between the same-prior teacher endpoint `y0` and the endpoint reached by an N=8 Cosmos teacher continuation starting from `z_r`.
2. `G_comp`: at `(r,s)=(750,500)` and `(500,250)`, the video MSE between the direct student map `Psi(z_r; r->0)` and the composed student map `Psi(Psi(z_r; r->s); s->0)`.
3. `video_ep`: the video MSE from the explicit finite student map `Psi(z_r; r->0)` to `y0`; it is not replaced by an equal-time velocity reconstruction.
4. `field_match`: the video MSE between equal-time student and Cosmos teacher fields at `z_r`.

`G_comp` is aggregated over its two valid deployed-grid pairs. The evaluator also keeps the pre-existing terminal endpoint and action metrics under separate, clearly named sanity keys. Optional decoded comparison artifacts are produced only by a verified Cosmos decoder; no Wan VAE may decode Cosmos latents.

## Closed-loop protocol

A dedicated progressive S4 runner receives the same raw LIBERO observation fields used by the Cosmos adapter: agent-view RGB, wrist RGB, 9-D Cosmos proprioception, and task text. It calls the official worker for the generated latent/action anchor, creates the matching text embeddings and FlowMap joint input, runs four student Euler steps, decodes the predicted action into LIBERO's 7-D action space, and executes the configured action chunk. It writes one immutable episode record per `(task, seed)` including success, decision timings, model/runtime identifiers, and action trace.

The orchestration script has three explicit modes:

- `smoke`: one-GPU, cache-backed student loading/forward check; it does not claim closed-loop behavior.
- `gate`: two-GPU live rollout on five shared seeds per task.
- `formal`: two two-GPU shards over the disjoint task sets `0..4` and `5..9`, each evaluating 50 shared seeds per task; a merge step validates exactly one result per requested `(task, seed)` before calculating summary statistics.

## Acceptance criteria

- Unit tests prove grid selection, the four metric definitions, task-macro aggregation, result-merge duplicate/missing detection, and launcher GPU partitioning.
- The public S4 checkpoint passes an actual one-GPU cache-backed 16-channel load-and-forward smoke using `--skip-same-state-velocity`.
- The launcher prints a dry-run contract with paths, devices, task shards, and commands without creating output directories.
- The two-GPU gate produces valid episode records before a formal 500-episode run is enabled.
- All source changes and tests live in the isolated branch; unrelated user changes remain untouched.
