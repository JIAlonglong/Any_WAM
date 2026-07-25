# Cosmos Worker Compatibility Design

## Goal

Restore the formal Stage-1 → Stage-2 → evaluation pipeline after the
official Cosmos raw worker failed before its first response because the clean
Cosmos source worktree could not load its CUDA extra and runtime dependencies.

## Scope

- Preserve the dirty checkout at
  `/kpfs-intern/jialongliu/projects/cosmos-predict2.5` exactly as-is.
- Create a separate clean Cosmos compatibility worktree from fixed commit
  `441b89740d91922737008a61e7f71407d47944e7`.
- Port only the six existing tracked runtime-compatibility changes from the
  dirty checkout and commit them on the isolated compatibility branch.
- Teach the Flash-WAM Stage-1 launcher to pass the same clean-repository
  package roots and NVIDIA dynamic-library roots already required by the
  Stage-2 launcher.
- Add a preflight that imports
  `cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils`
  under the exact worker Python, Python path, and library path used at runtime.
- Keep the failed output directory and all logs. A retry must use a new run tag.

## Runtime Contract

The worker process uses:

- Python:
  `/kpfs-intern/jialongliu/envs/cosmos-predict2-cu128-py310/bin/python`
- Clean Cosmos repository: the new compatibility worktree.
- Python extras:
  `<repo>/packages/cosmos-cuda:<repo>/packages/cosmos-oss`
- CUDA libraries: NVIDIA package library directories beneath the worker
  environment's `site-packages`.

The launcher must reject missing package roots, missing library directories, a
dirty Cosmos compatibility repository, or a failed full `cosmos_utils` import
before starting `torchrun`.

## Data and Output Safety

- Do not install or upgrade packages in shared environments.
- Do not modify, reset, clean, or commit the existing dirty Cosmos checkout.
- Do not delete the failed run root.
- Dry-run remains write-free.
- Fresh execution refuses an existing run root.

## Verification

1. A regression test must fail against the current Stage-1 launcher because
   worker Python/CUDA paths and the full import preflight are absent.
2. The test must pass after the minimal launcher change.
3. The isolated compatibility worktree must be clean and import
   `cosmos_utils.get_action` successfully using the exact runtime environment.
4. Stage-1 dry-run and the serial all-phase dry-run must remain write-free.
5. Focused launcher, provenance, and pipeline regression tests must pass.
6. No training process is launched by Codex.

## Follow-up

Matched teacher video/action K=1/2/4 evaluation remains a separate feature.
It resumes only after the training worker compatibility repair is verified.
