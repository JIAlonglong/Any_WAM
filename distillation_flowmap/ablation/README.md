# RobotWin StepWAM Ablation

This folder contains the metadata and launch utilities for the LingBot-VA-only
StepWAM ablation on RobotWin.

Scope:
- Teacher: LingBot-VA only.
- Benchmark: RobotWin representative subset.
- Student inference budget: fixed `K=4`.
- Final evaluation: 50 episodes per task.
- Smoke gate: 1 task x 5 episodes before launching bulk training.

Files:
- `robotwin_stepwam_tasks.json`: Easy/Hard task subset and episode counts.
- `robotwin_stepwam_variants.json`: Variant matrix, seeds, and environment overrides.
- `run_smoke.sh`: protocol-controlled smoke run. By default it uses one task
  with 3 train samples and 2 held-out samples; offline rollout metrics and
  videos read the held-out manifest and shared eval-pair JSON.

The ablation runner must write one manifest per run containing the git hash,
variant, seed, task list, checkpoint paths, and exact environment overrides.
Cosmos is intentionally excluded from these component ablations.
