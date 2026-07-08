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
- `summarize_robotwin_ablation.py`: report aggregator for protocol runs. It
  writes per-seed rows, mean/std summaries, raw delta vs `w_o_opd`, per-task
  rows, video/contact-sheet asset lists, and a trend-only markdown report.

The ablation runner must write one manifest per run containing the git hash,
variant, seed, task list, checkpoint paths, and exact environment overrides.
Cosmos is intentionally excluded from these component ablations.

Example:

```bash
python distillation_flowmap/ablation/summarize_robotwin_ablation.py \
  --root distillation_flowmap/output_robotwin_stepwam_ablation/protocol_core4 \
  --out distillation_flowmap/output_robotwin_stepwam_ablation/protocol_core4/report \
  --baseline-variant w_o_opd
```

Main outputs:
- `mini_ablation_per_seed.csv/jsonl`
- `mini_ablation_summary.csv/md`
- `mini_ablation_per_task.csv`
- `mini_ablation_video_assets.jsonl`
- `mini_ablation_report.md`
