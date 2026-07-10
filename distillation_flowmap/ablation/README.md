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

## OPD mechanism calibration

Before rerunning the final task matrix, use the guarded two-task calibration:

```bash
DRY_RUN=1 bash \
  distillation_flowmap/ablation/run_opd_mechanism_calibration.sh
```

The default protocol reuses the existing seed-0 Stage1 checkpoint and runs
Stage2 only. It is fixed to `place_a2b_right` plus `open_microwave`, 20 train
and 10 held-out samples per task, seed 0, and 750 optimizer steps. Its five
variants isolate no OPD, endpoint-only, velocity-only, full last-step credit,
and a two-step gradient suffix over a four-step rollout. Full four-step
backprop remains an optional memory diagnostic because it exceeds one H100
after optimizer-state initialization. OPD action and local auxiliary losses
are disabled for this first mechanism check.

The script refuses larger task/sample settings or more than 1000 steps unless
`ALLOW_LARGE_CALIBRATION=1` is set explicitly. Do not set that flag for the
first mechanism gate. Dry-run manifests are saved under
`protocol_opd_mechanism_calibration_v1/preflight/`.

After the five `step_750` checkpoints are present, run the matching evaluator:

```bash
bash distillation_flowmap/ablation/run_opd_mechanism_calibration_eval.sh
```

It reuses one teacher cache for every variant, evaluates the same held-out and
balanced train subsets, fixes the student to four rollout steps, and uses the
eight-step teacher selected by the held-out N=4 versus N=8 preflight. The
default eval remains on `core2`, seed 0, one GPU, and one representative video
pair per variant. Its summary uses `calib_w_o_opd` as the baseline and writes
trend-only tables under `summary_calibration/`.

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
