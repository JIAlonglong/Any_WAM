# Cosmos Aligned Full40 Resume Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one safe command that resumes the interrupted Cosmos Stage-1 checkpoint at step 1000 and then serially runs Stage-2 and the full student/teacher evaluation.

**Architecture:** A thin Bash wrapper supplies the validated experiment defaults and invokes the existing `run_cosmos_stage1_stage2_eval_8gpu.sh` once per phase. It owns no training, lineage, checkpoint, or evaluation logic; a Python contract test replaces the child launcher with a recording stub and verifies ordering, arguments, dry-run behavior, and preflight failure.

**Tech Stack:** Bash, Python `unittest`, existing Flash-WAM Cosmos launchers.

## Global Constraints

- Default command: `bash distillation_flowmap/resume_cosmos_aligned_full40_8gpu.sh`.
- Resume Stage-1 from checkpoint step 1000.
- Run Stage-1, Stage-2, and eval synchronously in that order.
- Require exactly eight visible GPUs for a live run.
- Do not delete, rename, overwrite, or duplicate existing checkpoints and training logic.
- `--dry-run` must not launch training or write production output.

---

### Task 1: Resume wrapper contract and implementation

**Files:**
- Create: `distillation_flowmap/resume_cosmos_aligned_full40_8gpu.sh`
- Create: `distillation_flowmap/tests/test_resume_cosmos_aligned_full40_8gpu.py`

**Interfaces:**
- Consumes: `run_cosmos_stage1_stage2_eval_8gpu.sh --phase stage1|stage2|eval`
- Produces: executable-compatible Bash entry point accepting `--dry-run` and environment overrides.

- [ ] **Step 1: Write the failing launcher contract test**

Create a temporary Stage-1 checkpoint containing
`target_student/transformer/config.json` and
`diffusion_pytorch_model.safetensors`. Set
`COSMOS_PIPELINE_LAUNCHER` to a temporary recording stub, set all path
overrides to temporary directories, and invoke the missing wrapper.

The test must assert that the recording stub receives exactly these phases in
order:

```python
assert [call["phase"] for call in calls] == ["stage1", "stage2", "eval"]
assert calls[0]["resume_stage"] == "stage1"
assert calls[0]["resume_step"] == "1000"
assert all(call["run_tag"] == "aligned-anchor-field-full40-20260725" for call in calls)
assert all(call["stage1_steps"] == "5000" for call in calls)
assert all(call["stage2_steps"] == "10000" for call in calls)
assert all(call["episodes"] == "50" for call in calls)
```

Add a second test that removes the resume checkpoint and asserts a non-zero
exit before the recording stub is called. Add a dry-run assertion that all
three recorded calls include `--dry-run`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_resume_cosmos_aligned_full40_8gpu.py
```

Expected: failure because
`distillation_flowmap/resume_cosmos_aligned_full40_8gpu.sh` does not exist.

- [ ] **Step 3: Implement the minimal Bash wrapper**

The wrapper must:

```bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PIPELINE_LAUNCHER="${COSMOS_PIPELINE_LAUNCHER:-$SCRIPT_DIR/run_cosmos_stage1_stage2_eval_8gpu.sh}"
RESUME_STEP="${RESUME_STEP:-1000}"
```

Resolve defaults for the existing output root, run tag, model/repository
paths, `STAGE1_STEPS=5000`, `STAGE2_STEPS=10000`,
`SAVE_INTERVAL=1000`, `EPISODES=50`, and ports 29671/29672. Parse only
`--dry-run`; reject unknown arguments. Validate eight comma-separated devices
for live mode and require the exact Stage-1 resume transformer before any
child call.

Build one common argument array, then invoke:

```bash
"$PIPELINE_LAUNCHER" --phase stage1 "${common[@]}" \
  --resume-stage stage1 --resume-step "$RESUME_STEP" "${dry_run[@]}"
"$PIPELINE_LAUNCHER" --phase stage2 "${common[@]}" "${dry_run[@]}"
"$PIPELINE_LAUNCHER" --phase eval "${common[@]}" "${dry_run[@]}"
```

- [ ] **Step 4: Run focused and related tests for GREEN**

Run:

```bash
bash -n distillation_flowmap/resume_cosmos_aligned_full40_8gpu.sh
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python \
  -m pytest -q \
  distillation_flowmap/tests/test_resume_cosmos_aligned_full40_8gpu.py \
  distillation_flowmap/tests/test_cosmos_progressive_runner.py
```

Expected: all tests pass.

- [ ] **Step 5: Verify the real wrapper in dry-run mode**

Run the wrapper with `--dry-run` and the existing experiment paths.

Expected:

- three resolved phases in Stage-1, Stage-2, eval order;
- Stage-1 contains `--resume-stage stage1 --resume-step 1000`;
- no `train.py`, rollout, or torchrun process is created;
- no Stage-2 or evaluation output directory is created.

- [ ] **Step 6: Commit the implementation**

```bash
git add \
  distillation_flowmap/resume_cosmos_aligned_full40_8gpu.sh \
  distillation_flowmap/tests/test_resume_cosmos_aligned_full40_8gpu.py
git commit -m "feat: add Cosmos full40 resume wrapper"
```
