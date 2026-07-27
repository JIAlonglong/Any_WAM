# Cosmos Independent Four-Way Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `s1`, `s2`, `s4`, and `universal` independently trainable Cosmos Stage-2 runs from one verified Stage-1 checkpoint.

**Architecture:** The config owns the endpoint/DanceOPD semantics for the four modes; the launcher owns direct Stage-1 initialization, isolated outputs, GPU validation, and resumability.  A minimal DanceOPD choice selector reuses the existing LingbotVA `2,4` convention without changing fixed S1/S2/S4 behavior.

**Tech Stack:** Bash, Python, PyTorch distributed training, pytest.

## Global Constraints

- Fresh `s1`, `s2`, `s4`, and `universal` runs all initialize from `/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/raw_stage1_5000` with `RESUME_ONLINE_FROM_TARGET=1`.
- Keep DanceOPD semantics exactly: S1=`1`/velocity `0`; S2=`2`/velocity `1`; S4=`4`/velocity `1`; universal=`2,4`/velocity `1`.
- Universal endpoint pairs are exactly `8,1;8,2;8,4`; its DanceOPD choice is independent, not pair-coupled.
- Steps are S1=3000, S2=3000, S4=5000, universal=5000; save interval is 1000.
- Fresh outputs are below `output_libero_cosmos_independent_dance_4way_8gpu_20260721/s1`, with matching `s2`, `s4`, and `universal` directories, and must never be overwritten.
- Every launch requires exactly eight matching visible GPU ordinals and does not start a GPU job during `--dry-run`.

---

### Task 1: Add universal Cosmos config and original DanceOPD choice semantics

**Files:**
- Modify: `distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py:22-160`
- Modify: `distillation_flowmap/flowmap_step.py:5013-5018`
- Modify: `distillation_flowmap/tests/test_cosmos_progressive_config.py:1-100`

**Interfaces:**
- Consumes: `COSMOS_PROGRESSIVE_STAGE` and `OPD_DANCEOPD_ROLLOUT_STEPS` environment variables.
- Produces: `cfg.opd_rollout_step_pairs`, `cfg.opd_danceopd_rollout_step_choices`, and `cfg.opd_danceopd_rollout_steps`.

- [ ] **Step 1: Write the failing universal config test**

```python
def test_progressive_universal_retains_original_lingbotva_definition(monkeypatch):
    monkeypatch.setenv("COSMOS_PROGRESSIVE_STAGE", "universal")
    monkeypatch.delenv("OPD_DANCEOPD_ROLLOUT_STEPS", raising=False)
    module = importlib.reload(importlib.import_module(
        "distillation_flowmap.config_libero_cosmos_policy_stage2_progressive"
    ))
    assert module.cfg.opd_rollout_step_pairs == [[8, 1], [8, 2], [8, 4]]
    assert module.cfg.opd_danceopd_rollout_step_choices == (2, 4)
    assert module.cfg.opd_danceopd_rollout_steps == 2
    assert module.cfg.opd_danceopd_velocity_weight == 1.0
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q distillation_flowmap/tests/test_cosmos_progressive_config.py::test_progressive_universal_retains_original_lingbotva_definition`

Expected: FAIL because `universal` is not a recognized `COSMOS_PROGRESSIVE_STAGE`.

- [ ] **Step 3: Add the strict choice parser and universal stage spec**

```python
def _parse_positive_step_choices(text, *, env_name):
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(f"{env_name} must be a comma-separated list of positive integers")
    try:
        choices = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"{env_name} must be a comma-separated list of positive integers"
        ) from exc
    if any(choice <= 0 for choice in choices) or len(set(choices)) != len(choices):
        raise ValueError(f"{env_name} must contain unique positive integers")
    return choices
```

Represent each stage with a `rollout_step_pairs` list; keep the existing S1/S2/S4 pairs and add:

```python
"universal": {
    "max_steps": 5000,
    "rollout_step_pairs": [[8, 1], [8, 2], [8, 4]],
    "focus_prob": 0.85,
    "velocity_weight": 1.0,
    "grad_mode": "last_step",
    "grad_steps": 1,
    "danceopd_rollout_steps": "2,4",
},
```

Parse `OPD_DANCEOPD_ROLLOUT_STEPS` through `_parse_positive_step_choices`, set the scalar compatibility attribute to its first value, and populate `cfg.opd_rollout_step_pairs` from the stage list.

- [ ] **Step 4: Make DanceOPD sample configured choices synchronously**

Replace the scalar selection in `_danceopd_aux_transition_step` with:

```python
rollout_step_choices = tuple(
    int(value)
    for value in getattr(
        self.config,
        "opd_danceopd_rollout_step_choices",
        (getattr(self.config, "opd_danceopd_rollout_steps", 4),),
    )
)
if not rollout_step_choices or any(value <= 0 for value in rollout_step_choices):
    raise ValueError("opd_danceopd_rollout_step_choices must be positive")
if len(rollout_step_choices) == 1:
    rollout_steps = rollout_step_choices[0]
else:
    choice_index = torch.randint(len(rollout_step_choices), (1,), device=self.device)
    if dist.is_initialized():
        dist.broadcast(choice_index, src=0)
    rollout_steps = rollout_step_choices[choice_index.item()]
```

- [ ] **Step 5: Run config tests and commit**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q distillation_flowmap/tests/test_cosmos_progressive_config.py`

Expected: all Cosmos progressive config tests pass, including universal.

```bash
git add distillation_flowmap/config_libero_cosmos_policy_stage2_progressive.py \
        distillation_flowmap/flowmap_step.py \
        distillation_flowmap/tests/test_cosmos_progressive_config.py
git commit -m "feat: add independent Cosmos universal stage"
```

### Task 2: Convert the launcher to four independent fresh runs

**Files:**
- Modify: `distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh:64-192`
- Modify: `distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py:58-178`

**Interfaces:**
- Consumes: one mode argument (`s1`, `s2`, `s4`, or `universal`) and existing optional `--dry-run`, `--master-port`, and `--resume-step` flags.
- Produces: one isolated output path and a `torchrun --nproc_per_node=8` command; all fresh modes use the verified Stage-1 root.

- [ ] **Step 1: Rewrite the parameterized launcher contract test first**

```python
@pytest.mark.parametrize(
    ("stage", "steps", "port"),
    [
        ("s1", "3000", "29663"),
        ("s2", "3000", "29662"),
        ("s4", "5000", "29661"),
        ("universal", "5000", "29664"),
    ],
)
def test_dry_run_prints_independent_stage_contract_without_creating_output(
    tmp_path, stage, steps, port
):
    env, output = _env(tmp_path)
    result = _run(stage, "--dry-run", env=env)
    assert result.returncode == 0, result.stderr
    assert f"MAX_TRAIN_STEPS={steps}" in result.stdout
    assert "RESUME_FROM_PATH=" + env["COSMOS_STAGE1_ROOT"] in result.stdout
    assert "RESUME_ONLINE_FROM_TARGET=1" in result.stdout
    assert f"--master_port={port}" in result.stdout
    assert not (output / stage).exists()
```

Delete the old tests that create S4/S2 predecessors or require an S2 predecessor, because independent fresh runs deliberately do not use them.

- [ ] **Step 2: Run the rewritten tests and verify they fail**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py`

Expected: S1, S2, and universal contract cases fail against the progressive-only launcher.

- [ ] **Step 3: Implement the independent stage table**

Set the default output root to:

```bash
OUTPUT_ROOT="${OUTPUT_ROOT:-/kpfs-intern/jialongliu/projects/Flash-WAM/distillation_flowmap/output_libero_cosmos_independent_dance_4way_8gpu_20260721}"
```

Use this exact stage table:

```bash
case "$stage" in
    s1) max_steps=3000; default_port=29663 ;;
    s2) max_steps=3000; default_port=29662 ;;
    s4) max_steps=5000; default_port=29661 ;;
    universal) max_steps=5000; default_port=29664 ;;
    *) die "stage must be one of: s1, s2, s4, universal" ;;
esac
```

For every fresh mode set `resume_from_path="$STAGE1_ROOT"`,
`resume_online_from_target=1`, `reset_resume_step=1`, and
`resume_optimizer_state=0`.  Retain strict Stage-1 online/target transformer
checks, existing checkpoint refusal, and self-resume behavior.

- [ ] **Step 4: Run launcher tests and commit**

Run: `bash -n distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh && PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py`

Expected: shell syntax succeeds and all launcher tests pass.

```bash
git add distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh \
        distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py
git commit -m "feat: launch four independent Cosmos stages"
```

### Task 3: Verify the real remote paths without GPUs

**Files:**
- Modify: none
- Test: `distillation_flowmap/tests/test_cosmos_progressive_config.py`
- Test: `distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py`

**Interfaces:**
- Consumes: verified Stage-1 root, official Cosmos worker environment, dataset, and four fresh output paths.
- Produces: four resolved 8-GPU command lines without creating output directories.

- [ ] **Step 1: Run the complete focused suite**

Run: `PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q distillation_flowmap/tests/test_cosmos_progressive_config.py distillation_flowmap/tests/test_run_cosmos_progressive_stage2_8gpu.py`

Expected: all focused tests pass.

- [ ] **Step 2: Run four real-path dry-runs**

Run each of:

```bash
bash distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh s1 --dry-run
bash distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh s2 --dry-run
bash distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh s4 --dry-run
bash distillation_flowmap/run_cosmos_progressive_stage2_8gpu.sh universal --dry-run
```

Expected: each prints its independent Stage-1 resume path, exact step budget,
mode-specific port, save interval 1000, and an eight-process `torchrun`
command; none creates its output directory or allocates GPUs.

- [ ] **Step 3: Record validation and leave the branch intact**

Run: `git status --short && git log --oneline -3`

Expected: clean worktree after commits; do not merge, push, or remove the
worktree without an explicit user request.
