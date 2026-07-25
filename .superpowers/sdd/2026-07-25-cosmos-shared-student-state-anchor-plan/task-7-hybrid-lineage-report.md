# Task 7 Hybrid Backend and Lineage Report

## Result

The executable chain is now explicitly and truthfully identified as:

- Student backend: `wan_flowmap` (`WanTransformer3DModel`)
- Frozen Teacher backend: `cosmos_policy`

This preserves the existing architecture. It does not claim or attempt a pure
Cosmos Policy Student.

## Corrections

- Added independent validators for the Wan Student base and official Cosmos
  Policy Teacher root.
- The Wan Student base must contain
  `transformer/config.json` with
  `_class_name == "WanTransformer3DModel"`.
- The Cosmos Teacher root must contain `config.json` with
  `model_type == "cosmos-policy"` and the official policy checkpoint,
  statistics, and T5 embedding assets.
- Stage-1 now requires an explicit `WAN_STUDENT_BASE_MODEL_PATH`. The former
  `CLEAN_STUDENT_BASE_MODEL_PATH` remains only as a deprecated explicit alias;
  conflicting values fail closed.
- Removed the Python config's machine-local Student fallback.
- Persisted `student_backend=wan_flowmap` in Stage-1 and Stage-2 checkpoints.
- Stage-1 and Stage-2 lineage validation now rejects missing or different
  Student/Teacher backend identities.
- Inference restoration now returns independently named
  `wan_student_base_model_path` and `cosmos_teacher_model_path`; it no longer
  calls the Wan path a Cosmos base or requires a nonexistent root
  `config.json`.
- Updated the serial Stage-1 → Stage-2 → evaluation wrapper wording and
  resolved environment to use the explicit Wan Student input.

## Test-first evidence

RED:

- New backend test initially failed because the validator module did not exist.
- Realistic LingBotVA layout failed because the old lineage code required a
  root `config.json`.
- Checkpoint writer test failed because `student_backend` was absent.

GREEN:

- Backend, lineage, checkpoint writer: `63 passed`.
- Stage-1 launcher and serial wrapper focused suite: all tests passed.
- Progressive config plus the hybrid config contract test: all tests passed.
- Shell syntax checks passed for both modified launchers.
- `git diff --check` passed.

The broader Cosmos policy suite still contains pre-existing simulator import
failures on this host (`robosuite` unavailable); those failures are unrelated
to this change and no training or evaluation was launched.
