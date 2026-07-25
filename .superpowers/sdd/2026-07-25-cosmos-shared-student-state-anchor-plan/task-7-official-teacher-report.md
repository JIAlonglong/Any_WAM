# Task 7: Official Cosmos Teacher matched-budget adapter

## Scope

Implemented a separate official-teacher inference contract. It does not load,
resolve, or advertise a FlowMap/Wan student transformer.

## Runtime contract

- Accepted budgets are exactly matched `video_steps == action_steps` in
  `{1, 2, 4}`.
- The policy root must contain the monolithic `.pt`, official `config.json`,
  LIBERO statistics, and T5 embeddings.
- Official requests leave the student-only compatibility field unset; they do
  not advertise student checkpoint metadata.
- `config.json` must identify `model_type=cosmos-policy`,
  `architecture=diffusion-transformer`, parallel generation, and the official
  `16 x 7` action contract.
- The raw worker starts with action and future-state denoising budgets both set
  to K.
- Every matched-budget request carries both requested K values.
- Every response carries requested/effective video and action K plus
  `matched_budget_verified=true`.
- Missing metadata, mismatched K, unsupported K=5, or a worker configured at a
  different K fails closed.
- Matched video/action evaluation requires future-video generation. Closed-loop
  control consumes the raw official `16 x 7` action chunk; future predictions
  remain available for diagnostics.

## Verification

- RED: 24 expected failures before implementation.
- GREEN: `24 passed` in the new official-teacher suite.
- Contract-focused regression: `27 passed, 37 deselected`.
- Raw-worker/policy focused regression: `126 passed, 3 deselected`; two
  unrelated failures remain from the concurrent Stage-1 environment change
  (`WAN_STUDENT_BASE_MODEL_PATH`) and the host missing `robosuite`.
- Python compilation and `git diff --check` passed.
- The real official model root resolved to the monolithic
  `Cosmos-Policy-LIBERO-Predict2-2B.pt` with backend `cosmos_policy`, a
  64-character contract identity, and no student transformer path.

## Evaluation integration

- Added an official Teacher service that returns the raw official action chunk,
  effective-K proof, role/checkpoint identity, and available future-prediction
  keys without student metadata.
- The client persists requested/effective video/action K and rejects a missing
  or mismatched official runtime proof.
- The formal evaluator supports the monolithic `--cosmos-policy-path` branch
  without passing a student transformer.
- The default matrix is now two independent roles (`stage2_target` and
  `official_teacher`) × four suites × K={1,2,4}. Role-specific directories,
  JSON, and CSV prevent result mixing.
- The merger requires all 240 role/task/K cells and rejects missing,
  duplicate/foreign, wrong-role, wrong-suite, wrong-K, wrong-task, and
  wrong-episode artifacts.
- Limited video selection and dry-run no-write behavior are preserved.

Focused eval-side integration regression: `109 passed`.

The outer Stage-1 → Stage-2 → evaluation wrapper is committed by a separate
task and is integrated in a follow-up commit so this change never overwrites
concurrent lineage work.

No live model evaluation, training, checkpoint mutation, or simulator run was
performed.
