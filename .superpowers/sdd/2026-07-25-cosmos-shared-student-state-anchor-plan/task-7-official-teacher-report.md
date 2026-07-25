# Task 7: Official Cosmos Teacher matched-budget adapter

## Scope

Implemented a separate official-teacher inference contract. It does not load,
resolve, or advertise a FlowMap/Wan student transformer.

## Runtime contract

- Accepted budgets are exactly matched `video_steps == action_steps` in
  `{1, 2, 4}`.
- The policy root must contain the monolithic `.pt`, official `config.json`,
  LIBERO statistics, and T5 embeddings.
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

## Deliberately deferred integration

The top-level LIBERO matrix/client/rollout files were concurrently owned by
another task and were not edited here. The integration point is
`OfficialTeacherMatchedBudgetAdapter`; the outer matrix should construct one
worker/service per K and persist the returned effective-K metadata.

No live model evaluation, training, checkpoint mutation, or simulator run was
performed.
