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
  `matched_budget_verified=true` and the observed joint denoiser NFE.
- Effective K is derived by instrumenting the denoiser actually called by the
  official sampler; it is not a copy of the configured step count.
- Raw action inference, same-prior continuation, and joint continuation all
  require the selected Cosmos repository to be clean at audited commit
  `1eb8457072b4a1adfe1f83c3076e4aa5452cbab2`. Each response records that commit
  and the SHA-256 of the imported `cosmos_utils.py`.
- The formal outer pipeline repeats the clean/commit check before launch.
- The full teacher artifact lock (including the monolithic weights) is hashed
  exactly once by the joint matrix launcher. It exports the verified 64-hex
  identity; shard/preflight processes receive `--teacher-contract-identity`
  and do not rehash multi-GB weights. The standalone official-teacher launcher
  performs the same one-time verification when no outer identity is supplied.
- Missing metadata, mismatched K/NFE, unaudited or dirty Cosmos source,
  unsupported K=5, or a worker configured at a different K fails closed.
- Matched video/action evaluation requires future-video generation. Closed-loop
  control consumes the raw official `16 x 7` action chunk; future predictions
  remain available for diagnostics.

## Final review fixes and verification

- Initial implementation RED/GREEN is retained in the commit history.
- Final-review RED reproduced the production lock rejection, unaudited
  `mode=actions`, missing requested/effective K and observed NFE, missing source
  identity, permissive string success values, and repeated per-shard lock use.
- Final GREEN regression: `196 passed in 25.69s`.
- `bash -n` passed for the serial pipeline, joint matrix launcher, and formal
  role launcher.
- Python compilation passed for the official adapter, policy adapter, raw
  worker, client, merger, and rollout entrypoint.
- `git diff --check` passed.
- The real official model root resolved to the monolithic
  `Cosmos-Policy-LIBERO-Predict2-2B.pt` with backend `cosmos_policy`, a
  64-character contract identity, and no student transformer path.

## Evaluation integration

- Added an official Teacher service that returns the raw official action chunk,
  effective-K proof, role/checkpoint identity, and available future-prediction
  keys without student metadata.
- The client persists requested/effective video/action K and rejects a missing
  or mismatched official runtime proof, unaudited commit, or malformed source
  digest.
- The formal evaluator supports the monolithic `--cosmos-policy-path` branch
  without passing a student transformer.
- The default matrix is now two independent roles (`stage2_target` and
  `official_teacher`) × four suites × K={1,2,4}. Role-specific directories,
  JSON, and CSV prevent result mixing.
- The merger requires all 240 role/task/K cells and rejects missing,
  duplicate/foreign, wrong-role, wrong-suite, wrong-K, wrong-task, and
  wrong-episode artifacts. Before publishing JSON/CSV, every Teacher episode
  must carry exact requested/effective K, exact observed NFE, a strict boolean
  success value, the audited repo commit, and one consistent source digest.
- Per-suite, per-role, and complete matrix summaries preserve the Teacher
  repository commit and source digest.
- Limited video selection and dry-run no-write behavior are preserved.

No live model evaluation, training, checkpoint mutation, or simulator run was
performed.
