# Cosmos Progressive S4 Evaluation: Reproducible Smoke, Gate, and Formal Run

Date: 2026-07-16
Status: orchestration only. This document does not claim a completed live Cosmos or closed-loop evaluation.

## Fixed artifact and no-overwrite policy

Use only the published 9.515-GiB ModelScope transformer subtree:

~~~text
progressive_stage2_full/s4/step_5000/online_student/transformer
~~~

The worktree does not pin a ModelScope model ID or revision. Obtain those two
values from the approved release record; do not substitute a similarly named
checkpoint or download a broad snapshot. Download only the subtree above into a
new cache root, then set:

~~~bash
export MODELSCOPE_S4_MODEL="<approved published ModelScope model ID>"
export MODELSCOPE_S4_REVISION="<approved published revision>"
export MODELSCOPE_S4_CACHE="<new empty local cache directory>"

# Use the approved ModelScope client/API with an allow-pattern restricted to:
# progressive_stage2_full/s4/step_5000/online_student/transformer/**
# The expected downloaded transformer subtree is exactly 9.515 GiB.
export S4_CKPT_ROOT="$MODELSCOPE_S4_CACHE/progressive_stage2_full/s4/step_5000/online_student/transformer"
~~~

Before loading, prove that this is the public 16-channel S4 artifact:

~~~bash
python - "$S4_CKPT_ROOT/config.json" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if int(config["in_channels"]) != 16 or int(config["out_channels"]) != 16:
    raise SystemExit(f"expected 16 input/output channels, got {config.get('in_channels')}/{config.get('out_channels')}")
print("S4_TRANSFORMER_CHANNELS=16,16")
PY
~~~

Every S4 launcher invocation requires a new, non-existing EVAL_ROOT. The
launcher rejects an existing path rather than mixing records from different
checkpoints or seed plans.

## Protocol and cache order

The teacher cache is an input to the smoke; it is never synthesized by the
launcher. Build protocol artifacts and the cache only in an official compatible
Cosmos worker environment, never on the current unsupported host.

~~~bash
export DATASET="<real latent LeRobot root containing empty_emb.pt>"
export PROTOCOL_ROOT="<new protocol directory>"
export COSMOS_POLICY_PATH="<official Cosmos Policy model path>"

python distillation_flowmap/prepare_cosmos_progressive_protocol.py \
  --dataset-path "$DATASET" \
  --output-dir "$PROTOCOL_ROOT" \
  --selection-per-task 3 \
  --test-per-task 5 \
  --protocol-seed 20260714 \
  --pairs 1000,0

# Official Cosmos environment only: this command contacts the teacher.
python distillation_flowmap/build_cosmos_progressive_teacher_cache.py \
  --dataset-path "$DATASET" \
  --manifest "$PROTOCOL_ROOT/test_manifest.json" \
  --pairs "$PROTOCOL_ROOT/eval_pairs.json" \
  --cache-dir "$PROTOCOL_ROOT/teacher_cache/test" \
  --teacher-steps 8 \
  --teacher-model-path "$COSMOS_POLICY_PATH"
~~~

The required cache path for a record index I and pair ID P is:

~~~text
$PROTOCOL_ROOT/teacher_cache/test/sample_IIIIII__P.pt
~~~

It must retain the exact test-manifest digest, exact same-prior pair with
t=1000 and r=0, teacher_steps=8, the four matching 16-channel video tensors
video_x0, video_noise, teacher_x_r, and teacher_v_r, plus teacher_action_x0.

## Cache-backed K=4 smoke

The smoke loads the real public S4 student and executes exactly one K=4
cache-backed forward. It does not resolve or construct a Cosmos teacher. Its
summary is explicitly non-paper: is_paper_metric=false. It is not a
closed-loop success claim.

~~~bash
export S4_SMOKE_DATASET_PATH="$DATASET"
export S4_SMOKE_MANIFEST="$PROTOCOL_ROOT/test_manifest.json"
export S4_SMOKE_PAIRS="$PROTOCOL_ROOT/eval_pairs.json"
export S4_SMOKE_CACHE_DIR="$PROTOCOL_ROOT/teacher_cache/test"
export EVAL_ROOT="<new empty smoke result root>"

bash evaluation/libero/run_cosmos_progressive_s4_eval.sh smoke
~~~

The launcher calls the cache-only evaluator with cache-only-smoke,
skip-same-state-velocity, student-steps=4, teacher-steps=8, and limit=1. It
checks that summary.json reports six 16-channel video shapes and a non-paper
K=4 result.

## Safe dry-run and live preflight

Dry-run emits KEY=value records and shell-quoted COMMAND lines, but creates no
result directory and invokes no child process. The dedicated `dry-run` mode
requires an existing S4_PROMPT_TABLE; it only plans and must not materialize a
table from S4_DATASET_PATH.

~~~bash
export EVAL_ROOT="<a path that must remain absent>"
S4_DRY_RUN=1 bash evaluation/libero/run_cosmos_progressive_s4_eval.sh formal
# Equivalently:
bash evaluation/libero/run_cosmos_progressive_s4_eval.sh dry-run
~~~

Live gate/formal commands call:

~~~text
python -m evaluation.libero.rollout_cosmos_progressive_s4
~~~

rather than a file path, so package imports work consistently. Before any
worker construction, the live module preflights CUDA driver and external
official Cosmos cu128 runtime compatibility. On the currently unsupported
host this fails safely; do not bypass it or launch a worker.

## Gate and formal allocation

Both live phases require S4_EMPTY_EMBEDDING. A task-keyed S4_PROMPT_TABLE may
be supplied directly. If it is absent, live gate/formal materializes
EVAL_ROOT/prompt_embeddings.pt from the first occurrence of each exact task in
S4_DATASET_PATH/latents/**/*.pth. It reads only existing text plus text_emb
[512,4096], verifies the exact ten prompts in meta/tasks.jsonl, unsqueezes each
embedding to [1,512,4096], and never invokes a text encoder. This materialization
is intentionally never performed in dry-run.

~~~bash
export S4_DATASET_PATH="$DATASET"
# Optional: omit this variable to materialize EVAL_ROOT/prompt_embeddings.pt.
export S4_PROMPT_TABLE="<task-keyed training prompt table>"
export S4_EMPTY_EMBEDDING="$DATASET/empty_emb.pt"

# Gate needs the compatible 0/1 student/worker pair; formal needs all four GPUs.
export EVAL_ROOT="<new empty gate root>"
bash evaluation/libero/run_cosmos_progressive_s4_eval.sh gate

export EVAL_ROOT="<new empty formal root>"
bash evaluation/libero/run_cosmos_progressive_s4_eval.sh formal
~~~

Gate is deliberately a single-pair acceptance phase:

| Phase | Student GPU | Cosmos worker GPU | Tasks |
| --- | ---: | ---: | --- |
| Gate | 0 | 1 | [0, 10) |

Formal's shard plan is fixed:

| Formal shard | Student GPU | Cosmos worker GPU | Tasks |
| --- | ---: | ---: | --- |
| 0 | 0 | 1 | [0, 5) |
| 1 | 2 | 3 | [5, 10) |

Gate uses five shared seeds for every task: 10 tasks × 5 = 50 requested
records on the 0/1 pair only. Formal uses 50 shared seeds for every task:
10 tasks × 50 = 500 requested records across the two formal shards. Each live
command receives the phase-specific CUDA_VISIBLE_DEVICES,
COSMOS_POLICY_WORKER_CUDA_VISIBLE_DEVICES, task range, and env seed.

Each durable trial record persists the integer CLI env seed as `seed`,
including success, server-failure, setup-failure, and skipped records. After
formal children finish, the embedded merge verifier rejects missing task/seed
records, duplicate task/seed records, a missing or path-mismatched record
`seed`, task-to-shard mismatch, or an S4 checkpoint mismatch before
writing formal_summary.json. It then reports equal-task macro success and a
deterministic bootstrap_ci_95.
