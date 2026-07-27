# Cosmos Progressive S4: Full Offline + Closed-Loop Evaluation Suite

This is the single entry point for both result families of the public Cosmos
Progressive S4 release. It is orchestration only; it does not turn a
cache-only smoke result into a paper or closed-loop result.

## Required runtime

Run on a four-GPU node whose NVIDIA driver is at least `570.124.06` and whose
official `COSMOS_POLICY_PYTHON` reports CUDA `12.8` or later. The current
Flash-WAM machine's CUDA 12.4 / driver 550 runtime is intentionally rejected
by the live Cosmos preflight and must not be used for this suite.

The released S4 checkpoint is:

```text
/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/progressive_stage2_full/s4/step_5000/online_student/transformer
```

## One-command suite

Choose a new path for `SUITE_ROOT`; it must not already exist. The launcher
never removes, resumes, or overwrites a suite root.

Before creating that root, a live suite validates every supplied path and runs
the existing non-worker Cosmos runtime preflight on the planned 0/1 pair. An
invalid checkpoint/path or incompatible CUDA driver therefore fails without
leaving a partial suite directory or starting the expensive teacher cache.

```bash
source /kpfs-intern/jialongliu/miniforge3/bin/activate && conda activate flashwam && cd /kpfs-intern/jialongliu/projects/Flash-WAM/.worktrees/cosmos-progressive-s4-eval && SUITE_ROOT=/kpfs-intern/jialongliu/results/cosmos_s4_full_$(date +%Y%m%d_%H%M%S) S4_CKPT_ROOT=/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/progressive_stage2_full/s4/step_5000/online_student/transformer S4_DATASET_PATH=/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot S4_EMPTY_EMBEDDING=/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot/empty_emb.pt COSMOS_POLICY_PATH=/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B COSMOS_POLICY_PYTHON=/path/to/official-cosmos-cu128/bin/python COSMOS_PREDICT2_REPO=/kpfs-intern/jialongliu/projects/cosmos-predict2.5 bash evaluation/libero/run_cosmos_progressive_s4_suite.sh run
```

`COSMOS_POLICY_PYTHON` must be replaced with the real official Cosmos cu128
environment on the compatible node. Supply `S4_PROMPT_TABLE=<task-keyed .pt>`
only if a verified existing table should be reused; otherwise the formal
launcher materializes it from the stored training latent embeddings.

Before allocating the node, inspect exactly what will run without creating a
directory or invoking Python/Cosmos:

```bash
SUITE_ROOT=/kpfs-intern/jialongliu/results/cosmos_s4_full_dry_run S4_CKPT_ROOT=/kpfs-intern/jialongliu/models/modelscope/JIAlonglong/any-wam-cosmos-checkpoints/progressive_stage2_full/s4/step_5000/online_student/transformer S4_DATASET_PATH=/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot S4_EMPTY_EMBEDDING=/kpfs-intern/jialongliu/projects/Flash-WAM/training_data/libero-long-lerobot/empty_emb.pt COSMOS_POLICY_PATH=/kpfs-intern/jialongliu/models/cosmos_predict2_5/checkpoints/nvidia/Cosmos-Policy-LIBERO-Predict2-2B COSMOS_POLICY_PYTHON=/path/to/official-cosmos-cu128/bin/python COSMOS_PREDICT2_REPO=/kpfs-intern/jialongliu/projects/cosmos-predict2.5 bash evaluation/libero/run_cosmos_progressive_s4_suite.sh dry-run
```

## Fixed protocol and outputs

The suite serializes GPU use as follows:

| Phase | Work | GPU allocation |
| --- | --- | --- |
| prerequisite | non-worker Cosmos cu128/runtime preflight | student 0 / worker 1 reservation only |
| `protocol` | deterministic 3-selection / 5-test split, pair `1000,0` | CPU |
| `teacher_cache` | Cosmos teacher cache, 8 Euler steps | student 0 / worker 1 |
| `offline_paper` | paper metrics, S4 K=4 vs Cosmos K=8 | student 0 / worker 1 |
| `closed_loop_formal` | 10 tasks × 50 shared seeds | shards 0/1 and 2/3 |

The root contains:

```text
$SUITE_ROOT/protocol/test_manifest.json
$SUITE_ROOT/protocol/eval_pairs.json
$SUITE_ROOT/protocol/teacher_cache/test/
$SUITE_ROOT/offline_paper/summary.json
$SUITE_ROOT/closed_loop_formal/formal_summary.json
$SUITE_ROOT/suite_status.jsonl
```

`offline_paper/summary.json` is the paper metric result and must report
`is_paper_metric=true`. `closed_loop_formal/formal_summary.json` is the
separately merged closed-loop success result (500 records, equal-task macro
success, deterministic bootstrap interval). `suite_status.jsonl` records the
start/completion of each serial phase and leaves the last started phase visible
if a live child fails.
