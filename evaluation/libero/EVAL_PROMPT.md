# LIBERO Evaluation Guide

当前保留两条评估链路：

1. 离线 checkpoint 指标：`distillation_flowmap/rollout_eval_stage2.py` 和 `rollout_eval_video_stage2.py`。
   这类评估在训练数据 latent/action 上比较 student 和 teacher，不会和 LIBERO 环境交互，也不会生成真实机器人执行视频。

2. 真实 LIBERO 环境评估：`evaluation/libero/run_eval_new.sh`。
   这条链路会启动 `wan_va_server.py`，再用 `evaluation/libero/client.py` 跑 LIBERO 环境，输出真实环境视频和 `succ_rate` JSON。

## 快速视频检查

```bash
cd /kpfs-intern/jialongliu/projects/Flash-WAM
bash evaluation/libero/run_eval_new.sh
```

默认评估 `distillation_flowmap/output_libero_fullft_stage2_anyflow/checkpoints/step_5000/online_student/transformer`，跑 `libero_10` 的任务 `0..3`，每个任务 3 个 episode。

输出：

```text
evaluation/outputs/libero_env_step_5000_online_student/videos/
```

## 成功率评估

```bash
cd /kpfs-intern/jialongliu/projects/Flash-WAM
EVAL_MODE=success TEST_NUM=50 bash evaluation/libero/run_eval_new.sh
```

输出 JSON：

```text
evaluation/outputs/libero_env_step_5000_online_student/results/libero_eval/*.json
```

## Teacher/Student Action 对比

```bash
cd /kpfs-intern/jialongliu/projects/Flash-WAM
EVAL_MODE=compare TEST_NUM=5 bash evaluation/libero/run_eval_new.sh
```

输出：

```text
evaluation/outputs/libero_env_step_5000_online_student/results/action_comparison.json
```

## 常用覆盖项

```bash
OUTPUT_ROOT=distillation_flowmap/output_libero_fullft_stage1_warmup \
EVAL_MODE=success TEST_NUM=20 TASK_START=0 TASK_END=10 \
bash evaluation/libero/run_eval_new.sh step_500 target_student
```

- `OUTPUT_ROOT`：切换 stage1/stage2 输出目录。
- `TEACHER_CKPT`：覆盖 teacher transformer 路径。
- `NUM_STEPS`：视频推理步数，默认 20。
- `ACTION_NUM_STEPS`：action 推理步数，默认 50。
- `TASK_START/TASK_END`：LIBERO task 范围。
- `SAVE_ROOT`：覆盖输出目录。
