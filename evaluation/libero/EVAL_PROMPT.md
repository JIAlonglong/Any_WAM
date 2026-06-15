请帮我评估 Flash-WAM Flow Map 蒸馏的阶段一效果。项目路径：/kpfs-intern/jialongliu/projects/Flash-WAM

## 背景
- 阶段一训练已完成，checkpoint 在 distillation_flowmap/output_libero/checkpoints/ 下（step_500 到 step_3500）
- 使用 flashwam conda 环境（/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python）
- 评估 client 需要 libero 环境（/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python）
- 评估架构：server（加载模型，WebSocket 推理）+ client（LIBERO 仿真，发观测收动作）
- GPU: A800 80GB

## 需要做的三件事

### 1. 视频可视化（快速 sanity check）
用 step_3500 的 student checkpoint 生成视频，看效果是否合理。

```bash
# Terminal 1: 启动 server
cd /kpfs-intern/jialongliu/projects/Flash-WAM
PYTHONPATH="$(pwd):$PYTHONPATH" \
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 29062 \
    wan_va/wan_va_server.py \
    --config-name libero --port 29056 \
    --checkpoint-path distillation_flowmap/output_libero/checkpoints/step_3500/online_student/transformer \
    --num-steps 2 \
    --save-root evaluation/outputs/step_3500/visualization

# Terminal 2: 等 server 加载完（约 60s）后跑 client
cd /kpfs-intern/jialongliu/projects/Flash-WAM
/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python evaluation/libero/client.py \
    --libero-benchmark libero_10 --port 29056 --test-num 5 \
    --task-range 0 3 --out-dir evaluation/outputs/step_3500/videos
```

视频保存在 evaluation/outputs/step_3500/videos/ 下，检查：
- 视频是否清晰（不是纯噪声/模糊）
- 机械臂是否朝目标移动
- 是否有帧间闪烁

### 2. 量化精度（student vs teacher 动作对比）
分别用 teacher 和 student checkpoint 跑推理，保存 .pt 文件，然后对比。

```bash
# Teacher server
cd /kpfs-intern/jialongliu/projects/Flash-WAM
PYTHONPATH="$(pwd):$PYTHONPATH" \
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 29063 \
    wan_va/wan_va_server.py \
    --config-name libero --port 29057 \
    --checkpoint-path checkpoints/libero/transformer \
    --num-steps 2 \
    --save-root evaluation/outputs/step_3500/actions/teacher

# Teacher client
/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python evaluation/libero/client.py \
    --libero-benchmark libero_10 --port 29057 --test-num 5 \
    --task-range 0 3 --out-dir evaluation/outputs/step_3500/videos_teacher

# Student server（换 checkpoint 和端口）
PYTHONPATH="$(pwd):$PYTHONPATH" \
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python -m torch.distributed.run \
    --nproc_per_node 1 --master_port 29064 \
    wan_va/wan_va_server.py \
    --config-name libero --port 29058 \
    --checkpoint-path distillation_flowmap/output_libero/checkpoints/step_3500/online_student/transformer \
    --num-steps 2 \
    --save-root evaluation/outputs/step_3500/actions/student

# Student client
/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python evaluation/libero/client.py \
    --libero-benchmark libero_10 --port 29058 --test-num 5 \
    --task-range 0 3 --out-dir evaluation/outputs/step_3500/videos_student

# 对比
/kpfs-intern/jialongliu/miniforge3/envs/flashwam/bin/python evaluation/libero/compare_actions.py \
    --teacher-dir evaluation/outputs/step_3500/actions/teacher \
    --student-dir evaluation/outputs/step_3500/actions/student \
    --output-file evaluation/outputs/step_3500/action_comparison.json
```

### 3. LIBERO 任务成功率（最终指标）
用 step_3500 跑 libero_10 的全部 10 个任务，每个任务 50 episode。

```bash
# Server（同上，用 step_3500）
# Client:
/kpfs-intern/jialongliu/miniforge3/envs/libero/bin/python evaluation/libero/client.py \
    --libero-benchmark libero_10 --port 29056 --test-num 50 \
    --task-range 0 10 --out-dir evaluation/outputs/step_3500/success_rate
```

成功率 JSON 在 evaluation/outputs/step_3500/success_rate/ 下。

## 注意事项
- server 和 client 必须用不同的终端（server 要常驻）
- server 启动后等 60s 再跑 client（模型加载需要时间）
- 如果要评估不同 step 的 checkpoint，换 --checkpoint-path 即可
- 可以同时评估 target_student（EMA 版本）：checkpoint 路径改为 step_3500/target_student/transformer
- 如果遇到 port 冲突，换 --master_port 和 --port

## 期望输出
请把以下结果汇总给我：
1. 视频可视化：生成的视频路径列表 + 肉眼评估（是否清晰、动作是否合理）
2. 量化精度：MSE、L1、余弦相似度（从 action_comparison.json 读取）
3. 任务成功率：每个任务的成功率 + 平均成功率
