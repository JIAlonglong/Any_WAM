# Any_WAM FlowMap Distillation Current Plan

## 1. 目标

当前 `distillation_flowmap` 的目标不是再做一个“待实现方案”，而是把已经落地的 FlowMap / on-policy 蒸馏实现、关键设计取舍、已修问题、以及后续建议整理清楚，方便继续训练、复现和团队协作。

目前系统的核心目标是：

1. 在 Flash-WAM / Any_WAM 的视频+动作联合建模框架上引入 FlowMap 蒸馏。
2. 支持视频蒸馏、动作蒸馏、以及 `video_action_aware` 辅助正则三种模式组合。
3. 在 Stage 1 中学习混合目标（diffusion / consistency / flowmap）。
4. 在 Stage 2 中切换到 on-policy transition matching，减少学生 rollout 时的分布偏移。
5. 在 Stage 3 中从同一个 Stage 1 checkpoint 再分叉出独立优化分支，用于你额外新增的训练目标，而不是强依赖 Stage 2。

---

## 2. 当前代码结构

当前相关实现主要在：

- `distillation_flowmap/flowmap_step.py`
- `distillation_flowmap/flowmap_trainer.py`
- `distillation_flowmap/model_flowmap.py`
- `distillation/patches.py`

职责划分如下：

- `flowmap_step.py`
  - 实现 Stage 1 的 `_train_step()`
  - 实现 Stage 2 的 `_onpolicy_transition_step()`
  - 实现混合时间步采样、teacher/student forward、中心差分、Euler rollout、action-aware loss
- `flowmap_trainer.py`
  - 初始化 teacher / student / target_student
  - 配置 LoRA、FSDP/FSDP1、optimizer、scheduler、dataset
  - 调度 Stage 1 / Stage 2 / Stage 3 训练
  - 执行 checkpoint 保存和 resume
- `model_flowmap.py`
  - 给 student / target_student 注入双时间步嵌入能力
  - patch forward，使模型支持 `r_timestep` / `action_r_timestep`
- `distillation/patches.py`
  - 安装 `flash_attn` stub
  - 提供 fail-fast 的 `SafeMultiLatentLeRobotDataset`

当前保留的配置文件：

- `distillation_flowmap/config.py`
  - RobotWin / 通用基础配置
- `distillation_flowmap/config_libero.py`
  - 原始 LIBERO 基础配置
- `distillation_flowmap/config_libero_optimized.py`
  - LIBERO FlowMap 共享基础配置
- `distillation_flowmap/config_libero_fullfinetune_stage1_warmup.py`
  - Stage 1 全量微调 warmup
- `distillation_flowmap/config_libero_fullfinetune_stage2_anyflow.py`
  - Stage 2 全量微调 continuation；集中包含 OPD aux、rollout eval 和 full-ft 参数

旧的 `config_libero_optimized_stage2.py`、`config_libero_optimized_stage2_distillstyle.py` 和
`config_libero_optimized_stage3.py` 已经合并/移除，避免 Stage2 配置链过长。

---

## 3. 当前训练设计

### 3.1 Stage 1: FlowMap Distillation

Stage 1 仍然是主基础训练阶段，当前设计是：

1. 对视频采样混合 `(t, r)`：
   - diffusion: `r = t`
   - consistency: `r = 0`
   - flowmap: `0 < r < t`
2. 对动作分支单独采样自己的 `(action_t, action_r)`。
3. teacher 在 `t` 处给出目标速度场，并通过中心差分近似 `dF/dt`。
4. student 在 `r` 处预测，与目标计算 loss。

当前混合采样逻辑已经改成真正的 per-sample 随机模式采样，不再依赖 `round(ratio * batch_size)` 之类的固定切块方式。

### 3.2 Stage 1 的监督项

当前实现支持以下损失项：

- 视频主损失
  - `video_loss` 或 `video_transition_loss`
- 动作蒸馏损失
  - `action_loss`
- 动作 GT 回归
  - `gt_regression_loss`
- 动作感知辅助正则
  - `action_aware_loss`
- 视频局部 FM 正则
  - `local_fm_loss`

动作分支当前设计不是简单绑定视频时间步，而是：

- `distill_action=True` 时，动作拥有独立的 `action_t/action_r`
- `action_aware=True` 但 `distill_action=False` 时，也会构建完整动作输入，只是不计算动作蒸馏主损失

这点现在已经在 `_train_step()` 和 `_onpolicy_transition_step()` 里统一。

### 3.3 Stage 2: AnyFlow + Student-State OPD Aux

Stage 2 当前不是旧的 replacement-style OPD，也不是单独的 Stage3/KTO 分支。
当前设计是：

1. 保留 AnyFlow / FlowMap 主目标。
2. 低频加入 OPD aux，默认 `OPD_AUX_INTERVAL=8`。
3. student 从 `x_t` 做 K 步 Euler rollout 到自己的 `student_x_r`。
4. teacher 直接在同一个 `student_x_r` 上 forward，提供 transition target。
5. velocity transition loss 只比较同一个 latent state 上的 teacher/student velocity。

当前入口是：

- `config_libero_fullfinetune_stage2_anyflow.py`
- `flowmap_trainer.py` 中的 OPD aux 调度
- `flowmap_step.py` 中的 `_opd_aux_transition_step()`

默认 rollout pair 收窄为 `[[1, 1], [2, 1]]`，action OPD 默认关闭，先验证 video
侧少步 rollout 是否改善。

### 3.4 已移除的历史分支

Stage3/KTO 配置和旧 stage2 run 脚本已经移除。后续如果要做 KTO/PCGrad/其它
多目标优化，建议在当前干净的 Stage2 配置稳定后单独新建实验配置。
3. `cfg.rollout_step_pairs = [[1, 1]]`
4. `cfg.teacher_micro_steps = 2`
5. `cfg.video_transition_weight = 0.2`
6. `cfg.local_fm_weight = 0.05`
7. `cfg.transition_loss_type = 'huber'`
8. `cfg.kto_adaptive = True`
9. `cfg.kto_good_weight = 0.3`
10. `cfg.kto_bad_weight = 1.0`
11. `cfg.max_train_steps = 5000`
12. `cfg.learning_rate = 5e-6`
13. 默认 `resume_from_path` 指向 Stage 1 checkpoint
14. 默认 `output_dir` 为 `output_libero_stage3_kto`

这说明 Stage 3 当前不是新的基础训练阶段，而是：

1. 同样从 Stage 1 checkpoint 热启动
2. 复用 Stage 2 的 on-policy transition matching 主体
3. 只是在视频 transition loss 上打开了 KTO-style pointwise adaptive weighting
4. 因此它更准确地说是 “Stage 2 on-policy 的 KTO 变体”

可以把当前阶段关系理解为：

```text
Stage 1 (FlowMap base)
├── Stage 2: on-policy transition matching
└── Stage 3: KTO-PAOPD (same on-policy path, different weighting config)
```

因此，当前最关键的不是“先 Stage 2 再 Stage 3”，而是：

- 先确保 Stage 1 checkpoint 可信
- 再从这个可信 Stage 1 checkpoint 分别拉出 Stage 2 / Stage 3

如果 Stage 1 本身带着旧 bug 训练出来，那么 Stage 2 和 Stage 3 都会一起继承这个问题。

### 3.5 Stage 2 和 Stage 3 的真实区别

从当前代码看，Stage 2 和 Stage 3 的区别主要不在训练框架，而在配置：

- Stage 2
  - 目标：标准 on-policy transition matching
  - 重点：减少学生 rollout 的 exposure bias
- Stage 3
  - 目标：在 on-policy transition matching 上再加入 KTO-style 自适应 token weighting
  - 重点：把更多梯度预算分给“学生还没学好”的 token

因此 Stage 3 的定位不是“另一个完全独立算法”，而是：

- 基于 Stage 2 框架
- 但从 Stage 1 checkpoint 直接启动
- 用不同的 loss weighting 做并行实验

---

## 4. 当前实现里的关键设计

### 4.1 双时间步条件

student / target_student 通过 `delta_embedder` 接收参考时间步：

- 视频：`r_timestep`
- 动作：`action_r_timestep`

teacher 保持原始结构，不做 FlowMap 改造。

### 4.2 视频和动作时间步解耦

当前实现明确支持：

- 视频独立的 `video_t/video_r`
- 动作独立的 `action_t/action_r`

动作分支不再被动复用视频时间步。这对于：

- `joint`
- `video_action_aware`
- on-policy 动作正则

都很关键。

### 4.3 On-policy student 语义统一

当前 on-policy 实现里，student rollout Phase 1 和 terminal loss Phase 2 现在都走同一套 student field 语义，不再出现：

- rollout 用 conditional-only
- final loss 用 CFG-combined

这种不一致。

这次修复后：

- `_student_euler_integrate()` 的 rollout 和 final prediction 都统一经过 `_student_cfg_forward()`
- `force_cfg=True` 用于 rollout 保持与 terminal loss 一致的向量场语义

### 4.4 Stage 3 的 KTO 自适应加权

当前 KTO 逻辑已经在 `flowmap_step.py` 的 on-policy 视频 loss 中接入，核心思想是：

1. 先计算 student 和 teacher 的 token 级误差
2. 估计 token similarity
3. 用当前 batch 的中位数作为动态阈值
4. 已学好的 token 用较小权重
5. 未学好的 token 用较大权重

这意味着 Stage 3 不是改 student / teacher rollout，本质上改的是 on-policy transition loss 的样本内重加权策略。

### 4.5 Resume 和 Checkpoint 设计

当前 checkpoint 设计已经更新为：

- 保存：
  - `online_student`
  - `target_student`
  - `optimizer.pt`
  - `lr_scheduler.pt`
  - `config.json`
  - `discriminator` 相关状态（如果启用 DMD）
- 恢复：
  - 强校验 `resume_from_step` 与 checkpoint 中的 `checkpoint_step`
  - LoRA checkpoint 必须配 `use_lora=True`
  - 如果有 `optimizer.pt` / `lr_scheduler.pt`，优先恢复完整状态
  - 如果是老 checkpoint 没有 scheduler 状态，则退回原来的 scheduler fast-forward

这比旧版本更安全，避免了 silent mismatch。

### 4.6 数据集加载策略

数据集加载当前默认是 fail-fast：

- 如果某些子数据集损坏或缺失，默认直接报错
- 只有显式设置 `allow_partial_datasets=True` 才允许跳过

这样做是为了避免训练无意中在“半截数据集”上继续跑，导致数据分布 silently 变掉。

---

## 5. 这轮已经完成的关键修复

下面这些问题已经修过，并且已经同步到远端代码。

### 5.1 Stage 1 / 通用训练逻辑

1. 混合时间步采样 bug
   - 之前 `batch_size=1` 或小 batch 时模式采样会失真
   - 现在已改成 per-sample 随机分配

2. `selective_cdiff` 子 batch 的 `empty_emb` 使用错误
   - 会导致 central difference 子路径条件不一致
   - 已修

3. `distill_action=False` 时共享动作路径崩溃
   - 已改成在 `action_aware` 场景下也准备完整动作输入

4. EMA action target 条件不一致
   - 之前 action target 计算时视频上下文和时间步不匹配
   - 已修成使用 `video_context_r`

5. 动作 consistency 分支时间步用错
   - 非 `x0` 参数化时错误用了 `action_t_sigma`
   - 现在已改为 `action_r_sigma`

6. per-sample timestep rollout 修复
   - 之前 Euler 积分部分用过 batch mean 时间步
   - 现在改为按样本插值路径

### 5.2 On-policy 路径

1. `video_action_aware` 没有真正闭环
   - `_onpolicy_transition_step()` 原来只在 `distill_action=True` 时构建动作路径
   - 已改为 `distill_action or action_aware`

2. on-policy 的 `action_r` 没有传进 video rollout
   - 之前 video rollout 用的是视频 `r`
   - 现在显式支持 `action_target_r`

3. rollout 和 final loss 语义不一致
   - 之前 Phase 1 / Phase 2 student field 不一致
   - 现在已统一

### 5.3 Resume / Checkpoint 路径

1. `resume_from_path` 时 step 丢失
   - 现在会从 checkpoint `config.json` 读取 `checkpoint_step`

2. `resume_from_path` / `resume_from_step` 不一致不报错
   - 现在会 fail fast

3. LoRA checkpoint 可以被错误地用非 LoRA 配置恢复
   - 现在会 fail fast

4. 主 optimizer / scheduler 状态没有保存恢复
   - 现在已经补齐

5. checkpoint 保存失败只记日志不终止
   - 现在会直接抛异常，防止后续基于坏 checkpoint 恢复

---

## 6. 当前训练建议

### 6.1 Stage 1

Stage 1 应该基于修复后的代码重新训练。

原因：

- 旧 Stage 1 训练时混合采样逻辑有 bug
- 旧代码下的 checkpoint 统计步数和真实状态可能不一致
- 部分动作路径和 on-policy 相关逻辑之前没有闭环

如果目标是得到可信的 Stage 2 / Stage 3 起点，建议不要继续沿用修复前生成的 Stage 1 checkpoint。

### 6.2 Stage 2

Stage 2 当前建议：

1. 只在修复后的 Stage 1 checkpoint 上启动
2. 先开较小的 `rollout_step_pairs` 做稳定性验证
3. 先验证：
   - loss 是否稳定
   - `video_transition_loss` 是否正常下降
   - `action_aware_loss` 是否量级合理
4. 再扩大 rollout 步数范围

### 6.3 Stage 3

Stage 3 当前建议按“独立分支”来理解，而不是默认接在 Stage 2 后面：

1. 也只从修复后的 Stage 1 checkpoint 启动
2. 不要把旧的、未修复 Stage 1 当作 Stage 3 的基座
3. 当前代码里的 Stage 3 实际就是 KTO-PAOPD，建议单独比较：
   - Stage 2 标准 on-policy
   - Stage 3 KTO 自适应加权
4. 重点观察：
   - `video_transition_loss` 是否更稳定
   - KTO weighting 是否导致过度关注少数 token
   - 动作相关 loss 是否被视频加权策略间接压制
5. 先确认 Stage 3 的新增加权不会掩盖 Stage 1 基础能力退化，再扩大训练

### 6.4 Resume 使用建议

恢复训练时建议遵守：

1. `resume_from_path` 和 `resume_from_step` 不要随便同时写两个互相矛盾的值
2. LoRA checkpoint 必须配套 `use_lora=True`
3. 同阶段中断恢复时，优先用新 checkpoint（包含 optimizer/scheduler 状态）

---

## 7. 当前已知限制

虽然这轮高优先问题都已经修了，但还有一些不是 bug、只是当前实现边界：

1. on-policy 训练当前仍然比 Stage 1 更复杂、更敏感
2. 当前 `video_t.mean(dim=-1)` 的时间步加权方式默认帧内共享时间步，因此现在没问题；如果未来改成 frame-wise 不同时间步，需要重新审视
3. `allow_partial_datasets=True` 是显式 opt-in；一旦开启，训练集分布可能被破坏，需要人工确认
4. Stage 3 作为独立分支时，新增目标本身的稳定性和对 base 能力的保持，还需要单独验证，不能默认继承 Stage 2 的结论

---

## 8. 建议的后续工作

建议按下面顺序继续：

1. 用当前修复后的代码重新跑 Stage 1
2. 产出新的、可信的 Stage 1 checkpoint
3. 基于新 Stage 1 checkpoint 启动 Stage 2 on-policy
4. 基于同一个新 Stage 1 checkpoint 启动 Stage 3 独立分支
5. 分别评估 Stage 2 和 Stage 3，再决定是否继续串联更高层训练

---

## 9. 当前状态总结

截至本次更新，`distillation_flowmap` 已经不是“概念验证代码”，而是一套已经补齐了以下关键能力的可训练实现：

- FlowMap Stage 1 混合蒸馏
- 双模态视频+动作路径
- `video_action_aware` 独立闭环
- on-policy transition matching
- 更安全的 checkpoint / resume
- fail-fast 的数据集加载策略

当前最重要的结论是：

- 老的 Stage 1 不建议继续信任
- 修复后的代码可以作为新的正式训练基线
- Stage 2 和 Stage 3 都应该从新的 Stage 1 checkpoint 分叉
- 当前更合理的结构是“Stage 1 为基座，Stage 2 / Stage 3 并行实验”
