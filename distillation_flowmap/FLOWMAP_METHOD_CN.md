# Distillation FlowMap 方法说明

本文档以论文 Method 的方式说明本仓库 `distillation_flowmap/` 中的 FlowMap 蒸馏实现。重点覆盖模型改造、时间步采样、视频与动作目标、Stage 1 warmup、Stage 2 OPD、训练/推理细节，以及当前 LIBERO 配置中的关键工程约束。

## 1. 任务设定

LingBot-VA 是一个联合视频-动作世界模型。给定语言指令和历史观测，它在 latent 空间中同时建模未来视频片段和动作序列。原始 teacher 需要多步 Flow Matching 推理，计算代价较高。Flash-WAM/FlowMap distillation 的目标是训练一个 student，使其在更少的 Euler 步数下逼近 teacher 的视频与动作生成行为。

我们把每个训练样本写成：

```text
z_0: 干净视频 latent，形状 [B, C_v, F, H, W]
a_0: 干净动作 latent，形状 [B, C_a, F, N, 1]
c:   文本条件 embedding
```

其中 LIBERO 配置中 `F=frame_chunk_size=4`，动作每帧有 `N=action_per_frame=4` 个低层动作 token，视频 latent 通道数通常为 48，动作通道补齐到 30。训练中视频和动作各自使用不同的 noise scheduler：视频使用 `snr_shift=5.0`，动作使用 `action_snr_shift=0.05`，因为动作流处于更低噪声、更连续的数值区间。

Flow Matching 加噪形式可写为：

```text
x_t = (1 - sigma_t) * x_0 + sigma_t * eps
v*(x_t, t) = eps - x_0
```

其中 `t` 在代码中以 `[0, num_train_timesteps]` 表示，`sigma=t/num_train_timesteps` 是归一化噪声强度。模型预测 velocity `v_theta(x_t, t, r, c)`，其中 `r` 是目标参考时间步。普通 diffusion 只需要 `t`；FlowMap 需要同时知道从 `t` 映射到哪个 `r`。

## 2. 模型改造：双时间步条件

核心改造位于 `distillation_flowmap/model_flowmap.py`。

原始 WanTransformer 只有一个时间步 embedding。FlowMap 需要建模从当前噪声时间 `t` 到参考时间 `r` 的映射，因此把原始 condition embedder 替换为 `WanTwoTimeTextImageEmbedding`：

```text
e_t     = TimeEmbed(t)
e_delta = DeltaEmbed(delta(t, r))
e       = (1 - g) * e_t + g * e_delta
```

其中 `g=gate_value`，当前 LIBERO full-finetune 配置中常用 `gate_value=0.25`。`delta(t, r)` 的定义由 `deltatime_type` 控制，目前默认是 `r`，即 delta embedder 直接编码参考时间步 `r`。

模型 forward 被 patch 后接受：

```text
r_timestep:        视频参考时间步 [B, F]
action_r_timestep: 动作参考时间步 [B, F_action]
```

这样 student 可以在同一个输入 `x_t` 上学习“到 `r` 的速度场/映射”。

## 3. 混合时间步采样

核心采样函数是 `FlowMapStepMixin.sample_timestep_mixed()`。

每个 batch sample 先采样两个均匀随机数：

```text
u1, u2 ~ Uniform(0, 1)
t = max(u1, u2)
r = min(u1, u2)
```

再按比例把 `(t, r)` 分成三类：

```text
diffusion:   r = t
consistency: r = 0
flowmap:     0 < r < t
```

默认比例来自配置：

```text
diffusion_ratio   = 0.5
consistency_ratio = 0.25
flowmap_ratio     = 0.25
```

然后对视频和动作分别应用各自的 SNR shift，并把标量扩展成 per-frame 时间步 `[B, F]`。这个设计有两个目的：保留普通 FlowMatch diffusion 学习，避免模型忘记局部 denoising；同时引入 `r=0` 和 `0<r<t` 的端点/中间点映射，使 student 学会跨时间步跳转。

## 4. Stage 1：AnyFlow/FlowMap 主目标

Stage 1 的核心训练函数是 `_train_step()`。它是整个 distillation flowmap 的主目标，也是 Stage 2 中继续保留的主 flowmap auxiliary。

### 4.1 Teacher CFG 速度场

对视频，从干净 latent `z_0` 和噪声 `eps` 得到 `z_t`。teacher 在 `z_t, t` 处做 CFG forward：

```text
v_T^cond   = T(z_t, t, c)
v_T^uncond = T(z_t, t, empty)
v_T        = v_T^uncond + s * (v_T^cond - v_T^uncond)
```

其中 `s` 从 `[cfg_min, cfg_max]` 采样。Stage 1 当前通常固定为 `fuse_guidance_scale=3.0`，因此 `cfg_min=cfg_max=3.0`。

### 4.2 FlowMap 目标

理论上 FlowMap 目标可以写成：

```text
target(t -> r) = v_T(x_t, t) - (t - r) * dF/dt
```

其中 `dF/dt` 可用中心差分估计：

```text
dF/dt ≈ [F(x_{t+eps}, t+eps) - F(x_{t-eps}, t-eps)] / (2 eps)
```

代码中保留了 `compute_central_difference()` 作为参考，并实现了 merged central-difference 路径以减少 teacher forward 次数。对于 diffusion 样本 `r=t`，`dF/dt=0`，目标退化为标准 FlowMatch teacher velocity。

实际训练目标是 student 在同一个 noisy state 上、带参考时间 `r` 的预测：

```text
v_S = S(z_t, t, r, c)
L_video = || v_S - target(t -> r) ||
```

视频 loss 支持 L2/Huber，当前常用 Huber 小阈值来限制 outlier。还会根据 `weight_type` 做 timestep weighting，并用 diffusion 样本的 loss 均值缩放非 diffusion 样本，平衡三种模式的梯度贡献。

### 4.3 Action 分支

动作分支也使用 FlowMatch 加噪，但数值特性不同，因此单独使用 action scheduler 和 `action_snr_shift=0.05`。LIBERO 中 action token 很密集，相邻动作差异小，训练 student 时动作时间维下采样：

```text
action_downsample_factor = 4
```

也就是 teacher/数据中保留全动作序列，但 student action forward 只取每 4 帧动作 token。这是当前 LIBERO Stage 1/Stage 2 的对齐项，推理 server 中也必须保持一致。

在当前 LIBERO full-finetune Stage 1/Stage 2 配置中，`action_use_flowmap=True`，动作侧也会对非 diffusion token 计算 action central difference；旧的 base config 可能仍保留 `False` 作为 ablation/兼容选项。动作目标有三部分：

```text
L_action:          action x0/teacher target distillation
L_gt_regression:   直接回归 GT action 的稳定项
L_action_local_fm: action velocity 对 GT FlowMatch target 的局部锚定
```

其中 action distillation 默认使用 target student/EMA 产生更平滑的目标。teacher 先把 action 从 `t` Euler 推到 `r`，target student 在 `r` 处预测 velocity，再转成 x0 目标：

```text
a_r^T         = a_t + v_T^a * (sigma_r - sigma_t)
v_target^EMA  = S_ema(a_r^T, r)
x0_target     = a_r^T - sigma_r * v_target^EMA
```

student 则预测：

```text
x0_student = a_t - sigma_r * v_S^a
```

再对 masked action token 做 MSE。动作 loss 使用 per-mask normalization，避免不同有效动作长度样本被隐式重加权。

### 4.4 Stage 1 总 loss

Stage 1 主 loss 可概括为：

```text
L_stage1 = w_video * L_video
         + w_action_block * (
             w_action * L_action
           + w_gt     * L_gt_regression
           + w_local  * L_action_local_fm
           )
```

Full-finetune Stage 1 使用较小学习率，例如 `1e-6`，并保存 online student 和 target student：

```text
checkpoints/step_N/online_student/transformer
checkpoints/step_N/target_student/transformer
```

Stage 2 通常从 Stage 1 target student 接续。

## 5. Target student 与 EMA

trainer 初始化中同时维护：

```text
teacher:        frozen LingBot-VA teacher
student:        online trainable student
target_student: frozen-for-gradient EMA student
```

teacher 用于视频/动作的 teacher velocity 或 rollout target；target student 不参与反向传播，但每次 optimizer step 后按 EMA 从 online student 更新。它用于更稳定的 action target 和 checkpoint 保存。保存 checkpoint 时同时写出 online/target 两份权重。

Stage 2 从 Stage 1 checkpoint 接续时通常设置：

```text
RESUME_ONLINE_FROM_TARGET=1
```

这会让 online student 和 target student 都从 Stage 1 的 `target_student/transformer` 起步，之后 online 正常训练、target 继续做 EMA 更新。

从 full-model checkpoint 恢复时，FlowMap 新增的 delta embedder 权重需要从 safetensors 中显式恢复；代码中有 `load_flowmap_delta_weights()`，用于补回 Wan loader 默认不会加载的 FlowMap delta 权重。

## 6. Stage 2：OPD 辅助目标

Stage 2 配置是 `config_libero_fullfinetune_stage2_anyflow.py`。它不是完全替换 Stage 1 目标，而是在 Stage 1/AnyFlow 主目标上增加 OPD auxiliary。实现上不是把两个 loss 先求和再做一次 backward，而是在同一个 gradient accumulation/optimizer step 内先执行 `_train_step()` 的 FlowMap backward，再按条件执行 `_opd_aux_transition_step()` 的 OPD backward，最终共享一次 optimizer update。

```text
grad_step = grad(FLOWMAP_AUX_WEIGHT * L_flowmap_main)
          + grad(OPD_AUX_WEIGHT     * L_OPD_aux)
```

当前配置中 `flowmap_aux_weight=0.25`，表示 Stage 1 的 FlowMap 主目标在 Stage 2 中降权保留，用作稳定项；OPD 是主要新增 correction。真实梯度预算仍取决于 raw loss 量级、`OPD_TRANSITION_GROUP_WEIGHT`、anchor cap 和各 component 权重，因此需要看日志中的 weighted contribution/ratio，而不能只看 raw loss。

### 6.1 为什么需要 OPD

Stage 1 主要在 teacher/data 分布上的 noisy state 学习一步或跨步映射。但真实快速推理时，student 会反复访问自己生成的 state，这些 state 可能偏离 teacher/data 分布。OPD 的目的就是在 student-induced state 上查询 teacher，让 student 在自己的 rollout 分布上对齐 teacher field。

### 6.2 Student rollout

OPD 先采样 `(t, r)`，加噪得到 `z_t`，再采样一组 rollout 步数对：

```text
(N_steps, K_steps) ∈ rollout_step_pairs
```

其中 `K_steps` 是 student 从 `t` 到 `r` 的 Euler 步数，`N_steps` 是 endpoint 模式下 teacher rollout 步数。当前默认覆盖低步数和中等步数，例如：

```text
[1,1], [2,1], [4,1], [4,2], [8,2], [8,4], [16,4]
```

student Euler 更新公式：

```text
z_{i+1} = z_i + v_S(z_i, t_i, r_{i+1}) * (sigma_{i+1} - sigma_i)
```

rollout 的梯度模式由 `OPD_ROLLOUT_GRAD_MODE` 控制：

```text
endpoint:  rollout no_grad，只在最终 state 上做有梯度 forward
last_step: 最后一个 Euler step 保留梯度
full:      全 rollout 保留梯度
```

当前默认是 `endpoint`，更省显存也更稳定。

### 6.3 Same-state teacher target

当前默认：

```text
OPD_TEACHER_TARGET_MODE=student_state
VIDEO_TRANSITION_PARAM=velocity
ACTION_TRANSITION_PARAM=velocity
```

这表示 teacher 不再独立 rollout 到另一个 endpoint，而是在 student 实际到达的 state 上查询 teacher velocity：

```text
z_r^S = StudentRollout(z_t, t -> r)
v_T   = Teacher(z_r^S, r)
v_S   = Student(z_r^S, r)
L_transition = || v_S - stopgrad(v_T) ||
```

这个 same-state velocity matching 是当前 Stage 2 OPD 的核心。它避免比较两个不同 endpoint 上的 velocity，减少 teacher/student state mismatch。若改成 `endpoint` 模式，则 velocity MSE 不再语义一致，代码会回退到 x0/endpoint loss。

### 6.4 Low-noise query bias

Stage 2 使用 DanceOPD 风格的低噪声查询偏置：

```text
OPD_QUERY_BIAS=low_t
OPD_LOW_NOISE_MAX_SIGMA=0.25
```

即 OPD 的 `r` 更偏向低噪声区域，使 teacher/student matching 更接近实际快速推理后段。低噪声 `r` 由 Beta 分布采样并限制到 `sigma <= max_sigma`。

### 6.5 OPD video losses

视频 OPD 包含三类项：

```text
L_video_transition: same-state teacher/student field matching
L_endpoint_aux:     endpoint/x0 辅助锚定
L_video_local_fm:   student velocity vs GT FlowMatch target
```

其中 transition 是主信号，endpoint/local-FM 是小权重 anchor。

### 6.6 OPD action losses

action OPD 与 video 对齐，也在 student-induced action state 上查询 teacher。当前实现中 action transition 和 action local-FM 被拆成两个不同权重：

```text
ACTION_TRANSITION_BLOCK_WEIGHT=4.0
ACTION_LOCAL_FM_BLOCK_WEIGHT=1.0
ACTION_LOCAL_FM_WEIGHT=0.003
```

这么做是为了避免 action local-FM 因 raw MSE 量级较大而主导 OPD。action loss 使用 per-sample masked mean 再 batch mean，使不同有效 action length 的样本不会被隐式重加权。

## 7. Transition group 与 anchor group

Stage 2 OPD 不直接把所有 component 简单相加，而是拆成两组：

```text
transition_group = video_transition + action_transition
anchor_group     = endpoint_aux + video_local_fm + action_local_fm
```

然后：

```text
scaled_transition = OPD_TRANSITION_GROUP_WEIGHT * transition_group
scaled_anchor     = min(anchor_group, OPD_ANCHOR_CAP_RATIO * |scaled_transition|)
```

当前默认：

```text
OPD_TRANSITION_GROUP_WEIGHT=25.0
OPD_ANCHOR_CAP_RATIO=0.25
```

目标是让 transition/field matching 成为 OPD 主预算，anchor 只提供稳定性。日志中会记录 raw component、weighted contribution 和 ratio，例如：

```text
loss_weighted/opd_transition_group_scaled
loss_weighted/opd_anchor_group_scaled
loss_ratio/opd_transition_total
loss_ratio/opd_anchor_total
```

理想状态下 transition/field 应占主要比例，anchor 不应长期压过 transition。

## 8. 推理与 i2va demo

推理入口在 `wan_va/wan_va_server.py`，调用 `flowmap_inference()` 执行少步 Euler。每一步都会构建 `(t_i, r_i)`，并同时更新视频和动作：

```text
x_{r_i} = x_{t_i} + v_theta(x_{t_i}, t_i, r_i) * (sigma_{r_i} - sigma_{t_i})
```

首个 chunk 会把第 0 帧视频 latent 替换为当前观测编码，作为视觉锚点；动作第 0 帧置 0。LIBERO 推理必须保持：

```text
action_downsample_factor=4
```

i2va demo 模式使用 `--config-name libero_i2av`。它从初始图片和 prompt 生成多个 chunk，先把 latent 按时间拼接，再一次性 VAE decode 成 `demo.mp4`。这和真实 LIBERO 环境 rollout 不同：i2va 是生成视频；环境 rollout 是把 action 接到 LIBERO simulator 里执行。

## 9. 关键工程约束

### 9.1 Gradient checkpointing

Stage 2 OPD 默认关闭：

```text
GRADIENT_CHECKPOINTING=0
```

原因是 OPD 有额外 student backward 路径，PyTorch FSDP2 在 activation-checkpoint recompute 中可能触发混合 Tensor/DTensor 报错：

```text
aten.addmm.default: got mixed torch.Tensor and DTensor
```

如果显存不够，优先调小 batch 或增加 gradient accumulation，不要直接把 checkpointing 改回 1。

### 9.2 FlexAttention mask

teacher CFG forward 可能使用 2B batch，而 student forward 使用 B batch。FlexAttention mask 是类级状态，可能被不同 batch shape 污染。因此 student forward 前会显式调用 `FlexAttnFunc.init_mask()`，checkpoint recompute 时也会刷新 mask。

### 9.3 FSDP 与 no-FSDP copy

teacher frozen 推理使用 `_teacher_nofsdp`，减少 FSDP all-gather 开销。部分 rollout 路径可使用 `_student_nofsdp`，避免在 no_grad rollout 中触发 DTensor dispatch。真正需要梯度的最终 student forward 仍走 FSDP student。

## 10. 主要文件索引

```text
distillation_flowmap/model_flowmap.py
  模型改造：双时间步 embedding、forward patch、checkpoint mask refresh。

distillation_flowmap/flowmap_step.py
  训练目标：混合时间步采样、主 FlowMap step、student/teacher Euler、Stage 2 OPD。

distillation_flowmap/flowmap_trainer.py
  模型加载、teacher/student/target_student、EMA、FSDP、optimizer、训练循环与日志。

distillation_flowmap/config_libero_fullfinetune_stage1_warmup.py
  Stage 1 full-parameter warmup 配置。

distillation_flowmap/config_libero_fullfinetune_stage2_anyflow.py
  Stage 2 AnyFlow/OPD continuation 配置。

wan_va/wan_va_server.py
  FlowMap inference server 与 i2va demo generation。
```

## 11. 方法小结

当前 distillation flowmap 可以概括为两阶段：

```text
Stage 1:
  在 teacher/data 分布上学习 AnyFlow 风格的 mixed (t, r) 映射。
  目标是让 student 获得基础的少步视频/动作生成能力。

Stage 2:
  从 Stage 1 checkpoint 继续训练，在 student-induced state 上做 OPD teacher matching。
  目标是修正 student rollout 分布上的 field，使 transition/field matching 主导，endpoint/local-FM 只做稳定 anchor。
```

设计原则是：视频和动作共享一个联合 Transformer，但 loss 和 scheduler 按模态区分；Stage 1 保证基础 FlowMap 能力，Stage 2 用 same-state OPD 把 student 生成轨迹拉回 teacher field。
