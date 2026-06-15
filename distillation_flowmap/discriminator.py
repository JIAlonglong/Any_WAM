"""
条件动作判别器模块 —— 用于 On-Policy DMD（Distribution Matching Distillation）蒸馏。

===============================================================================
DMD（Distribution Matching Distillation）原理概述
===============================================================================

标准的一致性蒸馏（LCM）只最小化学生输出与教师输出之间的逐样本距离（如 MSE/Huber），
但忽略了生成分布的整体匹配质量。DMD 引入一个判别器，通过对抗训练来对齐学生生成的
动作分布与真实动作分布。

核心思想：
  1. 学生模型生成"假"动作 fake_actions
  2. 判别器 D 判断动作是"真"（来自数据集）还是"假"（来自学生）
  3. DMD 梯度 = D(fake) - teacher_score
     - D(fake)：判别器对假样本的打分（越高越"真"）
     - teacher_score：教师模型对同一输入的打分（作为基线）
     - 差值 > 0：判别器认为假样本比教师预期更"真"，梯度鼓励学生往这个方向走
     - 差值 < 0：判别器认为假样本不够"真"，梯度推动学生改进
  4. 将 DMD 梯度注入学生的反向传播，引导生成分布向真实分布靠拢

为什么需要判别器？
  - 仅用 MSE/Huber 可能导致模式平均（mode averaging）：学生学到多个模式的"平均值"
  - 判别器提供分布级别的反馈，鼓励学生生成落在真实数据流形上的样本
  - 特别适合动作生成：机器人动作的分布往往是多模态的（同一任务有多种可行策略）

===============================================================================
架构设计
===============================================================================

本判别器是一个条件 Transformer 判别器，架构如下：

  动作编码器：Linear(action_dim → hidden_dim)
    将原始动作向量映射到隐藏空间。

  视频条件编码器：Conv3d → AdaptiveAvgPool3d → Linear → hidden_dim
    从视频潜在表示中提取时空特征，作为判别器的"上下文"。
    判别器需要知道"在什么场景下判断这个动作是否合理"。

  文本条件编码器：Linear(text_dim → hidden_dim)
    将文本嵌入映射到隐藏空间，提供任务语义信息。

  位置编码：可学习参数
    为每个 token 添加位置信息，帮助 Transformer 区分不同位置。

  Transformer 编码层：num_layers 层 TransformerEncoderLayer
    在动作 token + 条件 token 之间做交叉注意力，融合多模态信息。

  分类头：LayerNorm → Linear → 1
    对 [CLS] 位置的输出做二分类（真/假）。

输入形状：
  actions      : [B, C, F, N, 1]  — 动作 latent（C=动作维度的 latent 通道数，F=帧数，N=每帧步数）
  video_latent : [B, C_v, F_v, H, W] — 视频潜在表示
  text_emb     : [B, L, D]        — 文本嵌入（L=序列长度，D=嵌入维度）

输出形状：
  logits       : [B, 1]           — 判别器打分（>0 倾向"真"，<0 倾向"假"）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionDiscriminator(nn.Module):
    """
    条件动作判别器。

    给定视频上下文和文本嵌入，判断一组动作是"真实的"（来自数据集）
    还是"生成的"（来自学生模型）。

    设计思路：
      - 判别器不是简单地看动作本身，而是看"在给定视频帧和文本指令的条件下，
        这个动作是否合理"。这就是"条件"判别器的含义。
      - 使用 Transformer 而非简单的 MLP，是因为动作具有时序结构（多帧、每帧多步），
        Transformer 能更好地建模动作 token 之间的关系。
      - 视频条件通过 3D 卷积提取，保留时空信息；文本条件通过线性层映射。
    """

    def __init__(
        self,
        action_dim: int = 30,
        action_latent_channels: int = 64,
        video_latent_channels: int = 16,
        text_dim: int = 4096,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_action_tokens: int = 256,
        max_text_tokens: int = 77,
    ):
        """
        初始化判别器。

        参数:
            action_dim:            原始动作维度（如 30 维机器人自由度）
            action_latent_channels: 动作 latent 的通道数（经 VAE 编码后）
            video_latent_channels:  视频 latent 的通道数（经 VAE 编码后）
            text_dim:              文本嵌入的维度（如 CLIP 的 4096 维）
            hidden_dim:            Transformer 的隐藏维度
            num_layers:            Transformer 编码层的数量
            num_heads:             多头注意力的头数
            dropout:               Dropout 概率
            max_action_tokens:     动作 token 的最大数量（用于位置编码）
            max_text_tokens:       文本 token 的最大数量（用于位置编码）
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # ------------------------------------------------------------------
        # 动作编码器：将动作 latent 映射到隐藏空间
        # 输入: [B, C, F, N, 1] → 展平为 [B, F*N, C] → Linear → [B, F*N, hidden_dim]
        # ------------------------------------------------------------------
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(action_latent_channels),
            nn.Linear(action_latent_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ------------------------------------------------------------------
        # 视频条件编码器：从视频 latent 中提取时空特征
        # 输入: [B, C_v, F_v, H, W]
        #   → Conv3d: 3D 卷积提取局部时空特征
        #   → AdaptiveAvgPool3d: 全局平均池化，压缩到固定大小
        #   → Linear: 映射到 hidden_dim
        # 输出: [B, hidden_dim]（全局视频特征向量）
        #
        # 为什么用 3D 卷积？
        #   视频 latent 是 5D 张量（B, C, F, H, W），具有时空结构。
        #   3D 卷积能同时在时间和空间维度上提取特征，捕获运动信息。
        #   AdaptiveAvgPool3d 将特征压缩到固定大小，适应不同分辨率的输入。
        # ------------------------------------------------------------------
        self.video_condition_encoder = nn.Sequential(
            nn.Conv3d(
                video_latent_channels, hidden_dim,
                kernel_size=3, stride=1, padding=1,
            ),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv3d(
                hidden_dim, hidden_dim,
                kernel_size=3, stride=2, padding=1,
            ),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # [B, hidden_dim, 1, 1, 1]
            nn.Flatten(),                       # [B, hidden_dim]
        )

        # ------------------------------------------------------------------
        # 文本条件编码器：将文本嵌入映射到隐藏空间
        # 输入: [B, L, D] → Linear → [B, L, hidden_dim]
        #
        # 这里不做池化，保留文本的序列结构，让 Transformer 自己学习
        # 哪些文本 token 对判别更重要。
        # ------------------------------------------------------------------
        self.text_condition_encoder = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ------------------------------------------------------------------
        # 可学习的位置编码
        # 包含动作位置编码和文本位置编码，帮助 Transformer 区分不同位置的 token。
        #
        # 使用可学习的位置编码（而非正弦编码），因为：
        #   1. 动作和文本的序列长度相对固定
        #   2. 可学习编码可以自适应数据分布
        #   3. 实验表明在短序列上两者效果相近，可学习编码更灵活
        # ------------------------------------------------------------------
        self.action_pos_embed = nn.Parameter(
            torch.randn(1, max_action_tokens, hidden_dim) * 0.02
        )
        self.text_pos_embed = nn.Parameter(
            torch.randn(1, max_text_tokens, hidden_dim) * 0.02
        )

        # 视频条件 token 的投影（将全局特征向量扩展为单个 token）
        self.video_token_proj = nn.Linear(hidden_dim, hidden_dim)

        # [CLS] token：用于最终分类的特殊 token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # ------------------------------------------------------------------
        # Transformer 编码层
        # 在动作 token + 视频 token + 文本 token 之间做自注意力，
        # 融合多模态信息后，通过 [CLS] token 的输出做分类。
        #
        # Transformer 的优势：
        #   1. 自注意力机制能建模任意两个 token 之间的关系
        #   2. 动作 token 可以 attend 到视频和文本 token，理解"在什么场景下做什么动作"
        #   3. 多层堆叠逐步抽象，从低级特征到高级语义
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN Transformer，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_dim),  # 最终 LayerNorm
        )

        # ------------------------------------------------------------------
        # 分类头：LayerNorm → Linear → 1
        # 对 [CLS] token 的输出做二分类，输出 logits。
        # > 0 表示"真"，< 0 表示"假"。
        # ------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        actions: torch.Tensor,
        video_latent: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播：判断动作的真伪。

        参数:
            actions      : [B, C, F, N, 1]  — 动作 latent
                           C = latent 通道数, F = 帧数, N = 每帧动作步数
            video_latent : [B, C_v, F_v, H, W] — 视频潜在表示
            text_emb     : [B, L, D]        — 文本嵌入（如 CLIP 文本编码器输出）

        返回:
            logits       : [B, 1]  — 判别器打分（未经过 sigmoid）
                           > 0 倾向"真"，< 0 倾向"假"
        """
        B = actions.shape[0]

        # === 步骤 1: 编码动作 ===
        # [B, C, F, N, 1] → [B, F*N, C] → [B, F*N, hidden_dim]
        action_tokens = actions.squeeze(-1)           # [B, C, F, N]
        action_tokens = action_tokens.permute(0, 2, 3, 1)  # [B, F, N, C]
        F_act, N_act = action_tokens.shape[1], action_tokens.shape[2]
        action_tokens = action_tokens.reshape(B, F_act * N_act, -1)  # [B, F*N, C]
        action_tokens = self.action_encoder(action_tokens)  # [B, F*N, hidden_dim]

        # 添加动作位置编码（截取到实际序列长度）
        seq_len = action_tokens.shape[1]
        action_tokens = action_tokens + self.action_pos_embed[:, :seq_len, :]

        # === 步骤 2: 编码视频条件 ===
        # [B, C_v, F_v, H, W] → Conv3d → Pool → [B, hidden_dim] → [B, 1, hidden_dim]
        video_feat = self.video_condition_encoder(video_latent)  # [B, hidden_dim]
        video_token = self.video_token_proj(video_feat).unsqueeze(1)  # [B, 1, hidden_dim]

        # === 步骤 3: 编码文本条件 ===
        # [B, L, D] → [B, L, hidden_dim]
        text_tokens = self.text_condition_encoder(text_emb)  # [B, L, hidden_dim]

        # 添加文本位置编码（截取到实际序列长度）
        text_len = text_tokens.shape[1]
        text_tokens = text_tokens + self.text_pos_embed[:, :text_len, :]

        # === 步骤 4: 拼接所有 token ===
        # 序列结构: [CLS] + video_token + action_tokens + text_tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, hidden_dim]
        tokens = torch.cat([cls_tokens, video_token, action_tokens, text_tokens], dim=1)

        # === 步骤 5: Transformer 编码 ===
        # 多模态自注意力：动作 token 可以 attend 到视频和文本 token
        tokens = self.transformer(tokens)  # [B, 1+1+F*N+L, hidden_dim]

        # === 步骤 6: 分类 ===
        # 取 [CLS] token 的输出做分类
        cls_output = tokens[:, 0, :]  # [B, hidden_dim]
        logits = self.classifier(cls_output)  # [B, 1]

        return logits


def train_discriminator_step(
    discriminator: ActionDiscriminator,
    real_actions: torch.Tensor,
    fake_actions: torch.Tensor,
    video_latent: torch.Tensor,
    text_emb: torch.Tensor,
) -> torch.Tensor:
    """
    判别器的一步训练。

    标准的 GAN 判别器训练：
      - 真样本 label = 1，假样本 label = 0
      - 使用 BCE with logits 损失
      - 真样本和假样本各计算一次损失，取平均

    参数:
        discriminator  : ActionDiscriminator 判别器模型
        real_actions   : [B, C, F, N, 1]  — 真实动作（来自数据集）
        fake_actions   : [B, C, F, N, 1]  — 生成动作（来自学生模型，需 detach）
        video_latent   : [B, C_v, F_v, H, W] — 视频潜在表示（条件输入）
        text_emb       : [B, L, D]        — 文本嵌入（条件输入）

    返回:
        d_loss: 判别器损失（标量）

    设计说明：
      - fake_actions 应该 detach 后传入，因为判别器训练不需要梯度回传到学生模型
      - 使用 BCE with logits 而非 sigmoid + BCE，数值更稳定
      - 也可以使用 hinge loss 或 WGAN loss，这里选择最经典的 BCE
    """
    # 对真实动作计算判别器得分
    real_logits = discriminator(real_actions, video_latent, text_emb)  # [B, 1]

    # 对生成动作计算判别器得分（detach 切断到学生模型的梯度）
    fake_logits = discriminator(fake_actions.detach(), video_latent, text_emb)  # [B, 1]

    # 真样本标签 = 1，假样本标签 = 0
    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)

    # BCE with logits 损失
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)

    # 总损失 = (真样本损失 + 假样本损失) / 2
    d_loss = (real_loss + fake_loss) / 2.0

    return d_loss


def compute_dmd_gradient(
    discriminator: ActionDiscriminator,
    fake_actions: torch.Tensor,
    video_latent: torch.Tensor,
    text_emb: torch.Tensor,
    teacher_score: float = 0.0,
) -> torch.Tensor:
    """
    计算 DMD（Distribution Matching Distillation）梯度。

    DMD 的核心公式：
        grad = normalize(D(fake) - teacher_score)

    其中：
        D(fake)        : 判别器对生成动作的打分
        teacher_score  : 教师模型的打分（作为基线/参考）
        normalize      : L2 归一化，防止梯度爆炸

    DMD 梯度的直觉理解：
        - 如果 D(fake) > teacher_score：
            判别器认为生成的动作比教师预期更好，
            梯度指向"继续往这个方向生成"。
        - 如果 D(fake) < teacher_score：
            判别器认为生成的动作不够好，
            梯度指向"需要改进生成质量"。
        - teacher_score 通常设为 0（中性基线），
          也可以用教师模型对真实样本的打分作为动态基线。

    这个梯度会被注入到学生模型的反向传播中，
    引导学生生成的动作分布向真实数据分布靠拢。

    参数:
        discriminator  : ActionDiscriminator 判别器模型
        fake_actions   : [B, C, F, N, 1]  — 生成动作（来自学生模型）
        video_latent   : [B, C_v, F_v, H, W] — 视频潜在表示
        text_emb       : [B, L, D]        — 文本嵌入
        teacher_score  : float — 教师基线分数（默认 0.0）

    返回:
        grad: [B, C, F, N, 1] — DMD 梯度，形状与 fake_actions 相同
    """
    # 计算判别器对生成动作的打分
    fake_logits = discriminator(fake_actions, video_latent, text_emb)  # [B, 1]

    # DMD 信号 = 判别器打分 - 教师基线
    dmd_signal = fake_logits - teacher_score  # [B, 1]

    # 将 DMD 信号反向传播到 fake_actions，得到梯度
    # torch.autograd.grad 计算 fake_actions 相对于 dmd_signal 的梯度
    grad = torch.autograd.grad(
        outputs=dmd_signal,
        inputs=fake_actions,
        grad_outputs=torch.ones_like(dmd_signal),
        create_graph=False,  # 不需要二阶导数
        retain_graph=False,
    )[0]  # [B, C, F, N, 1]

    # L2 归一化：防止梯度爆炸，稳定训练
    # 对每个样本独立归一化
    grad_flat = grad.reshape(grad.shape[0], -1)  # [B, C*F*N*1]
    grad_norm = grad_flat.norm(dim=1, keepdim=True).clamp(min=1e-6)  # [B, 1]
    grad = grad / grad_norm.reshape(-1, 1, 1, 1, 1)

    return grad
