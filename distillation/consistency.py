"""
一致性函数的边界条件缩放（Karras boundary conditions）。

这个缩放确保一致性函数 f(x, σ) 满足边界条件：
  - 当 σ → 0 时，f(x, 0) = x（即从零噪声出发，输出等于输入）
  - 这是 LCM 蒸馏的数学基础：模型学习一个函数，使得从任意噪声水平
    出发都能一步预测到干净样本

数学原理：
  c_skip(σ) = σ_data² / (σ² + σ_data²)    # 跳跃连接系数
  c_out(σ)  = σ·σ_data / √(σ² + σ_data²)  # 网络输出系数

  一致性函数：f(x_σ, σ) = c_skip(σ)·x_σ + c_out(σ)·F(x_σ, σ)

  其中 F 是神经网络的预测（pred_x0），当 σ=0 时：
    c_skip(0) = 1, c_out(0) = 0  →  f(x, 0) = x ✓
"""


def scalings_for_boundary_conditions(sigma, sigma_data=0.5):
    """
    计算一致性边界条件的缩放系数。

    参数:
        sigma: 当前的噪声水平（σ），可以是标量或任意形状的张量
        sigma_data: 数据的噪声水平（默认 0.5），控制缩放的平衡点

    返回:
        c_skip: 跳跃连接系数，σ→0 时趋向 1（保留输入）
        c_out:  网络输出系数，σ→0 时趋向 0（忽略网络输出）

    直觉理解：
        - σ 大（高噪声）时：c_skip 小，c_out 大 → 主要依赖网络预测
        - σ 小（低噪声）时：c_skip 大，c_out 小 → 主要保留输入
        - σ=0 时：c_skip=1, c_out=0 → 完全保留输入（一致性条件）
    """
    c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
    c_out = sigma * sigma_data / (sigma ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out
