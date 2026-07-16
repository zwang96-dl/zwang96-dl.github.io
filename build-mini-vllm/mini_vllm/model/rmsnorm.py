"""RMSNorm（Root Mean Square Layer Normalization）—— Lesson 3。

为什么需要归一化？
    深层网络里，每一层的输出规模会漂移，导致训练不稳、数值爆炸。归一化把每个
    token 的向量拉回一个稳定的尺度。

RMSNorm 比 LayerNorm 更简单：不减均值、不加 bias，只按「均方根」缩放，再乘一个
可学习的权重向量：

    rms(x) = sqrt(mean(x_i^2) + eps)
    y_i    = x_i / rms(x) * weight_i

输入/输出 shape 都是 (seq, hidden)：逐 token（逐行）独立归一化。
"""

from __future__ import annotations

import math

from .matrix import Matrix, Vector


def rms_norm(x_rows: Matrix, weight: Vector, eps: float = 1e-5) -> Matrix:
    """对 (seq, hidden) 的每一行做 RMSNorm。"""
    if not x_rows:
        return []
    hidden = len(x_rows[0])
    if len(weight) != hidden:
        raise ValueError(f"RMSNorm 权重长度 {len(weight)} 应等于 hidden {hidden}")
    out: Matrix = []
    for row in x_rows:
        ms = sum(v * v for v in row) / hidden          # 均方（mean square）
        inv = 1.0 / math.sqrt(ms + eps)                # 1 / rms
        out.append([row[j] * inv * weight[j] for j in range(hidden)])
    return out
