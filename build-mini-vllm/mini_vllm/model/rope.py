"""RoPE（Rotary Position Embedding，旋转位置编码）—— Lesson 3。

Attention 本身对「顺序」是无感的（打乱 token 顺序，点积不变）。RoPE 通过把每个
位置的 Q/K 向量按位置相关的角度「旋转」，把位置信息注入进去——而且旋转后的点积
只依赖两个位置的**相对距离**，这正是我们想要的。

做法：把 head 维两两配对 (2i, 2i+1)，第 i 对以角度 θ = pos · freq_i 旋转：

    freq_i = theta ^ (-2i / head_dim)
    x'[2i]   = x[2i]·cosθ - x[2i+1]·sinθ
    x'[2i+1] = x[2i]·sinθ + x[2i+1]·cosθ

pos 越大、i 越小，旋转越快——低维捕捉近距离，高维捕捉远距离。
"""

from __future__ import annotations

import math

from .matrix import Matrix, Vector


def rope_freqs(head_dim: int, theta: float = 10000.0) -> list[float]:
    """返回 head_dim/2 个旋转频率（每对一个）。"""
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE 要求 head_dim 为偶数，收到 {head_dim}")
    return [theta ** (-(2.0 * i) / head_dim) for i in range(head_dim // 2)]


def apply_rope_vec(vec: Vector, pos: int, freqs: list[float]) -> Vector:
    """对单个 head 向量施加位置 pos 的旋转。"""
    out = list(vec)
    for i, f in enumerate(freqs):
        ang = pos * f
        c, s = math.cos(ang), math.sin(ang)
        a, b = vec[2 * i], vec[2 * i + 1]
        out[2 * i] = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out


def apply_rope_heads(x: Matrix, positions: list[int], num_heads: int,
                     head_dim: int, freqs: list[float]) -> Matrix:
    """对 (seq, num_heads*head_dim) 的每行、每个 head 分别施加 RoPE。

    第 i 行对应绝对位置 positions[i]；每行按 head 切成若干 head_dim 段分别旋转。
    """
    out: Matrix = []
    for row_idx, row in enumerate(x):
        pos = positions[row_idx]
        new_row: Vector = []
        for h in range(num_heads):
            seg = row[h * head_dim:(h + 1) * head_dim]
            new_row.extend(apply_rope_vec(seg, pos, freqs))
        out.append(new_row)
    return out
