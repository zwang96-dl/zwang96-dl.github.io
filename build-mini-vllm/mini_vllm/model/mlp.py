"""SwiGLU MLP（前馈网络）—— Lesson 3。

Transformer 每个 block 里，attention 之后是一个逐位置的前馈网络。现代 LLM 常用
SwiGLU：两条并行的线性投影，一条过 SiLU 激活当「门」，逐元素相乘后再投影回来。

    gate = x · W_gate            # (seq, intermediate)
    up   = x · W_up              # (seq, intermediate)
    h    = silu(gate) ⊙ up       # 逐元素相乘（门控）
    out  = h · W_down            # (seq, hidden)

其中 silu(x) = x · sigmoid(x)。intermediate 通常是 hidden 的 2~4 倍。
"""

from __future__ import annotations

import math

from .matrix import Matrix, matmul


def silu(x: float) -> float:
    """SiLU / swish：x * sigmoid(x)，数值稳定写法（避免 exp 溢出）。"""
    if x >= 0:
        return x / (1.0 + math.exp(-x))
    e = math.exp(x)
    return x * e / (1.0 + e)


def swiglu(x_rows: Matrix, w_gate: Matrix, w_up: Matrix, w_down: Matrix) -> Matrix:
    """SwiGLU 前馈：输入 (seq, hidden)，输出 (seq, hidden)。"""
    gate = matmul(x_rows, w_gate)   # (seq, intermediate)
    up = matmul(x_rows, w_up)       # (seq, intermediate)
    inter = len(gate[0]) if gate else 0
    h: Matrix = [
        [silu(gate[i][j]) * up[i][j] for j in range(inter)]
        for i in range(len(gate))
    ]
    return matmul(h, w_down)        # (seq, hidden)
