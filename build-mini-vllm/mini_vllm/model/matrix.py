"""纯 Python 的张量 / 矩阵工具（Lesson 2 的 Build 目标，Lesson 4 会复用）。

为什么不用 numpy？
------------------
本课程核心坚持**零第三方依赖**：用嵌套 list 表示张量，用最朴素的三重循环实现
矩阵乘法。这样每一步「行 × 列、逐元素乘、再求和」都摊在明面上，可手算、可对照。
真实系统会用高度优化的 kernel（BLAS / Metal / CUDA），但**规则完全一样**。

张量表示
--------
    标量 scalar : 3.0
    向量 vector : [1.0, 2.0, 3.0]              shape = (3,)
    矩阵 matrix : [[1, 2, 3], [4, 5, 6]]       shape = (2, 3)
    3D 张量     : [[[...]], [[...]]]            shape = (batch, seq, hidden)

约定：矩阵是「行优先」的 list of rows，即 ``a[i][j]`` 是第 i 行第 j 列。
"""

from __future__ import annotations

import math
from typing import Sequence

Number = float | int
Vector = list[float]
Matrix = list[list[float]]


# --------------------------------------------------------------------------- #
# 形状（shape）
# --------------------------------------------------------------------------- #
def shape(t) -> tuple[int, ...]:
    """返回嵌套 list 的形状，并顺带校验它是规整（rectangular）的。

    例：``shape([[1,2,3],[4,5,6]]) == (2, 3)``。
    若某一维长度不一致（锯齿数组），抛 ValueError——这类错误在张量计算里
    往往是最难查的 bug，提前暴露它。
    """
    if not isinstance(t, list):
        return ()
    n = len(t)
    if n == 0:
        return (0,)
    sub = shape(t[0])
    for i, row in enumerate(t):
        if shape(row) != sub:
            raise ValueError(
                f"张量形状不规整：第 0 个元素的 shape 是 {sub}，"
                f"但第 {i} 个元素的 shape 是 {shape(row)}。"
            )
    return (n, *sub)


def numel(t) -> int:
    s = shape(t)
    out = 1
    for d in s:
        out *= d
    return out


# --------------------------------------------------------------------------- #
# 基础构造
# --------------------------------------------------------------------------- #
def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def transpose(a: Matrix) -> Matrix:
    """二维转置：``(m, n) -> (n, m)``，即 ``aT[j][i] = a[i][j]``。"""
    sa = shape(a)
    if len(sa) != 2:
        raise ValueError(f"transpose 只支持二维矩阵，收到 shape={sa}")
    m, n = sa
    return [[a[i][j] for i in range(m)] for j in range(n)]


# --------------------------------------------------------------------------- #
# 乘法：row-column rule
# --------------------------------------------------------------------------- #
def dot(u: Sequence[float], v: Sequence[float]) -> float:
    """向量点积 ``sum_k u[k] * v[k]``（矩阵乘法的最小单元）。"""
    if len(u) != len(v):
        raise ValueError(f"点积要求等长：len(u)={len(u)} != len(v)={len(v)}")
    return float(sum(u[k] * v[k] for k in range(len(u))))


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """二维矩阵乘法 ``(m, k) @ (k, n) -> (m, n)``。

    规则（row-column rule）：``out[i][j] = Σ_k a[i][k] * b[k][j]``——
    结果第 i 行第 j 列 = A 的第 i 行 与 B 的第 j 列 的点积。

    形状必须相容：A 的列数（k）要等于 B 的行数（k），否则抛出带具体数字的错误。
    """
    sa, sb = shape(a), shape(b)
    if len(sa) != 2 or len(sb) != 2:
        raise ValueError(f"matmul 需要两个二维矩阵，收到 {sa} 与 {sb}")
    m, ka = sa
    kb, n = sb
    if ka != kb:
        raise ValueError(
            f"形状不相容：A 是 {sa}，B 是 {sb}；"
            f"要求 A 的列数({ka}) == B 的行数({kb})。"
            f"（记法：(m,k)@(k,n)->(m,n)）"
        )
    out = zeros(m, n)
    for i in range(m):
        ai = a[i]
        for j in range(n):
            s = 0.0
            for k in range(ka):
                s += ai[k] * b[k][j]
            out[i][j] = s
    return out


def matvec(a: Matrix, v: Vector) -> Vector:
    """矩阵 × 向量 ``(m, k) @ (k,) -> (m,)``。"""
    sa = shape(a)
    if len(sa) != 2:
        raise ValueError(f"matvec 需要二维矩阵，收到 {sa}")
    if sa[1] != len(v):
        raise ValueError(f"形状不相容：A 是 {sa}，v 长度 {len(v)}；要求 A 列数 == len(v)。")
    return [dot(row, v) for row in a]


def batched_matmul(a, b):
    """三维批量矩阵乘法 ``(B, m, k) @ (B, k, n) -> (B, m, n)``。

    对每个 batch 分别做二维 matmul。这是 attention 里「每个 head / 每个样本
    各算一份」的形态。
    """
    sa, sb = shape(a), shape(b)
    if len(sa) != 3 or len(sb) != 3:
        raise ValueError(f"batched_matmul 需要两个三维张量，收到 {sa} 与 {sb}")
    if sa[0] != sb[0]:
        raise ValueError(f"batch 维不一致：{sa[0]} != {sb[0]}")
    return [matmul(a[i], b[i]) for i in range(sa[0])]


# --------------------------------------------------------------------------- #
# 逐元素 & 广播（broadcasting）
# --------------------------------------------------------------------------- #
def scale(a: Matrix, s: float) -> Matrix:
    """逐元素数乘 ``a * s``。"""
    return [[x * s for x in row] for row in a]


def add(a: Matrix, b: Matrix) -> Matrix:
    """逐元素相加，要求 shape 完全一致。"""
    if shape(a) != shape(b):
        raise ValueError(f"逐元素相加要求同形：{shape(a)} != {shape(b)}")
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def add_row_bias(a: Matrix, bias: Vector) -> Matrix:
    """行广播：给 ``(m, n)`` 的每一行都加上长度为 ``n`` 的 bias 向量。

    这是最常见、最必要的 broadcasting：``(m, n) + (n,) -> (m, n)``。
    """
    sa = shape(a)
    if len(sa) != 2 or sa[1] != len(bias):
        raise ValueError(f"行广播要求 A 是 (m,n) 且 len(bias)=n；A={sa}, len(bias)={len(bias)}")
    return [[a[i][j] + bias[j] for j in range(sa[1])] for i in range(sa[0])]


def isclose_matrix(a: Matrix, b: Matrix, tol: float = 1e-9) -> bool:
    """比较两个矩阵是否逐元素接近（测试用）。"""
    if shape(a) != shape(b):
        return False
    for i in range(len(a)):
        for j in range(len(a[0])):
            if abs(a[i][j] - b[i][j]) > tol:
                return False
    return True


def max_abs_diff(a: Matrix, b: Matrix) -> float:
    """返回两个同形矩阵的最大逐元素绝对误差（用于对齐验证的证据输出）。"""
    if shape(a) != shape(b):
        raise ValueError(f"比较要求同形：{shape(a)} != {shape(b)}")
    d = 0.0
    for i in range(len(a)):
        for j in range(len(a[0])):
            d = max(d, abs(a[i][j] - b[i][j]))
    return d


def pretty(a: Matrix, width: int = 8, prec: int = 3) -> str:
    """把矩阵格式化成对齐的字符串（Trace / Inspector 用）。"""
    def fmt(x: float) -> str:
        return f"{x:>{width}.{prec}f}"
    return "\n".join("[" + " ".join(fmt(x) for x in row) + "]" for row in a)
