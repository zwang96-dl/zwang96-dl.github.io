"""参考实现：单头 causal scaled-dot-product attention（Lesson 4 的 Build 目标）。

这是全课程最核心的一段数学。用纯 Python + matrix.py 一步步写出来，让每个中间量
都能被打印、被手算核对：

    scores  = Q · Kᵀ                 # (Tq, Tk)  每个 query 与每个 key 的相似度
    scaled  = scores / sqrt(d)        # 缩放，防止点积随维度变大而过大
    masked  = 加 causal mask          # 未来位置置为 -inf（自回归不能偷看未来）
    weights = softmax(masked, 轴=行)  # 每个 query 的注意力分布，行和为 1
    out     = weights · V             # (Tq, dv)  用权重对 value 加权求和

符号与 shape（务必记住）：
    Q: (Tq, d)   —— query，每行是一个位置的查询向量
    K: (Tk, d)   —— key
    V: (Tk, dv)  —— value
    d            —— head 维度（Q/K 的列数），缩放因子是 1/sqrt(d)
    out: (Tq, dv)
"""

from __future__ import annotations

import math

from . import matrix as M

NEG_INF = float("-inf")


def softmax(row: list[float]) -> list[float]:
    """数值稳定的 softmax：先减去最大值再取 exp，避免溢出。

    ``softmax(x)_i = exp(x_i - max) / Σ_j exp(x_j - max)``，结果非负且和为 1。
    对 ``-inf``（被 mask 的位置）取 exp 得 0——这正是「屏蔽未来」的效果。
    """
    m = max(row)
    if m == NEG_INF:  # 整行都被 mask（不该发生），返回全 0 避免 0/0
        return [0.0 for _ in row]
    exps = [math.exp(x - m) if x != NEG_INF else 0.0 for x in row]
    s = sum(exps)
    return [e / s for e in exps]


def softmax_rows(a: M.Matrix) -> M.Matrix:
    """对矩阵逐行做 softmax。"""
    return [softmax(row) for row in a]


def causal_mask_apply(scores: M.Matrix) -> M.Matrix:
    """给方阵 scores 加 causal mask：位置 (i, j) 中 j > i 的置为 -inf。

    含义：第 i 个 query 只能看到 key 0..i（当前及之前），不能看未来。
    要求 Tq == Tk（prefill 时 query 与 key 对齐）。
    """
    s = M.shape(scores)
    if len(s) != 2:
        raise ValueError(f"causal mask 需要二维 scores，收到 {s}")
    tq, tk = s
    out = [row[:] for row in scores]
    for i in range(tq):
        for j in range(tk):
            if j > i:
                out[i][j] = NEG_INF
    return out


def scaled_dot_product_attention(
    Q: M.Matrix,
    K: M.Matrix,
    V: M.Matrix,
    causal: bool = True,
    return_stages: bool = False,
):
    """单头缩放点积注意力。

    参数
    ----
    Q: (Tq, d), K: (Tk, d), V: (Tk, dv)
    causal: 是否加 causal mask（自回归解码必须 True）
    return_stages: True 时额外返回每个中间量，供教学 / Trace / 可视化。

    返回
    ----
    ``out``: (Tq, dv)。若 ``return_stages`` 则返回 ``(out, stages_dict)``。
    """
    sq, sk, sv = M.shape(Q), M.shape(K), M.shape(V)
    if len(sq) != 2 or len(sk) != 2 or len(sv) != 2:
        raise ValueError(f"Q/K/V 需为二维，收到 {sq}, {sk}, {sv}")
    tq, d = sq
    tk, dk = sk
    tv, dv = sv
    if d != dk:
        raise ValueError(f"Q 与 K 的 head 维必须相同：d={d} != {dk}")
    if tk != tv:
        raise ValueError(f"K 与 V 的序列长度必须相同：Tk={tk} != Tv={tv}")
    if causal and tq != tk:
        raise ValueError(
            f"causal attention 要求 Tq==Tk（query 与 key 对齐），收到 {tq} 与 {tk}"
        )

    scores = M.matmul(Q, M.transpose(K))        # (Tq, Tk)
    inv = 1.0 / math.sqrt(d)
    scaled = M.scale(scores, inv)               # 缩放
    masked = causal_mask_apply(scaled) if causal else scaled
    weights = softmax_rows(masked)              # 逐行 softmax → 注意力分布
    out = M.matmul(weights, V)                  # (Tq, dv)

    if return_stages:
        return out, {
            "Q": Q, "K": K, "V": V,
            "scores": scores,
            "scale": inv,
            "scaled": scaled,
            "masked": masked,
            "weights": weights,
            "out": out,
            "shapes": {
                "Q": sq, "K": sk, "V": sv,
                "scores": M.shape(scores), "weights": M.shape(weights), "out": M.shape(out),
            },
        }
    return out


def sdpa_positions(
    Q: M.Matrix,
    K: M.Matrix,
    V: M.Matrix,
    q_pos: list[int],
    k_pos: list[int],
) -> M.Matrix:
    """按**绝对位置**做 causal attention，支持 KV Cache 的解码场景。

    与 :func:`scaled_dot_product_attention` 的区别：这里 query 与 key 的数量可以不同
    （decode 时 Tq=1，而 Tk=已缓存的全部长度）。是否可见由位置决定：

        query（绝对位置 q_pos[i]）可以看 key（绝对位置 k_pos[j]）当且仅当 k_pos[j] <= q_pos[i]。

    这样 prefill（q_pos=k_pos=[0..T-1]）就是普通 causal；decode（q_pos=[t], k_pos=[0..t]）
    则让新 token 看到全部历史——**结果与「重算整段」的 naive attention 完全一致**。
    """
    tq, d = M.shape(Q)
    tk = M.shape(K)[0]
    inv = 1.0 / math.sqrt(d)
    out: M.Matrix = []
    for i in range(tq):
        # 逐 query 计算与所有可见 key 的分数
        row_scores = []
        for j in range(tk):
            if k_pos[j] > q_pos[i]:
                row_scores.append(NEG_INF)
            else:
                s = 0.0
                for t in range(d):
                    s += Q[i][t] * K[j][t]
                row_scores.append(s * inv)
        w = softmax(row_scores)
        # 加权求和 value
        dv = len(V[0])
        acc = [0.0] * dv
        for j in range(tk):
            wj = w[j]
            if wj == 0.0:
                continue
            vj = V[j]
            for t in range(dv):
                acc[t] += wj * vj[t]
        out.append(acc)
    return out
