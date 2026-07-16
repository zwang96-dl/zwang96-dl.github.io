"""从 logits 采样下一个 token —— Lesson 5。

模型 forward 给出的是 **logits**：每个词表 token 的「原始分数」（未归一）。怎么从中
挑一个 token，决定了生成的确定性与多样性：

    greedy      —— 直接取分数最高的（temperature=0）。完全确定。
    temperature —— logits / T 再 softmax。T 越大越随机，越小越保守。
    top-k       —— 只在分数最高的 k 个里采样。
    top-p       —— 只在累计概率达到 p 的最小集合（nucleus）里采样。

本模块用一个确定性 RNG，保证给定 seed 结果可复现。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class _Rng:
    """确定性 splitmix64（与模型初始化用的同款，保证可复现）。"""

    def __init__(self, seed: int) -> None:
        self.state = (seed or 0x1234) & 0xFFFFFFFFFFFFFFFF

    def random(self) -> float:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        return z / 2.0 ** 64


@dataclass
class SamplingParams:
    temperature: float = 0.0   # 0 = greedy
    top_k: int = 0             # 0 = 关闭
    top_p: float = 1.0         # 1.0 = 关闭
    seed: int = 12345


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    ex = [math.exp(x - m) for x in logits]
    s = sum(ex)
    return [e / s for e in ex]


def apply_temperature(logits: list[float], t: float) -> list[float]:
    if t <= 0:
        return list(logits)
    return [x / t for x in logits]


def top_k_filter(probs: list[float], k: int) -> list[float]:
    """只保留概率最大的 k 个，其余置 0（不重新归一，交给采样时归一）。"""
    if k <= 0 or k >= len(probs):
        return list(probs)
    threshold_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    keep = set(threshold_idx)
    return [probs[i] if i in keep else 0.0 for i in range(len(probs))]


def top_p_filter(probs: list[float], p: float) -> list[float]:
    """nucleus：保留按概率降序累计首次达到 p 的最小集合，其余置 0。"""
    if p >= 1.0:
        return list(probs)
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    cum, keep = 0.0, set()
    for i in order:
        keep.add(i)
        cum += probs[i]
        if cum >= p:
            break
    return [probs[i] if i in keep else 0.0 for i in range(len(probs))]


def argmax(logits: list[float]) -> int:
    best, bi = logits[0], 0
    for i in range(1, len(logits)):
        if logits[i] > best:
            best, bi = logits[i], i
    return bi


class Sampler:
    """把 :class:`SamplingParams` 变成一个 ``logits -> token_id`` 的可调用对象。"""

    def __init__(self, params: SamplingParams | None = None) -> None:
        self.params = params or SamplingParams()
        self.rng = _Rng(self.params.seed)

    def __call__(self, logits: list[float]) -> int:
        p = self.params
        if p.temperature <= 0:                       # greedy
            return argmax(logits)
        probs = softmax(apply_temperature(logits, p.temperature))
        probs = top_k_filter(probs, p.top_k)
        probs = top_p_filter(probs, p.top_p)
        total = sum(probs)
        if total <= 0:                               # 兜底：退化为 greedy
            return argmax(logits)
        # 轮盘赌采样
        r = self.rng.random() * total
        acc = 0.0
        for i, pr in enumerate(probs):
            acc += pr
            if r <= acc:
                return i
        return len(probs) - 1
