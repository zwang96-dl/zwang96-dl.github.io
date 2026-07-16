"""KV Cache —— Lesson 7 的 Build 目标。

动机（Lesson 6 会先让你亲眼看到问题）：
    自回归解码时，生成第 t 个 token 需要它对**之前所有** token 的 attention。
    如果每步都把整段序列重新跑一遍 forward，前缀会被反复重算——越生成越慢（O(n²)）。

关键洞察：
    在 attention 里，历史 token 贡献的是它们的 **Key 和 Value**（被加权求和的对象）。
    而历史的 **Query 只用一次**（在它自己那一步）就不再需要。
    所以我们缓存每一层、每个历史 token 的 K 和 V；解码时只为**新** token 计算 Q/K/V，
    把新 K/V 追加进缓存，再对「新 Q × 全部缓存 K/V」做 attention。

这把每步的计算从「整段」降到「一个 token」，用**内存换计算**。

缓存内容 shape（本课教学版，按 token 行存储）：
    每层 K：list of rows，每行长度 = num_kv_heads · head_dim
    每层 V：同上
    positions：每个已缓存 token 的绝对位置（各层共享，因为是同一批 token）
"""

from __future__ import annotations

from ..config import ModelConfig
from ..model.matrix import Matrix


class KVCache:
    """逐层的 Key/Value 缓存。

    一个 :class:`KVCache` 对应**一条序列**的一次生成。prefill 时一次性追加整段 prompt
    的 K/V；之后每次 decode 追加 1 个 token 的 K/V。
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.num_layers = config.num_layers
        self.k: list[Matrix] = [[] for _ in range(self.num_layers)]
        self.v: list[Matrix] = [[] for _ in range(self.num_layers)]
        self.positions: list[int] = []

    @property
    def length(self) -> int:
        """已缓存的 token 数（= 上下文长度）。"""
        return len(self.positions)

    def add_positions(self, positions: list[int]) -> None:
        """登记新追加 token 的绝对位置（每次 forward 调用一次）。"""
        self.positions.extend(positions)

    def append(self, layer: int, k_rows: Matrix, v_rows: Matrix) -> None:
        """把某一层、新 token 的 K/V 行追加到缓存。

        约定：``k_rows`` / ``v_rows`` 的行数应等于本次新 token 的数量，
        且追加后各层长度应与 :attr:`positions` 一致（教学不变量）。
        """
        if len(k_rows) != len(v_rows):
            raise ValueError(f"K/V 行数不一致：{len(k_rows)} vs {len(v_rows)}")
        self.k[layer].extend([list(r) for r in k_rows])
        self.v[layer].extend([list(r) for r in v_rows])

    def get(self, layer: int) -> tuple[Matrix, Matrix]:
        """取某层的全部缓存 (K, V)。"""
        return self.k[layer], self.v[layer]

    def check_consistent(self) -> None:
        """自检：每层缓存长度应等于 positions 长度（无写漏/写重）。"""
        for li in range(self.num_layers):
            if len(self.k[li]) != self.length or len(self.v[li]) != self.length:
                raise AssertionError(
                    f"KV Cache 不一致：layer {li} 有 {len(self.k[li])} 行，"
                    f"但 positions 有 {self.length} 个。检查 forward 里 append 的时机。"
                )

    def memory_estimate(self) -> dict:
        """估算缓存占用（教学用）：元素个数与「每 token 每层」的公式。

        每 token 每层需存：2（K 和 V）× num_kv_heads × head_dim 个数。
        """
        cfg = self.config
        per_token_per_layer = 2 * cfg.num_kv_heads * cfg.head_dim
        total_elems = per_token_per_layer * cfg.num_layers * self.length
        return {
            "per_token_per_layer": per_token_per_layer,
            "num_layers": cfg.num_layers,
            "tokens": self.length,
            "total_elements": total_elems,
            "formula": "2 · num_kv_heads · head_dim · num_layers · seq_len",
        }
