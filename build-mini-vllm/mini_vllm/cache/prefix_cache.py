"""Prefix Caching —— 相同前缀共享 KV 物理块（Lesson 16 的 Build 目标）。

洞察：在 causal attention 里，位置 i 的 K/V 只依赖 token 0..i（前缀），与后面的 token
无关；而本课模型用**绝对位置**的 RoPE。所以两条请求只要 prompt 的前若干 token 完全
相同（且位置对齐，都从 0 开始），它们这段前缀的 K/V **逐值相同**——可以直接复用同一批
物理块，省掉重复的 prefill 计算。

实现要点：
    - 按 block_size 把 prompt 切成整块；对每个整块算一个**链式哈希**（包含其之前所有 token，
      即 parent hash 链），作为该前缀块的身份。
    - 命中：让新请求的 block table 直接引用已缓存的物理块（incref），并把 num_computed_tokens
      前移，跳过这段 prefill。
    - 未命中：正常 prefill；请求结束时把它的整块前缀**登记**进缓存（缓存持有一份引用，
      使块在请求结束后依然存活，供后续复用）。
    - 引用计数 + LRU 逐出：超出容量时逐出最久未用的缓存块（decref）。

**正确性**：命中不改变输出——与不开前缀缓存逐 token 一致（有测试保证）。
"""

from __future__ import annotations

from .block_allocator import BlockAllocator


class PrefixCache:
    def __init__(self, allocator: BlockAllocator, block_size: int, store: dict,
                 capacity_blocks: int = 128) -> None:
        self.alloc = allocator
        self.bs = block_size
        self.store = store
        self.capacity = capacity_blocks
        self.map: dict[int, int] = {}      # chained block hash -> physical block id
        self.order: list[int] = []         # LRU：越靠后越新
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    # ------------------------------------------------------------------ #
    def _block_hashes(self, token_ids: list[int]) -> list[int]:
        """对每个**整块**算链式 FNV 哈希（含其之前所有 token）。"""
        hashes: list[int] = []
        h = 1469598103934665603
        n_full = len(token_ids) // self.bs
        idx = 0
        for _ in range(n_full):
            for _ in range(self.bs):
                h ^= (token_ids[idx] + 1) & 0xFFFFFFFFFFFFFFFF
                h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
                idx += 1
            hashes.append(h)
        return hashes

    def _touch(self, h: int) -> None:
        if h in self.order:
            self.order.remove(h)
        self.order.append(h)

    # ------------------------------------------------------------------ #
    def attach(self, request) -> None:
        """尝试为请求复用已缓存的前缀块（在 prefill 之前调用）。"""
        kv = request.kv
        hashes = self._block_hashes(request.prompt_token_ids)
        shared = 0
        for h in hashes:
            if h in self.map:
                phys = self.map[h]
                self.alloc.incref(phys)            # 请求引用它
                kv.block_table.append(phys)
                self._touch(h)
                shared += 1
            else:
                break
        # 至少保留 1 个 prompt token 去跑 prefill，才能拿到「第一个输出 token」的 logits。
        prompt_len = len(request.prompt_token_ids)
        while shared > 0 and shared * self.bs >= prompt_len:
            shared -= 1
            self.alloc.free(kv.block_table.pop())  # 撤销最后一块的 incref
        if shared:
            kv.positions = list(range(shared * self.bs))
            request.num_computed_tokens = shared * self.bs
            self.hits += shared
        else:
            self.misses += 1

    def on_finish(self, request) -> None:
        """请求结束：登记其未缓存的整块前缀，然后释放请求对所有块的引用。"""
        kv = request.kv
        hashes = self._block_hashes(request.prompt_token_ids)
        for i, h in enumerate(hashes):
            if h not in self.map and i < len(kv.block_table):
                phys = kv.block_table[i]
                self.alloc.incref(phys)            # 缓存保留一份引用
                self.map[h] = phys
                self._touch(h)
        kv.free()                                  # 释放请求自己的引用（生成块归零回收）
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self.order) > self.capacity:
            h = self.order.pop(0)                   # 逐出最久未用
            phys = self.map.pop(h)
            self.alloc.free(phys)
            self.evictions += 1

    def flush(self) -> None:
        """引擎结束时释放缓存持有的全部块（保证无泄漏）。"""
        for h in list(self.order):
            self.alloc.free(self.map[h])
        self.map.clear()
        self.order.clear()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "cached_blocks": len(self.order), "evictions": self.evictions}
