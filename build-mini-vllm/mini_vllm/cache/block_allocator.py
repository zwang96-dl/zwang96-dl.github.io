"""BlockAllocator —— 物理 KV block 的分配器（Lesson 13 的 Build 目标）。

把 KV 显存切成固定大小的 **physical block**（每块容纳 block_size 个 token 的 KV），
用一个 free list 管理。这样：
    - 一条序列按需增长，逐块申请，不必一上来就为 max_seq_len 预留（消除 Lesson 12 的浪费）；
    - block 可被复用；配合引用计数还能被多条序列**共享**（前缀缓存，Lesson 16）。

引用计数（refcount）：
    allocate() 返回一个新 block，refcount=1。
    incref(b)  让另一条序列也引用它（共享），refcount+1。
    free(b)    refcount-1；归零才真正回收进 free list。
    重复 free 到负数 → double free（教学错误，主动报错）。

用 heap 管理 free list，总是分配**编号最小**的空闲块——确定、可读、易 Trace。
"""

from __future__ import annotations

import heapq


class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))
        heapq.heapify(self._free)
        self.ref: list[int] = [0] * num_blocks

    # ------------------------------------------------------------------ #
    def allocate(self) -> int:
        """分配一个新物理块（refcount 置 1）。无空闲块时抛 MemoryError（模拟 OOM）。"""
        if not self._free:
            raise MemoryError(
                f"KV blocks 耗尽（OOM）：{self.num_blocks} 块全部占用。"
                "真实系统会触发抢占/换出（本课在调度层处理）。"
            )
        b = heapq.heappop(self._free)
        self.ref[b] = 1
        return b

    def incref(self, block_id: int) -> None:
        """增加一个引用（用于前缀共享）。"""
        if self.ref[block_id] <= 0:
            raise RuntimeError(f"incref 了一个未分配的块 {block_id}")
        self.ref[block_id] += 1

    def free(self, block_id: int) -> None:
        """释放一个引用；refcount 归零才回收。"""
        if self.ref[block_id] <= 0:
            raise RuntimeError(
                f"double free：块 {block_id} 已是空闲状态，重复释放是典型的资源管理错误。"
            )
        self.ref[block_id] -= 1
        if self.ref[block_id] == 0:
            heapq.heappush(self._free, block_id)

    # ------------------------------------------------------------------ #
    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def blocks_needed(self, num_tokens: int) -> int:
        """容纳 num_tokens 个 token 需要多少块。"""
        import math
        return max(1, math.ceil(num_tokens / self.block_size)) if num_tokens > 0 else 0

    def check_no_leak(self) -> None:
        """自检：所有块 refcount 应为 0（无泄漏）。用于实验/测试收尾。"""
        leaked = [b for b in range(self.num_blocks) if self.ref[b] != 0]
        if leaked:
            raise AssertionError(f"KV block 泄漏：块 {leaked} 的 refcount 未归零。")

    def snapshot(self) -> dict:
        return {"num_blocks": self.num_blocks, "block_size": self.block_size,
                "free": self.num_free, "used": self.num_used,
                "refcounts": list(self.ref)}
