"""Block Table 与 Paged KV Cache（Lesson 14 的 Build 目标）。

Lesson 7 的 KVCache 把每层 K/V 存成**一整段连续**的行。问题（Lesson 12）：一条序列
要多长、显存就得连续预留多长，碎片和浪费严重。

分页（paging）借鉴操作系统虚拟内存的思路：
    - 把 KV 切成固定大小的 **physical block**（由 BlockAllocator 管理）；
    - 每条序列维护一张 **block table**：logical block i → physical block id；
    - token 的逻辑位置 pos → (logical_block = pos // block_size, offset = pos % block_size)；
    - **逻辑上连续，物理上可以分散**——不同序列的块在物理上交错也没关系。

本文件的 PagedKVCache 实现了与 :class:`KVCache` **完全相同的接口**
（add_positions / append / get / positions / length / check_consistent），
因此可以直接塞进 ``model.forward(..., kv_cache=paged)``。

第一版策略（教学）：attention 前先把分散的块 **gather** 成逻辑顺序的连续 K/V，
再走普通 attention——所以 **paged 的输出与连续版逐值一致**（可测试）。
真实 vLLM 用 PagedAttention kernel 直接在分页布局上算，省去 gather。
"""

from __future__ import annotations

from ..config import ModelConfig
from .block_allocator import BlockAllocator
from ..model.matrix import Matrix


class PagedKVCache:
    """一条序列的分页 KV 缓存。物理块从共享的 :class:`BlockAllocator` 借用。"""

    def __init__(self, config: ModelConfig, allocator: BlockAllocator,
                 store: dict | None = None) -> None:
        self.cfg = config
        self.alloc = allocator
        self.bs = allocator.block_size
        self.num_layers = config.num_layers
        self.block_table: list[int] = []      # logical block -> physical block id
        self.positions: list[int] = []
        # 物理存储：(layer, physical_block) -> 长度 block_size 的行数组（未写为 None）。
        # 可传入**共享** store，让不同序列复用同一物理块的 KV（前缀缓存，Lesson 16）。
        if store is None:
            store = {"k": {}, "v": {}}
        self.store = store
        self.kstore: dict[tuple[int, int], list] = store["k"]
        self.vstore: dict[tuple[int, int], list] = store["v"]

    @property
    def length(self) -> int:
        return len(self.positions)

    def _ensure_capacity(self) -> None:
        """按当前 token 数确保 block table 足够长，不够就分配新物理块。"""
        need = max(1, (self.length + self.bs - 1) // self.bs) if self.length else 0
        while len(self.block_table) < need:
            phys = self.alloc.allocate()
            self.block_table.append(phys)
            for li in range(self.num_layers):
                # 若该物理块已在共享 store 中（前缀复用），保留其 KV，不清空。
                if (li, phys) not in self.kstore:
                    self.kstore[(li, phys)] = [None] * self.bs
                    self.vstore[(li, phys)] = [None] * self.bs

    def add_positions(self, positions: list[int]) -> None:
        self.positions.extend(positions)
        self._ensure_capacity()

    def append(self, layer: int, k_rows: Matrix, v_rows: Matrix) -> None:
        if len(k_rows) != len(v_rows):
            raise ValueError("K/V 行数不一致")
        start = self.length - len(k_rows)   # 新 token 从这里开始
        for i in range(len(k_rows)):
            pos = start + i
            phys = self.block_table[pos // self.bs]
            off = pos % self.bs
            self.kstore[(layer, phys)][off] = list(k_rows[i])
            self.vstore[(layer, phys)][off] = list(v_rows[i])

    def get(self, layer: int) -> tuple[Matrix, Matrix]:
        """把分散在各物理块里的 K/V 按逻辑顺序 gather 成连续行。"""
        K: Matrix = []
        V: Matrix = []
        for pos in range(self.length):
            phys = self.block_table[pos // self.bs]
            off = pos % self.bs
            K.append(self.kstore[(layer, phys)][off])
            V.append(self.vstore[(layer, phys)][off])
        return K, V

    def check_consistent(self) -> None:
        for li in range(self.num_layers):
            k, _ = self.get(li)
            if len(k) != self.length or any(r is None for r in k):
                raise AssertionError(f"Paged KV 不一致：layer {li} 有空槽或长度不符。")

    def free(self) -> None:
        """释放本序列占用的物理块（每个物理块释放一次引用）。"""
        for phys in self.block_table:
            self.alloc.free(phys)
        self.block_table = []

    def logical_to_physical(self) -> list[dict]:
        """返回 block table 的可读映射（Trace / 可视化用）。"""
        return [{"logical": i, "physical": p} for i, p in enumerate(self.block_table)]
