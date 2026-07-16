"""Lesson 14 检查：Paged KV Cache 与连续 KV 逐值一致，block table 正确，无泄漏。"""
import unittest
from mini_vllm.config import ModelConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.model import matrix as M
from mini_vllm.cache.kv_cache import KVCache
from mini_vllm.cache.block_allocator import BlockAllocator
from mini_vllm.cache.block_table import PagedKVCache


class TestPagedKV(unittest.TestCase):
    def setUp(self):
        self.model = TinyTextModel(ModelConfig())
        self.ids = [256, 72, 101, 108, 108, 111]

    def test_paged_matches_contiguous_prefill(self):
        cont = self.model.forward(self.ids, list(range(len(self.ids))), KVCache(self.model.cfg))
        alloc = BlockAllocator(num_blocks=32, block_size=4)
        paged = self.model.forward(self.ids, list(range(len(self.ids))),
                                   PagedKVCache(self.model.cfg, alloc))
        self.assertEqual(M.max_abs_diff(cont, paged), 0.0,
                         msg="分页 KV 的 prefill 输出必须与连续 KV 逐值一致。")

    def test_block_table_mapping(self):
        alloc = BlockAllocator(num_blocks=32, block_size=4)
        kv = PagedKVCache(self.model.cfg, alloc)
        self.model.forward(self.ids, list(range(len(self.ids))), kv)
        self.assertEqual(len(kv.block_table), 2,
                         msg="6 tokens、block_size=4 → 2 个逻辑块。")

    def test_physically_noncontiguous_still_matches_contiguous(self):
        # 交错释放物理块 → 序列拿到分散的块 → block table 非连续；但输出仍与连续 KV 一致。
        cont = self.model.forward(self.ids, list(range(len(self.ids))), KVCache(self.model.cfg))
        alloc = BlockAllocator(num_blocks=64, block_size=4)
        held = [alloc.allocate() for _ in range(8)]
        for h in held[1::2]:          # 释放 1,3,5,7 制造交错空洞
            alloc.free(h)
        kv = PagedKVCache(self.model.cfg, alloc)
        paged = self.model.forward(self.ids, list(range(len(self.ids))), kv)
        bt = list(kv.block_table)
        noncontig = any(bt[i + 1] - bt[i] != 1 for i in range(len(bt) - 1))
        self.assertTrue(noncontig,
                        msg=f"block table 应物理不连续（Paged 的核心卖点），实际 {bt}。")
        self.assertEqual(M.max_abs_diff(cont, paged), 0.0,
                         msg="物理不连续时，gather 后输出仍必须与连续 KV 逐值一致。")
        kv.free()
        for h in held[0::2]:
            alloc.free(h)
        alloc.check_no_leak()

    def test_decode_matches_and_no_leak(self):
        from mini_vllm.sampling import argmax
        alloc = BlockAllocator(num_blocks=32, block_size=4)
        kv = PagedKVCache(self.model.cfg, alloc)
        cont_kv = KVCache(self.model.cfg)
        c = self.model.forward(self.ids, list(range(len(self.ids))), cont_kv)
        p = self.model.forward(self.ids, list(range(len(self.ids))), kv)
        nxt = argmax(c[-1])
        dc = self.model.forward([nxt], [len(self.ids)], cont_kv)
        dp = self.model.forward([nxt], [len(self.ids)], kv)
        self.assertEqual(M.max_abs_diff([dc[0]], [dp[0]]), 0.0,
                         msg="分页 KV 的 decode 输出也必须与连续一致。")
        kv.free()
        alloc.check_no_leak()


if __name__ == "__main__":
    unittest.main()
