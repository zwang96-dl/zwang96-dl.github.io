"""Lesson 13 检查：BlockAllocator 的分配/释放/引用计数/OOM/double-free/leak。"""
import unittest
from mini_vllm.cache.block_allocator import BlockAllocator


class TestBlockAllocator(unittest.TestCase):
    def test_allocate_returns_lowest_and_tracks_free(self):
        a = BlockAllocator(num_blocks=3)
        self.assertEqual(a.allocate(), 0, msg="应分配编号最小的空闲块（确定可读）。")
        self.assertEqual(a.allocate(), 1)
        self.assertEqual(a.num_free, 1)
        self.assertEqual(a.num_used, 2)

    def test_free_recycles(self):
        a = BlockAllocator(num_blocks=2)
        b = a.allocate()
        a.free(b)
        self.assertEqual(a.num_free, 2)
        self.assertEqual(a.allocate(), b, msg="释放后应可重新分配同一块。")

    def test_refcount_sharing(self):
        a = BlockAllocator(num_blocks=2)
        b = a.allocate()
        a.incref(b)                       # 两个引用（共享）
        a.free(b)
        self.assertEqual(a.num_used, 1, msg="仍有一个引用，块不应被回收。")
        a.free(b)
        self.assertEqual(a.num_used, 0, msg="引用归零后才回收。")

    def test_oom_raises(self):
        a = BlockAllocator(num_blocks=1)
        a.allocate()
        with self.assertRaises(MemoryError, msg="耗尽后再分配应抛 MemoryError（OOM）。"):
            a.allocate()

    def test_double_free_raises(self):
        a = BlockAllocator(num_blocks=1)
        b = a.allocate()
        a.free(b)
        with self.assertRaises(RuntimeError, msg="重复释放应被检测为 double free。"):
            a.free(b)

    def test_leak_detection(self):
        a = BlockAllocator(num_blocks=2)
        a.allocate()
        with self.assertRaises(AssertionError, msg="未释放即自检应报泄漏。"):
            a.check_no_leak()

    def test_no_leak_after_balanced_free(self):
        a = BlockAllocator(num_blocks=4)
        xs = [a.allocate() for _ in range(4)]
        for x in xs:
            a.free(x)
        a.check_no_leak()   # 不应抛异常


if __name__ == "__main__":
    unittest.main()
