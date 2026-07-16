"""Lesson 12 检查：连续预留 vs 分页的碎片对比。"""
import unittest
from experiments.lesson_12_kv_waste import run_experiment
from experiments._common import read_config
from mini_vllm.trace import Tracer


class TestKVWaste(unittest.TestCase):
    def test_paged_wastes_less_than_contiguous(self):
        cfg = read_config("configs/lesson_12_quick.json")
        r = run_experiment(cfg, Tracer("quiet"))
        self.assertLess(r["paged_waste"], r["contiguous_waste"],
                        msg="分页的浪费应远小于连续预留（后者按 max_seq 预留）。")

    def test_paged_waste_below_one_block_per_seq(self):
        cfg = read_config("configs/lesson_12_quick.json")
        r = run_experiment(cfg, Tracer("quiet"))
        max_possible = (r["block_size"] - 1) * len(r["seq_lens"])
        self.assertLessEqual(r["paged_waste"], max_possible,
                             msg="分页浪费每序列不超过「不足一块」的零头。")


if __name__ == "__main__":
    unittest.main()
