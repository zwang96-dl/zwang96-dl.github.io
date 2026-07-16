"""Lesson 30 检查：多模态综合场景的全部自检通过。"""
import unittest
from experiments.lesson_30_mm_final import run_experiment
from experiments._common import read_config
from mini_vllm.trace import Tracer


class TestMMFinal(unittest.TestCase):
    def test_all_checks_pass(self):
        rep = run_experiment(read_config("configs/lesson_30_quick.json"), Tracer("quiet"))
        c = rep["checks"]
        self.assertTrue(c["no_cross_request_bleed"],
                        msg="不同请求的视觉 embedding 不得串用（输出与单独生成一致）。")
        self.assertTrue(c["placeholder_aligned"], msg="placeholder 必须与媒体严格对齐。")
        self.assertTrue(c["timestamp_preserved"], msg="视频 timestamp metadata 必须保留。")
        self.assertTrue(c["encoder_cache_hit"], msg="相同媒体应命中 encoder 缓存。")


if __name__ == "__main__":
    unittest.main()
