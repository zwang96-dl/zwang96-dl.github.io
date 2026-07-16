"""Lesson 09 检查：静态批处理 padding 浪费的计算，与 padding mask 语义。"""
import unittest
from pathlib import Path
from experiments.lesson_09_static_batching import run_experiment
from experiments._common import read_config
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.trace import Tracer


class TestStaticBatching(unittest.TestCase):
    def test_waste_is_positive_for_uneven_lengths(self):
        cfg = read_config("configs/lesson_09_quick.json")
        r = run_experiment(cfg, Tracer("quiet"))
        self.assertEqual(r["static_cost"], r["batch_size"] * r["max_len"],
                         msg="静态成本 = 批大小 × 最长长度。")
        self.assertEqual(r["continuous_cost"], sum(r["lens"]),
                         msg="连续成本 = 各请求真实长度之和。")
        self.assertGreater(r["waste"], 0,
                           msg="长度不齐时静态批处理必有浪费。")

    def test_pad_mask_marks_real_tokens(self):
        tok = ByteTokenizer()
        ids, mask = tok.pad([tok.encode("hi", add_bos=False), tok.encode("hello!", add_bos=False)])
        self.assertEqual(mask[0], [1, 1, 0, 0, 0, 0],
                         msg="mask 应标记真实 token(1) 与 padding(0)。")


if __name__ == "__main__":
    unittest.main()
