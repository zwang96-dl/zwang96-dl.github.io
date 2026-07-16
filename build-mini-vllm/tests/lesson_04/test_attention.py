"""Lesson 4 检查：单头 causal attention 的五步。

用手算期望值断言 softmax、causal mask 与整体 attention 输出。
"""

import math
import unittest

from mini_vllm.model import attention_ref as A


class TestLesson04Attention(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        w = A.softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(w), 1.0, places=9,
                               msg="softmax 输出应非负且和为 1。")

    def test_softmax_numerically_stable(self):
        # 大数值不应溢出（实现先减去最大值）。
        w = A.softmax([1000.0, 1000.0])
        self.assertAlmostEqual(w[0], 0.5, places=9,
                               msg="相等的大 logits 应得到均匀分布；若 NaN/inf 说明未做数值稳定。")

    def test_masked_position_becomes_zero_weight(self):
        # -inf 经 softmax 应得 0 权重（屏蔽未来）。
        w = A.softmax([0.0, float("-inf")])
        self.assertEqual(w, [1.0, 0.0],
                         msg="被 mask（-inf）的位置权重应为 0。")

    def test_causal_mask_upper_triangle(self):
        masked = A.causal_mask_apply([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(masked[0][1], float("-inf"), msg="位置(0,1) j>i 应被 mask。")
        self.assertEqual(masked[0][2], float("-inf"))
        self.assertEqual(masked[1][2], float("-inf"))
        self.assertEqual(masked[2][2], 9, msg="对角线及以下不应被 mask。")

    def test_attention_output_shape(self):
        Q = [[1, 0], [0, 1], [1, 1]]
        K = [[1, 0], [0, 1], [1, 1]]
        V = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # dv=3
        out = A.scaled_dot_product_attention(Q, K, V, causal=True)
        self.assertEqual(len(out), 3, msg="out 行数应为 Tq=3。")
        self.assertEqual(len(out[0]), 3, msg="out 列数应为 dv=3。")

    def test_first_query_sees_only_first_key(self):
        # causal 下，第 0 个 query 权重应为 [1,0,0]，故 out[0] == V[0]。
        Q = [[1, 0], [1, 1], [0, 1]]
        K = [[1, 0], [0, 1], [1, 1]]
        V = [[10, 0], [0, 10], [5, 5]]
        out, st = A.scaled_dot_product_attention(Q, K, V, causal=True, return_stages=True)
        self.assertEqual(st["weights"][0], [1.0, 0.0, 0.0],
                         msg="第 0 个 query 只能看到 key 0；检查 causal_mask_apply/softmax。")
        self.assertEqual(out[0], [10.0, 0.0], msg="out[0] 应等于 V[0]。")

    def test_second_query_is_average_of_first_two_values(self):
        # query1=[1,1] 对 key0/key1 打平（各 0.5），out[1]=0.5*V0+0.5*V1。
        Q = [[1, 0], [1, 1], [0, 1]]
        K = [[1, 0], [0, 1], [1, 1]]
        V = [[10, 0], [0, 10], [5, 5]]
        out = A.scaled_dot_product_attention(Q, K, V, causal=True)
        self.assertAlmostEqual(out[1][0], 5.0, places=6)
        self.assertAlmostEqual(out[1][1], 5.0, places=6)

    def test_causal_differs_from_noncausal(self):
        Q = [[1, 0], [1, 1]]
        K = [[1, 0], [0, 1]]
        V = [[10, 0], [0, 10]]
        c = A.scaled_dot_product_attention(Q, K, V, causal=True)
        nc = A.scaled_dot_product_attention(Q, K, V, causal=False)
        self.assertNotEqual(c[0], nc[0],
                            msg="causal 与非 causal 在第 0 行应不同（前者看不到 key1）。")

    def test_experiment_weight_rows_sum_to_one(self):
        from pathlib import Path
        from experiments.lesson_04_attention import run_experiment
        from mini_vllm.trace import Tracer
        root = Path(__file__).resolve().parent.parent.parent
        result = run_experiment(root / "configs/lesson_04_quick.json", "quick", Tracer("quiet"))
        for s in result["row_weight_sums"]:
            self.assertTrue(math.isclose(s, 1.0, abs_tol=1e-9),
                            msg="每个 query 的注意力权重和应为 1。")


if __name__ == "__main__":
    unittest.main()
