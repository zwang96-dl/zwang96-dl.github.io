"""Lesson 2 检查：Tensor shape 与矩阵乘法。

用手算得到的期望值来断言 matmul / transpose / 广播，并确认形状不相容会被明确拒绝。
"""

import unittest

from mini_vllm.model import matrix as M


class TestLesson02Matrix(unittest.TestCase):
    def test_shape_rectangular(self):
        self.assertEqual(M.shape([[1, 2, 3], [4, 5, 6]]), (2, 3))
        self.assertEqual(M.shape([[[1, 2], [3, 4]]]), (1, 2, 2))

    def test_shape_jagged_raises(self):
        with self.assertRaises(ValueError,
                               msg="锯齿（不规整）张量应抛 ValueError，提前暴露最难查的 shape bug。"):
            M.shape([[1, 2, 3], [4, 5]])

    def test_matmul_hand_value(self):
        # [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]] = [[4,5],[10,11]]
        out = M.matmul([[1, 2, 3], [4, 5, 6]], [[1, 0], [0, 1], [1, 1]])
        self.assertEqual(
            out, [[4.0, 5.0], [10.0, 11.0]],
            msg="matmul 结果不对；out[i][j]=Σ_k A[i][k]·B[k][j]。检查 matrix.matmul()。",
        )

    def test_matmul_shape(self):
        out = M.matmul(M.zeros(2, 5), M.zeros(5, 3))
        self.assertEqual(M.shape(out), (2, 3),
                         msg="(m,k)@(k,n) 的结果应为 (m,n)=(2,3)。")

    def test_matmul_incompatible_raises_with_numbers(self):
        with self.assertRaises(ValueError) as ctx:
            M.matmul([[1, 2, 3]], [[1, 2, 3]])  # (1,3)@(1,3) 不相容
        self.assertIn("3", str(ctx.exception),
                      msg="不相容错误信息应包含具体维度数字，便于定位。")

    def test_transpose(self):
        self.assertEqual(M.transpose([[1, 2, 3], [4, 5, 6]]),
                         [[1, 4], [2, 5], [3, 6]])

    def test_add_row_bias_broadcast(self):
        # (2,2) + (2,) 行广播：每行都加 [10,20]
        out = M.add_row_bias([[1, 2], [3, 4]], [10, 20])
        self.assertEqual(out, [[11, 22], [13, 24]],
                         msg="行广播 (m,n)+(n,) 应给每一行加同一个 bias。")

    def test_batched_matmul(self):
        a = [[[1, 0], [0, 1]], [[2, 0], [0, 2]]]  # (2,2,2)
        b = [[[5, 6], [7, 8]], [[1, 1], [1, 1]]]
        out = M.batched_matmul(a, b)
        self.assertEqual(out[0], [[5, 6], [7, 8]], msg="第 0 个 batch 是单位阵乘 b[0]。")
        self.assertEqual(out[1], [[2, 2], [2, 2]], msg="第 1 个 batch 是 2*单位阵乘 全1阵。")

    def test_experiment_rejects_incompatible(self):
        from pathlib import Path
        from experiments.lesson_02_tensor import run_experiment
        from mini_vllm.trace import Tracer
        root = Path(__file__).resolve().parent.parent.parent
        result = run_experiment(root / "configs/lesson_02_quick.json", "quick", Tracer("quiet"))
        self.assertTrue(result["incompatible_rejected"],
                        msg="实验应演示形状不相容的 matmul 被明确拒绝。")
        self.assertEqual(result["C"], [[4.0, 5.0], [10.0, 11.0]])


if __name__ == "__main__":
    unittest.main()
