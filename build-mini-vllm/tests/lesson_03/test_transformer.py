"""Lesson 3 检查：极小 Transformer 的组件与前向。"""

import math
import unittest

from mini_vllm.config import ModelConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.model import matrix as M
from mini_vllm.model.rmsnorm import rms_norm
from mini_vllm.model.rope import rope_freqs, apply_rope_vec
from mini_vllm.model.mlp import swiglu, silu


class TestComponents(unittest.TestCase):
    def test_rmsnorm_normalizes_to_unit_rms(self):
        # weight=1 时，归一化后每行的 RMS 应约等于 1。
        x = [[3.0, 4.0, 0.0, 0.0]]  # rms = sqrt((9+16)/4)=2.5
        out = rms_norm(x, [1.0] * 4, eps=0.0)
        rms = math.sqrt(sum(v * v for v in out[0]) / 4)
        self.assertAlmostEqual(rms, 1.0, places=6,
                               msg="RMSNorm 后（weight=1）每行 RMS 应为 1。")

    def test_rope_identity_at_position_zero(self):
        # pos=0 时角度全为 0，旋转是恒等变换。
        freqs = rope_freqs(4, 10000.0)
        vec = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(apply_rope_vec(vec, 0, freqs), vec,
                         msg="位置 0 的 RoPE 应为恒等（角度为 0）。")

    def test_rope_preserves_norm(self):
        # 旋转不改变向量长度。
        freqs = rope_freqs(4, 10000.0)
        vec = [1.0, 2.0, 3.0, 4.0]
        r = apply_rope_vec(vec, 5, freqs)
        n0 = math.sqrt(sum(v * v for v in vec))
        n1 = math.sqrt(sum(v * v for v in r))
        self.assertAlmostEqual(n0, n1, places=6, msg="RoPE 是旋转，应保持向量范数不变。")

    def test_silu_and_swiglu_shape(self):
        self.assertAlmostEqual(silu(0.0), 0.0, places=9)
        x = [[1.0, 2.0]]
        wg = [[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]]  # (2,3)
        wu = [[0.1, 0.0, 0.1], [0.2, 0.1, 0.0]]
        wd = [[0.1, 0.2], [0.0, 0.1], [0.1, 0.1]]  # (3,2)
        out = swiglu(x, wg, wu, wd)
        self.assertEqual(M.shape(out), (1, 2), msg="SwiGLU 输出 shape 应为 (seq, hidden)。")


class TestForward(unittest.TestCase):
    def setUp(self):
        self.model = TinyTextModel(ModelConfig())

    def test_forward_shape(self):
        ids = [256, 72, 105]
        logits = self.model.forward(ids, [0, 1, 2])
        self.assertEqual(M.shape(logits), (3, 259),
                         msg="logits 形状应为 (seq_len, vocab)=(3,259)。")

    def test_deterministic_same_seed(self):
        a = TinyTextModel(ModelConfig(seed=1234))
        b = TinyTextModel(ModelConfig(seed=1234))
        ids = [256, 100, 101]
        self.assertEqual(
            M.max_abs_diff(a.forward(ids, [0, 1, 2]), b.forward(ids, [0, 1, 2])), 0.0,
            msg="相同 seed 的两个模型前向应逐位一致（确定性初始化）。",
        )

    def test_different_seed_differs(self):
        a = TinyTextModel(ModelConfig(seed=1))
        b = TinyTextModel(ModelConfig(seed=2))
        ids = [256, 100, 101]
        self.assertGreater(M.max_abs_diff(a.forward(ids, [0, 1, 2]), b.forward(ids, [0, 1, 2])), 0.0,
                           msg="不同 seed 应产生不同权重、不同输出。")

    def test_gqa_group_size(self):
        self.assertEqual(self.model.cfg.group_size, 2,
                         msg="GQA：4 个 query head / 2 个 kv head = 每 kv head 服务 2 个 query head。")


if __name__ == "__main__":
    unittest.main()
