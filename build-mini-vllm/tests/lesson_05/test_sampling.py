"""Lesson 5 检查：采样与自回归生成。"""

import unittest

from mini_vllm.config import ModelConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.sampling import (Sampler, SamplingParams, argmax, softmax,
                                top_k_filter, top_p_filter)
from mini_vllm.engine.generate import generate_cached


class TestSampling(unittest.TestCase):
    def test_greedy_is_argmax(self):
        logits = [0.1, 5.0, 0.2, -1.0]
        s = Sampler(SamplingParams(temperature=0.0))
        self.assertEqual(s(logits), 1, msg="greedy 应选分数最高的 index。")
        self.assertEqual(argmax(logits), 1)

    def test_softmax_sums_to_one(self):
        p = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(p), 1.0, places=9)

    def test_top_k_keeps_only_k(self):
        probs = [0.4, 0.3, 0.2, 0.1]
        filt = top_k_filter(probs, 2)
        self.assertEqual(sum(1 for x in filt if x > 0), 2,
                         msg="top-k=2 应只保留 2 个非零概率。")
        self.assertEqual(filt[2], 0.0)
        self.assertEqual(filt[3], 0.0)

    def test_top_p_keeps_nucleus(self):
        probs = [0.5, 0.3, 0.15, 0.05]
        filt = top_p_filter(probs, 0.8)  # 0.5+0.3=0.8 达标
        self.assertGreater(filt[0], 0)
        self.assertGreater(filt[1], 0)
        self.assertEqual(filt[3], 0.0, msg="尾部低概率应被 nucleus 截断。")

    def test_temperature_reproducible_with_seed(self):
        model = TinyTextModel(ModelConfig())
        tok = ByteTokenizer()
        ids = tok.encode("Hi", add_bos=True)
        a = generate_cached(model, ids, 6, Sampler(SamplingParams(temperature=1.0, seed=3)),
                            stop_on_eos=False)
        b = generate_cached(model, ids, 6, Sampler(SamplingParams(temperature=1.0, seed=3)),
                            stop_on_eos=False)
        self.assertEqual(a.generated, b.generated,
                         msg="相同 seed 的温度采样应可复现。")

    def test_greedy_generation_deterministic(self):
        model = TinyTextModel(ModelConfig())
        tok = ByteTokenizer()
        ids = tok.encode("Hi", add_bos=True)
        a = generate_cached(model, ids, 6, Sampler(SamplingParams()), stop_on_eos=False)
        b = generate_cached(model, ids, 6, Sampler(SamplingParams()), stop_on_eos=False)
        self.assertEqual(a.generated, b.generated, msg="greedy 生成应完全确定。")


if __name__ == "__main__":
    unittest.main()
