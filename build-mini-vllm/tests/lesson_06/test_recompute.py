"""Lesson 6 检查：重复计算侦探（naive 的浪费 vs cached）。"""

import unittest

from mini_vllm.config import ModelConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.sampling import Sampler, SamplingParams
from mini_vllm.engine.generate import generate_naive, generate_cached, processed_token_curves


class TestRecompute(unittest.TestCase):
    def setUp(self):
        self.model = TinyTextModel(ModelConfig())
        self.tok = ByteTokenizer()
        self.ids = self.tok.encode("Hello", add_bos=True)

    def test_naive_and_cached_same_output(self):
        gn = generate_naive(self.model, self.ids, 8, Sampler(SamplingParams()), stop_on_eos=False)
        gc = generate_cached(self.model, self.ids, 8, Sampler(SamplingParams()), stop_on_eos=False)
        self.assertEqual(gn.generated, gc.generated,
                         msg="naive 与 cached 在 greedy 下必须产出相同结果。")

    def test_naive_processes_more_than_cached(self):
        gn = generate_naive(self.model, self.ids, 8, Sampler(SamplingParams()), stop_on_eos=False)
        gc = generate_cached(self.model, self.ids, 8, Sampler(SamplingParams()), stop_on_eos=False)
        self.assertGreater(gn.total_processed_tokens, gc.total_processed_tokens,
                           msg="naive 应处理更多 token（重算前缀）——这正是 KV Cache 的动机。")

    def test_naive_step_processing_grows(self):
        gn = generate_naive(self.model, self.ids, 6, Sampler(SamplingParams()), stop_on_eos=False)
        processed = [s.processed_tokens for s in gn.steps]
        self.assertEqual(processed, sorted(processed),
                         msg="naive 每步处理量应单调不减（前缀越来越长）。")
        self.assertTrue(processed[-1] > processed[0])

    def test_cached_decode_processes_one(self):
        gc = generate_cached(self.model, self.ids, 6, Sampler(SamplingParams()), stop_on_eos=False)
        decode = [s.processed_tokens for s in gc.steps if s.phase == "decode"]
        self.assertTrue(all(p == 1 for p in decode),
                        msg="cached 的每个 decode 步应只处理 1 个 token。")

    def test_curve_analytic_matches_shape(self):
        c = processed_token_curves(prompt_len=6, n_new=5)
        self.assertEqual(len(c["naive"]), 5)
        self.assertGreater(c["naive_total"], c["cached_total"])
        # cached_total = prompt_len(prefill) + (n_new-1)*1
        self.assertEqual(c["cached_total"], 6 + 4)


if __name__ == "__main__":
    unittest.main()
