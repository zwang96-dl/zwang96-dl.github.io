"""Lesson 7 检查：KV Cache 正确性对齐、一致性与内存公式。

最关键的一条：cached 前向必须与 naive（重算整段）前向**逐值一致**——否则 KV Cache
就是错的，会悄悄改变模型输出。
"""

import unittest

from mini_vllm.config import ModelConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.model import matrix as M
from mini_vllm.cache.kv_cache import KVCache
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.sampling import Sampler, SamplingParams, argmax
from mini_vllm.engine.generate import generate_naive, generate_cached


class TestKVCache(unittest.TestCase):
    def setUp(self):
        self.model = TinyTextModel(ModelConfig())
        self.tok = ByteTokenizer()
        self.ids = self.tok.encode("Hello", add_bos=True)

    def test_prefill_logits_match_naive(self):
        naive = self.model.forward(self.ids, list(range(len(self.ids))))
        cache = KVCache(self.model.cfg)
        cached = self.model.forward(self.ids, list(range(len(self.ids))), cache)
        diff = M.max_abs_diff(naive, cached)
        self.assertEqual(diff, 0.0,
                         msg=f"prefill 时 cached 与 naive 的 logits 应完全一致，实际最大误差 {diff}。"
                             "检查 KVCache.append 与 attention 的 positions/mask。")

    def test_decode_logits_match_naive(self):
        # prefill 后 decode 一个 token，比较最后一行 logits。
        cache = KVCache(self.model.cfg)
        self.model.forward(self.ids, list(range(len(self.ids))), cache)
        nxt = argmax(self.model.forward(self.ids, list(range(len(self.ids))))[-1])
        naive2 = self.model.forward(self.ids + [nxt], list(range(len(self.ids) + 1)))
        dec = self.model.forward([nxt], [len(self.ids)], cache)
        diff = M.max_abs_diff([naive2[-1]], [dec[0]])
        self.assertEqual(diff, 0.0,
                         msg=f"decode 时 cached 与 naive 最后一行 logits 应一致，实际误差 {diff}。")

    def test_full_generation_identical(self):
        gn = generate_naive(self.model, self.ids, 10, Sampler(SamplingParams()), stop_on_eos=False)
        gc = generate_cached(self.model, self.ids, 10, Sampler(SamplingParams()), stop_on_eos=False)
        self.assertEqual(gn.generated, gc.generated,
                         msg="整段生成：cached 必须与 naive 逐 token 一致。")

    def test_cache_length_and_consistency(self):
        cache = KVCache(self.model.cfg)
        self.model.forward(self.ids, list(range(len(self.ids))), cache)
        self.assertEqual(cache.length, len(self.ids),
                         msg="prefill 后缓存长度应等于 prompt 长度。")
        self.model.forward([5], [len(self.ids)], cache)
        self.assertEqual(cache.length, len(self.ids) + 1,
                         msg="decode 一步后缓存长度应 +1。")
        cache.check_consistent()  # 每层长度应等于 positions 长度，否则抛异常

    def test_memory_formula(self):
        cache = KVCache(self.model.cfg)
        cache.positions = list(range(4))
        mem = cache.memory_estimate()
        cfg = self.model.cfg
        self.assertEqual(mem["per_token_per_layer"], 2 * cfg.num_kv_heads * cfg.head_dim)
        self.assertEqual(mem["total_elements"],
                         2 * cfg.num_kv_heads * cfg.head_dim * cfg.num_layers * 4,
                         msg="内存 = 2·kv_heads·head_dim·num_layers·seq_len。")


if __name__ == "__main__":
    unittest.main()
