"""Lesson 16 检查：前缀缓存命中、命中不改变输出、无泄漏、逐出。"""
import unittest
from mini_vllm.config import ModelConfig, EngineConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.sampling import Sampler, SamplingParams
from mini_vllm.engine.generate import generate_cached
from mini_vllm.engine.engine import LLMEngine


class TestPrefixCache(unittest.TestCase):
    def setUp(self):
        self.model = TinyTextModel(ModelConfig()); self.tok = ByteTokenizer()
        self.shared = "System: you are a helpful assistant. "

    def _engine(self):
        ec = EngineConfig(block_size=4, num_blocks=256, max_num_seqs=1, max_num_batched_tokens=128)
        return LLMEngine(self.model, self.tok, ec, enable_prefix_cache=True)

    def test_second_request_hits_prefix(self):
        eng = self._engine()
        eng.add_request("q0", self.shared + "one?", 5, arrival=0); eng.run()
        before = eng.prefix_cache.stats()["hits"]
        eng.add_request("q1", self.shared + "two?", 5, arrival=0); eng.run()
        after = eng.prefix_cache.stats()["hits"]
        self.assertGreater(after, before, msg="共享前缀的第二个请求应命中缓存块。")
        eng.shutdown()

    def test_hit_does_not_change_output(self):
        eng = self._engine()
        eng.add_request("q0", self.shared + "one?", 5, arrival=0); eng.run()
        eng.add_request("q1", self.shared + "two?", 5, arrival=0); r = eng.run()
        got = next(x.output_token_ids for x in r.requests if x.request_id == "q1")
        ref = generate_cached(self.model, self.tok.encode(self.shared + "two?", add_bos=True), 5,
                              Sampler(SamplingParams()), stop_on_eos=False).generated
        self.assertEqual(got, ref, msg="前缀命中不得改变输出（与标准生成逐 token 一致）。")
        eng.shutdown()

    def test_no_leak_after_shutdown(self):
        eng = self._engine()
        eng.add_request("q0", self.shared + "one?", 5, arrival=0); eng.run()
        eng.add_request("q1", self.shared + "two?", 5, arrival=0); eng.run()
        eng.shutdown()
        self.assertEqual(eng.allocator.num_used, 0, msg="shutdown 后应无 KV 块泄漏。")


if __name__ == "__main__":
    unittest.main()
