"""Lesson 17 检查：完整引擎端到端正确、无泄漏、指标可得。"""
import unittest
from mini_vllm.config import ModelConfig, EngineConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.sampling import Sampler, SamplingParams
from mini_vllm.engine.generate import generate_cached
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.scheduler.scheduler import SchedulerConfig
from benchmarks.report import build_report


class TestTextEngine(unittest.TestCase):
    def _run(self, prefix_cache):
        model = TinyTextModel(ModelConfig()); tok = ByteTokenizer()
        ec = EngineConfig(block_size=8, num_blocks=128, max_num_seqs=3, max_num_batched_tokens=32)
        eng = LLMEngine(model, tok, ec, SchedulerConfig(
            max_num_seqs=3, max_num_batched_tokens=32, policy="balanced",
            enable_chunked_prefill=True), enable_prefix_cache=prefix_cache)
        specs = [("chat1", "System: assistant. Hi", 6, 0),
                 ("chat2", "System: assistant. Hello", 8, 1),
                 ("long", "The quick brown fox jumps over the lazy dog", 5, 2)]
        for rid, p, n, a in specs:
            eng.add_request(rid, p, n, arrival=a)
        res = eng.run()
        got = {r.request_id: r.output_token_ids for r in res.requests}
        ref = {rid: generate_cached(model, tok.encode(p, add_bos=True), n,
                                    Sampler(SamplingParams()), stop_on_eos=True).generated
               for rid, p, n, a in specs}
        eng.shutdown()
        return eng, res, got, ref

    def test_engine_correct_and_no_leak(self):
        eng, res, got, ref = self._run(prefix_cache=False)
        self.assertEqual(got, ref, msg="引擎输出必须与逐请求参考一致（无串请求/无重复执行）。")
        self.assertEqual(eng.allocator.num_used, 0, msg="运行后应无 KV 泄漏。")

    def test_engine_correct_with_prefix_cache(self):
        eng, res, got, ref = self._run(prefix_cache=True)
        self.assertEqual(got, ref, msg="开启前缀缓存后输出仍必须正确。")

    def test_report_has_core_metrics(self):
        eng, res, got, ref = self._run(prefix_cache=False)
        rep = build_report(eng, res)
        for k in ("requests", "iterations", "output_tokens", "kv_utilization", "peak_blocks_used"):
            self.assertIn(k, rep, msg=f"性能报告应包含指标 {k}。")


if __name__ == "__main__":
    unittest.main()
