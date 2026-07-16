"""Lesson 15 检查：chunked prefill 保持输出正确；预算不足且未开 chunked 时明确停滞。"""
import unittest
from mini_vllm.config import ModelConfig, EngineConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.sampling import Sampler, SamplingParams
from mini_vllm.engine.generate import generate_cached
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.scheduler.scheduler import SchedulerConfig


class TestChunkedPrefill(unittest.TestCase):
    def setUp(self):
        self.model = TinyTextModel(ModelConfig()); self.tok = ByteTokenizer()
        self.long = "The quick brown fox jumps over the lazy dog again today"

    def _engine(self, chunked, budget=8):
        ec = EngineConfig(block_size=8, num_blocks=128, max_num_seqs=2, max_num_batched_tokens=budget)
        return LLMEngine(self.model, self.tok, ec, SchedulerConfig(
            max_num_seqs=2, max_num_batched_tokens=budget, policy="balanced",
            enable_chunked_prefill=chunked))

    def test_chunked_prefill_output_correct(self):
        eng = self._engine(chunked=True)
        eng.add_request("long", self.long, 4, arrival=0)
        eng.add_request("short", "Hi", 6, arrival=0)
        res = eng.run()
        got = {r.request_id: r.output_token_ids for r in res.requests}
        for rid, prompt in [("long", self.long), ("short", "Hi")]:
            ref = generate_cached(self.model, self.tok.encode(prompt, add_bos=True),
                                  {"long": 4, "short": 6}[rid], Sampler(SamplingParams()),
                                  stop_on_eos=False).generated
            self.assertEqual(got[rid], ref,
                             msg=f"chunked prefill 改变了 {rid} 的输出——切块不应影响结果。")

    def test_stall_without_chunked_when_prompt_exceeds_budget(self):
        eng = self._engine(chunked=False, budget=8)   # long prompt >> 8 tokens
        eng.add_request("long", self.long, 4, arrival=0)
        with self.assertRaises(RuntimeError,
                               msg="预算 < prompt 且未开 chunked prefill 时，应明确报「调度停滞」而非死循环。"):
            eng.run()


if __name__ == "__main__":
    unittest.main()
