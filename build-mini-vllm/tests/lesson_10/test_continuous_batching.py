"""Lesson 10 检查：请求状态机 + 连续批处理（引擎输出与逐请求参考一致，动态准入）。"""
import unittest
from mini_vllm.config import ModelConfig, EngineConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.sampling import Sampler, SamplingParams
from mini_vllm.engine.generate import generate_cached
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.scheduler.request import Request, RequestStatus


class TestRequestStateMachine(unittest.TestCase):
    def test_status_and_finish(self):
        r = Request("r", [256, 1, 2], max_new_tokens=2)
        self.assertEqual(r.status, RequestStatus.WAITING)
        self.assertFalse(r.prefill_done)
        r.num_computed_tokens = 3
        self.assertTrue(r.prefill_done)
        r.output_token_ids = [5, 6]
        self.assertTrue(r.is_finished(), msg="生成达到 max_new_tokens 应判定完成。")


class TestContinuousBatching(unittest.TestCase):
    def test_engine_matches_reference(self):
        model = TinyTextModel(ModelConfig()); tok = ByteTokenizer()
        prompts = {"A": "Hello there", "B": "Hi", "C": "The fox"}
        ref = {k: generate_cached(model, tok.encode(v, add_bos=True), 5,
                                  Sampler(SamplingParams()), stop_on_eos=False).generated
               for k, v in prompts.items()}
        ec = EngineConfig(block_size=8, num_blocks=64, max_num_seqs=3, max_num_batched_tokens=64)
        eng = LLMEngine(model, tok, ec)
        eng.add_request("A", prompts["A"], 5, arrival=0)
        eng.add_request("B", prompts["B"], 5, arrival=1)
        eng.add_request("C", prompts["C"], 5, arrival=3)
        res = eng.run()
        got = {r.request_id: r.output_token_ids for r in res.requests}
        self.assertEqual(got, ref,
                         msg="连续批处理（含错峰到达）的输出必须与逐请求参考一致。")

    def test_later_arrival_admitted_after_start(self):
        model = TinyTextModel(ModelConfig()); tok = ByteTokenizer()
        ec = EngineConfig(block_size=8, num_blocks=64, max_num_seqs=3, max_num_batched_tokens=64)
        eng = LLMEngine(model, tok, ec)
        eng.add_request("A", "Hello", 6, arrival=0)
        eng.add_request("B", "Hi", 4, arrival=2)
        res = eng.run()
        first_iter_scheduled = res.snapshots[0].scheduled
        self.assertNotIn("B", first_iter_scheduled,
                         msg="arrival=2 的请求不应出现在第 1 迭代（continuous batching）。")
        joined = any("B" in s.scheduled for s in res.snapshots)
        self.assertTrue(joined, msg="B 应在后续迭代加入运行 batch。")


if __name__ == "__main__":
    unittest.main()
