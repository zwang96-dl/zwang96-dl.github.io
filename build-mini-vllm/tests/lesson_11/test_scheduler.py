"""Lesson 11 检查：调度器预算约束、策略排序，以及「策略不改变输出」。"""
import unittest
from mini_vllm.config import ModelConfig, EngineConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.scheduler.scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from mini_vllm.scheduler.request import Request


class TestSchedulerUnit(unittest.TestCase):
    def test_token_budget_respected(self):
        s = Scheduler(SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=10,
                                      enable_chunked_prefill=True))
        for i in range(4):
            s.add(Request(f"r{i}", [1, 2, 3, 4, 5], max_new_tokens=3, arrival=0))
        out = s.schedule(iteration=1, free_blocks=100, block_size=8)
        self.assertLessEqual(out.scheduled_tokens, 10,
                             msg="一次迭代调度的 token 数不得超过 max_num_batched_tokens。")

    def test_max_num_seqs_respected(self):
        s = Scheduler(SchedulerConfig(max_num_seqs=2, max_num_batched_tokens=100))
        for i in range(5):
            s.add(Request(f"r{i}", [1, 2], max_new_tokens=3, arrival=0))
        s.schedule(iteration=1, free_blocks=100, block_size=8)
        self.assertLessEqual(len(s.running), 2, msg="同时 RUNNING 的请求数不得超过 max_num_seqs。")


class TestSchedulerPolicies(unittest.TestCase):
    def _run(self, policy):
        model = TinyTextModel(ModelConfig()); tok = ByteTokenizer()
        ec = EngineConfig(block_size=8, num_blocks=128, max_num_seqs=1, max_num_batched_tokens=64)
        eng = LLMEngine(model, tok, ec, SchedulerConfig(
            max_num_seqs=1, max_num_batched_tokens=64, policy=policy))
        eng.add_request("long", "The quick brown fox jumps", 6, arrival=0)
        eng.add_request("short1", "Hi", 2, arrival=0)
        eng.add_request("short2", "Yo", 2, arrival=0)
        res = eng.run()
        outs = {r.request_id: r.output_token_ids for r in res.requests}
        ttft = {r.request_id: r.first_token_iter for r in res.requests}
        return outs, ttft

    def test_outputs_identical_across_policies(self):
        base, _ = self._run("fifo")
        for p in ("decode-first", "sjf", "balanced"):
            outs, _ = self._run(p)
            self.assertEqual(outs, base,
                             msg=f"策略 {p} 改变了输出——调度只应影响顺序，不影响结果。")

    def test_sjf_prioritizes_short_jobs(self):
        _, fifo_ttft = self._run("fifo")
        _, sjf_ttft = self._run("sjf")
        # SJF 下短请求应比 FIFO 更早拿到首 token
        self.assertLess(sjf_ttft["short1"], fifo_ttft["short1"],
                        msg="SJF 应让短请求更早开始（首 token 更早）。")


if __name__ == "__main__":
    unittest.main()
