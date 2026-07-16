"""Lesson 0 测试：请求生命周期机制模拟器的正确性与可复现性。

这些测试断言的是**机制**，不是玩具输出的具体文本：状态机是否走通、
prefill/decode 的 token 计数是否正确、KV block 是否无泄漏、结果是否可复现。
"""

import unittest

from mini_vllm.simulator import LifecycleSimulator, SimRequest, RequestState


def make_requests():
    return [
        SimRequest(request_id="req_0", prompt="Hello", max_new_tokens=3, arrival=0),
        SimRequest(request_id="req_1", prompt="世界", max_new_tokens=2, arrival=1),
    ]


class TestLifecycleSimulator(unittest.TestCase):
    def setUp(self):
        self.sim = LifecycleSimulator(
            block_size=8, num_blocks=16, max_num_seqs=2, token_budget=32
        )

    def test_all_requests_finish(self):
        result = self.sim.run(make_requests())
        for r in result.requests:
            self.assertEqual(
                r.state, RequestState.FINISHED,
                msg=f"{r.request_id} 应走到 FINISHED，实际停在 {r.state}；"
                    "检查 mini_vllm/simulator/text_pipeline.py 的 run() 收尾逻辑。",
            )

    def test_generated_token_counts_match_request(self):
        result = self.sim.run(make_requests())
        counts = {r.request_id: len(r.generated) for r in result.requests}
        self.assertEqual(
            counts, {"req_0": 3, "req_1": 2},
            msg="每个请求生成的 token 数应等于其 max_new_tokens；"
                "预期 {'req_0': 3, 'req_1': 2}，实际 " + str(counts) + "。"
                "可能原因：done 判定或 decode 步进有误。",
        )

    def test_prefill_processes_whole_prompt_decode_one(self):
        # req_0 prompt = BOS + "Hello" = 6 个 token；prefill 应一次调度 6 个。
        result = self.sim.run(make_requests())
        first = result.records[0]
        self.assertEqual(
            first.scheduled, ["req_0"],
            msg="第一迭代应只调度已到达的 req_0（req_1 arrival=1，尚未到达）。",
        )
        self.assertEqual(
            first.scheduled_tokens, 6,
            msg="prefill 应一次处理整段 prompt（BOS+'Hello'=6）；"
                "若为 1，说明 prefill 被当成了 decode，检查 phase 判断。",
        )
        # IterationRecord 记录的是「本步实际执行的 phase」——第一步是 prefill。
        # （此处历史上曾误记成翻转后的 "decode"，是 inspect scheduler 显示错误的根因，已修复。）
        self.assertEqual(first.phases["req_0"], "prefill",
                         msg="第一步是 prefill，记录的 phase 就应是 'prefill'（不应记成翻转后的 decode）。")
        # 到第 2 步，req_0 才进入 decode。
        self.assertEqual(result.records[1].phases["req_0"], "decode",
                         msg="prefill 完成后，req_0 在下一步进入 decode。")

    def test_first_token_recorded(self):
        result = self.sim.run(make_requests())
        r0 = result.requests[0]
        self.assertEqual(
            r0.first_token_iter, 1,
            msg="req_0 应在第 1 次迭代（prefill）产出首 token；"
                "first_token_iter 是后续 TTFT 指标的代理。",
        )

    def test_no_kv_block_leak(self):
        # 结束后所有 block 必须归还。模拟器内部已有 assert，这里再从结果侧确认。
        result = self.sim.run(make_requests())
        last = result.records[-1]
        self.assertEqual(
            last.blocks_in_use, 0,
            msg="所有请求结束后，正在使用的 KV block 必须为 0（无泄漏）；"
                "实际 " + str(last.blocks_in_use) + "。检查 run() 中 FINISHED 时的 pool.release。",
        )
        self.assertEqual(last.free_blocks, result.num_blocks,
                         msg="空闲 block 应恢复到 num_blocks。")

    def test_deterministic_reproducible(self):
        # 相同输入必须产出完全相同的生成序列——这是无神经网络玩具函数的意义。
        r1 = self.sim.run(make_requests())
        r2 = LifecycleSimulator(
            block_size=8, num_blocks=16, max_num_seqs=2, token_budget=32
        ).run(make_requests())
        g1 = {r.request_id: r.generated for r in r1.requests}
        g2 = {r.request_id: r.generated for r in r2.requests}
        self.assertEqual(
            g1, g2,
            msg="两次相同运行的生成结果应完全一致（确定性）；"
                "若不一致，说明引入了随机性或依赖了运行环境。",
        )

    def test_continuous_batching_admits_later_arrival(self):
        # req_1 arrival=1，应在 req_0 已经 RUNNING 之后才被准入，体现 continuous batching。
        result = self.sim.run(make_requests())
        self.assertNotIn("req_1", result.records[0].scheduled,
                         msg="arrival=1 的请求不应出现在第 1 迭代。")
        joined = any("req_1" in rec.scheduled for rec in result.records)
        self.assertTrue(joined, msg="req_1 应在后续迭代加入同一个运行 batch。")

    def test_oom_raises_when_blocks_exhausted(self):
        # 只给 1 个 block，长 prompt 需要多个 block → 应触发模拟 OOM。
        tiny = LifecycleSimulator(block_size=2, num_blocks=1, max_num_seqs=1, token_budget=64)
        with self.assertRaises(MemoryError,
                               msg="KV block 不足时应抛 MemoryError（模拟 OOM），"
                                   "而不是静默越界。检查 _BlockPool.allocate。"):
            tiny.run([SimRequest(request_id="big", prompt="a long prompt here", max_new_tokens=1)])


if __name__ == "__main__":
    unittest.main()
