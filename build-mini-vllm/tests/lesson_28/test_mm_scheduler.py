"""Lesson 28 检查：visual token 预算限制准入；预算不改变输出。"""
import unittest
from mini_vllm.config import ModelConfig, VisionConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.multimodal.mm_engine import MultiModalEngine
from mini_vllm.multimodal.budget import MultiModalBudget
from mini_vllm.multimodal import messages as msg


class TestMMScheduler(unittest.TestCase):
    def _run(self, budget):
        eng = MultiModalEngine(TinyTextModel(ModelConfig()), VisionConfig(),
                               budget=budget, enable_caches=True)
        img = lambda s: {"synth": {"h": 16, "w": 16, "seed": s}}
        for i in range(3):
            eng.add_request(f"r{i}", [msg.user(msg.text("q"), msg.image(img(i)))], 4, arrival=0)
        res = eng.run()
        return res, {r.request_id: r.output_token_ids for r in res.requests}

    def test_tight_visual_budget_admits_fewer_first_iter(self):
        res_tight, _ = self._run(MultiModalBudget(visual_token_budget=4, encoder_budget=1, max_num_seqs=3))
        res_loose, _ = self._run(MultiModalBudget(visual_token_budget=64, encoder_budget=4, max_num_seqs=3))
        self.assertLessEqual(len(res_tight.snapshots[0].admitted),
                             len(res_loose.snapshots[0].admitted),
                             msg="更紧的 visual 预算，第一迭代应准入更少请求。")

    def test_budget_does_not_change_output(self):
        _, tight = self._run(MultiModalBudget(visual_token_budget=4, encoder_budget=1, max_num_seqs=3))
        _, loose = self._run(MultiModalBudget(visual_token_budget=64, encoder_budget=4, max_num_seqs=3))
        self.assertEqual(tight, loose, msg="预算只改「何时算」，不改「算什么」。")


if __name__ == "__main__":
    unittest.main()
