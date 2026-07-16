"""Lesson 29 检查：完整多模态引擎——正确、不串请求、encoder 不在 decode 重复运行。"""
import unittest
from mini_vllm.config import ModelConfig, VisionConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.multimodal.runner import MultiModalRunner
from mini_vllm.multimodal.mm_engine import MultiModalEngine
from mini_vllm.multimodal.budget import MultiModalBudget
from mini_vllm.multimodal import messages as msg
from mini_vllm.sampling import SamplingParams


class TestMMEngine(unittest.TestCase):
    def setUp(self):
        self.model = TinyTextModel(ModelConfig())
        img = lambda s: {"synth": {"h": 16, "w": 16, "seed": s}}
        self.specs = [
            ("text", [msg.user(msg.text("hello"))], 4),
            ("img", [msg.user(msg.text("see"), msg.image(img(1)))], 5),
            ("img_same", [msg.user(msg.text("again"), msg.image(img(1)))], 4),
            ("multi", [msg.user(msg.image(img(1)), msg.image(img(2)))], 4),
        ]

    def _engine(self):
        eng = MultiModalEngine(self.model, VisionConfig(),
                               budget=MultiModalBudget(text_token_budget=32, visual_token_budget=32,
                                                       encoder_budget=3, max_num_seqs=3),
                               enable_caches=True)
        for rid, m, n in self.specs:
            eng.add_request(rid, m, n, arrival=0)
        return eng

    def test_outputs_match_standalone(self):
        eng = self._engine(); res = eng.run()
        outs = {r.request_id: r.output_token_ids for r in res.requests}
        ref_runner = MultiModalRunner(self.model, VisionConfig())
        for rid, m, n in self.specs:
            ref = ref_runner.generate(m, max_new_tokens=n, sampling=SamplingParams())["generated"]
            self.assertEqual(outs[rid], ref,
                             msg=f"{rid} 的引擎输出必须与单独生成一致（不跨请求串视觉 embedding）。")

    def test_encoder_cache_hit_for_duplicate_image(self):
        eng = self._engine(); eng.run()
        self.assertGreater(eng.encoder_cache.stats()["hits"], 0,
                           msg="重复出现的图应命中 encoder 缓存。")

    def test_encoder_not_run_in_decode(self):
        # mm_engine.run() 内含断言「decode 阶段 encoder_runs 不变」；跑通即通过。
        eng = self._engine(); eng.run()


if __name__ == "__main__":
    unittest.main()
