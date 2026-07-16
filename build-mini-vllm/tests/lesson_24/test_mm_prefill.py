"""Lesson 24 检查：multimodal prefill 只跑一次 encoder，decode 是纯文本。"""
import unittest
from mini_vllm.config import ModelConfig, VisionConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.multimodal.runner import MultiModalRunner
from mini_vllm.multimodal import messages as msg
from mini_vllm.sampling import SamplingParams


class TestMMPrefill(unittest.TestCase):
    def setUp(self):
        self.runner = MultiModalRunner(TinyTextModel(ModelConfig()), VisionConfig())
        self.m = [msg.user(msg.text("desc"), msg.image({"synth": {"h": 16, "w": 16, "seed": 1}}))]

    def test_encoder_runs_once_per_media(self):
        before = self.runner.encoder_runs
        self.runner.generate(self.m, max_new_tokens=6, sampling=SamplingParams())
        self.assertEqual(self.runner.encoder_runs - before, 1,
                         msg="单图请求整个 prefill+decode 只应运行 1 次 vision encoder。")

    def test_prefill_includes_visual_tokens(self):
        pre = self.runner.prefill(self.m)
        self.assertEqual(sum(pre.visual_token_counts), self.runner.vcfg.num_patches,
                         msg="prefill 序列应含该图的视觉 token。")
        self.assertEqual(len(pre.merged_embeds), len(pre.input_ids))

    def test_generate_deterministic(self):
        a = self.runner.generate(self.m, 6, SamplingParams())["generated"]
        b = self.runner.generate(self.m, 6, SamplingParams())["generated"]
        self.assertEqual(a, b, msg="greedy 多模态生成应确定。")


if __name__ == "__main__":
    unittest.main()
