"""Lesson 25 检查：多图与动态 visual token（不同请求视觉 token 数不同）。"""
import unittest
from mini_vllm.config import ModelConfig, VisionConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.multimodal.runner import MultiModalRunner
from mini_vllm.multimodal import messages as msg


class TestMultiImage(unittest.TestCase):
    def setUp(self):
        self.runner = MultiModalRunner(TinyTextModel(ModelConfig()), VisionConfig())
        self.pp = self.runner.vcfg.num_patches

    def _visual(self, m):
        _, _, vembeds, _ = self.runner.build_inputs(m)
        return sum(len(v) for v in vembeds)

    def test_visual_tokens_scale_with_num_images(self):
        img = lambda s: msg.image({"synth": {"h": 16, "w": 16, "seed": s}})
        self.assertEqual(self._visual([msg.user(msg.text("t"))]), 0, msg="纯文本无视觉 token。")
        self.assertEqual(self._visual([msg.user(img(1))]), self.pp)
        self.assertEqual(self._visual([msg.user(img(1), img(2))]), 2 * self.pp,
                         msg="两张图的视觉 token 应翻倍。")

    def test_media_order_preserved(self):
        m = [msg.user(msg.image({"synth": {"h": 16, "w": 16, "seed": 1}}), msg.text("mid"),
                      msg.image({"synth": {"h": 16, "w": 16, "seed": 2}}))]
        _, ranges, _, _ = self.runner.build_inputs(m)
        self.assertEqual([r.media_index for r in ranges], [0, 1])
        self.assertLess(ranges[0].offset, ranges[1].offset, msg="媒体在序列中的顺序应保持。")


if __name__ == "__main__":
    unittest.main()
