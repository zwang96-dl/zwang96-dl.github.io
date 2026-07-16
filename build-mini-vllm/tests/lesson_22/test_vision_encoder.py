"""Lesson 22 检查：vision encoder 与 projector 的形状与维度对齐。"""
import unittest
from mini_vllm.config import VisionConfig
from mini_vllm.model import matrix as M
from mini_vllm.multimodal.vision_encoder import TinyVisionEncoder, MultimodalProjector
from mini_vllm.multimodal.image_processor import TinyImageProcessor
from mini_vllm.multimodal.media import synth_image


class TestVisionEncoder(unittest.TestCase):
    def setUp(self):
        self.vc = VisionConfig()
        self.chw = TinyImageProcessor(image_size=self.vc.image_size).preprocess(
            synth_image(16, 16, 1)).pixels_norm_chw

    def test_encoder_output_shape(self):
        vis = TinyVisionEncoder(self.vc).encode(self.chw)
        self.assertEqual(M.shape(vis), (self.vc.num_patches, self.vc.vision_hidden_size))

    def test_projector_maps_to_text_hidden(self):
        vis = TinyVisionEncoder(self.vc).encode(self.chw)
        proj = MultimodalProjector(self.vc)(vis)
        self.assertEqual(M.shape(proj), (self.vc.num_patches, self.vc.text_hidden_size),
                         msg="projector 应把 vision_hidden 投影到 text_hidden，才能与文本 embedding 合并。")

    def test_deterministic(self):
        a = TinyVisionEncoder(self.vc).encode(self.chw)
        b = TinyVisionEncoder(self.vc).encode(self.chw)
        self.assertEqual(M.max_abs_diff(a, b), 0.0, msg="相同配置的编码应确定可复现。")


if __name__ == "__main__":
    unittest.main()
