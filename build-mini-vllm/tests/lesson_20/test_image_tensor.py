"""Lesson 20 检查：图片处理的 resize / layout / normalize。"""
import unittest
from mini_vllm.multimodal.image_processor import TinyImageProcessor
from mini_vllm.multimodal.media import synth_image
from mini_vllm.model import matrix as M


class TestImageProcessor(unittest.TestCase):
    def test_resize_to_square(self):
        out = TinyImageProcessor(image_size=16).preprocess(synth_image(20, 24, seed=1))
        self.assertEqual(len(out.pixels_norm_hwc), 16, msg="应 resize 到 image_size×image_size。")
        self.assertEqual(len(out.pixels_norm_hwc[0]), 16)

    def test_channel_first_layout(self):
        out = TinyImageProcessor(image_size=8).preprocess(synth_image(8, 8, seed=1))
        self.assertEqual(M.shape(out.pixels_norm_chw), (3, 8, 8),
                         msg="channel-first 应为 (3, S, S)。")
        self.assertEqual(M.shape(out.pixels_norm_hwc), (8, 8, 3))

    def test_normalize_consistency(self):
        # channel-first 与 channel-last 应表示同一像素
        out = TinyImageProcessor(image_size=8).preprocess(synth_image(8, 8, seed=2))
        self.assertAlmostEqual(out.pixels_norm_chw[0][3][5], out.pixels_norm_hwc[3][5][0], places=9)


if __name__ == "__main__":
    unittest.main()
