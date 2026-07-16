"""Lesson 21 检查：patchify 的 grid / visual token 数 / 维度。"""
import unittest
from mini_vllm.config import VisionConfig
from mini_vllm.model.transformer import Rng
from mini_vllm.model import matrix as M
from mini_vllm.multimodal.patch_embed import PatchEmbed
from mini_vllm.multimodal.image_processor import TinyImageProcessor
from mini_vllm.multimodal.media import synth_image


class TestPatch(unittest.TestCase):
    def test_num_patches_and_dim(self):
        vc = VisionConfig(image_size=16, patch_size=8)
        pe = PatchEmbed(vc, Rng(vc.seed))
        self.assertEqual(vc.num_patches, 4, msg="16/8 → grid 2×2 = 4 个 visual token。")
        self.assertEqual(pe.patch_dim, 3 * 8 * 8, msg="patch_dim = 3×patch×patch。")

    def test_embed_shape(self):
        vc = VisionConfig(image_size=16, patch_size=8)
        pe = PatchEmbed(vc, Rng(vc.seed))
        chw = TinyImageProcessor(image_size=16).preprocess(synth_image(16, 16, 1)).pixels_norm_chw
        self.assertEqual(M.shape(pe(chw)), (4, vc.vision_hidden_size),
                         msg="patch embedding 形状应为 (num_patches, vision_hidden)。")


if __name__ == "__main__":
    unittest.main()
