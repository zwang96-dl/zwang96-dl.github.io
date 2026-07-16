"""Lesson 27 检查：三层缓存命中/未命中，以及 stale cache 防护。"""
import unittest
from mini_vllm.config import ModelConfig, VisionConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.multimodal.runner import MultiModalRunner
from mini_vllm.multimodal.cache import ProcessorCache, EncoderOutputCache, content_hash
from mini_vllm.multimodal import messages as msg
from mini_vllm.multimodal.media import synth_image


class TestThreeCache(unittest.TestCase):
    def setUp(self):
        self.model = TinyTextModel(ModelConfig())

    def test_same_image_hits_encoder_cache(self):
        pc, ec = ProcessorCache(), EncoderOutputCache()
        runner = MultiModalRunner(self.model, VisionConfig(), processor_cache=pc, encoder_cache=ec)
        m = [msg.user(msg.image({"synth": {"h": 16, "w": 16, "seed": 5}}))]
        runner.build_inputs(m)
        first = runner.encoder_runs
        runner.build_inputs(m)
        self.assertEqual(runner.encoder_runs, first, msg="第二次相同图应命中缓存、不再编码。")
        self.assertGreater(ec.stats()["hits"], 0)

    def test_stale_cache_avoided_on_encoder_change(self):
        pc, ec = ProcessorCache(), EncoderOutputCache()
        m = [msg.user(msg.image({"synth": {"h": 16, "w": 16, "seed": 5}}))]
        r1 = MultiModalRunner(self.model, VisionConfig(seed=1), processor_cache=pc, encoder_cache=ec)
        r1.build_inputs(m)
        r2 = MultiModalRunner(self.model, VisionConfig(seed=2), processor_cache=pc, encoder_cache=ec)
        before = r2.encoder_runs
        r2.build_inputs(m)
        self.assertEqual(r2.encoder_runs - before, 1,
                         msg="换了 encoder 身份应重新编码，而不是误命中旧缓存（stale cache）。")

    def test_content_hash_stable(self):
        img = synth_image(8, 8, 3)
        self.assertEqual(content_hash(img), content_hash(synth_image(8, 8, 3)))
        self.assertNotEqual(content_hash(img), content_hash(synth_image(8, 8, 4)))


if __name__ == "__main__":
    unittest.main()
