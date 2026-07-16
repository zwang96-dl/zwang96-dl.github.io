"""Lesson 19 检查：多模态消息解析——阶段边界、占位与媒体对齐。"""
import unittest
from mini_vllm.config import ModelConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.multimodal.runner import MultiModalRunner
from mini_vllm.multimodal.chat_template import MultiModalChatTemplate
from mini_vllm.multimodal import messages as msg


class TestMMRequest(unittest.TestCase):
    def setUp(self):
        self.runner = MultiModalRunner(TinyTextModel(ModelConfig()))

    def test_template_marks_media(self):
        segs, rendered = MultiModalChatTemplate().render(
            [msg.user(msg.text("hi"), msg.image({"synth": {"h": 16, "w": 16, "seed": 1}}))])
        self.assertIn("<image>", rendered, msg="模板应在图片处放占位标记，而非编码像素。")
        self.assertTrue(any(k == "media" for k, _ in segs))

    def test_placeholder_count_matches_media(self):
        m = [msg.user(msg.image({"synth": {"h": 16, "w": 16, "seed": 1}}),
                      msg.image({"synth": {"h": 16, "w": 16, "seed": 2}}))]
        ids, ranges, vembeds, meta = self.runner.build_inputs(m)
        self.assertEqual(len(ranges), 2, msg="两张图应产生两段 placeholder。")
        self.assertEqual([r.media_index for r in ranges], [0, 1],
                         msg="placeholder 的 media_index 顺序应与媒体顺序一致。")


if __name__ == "__main__":
    unittest.main()
