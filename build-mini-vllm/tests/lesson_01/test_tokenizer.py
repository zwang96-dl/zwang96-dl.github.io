"""Lesson 1 检查：byte-level tokenizer 的学习目标。

断言的是学习目标本身：byte 语义、无损往返、特殊 token、padding + mask，
并用一个集成测试确认 Lesson 1 实验里所有文本都能无损往返。
"""

import unittest

from mini_vllm.tokenizer import ByteTokenizer, BOS_ID, PAD_ID


class TestLesson01Tokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()

    def test_chinese_char_is_three_bytes(self):
        # 学习目标：byte-level 下，一个中文字 = 3 个 UTF-8 字节 = 3 个 token。
        ids = self.tok.encode("你", add_bos=False)
        self.assertEqual(
            len(ids), 3,
            msg="一个中文字应编码为 3 个 token（UTF-8 3 字节）；"
                "实际 " + str(len(ids)) + "。检查 ByteTokenizer.encode()。",
        )

    def test_lossless_roundtrip_mixed(self):
        text = "mini-vLLM 🚀 你好"
        self.assertEqual(
            self.tok.decode(self.tok.encode(text, add_bos=False)), text,
            msg="混合 ASCII/中文/emoji 应无损往返；若失败多为 decode 截断多字节字符。",
        )

    def test_bos_added_and_stripped(self):
        ids = self.tok.encode("hi")  # 默认加 BOS
        self.assertEqual(ids[0], BOS_ID)
        self.assertEqual(self.tok.decode(ids), "hi",
                         msg="decode 应跳过 BOS，只还原真实字节。")

    def test_pad_produces_correct_mask(self):
        batch = [self.tok.encode("hi", add_bos=False),
                 self.tok.encode("hello", add_bos=False)]
        ids, mask = self.tok.pad(batch)
        self.assertEqual(len(ids[0]), 5, msg="应补齐到最长长度 5。")
        self.assertEqual(ids[0][2:], [PAD_ID, PAD_ID, PAD_ID],
                         msg="较短序列右侧应补 PAD_ID。")
        self.assertEqual(mask[0], [1, 1, 0, 0, 0],
                         msg="mask 中真实 token 为 1、padding 为 0。")

    def test_experiment_all_roundtrip_ok(self):
        # 集成：跑一遍 Lesson 1 实验，确认全部文本无损往返。
        from pathlib import Path
        from experiments.lesson_01_tokenizer import run_experiment
        from mini_vllm.trace import Tracer
        root = Path(__file__).resolve().parent.parent.parent
        result = run_experiment(root / "configs/lesson_01_quick.json", "quick", Tracer("quiet"))
        self.assertTrue(
            all(r["roundtrip_ok"] for r in result["rows"]),
            msg="Lesson 1 实验中应有全部文本 encode→decode 无损往返。",
        )


if __name__ == "__main__":
    unittest.main()
