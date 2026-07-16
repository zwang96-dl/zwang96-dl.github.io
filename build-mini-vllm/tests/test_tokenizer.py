"""ByteTokenizer 的单元测试（Lesson 1 的地基，Phase 1 已落地）。"""

import unittest

from mini_vllm.tokenizer import ByteTokenizer, BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE


class TestByteTokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()

    def test_vocab_size_is_259(self):
        self.assertEqual(
            self.tok.vocab_size, 259,
            msg="词表应为 256 字节 + BOS/EOS/PAD = 259；"
                "检查 mini_vllm/tokenizer.py 的 VOCAB_SIZE。",
        )

    def test_roundtrip_ascii(self):
        text = "Hello, mini-vLLM!"
        ids = self.tok.encode(text, add_bos=False)
        self.assertEqual(
            self.tok.decode(ids), text,
            msg="ASCII 文本 encode→decode 应无损还原；检查 ByteTokenizer.decode()。",
        )

    def test_roundtrip_unicode(self):
        # 中文与 emoji 都是多字节 UTF-8；byte-level tokenizer 必须能无损往返。
        text = "你好，推理引擎 🚀"
        ids = self.tok.encode(text, add_bos=False)
        self.assertEqual(
            self.tok.decode(ids), text,
            msg="多字节 UTF-8（中文/emoji）应无损还原；"
                "若失败，多半是 decode 时把字节拆断了，检查 ByteTokenizer.decode()。",
        )

    def test_bos_prepended_by_default(self):
        ids = self.tok.encode("A")  # add_bos 默认 True
        self.assertEqual(ids[0], BOS_ID,
                         msg="默认应在序列开头加 BOS；检查 encode() 的 add_bos 分支。")
        self.assertEqual(ids[1], ord("A"))

    def test_eos_optional(self):
        ids = self.tok.encode("A", add_bos=False, add_eos=True)
        self.assertEqual(ids[-1], EOS_ID,
                         msg="add_eos=True 时应在末尾加 EOS。")

    def test_decode_skips_special_tokens(self):
        ids = [BOS_ID, ord("h"), ord("i"), EOS_ID]
        self.assertEqual(
            self.tok.decode(ids, skip_special=True), "hi",
            msg="skip_special=True 时应丢弃 BOS/EOS/PAD，只保留真实字节。",
        )

    def test_pad_right_and_mask(self):
        batch = [self.tok.encode("hi", add_bos=False),
                 self.tok.encode("hello", add_bos=False)]
        ids, mask = self.tok.pad(batch)  # 右侧 padding 到最长长度 5
        self.assertEqual([len(x) for x in ids], [5, 5],
                         msg="pad 后每个序列长度应相等（= 最长序列长度）。")
        self.assertEqual(
            ids[0], [ord("h"), ord("i"), PAD_ID, PAD_ID, PAD_ID],
            msg="较短序列应在右侧补 PAD_ID。",
        )
        self.assertEqual(
            mask[0], [1, 1, 0, 0, 0],
            msg="attention_mask 中真实 token 为 1、padding 为 0——"
                "这正是 Lesson 9 里 static batching 计算浪费的来源。",
        )

    def test_id_to_piece_specials(self):
        self.assertEqual(self.tok.id_to_piece(BOS_ID), "<bos>")
        self.assertEqual(self.tok.id_to_piece(ord("A")), "A")

    def test_decode_rejects_out_of_range(self):
        with self.assertRaises(ValueError,
                               msg="超出词表范围的 id 应抛 ValueError，而不是静默产生错误输出。"):
            self.tok.decode([VOCAB_SIZE + 1])


if __name__ == "__main__":
    unittest.main()
