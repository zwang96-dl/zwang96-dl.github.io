"""Lesson 8 检查：prefill/decode 结构与 TTFT/TPOT 指标。"""

import unittest

from mini_vllm.config import ModelConfig
from mini_vllm.model.transformer import TinyTextModel
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.sampling import Sampler, SamplingParams
from mini_vllm.engine.generate import generate_cached


class TestPrefillDecode(unittest.TestCase):
    def setUp(self):
        self.model = TinyTextModel(ModelConfig())
        self.tok = ByteTokenizer()

    def _gen(self, prompt, n=8):
        ids = self.tok.encode(prompt, add_bos=True)
        return ids, generate_cached(self.model, ids, n, Sampler(SamplingParams()),
                                    stop_on_eos=False)

    def test_first_step_is_prefill_over_whole_prompt(self):
        ids, g = self._gen("Hello")
        self.assertEqual(g.steps[0].phase, "prefill")
        self.assertEqual(g.steps[0].processed_tokens, len(ids),
                         msg="prefill 应一次处理整段 prompt。")

    def test_decode_steps_process_single_token(self):
        _, g = self._gen("Hello")
        for s in g.steps[1:]:
            self.assertEqual(s.phase, "decode")
            self.assertEqual(s.input_len, 1,
                             msg="每个 decode 步只喂 1 个 token。")

    def test_ttft_is_first_step_time(self):
        _, g = self._gen("Hello")
        self.assertEqual(g.ttft, g.steps[0].dt,
                         msg="TTFT 应等于第一步（prefill）的耗时。")
        self.assertGreaterEqual(g.ttft, 0.0)

    def test_context_len_grows_by_one_each_decode(self):
        ids, g = self._gen("Hello", n=5)
        ctx = [s.context_len for s in g.steps]
        for a, b in zip(ctx, ctx[1:]):
            self.assertEqual(b - a, 1, msg="上下文长度每步应 +1。")

    def test_tpot_defined_when_decoding(self):
        _, g = self._gen("Hello", n=4)
        self.assertGreater(len(g.decode_times), 0)
        self.assertGreaterEqual(g.tpot, 0.0)


if __name__ == "__main__":
    unittest.main()
