"""Lesson 23 检查：placeholder 校验与 embedding 合并（含对齐错误检测）。"""
import unittest
from mini_vllm.multimodal.placeholders import PlaceholderRange, validate_placeholders
from mini_vllm.multimodal.embedding_merge import merge_multimodal_embeddings


class TestPlaceholderMerge(unittest.TestCase):
    def test_merge_replaces_placeholder_positions(self):
        text = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ranges = [PlaceholderRange(1, 2, 0, "image")]
        vis = [[[1.0, 2.0], [3.0, 4.0]]]
        merged = merge_multimodal_embeddings(text, ranges, vis)
        self.assertEqual(merged[1], [1.0, 2.0])
        self.assertEqual(merged[2], [3.0, 4.0])
        self.assertEqual(merged[0], [0.0, 0.0], msg="非占位位置不应改变。")

    def test_count_mismatch_detected(self):
        with self.assertRaises(ValueError, msg="placeholder 与媒体数量不一致应报错。"):
            validate_placeholders([PlaceholderRange(0, 2, 0, "image")], [2, 2], 4)

    def test_length_mismatch_detected(self):
        with self.assertRaises(ValueError, msg="visual token 长度不符应报错。"):
            validate_placeholders([PlaceholderRange(0, 3, 0, "image")], [2], 4)

    def test_overlap_detected(self):
        with self.assertRaises(ValueError, msg="占位区间重叠应报错。"):
            validate_placeholders(
                [PlaceholderRange(0, 2, 0, "image"), PlaceholderRange(1, 2, 1, "image")], [2, 2], 6)

    def test_out_of_bounds_detected(self):
        with self.assertRaises(ValueError, msg="越界应报错。"):
            validate_placeholders([PlaceholderRange(3, 2, 0, "image")], [2], 4)


if __name__ == "__main__":
    unittest.main()
