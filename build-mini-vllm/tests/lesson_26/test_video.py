"""Lesson 26 检查：视频抽帧策略、帧数、时间戳与顺序。"""
import unittest
from mini_vllm.multimodal.media import synth_image, VideoFrame
from mini_vllm.multimodal.video_sampler import VideoFrameSampler


class TestVideoSampler(unittest.TestCase):
    def setUp(self):
        self.fps = 2.0
        self.frames = [VideoFrame(synth_image(8, 8, k), k, k / self.fps) for k in range(10)]
        self.s = VideoFrameSampler()

    def test_uniform_count_and_order(self):
        out = self.s.sample(self.frames, "uniform", num_frames=4)
        self.assertLessEqual(len(out), 4)
        idxs = [f.frame_index for f in out]
        self.assertEqual(idxs, sorted(idxs), msg="抽帧应保持时间顺序。")
        self.assertEqual(idxs[0], 0, msg="uniform 应含首帧。")

    def test_head_and_tail(self):
        head = self.s.sample(self.frames, "head", num_frames=3)
        tail = self.s.sample(self.frames, "tail", num_frames=3)
        self.assertEqual([f.frame_index for f in head], [0, 1, 2])
        self.assertEqual([f.frame_index for f in tail], [7, 8, 9])

    def test_timestamps_preserved(self):
        out = self.s.sample(self.frames, "head", num_frames=2)
        tl = self.s.timeline(out)
        self.assertEqual(tl[1]["timestamp"], 1 / self.fps,
                         msg="每帧的 presentation timestamp 应被保留。")

    def test_fixed_fps(self):
        out = self.s.sample(self.frames, "fixed_fps", fps=1.0, source_fps=2.0)
        self.assertEqual([f.frame_index for f in out], [0, 2, 4, 6, 8],
                         msg="源 2fps 抽到 1fps 应每隔一帧取一帧。")


if __name__ == "__main__":
    unittest.main()
