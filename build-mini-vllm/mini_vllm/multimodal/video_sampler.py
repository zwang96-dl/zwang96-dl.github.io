"""VideoFrameSampler —— 从帧序列里抽样（Lesson 26 的 Build 目标）。

视频进模型前要先**抽帧**（不可能每帧都算）。不同策略在「覆盖度」和「成本」之间权衡：

    uniform    —— 均匀抽 N 帧（覆盖全片）
    fixed_fps  —— 按固定帧率抽（时长越长帧越多）
    head       —— 取前 N 帧
    tail       —— 取后 N 帧

**重要**：每个被抽的帧都带 frame_index 与 timestamp（presentation timestamp）。
timestamp 是 processor 侧的 metadata——它不会被 LLM 自动理解，除非显式喂进 prompt。
"""

from __future__ import annotations

import math

from .media import VideoFrame


class VideoFrameSampler:
    def sample(self, frames: list[VideoFrame], strategy: str = "uniform",
               num_frames: int = 4, fps: float | None = None,
               source_fps: float = 1.0) -> list[VideoFrame]:
        n = len(frames)
        if n == 0:
            return []
        if strategy == "uniform":
            k = min(num_frames, n)
            idxs = [min(n - 1, round(i * (n - 1) / max(1, k - 1))) for i in range(k)] if k > 1 else [0]
            # 去重并保序
            seen, out = set(), []
            for i in idxs:
                if i not in seen:
                    seen.add(i); out.append(frames[i])
            return out
        if strategy == "fixed_fps":
            target = fps or 1.0
            step = max(1, int(round(source_fps / target)))
            return [frames[i] for i in range(0, n, step)]
        if strategy == "head":
            return frames[:min(num_frames, n)]
        if strategy == "tail":
            return frames[max(0, n - num_frames):]
        raise ValueError(f"未知抽帧策略：{strategy}（可选 uniform/fixed_fps/head/tail）")

    def timeline(self, sampled: list[VideoFrame]) -> list[dict]:
        """返回被抽帧的时间线 metadata（frame_index + timestamp）。"""
        return [{"frame_index": f.frame_index, "timestamp": f.timestamp_seconds}
                for f in sampled]
