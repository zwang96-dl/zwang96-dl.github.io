"""Placeholder 与对齐校验（Lesson 23 的 Build 目标之一）。

在 token 序列里，每段媒体（图/视频）占用一段连续的**占位 token**（本课用 PAD 作为占位），
稍后这段会被该媒体的视觉 embedding 逐位替换。PlaceholderRange 精确记录这段的位置与归属。

对齐是多模态最容易出错、也最需要守护的地方——占位符数量必须与媒体数量一致、每段长度
必须等于该媒体的 visual token 数、区间不能越界或重叠、顺序要与媒体顺序一致。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlaceholderRange:
    offset: int        # 在 input_ids 中的起始位置
    length: int        # 占用多少个 token（= 该媒体的 visual token 数）
    media_index: int   # 对应第几个媒体
    modality: str      # "image" | "video"


def validate_placeholders(ranges: list[PlaceholderRange], media_visual_lens: list[int],
                          seq_len: int) -> None:
    """校验占位区间与媒体严格对齐；任何不一致都抛出**带定位信息**的错误。"""
    if len(ranges) != len(media_visual_lens):
        raise ValueError(
            f"Placeholder 与媒体数量不一致：placeholders={len(ranges)}, "
            f"media items={len(media_visual_lens)}。"
            "检查 mini_vllm/multimodal/inputs.py 的 parse()。")
    prev_end = -1
    for k, r in enumerate(ranges):
        if r.media_index != k:
            raise ValueError(f"媒体顺序错乱：第 {k} 个 placeholder 的 media_index={r.media_index}。")
        if r.length != media_visual_lens[k]:
            raise ValueError(
                f"visual token 长度错误：媒体 {k} 需要 {media_visual_lens[k]} 个 visual token，"
                f"但 placeholder 长度为 {r.length}。")
        if r.offset < 0 or r.offset + r.length > seq_len:
            raise ValueError(f"placeholder 越界：媒体 {k} [{r.offset}, {r.offset + r.length}) "
                             f"超出序列长度 {seq_len}。")
        if r.offset <= prev_end:
            raise ValueError(f"placeholder 区间重叠：媒体 {k} 起点 {r.offset} <= 上一段终点 {prev_end}。")
        prev_end = r.offset + r.length - 1
