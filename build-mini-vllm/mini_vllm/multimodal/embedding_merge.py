"""文本 / 视觉 embedding 合并（Lesson 23 的 Build 目标之二）。

先按 token id 查出**文本 embedding**（占位 token 的 embedding 是无所谓的），再把每段
placeholder 位置的 embedding **逐位替换**成对应媒体的视觉 embedding。合并后的
(seq, hidden) 直接喂给文本模型的 forward(inputs_embeds=...) 做多模态 prefill。
"""

from __future__ import annotations

from ..model import matrix as M
from .placeholders import PlaceholderRange, validate_placeholders


def merge_multimodal_embeddings(text_embeds: M.Matrix,
                                ranges: list[PlaceholderRange],
                                visual_embeds: list[M.Matrix]) -> M.Matrix:
    """把视觉 embedding 合并进文本 embedding。

    参数
    ----
    text_embeds: (seq, hidden)，按 input_ids 查表得到。
    ranges: 每段媒体的 placeholder 区间（已按出现顺序）。
    visual_embeds: 与 ranges 一一对应，visual_embeds[k] 形状 (ranges[k].length, hidden)。
    """
    seq = len(text_embeds)
    hidden = len(text_embeds[0]) if seq else 0
    validate_placeholders(ranges, [len(v) for v in visual_embeds], seq)
    out = [list(row) for row in text_embeds]
    for k, r in enumerate(ranges):
        vis = visual_embeds[k]
        if vis and len(vis[0]) != hidden:
            raise ValueError(
                f"视觉 embedding 维度 {len(vis[0])} 与文本 hidden {hidden} 不一致（媒体 {k}）——"
                "多半是 projector 的输出维度没对齐 text_hidden。")
        for i in range(r.length):
            out[r.offset + i] = list(vis[i])
    return out
