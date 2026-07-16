"""MultiModalInputParser —— 把「文本段 + 媒体（已知 visual token 数）」拼成 token 序列。

给定 chat template 渲染出的 segments，以及每段媒体已经算好的 visual token 数，
本解析器负责：
    - 对文本段做 byte-level tokenize；
    - 在媒体处插入相应数量的**占位 token**（用 PAD 作占位，稍后被视觉 embedding 替换）；
    - 记录每段媒体的 PlaceholderRange（offset/length/media_index/modality）。
返回 (input_ids, placeholder_ranges)。
"""

from __future__ import annotations

from ..tokenizer import ByteTokenizer, PAD_ID
from .placeholders import PlaceholderRange


class MultiModalInputParser:
    def __init__(self, tokenizer: ByteTokenizer | None = None, add_bos: bool = True) -> None:
        self.tok = tokenizer or ByteTokenizer()
        self.add_bos = add_bos

    def parse(self, segments: list, media_visual_lens: list[int]):
        """segments: [("text",str)|("media",dict)]；media_visual_lens: 每段媒体的 visual token 数。"""
        input_ids: list[int] = []
        ranges: list[PlaceholderRange] = []
        if self.add_bos:
            input_ids.append(self.tok.bos_id)
        media_i = 0
        for kind, payload in segments:
            if kind == "text":
                input_ids.extend(self.tok.encode(payload, add_bos=False))
            else:  # media
                n = media_visual_lens[media_i]
                offset = len(input_ids)
                input_ids.extend([PAD_ID] * n)      # 占位 token
                ranges.append(PlaceholderRange(offset=offset, length=n,
                                               media_index=media_i, modality=payload["modality"]))
                media_i += 1
        if media_i != len(media_visual_lens):
            raise ValueError(f"媒体数量不一致：segments 中有 {media_i} 段媒体，"
                             f"但提供了 {len(media_visual_lens)} 个 visual 长度。")
        return input_ids, ranges
