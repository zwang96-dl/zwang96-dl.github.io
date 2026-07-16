"""MultiModalBudget —— 多模态调度预算（Lesson 28 的 Build 目标）。

多模态请求的 prefill 不只有文本 token，还有大量 **visual token**（每张图/每帧若干个），
而视觉编码本身也很贵。所以多模态调度要同时约束：

    text_token_budget    —— 一次迭代处理的文本 token 上限
    visual_token_budget  —— 一次迭代处理的 visual token 上限
    encoder_budget       —— 一次迭代运行 vision encoder 的媒体数上限（编码算力）
    max_num_seqs         —— 并发请求数上限

一个多模态请求 prefill 的成本 = 文本 token 数 + visual token 数（因为 visual token 也占
序列位置、也进 KV）。visual token 命中 encoder 缓存时，编码成本可省，但序列位置仍占。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MultiModalBudget:
    text_token_budget: int = 64
    visual_token_budget: int = 64
    encoder_budget: int = 4          # 一次迭代最多编码多少个媒体
    max_num_seqs: int = 4

    def fits(self, text_tokens: int, visual_tokens: int, num_media_to_encode: int,
             used_text: int, used_visual: int, used_encoder: int) -> bool:
        """判断在当前已用量下，是否还能接纳一个 (text, visual, encode) 的 prefill。"""
        return (used_text + text_tokens <= self.text_token_budget and
                used_visual + visual_tokens <= self.visual_token_budget and
                used_encoder + num_media_to_encode <= self.encoder_budget)
