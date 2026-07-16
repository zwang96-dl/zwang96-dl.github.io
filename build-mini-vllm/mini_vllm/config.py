"""模型与引擎配置。

这些 dataclass 是整套系统的「旋钮」。课程里所有关于 shape、内存、
调度预算的讨论，最终都落到这几个字段上。Phase 1 只用到其中一部分，
但把它们集中定义好，后续 Lesson 才能在同一套配置上演化。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """极小 decoder-only Transformer 的配置（课程推荐的文本配置）。"""

    vocab_size: int = 259
    hidden_size: int = 32
    num_layers: int = 2
    num_attention_heads: int = 4
    num_kv_heads: int = 2  # < num_attention_heads → GQA（Lesson 7）
    intermediate_size: int = 64  # SwiGLU 中间维度（通常 2~4 倍 hidden）
    max_seq_len: int = 64
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    seed: int = 1234  # 确定性初始化种子（保证 checkpoint 可复现、可离线重建）

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        return self.hidden_size // self.num_attention_heads

    @property
    def group_size(self) -> int:
        """GQA：每个 KV head 被多少个 query head 共享。"""
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_kv_heads 整除")
        return self.num_attention_heads // self.num_kv_heads


@dataclass
class VisionConfig:
    """极小 vision encoder 的配置（课程推荐的视觉配置，多模态 Phase 用）。"""

    image_size: int = 16       # 小图以保证纯 Python 处理速度（grid 2×2 → 4 个 visual token）
    patch_size: int = 8
    vision_hidden_size: int = 48
    vision_layers: int = 1
    vision_heads: int = 4
    text_hidden_size: int = 32  # 与 tiny text model 的 hidden_size 对齐
    rms_norm_eps: float = 1e-5
    seed: int = 4321

    @property
    def grid(self) -> int:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size 必须能被 patch_size 整除")
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid * self.grid


@dataclass
class EngineConfig:
    """推理引擎 / 调度器的配置（Lesson 11、13、15 会逐步用到）。"""

    block_size: int = 16               # 每个 KV block 容纳的 token 数（Lesson 14）
    num_blocks: int = 64               # 物理 block 总数（Lesson 13）
    max_num_seqs: int = 8              # 一个 batch 最多同时跑几个序列
    max_num_batched_tokens: int = 256  # 一次迭代的 token 预算（Lesson 15）
    scheduler_policy: str = "fifo"     # fifo | decode-first | sjf | balanced


# 课程默认使用的三套「规模」（Quick / Normal / Stress）。
# Stress 也刻意保持在普通 MacBook Air 内存之内。
RUN_MODES = {
    "quick": {"description": "数秒级，用于快速验证", "scale": 1},
    "normal": {"description": "默认规模", "scale": 4},
    "stress": {"description": "更大规模，但不会耗尽 8GB 内存", "scale": 16},
}
