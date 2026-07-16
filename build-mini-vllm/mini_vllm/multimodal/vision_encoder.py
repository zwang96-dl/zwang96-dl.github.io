"""TinyVisionEncoder + MultimodalProjector（Lesson 22 的 Build 目标）。

TinyVisionEncoder：在 patch embedding 上做若干层（本课 1 层）「双向」自注意力 + 前馈
（视觉编码器**不加 causal mask**——图片里 patch 之间可以互相看）。输出仍是
(num_patches, vision_hidden)。

MultimodalProjector：把视觉的 vision_hidden 维**投影到文本模型的 text_hidden 维**，
使视觉 embedding 能和文本 embedding 拼在同一个空间里（Lesson 23 合并时用）。

要点：视觉 embedding **不是 token id**——它是连续向量，直接进入文本模型的 hidden 空间。
"""

from __future__ import annotations

from ..config import VisionConfig
from ..model import matrix as M
from ..model import attention_ref as A
from ..model.rmsnorm import rms_norm
from ..model.mlp import swiglu
from ..model.transformer import Rng, _rand_matrix
from .patch_embed import PatchEmbed


def _full_attention(q: M.Matrix, k: M.Matrix, v: M.Matrix, num_heads: int, hd: int) -> M.Matrix:
    """多头**非因果**注意力（patch 之间互相可见）。"""
    seq = len(q)
    out = M.zeros(seq, num_heads * hd)
    for h in range(num_heads):
        Qh = [r[h * hd:(h + 1) * hd] for r in q]
        Kh = [r[h * hd:(h + 1) * hd] for r in k]
        Vh = [r[h * hd:(h + 1) * hd] for r in v]
        oh = A.scaled_dot_product_attention(Qh, Kh, Vh, causal=False)
        for i in range(seq):
            for d in range(hd):
                out[i][h * hd + d] = oh[i][d]
    return out


class TinyVisionEncoder:
    def __init__(self, cfg: VisionConfig | None = None) -> None:
        self.cfg = cfg or VisionConfig()
        c = self.cfg
        rng = Rng(c.seed)
        self.patch_embed = PatchEmbed(c, rng)
        vh = c.vision_hidden_size
        self.hd = vh // c.vision_heads
        self.inter = vh * 2
        self.layers = []
        for _ in range(c.vision_layers):
            self.layers.append({
                "ln1": [1.0] * vh,
                "wq": _rand_matrix(rng, vh, vh, 0.05),
                "wk": _rand_matrix(rng, vh, vh, 0.05),
                "wv": _rand_matrix(rng, vh, vh, 0.05),
                "wo": _rand_matrix(rng, vh, vh, 0.05),
                "ln2": [1.0] * vh,
                "w_gate": _rand_matrix(rng, vh, self.inter, 0.05),
                "w_up": _rand_matrix(rng, vh, self.inter, 0.05),
                "w_down": _rand_matrix(rng, self.inter, vh, 0.05),
            })

    def encode(self, chw: list) -> M.Matrix:
        """(3, S, S) → (num_patches, vision_hidden)。"""
        c = self.cfg
        h = self.patch_embed(chw)                  # (num_patches, vision_hidden)
        for layer in self.layers:
            hn = rms_norm(h, layer["ln1"], c.rms_norm_eps)
            q = M.matmul(hn, layer["wq"])
            k = M.matmul(hn, layer["wk"])
            v = M.matmul(hn, layer["wv"])
            attn = _full_attention(q, k, v, c.vision_heads, self.hd)
            h = M.add(h, M.matmul(attn, layer["wo"]))
            hn2 = rms_norm(h, layer["ln2"], c.rms_norm_eps)
            h = M.add(h, swiglu(hn2, layer["w_gate"], layer["w_up"], layer["w_down"]))
        return h


class MultimodalProjector:
    """把 vision_hidden 维投影到 text_hidden 维（两层 + 中间激活）。"""

    def __init__(self, cfg: VisionConfig | None = None) -> None:
        self.cfg = cfg or VisionConfig()
        c = self.cfg
        rng = Rng(c.seed + 999)
        self.w1 = _rand_matrix(rng, c.vision_hidden_size, c.text_hidden_size, 0.05)
        self.w2 = _rand_matrix(rng, c.text_hidden_size, c.text_hidden_size, 0.05)

    def __call__(self, visual: M.Matrix) -> M.Matrix:
        """(num_patches, vision_hidden) → (num_patches, text_hidden)。"""
        from ..model.mlp import silu
        h = M.matmul(visual, self.w1)
        h = [[silu(x) for x in row] for row in h]
        return M.matmul(h, self.w2)
