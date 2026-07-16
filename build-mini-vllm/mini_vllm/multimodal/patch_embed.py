"""PatchEmbed —— 图片切 patch 并线性投影成 patch embedding（Lesson 21 的 Build 目标）。

一张 (3, S, S) 的图片，按 patch_size 切成 grid×grid 个不重叠的小块；每块展平成一个
向量（3·patch·patch 维），再经一个线性层投影到 vision_hidden 维——就得到「visual token」。
visual token 数 = grid×grid（例如 16×16 图、8×8 patch → 2×2 = 4 个 visual token）。
"""

from __future__ import annotations

from ..config import VisionConfig
from ..model import matrix as M
from ..model.transformer import Rng, _rand_matrix


class PatchEmbed:
    def __init__(self, cfg: VisionConfig, rng: Rng) -> None:
        self.cfg = cfg
        self.patch_dim = 3 * cfg.patch_size * cfg.patch_size
        self.proj = _rand_matrix(rng, self.patch_dim, cfg.vision_hidden_size, 0.05)  # (patch_dim, vhid)

    def num_patches(self) -> int:
        return self.cfg.grid * self.cfg.grid

    def flatten_patches(self, chw: list) -> M.Matrix:
        """(3, S, S) → (num_patches, patch_dim)，每行是一个展平的 patch。"""
        cfg = self.cfg
        ps, g = cfg.patch_size, cfg.grid
        rows = []
        for pi in range(g):
            for pj in range(g):
                vec = []
                for c in range(3):
                    for i in range(ps):
                        for j in range(ps):
                            vec.append(chw[c][pi * ps + i][pj * ps + j])
                rows.append(vec)
        return rows

    def __call__(self, chw: list) -> M.Matrix:
        """(3, S, S) → (num_patches, vision_hidden)。"""
        patches = self.flatten_patches(chw)
        return M.matmul(patches, self.proj)
