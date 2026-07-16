"""TinyImageProcessor —— 把图片变成模型能吃的 Tensor（Lesson 20 的 Build 目标）。

图片在这里用最朴素的方式表示：``list[H][W][3]`` 的整数（0–255）RGB 像素，纯 Python，
无需 Pillow。处理流程（与真实 image processor 概念一致）：

    原图 (H, W, 3)
      → resize 到目标边长（最近邻，简单可懂）
      → center crop / pad 到正方形
      → normalize：像素 / 255，再按 mean/std 标准化
      → channel-first 布局 (3, H, W)（很多视觉模型的约定）

输出既给出 channel-last 也给出 channel-first，方便对照理解「布局」这件事。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProcessedImage:
    size: int                      # 处理后的边长
    pixels_norm_hwc: list          # (H, W, 3) 归一化后
    pixels_norm_chw: list          # (3, H, W) 归一化后
    meta: dict = field(default_factory=dict)


class TinyImageProcessor:
    def __init__(self, image_size: int = 16,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> None:
        self.image_size = image_size
        self.mean = mean
        self.std = std

    # ------------------------------------------------------------------ #
    def _resize_nearest(self, img: list, size: int) -> list:
        """最近邻 resize 到 (size, size, 3)。"""
        h = len(img)
        w = len(img[0]) if h else 0
        out = []
        for i in range(size):
            src_i = min(h - 1, int(i * h / size)) if h else 0
            row = []
            for j in range(size):
                src_j = min(w - 1, int(j * w / size)) if w else 0
                row.append(list(img[src_i][src_j]))
            out.append(row)
        return out

    def preprocess(self, img: list) -> ProcessedImage:
        """图片 (H,W,3, 0..255) → ProcessedImage。"""
        s = self.image_size
        resized = self._resize_nearest(img, s)
        # normalize
        hwc = [[[ (resized[i][j][c] / 255.0 - self.mean[c]) / self.std[c]
                  for c in range(3)] for j in range(s)] for i in range(s)]
        # channel-first
        chw = [[[hwc[i][j][c] for j in range(s)] for i in range(s)] for c in range(3)]
        return ProcessedImage(size=s, pixels_norm_hwc=hwc, pixels_norm_chw=chw,
                              meta={"orig_h": len(img), "orig_w": len(img[0]) if img else 0,
                                    "resized_to": s, "mean": list(self.mean), "std": list(self.std)})
