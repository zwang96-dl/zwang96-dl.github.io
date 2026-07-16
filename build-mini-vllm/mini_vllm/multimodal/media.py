"""MediaLoader —— 从本地加载图片与视频帧（Lesson 20/26）。

图片以 JSON 存放：``{"h":H,"w":W,"pixels":[[[r,g,b],...],...]}``（RGB, 0..255）。
视频是一个帧目录，含 ``manifest.json``（fps、帧文件名、时间戳）与若干帧 JSON。
全部本地、离线、无需 Pillow。也可用 :func:`synth_image` 按种子确定性生成合成图片。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def synth_image(h: int, w: int, seed: int = 0) -> list:
    """确定性生成一张合成 RGB 图片（渐变 + 方块图案），用于教学、无需外部素材。"""
    img = []
    for i in range(h):
        row = []
        for j in range(w):
            r = (i * 255) // max(1, h - 1)
            g = (j * 255) // max(1, w - 1)
            b = ((i + j + seed) * 37) % 256
            row.append([r, g, b])
        img.append(row)
    return img


def load_image(path: str | Path) -> list:
    data = json.loads(Path(path).read_text("utf-8"))
    return data["pixels"]


def save_image(path: str | Path, pixels: list) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"h": len(pixels), "w": len(pixels[0]) if pixels else 0,
                             "pixels": pixels}), "utf-8")


@dataclass
class VideoFrame:
    pixels: list
    frame_index: int
    timestamp_seconds: float


def load_video(dir_path: str | Path) -> tuple[list[VideoFrame], dict]:
    """加载一个帧目录，返回 (frames, manifest)。"""
    d = Path(dir_path)
    manifest = json.loads((d / "manifest.json").read_text("utf-8"))
    frames = []
    for entry in manifest["frames"]:
        pixels = load_image(d / entry["file"])
        frames.append(VideoFrame(pixels=pixels, frame_index=entry["frame_index"],
                                 timestamp_seconds=entry["timestamp"]))
    return frames, manifest
