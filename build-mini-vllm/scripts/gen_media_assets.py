#!/usr/bin/env python3
"""生成本地多模态素材（合成图片 + 视频帧目录），全部离线、确定性、无需 Pillow。

产出：
    assets/images/demo_a.json, demo_b.json        —— 单张合成图片
    assets/videos/demo/manifest.json + frame_*.json —— 一个帧目录（含 fps 与 timestamp）

课程实验默认用内存里的合成图片（synth_image）即可离线运行；这些本地文件用于演示
「从本地路径加载媒体」以及让离线打包包含真实媒体素材。
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mini_vllm.multimodal.media import synth_image, save_image  # noqa: E402
import json  # noqa: E402


def main() -> int:
    save_image(ROOT / "assets/images/demo_a.json", synth_image(16, 16, seed=11))
    save_image(ROOT / "assets/images/demo_b.json", synth_image(16, 16, seed=22))

    vdir = ROOT / "assets/videos/demo"
    fps = 2.0
    frames = []
    for k in range(6):
        fname = f"frame_{k:03d}.json"
        save_image(vdir / fname, synth_image(16, 16, seed=100 + k))
        frames.append({"file": fname, "frame_index": k, "timestamp": k / fps})
    (vdir / "manifest.json").write_text(
        json.dumps({"fps": fps, "num_frames": len(frames), "frames": frames},
                   indent=2), "utf-8")
    print("  generated assets/images/demo_a.json, demo_b.json")
    print(f"  generated assets/videos/demo/ ({len(frames)} frames + manifest.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
