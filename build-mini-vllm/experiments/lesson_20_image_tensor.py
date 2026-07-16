"""Lesson 20 实验：图片如何变成 Tensor（resize/normalize/layout）。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result
from mini_vllm.model import matrix as M
from mini_vllm.multimodal.image_processor import TinyImageProcessor
from mini_vllm.multimodal.media import synth_image
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    h, w = cfg.get("h", 20), cfg.get("w", 24)
    img = synth_image(h, w, seed=cfg.get("seed", 1))
    proc = TinyImageProcessor(image_size=cfg.get("image_size", 16))
    out = proc.preprocess(img)
    with tracer.section("image processing"):
        tracer.event("input", h=h, w=w, channels=3, dtype="uint8[0..255]")
        tracer.event("resized", size=out.size)
        tracer.event("layouts", channel_last=[out.size, out.size, 3],
                     channel_first=[3, out.size, out.size])
    return {"orig": [h, w, 3], "resized_to": out.size,
            "hwc_shape": [out.size, out.size, 3], "chw_shape": [3, out.size, out.size],
            "sample_pixel_norm": round(out.pixels_norm_hwc[0][0][0], 4), "meta": out.meta}


def print_summary(r):
    print("\n" + "=" * 64)
    print("  Lesson 20 · 图片如何变成 Tensor —— 运行成功 ✓")
    print("=" * 64)
    print(f"  原图 (H,W,C) = {tuple(r['orig'])}（uint8 0..255）")
    print(f"  resize → {r['resized_to']}×{r['resized_to']}，normalize：(x/255 − mean)/std")
    print(f"  channel-last (H,W,C) = {tuple(r['hwc_shape'])}")
    print(f"  channel-first (C,H,W) = {tuple(r['chw_shape'])}  ← 很多视觉模型的约定")
    print(f"  示例归一化像素 [0][0][0] = {r['sample_pixel_norm']}")
    print("=" * 64)
    print("  下一步：python3 course.py check 20   或   Lesson 21 切 patch。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_20_image_tensor",
                    "configs/lesson_20_quick.json", "outputs/lesson_20").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
