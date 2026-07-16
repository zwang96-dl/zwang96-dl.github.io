"""Lesson 22 实验：Tiny Vision Encoder 与 Projector（vision_hidden → text_hidden）。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result
from mini_vllm.config import VisionConfig
from mini_vllm.model import matrix as M
from mini_vllm.multimodal.image_processor import TinyImageProcessor
from mini_vllm.multimodal.vision_encoder import TinyVisionEncoder, MultimodalProjector
from mini_vllm.multimodal.media import synth_image
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    vc = VisionConfig()
    chw = TinyImageProcessor(image_size=vc.image_size).preprocess(
        synth_image(18, 18, seed=cfg.get("seed", 1))).pixels_norm_chw
    enc = TinyVisionEncoder(vc); proj = MultimodalProjector(vc)
    vis = enc.encode(chw)          # (num_patches, vision_hidden)
    projected = proj(vis)          # (num_patches, text_hidden)
    with tracer.section("vision encode + project"):
        tracer.event("vision_out", shape=list(M.shape(vis)), dim=vc.vision_hidden_size)
        tracer.event("projected", shape=list(M.shape(projected)), dim=vc.text_hidden_size)
    return {"num_patches": vc.num_patches, "vision_hidden": vc.vision_hidden_size,
            "text_hidden": vc.text_hidden_size,
            "vision_shape": list(M.shape(vis)), "projected_shape": list(M.shape(projected))}


def print_summary(r):
    print("\n" + "=" * 64)
    print("  Lesson 22 · Tiny Vision Encoder 与 Projector —— 运行成功 ✓")
    print("=" * 64)
    print(f"  patch embedding → vision encoder（非因果自注意力）→ {tuple(r['vision_shape'])}"
          f"（vision_hidden={r['vision_hidden']}）")
    print(f"  projector：vision_hidden {r['vision_hidden']} → text_hidden {r['text_hidden']}")
    print(f"  投影后视觉 embedding：{tuple(r['projected_shape'])}  ← 已在文本模型的 hidden 空间")
    print("-" * 64)
    print("  证据：视觉 embedding 是连续向量（不是 token id）；projector 负责维度对齐。")
    print("=" * 64)
    print("  下一步：python3 course.py check 22   或   Lesson 23 placeholder 与合并。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_22_vision_encoder",
                    "configs/lesson_22_quick.json", "outputs/lesson_22").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
