"""Lesson 21 实验：图片如何变成 patch token。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result
from mini_vllm.config import VisionConfig
from mini_vllm.model import matrix as M
from mini_vllm.model.transformer import Rng
from mini_vllm.multimodal.image_processor import TinyImageProcessor
from mini_vllm.multimodal.patch_embed import PatchEmbed
from mini_vllm.multimodal.media import synth_image
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    vc = VisionConfig(image_size=cfg.get("image_size", 16), patch_size=cfg.get("patch_size", 8))
    img = synth_image(20, 20, seed=cfg.get("seed", 1))
    chw = TinyImageProcessor(image_size=vc.image_size).preprocess(img).pixels_norm_chw
    pe = PatchEmbed(vc, Rng(vc.seed))
    patches = pe.flatten_patches(chw)          # (num_patches, patch_dim)
    embeds = pe(chw)                            # (num_patches, vision_hidden)
    with tracer.section("patchify"):
        tracer.event("grid", grid=f"{vc.grid}x{vc.grid}", num_patches=vc.num_patches,
                     patch_dim=pe.patch_dim, vision_hidden=vc.vision_hidden_size)
    return {"image_size": vc.image_size, "patch_size": vc.patch_size, "grid": vc.grid,
            "num_patches": vc.num_patches, "patch_dim": pe.patch_dim,
            "patches_shape": list(M.shape(patches)), "embeds_shape": list(M.shape(embeds))}


def print_summary(r):
    print("\n" + "=" * 64)
    print("  Lesson 21 · 图片如何变成 Patch Token —— 运行成功 ✓")
    print("=" * 64)
    print(f"  {r['image_size']}×{r['image_size']} 图，patch={r['patch_size']} → grid {r['grid']}×{r['grid']}")
    print(f"  visual token 数 = grid×grid = {r['num_patches']}")
    print(f"  每个 patch 展平维度 = 3×{r['patch_size']}×{r['patch_size']} = {r['patch_dim']}")
    print(f"  patches {tuple(r['patches_shape'])} --线性投影--> patch embedding {tuple(r['embeds_shape'])}")
    print("=" * 64)
    print("  下一步：python3 course.py check 21   或   Lesson 22 vision encoder + projector。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_21_image_patch",
                    "configs/lesson_21_quick.json", "outputs/lesson_21").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
