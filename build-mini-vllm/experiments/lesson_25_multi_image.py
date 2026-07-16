"""Lesson 25 实验：多图片与动态 visual token（不同请求的视觉 token 数不同）。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.multimodal.runner import MultiModalRunner
from mini_vllm.multimodal import messages as msg
from mini_vllm.multimodal.media import synth_image
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    model = load_model(); runner = MultiModalRunner(model)
    img = lambda s: {"synth": {"h": 16, "w": 16, "seed": s}}
    reqs = {
        "text_only": [msg.user(msg.text("just text"))],
        "one_image": [msg.user(msg.text("A:"), msg.image(img(1)))],
        "two_images": [msg.user(msg.image(img(2)), msg.text("vs"), msg.image(img(3)))],
        "video_3f": [msg.user(msg.video({"frames": [synth_image(16, 16, k) for k in range(6)],
                                         "fps": 2.0, "num_frames": 3}))],
    }
    result = {}
    with tracer.section("dynamic visual tokens"):
        for name, m in reqs.items():
            ids, ranges, vembeds, meta = runner.build_inputs(m)
            total = sum(len(v) for v in vembeds)
            result[name] = {"num_media": len(ranges), "visual_tokens": total,
                            "per_media": [len(v) for v in vembeds], "seq_len": len(ids)}
            tracer.event(name, media=len(ranges), visual_tokens=total)
    return {"requests": result}


def print_summary(r):
    print("\n" + "=" * 66)
    print("  Lesson 25 · 多图片与动态 Visual Token —— 运行成功 ✓")
    print("=" * 66)
    print(f"  {'请求':<12}{'媒体数':>6}{'视觉token':>10}{'各媒体':>12}{'序列长':>8}")
    for name, d in r["requests"].items():
        print(f"  {name:<12}{d['num_media']:>6}{d['visual_tokens']:>10}"
              f"{str(d['per_media']):>12}{d['seq_len']:>8}")
    print("-" * 66)
    print("  证据：不同请求的视觉 token 数不同（0/4/8/12）→ 序列长度与 KV 需求也不同。")
    print("        真实 VLM 里，不同分辨率的图还会产生不同的 per-image token 数。")
    print("=" * 66)
    print("  下一步：python3 course.py check 25   或   Lesson 26 视频抽帧。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_25_multi_image",
                    "configs/lesson_25_quick.json", "outputs/lesson_25").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
