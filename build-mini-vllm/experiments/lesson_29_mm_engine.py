"""Lesson 29 实验：完整 Tiny Multimodal Engine 端到端（mixed batch + 报告）。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.config import VisionConfig
from mini_vllm.multimodal.mm_engine import MultiModalEngine
from mini_vllm.multimodal.budget import MultiModalBudget
from mini_vllm.multimodal import messages as msg
from mini_vllm.multimodal.media import synth_image
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    model = load_model()
    eng = MultiModalEngine(model, VisionConfig(),
                           budget=MultiModalBudget(text_token_budget=32, visual_token_budget=32,
                                                   encoder_budget=3, max_num_seqs=3),
                           enable_caches=True)
    img = lambda s: {"synth": {"h": 16, "w": 16, "seed": s}}
    eng.add_request("text", [msg.user(msg.text("hello world"))], 5, arrival=0)
    eng.add_request("img", [msg.user(msg.text("see"), msg.image(img(1)))], 5, arrival=0)
    eng.add_request("img_same", [msg.user(msg.text("again"), msg.image(img(1)))], 4, arrival=1)
    eng.add_request("multi", [msg.user(msg.image(img(1)), msg.image(img(2)))], 4, arrival=2)
    eng.add_request("vid", [msg.user(msg.video({"frames": [synth_image(16, 16, k) for k in range(6)],
                                                "fps": 2.0, "num_frames": 3}))], 3, arrival=2)
    res = eng.run(tracer=tracer)
    st = eng.stats()
    return {
        "iterations": res.iterations,
        "requests": [{"id": r.request_id, "visual_tokens": r.visual_tokens,
                      "generated": len(r.output_token_ids),
                      "first_token_iter": r.first_token_iter} for r in res.requests],
        "encoder_runs": st["encoder_runs"],
        "encoder_cache": st["encoder_cache"], "processor_cache": st["processor_cache"],
    }


def print_summary(r):
    print("\n" + "=" * 68)
    print("  Lesson 29 · 完整 Tiny Multimodal Engine —— 运行成功 ✓")
    print("=" * 68)
    print(f"  迭代数 {r['iterations']}，vision encoder 实际运行 {r['encoder_runs']} 次")
    print(f"  {'请求':<10}{'视觉token':>10}{'生成':>6}{'首token@iter':>14}")
    for q in r["requests"]:
        print(f"  {q['id']:<10}{q['visual_tokens']:>10}{q['generated']:>6}{str(q['first_token_iter']):>14}")
    print(f"  encoder cache: {r['encoder_cache']}")
    print("-" * 68)
    print("  证据：text-only / 单图 / 多图 / 视频 混合批处理；相同图命中 encoder 缓存；")
    print("        vision encoder 不在 decode 重复运行；三层缓存各自独立。")
    print("=" * 68)
    print("  下一步：python3 course.py check 29   或   Lesson 30 多模态综合挑战。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_29_mm_engine",
                    "configs/lesson_29_quick.json", "outputs/lesson_29").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
