"""Lesson 27 实验：多模态三层缓存（processor / encoder / LLM KV），以及避免 stale cache。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.config import VisionConfig
from mini_vllm.multimodal.runner import MultiModalRunner
from mini_vllm.multimodal.cache import ProcessorCache, EncoderOutputCache
from mini_vllm.multimodal import messages as msg
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    model = load_model()
    pc, ec = ProcessorCache(), EncoderOutputCache()
    runner = MultiModalRunner(model, VisionConfig(), processor_cache=pc, encoder_cache=ec)
    img = {"synth": {"h": 16, "w": 16, "seed": 7}}
    m = [msg.user(msg.image(img))]

    r1_runs = runner.encoder_runs
    runner.build_inputs(m); miss_runs = runner.encoder_runs - r1_runs   # 首次：miss，编码
    runner.build_inputs(m); hit_runs = runner.encoder_runs - r1_runs - miss_runs  # 再次：命中，不编码

    # stale cache 防护：换 encoder identity（不同 seed）→ key 变化 → 不会误命中旧结果
    runner2 = MultiModalRunner(model, VisionConfig(seed=9999),
                               processor_cache=pc, encoder_cache=ec)
    before = runner2.encoder_runs
    runner2.build_inputs(m)
    stale_avoided = (runner2.encoder_runs - before) == 1   # 重新编码而非误命中

    with tracer.section("three caches"):
        tracer.event("encoder_cache", **ec.stats())
        tracer.event("processor_cache", **pc.stats())
    return {"encoder_runs_first": miss_runs, "encoder_runs_second": hit_runs,
            "encoder_cache": ec.stats(), "processor_cache": pc.stats(),
            "stale_cache_avoided": stale_avoided}


def print_summary(r):
    print("\n" + "=" * 66)
    print("  Lesson 27 · 多模态三层缓存 —— 运行成功 ✓")
    print("=" * 66)
    print(f"  同一张图第一次编码次数={r['encoder_runs_first']}（miss），第二次={r['encoder_runs_second']}（hit，跳过编码）")
    print(f"  encoder cache: {r['encoder_cache']}")
    print(f"  processor cache: {r['processor_cache']}")
    print("-" * 66)
    print("  三层缓存各司其职：ProcessorCache（预处理）/ EncoderOutputCache（视觉 embedding）/ LLM KV Cache（文本上下文）。")
    print(f"  ✓ 换 encoder 身份后不会误命中旧缓存（stale cache 已避免）：{r['stale_cache_avoided']}")
    print("=" * 66)
    print("  下一步：python3 course.py check 27   或   Lesson 28 多模态调度。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_27_three_cache",
                    "configs/lesson_27_quick.json", "outputs/lesson_27").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
