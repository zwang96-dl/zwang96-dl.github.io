"""Lesson 24 实验：Multimodal Prefill 与 Text Decode（vision encoder 只跑一次）。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.config import VisionConfig
from mini_vllm.multimodal.runner import MultiModalRunner
from mini_vllm.multimodal.cache import ProcessorCache, EncoderOutputCache
from mini_vllm.multimodal import messages as msg
from mini_vllm.sampling import SamplingParams
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    model = load_model()
    runner = MultiModalRunner(model, VisionConfig(), processor_cache=ProcessorCache(),
                              encoder_cache=EncoderOutputCache())
    messages = [msg.user(msg.text(cfg.get("text", "Describe:")),
                         msg.image({"synth": cfg.get("image", {"h": 16, "w": 16, "seed": 1})}))]
    enc_before = runner.encoder_runs
    out = runner.generate(messages, max_new_tokens=cfg.get("max_new_tokens", 6),
                          sampling=SamplingParams())
    enc_after = runner.encoder_runs
    with tracer.section("mm prefill + decode"):
        tracer.event("encoder_runs", before=enc_before, after=enc_after)
        tracer.event("generated", ids=out["generated"])
    return {"input_ids_len": len(out["input_ids"]),
            "visual_token_counts": out["visual_token_counts"],
            "encoder_runs_total": enc_after,
            "encoder_runs_during_this_request": enc_after - enc_before,
            "generated": out["generated"], "decode_steps": len(out["generated"])}


def print_summary(r):
    print("\n" + "=" * 66)
    print("  Lesson 24 · Multimodal Prefill 与 Text Decode —— 运行成功 ✓")
    print("=" * 66)
    print(f"  多模态 prefill 序列长度 {r['input_ids_len']}（含 {sum(r['visual_token_counts'])} 视觉 token）")
    print(f"  decode 步数：{r['decode_steps']}，生成 {r['generated']}")
    print("-" * 66)
    print("  证据：")
    print(f"    ✓ 本请求只在 prefill 运行 {r['encoder_runs_during_this_request']} 次 vision encoder（=媒体数）")
    print("    ✓ decode 阶段是纯文本自回归，复用 LLM KV Cache，不再运行 vision encoder")
    print("    ✓ processor cache / encoder cache / LLM KV cache 是三个不同层级")
    print("=" * 66)
    print("  下一步：python3 course.py check 24   或   Lesson 25 多图与动态 visual token。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_24_mm_prefill",
                    "configs/lesson_24_quick.json", "outputs/lesson_24").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
