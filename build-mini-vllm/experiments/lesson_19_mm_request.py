"""Lesson 19 实验：多模态请求经历的阶段（结构化消息 → 模板 → token+占位 → 媒体）。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.multimodal.runner import MultiModalRunner
from mini_vllm.multimodal.chat_template import MultiModalChatTemplate
from mini_vllm.multimodal import messages as msg
from mini_vllm.trace import Tracer


def _messages(cfg):
    return [msg.user(msg.text(cfg.get("text", "What is in this image?")),
                     msg.image({"synth": cfg.get("image", {"h": 16, "w": 16, "seed": 1})}))]


def run_experiment(cfg, tracer):
    model = load_model(); runner = MultiModalRunner(model)
    messages = _messages(cfg)
    segments, rendered = MultiModalChatTemplate().render(messages)
    input_ids, ranges, vembeds, meta = runner.build_inputs(messages)
    with tracer.section("multimodal request stages"):
        tracer.event("rendered", text=rendered.replace("\n", "\\n"))
        tracer.event("tokens", input_ids_len=len(input_ids))
        tracer.event("placeholders", ranges=[(r.offset, r.length, r.modality) for r in ranges])
    return {"rendered": rendered, "num_segments": len(segments),
            "num_media": len(ranges), "input_ids_len": len(input_ids),
            "placeholder_ranges": [{"offset": r.offset, "length": r.length,
                                    "modality": r.modality} for r in ranges],
            "visual_token_counts": [len(v) for v in vembeds]}


def print_summary(r):
    print("\n" + "=" * 66)
    print("  Lesson 19 · 多模态请求是什么 —— 运行成功 ✓")
    print("=" * 66)
    print(f"  chat template 渲染：{r['rendered']!r}")
    print(f"  文本段/媒体数：{r['num_segments']} 段 / {r['num_media']} 个媒体")
    print(f"  token 序列长度：{r['input_ids_len']}（含 {sum(r['visual_token_counts'])} 个视觉占位）")
    print(f"  placeholder 区间：{r['placeholder_ranges']}")
    print("-" * 66)
    print("  证据（阶段边界）：")
    print("    ✓ 图片引用不是 embedding；chat template 不编码像素；tokenizer 不处理像素")
    print("    ✓ 占位 token 稍后才被视觉 embedding 替换（Lesson 23）")
    print("=" * 66)
    print("  下一步：python3 course.py check 19   或   Lesson 20 图片如何变成 Tensor。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_19_mm_request",
                    "configs/lesson_19_quick.json", "outputs/lesson_19").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
