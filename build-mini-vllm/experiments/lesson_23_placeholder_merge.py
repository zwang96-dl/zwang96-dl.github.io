"""Lesson 23 实验：Placeholder 与 Embedding Merge（含对齐错误检测）。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.model import matrix as M
from mini_vllm.multimodal.runner import MultiModalRunner
from mini_vllm.multimodal.placeholders import PlaceholderRange, validate_placeholders
from mini_vllm.multimodal.embedding_merge import merge_multimodal_embeddings
from mini_vllm.multimodal import messages as msg
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    model = load_model(); runner = MultiModalRunner(model)
    messages = [msg.user(msg.text("Left"), msg.image({"synth": {"h": 16, "w": 16, "seed": 1}}),
                         msg.text("Right"), msg.image({"synth": {"h": 16, "w": 16, "seed": 2}}))]
    input_ids, ranges, vembeds, meta = runner.build_inputs(messages)
    text_embeds = [list(model.embed[t]) for t in input_ids]
    merged = merge_multimodal_embeddings(text_embeds, ranges, vembeds)
    # 验证：合并后占位处的 embedding 确实等于视觉 embedding
    ok = True
    for k, rg in enumerate(ranges):
        for i in range(rg.length):
            if merged[rg.offset + i] != vembeds[k][i]:
                ok = False
    # 故意制造对齐错误，确认被检测
    errs = {}
    try:
        validate_placeholders(ranges[:1], [len(v) for v in vembeds], len(input_ids))
    except ValueError as e:
        errs["count_mismatch"] = str(e).split("。")[0]
    try:
        bad = [PlaceholderRange(rg.offset, rg.length + 1, rg.media_index, rg.modality) for rg in ranges]
        validate_placeholders(bad, [len(v) for v in vembeds], len(input_ids))
    except ValueError as e:
        errs["length_mismatch"] = str(e).split("。")[0]
    with tracer.section("merge"):
        tracer.event("ranges", ranges=[(r.offset, r.length, r.media_index) for r in ranges])
        tracer.event("merged_shape", shape=list(M.shape(merged)))
    return {"num_media": len(ranges), "visual_token_counts": [len(v) for v in vembeds],
            "merged_shape": list(M.shape(merged)), "merge_correct": ok,
            "errors_detected": errs}


def print_summary(r):
    print("\n" + "=" * 66)
    print("  Lesson 23 · Placeholder 与 Embedding Merge —— 运行成功 ✓")
    print("=" * 66)
    print(f"  媒体数 {r['num_media']}，visual token 数 {r['visual_token_counts']}")
    print(f"  合并后 embedding 形状：{tuple(r['merged_shape'])}")
    print(f"  ✓ 占位处 embedding 已被视觉 embedding 逐位替换：{r['merge_correct']}")
    print("  ✓ 对齐错误被检测：")
    for k, v in r["errors_detected"].items():
        print(f"      {k}: {v}")
    print("=" * 66)
    print("  下一步：python3 course.py check 23   或   Lesson 24 multimodal prefill。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_23_placeholder_merge",
                    "configs/lesson_23_quick.json", "outputs/lesson_23").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
