"""Lesson 28 实验：Multimodal Scheduler —— visual token 预算如何影响准入。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.config import VisionConfig
from mini_vllm.multimodal.mm_engine import MultiModalEngine
from mini_vllm.multimodal.budget import MultiModalBudget
from mini_vllm.multimodal import messages as msg
from mini_vllm.trace import Tracer


def _workload(eng):
    img = lambda s: {"synth": {"h": 16, "w": 16, "seed": s}}
    eng.add_request("A", [msg.user(msg.text("a"), msg.image(img(1)))], 4, arrival=0)
    eng.add_request("B", [msg.user(msg.text("b"), msg.image(img(2)))], 4, arrival=0)
    eng.add_request("C", [msg.user(msg.text("c"), msg.image(img(3)))], 4, arrival=0)


def _run(model, budget):
    eng = MultiModalEngine(model, VisionConfig(), budget=budget, enable_caches=True)
    _workload(eng)
    res = eng.run()
    # 第一迭代准入了几个（受 visual 预算限制）
    first_admitted = res.snapshots[0].admitted if res.snapshots else []
    return {"iterations": res.iterations, "first_iter_admitted": first_admitted,
            "outputs": {r.request_id: r.output_token_ids for r in res.requests}}


def run_experiment(cfg, tracer):
    model = load_model()
    tight = _run(model, MultiModalBudget(visual_token_budget=4, encoder_budget=1, max_num_seqs=3))
    loose = _run(model, MultiModalBudget(visual_token_budget=64, encoder_budget=4, max_num_seqs=3))
    tracer.event("tight", first_admitted=tight["first_iter_admitted"])
    tracer.event("loose", first_admitted=loose["first_iter_admitted"])
    return {"tight_budget": {k: v for k, v in tight.items() if k != "outputs"},
            "loose_budget": {k: v for k, v in loose.items() if k != "outputs"},
            "outputs_identical": tight["outputs"] == loose["outputs"]}


def print_summary(r):
    print("\n" + "=" * 66)
    print("  Lesson 28 · Multimodal Scheduler —— 运行成功 ✓")
    print("=" * 66)
    print(f"  紧预算(visual=4, encoder=1)：第 1 迭代准入 {r['tight_budget']['first_iter_admitted']}，"
          f"共 {r['tight_budget']['iterations']} 迭代")
    print(f"  松预算(visual=64,encoder=4)：第 1 迭代准入 {r['loose_budget']['first_iter_admitted']}，"
          f"共 {r['loose_budget']['iterations']} 迭代")
    print(f"  ✓ 两种预算输出完全一致：{r['outputs_identical']}（预算只改「何时算」）")
    print("-" * 66)
    print("  证据：visual token / encoder 预算限制一次能准入多少多模态请求（视觉工作量入调度）。")
    print("=" * 66)
    print("  下一步：python3 course.py check 28   或   Lesson 29 完整多模态引擎。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_28_mm_scheduler",
                    "configs/lesson_28_quick.json", "outputs/lesson_28").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
