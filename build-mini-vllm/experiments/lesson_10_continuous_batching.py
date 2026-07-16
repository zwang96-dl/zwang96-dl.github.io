"""Lesson 10 实验：Request 状态机与 Continuous Batching。

请求错峰到达，引擎在**每次迭代**动态地准入/推进/退出请求——新请求可以在别人正在
decode 时加入同一个运行 batch，完成的请求立即让出资源。观察状态转移与逐迭代快照。
"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model, ROOT
from mini_vllm.config import EngineConfig
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.scheduler.scheduler import SchedulerConfig
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    model = load_model(); tok = ByteTokenizer()
    ec = EngineConfig(block_size=cfg.get("block_size", 8), num_blocks=cfg.get("num_blocks", 64),
                      max_num_seqs=cfg.get("max_num_seqs", 3),
                      max_num_batched_tokens=cfg.get("max_num_batched_tokens", 64))
    eng = LLMEngine(model, tok, ec, SchedulerConfig(
        max_num_seqs=ec.max_num_seqs, max_num_batched_tokens=ec.max_num_batched_tokens,
        policy="fifo", enable_chunked_prefill=True))
    for r in cfg["requests"]:
        eng.add_request(r["id"], r["prompt"], r["max_new_tokens"], arrival=r.get("arrival", 0))
    res = eng.run(tracer=tracer)
    return {
        "snapshots": [{"iter": s.iteration, "scheduled": s.scheduled,
                       "running": s.running, "waiting": s.waiting,
                       "prefill": s.num_prefill, "decode": s.num_decode} for s in res.snapshots],
        "requests": [{"id": r.request_id, "arrival": r.arrival,
                      "first_token_iter": r.first_token_iter, "finish_iter": r.finish_iter,
                      "generated": len(r.output_token_ids)} for r in res.requests],
        "iterations": res.num_iterations,
    }


def print_summary(r):
    print("\n" + "=" * 70)
    print("  Lesson 10 · Request 状态机与 Continuous Batching —— 运行成功 ✓")
    print("=" * 70)
    print(f"  {'iter':>4} {'scheduled':<22}{'running':>8}{'waiting':>8}")
    for s in r["snapshots"]:
        print(f"  {s['iter']:>4} {str(s['scheduled']):<22}{s['running']:>8}{s['waiting']:>8}")
    print("-" * 70)
    for q in r["requests"]:
        print(f"    {q['id']}: 到达@{q['arrival']}  首token@{q['first_token_iter']}  "
              f"完成@{q['finish_iter']}  生成 {q['generated']}")
    print("-" * 70)
    print("  证据：请求在不同迭代加入/退出同一运行 batch（continuous / iteration-level scheduling）。")
    print("=" * 70)
    print("  下一步：python3 course.py check 10   或   Lesson 11 比较调度策略。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_10_continuous_batching",
                    "configs/lesson_10_quick.json", "outputs/lesson_10").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
