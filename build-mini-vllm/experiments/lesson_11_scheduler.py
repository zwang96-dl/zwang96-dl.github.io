"""Lesson 11 实验：比较调度策略（fifo / decode-first / sjf / balanced）。

同一个 workload，四种策略跑出的迭代数、平均首 token 延迟（迭代级）各不相同。
输出结果一致（调度只影响「什么时候算」，不影响「算出什么」）。
"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.config import EngineConfig
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.scheduler.scheduler import SchedulerConfig
from mini_vllm.trace import Tracer


def _run(policy, cfg, model, tok):
    ec = EngineConfig(block_size=cfg.get("block_size", 8), num_blocks=cfg.get("num_blocks", 128),
                      max_num_seqs=cfg.get("max_num_seqs", 2),
                      max_num_batched_tokens=cfg.get("max_num_batched_tokens", 16))
    eng = LLMEngine(model, tok, ec, SchedulerConfig(
        max_num_seqs=ec.max_num_seqs, max_num_batched_tokens=ec.max_num_batched_tokens,
        policy=policy, enable_chunked_prefill=True))
    for r in cfg["requests"]:
        eng.add_request(r["id"], r["prompt"], r["max_new_tokens"], arrival=r.get("arrival", 0))
    res = eng.run()
    ttfts = [r.first_token_iter - r.arrival for r in res.requests if r.first_token_iter is not None]
    outs = {r.request_id: r.output_token_ids for r in res.requests}
    return {"iterations": res.num_iterations,
            "avg_ttft_iters": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "max_ttft_iters": max(ttfts) if ttfts else None, "outputs": outs}


def run_experiment(cfg, tracer):
    model = load_model(); tok = ByteTokenizer()
    policies = ["fifo", "decode-first", "sjf", "balanced"]
    results = {p: _run(p, cfg, model, tok) for p in policies}
    # 一致性：各策略输出应完全相同
    base = results["fifo"]["outputs"]
    consistent = all(results[p]["outputs"] == base for p in policies)
    for p in policies:
        tracer.event(p, iterations=results[p]["iterations"],
                     avg_ttft=results[p]["avg_ttft_iters"])
    return {"policies": {p: {k: v for k, v in results[p].items() if k != "outputs"}
                         for p in policies},
            "outputs_consistent_across_policies": consistent}


def print_summary(r):
    print("\n" + "=" * 68)
    print("  Lesson 11 · Scheduler 策略比较 —— 运行成功 ✓")
    print("=" * 68)
    print(f"  {'policy':<14}{'iterations':>12}{'avg_TTFT(it)':>14}{'max_TTFT(it)':>14}")
    for p, m in r["policies"].items():
        print(f"  {p:<14}{m['iterations']:>12}{str(m['avg_ttft_iters']):>14}{str(m['max_ttft_iters']):>14}")
    print("-" * 68)
    print(f"  各策略输出完全一致：{r['outputs_consistent_across_policies']}"
          "（调度只改变「何时算」，不改变「算什么」）")
    print("=" * 68)
    print("  下一步：python3 course.py check 11   或   Lesson 12 看连续 KV 分配为何浪费。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_11_scheduler",
                    "configs/lesson_11_quick.json", "outputs/lesson_11").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
