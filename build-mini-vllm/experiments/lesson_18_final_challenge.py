"""Lesson 18 实验：Text Final Incident Challenge。

一个综合场景：短聊天 + 长 prompt + 不同输出长度 + 共享前缀 + 有限 KV 块 + 动态到达。
引擎必须：正确、无 KV 泄漏、无重复执行（cached==reference）、无明显 starvation、
输出性能报告并与 naive 基线对比。本实验运行并自检这些性质。
"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.config import EngineConfig
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.sampling import Sampler, SamplingParams
from mini_vllm.engine.generate import generate_cached, generate_naive
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.scheduler.scheduler import SchedulerConfig
from mini_vllm.trace import Tracer
from benchmarks.report import build_report, format_report


def run_experiment(cfg, tracer):
    model = load_model(); tok = ByteTokenizer()
    ec = EngineConfig(block_size=cfg.get("block_size", 4), num_blocks=cfg.get("num_blocks", 48),
                      max_num_seqs=cfg.get("max_num_seqs", 3),
                      max_num_batched_tokens=cfg.get("max_num_batched_tokens", 16),
                      scheduler_policy=cfg.get("policy", "balanced"))
    eng = LLMEngine(model, tok, ec, SchedulerConfig(
        max_num_seqs=ec.max_num_seqs, max_num_batched_tokens=ec.max_num_batched_tokens,
        policy=ec.scheduler_policy, enable_chunked_prefill=True), enable_prefix_cache=True)
    for r in cfg["requests"]:
        eng.add_request(r["id"], r["prompt"], r["max_new_tokens"], arrival=r.get("arrival", 0),
                        stop_on_eos=False)
    res = eng.run(tracer=tracer)

    # 自检 1：正确性——每个请求输出与「单独 cached 生成」逐 token 一致（无重复执行/无串请求）
    correct = True
    for r in cfg["requests"]:
        ref = generate_cached(model, tok.encode(r["prompt"], add_bos=True), r["max_new_tokens"],
                              Sampler(SamplingParams()), stop_on_eos=False).generated
        got = next(x.output_token_ids for x in res.requests if x.request_id == r["id"])
        if got != ref:
            correct = False
    # 自检 2：无 starvation——所有请求都完成了
    all_finished = all(x.finish_iter is not None for x in res.requests)
    # 自检 3：TTFT 分布（迭代级）——最大值不应过分离谱（粗略反 starvation）
    ttfts = [x.first_token_iter - x.arrival for x in res.requests if x.first_token_iter is not None]

    baseline = sum(generate_naive(model, tok.encode(r["prompt"], add_bos=True),
                                  r["max_new_tokens"], Sampler(SamplingParams()),
                                  stop_on_eos=False).total_processed_tokens
                   for r in cfg["requests"])
    pstats = eng.prefix_cache.stats()
    eng.shutdown()   # 触发无泄漏自检（shutdown 内含 check_no_leak）
    no_leak = eng.allocator.num_used == 0

    report = build_report(eng, res, prefix_stats=pstats, baseline_processed=baseline)
    report["checks"] = {"correct_vs_reference": correct, "all_requests_finished": all_finished,
                        "no_kv_leak": no_leak, "max_ttft_iters": max(ttfts) if ttfts else None}
    return report


def print_summary(r):
    print("\n" + format_report("Lesson 18 · Text Final Incident · 性能报告", r))
    c = r["checks"]
    print("-" * 48)
    print("  综合自检：")
    print(f"    ✓ 正确性（cached==reference，无重复执行/无串请求）：{c['correct_vs_reference']}")
    print(f"    ✓ 所有请求完成（无 starvation）：{c['all_requests_finished']}")
    print(f"    ✓ 无 KV 泄漏：{c['no_kv_leak']}")
    print(f"    · 最大首 token 延迟（迭代级）：{c['max_ttft_iters']}")
    print("=" * 48)
    print("  文本主线（Lesson 0–18）完成！下一步进入多模态（Phase 4+）。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_18_final_challenge",
                    "configs/lesson_18_quick.json", "outputs/lesson_18").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
