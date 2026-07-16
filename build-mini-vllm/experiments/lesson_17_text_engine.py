"""Lesson 17 实验：完整 Text Mini-vLLM 端到端。

把 tokenizer + 状态机 + scheduler + block allocator + paged KV + model runner + sampler
组装成引擎，跑一个混合 workload，输出性能报告，并与 naive（无缓存、重算整段）基线对比。
"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.config import EngineConfig
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.sampling import Sampler, SamplingParams
from mini_vllm.engine.generate import generate_naive
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.scheduler.scheduler import SchedulerConfig
from mini_vllm.trace import Tracer
from benchmarks.report import build_report, format_report


def run_experiment(cfg, tracer):
    model = load_model(); tok = ByteTokenizer()
    ec = EngineConfig(block_size=cfg.get("block_size", 8), num_blocks=cfg.get("num_blocks", 128),
                      max_num_seqs=cfg.get("max_num_seqs", 3),
                      max_num_batched_tokens=cfg.get("max_num_batched_tokens", 32),
                      scheduler_policy=cfg.get("policy", "balanced"))
    eng = LLMEngine(model, tok, ec, SchedulerConfig(
        max_num_seqs=ec.max_num_seqs, max_num_batched_tokens=ec.max_num_batched_tokens,
        policy=ec.scheduler_policy, enable_chunked_prefill=True),
        enable_prefix_cache=cfg.get("prefix_cache", True))
    for r in cfg["requests"]:
        eng.add_request(r["id"], r["prompt"], r["max_new_tokens"], arrival=r.get("arrival", 0))
    res = eng.run(tracer=tracer)

    # naive 基线：逐请求无缓存重算，累计处理 token
    baseline = 0
    for r in cfg["requests"]:
        g = generate_naive(model, tok.encode(r["prompt"], add_bos=True),
                           r["max_new_tokens"], Sampler(SamplingParams()), stop_on_eos=True)
        baseline += g.total_processed_tokens
    pstats = eng.prefix_cache.stats() if eng.prefix_cache else None
    eng.shutdown()
    report = build_report(eng, res, prefix_stats=pstats, baseline_processed=baseline)
    report["outputs"] = {r.request_id: tok.decode(r.output_token_ids) for r in res.requests}
    return report


def print_summary(r):
    print("\n" + format_report("Lesson 17 · 完整 Text Mini-vLLM · 性能报告", r))
    print("  下一步：python3 course.py check 17   或   Lesson 18 综合故障挑战。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_17_text_engine",
                    "configs/lesson_17_quick.json", "outputs/lesson_17").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
