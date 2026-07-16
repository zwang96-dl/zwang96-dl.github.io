"""Lesson 16 实验：Prefix Caching。

多个请求共享同一段长前缀（如相同的 system prompt）。第一个请求跑完后把前缀块登记进
缓存；后续请求命中前缀、跳过这段 prefill。输出**不变**，但省下大量 prefill 计算与块。
"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.config import EngineConfig
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.sampling import Sampler, SamplingParams
from mini_vllm.engine.generate import generate_cached
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    model = load_model(); tok = ByteTokenizer()
    shared = cfg.get("shared_prefix", "System: you are a helpful assistant. ")
    tails = cfg.get("tails", ["Question one?", "Question two?", "Question three?"])
    n = cfg.get("max_new_tokens", 6)
    ec = EngineConfig(block_size=cfg.get("block_size", 4), num_blocks=256,
                      max_num_seqs=1, max_num_batched_tokens=128)
    eng = LLMEngine(model, tok, ec, enable_prefix_cache=True)
    outs = {}
    for i, t in enumerate(tails):
        rid = f"q{i}"
        eng.add_request(rid, shared + t, n, arrival=0)
        res = eng.run()               # 逐个跑，让前缀在请求间复用
        for r in res.requests:
            if r.request_id == rid:
                outs[rid] = r.output_token_ids
        tracer.event(f"after {rid}", **eng.prefix_cache.stats())
    stats = eng.prefix_cache.stats()
    eng.shutdown()

    # 正确性：与不开前缀缓存的标准生成逐 token 一致
    ref_ok = True
    for i, t in enumerate(tails):
        ref = generate_cached(model, tok.encode(shared + t, add_bos=True), n,
                              Sampler(SamplingParams()), stop_on_eos=False).generated
        if outs[f"q{i}"] != ref:
            ref_ok = False
    return {"shared_prefix": shared, "num_requests": len(tails),
            "prefix_stats": stats, "outputs_match_reference": ref_ok}


def print_summary(r):
    print("\n" + "=" * 66)
    print("  Lesson 16 · Prefix Caching —— 运行成功 ✓")
    print("=" * 66)
    print(f"  共享前缀：{r['shared_prefix']!r}")
    s = r["prefix_stats"]
    print(f"  命中块 hits={s['hits']}  未命中 misses={s['misses']}  "
          f"命中率={s['hit_rate']:.2f}  逐出={s['evictions']}")
    print(f"  ✓ 命中不改变输出（与标准生成逐 token 一致）：{r['outputs_match_reference']}")
    print("-" * 66)
    print("  证据：后续请求复用前缀块，省下重复 prefill；输出保持正确。")
    print("=" * 66)
    print("  下一步：python3 course.py check 16   或   Lesson 17 组装完整引擎。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_16_prefix_cache",
                    "configs/lesson_16_quick.json", "outputs/lesson_16").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
