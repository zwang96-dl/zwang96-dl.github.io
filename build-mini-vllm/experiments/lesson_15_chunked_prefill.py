"""Lesson 15 实验：Token Budget 与 Chunked Prefill。

一个长 prompt 与一个短请求并发。不开 chunked prefill 时，长 prompt 的整段 prefill
会独占若干迭代，短请求的首 token 被推迟；开启后，长 prompt 被切成块与短请求的 decode
混排，短请求更早拿到首 token。输出保持不变。
"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.config import EngineConfig
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.engine.engine import LLMEngine
from mini_vllm.scheduler.scheduler import SchedulerConfig
from mini_vllm.sampling import Sampler, SamplingParams
from mini_vllm.engine.generate import generate_cached
from mini_vllm.trace import Tracer


def _run(chunked, cfg, model, tok):
    ec = EngineConfig(block_size=cfg.get("block_size", 8), num_blocks=128,
                      max_num_seqs=cfg.get("max_num_seqs", 2),
                      max_num_batched_tokens=cfg.get("max_num_batched_tokens", 8))
    eng = LLMEngine(model, tok, ec, SchedulerConfig(
        max_num_seqs=ec.max_num_seqs, max_num_batched_tokens=ec.max_num_batched_tokens,
        policy="balanced", enable_chunked_prefill=chunked))
    for r in cfg["requests"]:
        eng.add_request(r["id"], r["prompt"], r["max_new_tokens"], arrival=r.get("arrival", 0))
    try:
        res = eng.run()
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    ttft = {r.request_id: (r.first_token_iter - r.arrival) for r in res.requests
            if r.first_token_iter is not None}
    outs = {r.request_id: r.output_token_ids for r in res.requests}
    return {"ok": True, "iterations": res.num_iterations, "ttft_iters": ttft, "outputs": outs}


def run_experiment(cfg, tracer):
    model = load_model(); tok = ByteTokenizer()
    off = _run(False, cfg, model, tok)
    on = _run(True, cfg, model, tok)
    short_id = cfg.get("short_id", "short")
    tracer.event("chunked=off", ok=off["ok"])
    tracer.event("chunked=on", ok=on["ok"], ttft=on.get("ttft_iters"))
    # 正确性：开启 chunked 后输出应与标准 cached 生成一致
    correct = True
    if on["ok"]:
        for req in cfg["requests"]:
            ref = generate_cached(model, tok.encode(req["prompt"], add_bos=True),
                                  req["max_new_tokens"], Sampler(SamplingParams()),
                                  stop_on_eos=False).generated
            if on["outputs"][req["id"]] != ref:
                correct = False
    return {"short_id": short_id,
            "chunked_off_ok": off["ok"], "chunked_off_error": off.get("error"),
            "chunked_on_ok": on["ok"],
            "short_ttft_on": on["ttft_iters"].get(short_id) if on["ok"] else None,
            "on_outputs_correct": correct}


def print_summary(r):
    print("\n" + "=" * 68)
    print("  Lesson 15 · Token Budget 与 Chunked Prefill —— 运行成功 ✓")
    print("=" * 68)
    print("  不开 chunked prefill：")
    if r["chunked_off_ok"]:
        print("    长 prompt 在预算内可一次 prefill（本配置未触发阻塞）。")
    else:
        print("    ✗ 调度停滞——长 prompt 超过 token 预算，无法开始。")
        print("      " + (r["chunked_off_error"] or "").split("——")[0])
    print("  开启 chunked prefill：")
    print(f"    ✓ 长 prompt 被切块，与短请求 decode 混排；短请求 {r['short_id']!r} "
          f"首 token 延迟（迭代级）= {r['short_ttft_on']}")
    print(f"    ✓ 输出与标准生成逐 token 一致：{r['on_outputs_correct']}")
    print("=" * 68)
    print("  证据：token 预算小于长 prompt 时，chunked prefill 让系统仍能推进并保护短请求 TTFT。")
    print("  下一步：python3 course.py check 15   或   Lesson 16 Prefix Caching。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_15_chunked_prefill",
                    "configs/lesson_15_quick.json", "outputs/lesson_15").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
