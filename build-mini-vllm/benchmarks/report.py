"""从引擎运行结果计算性能报告（含测量方法说明）。

指标（Lesson 24 的多模态报告也复用其中一部分）：
    - requests / prompt tokens / output tokens
    - iterations / wall_time
    - TTFT：这里用「首 token 所在迭代号 - 到达迭代号」作为**迭代级**代理（教学清晰、可复现）；
      也给出 wall-clock 版本供参考。
    - throughput：output tokens / wall_time
    - KV / block 利用率：peak_blocks_used / num_blocks
    - prefix hit rate（若开启前缀缓存）
"""

from __future__ import annotations


def build_report(engine, result, prefix_stats: dict | None = None,
                 baseline_processed: int | None = None) -> dict:
    reqs = result.requests
    total_prompt = sum(r.num_prompt_tokens for r in reqs)
    total_output = sum(len(r.output_token_ids) for r in reqs)
    ttfts = [(r.first_token_iter - r.arrival) for r in reqs if r.first_token_iter is not None]
    comps = [(r.finish_iter - r.arrival) for r in reqs if r.finish_iter is not None]
    num_blocks = engine.allocator.num_blocks
    report = {
        "measurement_notes": {
            "TTFT_iters": "首 token 所在迭代号减去到达迭代号（迭代级代理，确定可复现）",
            "throughput": "总输出 token 数 / 墙钟时间（秒）",
            "kv_utilization": "峰值占用物理块 / 物理块总数",
        },
        "scheduler_policy": engine.scheduler.cfg.policy,
        "requests": len(reqs),
        "prompt_tokens": total_prompt,
        "output_tokens": total_output,
        "iterations": result.num_iterations,
        "wall_time_s": round(result.wall_time_s, 4),
        "avg_ttft_iters": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
        "max_ttft_iters": max(ttfts) if ttfts else None,
        "avg_completion_iters": round(sum(comps) / len(comps), 2) if comps else None,
        "token_throughput_per_s": round(total_output / result.wall_time_s, 1) if result.wall_time_s else None,
        "peak_blocks_used": result.peak_blocks_used,
        "num_blocks": num_blocks,
        "kv_utilization": round(result.peak_blocks_used / num_blocks, 3) if num_blocks else 0,
    }
    if prefix_stats is not None:
        report["prefix_cache"] = prefix_stats
    if baseline_processed is not None:
        report["baseline_naive_processed_tokens"] = baseline_processed
    return report


def format_report(title: str, report: dict) -> str:
    lines = [title, "=" * len(title)]
    order = ["scheduler_policy", "requests", "prompt_tokens", "output_tokens", "iterations",
             "wall_time_s", "avg_ttft_iters", "max_ttft_iters", "avg_completion_iters",
             "token_throughput_per_s", "peak_blocks_used", "num_blocks", "kv_utilization"]
    for k in order:
        if k in report and report[k] is not None:
            lines.append(f"  {k:<24} : {report[k]}")
    if "prefix_cache" in report:
        pc = report["prefix_cache"]
        lines.append(f"  prefix_cache             : hits={pc['hits']} rate={pc['hit_rate']:.2f} "
                     f"cached_blocks={pc['cached_blocks']} evictions={pc['evictions']}")
    if "baseline_naive_processed_tokens" in report:
        lines.append(f"  baseline(naive)_processed: {report['baseline_naive_processed_tokens']} tokens")
    lines.append("  （measurement_notes 见 result.json）")
    return "\n".join(lines)
