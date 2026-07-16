"""Lesson 12 实验：连续 KV 分配为何浪费。

朴素做法给每条序列**按 max_seq_len 预留**一段连续 KV：实际用不满 → internal
fragmentation；不同序列的预留段之间形成空洞 → external fragmentation。分页按需
分配、可复用，几乎消除这两种浪费。本实验量化对比。
"""
from __future__ import annotations
import math
from experiments._common import make_parser, read_config, write_result
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    max_seq = cfg.get("max_seq_len", 64)
    block_size = cfg.get("block_size", 8)
    seqs = cfg["seq_lens"]     # 各请求实际用到的长度
    # 连续预留：每条按 max_seq 预留
    reserved = max_seq * len(seqs)
    used = sum(seqs)
    internal_frag = reserved - used
    # 分页：按 block 向上取整分配
    paged = sum(math.ceil(s / block_size) * block_size for s in seqs)
    paged_waste = paged - used
    with tracer.section("fragmentation"):
        for i, s in enumerate(seqs):
            tracer.detail(f"seq[{i}]", used=s, reserved_contig=max_seq,
                          paged=math.ceil(s / block_size) * block_size)
    return {"max_seq_len": max_seq, "block_size": block_size, "seq_lens": seqs,
            "contiguous_reserved": reserved, "used": used,
            "contiguous_waste": internal_frag,
            "contiguous_waste_pct": round(100 * internal_frag / reserved, 1),
            "paged_allocated": paged, "paged_waste": paged_waste,
            "paged_waste_pct": round(100 * paged_waste / paged, 1)}


def print_summary(r):
    print("\n" + "=" * 66)
    print("  Lesson 12 · 连续 KV 分配为何浪费 —— 运行成功 ✓")
    print("=" * 66)
    print(f"  实际用量 Σ = {r['used']}  （各序列长度 {r['seq_lens']}）")
    print(f"  连续预留（按 max_seq={r['max_seq_len']}）= {r['contiguous_reserved']}  "
          f"→ 浪费 {r['contiguous_waste']}（{r['contiguous_waste_pct']}%）")
    print(f"  分页分配（block={r['block_size']} 向上取整）= {r['paged_allocated']}  "
          f"→ 浪费仅 {r['paged_waste']}（{r['paged_waste_pct']}%，只剩块内零头）")
    print("-" * 66)
    print("  证据：连续预留的浪费随 max_seq 与并发数暴涨；分页把浪费压到「不足一块」的零头。")
    print("=" * 66)
    print("  下一步：python3 course.py check 12   或   Lesson 13 实现 Block Allocator。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_12_kv_waste",
                    "configs/lesson_12_quick.json", "outputs/lesson_12").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
