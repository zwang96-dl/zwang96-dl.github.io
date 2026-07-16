"""Lesson 14 实验：Block Table 与 Paged KV Cache。

用真实 tiny model 对同一 prompt 分别跑「连续 KV」与「分页 KV」，证明输出**逐值一致**；
展示 block table（logical→physical）与「逻辑连续、物理分散」的现象。
"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result, load_model
from mini_vllm.model import matrix as M
from mini_vllm.cache.kv_cache import KVCache
from mini_vllm.cache.block_allocator import BlockAllocator
from mini_vllm.cache.block_table import PagedKVCache
from mini_vllm.tokenizer import ByteTokenizer
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    model = load_model(); tok = ByteTokenizer()
    ids = tok.encode(cfg.get("prompt", "Hello paged world"), add_bos=True)
    bs = cfg.get("block_size", 4)

    contiguous = model.forward(ids, list(range(len(ids))), KVCache(model.cfg))
    alloc = BlockAllocator(num_blocks=64, block_size=bs)
    # 制造真正「交错」的空洞：先占用前 N 个物理块，再**隔一个释放一个**（1,3,5,7…），
    # 这样后续序列会拿到这些分散的空闲块 → block table 物理上不连续。
    prealloc = cfg.get("prealloc_blocks", 8)
    held = [alloc.allocate() for _ in range(prealloc)]        # 物理块 0..N-1
    holes = held[1::2]                                        # 1,3,5,7… 交错
    for h in holes:
        alloc.free(h)

    paged = PagedKVCache(model.cfg, alloc)                    # 序列拿到分散的空闲块
    logits = model.forward(ids, list(range(len(ids))), paged)
    diff = M.max_abs_diff(contiguous, logits)
    bt = list(paged.block_table)
    noncontig = any(bt[i + 1] - bt[i] != 1 for i in range(len(bt) - 1))

    with tracer.section("paged"):
        tracer.event("block_table", mapping=paged.logical_to_physical())
        tracer.event("physically_noncontiguous", value=noncontig, block_table=bt)
        tracer.event("alignment", max_abs_diff=diff)

    # 清理：释放序列自身与仍占用的偶数号块，保证无泄漏
    paged.free()
    for h in held[0::2]:
        alloc.free(h)
    alloc.check_no_leak()
    return {"prompt_tokens": len(ids), "block_size": bs,
            "block_table": [{"logical": i, "physical": p} for i, p in enumerate(bt)],
            "paged_vs_contiguous_max_abs_diff": diff,
            "physically_noncontiguous": noncontig}


def print_summary(r):
    print("\n" + "=" * 66)
    print("  Lesson 14 · Block Table 与 Paged KV Cache —— 运行成功 ✓")
    print("=" * 66)
    print(f"  prompt {r['prompt_tokens']} tokens，block_size={r['block_size']}")
    print("  block table（逻辑块 → 物理块）：")
    for m in r["block_table"]:
        print(f"    logical {m['logical']} → physical {m['physical']}")
    print(f"  物理块非连续（有空洞时）：{r['physically_noncontiguous']}")
    print(f"  ✓ paged 与连续 KV 输出最大绝对误差 = {r['paged_vs_contiguous_max_abs_diff']:.3e}")
    print("=" * 66)
    print("  下一步：python3 course.py check 14   或   Lesson 15 Chunked Prefill。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_14_paged_kv",
                    "configs/lesson_14_quick.json", "outputs/lesson_14").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
