"""Lesson 13 实验：Block Allocator 的行为——allocate/free/incref、OOM、double free、leak。"""
from __future__ import annotations
from experiments._common import make_parser, read_config, write_result
from mini_vllm.cache.block_allocator import BlockAllocator
from mini_vllm.trace import Tracer


def run_experiment(cfg, tracer):
    nb = cfg.get("num_blocks", 4)
    alloc = BlockAllocator(num_blocks=nb, block_size=cfg.get("block_size", 8))
    log = []
    with tracer.section("allocator"):
        a = alloc.allocate(); log.append(("allocate", a, alloc.num_free))
        b = alloc.allocate(); log.append(("allocate", b, alloc.num_free))
        alloc.incref(a); log.append(("incref", a, alloc.ref[a]))     # 共享 a（前缀缓存的雏形）
        alloc.free(a); log.append(("free(refcount--)", a, alloc.ref[a]))  # 仍有引用，不回收
        alloc.free(a); log.append(("free(->0, 回收)", a, alloc.num_free))
        tracer.event("state", free=alloc.num_free, used=alloc.num_used)

    # OOM：耗尽后再分配
    oom = False
    small = BlockAllocator(num_blocks=1)
    small.allocate()
    try:
        small.allocate()
    except MemoryError:
        oom = True

    # double free
    dbl = False
    d = BlockAllocator(num_blocks=2); x = d.allocate(); d.free(x)
    try:
        d.free(x)
    except RuntimeError:
        dbl = True

    # leak detection
    leak = False
    lk = BlockAllocator(num_blocks=2); lk.allocate()
    try:
        lk.check_no_leak()
    except AssertionError:
        leak = True

    return {"num_blocks": nb, "op_log": [{"op": o, "block": bl, "info": inf} for o, bl, inf in log],
            "final_free": alloc.num_free, "oom_detected": oom,
            "double_free_detected": dbl, "leak_detected": leak}


def print_summary(r):
    print("\n" + "=" * 64)
    print("  Lesson 13 · Block Allocator —— 运行成功 ✓")
    print("=" * 64)
    for e in r["op_log"]:
        print(f"    {e['op']:<20} block={e['block']}  info={e['info']}")
    print(f"  最终空闲块 = {r['final_free']} / {r['num_blocks']}")
    print("-" * 64)
    print(f"    ✓ OOM 被检测（耗尽后分配抛 MemoryError）：{r['oom_detected']}")
    print(f"    ✓ double free 被检测：{r['double_free_detected']}")
    print(f"    ✓ leak 被检测（未释放即自检报错）：{r['leak_detected']}")
    print("=" * 64)
    print("  下一步：python3 course.py check 13   或   Lesson 14 Block Table 与 Paged KV。")


def main(argv=None) -> int:
    a = make_parser("experiments.lesson_13_block_allocator",
                    "configs/lesson_13_quick.json", "outputs/lesson_13").parse_args(argv)
    r = run_experiment(read_config(a.config), Tracer.from_flags(a.verbose, a.trace))
    rel = write_result(a.out, r); print_summary(r); print(f"  结果已写入：{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
