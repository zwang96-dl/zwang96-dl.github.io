"""Lesson 9 实验：Static Batching 的 padding 浪费。

静态批处理把一批请求 padding 到同一长度、一起跑到最长者结束：短请求早就该完成，
却要陪着长请求空转（padding/finished slot 上的计算全是浪费）。本实验量化这份浪费，
并与「按实际长度计费」的连续批处理对比。
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from mini_vllm.config import RUN_MODES  # noqa: E402
from mini_vllm.tokenizer import ByteTokenizer  # noqa: E402
from mini_vllm.trace import Tracer  # noqa: E402


def run_experiment(cfg, tracer: Tracer):
    reqs = cfg["requests"]  # [{prompt_len, output_len}]
    lens = [r["prompt_len"] + r["output_len"] for r in reqs]
    max_len = max(lens)
    bs = len(reqs)
    static_cost = bs * max_len          # 所有 slot 补齐到 max，跑满 max 步
    actual_cost = sum(lens)             # 真实需要处理的总量
    waste = static_cost - actual_cost
    with tracer.section("static vs continuous"):
        for i, l in enumerate(lens):
            tracer.detail(f"req[{i}]", real_len=l, padded_to=max_len, wasted=max_len - l)
        tracer.event("totals", static_cost=static_cost, actual=actual_cost, waste=waste)
    # padding mask 小示例
    tok = ByteTokenizer()
    batch = [tok.encode("hi", add_bos=False), tok.encode("hello!", add_bos=False)]
    ids, mask = tok.pad(batch)
    return {"lens": lens, "max_len": max_len, "batch_size": bs,
            "static_cost": static_cost, "continuous_cost": actual_cost,
            "waste": waste, "waste_pct": round(100 * waste / static_cost, 1),
            "pad_example": {"ids": ids, "mask": mask}}


def print_summary(r):
    print("\n" + "=" * 64)
    print("  Lesson 9 · Static Batching 的 padding 浪费 —— 运行成功 ✓")
    print("=" * 64)
    print(f"  批大小={r['batch_size']}  各请求真实长度={r['lens']}  padding 到={r['max_len']}")
    print(f"  静态批处理成本 = {r['batch_size']}×{r['max_len']} = {r['static_cost']}")
    print(f"  连续批处理成本 = Σ实际长度 = {r['continuous_cost']}")
    print(f"  浪费 = {r['waste']}（{r['waste_pct']}%）—— 花在 padding / 已完成 slot 上")
    print(f"  padding mask 示例：mask={r['pad_example']['mask']}（1=真实, 0=PAD）")
    print("-" * 64)
    print("  证据：请求长度差异越大，静态批处理浪费越多 → 引出 continuous batching（Lesson 10）。")
    print("=" * 64)
    print("  下一步：python3 course.py check 9   或   Lesson 10 看 continuous batching。")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments.lesson_09_static_batching")
    p.add_argument("--config", default="configs/lesson_09_quick.json")
    p.add_argument("--mode", default="quick", choices=list(RUN_MODES))
    p.add_argument("--verbose", action="store_true"); p.add_argument("--trace", action="store_true")
    p.add_argument("--out", default="outputs/lesson_09")
    a = p.parse_args(argv)
    cp = (_ROOT / a.config).resolve()
    if not cp.exists():
        print(f"[错误] 找不到配置：{cp}", file=sys.stderr); return 2
    cfg = json.loads(cp.read_text("utf-8"))
    r = run_experiment(cfg, Tracer.from_flags(a.verbose, a.trace))
    od = (_ROOT / a.out).resolve(); od.mkdir(parents=True, exist_ok=True)
    (od / "result.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), "utf-8")
    print_summary(r); print(f"  结果已写入：{(od/'result.json').relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
