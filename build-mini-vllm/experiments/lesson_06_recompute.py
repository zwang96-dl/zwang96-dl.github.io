"""Lesson 6 实验：重复计算侦探。

对同一个 prompt 分别用 naive（重算整段）与 cached（KV Cache）生成，逐步对比每步
「处理了多少 token」以及耗时——亲眼看到 naive 的 O(n²) 浪费。

用法::

    python3 -m experiments.lesson_06_recompute --config configs/lesson_06_quick.json
    python3 -m experiments.lesson_06_recompute --trace

输出：outputs/lesson_06/result.json；不修改源代码。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mini_vllm.config import RUN_MODES  # noqa: E402
from mini_vllm.model.transformer import load_checkpoint  # noqa: E402
from mini_vllm.tokenizer import ByteTokenizer  # noqa: E402
from mini_vllm.sampling import Sampler, SamplingParams  # noqa: E402
from mini_vllm.engine.generate import generate_naive, generate_cached, processed_token_curves  # noqa: E402
from mini_vllm.trace import Tracer  # noqa: E402


def run_experiment(config_path: Path, mode: str, tracer: Tracer):
    cfg = json.loads(config_path.read_text("utf-8"))
    model = load_checkpoint(_ROOT / cfg["checkpoint"])
    tok = ByteTokenizer()
    prompt = cfg.get("prompt", "Hello, world")
    n = int(cfg.get("max_new_tokens", 12))
    prompt_ids = tok.encode(prompt, add_bos=True)

    with tracer.section("naive vs cached"):
        gn = generate_naive(model, prompt_ids, n, Sampler(SamplingParams()), stop_on_eos=False)
        gc = generate_cached(model, prompt_ids, n, Sampler(SamplingParams()), stop_on_eos=False)
        for s in gn.steps:
            tracer.detail("naive step", step=s.step, input_len=s.input_len,
                          processed=s.processed_tokens)
        tracer.event("totals", naive=gn.total_processed_tokens, cached=gc.total_processed_tokens)

    curves = processed_token_curves(len(prompt_ids), n)
    return {
        "prompt": prompt, "prompt_len": len(prompt_ids), "max_new_tokens": n,
        "identical_output": gn.generated == gc.generated,
        "naive_total": gn.total_processed_tokens,
        "cached_total": gc.total_processed_tokens,
        "naive_steps": [{"step": s.step, "input_len": s.input_len,
                         "processed": s.processed_tokens} for s in gn.steps],
        "cached_steps": [{"step": s.step, "phase": s.phase, "input_len": s.input_len,
                          "processed": s.processed_tokens} for s in gc.steps],
        "curves": curves,
    }


def print_summary(r) -> None:
    print()
    print("=" * 70)
    print("  Lesson 6 · 重复计算侦探 —— 运行成功 ✓")
    print("=" * 70)
    print(f"  prompt_len={r['prompt_len']}  max_new_tokens={r['max_new_tokens']}")
    print(f"  {'step':>4}  {'naive 处理':>10}  {'cached 处理':>11}")
    for a, b in zip(r["naive_steps"], r["cached_steps"]):
        print(f"  {a['step']:>4}  {a['processed']:>10}  {b['processed']:>11}   "
              + ("█" * a['processed']))
    print("-" * 70)
    ratio = r["naive_total"] / max(1, r["cached_total"])
    print(f"  累计处理 token：naive={r['naive_total']}  vs  cached={r['cached_total']}  "
          f"（naive 多做 {ratio:.1f}×）")
    print(f"  两种方法生成结果一致：{r['identical_output']}")
    print("-" * 70)
    print("  证据：naive 每步都重算整段前缀（条形逐步变长），cached 每步只处理 1 个 token。")
    print("        这就是下一课 KV Cache 要消除的浪费。")
    print("=" * 70)
    print("  下一步：python3 course.py check 6   或   Lesson 7 亲手实现 KV Cache。")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments.lesson_06_recompute")
    p.add_argument("--config", default="configs/lesson_06_quick.json")
    p.add_argument("--mode", default="quick", choices=list(RUN_MODES))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", default="outputs/lesson_06")
    args = p.parse_args(argv)
    config_path = (_ROOT / args.config).resolve()
    if not config_path.exists():
        print(f"[错误] 找不到配置文件：{config_path}", file=sys.stderr)
        return 2
    tracer = Tracer.from_flags(verbose=args.verbose, trace=args.trace)
    r = run_experiment(config_path, args.mode, tracer)
    out_dir = (_ROOT / args.out).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), "utf-8")
    print_summary(r)
    print(f"  结果已写入：{(out_dir / 'result.json').relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
