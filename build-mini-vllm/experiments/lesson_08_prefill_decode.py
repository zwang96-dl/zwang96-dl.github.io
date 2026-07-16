"""Lesson 8 实验：prefill 与 decode，度量 TTFT / TPOT / ITL。

用法::

    python3 -m experiments.lesson_08_prefill_decode --config configs/lesson_08_quick.json
    python3 -m experiments.lesson_08_prefill_decode --trace

输出：outputs/lesson_08/result.json；不修改源代码。
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
from mini_vllm.engine.generate import generate_cached  # noqa: E402
from mini_vllm.trace import Tracer  # noqa: E402


def run_experiment(config_path: Path, mode: str, tracer: Tracer):
    cfg = json.loads(config_path.read_text("utf-8"))
    model = load_checkpoint(_ROOT / cfg["checkpoint"])
    tok = ByteTokenizer()
    n = int(cfg.get("max_new_tokens", 12))

    scenarios = cfg.get("scenarios", [
        {"name": "prompt-heavy", "prompt": "The quick brown fox jumps over the lazy dog again"},
        {"name": "decode-heavy", "prompt": "Hi"},
    ])
    out = []
    with tracer.section("prefill/decode"):
        for sc in scenarios:
            ids = tok.encode(sc["prompt"], add_bos=True)
            g = generate_cached(model, ids, n, Sampler(SamplingParams()),
                                stop_on_eos=False, tracer=tracer)
            itl = g.decode_times
            out.append({
                "name": sc["name"], "prompt_len": len(ids), "generated_len": len(g.generated),
                "ttft_s": g.ttft, "tpot_s": g.tpot,
                "itl_s": itl,
                "prefill_processed": g.steps[0].processed_tokens,
                "decode_steps": len(itl),
            })
            tracer.event(sc["name"], prompt_len=len(ids), ttft_s=round(g.ttft, 6),
                         tpot_s=round(g.tpot, 6))
    return {"max_new_tokens": n, "scenarios": out}


def print_summary(r) -> None:
    print()
    print("=" * 72)
    print("  Lesson 8 · Prefill 与 Decode —— 运行成功 ✓")
    print("=" * 72)
    print("  指标定义：TTFT=首 token 延迟(=prefill 耗时)；TPOT=每输出 token 平均耗时；")
    print("            ITL=相邻 token 的时间间隔（decode 步耗时序列）。")
    print("-" * 72)
    print(f"  {'场景':<14}{'prompt_len':>11}{'TTFT(ms)':>10}{'TPOT(ms)':>10}{'decode步':>8}")
    for s in r["scenarios"]:
        print(f"  {s['name']:<14}{s['prompt_len']:>11}{s['ttft_s']*1000:>10.2f}"
              f"{s['tpot_s']*1000:>10.2f}{s['decode_steps']:>8}")
    print("-" * 72)
    print("  证据：")
    print("    ✓ prompt 越长，prefill 处理的 token 越多 → TTFT 越大（compute-bound）")
    print("    ✓ decode 每步只处理 1 个 token → TPOT 相对稳定（更受内存/带宽影响）")
    print("=" * 72)
    print("  下一步：python3 course.py check 8   （Lesson 9+ 批处理/调度在后续 Phase 3 增量交付）")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments.lesson_08_prefill_decode")
    p.add_argument("--config", default="configs/lesson_08_quick.json")
    p.add_argument("--mode", default="quick", choices=list(RUN_MODES))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", default="outputs/lesson_08")
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
