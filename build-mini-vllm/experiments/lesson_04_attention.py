"""Lesson 4 实验：单头 causal attention 手算复现。

用法::

    python3 -m experiments.lesson_04_attention --config configs/lesson_04_quick.json
    python3 -m experiments.lesson_04_attention --trace

输入：configs/lesson_04_quick.json（引用 assets/workloads/lesson_04.json）
输出：outputs/lesson_04/result.json
不会修改任何源代码。
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
from mini_vllm.model import matrix as M  # noqa: E402
from mini_vllm.model import attention_ref as A  # noqa: E402
from mini_vllm.trace import Tracer  # noqa: E402


def run_experiment(config_path: Path, mode: str, tracer: Tracer):
    cfg = json.loads(config_path.read_text("utf-8"))
    wl = json.loads((_ROOT / cfg["workload"]).read_text("utf-8"))
    Q, K, V = wl["Q"], wl["K"], wl["V"]
    causal = cfg.get("causal", True)

    out, st = A.scaled_dot_product_attention(Q, K, V, causal=causal, return_stages=True)

    with tracer.section("Attention (single head, causal=%s)" % causal):
        tracer.event("shapes", Q=list(M.shape(Q)), K=list(M.shape(K)), V=list(M.shape(V)),
                     out=list(M.shape(out)))
        tracer.detail("1) scores = Q·Kᵀ", rows=st["scores"])
        tracer.detail("2) scaled = scores/√d", scale=round(st["scale"], 6), rows=st["scaled"])
        tracer.detail("3) masked (未来置 -inf)", rows=[[("-inf" if x == float("-inf") else round(x, 4)) for x in r] for r in st["masked"]])
        tracer.detail("4) weights = softmax(masked)", rows=[[round(x, 4) for x in r] for r in st["weights"]])
        tracer.detail("5) out = weights·V", rows=st["out"])
        # 验证每行权重和为 1
        sums = [round(sum(r), 6) for r in st["weights"]]
        tracer.event("row weight sums (应全为 1)", sums=sums)

    return {
        "shapes": {k: list(v) for k, v in st["shapes"].items()},
        "scores": st["scores"], "scaled": st["scaled"],
        "weights": st["weights"], "out": st["out"],
        "row_weight_sums": [sum(r) for r in st["weights"]],
        "causal": causal,
    }


def print_summary(result) -> None:
    print()
    print("=" * 64)
    print("  Lesson 4 · Attention 手算（单头 causal）—— 运行成功 ✓")
    print("=" * 64)
    print(f"  shapes: Q{tuple(result['shapes']['Q'])} K{tuple(result['shapes']['K'])} "
          f"V{tuple(result['shapes']['V'])} → out{tuple(result['shapes']['out'])}")
    print("  1) scores = Q·Kᵀ:")
    print(M.pretty(result["scores"], width=8, prec=3))
    print("  4) weights = softmax(mask(scaled))（逐行和为 1）:")
    print(M.pretty(result["weights"], width=8, prec=3))
    print("  5) out = weights·V:")
    print(M.pretty(result["out"], width=8, prec=3))
    print("-" * 64)
    print("  证据：")
    sums = result["row_weight_sums"]
    max_err = max(abs(s - 1.0) for s in sums)
    print(f"    ✓ 每个 query 的注意力权重和为 1（最大误差 {max_err:.2e}）")
    if result["causal"]:
        w0 = result["weights"][0]
        print(f"    ✓ causal：第 0 个 query 只看得到 key 0 —— weights[0] = "
              f"{[round(x,3) for x in w0]}")
    print("=" * 64)
    print("  下一步：python3 course.py check 4   或   网页 Attention Stepper 逐步观察五步。")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="experiments.lesson_04_attention",
                                description="Lesson 4：单头 causal attention")
    p.add_argument("--config", default="configs/lesson_04_quick.json")
    p.add_argument("--mode", default="quick", choices=list(RUN_MODES))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", default="outputs/lesson_04")
    args = p.parse_args(argv)

    config_path = (_ROOT / args.config).resolve()
    if not config_path.exists():
        print(f"[错误] 找不到配置文件：{config_path}", file=sys.stderr)
        return 2

    tracer = Tracer.from_flags(verbose=args.verbose, trace=args.trace)
    result = run_experiment(config_path, args.mode, tracer)

    out_dir = (_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
    print_summary(result)
    print(f"  结果已写入：{(out_dir / 'result.json').relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
