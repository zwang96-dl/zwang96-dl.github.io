"""Lesson 2 实验：Tensor shape 与矩阵乘法（row-column rule + broadcasting）。

用法::

    python3 -m experiments.lesson_02_tensor --config configs/lesson_02_quick.json
    python3 -m experiments.lesson_02_tensor --trace

输入：configs/lesson_02_quick.json（引用 assets/workloads/lesson_02.json）
输出：outputs/lesson_02/result.json
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
from mini_vllm.trace import Tracer  # noqa: E402


def run_experiment(config_path: Path, mode: str, tracer: Tracer):
    cfg = json.loads(config_path.read_text("utf-8"))
    wl = json.loads((_ROOT / cfg["workload"]).read_text("utf-8"))
    A, B, bias = wl["A"], wl["B"], wl["bias"]

    with tracer.section("Shapes"):
        tracer.event("A", shape=list(M.shape(A)))
        tracer.event("B", shape=list(M.shape(B)))
        # 一个 (batch, seq, hidden) 例子，解释三维语义
        example = wl.get("tensor3d")
        if example is not None:
            tracer.event("3D tensor (batch, seq, hidden)", shape=list(M.shape(example)))

    with tracer.section("matmul A@B (row-column rule)"):
        C = M.matmul(A, B)
        tracer.event("result shape", shape=list(M.shape(C)))
        # 展示一个被算出的元素 out[i][j] = Σ_k A[i][k]*B[k][j]
        i = j = 0
        terms = [f"{A[i][k]}*{B[k][j]}" for k in range(len(B))]
        worked = " + ".join(terms) + f" = {C[i][j]}"
        tracer.detail("worked element", position=f"out[{i}][{j}]", compute=worked)
        for r_i, row in enumerate(C):
            tracer.fine(f"C[{r_i}]", row=row)

    with tracer.section("transpose"):
        AT = M.transpose(A)
        tracer.event("A^T shape", shape=list(M.shape(AT)))

    with tracer.section("broadcasting: add_row_bias (m,n)+(n,)"):
        Cb = M.add_row_bias(C, bias)
        tracer.detail("bias", bias=bias)
        for r_i, row in enumerate(Cb):
            tracer.fine(f"C+bias[{r_i}]", row=row)

    # 演示 shape 不相容会被明确拒绝（不静默出错）
    incompat_msg = None
    try:
        M.matmul(A, A)  # (m,k)@(m,k) 通常不相容
    except ValueError as e:
        incompat_msg = str(e)
    with tracer.section("shape check"):
        tracer.event("incompatible matmul rejected", ok=incompat_msg is not None)

    return {
        "shape_A": list(M.shape(A)), "shape_B": list(M.shape(B)),
        "A": A, "B": B, "C": C, "AT": AT, "C_plus_bias": Cb,
        "worked_element": {"pos": [0, 0], "value": C[0][0]},
        "incompatible_rejected": incompat_msg is not None,
        "incompatible_message": incompat_msg,
    }


def print_summary(result) -> None:
    print()
    print("=" * 64)
    print("  Lesson 2 · Tensor Shape 与矩阵乘法 —— 运行成功 ✓")
    print("=" * 64)
    print(f"  A shape = {tuple(result['shape_A'])}   B shape = {tuple(result['shape_B'])}")
    print("  A @ B =")
    print(M.pretty(result["C"], width=8, prec=2))
    we = result["worked_element"]
    print(f"  手算校验 out[0][0] = {we['value']}（= A 第0行 · B 第0列）")
    print("  A + bias（行广播）=")
    print(M.pretty(result["C_plus_bias"], width=8, prec=2))
    print("-" * 64)
    print("  证据：")
    print(f"    ✓ (m,k)@(k,n) 得到 shape {tuple(M.shape(result['C']))}")
    print(f"    ✓ 形状不相容的 matmul 被明确拒绝：{result['incompatible_rejected']}")
    print("=" * 64)
    print("  下一步：python3 course.py check 2   或   网页 Matrix Multiplication Explorer 点格子看乘加。")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="experiments.lesson_02_tensor",
                                description="Lesson 2：Tensor shape 与矩阵乘法")
    p.add_argument("--config", default="configs/lesson_02_quick.json")
    p.add_argument("--mode", default="quick", choices=list(RUN_MODES))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", default="outputs/lesson_02")
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
