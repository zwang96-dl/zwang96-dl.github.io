"""Lesson 0 实验：跑通请求生命周期机制模拟器。

用法::

    python3 -m experiments.lesson_00_intro --config configs/lesson_00_quick.json
    python3 -m experiments.lesson_00_intro --mode normal --trace
    python3 -m experiments.lesson_00_intro --verbose

输入：configs/lesson_00_quick.json（引用 assets/workloads/lesson_00.json）
输出：outputs/lesson_00/result.json
不会修改任何源代码。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 允许 `python3 -m experiments.lesson_00_intro` 以及从任意 cwd 运行。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mini_vllm.config import RUN_MODES  # noqa: E402
from mini_vllm.simulator import LifecycleSimulator, SimRequest  # noqa: E402
from mini_vllm.trace import Tracer  # noqa: E402


def load_workload(path: Path) -> list[SimRequest]:
    data = json.loads(path.read_text("utf-8"))
    reqs = []
    for item in data["requests"]:
        reqs.append(SimRequest(
            request_id=item["request_id"],
            prompt=item["prompt"],
            max_new_tokens=int(item["max_new_tokens"]),
            arrival=int(item.get("arrival", 0)),
        ))
    return reqs


def build_simulator(cfg: dict, tracer: Tracer) -> LifecycleSimulator:
    return LifecycleSimulator(
        block_size=int(cfg.get("block_size", 16)),
        num_blocks=int(cfg.get("num_blocks", 32)),
        max_num_seqs=int(cfg.get("max_num_seqs", 4)),
        token_budget=int(cfg.get("token_budget", 64)),
        tracer=tracer,
    )


def run_experiment(config_path: Path, mode: str, tracer: Tracer) -> tuple[object, dict]:
    cfg = json.loads(config_path.read_text("utf-8"))
    workload_path = (_ROOT / cfg["workload"]).resolve()
    requests = load_workload(workload_path)

    # mode 通过复制请求来放大规模（quick=1x, normal=4x, stress=16x）。
    scale = RUN_MODES.get(mode, RUN_MODES["quick"])["scale"]
    if scale > 1:
        expanded = []
        for k in range(scale):
            for r in requests:
                expanded.append(SimRequest(
                    request_id=f"{r.request_id}_{k}",
                    prompt=r.prompt,
                    max_new_tokens=r.max_new_tokens,
                    arrival=r.arrival + (k if cfg.get("stagger_arrivals") else 0),
                ))
        requests = expanded

    sim = build_simulator(cfg, tracer)
    result = sim.run(requests)
    return result, cfg


def print_summary(result, cfg: dict, mode: str) -> None:
    outputs = result.outputs()
    print()
    print("=" * 64)
    print("  Lesson 0 · 请求生命周期机制模拟器 —— 运行成功 ✓")
    print("=" * 64)
    print(f"  运行规模 (mode)         : {mode}")
    print(f"  请求数 (requests)       : {len(result.requests)}")
    print(f"  总迭代数 (iterations)   : {result.total_iterations}")
    print(f"  调度 token 总数         : {result.total_scheduled_tokens}")
    print(f"  KV block 峰值占用       : {result.peak_blocks_in_use} / {result.num_blocks}"
          f"  (block_size={result.block_size})")
    print("-" * 64)
    print("  每个请求的结果：")
    for r in result.requests:
        state = r.state.value
        ttft = r.first_token_iter
        gen = len(r.generated)
        preview = outputs[r.request_id]
        if len(preview) > 24:
            preview = preview[:24] + "…"
        print(f"    {r.request_id:<12} state={state:<9} "
              f"prompt_len={r.prompt_len:<3} generated={gen:<3} "
              f"首token@iter={ttft}  输出≈{preview!r}")
    print("-" * 64)
    print("  证据（这些数字应当稳定可复现，因为下一 token 由确定性函数产生）：")
    r0 = result.requests[0]
    print(f"    ✓ 请求状态已从 WAITING 走到 {r0.state.value}")
    print(f"    ✓ prefill 一次处理 {r0.prompt_len} 个 token，decode 每步只处理 1 个")
    print(f"    ✓ 结束后 KV block 全部归还（无泄漏），断言已在模拟器内通过")
    print("=" * 64)
    print("  下一步：运行测试  python3 -m unittest tests.lesson_00.test_intro")
    print("         或返回网页 Lesson 0，用动画单步观察同一套过程。")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="experiments.lesson_00_intro",
        description="Lesson 0：请求生命周期机制模拟器（无神经网络，纯标准库）",
    )
    p.add_argument("--config", default="configs/lesson_00_quick.json",
                   help="配置文件路径（相对项目根目录）")
    p.add_argument("--mode", default="quick", choices=list(RUN_MODES),
                   help="运行规模：quick(数秒) / normal / stress")
    p.add_argument("--verbose", action="store_true", help="打印更多中间状态")
    p.add_argument("--trace", action="store_true", help="打印完整执行时序（每步状态）")
    p.add_argument("--out", default="outputs/lesson_00", help="结果输出目录")
    args = p.parse_args(argv)

    config_path = (_ROOT / args.config).resolve()
    if not config_path.exists():
        print(f"[错误] 找不到配置文件：{config_path}", file=sys.stderr)
        print("       请确认在 build-mini-vllm/ 目录下运行，或用 --config 指定正确路径。",
              file=sys.stderr)
        return 2

    tracer = Tracer.from_flags(verbose=args.verbose, trace=args.trace)
    result, cfg = run_experiment(config_path, args.mode, tracer)

    out_dir = (_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "result.json"
    out_file.write_text(json.dumps(result.to_json(), indent=2, ensure_ascii=False), "utf-8")

    print_summary(result, cfg, args.mode)
    print(f"  结果已写入：{out_file.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
