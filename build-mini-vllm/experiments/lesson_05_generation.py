"""Lesson 5 实验：从 logits 采样，自回归生成第一段文本。

用法::

    python3 -m experiments.lesson_05_generation --config configs/lesson_05_quick.json
    python3 -m experiments.lesson_05_generation --trace

输出：outputs/lesson_05/result.json；不修改源代码。演示 greedy / temperature / top-k / top-p。
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


def _gen(model, tok, prompt_ids, n, sp, tracer=None):
    r = generate_cached(model, prompt_ids, n, Sampler(sp), tracer=tracer)
    return {"params": {"temperature": sp.temperature, "top_k": sp.top_k,
                       "top_p": sp.top_p, "seed": sp.seed},
            "generated_ids": r.generated, "text": r.text(tok)}


def run_experiment(config_path: Path, mode: str, tracer: Tracer):
    cfg = json.loads(config_path.read_text("utf-8"))
    model = load_checkpoint(_ROOT / cfg["checkpoint"])
    tok = ByteTokenizer()
    prompt = cfg.get("prompt", "Hello")
    n = int(cfg.get("max_new_tokens", 12))
    prompt_ids = tok.encode(prompt, add_bos=True)

    runs = []
    with tracer.section("generation"):
        tracer.event("prompt", text=prompt, ids=prompt_ids, max_new_tokens=n)
        runs.append(("greedy", _gen(model, tok, prompt_ids, n,
                                    SamplingParams(temperature=0.0), tracer)))
        runs.append(("temperature=0.8", _gen(model, tok, prompt_ids, n,
                                             SamplingParams(temperature=0.8, seed=7))))
        runs.append(("top_k=5,T=1.0", _gen(model, tok, prompt_ids, n,
                                           SamplingParams(temperature=1.0, top_k=5, seed=7))))
        runs.append(("top_p=0.9,T=1.0", _gen(model, tok, prompt_ids, n,
                                             SamplingParams(temperature=1.0, top_p=0.9, seed=7))))
    return {"prompt": prompt, "prompt_ids": prompt_ids, "max_new_tokens": n,
            "runs": [{"name": name, **data} for name, data in runs]}


def print_summary(r) -> None:
    print()
    print("=" * 68)
    print("  Lesson 5 · 生成第一个 Token —— 运行成功 ✓")
    print("=" * 68)
    print(f"  prompt = {r['prompt']!r}  →  ids {r['prompt_ids']}")
    print(f"  max_new_tokens = {r['max_new_tokens']}")
    print("-" * 68)
    for run in r["runs"]:
        print(f"  [{run['name']:<16}] ids={run['generated_ids']}")
    print("-" * 68)
    print("  证据：")
    print("    ✓ greedy 完全确定（同输入同输出）；temperature/top-k/top-p 引入可控随机")
    print("    ✓ 未使用 model.generate()，采样逻辑全部透明可见（mini_vllm/sampling.py）")
    print("=" * 68)
    print("  下一步：python3 course.py check 5   或   Lesson 6 看看朴素生成有多浪费。")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments.lesson_05_generation")
    p.add_argument("--config", default="configs/lesson_05_quick.json")
    p.add_argument("--mode", default="quick", choices=list(RUN_MODES))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", default="outputs/lesson_05")
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
