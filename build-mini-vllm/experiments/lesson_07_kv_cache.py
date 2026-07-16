"""Lesson 7 实验：KV Cache 正确性对齐与内存公式。

验证 cached 生成与 naive 生成**逐 token 一致**，并展示缓存的 shape 与内存估算。

用法::

    python3 -m experiments.lesson_07_kv_cache --config configs/lesson_07_quick.json
    python3 -m experiments.lesson_07_kv_cache --trace

输出：outputs/lesson_07/result.json；不修改源代码。
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
from mini_vllm.model import matrix as M  # noqa: E402
from mini_vllm.cache.kv_cache import KVCache  # noqa: E402
from mini_vllm.tokenizer import ByteTokenizer  # noqa: E402
from mini_vllm.sampling import Sampler, SamplingParams, argmax  # noqa: E402
from mini_vllm.engine.generate import generate_naive, generate_cached  # noqa: E402
from mini_vllm.trace import Tracer  # noqa: E402


def run_experiment(config_path: Path, mode: str, tracer: Tracer):
    cfg = json.loads(config_path.read_text("utf-8"))
    model = load_checkpoint(_ROOT / cfg["checkpoint"])
    tok = ByteTokenizer()
    prompt = cfg.get("prompt", "Hello")
    n = int(cfg.get("max_new_tokens", 10))
    ids = tok.encode(prompt, add_bos=True)

    # 1) 逐 logits 对齐：prefill 后再 decode 一步，比较 naive 与 cached 的最后一行 logits
    naive_prefill = model.forward(ids, list(range(len(ids))))
    cache = KVCache(model.cfg)
    cached_prefill = model.forward(ids, list(range(len(ids))), cache)
    prefill_diff = M.max_abs_diff(naive_prefill, cached_prefill)
    nxt = argmax(naive_prefill[-1])
    naive2 = model.forward(ids + [nxt], list(range(len(ids) + 1)))
    dec = model.forward([nxt], [len(ids)], cache)
    decode_diff = M.max_abs_diff([naive2[-1]], [dec[0]])

    # 2) 整段生成对齐
    gn = generate_naive(model, ids, n, Sampler(SamplingParams()), stop_on_eos=False)
    gc = generate_cached(model, ids, n, Sampler(SamplingParams()), stop_on_eos=False)

    with tracer.section("KV Cache"):
        tracer.event("prefill_logits_diff", max_abs=prefill_diff)
        tracer.event("decode_logits_diff", max_abs=decode_diff)
        # 注意：这是「对齐检查」用的短缓存（prefill + 1 步 decode 后），length 较小；
        # 下方摘要的内存数字用的是「完整生成 prompt+max_new 后」的估算，两者快照点不同，都对。
        tracer.event("cache(对齐检查快照:prefill+1decode)", length=cache.length,
                     **cache.memory_estimate())

    mem = KVCache(model.cfg)
    mem.positions = list(range(len(ids) + n))  # 用于展示公式
    return {
        "prompt": prompt, "prompt_len": len(ids), "max_new_tokens": n,
        "prefill_logits_max_abs_diff": prefill_diff,
        "decode_logits_max_abs_diff": decode_diff,
        "generation_identical": gn.generated == gc.generated,
        "cache_memory": mem.memory_estimate(),
        "config": {"layers": model.cfg.num_layers, "kv_heads": model.cfg.num_kv_heads,
                   "head_dim": model.cfg.head_dim},
    }


def print_summary(r) -> None:
    print()
    print("=" * 68)
    print("  Lesson 7 · KV Cache —— 运行成功 ✓")
    print("=" * 68)
    print(f"  prompt_len={r['prompt_len']}  max_new_tokens={r['max_new_tokens']}")
    print("-" * 68)
    print("  正确性对齐（cached 必须与 naive 逐值一致）：")
    print(f"    ✓ prefill logits 最大绝对误差 = {r['prefill_logits_max_abs_diff']:.3e}")
    print(f"    ✓ decode  logits 最大绝对误差 = {r['decode_logits_max_abs_diff']:.3e}")
    print(f"    ✓ 整段生成结果与 naive 完全一致：{r['generation_identical']}")
    print("-" * 68)
    mem = r["cache_memory"]
    print(f"  内存估算（完整生成后：prompt {r['prompt_len']} + max_new {r['max_new_tokens']} = "
          f"{mem['tokens']} tokens）：")
    print(f"  内存公式：{mem['formula']}")
    print(f"    每 token 每层 = 2·kv_heads·head_dim = {mem['per_token_per_layer']}")
    print(f"    共 {mem['tokens']} tokens × {mem['num_layers']} 层 → {mem['total_elements']} 个数")
    print("=" * 68)
    print("  下一步：python3 course.py check 7   或   Lesson 8 度量 TTFT/TPOT/ITL。")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments.lesson_07_kv_cache")
    p.add_argument("--config", default="configs/lesson_07_quick.json")
    p.add_argument("--mode", default="quick", choices=list(RUN_MODES))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", default="outputs/lesson_07")
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
