"""Lesson 3 实验：极小 decoder-only Transformer 的一次前向。

用法::

    python3 -m experiments.lesson_03_transformer --config configs/lesson_03_quick.json
    python3 -m experiments.lesson_03_transformer --trace

输出：outputs/lesson_03/result.json；不修改源代码。
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
from mini_vllm.tokenizer import ByteTokenizer  # noqa: E402
from mini_vllm.trace import Tracer  # noqa: E402
from mini_vllm.sampling import argmax  # noqa: E402


def run_experiment(config_path: Path, mode: str, tracer: Tracer):
    cfg = json.loads(config_path.read_text("utf-8"))
    model = load_checkpoint(_ROOT / cfg["checkpoint"])
    tok = ByteTokenizer()
    text = cfg.get("prompt", "Hello")
    ids = tok.encode(text, add_bos=True)

    with tracer.section("TinyTextModel.forward"):
        tracer.event("model",
                     hidden=model.cfg.hidden_size, layers=model.cfg.num_layers,
                     heads=model.cfg.num_attention_heads, kv_heads=model.cfg.num_kv_heads,
                     head_dim=model.cfg.head_dim, vocab=model.cfg.vocab_size)
        tracer.event("input", prompt=text, input_ids=ids, seq_len=len(ids))
        logits = model.forward(ids, list(range(len(ids))), tracer=tracer)

    next_id = argmax(logits[-1])
    return {
        "config": {"hidden": model.cfg.hidden_size, "layers": model.cfg.num_layers,
                   "heads": model.cfg.num_attention_heads, "kv_heads": model.cfg.num_kv_heads,
                   "head_dim": model.cfg.head_dim, "vocab": model.cfg.vocab_size},
        "prompt": text, "input_ids": ids, "seq_len": len(ids),
        "logits_shape": list(M.shape(logits)),
        "next_token_id": next_id, "next_token_piece": tok.id_to_piece(next_id),
    }


def print_summary(r) -> None:
    c = r["config"]
    print()
    print("=" * 66)
    print("  Lesson 3 · 极小 Decoder-only Transformer —— 前向成功 ✓")
    print("=" * 66)
    print(f"  模型：hidden={c['hidden']} layers={c['layers']} heads={c['heads']} "
          f"kv_heads={c['kv_heads']}(GQA) head_dim={c['head_dim']} vocab={c['vocab']}")
    print(f"  输入：{r['prompt']!r} → input_ids {r['input_ids']}（seq_len={r['seq_len']}）")
    print(f"  数据流：embedding (seq,{c['hidden']}) → [RMSNorm→QKV→RoPE→attn→残差→"
          f"RMSNorm→SwiGLU→残差]×{c['layers']} → RMSNorm → LM head")
    print(f"  输出：logits.shape = {tuple(r['logits_shape'])}  "
          f"（最后一维 {c['vocab']} = 词表大小）")
    print("-" * 66)
    print(f"  证据：")
    print(f"    ✓ 前向产出 logits，形状 (seq_len, vocab) = {tuple(r['logits_shape'])}")
    print(f"    ✓ 预测的下一个 token id = {r['next_token_id']}（piece={r['next_token_piece']!r}）")
    print("=" * 66)
    print("  下一步：python3 course.py check 3   或   Lesson 5 让它真正「生成」。")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="experiments.lesson_03_transformer")
    p.add_argument("--config", default="configs/lesson_03_quick.json")
    p.add_argument("--mode", default="quick", choices=list(RUN_MODES))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", default="outputs/lesson_03")
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
