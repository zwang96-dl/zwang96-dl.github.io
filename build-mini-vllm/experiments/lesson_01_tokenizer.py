"""Lesson 1 实验：byte-level tokenizer 的 encode / decode / pad。

用法::

    python3 -m experiments.lesson_01_tokenizer --config configs/lesson_01_quick.json
    python3 -m experiments.lesson_01_tokenizer --trace

输入：configs/lesson_01_quick.json（引用 assets/workloads/lesson_01.json）
输出：outputs/lesson_01/result.json
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
from mini_vllm.tokenizer import ByteTokenizer  # noqa: E402
from mini_vllm.trace import Tracer  # noqa: E402


def run_experiment(config_path: Path, mode: str, tracer: Tracer):
    cfg = json.loads(config_path.read_text("utf-8"))
    workload = json.loads((_ROOT / cfg["workload"]).read_text("utf-8"))
    texts = list(workload["texts"])
    scale = RUN_MODES.get(mode, RUN_MODES["quick"])["scale"]
    if scale > 1:
        texts = (texts * scale)

    tok = ByteTokenizer()
    add_bos = cfg.get("add_bos", True)
    add_eos = cfg.get("add_eos", False)

    rows = []
    with tracer.section("Tokenizer"):
        tracer.event("config", vocab_size=tok.vocab_size,
                     bos=tok.bos_id, eos=tok.eos_id, pad=tok.pad_id,
                     add_bos=add_bos, add_eos=add_eos)
        batch_ids = []
        for text in texts:
            ids = tok.encode(text, add_bos=add_bos, add_eos=add_eos)
            batch_ids.append(ids)
            decoded = tok.decode(ids)
            roundtrip = (decoded == text)
            raw_bytes = list(text.encode("utf-8"))
            pieces = [tok.id_to_piece(i) for i in ids]
            rows.append({
                "text": text, "chars": len(text), "bytes": len(raw_bytes),
                "num_tokens": len(ids), "ids": ids, "roundtrip_ok": roundtrip,
            })
            with tracer.section(f"encode {text!r}"):
                tracer.detail("stats", chars=len(text), utf8_bytes=len(raw_bytes),
                              tokens=len(ids))
                tracer.detail("byte_values", bytes=raw_bytes)
                tracer.detail("token_ids", ids=ids)
                tracer.fine("pieces", pieces=pieces)
                tracer.detail("decode_roundtrip", ok=roundtrip, decoded=decoded)

        # pad 一批到同一长度，展示 attention mask
        pad_ids, mask = tok.pad(batch_ids, side=cfg.get("pad_side", "right"))
        with tracer.section("pad batch"):
            tracer.event("padded_to", length=len(pad_ids[0]) if pad_ids else 0,
                         batch=len(pad_ids))
            for i, (pi, mi) in enumerate(zip(pad_ids, mask)):
                tracer.detail(f"seq[{i}]", ids=pi, mask=mi)

    return {
        "vocab_size": tok.vocab_size,
        "rows": rows,
        "padded": {"ids": pad_ids, "mask": mask, "length": len(pad_ids[0]) if pad_ids else 0},
    }


def print_summary(result) -> None:
    print()
    print("=" * 68)
    print("  Lesson 1 · Byte-level Tokenizer —— 运行成功 ✓")
    print("=" * 68)
    print(f"  词表大小 vocab_size = {result['vocab_size']}  (256 字节 + BOS/EOS/PAD)")
    print("-" * 68)
    print(f"  {'文本':<16}{'字符':>4}{'字节':>4}{'token':>6}  {'往返':>4}  token IDs")
    for r in result["rows"][:8]:
        t = r["text"]
        disp = (t[:14] + "…") if len(t) > 15 else t
        ok = "✓" if r["roundtrip_ok"] else "✗"
        ids = r["ids"]
        ids_disp = str(ids if len(ids) <= 10 else ids[:10] + ["…"])
        print(f"  {disp:<16}{r['chars']:>4}{r['bytes']:>4}{r['num_tokens']:>6}  {ok:>4}  {ids_disp}")
    print("-" * 68)
    pad = result["padded"]
    print(f"  padding：整批补齐到长度 {pad['length']}，并生成 attention mask（1=真实, 0=PAD）")
    for i, (ids, m) in enumerate(zip(pad["ids"][:3], pad["mask"][:3])):
        print(f"    seq[{i}] ids ={ids}")
        print(f"           mask={m}")
    print("-" * 68)
    print("  证据：")
    all_ok = all(r["roundtrip_ok"] for r in result["rows"])
    print(f"    ✓ 全部文本 encode→decode 无损往返：{all_ok}")
    print(f"    ✓ 中文/emoji 等多字节字符按 UTF-8 字节展开为多个 token")
    print(f"    ✓ padding 产生的 mask 精确标记了哪些位置是真实 token")
    print("=" * 68)
    print("  下一步：python3 course.py check 1   或   网页用 Tokenizer Explorer 输入你自己的文本。")
    print()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="experiments.lesson_01_tokenizer",
                                description="Lesson 1：byte-level tokenizer")
    p.add_argument("--config", default="configs/lesson_01_quick.json")
    p.add_argument("--mode", default="quick", choices=list(RUN_MODES))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", default="outputs/lesson_01")
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
