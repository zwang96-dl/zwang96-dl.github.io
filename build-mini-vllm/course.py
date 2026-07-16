#!/usr/bin/env python3
"""course.py —— Build Your Own Mini-vLLM 的统一、透明入口。

设计原则（对应 Prompt 第一章「产品哲学」与第二十一章「Course Runner」）：

    - **透明的薄封装**：每个封装命令在执行前都会打印它等价的底层命令、
      输入、输出、以及「是否修改文件」。你随时可以绕过 course.py 直接运行
      底层的 `python3 -m ...`。
    - **默认不修改源代码**：没有任何命令会在后台自动改你的代码或做 git restore。
    - **友好的错误**：默认给出可读的错误信息；加 `--debug` 才显示完整 traceback。
    - **不接受任意 shell 命令**：只暴露一组固定的、可审计的子命令。
    - **simulator-only 可运行**：核心命令只依赖 Python 标准库。

用法::

    python3 course.py                      # 终端版首页：我在哪、下一步做什么
    python3 course.py serve                # 启动本地静态课程网页
    python3 course.py doctor [--offline]   # 环境自检
    python3 course.py verify-offline-bundle
    python3 course.py where-am-i
    python3 course.py lesson 0
    python3 course.py run 0 [--mode quick] [--trace]
    python3 course.py check 0
    python3 course.py hint 0 --level 1
    python3 course.py inspect blocks|scheduler|kv-cache
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_vllm import MIN_PYTHON, __version__  # noqa: E402
from mini_vllm import course_meta  # noqa: E402
from mini_vllm.offline_check import verify_bundle, format_report  # noqa: E402

PROGRESS_FILE = ROOT / ".course_progress.json"

# ANSI 颜色（仅在 TTY 时启用，保持输出在重定向/CI 下干净）。
_TTY = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def bold(s: str) -> str: return _c(s, "1")
def green(s: str) -> str: return _c(s, "32")
def yellow(s: str) -> str: return _c(s, "33")
def cyan(s: str) -> str: return _c(s, "36")
def red(s: str) -> str: return _c(s, "31")
def dim(s: str) -> str: return _c(s, "2")


# --------------------------------------------------------------------------- #
# 进度追踪（只记录阅读/完成进度，绝不修改源代码）
# --------------------------------------------------------------------------- #
def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text("utf-8"))
        except json.JSONDecodeError:
            pass
    return {"current_lesson": 0, "completed_checks": [], "started": False}


def save_progress(p: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(p, indent=2, ensure_ascii=False), "utf-8")


# --------------------------------------------------------------------------- #
# 透明性辅助：打印「等价的底层命令」
# --------------------------------------------------------------------------- #
def announce_equivalent(cmd: str, *, inputs: str = "", outputs: str = "",
                        modifies_source: bool = False) -> None:
    print(dim("─" * 60))
    print(dim("该命令等价于："))
    print("  " + cyan(cmd))
    if inputs:
        print(dim(f"输入：{inputs}"))
    if outputs:
        print(dim(f"输出：{outputs}"))
    print(dim("修改源代码：" + ("是" if modifies_source else "否")))
    print(dim("─" * 60))
    # 必须先把「等价命令」刷出去，再启动子进程；否则在管道/重定向(非 TTY)下，
    # 父进程的这段是块缓冲的，会晚于子进程输出打印，显得「等价命令在结果之后」。
    sys.stdout.flush()


def _run(cmd: list[str]) -> int:
    """在项目根目录下运行子进程，透传输出。"""
    proc = subprocess.run(cmd, cwd=str(ROOT), env=os.environ.copy())
    return proc.returncode


def _git(args: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(["git", *args], cwd=str(ROOT),
                              capture_output=True, text=True)
        return proc.returncode, proc.stdout
    except FileNotFoundError:
        return 127, ""


# --------------------------------------------------------------------------- #
# 命令：serve
# --------------------------------------------------------------------------- #
def cmd_serve(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py serve", add_help=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args(argv)

    docs = ROOT / "docs"
    if not docs.exists():
        print(red("找不到 docs/ 目录。"), file=sys.stderr)
        return 1

    announce_equivalent(
        f"python3 -m http.server {args.port} --bind {args.host} --directory docs",
        inputs="docs/（静态课程网页）",
        outputs="（仅提供 HTTP 服务，不写文件）",
        modifies_source=False,
    )
    url = f"http://{args.host}:{args.port}/index.html"
    print(bold("启动本地课程网页（仅本机，无需网络）"))
    print("  在浏览器打开： " + cyan(url))
    print("  Lesson 0 直达： " + cyan(f"http://{args.host}:{args.port}/lessons/lesson_00.html"))
    print(dim("  按 Ctrl+C 停止。也可直接双击 docs/index.html 离线打开（部分功能需 localhost）。"))
    print()

    import http.server
    import functools

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(docs))
    try:
        with http.server.ThreadingHTTPServer((args.host, args.port), handler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n" + dim("已停止课程网页服务。"))
    except OSError as e:
        print(red(f"无法在 {args.host}:{args.port} 启动服务：{e}"), file=sys.stderr)
        print(dim("可能端口被占用，试试 --port 8001。"), file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# 命令：doctor
# --------------------------------------------------------------------------- #
def cmd_doctor(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py doctor")
    p.add_argument("--offline", action="store_true",
                   help="额外确认离线可用（运行离线自检、检查无需联网）")
    args = p.parse_args(argv)

    print(bold("环境自检（doctor）"))
    print("=" * 56)
    ok = True

    # Python 版本
    ver = sys.version_info
    py_ok = (ver.major, ver.minor) >= MIN_PYTHON
    print(f"  [{'✓' if py_ok else '✗'}] Python 版本 {ver.major}.{ver.minor}.{ver.micro}"
          f"（要求 ≥ {MIN_PYTHON[0]}.{MIN_PYTHON[1]}）")
    if not py_ok:
        ok = False
        print(dim(f"      请用 python3.{MIN_PYTHON[1]} 或更高版本运行。"))

    # 平台
    import platform
    print(f"  [✓] 平台 {platform.system()} {platform.machine()}"
          + dim("（Apple Silicon 首选，但仅 CPU + 标准库即可）"))

    # 核心可导入（simulator-only）
    try:
        from mini_vllm.simulator import LifecycleSimulator  # noqa: F401
        print("  [✓] 核心模块可导入（机制模拟器可运行，无需第三方库）")
    except Exception as e:  # pragma: no cover
        ok = False
        print(red(f"  [✗] 核心模块导入失败：{e}"))

    # 可选加速库（信息性，不影响核心）
    def _probe(name: str) -> str:
        try:
            __import__(name)
            return "已安装"
        except ImportError:
            return "未安装（可选，不影响核心与 Level A）"
    for lib in ("torch", "numpy", "PIL"):
        print(dim(f"  [·] 可选库 {lib}: {_probe(lib)}"))

    # outputs 可写
    try:
        outdir = ROOT / "outputs"
        outdir.mkdir(exist_ok=True)
        (outdir / ".write_test").write_text("ok", "utf-8")
        (outdir / ".write_test").unlink()
        print("  [✓] outputs/ 目录可写")
    except Exception as e:
        ok = False
        print(red(f"  [✗] outputs/ 不可写：{e}"))

    if args.offline:
        print("-" * 56)
        print(bold("  离线附加检查"))
        report = verify_bundle(ROOT, check_wheels=True)
        for c in report.checks:
            print(f"  [{'✓' if c.ok else '✗'}] {c.name}")
            if not c.ok:
                ok = False
        # 环境变量提示（不强制）
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            val = os.environ.get(var)
            print(dim(f"  [·] {var} = {val or '(未设置，本课程本就不联网，可不设)'}"))

    print("=" * 56)
    print(green("  结论：环境就绪 ✓") if ok else red("  结论：存在问题，请按上面提示修复 ✗"))
    if ok:
        print(dim("  下一步： python3 course.py serve   或   python3 course.py run 0"))
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# 命令：verify-offline-bundle
# --------------------------------------------------------------------------- #
def cmd_verify_offline(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py verify-offline-bundle")
    p.add_argument("--no-wheels", action="store_true",
                   help="跳过 wheel 校验（仅检查静态资源与代码）")
    args = p.parse_args(argv)
    report = verify_bundle(ROOT, check_wheels=not args.no_wheels)
    print(format_report(report))
    return 0 if report.ok else 1


# --------------------------------------------------------------------------- #
# 命令：where-am-i
# --------------------------------------------------------------------------- #
def cmd_where_am_i(argv: list[str]) -> int:
    prog = load_progress()
    cur = prog.get("current_lesson", 0)
    les = course_meta.get(cur)
    print(bold("你现在在哪里（where-am-i）"))
    print("=" * 56)
    print(f"  当前课程：\n    Lesson {les.number} — {les.title}  [{les.status}]")
    done = sorted(prog.get("completed_checks", []))
    print(f"  已完成检查：{done if done else '（暂无）'}")
    print()
    if les.status == "ready":
        print("  下一步：")
        print(f"    1) 阅读网页  docs/lessons/lesson_{les.number:02d}.html")
        if les.modify:
            print(f"    2) 修改文件  {les.modify[0]}")
        print(f"    运行实验：  {cyan('python3 course.py run ' + str(les.number))}")
        print(f"    运行测试：  {cyan('python3 course.py check ' + str(les.number))}")
    else:
        print(f"  Lesson {les.number} 计划在 {les.phase} 实现（当前 Phase 尚未提供实验/网页）。")
        ready = course_meta.READY_LESSONS
        print(f"    现在可学的已就绪课程：{ready}")
        print(f"    建议先运行： {cyan('python3 course.py run ' + str(ready[0]))}")

    # Git 状态（只读，绝不自动修改）
    print("-" * 56)
    code, out = _git(["status", "--porcelain"])
    if code == 0:
        lines = [l for l in out.splitlines() if l.strip()]
        untracked = [l for l in lines if l.startswith("??")]
        modified = [l for l in lines if not l.startswith("??")]
        print(f"  当前 Git 状态：{len(modified)} 个已跟踪文件有改动"
              + (f"，{len(untracked)} 项未跟踪" if untracked else ""))
        for l in (modified + untracked)[:8]:
            print(dim("    " + l))
        if len(lines) > 8:
            print(dim(f"    …… 还有 {len(lines) - 8} 项"))
    else:
        print(dim("  （未检测到 git，可跳过版本管理相关步骤）"))
    print("=" * 56)
    print(dim("  本命令不会自动修改或恢复任何文件。"))
    return 0


# --------------------------------------------------------------------------- #
# 命令：lesson N
# --------------------------------------------------------------------------- #
def cmd_lesson(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py lesson")
    p.add_argument("number", type=int)
    args = p.parse_args(argv)
    try:
        les = course_meta.get(args.number)
    except KeyError as e:
        print(red(e.args[0] if e.args else str(e)), file=sys.stderr)
        return 2

    print(bold(f"Lesson {les.number} — {les.title}") + f"   [{les.phase} · {les.status}]")
    print("=" * 60)
    print("当前目标：")
    print("  " + les.goal)
    if les.why:
        print("为什么需要：")
        print("  " + les.why)
    print(f"本课依赖：{les.prereqs or '（无）'}")
    print(f"需要阅读的文件：\n    " + ("\n    ".join(les.read) if les.read else "（无）"))
    print(f"需要修改的文件：\n    " + ("\n    ".join(les.modify) if les.modify else "（本课无需改代码，重在运行与观察）"))
    print(f"需要关注的类或函数：\n    " + ("\n    ".join(les.focus) if les.focus else "（无）"))
    print(f"实验命令：\n    {les.experiment or '（本 Phase 暂未提供）'}")
    print(f"测试命令：\n    " + (f"python3 -m unittest {les.test}" if les.test else "（本 Phase 暂未提供）"))
    nxt = f"Lesson {les.next}" if les.next is not None else "（已是最后一课）"
    print(f"完成后下一步：{nxt}")
    print("=" * 60)
    if les.status == "ready":
        print(f"网页： docs/lessons/lesson_{les.number:02d}.html")
        print(green(f"上手： python3 course.py run {les.number}"))
    else:
        print(yellow(f"该课程尚未在当前 Phase 实现（计划：{les.phase}）。"))
        print(dim(f"已就绪课程：{course_meta.READY_LESSONS}"))
    return 0


# --------------------------------------------------------------------------- #
# 命令：run N
# --------------------------------------------------------------------------- #
def cmd_run(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py run")
    p.add_argument("number", type=int)
    p.add_argument("--mode", default="quick")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    try:
        les = course_meta.get(args.number)
    except KeyError as e:
        print(red(e.args[0] if e.args else str(e)), file=sys.stderr)
        return 2

    if les.status != "ready" or not les.experiment:
        print(yellow(f"Lesson {les.number} 的实验尚未在当前 Phase 实现（计划：{les.phase}）。"))
        print(dim(f"现在可运行的：{[f'run {n}' for n in course_meta.READY_LESSONS]}"))
        return 1

    # 把 course_meta 里的等价命令拼上用户附加的 flag，然后透明打印并执行。
    base = les.experiment.split()
    extra = ["--mode", args.mode]
    if args.trace:
        extra.append("--trace")
    if args.verbose:
        extra.append("--verbose")
    display_cmd = " ".join(base + extra)
    announce_equivalent(
        display_cmd,
        inputs=les.workload or les.config,
        outputs=les.outputs or "outputs/",
        modifies_source=False,
    )
    # 用当前解释器执行真实模块（base[0] 是 'python3'）。
    real = [sys.executable] + base[1:] + extra
    return _run(real)


# --------------------------------------------------------------------------- #
# 命令：check N
# --------------------------------------------------------------------------- #
def cmd_check(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py check")
    p.add_argument("number", type=int)
    args = p.parse_args(argv)

    try:
        les = course_meta.get(args.number)
    except KeyError as e:
        print(red(e.args[0] if e.args else str(e)), file=sys.stderr)
        return 2

    if les.status != "ready" or not les.test:
        print(yellow(f"Lesson {les.number} 的测试尚未在当前 Phase 实现（计划：{les.phase}）。"))
        return 1

    announce_equivalent(
        f"python3 -m unittest {les.test} -v",
        inputs="（测试代码与被测模块）",
        outputs="（仅打印测试结果，不写文件）",
        modifies_source=False,
    )
    code = _run([sys.executable, "-m", "unittest", les.test, "-v"])
    if code == 0:
        print(green(f"\n✓ Lesson {les.number} 检查通过！"))
        prog = load_progress()
        done = set(prog.get("completed_checks", []))
        done.add(les.number)
        prog["completed_checks"] = sorted(done)
        prog["started"] = True
        if les.next is not None:
            prog["current_lesson"] = les.next
        save_progress(prog)
        nxt = les.next
        if nxt is not None:
            print(dim(f"进度已更新。下一课： python3 course.py lesson {nxt}"))
    else:
        print(red(f"\n✗ Lesson {les.number} 检查未通过。"))
        print(dim("  看上面的失败信息（含预期/实际/对应文件与函数），或用："))
        print(dim(f"  python3 course.py hint {les.number} --level 1"))
    return code


# --------------------------------------------------------------------------- #
# 命令：hint N --level L
# --------------------------------------------------------------------------- #
def cmd_hint(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py hint")
    p.add_argument("number", type=int)
    p.add_argument("--level", type=int, default=1, choices=[1, 2, 3, 4])
    args = p.parse_args(argv)
    try:
        les = course_meta.get(args.number)
    except KeyError as e:
        print(red(e.args[0] if e.args else str(e)), file=sys.stderr)
        return 2
    if not les.hints:
        print(yellow(f"Lesson {les.number} 暂无提示（计划：{les.phase}）。"))
        return 1
    lvl = min(args.level, len(les.hints))
    print(bold(f"Lesson {les.number} · 提示 Level {lvl}/{len(les.hints)}"))
    print("  " + les.hints[lvl - 1])
    if lvl < len(les.hints):
        print(dim(f"  想要更具体？ python3 course.py hint {les.number} --level {lvl + 1}"))
    return 0


# --------------------------------------------------------------------------- #
# 命令：inspect <target>
# --------------------------------------------------------------------------- #
def cmd_inspect(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py inspect")
    p.add_argument("target",
                   choices=["blocks", "scheduler", "kv-cache",
                            "tokenizer", "matmul", "attention",
                            "model", "generate"])
    p.add_argument("--text", default="你好 mini-vLLM", help="inspect tokenizer 用的文本")
    args = p.parse_args(argv)

    # --- Lesson 3: 真实 tiny model 前向 ---
    if args.target == "model":
        from mini_vllm.model.transformer import load_checkpoint
        from mini_vllm.model import matrix as MX
        from mini_vllm.tokenizer import ByteTokenizer
        model = load_checkpoint(ROOT / "assets/checkpoints/tiny_text.json")
        tok = ByteTokenizer()
        ids = tok.encode("Hello", add_bos=True)
        c = model.cfg
        print(bold("Inspector · TinyTextModel（真实前向）"))
        print(f"  config: hidden={c.hidden_size} layers={c.num_layers} heads={c.num_attention_heads} "
              f"kv_heads={c.num_kv_heads}(GQA) head_dim={c.head_dim} vocab={c.vocab_size}")
        print(f"  input : 'Hello' → {ids}  (seq_len={len(ids)})")
        logits = model.forward(ids, list(range(len(ids))))
        print(f"  数据流: embedding (seq,{c.hidden_size}) → "
              f"[RMSNorm→QKV→RoPE→attn→+→RMSNorm→SwiGLU→+]×{c.num_layers} → RMSNorm → LM head")
        print(f"  output: logits.shape = {tuple(MX.shape(logits))}")
        return 0

    # --- Lesson 5: 生成 ---
    if args.target == "generate":
        from mini_vllm.model.transformer import load_checkpoint
        from mini_vllm.tokenizer import ByteTokenizer
        from mini_vllm.sampling import Sampler, SamplingParams
        from mini_vllm.engine.generate import generate_cached
        model = load_checkpoint(ROOT / "assets/checkpoints/tiny_text.json")
        tok = ByteTokenizer()
        ids = tok.encode(args.text, add_bos=True)
        g = generate_cached(model, ids, 12, Sampler(SamplingParams()), stop_on_eos=False)
        print(bold("Inspector · Generate（greedy, KV cached）"))
        print(f"  prompt   : {args.text!r} → {ids}")
        print(f"  generated: {g.generated}")
        print(f"  TTFT={g.ttft*1000:.2f}ms  TPOT={g.tpot*1000:.2f}ms  "
              f"processed={g.total_processed_tokens} tokens")
        return 0

    # --- Lesson 1: tokenizer ---
    if args.target == "tokenizer":
        from mini_vllm.tokenizer import ByteTokenizer
        tok = ByteTokenizer()
        ids = tok.encode(args.text, add_bos=True)
        print(bold("Inspector · Tokenizer（byte-level, vocab=%d）" % tok.vocab_size))
        print(f"  文本      : {args.text!r}  （{len(args.text)} 字符, "
              f"{len(args.text.encode('utf-8'))} 字节）")
        print(f"  token ids : {ids}")
        print(f"  pieces    : {[tok.id_to_piece(i) for i in ids]}")
        print(f"  decode    : {tok.decode(ids)!r}  （往返一致: {tok.decode(ids) == args.text}）")
        return 0

    # --- Lesson 2: matmul ---
    if args.target == "matmul":
        from mini_vllm.model import matrix as MX
        A_, B_ = [[1, 2, 3], [4, 5, 6]], [[1, 0], [0, 1], [1, 1]]
        C_ = MX.matmul(A_, B_)
        print(bold("Inspector · 矩阵乘法（row-column rule）"))
        print(f"  A {MX.shape(A_)} @ B {MX.shape(B_)} -> C {MX.shape(C_)}")
        terms = " + ".join(f"{A_[0][k]}*{B_[k][0]}" for k in range(len(B_)))
        print(f"  worked: out[0][0] = {terms} = {C_[0][0]}")
        print("  C =")
        print(MX.pretty(C_, width=7, prec=1))
        return 0

    # --- Lesson 4: attention ---
    if args.target == "attention":
        from mini_vllm.model import matrix as MX
        from mini_vllm.model import attention_ref as AT
        Q = [[1, 0], [1, 1], [0, 1]]
        K = [[1, 0], [0, 1], [1, 1]]
        V = [[10, 0], [0, 10], [5, 5]]
        out, st = AT.scaled_dot_product_attention(Q, K, V, causal=True, return_stages=True)
        print(bold("Inspector · Attention（单头 causal, d=2, T=3）"))
        print("  scores = Q·Kᵀ:");  print(MX.pretty(st["scores"], width=7, prec=2))
        print("  weights = softmax(mask(scaled)):"); print(MX.pretty(st["weights"], width=7, prec=3))
        print("  out = weights·V:"); print(MX.pretty(out, width=7, prec=3))
        print(dim("  第 0 个 query 只看 key0 → weights[0]=%s" % [round(x, 2) for x in st["weights"][0]]))
        return 0

    from mini_vllm.simulator import LifecycleSimulator, SimRequest
    from mini_vllm.trace import Tracer

    # 与 `course.py run 0` 用完全相同的 config + workload，保证 inspect 与 run 0 --trace、
    # 网页动画、Lesson 0 手算表「四者一致」。
    _cfg = json.loads((ROOT / "configs/lesson_00_quick.json").read_text("utf-8"))
    _wl = json.loads((ROOT / _cfg["workload"]).read_text("utf-8"))
    reqs = [SimRequest(request_id=it["request_id"], prompt=it["prompt"],
                       max_new_tokens=int(it["max_new_tokens"]), arrival=int(it.get("arrival", 0)))
            for it in _wl["requests"]]

    def _mk_sim(**extra):
        return LifecycleSimulator(block_size=int(_cfg.get("block_size", 8)),
                                  num_blocks=int(_cfg.get("num_blocks", 16)),
                                  max_num_seqs=int(_cfg.get("max_num_seqs", 2)),
                                  token_budget=int(_cfg.get("token_budget", 32)), **extra)

    if args.target == "scheduler":
        print(bold("Inspector · Scheduler（每次迭代的调度决策，与 `run 0` 一致）"))
        result = _mk_sim().run([SimRequest(r.request_id, r.prompt, r.max_new_tokens, r.arrival) for r in reqs])
        print(f"{'iter':>4}  {'scheduled':<20} {'tokens':>6} {'phases'}")
        for rec in result.records:
            phases = ", ".join(f"{k}:{v}" for k, v in rec.phases.items())
            print(f"{rec.iteration:>4}  {str(rec.scheduled):<20} {rec.scheduled_tokens:>6}  {phases}")
        return 0

    if args.target == "blocks":
        print(bold("Inspector · KV Block 分配（逐迭代占用/空闲）"))
        result = _mk_sim().run([SimRequest(r.request_id, r.prompt, r.max_new_tokens, r.arrival) for r in reqs])
        print(f"{'iter':>4}  {'in_use':>6} {'free':>6}   {'占用条'}")
        for rec in result.records:
            bar = "█" * rec.blocks_in_use + "·" * rec.free_blocks
            print(f"{rec.iteration:>4}  {rec.blocks_in_use:>6} {rec.free_blocks:>6}   {bar}")
        print(dim(f"峰值占用 {result.peak_blocks_in_use}/{result.num_blocks}，"
                  f"结束后必须归零（无泄漏）。"))
        return 0

    if args.target == "kv-cache":
        print(bold("Inspector · KV Cache（每个请求的上下文长度增长与 block 数）"))
        result = _mk_sim(tracer=Tracer("quiet")).run(
            [SimRequest(r.request_id, r.prompt, r.max_new_tokens, r.arrival) for r in reqs])
        import math
        for r in result.requests:
            need = max(1, math.ceil(r.total_len / result.block_size))
            print(f"  {r.request_id}: prompt={r.prompt_len} + gen={len(r.generated)} "
                  f"= 上下文 {r.total_len} tokens → 需 {need} 个 block（block_size={result.block_size}）")
        print(dim("  说明：这是机制预览。真实 KV Cache 在每层缓存 K/V（Lesson 7/14 展开）。"))
        return 0
    return 1


# --------------------------------------------------------------------------- #
# 命令：benchmark N
# --------------------------------------------------------------------------- #
def cmd_benchmark(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py benchmark")
    p.add_argument("number", type=int)
    p.add_argument("--mode", default="quick")
    args = p.parse_args(argv)
    modules = {
        8: "experiments.lesson_08_prefill_decode",
        17: "experiments.lesson_17_text_engine",
        18: "experiments.lesson_18_final_challenge",
    }
    if args.number not in modules:
        print(yellow(f"benchmark 目前支持 Lesson {sorted(modules)}（引擎/性能相关）。"))
        return 1
    module = modules[args.number]
    announce_equivalent(f"python3 -m {module} --mode {args.mode}",
                        inputs=f"configs/lesson_{args.number:02d}_quick.json",
                        outputs=f"outputs/lesson_{args.number:02d}/", modifies_source=False)
    return _run([sys.executable, "-m", module, "--mode", args.mode])


# --------------------------------------------------------------------------- #
# 命令：mm-demo / mm-inspect / mm-benchmark
# --------------------------------------------------------------------------- #
def _run_module(module: str, extra: list[str], inputs: str = "", outputs: str = "") -> int:
    announce_equivalent("python3 -m " + module + (" " + " ".join(extra) if extra else ""),
                        inputs=inputs, outputs=outputs, modifies_source=False)
    return _run([sys.executable, "-m", module, *extra])


def cmd_mm_demo(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py mm-demo")
    p.add_argument("kind", choices=["image", "video"])
    p.add_argument("--mode", default="quick")
    args = p.parse_args(argv)
    module = {"image": "experiments.lesson_24_mm_prefill",
              "video": "experiments.lesson_26_video"}[args.kind]
    return _run_module(module, ["--mode", args.mode], outputs="outputs/")


def cmd_mm_inspect(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py mm-inspect")
    p.add_argument("target", choices=["placeholders"])
    args = p.parse_args(argv)
    return _run_module("experiments.lesson_23_placeholder_merge", [], outputs="outputs/lesson_23/")


def cmd_mm_benchmark(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="course.py mm-benchmark")
    p.add_argument("--mode", default="quick")
    args = p.parse_args(argv)
    return _run_module("experiments.lesson_29_mm_engine", ["--mode", args.mode],
                       outputs="outputs/lesson_29/")


# --------------------------------------------------------------------------- #
# 计划中的命令（透明占位：明确说明未实现，不伪装成功）
# --------------------------------------------------------------------------- #
def _planned_cmd(name: str, phase: str):
    def handler(argv: list[str]) -> int:
        print(yellow(f"`{name}` 计划在 {phase} 实现，当前 Phase 尚未提供。"))
        print(dim("本课程遵循「诚实标注」原则：不提供伪装成功的占位命令。"))
        print(dim(f"当前可用命令： {', '.join(sorted(k for k in COMMANDS if not k.startswith('mm-') and k not in ('benchmark',)))}"))
        return 1
    return handler


# --------------------------------------------------------------------------- #
# 无参数：终端版首页
# --------------------------------------------------------------------------- #
def cmd_home(argv: list[str]) -> int:
    prog = load_progress()
    cur = prog.get("current_lesson", 0)
    les = course_meta.get(cur)
    print(bold("Build Your Own Mini-vLLM") + dim(f"   v{__version__}"))
    print("从零构建支持文本、图片和视频的大模型推理引擎")
    print("=" * 60)
    print(f"  已就绪课程：{course_meta.READY_LESSONS}   "
          f"（共 {course_meta.TOTAL_LESSONS} 课，其余为路线图）")
    print(f"  当前主线：Lesson {les.number} — {les.title} [{les.status}]")
    done = sorted(prog.get("completed_checks", []))
    print(f"  已完成检查：{done if done else '（暂无）'}")
    print("-" * 60)
    print(bold("  唯一推荐的下一步："))
    if not prog.get("started"):
        print("    1) " + cyan("python3 course.py doctor --offline") + "   # 确认环境与离线资源")
        print("    2) " + cyan("python3 course.py serve") + "              # 打开网页 Lesson 0")
        print("    3) " + cyan("python3 course.py run 0") + "              # 运行第一个模拟器")
        print("    4) " + cyan("python3 course.py check 0") + "            # 跑通测试，获得第一次成功")
    else:
        target = les.number if les.status == "ready" else course_meta.READY_LESSONS[-1]
        print("    " + cyan(f"python3 course.py run {target}") + "   然后  "
              + cyan(f"python3 course.py check {target}"))
    print("=" * 60)
    print(dim("  全部命令： python3 course.py --help"))
    return 0


# --------------------------------------------------------------------------- #
# 命令注册表
# --------------------------------------------------------------------------- #
COMMANDS = {
    "serve": (cmd_serve, "启动本地静态课程网页"),
    "doctor": (cmd_doctor, "环境自检（--offline 附加离线检查）"),
    "verify-offline-bundle": (cmd_verify_offline, "校验离线资源是否齐全"),
    "where-am-i": (cmd_where_am_i, "显示当前课程/进度/Git 状态与下一步"),
    "lesson": (cmd_lesson, "显示某课的固定头部信息（目标/文件/命令/下一步）"),
    "run": (cmd_run, "运行某课的实验（先打印等价底层命令）"),
    "check": (cmd_check, "运行某课的测试并更新进度"),
    "hint": (cmd_hint, "查看某课的四级提示"),
    "inspect": (cmd_inspect, "查看内部状态：blocks | scheduler | kv-cache"),
    "benchmark": (cmd_benchmark, "性能报告（Lesson 8 / 17 / 18）"),
    "mm-demo": (cmd_mm_demo, "多模态演示：image | video"),
    "mm-inspect": (cmd_mm_inspect, "多模态检查：placeholders"),
    "mm-benchmark": (cmd_mm_benchmark, "多模态端到端性能报告"),
}


def print_help() -> None:
    print(bold("course.py — Build Your Own Mini-vLLM 统一入口"))
    print("用法： python3 course.py [--debug] <命令> [参数]")
    print()
    print("命令：")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:<22} {desc}")
    print()
    print("全局选项：")
    print("  --debug        出错时显示完整 traceback（默认只显示友好错误）")
    print()
    print("直接运行 `python3 course.py`（无命令）会显示终端版首页与推荐的下一步。")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    debug = False
    if "--debug" in argv:
        debug = True
        argv.remove("--debug")

    if not argv:
        return cmd_home([])

    cmd = argv[0]
    if cmd in ("-h", "--help", "help"):
        print_help()
        return 0

    if cmd not in COMMANDS:
        print(red(f"未知命令：{cmd}"), file=sys.stderr)
        print_help()
        return 2

    handler, _ = COMMANDS[cmd]
    try:
        return handler(argv[1:])
    except KeyboardInterrupt:
        print("\n" + dim("已取消。"))
        return 130
    except SystemExit as e:  # argparse 的退出
        return int(e.code) if e.code is not None else 0
    except Exception as e:
        if debug:
            raise
        print(red(f"[错误] {type(e).__name__}: {e}"), file=sys.stderr)
        print(dim("加 --debug 可查看完整 traceback，例如：python3 course.py --debug " + cmd),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
