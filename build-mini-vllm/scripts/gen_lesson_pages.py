#!/usr/bin/env python3
"""从 course_meta + 每课页面规格生成 Lesson HTML 页面（离线、零远程依赖）。

用途：为机制/引擎类课程（9–18）与多模态课程（19–30）批量生成风格一致、且满足
课程契约（固定头部、动画控件、真实命令、四级提示、vLLM 映射）的页面。手写精修的
Lesson 0–8 不经过本生成器。

页面规格（PAGE_SPECS[n]）字段：
    anim_scripts : 需要引入的动画 JS（相对 ../js/anim/）
    anim_id      : 动画容器 id
    anim_q       : 「这个动画要回答的问题」
    mount_js     : 挂载动画的内联 JS
    intro        : 开头一段话
    concepts     : [(小标题, HTML)]  额外知识小节
    errors       : [(现象, 原因, 解决)]
    challenge    : [问题, ...]
    vllm         : [(教学实现, 真实vLLM, 相同点, 简化点)]
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mini_vllm import course_meta  # noqa: E402


def _dl(les):
    read = "、".join(f"<code>{x}</code>" for x in les.read) or "（无）"
    modify = "、".join(f"<code>{x}</code>" for x in les.modify) or "（本课重在运行与观察）"
    focus = "、".join(f"<code>{x}</code>" for x in les.focus) or "（无）"
    nxt = f"Lesson {les.next}" if les.next is not None else "（已是最后一课）"
    return f"""      <dl>
        <dt>当前目标</dt><dd>{les.goal}</dd>
        <dt>需要阅读的文件</dt><dd>{read}</dd>
        <dt>需要修改的文件</dt><dd>{modify}</dd>
        <dt>需要关注的类或函数</dt><dd>{focus}</dd>
        <dt>实验命令</dt><dd><code>python3 course.py run {les.number}</code></dd>
        <dt>测试命令</dt><dd><code>python3 course.py check {les.number}</code></dd>
        <dt>本课依赖</dt><dd>{('Lesson ' + str(les.prereqs[0])) if les.prereqs else '（无）'}</dd>
        <dt>完成后下一步</dt><dd>{nxt}</dd>
      </dl>"""


def _hints(les):
    out = []
    names = ["方向", "位置", "机制", "接近答案"]
    for i, h in enumerate(les.hints[:4]):
        out.append(f'    <details><summary>提示 Level {i+1} · {names[i]}</summary>{h}</details>')
    return "\n".join(out) if out else ""


ANIM_SCAFFOLD = """    <div class="anim" id="{aid}">
      <div class="anim-head"><div class="anim-q">这个动画要回答的问题：<small>{q}</small></div></div>
      <div class="anim-stage" style="min-height:120px"></div>
      <div class="anim-step-desc"><span class="step-no"></span><span class="step-text">按「▶ 播放」或「下一步」。</span>
        <div class="step-code" style="font-family:var(--mono);font-size:.82rem;color:var(--fg-dim);margin-top:4px;"></div></div>
      <div class="anim-controls">
        <button class="btn" data-anim-control="prev" type="button">⏮ 上一步</button>
        <button class="btn btn-accent" data-anim-control="play" type="button">▶ 播放</button>
        <button class="btn" data-anim-control="pause" type="button" hidden>⏸ 暂停</button>
        <button class="btn" data-anim-control="next" type="button">下一步 ⏭</button>
        <button class="btn" data-anim-control="reset" type="button">↺ 重置</button>
        <span class="spacer"></span>
        <label>速度 <input type="range" data-anim-control="speed" min="0.5" max="3" step="0.5" value="1"></label>
        <span class="rl-progress" style="font-family:var(--mono);font-size:.8rem;color:var(--fg-dim);"></span>
      </div>
    </div>"""


def render_page(n: int, spec: dict) -> str:
    les = course_meta.get(n)
    badge = "phase-build" if les.track == "text" else "phase-learn"
    concepts = "\n".join(
        f'    <h2>{t} <span class="phase-badge phase-learn">Learn</span></h2>\n    {html}'
        for t, html in spec.get("concepts", []))
    errors = ""
    if spec.get("errors"):
        rows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in spec["errors"])
        errors = ('    <h2 id="errors">常见错误 <span class="phase-badge phase-build">Build</span></h2>\n'
                  f'    <table><thead><tr><th>现象</th><th>原因</th><th>解决</th></tr></thead><tbody>{rows}</tbody></table>')
    ch = ""
    if spec.get("challenge"):
        items = "".join(f"<li>{q}</li>" for q in spec["challenge"])
        ch = ('    <h2 id="challenge">因果理解题 <span class="phase-badge phase-challenge">Challenge</span></h2>\n'
              f'    <div class="callout question"><ol style="margin:.3em 0">{items}</ol></div>')
    vrows = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in spec.get("vllm", []))
    vllm = ('    <h2 id="vllm">真实 vLLM 映射 <span class="phase-badge phase-deep">Deep Dive</span></h2>\n'
            f'    <table><thead><tr><th>教学实现</th><th>真实 vLLM</th><th>相同点</th><th>简化点</th></tr></thead>'
            f'<tbody>{vrows}</tbody></table>') if vrows else ""
    scripts = "\n".join(f'<script src="../js/anim/{s}"></script>' for s in spec["anim_scripts"])
    nxt_href = (f"lesson_{les.next:02d}.html" if les.next is not None
                and course_meta.get(les.next).status == "ready" else "../index.html")
    nxt_label = (f"下一步：Lesson {les.next} · {course_meta.get(les.next).title} →"
                 if les.next is not None and course_meta.get(les.next).status == "ready"
                 else "返回课程首页 →")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Lesson {n} · {les.title} — Build Your Own Mini-vLLM</title>
  <meta name="doc-description" content="{les.goal}">
  <link rel="stylesheet" href="../css/course.css">
</head>
<body data-total-lessons="31">
<a class="skip-link" href="#main">跳到主内容</a>
<div class="topbar">
  <span class="brand">Mini-vLLM <small>· Lesson {n}</small></span>
  <span class="spacer"></span>
  <a class="btn" href="../index.html">← 课程首页</a>
  <button class="btn" id="theme-toggle" type="button">主题：跟随系统</button>
</div>
<main class="content" id="main" style="max-width:900px;margin:0 auto">
  <h1>Lesson {n} · {les.title} <span class="phase-badge {badge}">{les.phase}</span></h1>
  <p>{spec.get('intro', les.why)}</p>

  <section id="goal"><div class="lesson-header">
{_dl(les)}
  </div></section>

  <h2 id="why">为什么需要这个知识 <span class="phase-badge phase-learn">Learn</span></h2>
  <p>{les.why}</p>

{concepts}

  <h2 id="anim">交互式动画 <span class="phase-badge phase-learn">Learn</span></h2>
{ANIM_SCAFFOLD.format(aid=spec['anim_id'], q=spec['anim_q'])}

  <h2 id="build">动手运行（Build） <span class="phase-badge phase-build">Build</span></h2>
  <pre class="cmd"><code>python3 course.py run {n}
python3 course.py run {n} --trace
python3 course.py check {n}</code></pre>

  <h2 id="hints">四级提示 <span class="phase-badge phase-build">Build</span></h2>
{_hints(les)}

{errors}

  <h2 id="trace">Trace 与 Inspector <span class="phase-badge phase-deep">Deep Dive</span></h2>
  <pre class="cmd"><code>python3 course.py run {n} --trace{spec.get('inspect_cmd','')}</code></pre>

{ch}

{vllm}

  <h2 id="summary">本节总结 & 下一步</h2>
  <p>{spec.get('summary', les.goal)}</p>
  <p style="margin-top:1em">
    <button class="btn btn-accent" type="button" data-mark-lesson="{n}">✅ 标记本课已学</button>
    &nbsp; <a class="btn" href="{nxt_href}">{nxt_label}</a></p>
  <div class="footer">Build Your Own Mini-vLLM · Lesson {n} · 完全离线 · 无第三方依赖</div>
</main>
<script src="../js/course.js"></script>
<script src="../js/anim/stepper_core.js"></script>
{scripts}
<script>
{spec['mount_js']}
</script>
</body>
</html>
"""


def generate(numbers, specs) -> None:
    out_dir = ROOT / "docs" / "lessons"
    for n in numbers:
        (out_dir / f"lesson_{n:02d}.html").write_text(render_page(n, specs[n]), "utf-8")
        print(f"  generated docs/lessons/lesson_{n:02d}.html")


if __name__ == "__main__":
    from scripts.lesson_specs import SPECS as S_TEXT  # noqa
    try:
        from scripts.lesson_specs_mm import SPECS as S_MM  # noqa
    except Exception:
        S_MM = {}
    SPECS = {**S_TEXT, **S_MM}
    nums = [int(x) for x in sys.argv[1:]] or sorted(SPECS)
    generate(nums, SPECS)
