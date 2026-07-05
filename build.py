#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成门户主页 index.html。

用法:
    python3 build.py

组织约定(重要):
    每个项目 = 仓库根目录下的一个文件夹,文件夹里放一个 index.html。
        my-project/index.html   ->  一张卡片,标题默认就是文件夹名 "my-project"
    访问时 URL 很干净: https://<用户名>.github.io/my-project/

    - 不含 index.html 的文件夹、以及以 "." 开头的文件夹(如 .github),会自动忽略。
    - 想精细控制某个项目在主页上的展示,可在它的 index.html <head> 里加可选 meta(见 README)。

你几乎不用改这个脚本,只需改下面的 SITE 配置,然后往仓库里建项目文件夹即可。
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

# ============================================================
#  站点配置 —— 改成你自己的信息
# ============================================================
SITE = {
    "title": "我的文档库",
    "subtitle": "个人技术笔记 · 总结 · 速查",
    "author": "zwang96-dl",
    "github": "https://github.com/zwang96-dl",   # 改成你的 GitHub 主页
    "footer": "用 GitHub Pages 托管 · 内容自动生成",
}

# 输出文件(门户主页)
OUTPUT = Path("index.html")

# 额外要忽略的文件夹(以 "." 开头的和不含 index.html 的已自动忽略)
IGNORE_DIRS = {"node_modules", "assets", "scripts", "templates"}


# ============================================================
#  下面一般不用改
# ============================================================

def read_text(path: Path) -> str:
    for enc in ("utf-8", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, OSError):
            continue
    return ""


def get_meta(html: str, name: str):
    """从 HTML 中提取 <meta name="NAME" content="..."> 的 content(顺序、引号不敏感)。"""
    for m in re.finditer(r"<meta\b[^>]*>", html, re.I):
        tag = m.group(0)
        if re.search(r'name\s*=\s*["\']%s["\']' % re.escape(name), tag, re.I):
            cm = re.search(r'content\s*=\s*["\'](.*?)["\']', tag, re.I | re.S)
            if cm:
                return unescape(cm.group(1).strip())
    return None


def get_title(html: str):
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    t = unescape(m.group(1).strip()) if m else None
    return t or None


def unescape(s: str) -> str:
    import html as _html
    return _html.unescape(s)


def git_last_date(path: Path):
    """用 git 最后一次提交日期作为兜底(在 GitHub Actions 里文件 mtime 不可靠)。"""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cs", "--", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        d = out.stdout.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            return d
    except Exception:
        pass
    return None


def collect_docs():
    """扫描仓库根目录下每个含 index.html 的文件夹,每个作为一张卡片。"""
    docs = []
    for sub in sorted(Path(".").iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith(".") or sub.name in IGNORE_DIRS:
            continue
        index = sub / "index.html"
        if not index.exists():
            continue

        html = read_text(index)

        # 标题: meta doc-title > <title> > 文件夹名
        title = get_meta(html, "doc-title") or get_title(html) or sub.name

        # 分类(可选): meta doc-category,不填则不分类
        category = get_meta(html, "doc-category") or ""

        # 简介: meta doc-description > meta description > 空
        desc = get_meta(html, "doc-description") or get_meta(html, "description") or ""

        # 日期: meta doc-date > git 提交日期 > 文件修改时间
        date = get_meta(html, "doc-date") or git_last_date(index)
        if not date:
            date = datetime.fromtimestamp(index.stat().st_mtime).strftime("%Y-%m-%d")

        # 标签(可选): meta doc-tags,逗号分隔
        tags_raw = get_meta(html, "doc-tags") or ""
        tags = [t.strip() for t in re.split(r"[,,]", tags_raw) if t.strip()]

        docs.append({
            "title": title,
            "category": category,
            "description": desc,
            "date": date,
            "tags": tags,
            "href": sub.name + "/index.html",   # 干净 URL: 访问 项目名/ 即可
        })

    docs.sort(key=lambda d: d["date"], reverse=True)
    return docs


def build():
    docs = collect_docs()
    docs_json = json.dumps(docs, ensure_ascii=False).replace("</", "<\\/")
    site_json = json.dumps(SITE, ensure_ascii=False).replace("</", "<\\/")

    out = PAGE.replace("__DOCS__", docs_json).replace("__SITE__", site_json)
    OUTPUT.write_text(out, encoding="utf-8")

    print(f"✓ 已生成 {OUTPUT} —— 收录 {len(docs)} 个项目")
    for d in docs:
        cat = f"[{d['category']}] " if d["category"] else ""
        print(f"    · {cat}{d['title']}  ({d['href']})")


# ============================================================
#  页面模板(HTML + CSS + JS 全内联,生成单文件)
# ============================================================
PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title></title>
<style>
  :root{
    --bg:#f7f8fa; --card:#ffffff; --text:#1a1a2e; --muted:#6b7280;
    --border:#e5e7eb; --accent:#4f46e5; --accent-soft:#eef2ff; --shadow:0 1px 3px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.04);
  }
  :root[data-theme="dark"]{
    --bg:#0f1117; --card:#181b23; --text:#e8eaed; --muted:#9aa0a8;
    --border:#272b35; --accent:#818cf8; --accent-soft:#1e2130; --shadow:0 1px 3px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.25);
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
    line-height:1.6; -webkit-font-smoothing:antialiased;
  }
  .wrap{max-width:1000px; margin:0 auto; padding:0 20px}
  header{padding:56px 0 28px}
  .top{display:flex; justify-content:space-between; align-items:flex-start; gap:16px}
  h1{margin:0; font-size:2rem; letter-spacing:-.02em}
  .subtitle{color:var(--muted); margin:6px 0 0; font-size:1.05rem}
  .meta-row{margin-top:14px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; color:var(--muted); font-size:.9rem}
  .meta-row a{color:var(--accent); text-decoration:none}
  .meta-row a:hover{text-decoration:underline}
  .theme-btn{
    background:var(--card); border:1px solid var(--border); color:var(--text);
    width:40px; height:40px; border-radius:10px; cursor:pointer; font-size:1.1rem;
    display:flex; align-items:center; justify-content:center; flex-shrink:0; transition:.15s;
  }
  .theme-btn:hover{border-color:var(--accent)}
  .controls{position:sticky; top:0; background:var(--bg); padding:14px 0; z-index:5; border-bottom:1px solid var(--border)}
  .search{
    width:100%; padding:11px 14px; border:1px solid var(--border); border-radius:10px;
    background:var(--card); color:var(--text); font-size:.95rem; outline:none; transition:.15s;
  }
  .search:focus{border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-soft)}
  .chips{display:flex; gap:8px; flex-wrap:wrap; margin-top:12px}
  .chip{
    padding:5px 13px; border:1px solid var(--border); border-radius:999px; background:var(--card);
    color:var(--muted); cursor:pointer; font-size:.85rem; transition:.15s; user-select:none;
  }
  .chip:hover{border-color:var(--accent)}
  .chip.active{background:var(--accent); color:#fff; border-color:var(--accent)}
  main{padding:24px 0 60px}
  .cat-title{font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:28px 0 12px; font-weight:600}
  .grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:14px}
  .card{
    display:block; background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:18px 18px 16px; text-decoration:none; color:inherit; transition:.18s; box-shadow:var(--shadow);
  }
  .card:hover{transform:translateY(-3px); border-color:var(--accent)}
  .card h3{margin:0 0 6px; font-size:1.05rem; line-height:1.35}
  .card .desc{color:var(--muted); font-size:.9rem; margin:0 0 12px; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden}
  .card .foot{display:flex; justify-content:space-between; align-items:center; gap:8px}
  .card .date{color:var(--muted); font-size:.8rem}
  .tags{display:flex; gap:6px; flex-wrap:wrap}
  .tag{background:var(--accent-soft); color:var(--accent); font-size:.72rem; padding:2px 8px; border-radius:6px}
  .empty{text-align:center; color:var(--muted); padding:60px 0}
  footer{text-align:center; color:var(--muted); font-size:.85rem; padding:30px 0 50px; border-top:1px solid var(--border)}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="top">
      <div>
        <h1 id="site-title"></h1>
        <p class="subtitle" id="site-subtitle"></p>
        <div class="meta-row">
          <span id="site-author"></span>
          <a id="site-github" href="#" target="_blank" rel="noopener">GitHub ↗</a>
          <span id="doc-count"></span>
        </div>
      </div>
      <button class="theme-btn" id="theme-btn" title="切换深色/浅色">🌙</button>
    </div>
  </header>

  <div class="controls">
    <input class="search" id="search" type="search" placeholder="🔍  搜索项目标题、简介或标签…" autocomplete="off">
    <div class="chips" id="chips"></div>
  </div>

  <main id="main"></main>

  <footer id="footer"></footer>
</div>

<script>
const DOCS = __DOCS__;
const SITE = __SITE__;

// ---- 头部信息 ----
document.title = SITE.title;
document.getElementById('site-title').textContent = SITE.title;
document.getElementById('site-subtitle').textContent = SITE.subtitle;
document.getElementById('site-author').textContent = SITE.author;
document.getElementById('site-github').href = SITE.github;
document.getElementById('doc-count').textContent = DOCS.length + ' 个项目';
document.getElementById('footer').innerHTML = '© ' + (SITE.author||'') + ' · ' + (SITE.footer||'');

// ---- 深色模式 ----
const themeBtn = document.getElementById('theme-btn');
function applyTheme(t){
  document.documentElement.setAttribute('data-theme', t);
  themeBtn.textContent = t === 'dark' ? '☀️' : '🌙';
}
let theme = localStorage.getItem('theme');
if(!theme) theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
applyTheme(theme);
themeBtn.onclick = () => { theme = theme === 'dark' ? 'light' : 'dark'; localStorage.setItem('theme', theme); applyTheme(theme); };

// ---- 是否启用分类(只要有任意一篇设了 doc-category 就启用)----
const hasCats = DOCS.some(d => d.category && d.category.trim());

// ---- 分类过滤 ----
let activeCat = 'all';
const chipsEl = document.getElementById('chips');
if(hasCats){
  const cats = ['all', ...Array.from(new Set(DOCS.map(d => d.category || '未分类')))];
  chipsEl.innerHTML = cats.map(c =>
    `<span class="chip${c==='all'?' active':''}" data-cat="${encodeURIComponent(c)}">${c==='all'?'全部':escapeHtml(c)}</span>`
  ).join('');
  chipsEl.querySelectorAll('.chip').forEach(el => {
    el.onclick = () => {
      activeCat = decodeURIComponent(el.dataset.cat);
      chipsEl.querySelectorAll('.chip').forEach(x => x.classList.remove('active'));
      el.classList.add('active');
      render();
    };
  });
}else{
  chipsEl.style.display = 'none';
}

// ---- 搜索 ----
const searchEl = document.getElementById('search');
searchEl.oninput = render;

// ---- 工具 ----
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function cardHtml(d){
  return `
    <a class="card" href="${encodeURI(d.href)}">
      <h3>${escapeHtml(d.title)}</h3>
      ${d.description ? `<p class="desc">${escapeHtml(d.description)}</p>` : '<p class="desc"></p>'}
      <div class="foot">
        <div class="tags">${(d.tags||[]).slice(0,3).map(t=>`<span class="tag">${escapeHtml(t)}</span>`).join('')}</div>
        <span class="date">${escapeHtml(d.date)}</span>
      </div>
    </a>`;
}

// ---- 渲染 ----
function render(){
  const q = searchEl.value.trim().toLowerCase();
  let list = DOCS.filter(d => {
    if(hasCats && activeCat !== 'all' && (d.category || '未分类') !== activeCat) return false;
    if(!q) return true;
    const hay = (d.title + ' ' + d.description + ' ' + (d.tags||[]).join(' ')).toLowerCase();
    return hay.includes(q);
  });

  const main = document.getElementById('main');
  if(!list.length){ main.innerHTML = '<div class="empty">没有匹配的项目 🤔</div>'; return; }

  if(!hasCats){
    // 无分类: 直接平铺
    main.innerHTML = `<div class="grid">${list.map(cardHtml).join('')}</div>`;
    return;
  }

  // 有分类: 按分类分组
  const groups = {};
  list.forEach(d => { const k = d.category || '未分类'; (groups[k] = groups[k] || []).push(d); });
  main.innerHTML = Object.keys(groups).map(cat =>
    `<div class="cat-title">${escapeHtml(cat)} · ${groups[cat].length}</div><div class="grid">${groups[cat].map(cardHtml).join('')}</div>`
  ).join('');
}

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
