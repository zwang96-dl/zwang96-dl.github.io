#!/bin/bash
# start.command —— 双击即可启动课程网页（macOS）。
# 这个脚本只做**透明、简单**的事：进入项目目录、激活虚拟环境（如果存在）、启动网页。
# 它不会静默安装依赖、不下载资源、不修改 Python 环境、不覆盖代码、不做 git 恢复。

set -e
cd "$(dirname "$0")"

# 若存在虚拟环境则激活（可选，核心不装任何包也能运行）
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "启动 Build Your Own Mini-vLLM 课程网页（仅本机，无需网络）…"
echo "浏览器打开： http://127.0.0.1:8000/index.html"
echo "按 Ctrl+C 停止。"
exec python3 course.py serve
