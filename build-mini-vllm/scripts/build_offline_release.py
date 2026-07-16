#!/usr/bin/env python3
"""构建离线发布包（offline release）。

生成：

    dist/build-mini-vllm-offline-macos-arm64/

内含：完整教材、Python 源码、测试、本地模型/tokenizer、图片、视频帧、workload、
wheel（若有）、安装说明、licenses、以及一份覆盖全树的 SHA-256 manifest。

用法::

    python3 scripts/build_offline_release.py [--out dist]

这个脚本只**读取**当前仓库、**写入** dist/；不修改源代码，不联网。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_vllm.offline_check import verify_bundle, format_report  # noqa: E402

RELEASE_NAME = "build-mini-vllm-offline-macos-arm64"

# 不打包进发布物的路径（生成物 / 版本控制 / 缓存 / 虚拟环境）。
EXCLUDE_DIRS = {".git", "__pycache__", "outputs", "dist", ".pytest_cache",
                ".venv", ".venv-offline-test", ".mypy_cache"}
EXCLUDE_SUFFIX = {".pyc", ".pyo"}


def _should_skip(rel: Path) -> bool:
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if rel.suffix in EXCLUDE_SUFFIX:
        return True
    if rel.name.startswith(".course_progress"):
        return True
    return False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="构建 mini-vLLM 离线发布包")
    p.add_argument("--out", default="dist", help="输出目录（默认 dist/）")
    args = p.parse_args(argv)

    # 先做离线自检；不通过则拒绝打包（避免发布残缺包）。
    report = verify_bundle(ROOT, check_wheels=True)
    print(format_report(report))
    if not report.ok:
        print("\n[拒绝] 离线自检未通过，已中止打包。请先修复上面的问题。", file=sys.stderr)
        return 1

    out_root = (ROOT / args.out).resolve()
    dest = out_root / RELEASE_NAME
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # 复制全树（跳过生成物 / VCS / 缓存）。
    copied = 0
    manifest_lines: list[str] = []
    for src in sorted(ROOT.rglob("*")):
        rel = src.relative_to(ROOT)
        if _should_skip(rel):
            continue
        # 不把 dist 自己复制进 dist。
        if rel.parts and rel.parts[0] == Path(args.out).name:
            continue
        target = dest / rel
        if src.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            copied += 1
            manifest_lines.append(f"{sha256(target)}  {rel.as_posix()}")

    # 写入覆盖全树的 SHA-256 manifest（发布包内 + 仓库内各一份）。
    manifest_text = "\n".join(manifest_lines) + "\n"
    (dest / "offline" / "manifests").mkdir(parents=True, exist_ok=True)
    (dest / "offline" / "manifests" / "sha256sums.txt").write_text(manifest_text, "utf-8")
    (ROOT / "offline" / "manifests" / "sha256sums.txt").write_text(manifest_text, "utf-8")

    # 复制 LICENSE（若存在于上级仓库根，可选）。
    print("\n" + "=" * 56)
    print(f"  离线发布包已生成：{dest.relative_to(ROOT)}")
    print(f"  文件数：{copied}")
    print(f"  SHA-256 manifest：offline/manifests/sha256sums.txt（{len(manifest_lines)} 条）")
    print("=" * 56)
    print("  验证发布包（在发布目录内运行）：")
    print(f"    cd {dest.relative_to(ROOT)} && python3 course.py verify-offline-bundle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
