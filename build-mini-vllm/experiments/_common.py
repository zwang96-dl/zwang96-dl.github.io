"""实验共用的小工具（减少样板）。所有实验仍是独立可运行的 `python3 -m ...` 模块。"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_vllm.config import RUN_MODES  # noqa: E402


def load_model():
    from mini_vllm.model.transformer import load_checkpoint
    return load_checkpoint(ROOT / "assets/checkpoints/tiny_text.json")


def make_parser(prog: str, default_config: str, default_out: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog)
    p.add_argument("--config", default=default_config)
    p.add_argument("--mode", default="quick", choices=list(RUN_MODES))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", default=default_out)
    return p


def read_config(rel: str) -> dict:
    cp = (ROOT / rel).resolve()
    if not cp.exists():
        raise SystemExit(f"[错误] 找不到配置：{cp}（请在 build-mini-vllm/ 目录下运行）")
    return json.loads(cp.read_text("utf-8"))


def write_result(out: str, result: dict) -> Path:
    od = (ROOT / out).resolve()
    od.mkdir(parents=True, exist_ok=True)
    fp = od / "result.json"
    fp.write_text(json.dumps(result, indent=2, ensure_ascii=False), "utf-8")
    return fp.relative_to(ROOT)
