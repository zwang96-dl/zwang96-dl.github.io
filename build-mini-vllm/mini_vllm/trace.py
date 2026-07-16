"""统一的 Trace / 可观测性基础设施。

课程要求每个核心实验都支持三个观察层级：

    普通（normal）   —— 精简的过程摘要
    详细（--verbose）—— 更多中间状态
    追踪（--trace）  —— 完整执行时序：请求、token IDs、shape、
                        scheduler 输入输出、KV 分配、block table、
                        cache append、timing 等。

设计目标：**让复杂度按正确顺序显现**。Tracer 是一个薄薄的、显式的
事件记录器——它不隐藏任何东西，只是把系统内部发生的事情按时间顺序
打印出来，并可选择性地收集成结构化事件列表供测试断言。

用法::

    tr = Tracer(level="trace")
    with tr.section("Prefill"):
        tr.event("ModelRunner Input", input_ids_shape=[1, 4])
    tr.kv("logits.shape", [1, 4, 259])

三个层级的语义：
    - ``event`` / ``kv`` 在 verbose 与 trace 下都会打印。
    - ``detail`` 只在 verbose / trace 下打印。
    - ``fine`` 只在 trace 下打印（最细粒度）。
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

# Trace 层级从低到高。数字越大，输出越详细。
_LEVELS = {"quiet": 0, "normal": 1, "verbose": 2, "trace": 3}


@dataclass
class TraceEvent:
    """一条结构化的追踪事件，便于测试断言与后续可视化。"""

    section: str
    label: str
    fields: dict[str, Any] = field(default_factory=dict)
    depth: int = 0


class Tracer:
    """薄封装的事件记录器：既打印给人看，也收集成结构给机器用。

    参数
    ----
    level:
        ``"quiet" | "normal" | "verbose" | "trace"``。也可用布尔快捷方式
        通过 :meth:`from_flags` 构造。
    stream:
        输出流，默认 ``sys.stdout``。
    collect:
        是否把事件收集进 :attr:`events`（测试时很有用）。
    """

    def __init__(
        self,
        level: str = "normal",
        stream: Any | None = None,
        collect: bool = True,
    ) -> None:
        if level not in _LEVELS:
            raise ValueError(
                f"未知的 trace level: {level!r}，"
                f"可选：{sorted(_LEVELS)}"
            )
        self.level_name = level
        self.level = _LEVELS[level]
        self.stream = stream if stream is not None else sys.stdout
        self.collect = collect
        self.events: list[TraceEvent] = []
        self._depth = 0
        self._section_stack: list[str] = []

    # ------------------------------------------------------------------ #
    # 构造快捷方式
    # ------------------------------------------------------------------ #
    @classmethod
    def from_flags(cls, verbose: bool = False, trace: bool = False, **kw: Any) -> "Tracer":
        """从 ``--verbose`` / ``--trace`` 命令行开关构造。"""
        level = "trace" if trace else "verbose" if verbose else "normal"
        return cls(level=level, **kw)

    # ------------------------------------------------------------------ #
    # 输出原语
    # ------------------------------------------------------------------ #
    def _emit(self, min_level: int, label: str, fields: dict[str, Any]) -> None:
        section = self._section_stack[-1] if self._section_stack else ""
        if self.collect:
            self.events.append(
                TraceEvent(section=section, label=label, fields=dict(fields), depth=self._depth)
            )
        if self.level < min_level:
            return
        indent = "  " * self._depth
        if fields:
            rendered = "  ".join(f"{k} = {self._fmt(v)}" for k, v in fields.items())
            line = f"{indent}{label}: {rendered}" if label else f"{indent}{rendered}"
        else:
            line = f"{indent}{label}"
        print(line, file=self.stream)

    @staticmethod
    def _fmt(value: Any) -> str:
        # 让 shape 之类的 list 打印得整齐：[1, 4] 而不是 [1, 4]（保持简单）。
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    # ------------------------------------------------------------------ #
    # 公共 API
    # ------------------------------------------------------------------ #
    def log(self, message: str) -> None:
        """普通层级的一行叙述。"""
        self._ensure_stack()
        self._emit(_LEVELS["normal"], message, {})

    def event(self, label: str, **fields: Any) -> None:
        """一条关键事件（normal 及以上都会打印）。"""
        self._ensure_stack()
        self._emit(_LEVELS["normal"], label, fields)

    def kv(self, key: str, value: Any) -> None:
        """打印一个键值对，例如 ``tr.kv("logits.shape", [1, 4, 259])``。"""
        self._ensure_stack()
        self._emit(_LEVELS["normal"], "", {key: value})

    def detail(self, label: str, **fields: Any) -> None:
        """详细层级（verbose 及以上）。"""
        self._ensure_stack()
        self._emit(_LEVELS["verbose"], label, fields)

    def fine(self, label: str, **fields: Any) -> None:
        """最细粒度（仅 trace）。"""
        self._ensure_stack()
        self._emit(_LEVELS["trace"], label, fields)

    # ------------------------------------------------------------------ #
    # 分段
    # ------------------------------------------------------------------ #
    def _ensure_stack(self) -> None:
        if not hasattr(self, "_section_stack"):
            self._section_stack = []

    @contextmanager
    def section(self, title: str) -> Iterator["Tracer"]:
        """进入一个命名段落，缩进其中的事件。"""
        self._ensure_stack()
        header_level = _LEVELS["normal"]
        if self.level >= header_level:
            indent = "  " * self._depth
            print(f"{indent}[{title}]", file=self.stream)
        self._section_stack.append(title)
        self._depth += 1
        try:
            yield self
        finally:
            self._depth -= 1
            self._section_stack.pop()

    # ------------------------------------------------------------------ #
    # 测试辅助
    # ------------------------------------------------------------------ #
    def labels(self) -> list[str]:
        """返回所有已收集事件的 label 列表（供测试断言时序）。"""
        return [e.label for e in self.events]

    def find(self, label: str) -> TraceEvent | None:
        """返回第一个匹配 label 的事件。"""
        for e in self.events:
            if e.label == label:
                return e
        return None
