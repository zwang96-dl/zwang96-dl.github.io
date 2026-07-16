"""Request 与状态机（Lesson 10 的 Build 目标）。

一个请求在推理引擎里是一个**状态机**：

    WAITING  ──调度器准入──▶  RUNNING  ──生成完成──▶  FINISHED
                               │
                               └──被中止──▶ ABORTED

为支持 **chunked prefill**（Lesson 15），Request 记录 ``num_computed_tokens``：
已经算进 KV 的 prompt token 数。它 < prompt 长度时还在 prefill，之后才进入 decode。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ..tokenizer import EOS_ID


class RequestStatus(enum.Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"
    ABORTED = "ABORTED"


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    arrival: int = 0
    status: RequestStatus = RequestStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0      # 已算进 KV 的 prompt token 数（chunked prefill）
    kv: object | None = None          # 该请求的 PagedKVCache
    first_token_iter: int | None = None
    finish_iter: int | None = None
    stop_on_eos: bool = True

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def prefill_done(self) -> bool:
        return self.num_computed_tokens >= self.num_prompt_tokens

    @property
    def total_len(self) -> int:
        return self.num_prompt_tokens + len(self.output_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def remaining_prompt(self) -> int:
        return self.num_prompt_tokens - self.num_computed_tokens

    @property
    def remaining_work(self) -> int:
        """粗略的「剩余工作量」：未算 prompt + 未生成 token（SJF 用）。"""
        return self.remaining_prompt + (self.max_new_tokens - self.num_generated)

    def is_finished(self) -> bool:
        if self.num_generated >= self.max_new_tokens:
            return True
        if self.stop_on_eos and self.output_token_ids and self.output_token_ids[-1] == EOS_ID:
            return True
        return False
