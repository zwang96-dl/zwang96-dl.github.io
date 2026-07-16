"""Scheduler —— 迭代级调度器（Lesson 11 + 15 的 Build 目标）。

每次迭代，调度器决定「这一步跑哪些请求、各处理多少 token」，受两个预算约束：

    max_num_seqs           —— 同时 RUNNING 的请求数上限（并发度）
    max_num_batched_tokens —— 一次迭代处理的 token 总数上限（token 预算）

并支持 **chunked prefill**（Lesson 15）：长 prompt 不必一次算完，可切成若干块，
和其它请求的 decode 混排，避免长 prompt 长时间「堵住」短请求的首 token。

策略（Lesson 11）：
    fifo         —— 先到先服务
    decode-first —— 优先推进已在解码的请求（利于 TPOT / 吞吐）
    sjf          —— 最短作业优先（教学版，按剩余工作量）
    balanced     —— decode 优先但给等待中的 prefill 保留一部分预算（防饥饿）
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .request import Request, RequestStatus


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 64
    policy: str = "fifo"                 # fifo | decode-first | sjf | balanced
    enable_chunked_prefill: bool = True


@dataclass
class ScheduledItem:
    request: Request
    is_prefill: bool
    num_tokens: int                      # 本步为该请求处理的 token 数（decode=1）


@dataclass
class SchedulerOutput:
    iteration: int
    items: list[ScheduledItem] = field(default_factory=list)
    admitted: list[str] = field(default_factory=list)
    budget_used: int = 0

    @property
    def scheduled_tokens(self) -> int:
        return sum(i.num_tokens for i in self.items)


class Scheduler:
    def __init__(self, config: SchedulerConfig | None = None) -> None:
        self.cfg = config or SchedulerConfig()
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add(self, request: Request) -> None:
        request.status = RequestStatus.WAITING
        self.waiting.append(request)

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------------ #
    def _order_running(self, running: list[Request]) -> list[Request]:
        p = self.cfg.policy
        if p == "decode-first" or p == "balanced":
            return sorted(running, key=lambda r: (r.prefill_done is False, r.arrival, r.request_id))
        if p == "sjf":
            return sorted(running, key=lambda r: (r.remaining_work, r.arrival, r.request_id))
        return sorted(running, key=lambda r: (r.arrival, r.request_id))   # fifo

    def _order_waiting(self, cands: list[Request]) -> list[Request]:
        """准入顺序：sjf 先收最短作业，其余按到达先后。"""
        if self.cfg.policy == "sjf":
            return sorted(cands, key=lambda r: (r.remaining_work, r.arrival, r.request_id))
        return sorted(cands, key=lambda r: (r.arrival, r.request_id))

    # ------------------------------------------------------------------ #
    def schedule(self, iteration: int, free_blocks: int, block_size: int) -> SchedulerOutput:
        """产出本迭代的调度决策。``free_blocks`` 让调度器避免安排放不下的增长。"""
        cfg = self.cfg
        out = SchedulerOutput(iteration=iteration)
        budget = cfg.max_num_batched_tokens
        # 预留：balanced 给等待中的 prefill 至少留 1/4 预算，防止 decode 长期饿死新请求
        prefill_reserve = budget // 4 if cfg.policy == "balanced" else 0

        # 1) 推进已 RUNNING 的请求
        for r in self._order_running(self.running):
            if r.is_finished():
                continue
            if not r.prefill_done:
                remaining = r.remaining_prompt
                take = min(remaining, budget) if cfg.enable_chunked_prefill else remaining
                if take <= 0 or take > budget:
                    continue
                budget -= take
                out.items.append(ScheduledItem(r, is_prefill=True, num_tokens=take))
            else:
                usable = budget - prefill_reserve if prefill_reserve and self.waiting else budget
                if usable < 1:
                    continue
                budget -= 1
                out.items.append(ScheduledItem(r, is_prefill=False, num_tokens=1))

        # 2) 准入等待队列中的新请求（受 max_num_seqs 与 token 预算约束，顺序按策略）
        for r in self._order_waiting([w for w in self.waiting if w.arrival <= iteration]):
            if len(self.running) >= cfg.max_num_seqs:
                break
            need = r.remaining_prompt   # 已被前缀缓存命中的部分不再重复 prefill
            if need <= 0:
                # 整段 prompt 都命中前缀（极少见）：直接进入 running，下一轮走 decode
                self.waiting.remove(r)
                r.status = RequestStatus.RUNNING
                self.running.append(r)
                out.admitted.append(r.request_id)
                continue
            take = min(need, budget) if cfg.enable_chunked_prefill else need
            if take <= 0:
                continue
            if not cfg.enable_chunked_prefill and need > budget:
                continue   # 这个放不下，试更短的候选
            if free_blocks < 1:
                break
            self.waiting.remove(r)
            r.status = RequestStatus.RUNNING
            self.running.append(r)
            out.admitted.append(r.request_id)
            budget -= take
            out.items.append(ScheduledItem(r, is_prefill=True, num_tokens=take))

        out.budget_used = cfg.max_num_batched_tokens - budget
        return out

    def remove_finished(self) -> list[Request]:
        """把已完成的请求移出 running，返回它们（引擎据此回收 KV）。"""
        finished = [r for r in self.running if r.is_finished()]
        for r in finished:
            r.status = RequestStatus.FINISHED
        self.running = [r for r in self.running if not r.is_finished()]
        return finished
