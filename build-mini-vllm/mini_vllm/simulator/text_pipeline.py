"""LifecycleSimulator —— 一个请求在推理引擎里的完整生命周期（Lesson 0）。

这个模拟器把后面十几课的核心机制，用最简单的形式**一次性预演**给你看：

    请求到达 (WAITING)
        → 被调度器选中 (RUNNING)
        → prefill：一次吃掉整段 prompt，产出第一个 token（首 token）
        → decode：每次迭代只处理 1 个 token，直到写满 max_new_tokens
        → 结束 (FINISHED)，释放它占用的 KV block

它刻意展示了这些「将来会各自展开成一整课」的概念：

    - 请求状态机（Lesson 10）：WAITING / RUNNING / FINISHED / ABORTED
    - prefill vs decode（Lesson 8）：一次 N 个 token vs 每次 1 个 token
    - token 预算（Lesson 15）：一次迭代最多调度多少 token
    - continuous batching（Lesson 10/11）：请求随时加入 / 离开同一个 batch
    - KV block 分配（Lesson 13/14）：logical block → physical block，含释放

**没有神经网络**：下一个 token 由一个确定性的玩具函数产生（见 `_toy_next_token`），
所以每次运行结果完全一致，方便手算核对与测试。Lesson 5 会用真实模型替换它。

这里的 shape / 计数都是「真实形状、玩具数值」——例如 prefill 的
``input_ids.shape = [1, prompt_len]``、decode 的 ``[1, 1]``、
``logits.shape = [1, T, vocab_size]``——和真实 vLLM 的数据流一致。
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from ..tokenizer import ByteTokenizer
from ..trace import Tracer


class RequestState(enum.Enum):
    """请求状态机（预览 Lesson 10）。"""

    WAITING = "WAITING"    # 已到达，等待被调度
    RUNNING = "RUNNING"    # 正在被处理（prefill 或 decode）
    FINISHED = "FINISHED"  # 正常结束
    ABORTED = "ABORTED"    # 被中止（本课不触发，仅占位）


@dataclass
class SimRequest:
    """一个模拟请求。"""

    request_id: str
    prompt: str
    max_new_tokens: int
    arrival: int = 0                      # 第几个迭代到达（模拟动态到达）
    prompt_tokens: list[int] = field(default_factory=list)
    generated: list[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING
    phase: str | None = None              # "prefill" | "decode" | None
    blocks: list[int] = field(default_factory=list)  # 占用的 physical block id
    first_token_iter: int | None = None   # 产出首 token 的迭代号（TTFT 的代理）
    finish_iter: int | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_tokens)

    @property
    def total_len(self) -> int:
        """当前上下文长度 = prompt + 已生成。"""
        return len(self.prompt_tokens) + len(self.generated)

    @property
    def done(self) -> bool:
        return len(self.generated) >= self.max_new_tokens


@dataclass
class IterationRecord:
    """一次迭代的快照（供 Trace、测试、动画使用）。"""

    iteration: int
    scheduled: list[str]
    scheduled_tokens: int
    phases: dict[str, str]                # request_id -> "prefill"/"decode"
    states: dict[str, str]                # request_id -> RequestState.value
    blocks_in_use: int
    free_blocks: int
    new_tokens: dict[str, int | None]     # 本次迭代每个请求新产出的 token id
    sim_cost: int                         # 简化的计算量代理：scheduled_tokens


@dataclass
class SimResult:
    """整轮模拟的结果。"""

    requests: list[SimRequest]
    records: list[IterationRecord]
    total_iterations: int
    total_scheduled_tokens: int
    peak_blocks_in_use: int
    block_size: int
    num_blocks: int

    def outputs(self) -> dict[str, str]:
        """把每个请求生成的 token 解码成文本（玩具内容）。"""
        tok = ByteTokenizer()
        return {r.request_id: tok.decode(r.generated) for r in self.requests}

    def to_json(self) -> dict:
        """转成可序列化 dict（写入 outputs/，也可喂给可视化）。"""
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "total_iterations": self.total_iterations,
            "total_scheduled_tokens": self.total_scheduled_tokens,
            "peak_blocks_in_use": self.peak_blocks_in_use,
            "requests": [
                {
                    "request_id": r.request_id,
                    "prompt": r.prompt,
                    "prompt_len": r.prompt_len,
                    "max_new_tokens": r.max_new_tokens,
                    "generated_len": len(r.generated),
                    "state": r.state.value,
                    "first_token_iter": r.first_token_iter,
                    "finish_iter": r.finish_iter,
                }
                for r in self.requests
            ],
            "records": [
                {
                    "iteration": rec.iteration,
                    "scheduled": rec.scheduled,
                    "scheduled_tokens": rec.scheduled_tokens,
                    "phases": rec.phases,
                    "states": rec.states,
                    "blocks_in_use": rec.blocks_in_use,
                    "free_blocks": rec.free_blocks,
                    "sim_cost": rec.sim_cost,
                }
                for rec in self.records
            ],
        }


def _toy_next_token(context: list[int], step: int) -> int:
    """确定性的玩具「下一个 token」函数（FNV-1a 哈希 → 可打印 ASCII）。

    **它不是模型**。它的唯一目的：给定相同输入，永远产出相同输出，让整个
    生命周期可复现、可测试。Lesson 5 会用真实模型的 logits + 采样替换它。
    返回值落在可打印 ASCII 区间 [32, 126]，方便 decode 出可读字符。
    """
    h = 1469598103934665603  # FNV-1a 64-bit offset basis
    for t in context:
        h ^= (t + 1) & 0xFFFFFFFFFFFFFFFF
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    h ^= (step + 1)
    h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return 32 + (h % 95)  # 32..126，可打印


class _BlockPool:
    """极简 KV block 池（预览 Lesson 13 的 BlockAllocator）。

    维护一个空闲 physical block 的 free list；分配时从小到大取号，释放时归还。
    模拟器用它来演示「logical block → physical block」以及「结束后归还、无泄漏」。
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self.free: deque[int] = deque(range(num_blocks))
        self.in_use: int = 0

    def allocate(self) -> int:
        if not self.free:
            raise MemoryError("KV blocks 耗尽（模拟 OOM）——真实系统会触发抢占/换出")
        self.in_use += 1
        return self.free.popleft()

    def release(self, block_id: int) -> None:
        self.in_use -= 1
        self.free.append(block_id)


class LifecycleSimulator:
    """把一批请求跑完整个生命周期，产出逐迭代的快照。

    参数
    ----
    block_size:
        每个 KV block 容纳多少 token（Lesson 14）。
    num_blocks:
        物理 block 总数（Lesson 13）。
    max_num_seqs:
        一个 batch 里最多同时有几个 RUNNING 请求（Lesson 15）。
    token_budget:
        一次迭代最多调度多少 token，对应 vLLM 的 ``max_num_batched_tokens``。
    tracer:
        观测器；默认安静。传入 ``Tracer("trace")`` 可看到每步完整状态。
    """

    def __init__(
        self,
        block_size: int = 16,
        num_blocks: int = 32,
        max_num_seqs: int = 4,
        token_budget: int = 64,
        tracer: Tracer | None = None,
        next_token_fn: Callable[[list[int], int], int] = _toy_next_token,
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.max_num_seqs = max_num_seqs
        self.token_budget = token_budget
        self.tr = tracer or Tracer("quiet")
        self.next_token_fn = next_token_fn
        self.tokenizer = ByteTokenizer()

    # ------------------------------------------------------------------ #
    def _blocks_needed(self, length: int) -> int:
        return max(1, math.ceil(length / self.block_size))

    def _ensure_blocks(self, req: SimRequest, pool: _BlockPool) -> list[int]:
        """确保 req 有足够 block 容纳当前上下文；返回本次新分配的 block。"""
        need = self._blocks_needed(req.total_len)
        newly: list[int] = []
        while len(req.blocks) < need:
            phys = pool.allocate()
            req.blocks.append(phys)
            newly.append(phys)
        return newly

    # ------------------------------------------------------------------ #
    def run(self, requests: list[SimRequest]) -> SimResult:
        """执行迭代级调度循环，返回 :class:`SimResult`。"""
        # 先 tokenize 所有 prompt（预览 Lesson 1 的 encode）。
        for r in requests:
            if not r.prompt_tokens:
                r.prompt_tokens = self.tokenizer.encode(r.prompt, add_bos=True)

        pool = _BlockPool(self.num_blocks)
        pending = deque(sorted(requests, key=lambda r: (r.arrival, r.request_id)))
        running: list[SimRequest] = []
        records: list[IterationRecord] = []
        iteration = 0
        total_scheduled = 0
        peak_blocks = 0

        with self.tr.section("LifecycleSimulator"):
            self.tr.event("config",
                          block_size=self.block_size, num_blocks=self.num_blocks,
                          max_num_seqs=self.max_num_seqs, token_budget=self.token_budget)

            while pending or running:
                iteration += 1

                # 1) 准入：把已到达、还在等待的请求加入 running（受 max_num_seqs 限制）。
                admitted: list[SimRequest] = []
                while (pending and len(running) < self.max_num_seqs
                       and pending[0].arrival < iteration):
                    req = pending.popleft()
                    req.state = RequestState.RUNNING
                    req.phase = "prefill"
                    running.append(req)
                    admitted.append(req)

                if not running:
                    # 还没有请求到达（arrival 在未来），空转一步推进时间。
                    self.tr.fine("idle-iteration", iteration=iteration)
                    continue

                # 2) 调度：在 token 预算内，先安排 prefill（吃 prompt），再安排 decode（1 token）。
                budget = self.token_budget
                scheduled: list[SimRequest] = []
                sched_tokens = 0
                for req in running:
                    if req.phase == "prefill":
                        cost = req.prompt_len
                    else:  # decode
                        cost = 1
                    if cost > budget:
                        continue  # 预算不足，本迭代先跳过（长 prompt 阻塞 → Lesson 15 chunked prefill）
                    budget -= cost
                    sched_tokens += cost
                    scheduled.append(req)

                total_scheduled += sched_tokens

                with self.tr.section(f"iteration {iteration}"):
                    self.tr.event("Scheduler",
                                  scheduled=[r.request_id for r in scheduled],
                                  scheduled_tokens=sched_tokens,
                                  waiting=len(pending), running=len(running))

                    # 记录本步「实际执行的」phase（在下面 prefill 分支把 phase 翻成 decode 之前），
                    # 否则 IterationRecord 会把 prefill 误记成 decode。
                    phase_used = {r.request_id: r.phase for r in scheduled}
                    new_tokens: dict[str, int | None] = {}
                    for req in scheduled:
                        newly = self._ensure_blocks(req, pool)
                        if newly:
                            base = len(req.blocks) - len(newly)
                            for i, phys in enumerate(newly):
                                self.tr.detail("KV Allocation",
                                               request=req.request_id,
                                               logical_block=base + i,
                                               physical_block=phys)

                        if req.phase == "prefill":
                            self.tr.detail("ModelRunner Input",
                                           request=req.request_id,
                                           input_ids_shape=[1, req.prompt_len],
                                           positions_shape=[1, req.prompt_len],
                                           phase="PREFILL")
                            tok = self.next_token_fn(req.prompt_tokens, 0)
                            req.generated.append(tok)
                            req.first_token_iter = iteration
                            req.phase = "decode"
                            new_tokens[req.request_id] = tok
                            self.tr.detail("ModelRunner Output",
                                           request=req.request_id,
                                           logits_shape=[1, req.prompt_len, self.tokenizer.vocab_size],
                                           first_token=tok)
                        else:  # decode
                            step = len(req.generated)
                            ctx = req.prompt_tokens + req.generated
                            self.tr.detail("ModelRunner Input",
                                           request=req.request_id,
                                           input_ids_shape=[1, 1],
                                           positions_shape=[1, 1],
                                           cache_len=req.total_len - 1,
                                           phase="DECODE")
                            tok = self.next_token_fn(ctx, step)
                            req.generated.append(tok)
                            new_tokens[req.request_id] = tok
                            self.tr.detail("ModelRunner Output",
                                           request=req.request_id,
                                           logits_shape=[1, 1, self.tokenizer.vocab_size],
                                           next_token=tok)

                    # 3) 收尾：把写满的请求标记 FINISHED，并归还其 block。
                    still_running: list[SimRequest] = []
                    for req in running:
                        if req.done:
                            req.state = RequestState.FINISHED
                            req.finish_iter = iteration
                            for phys in req.blocks:
                                pool.release(phys)
                            self.tr.event("Finished",
                                          request=req.request_id,
                                          generated=len(req.generated),
                                          freed_blocks=len(req.blocks))
                            req.blocks = []
                        else:
                            still_running.append(req)
                    running = still_running

                    peak_blocks = max(peak_blocks, pool.in_use)
                    records.append(IterationRecord(
                        iteration=iteration,
                        scheduled=[r.request_id for r in scheduled],
                        scheduled_tokens=sched_tokens,
                        phases={r.request_id: (phase_used.get(r.request_id) or "done") for r in scheduled},
                        states={r.request_id: r.state.value for r in requests},
                        blocks_in_use=pool.in_use,
                        free_blocks=len(pool.free),
                        new_tokens=new_tokens,
                        sim_cost=sched_tokens,
                    ))

                # 安全阀：防止（不该发生的）死循环。
                if iteration > 10000:
                    raise RuntimeError("模拟迭代次数异常，疑似死循环")

        # 结束：断言没有 block 泄漏（教学要点——FINISHED 必须归还全部 block）。
        assert pool.in_use == 0, f"KV block 泄漏：仍有 {pool.in_use} 个 block 未释放"

        return SimResult(
            requests=requests,
            records=records,
            total_iterations=iteration,
            total_scheduled_tokens=total_scheduled,
            peak_blocks_in_use=peak_blocks,
            block_size=self.block_size,
            num_blocks=self.num_blocks,
        )
