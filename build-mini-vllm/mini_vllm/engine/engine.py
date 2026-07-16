"""LLMEngine —— 把各部件组装成完整推理引擎（Lesson 17 的 Build 目标）。

engine loop（迭代级、continuous batching）：

    while 还有未完成请求:
        out = scheduler.schedule(...)            # 决定这一步跑谁、各多少 token
        for item in out.items:
            运行模型（prefill 块 或 decode 一个 token）→ 采样 → 更新请求
        finished = scheduler.remove_finished()   # 完成的移出
        for r in finished: r.kv.free()           # 归还 KV block（无泄漏）

组合：tokenizer + Request/状态机 + Scheduler + BlockAllocator + PagedKVCache +
TinyTextModel(model runner) + Sampler。每条请求各持一份 PagedKVCache，从共享的
BlockAllocator 借用物理块——这就是一个（教学规模的）真实 mini-vLLM。

可选 ``prefix_cache``：相同 prompt 前缀的请求共享物理块（Lesson 16）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import ModelConfig, EngineConfig
from ..tokenizer import ByteTokenizer
from ..cache.block_allocator import BlockAllocator
from ..cache.block_table import PagedKVCache
from ..sampling import Sampler, SamplingParams
from ..scheduler.request import Request as _Req
from ..scheduler.scheduler import Scheduler, SchedulerConfig


@dataclass
class IterationSnapshot:
    iteration: int
    scheduled: list[str]
    scheduled_tokens: int
    num_prefill: int
    num_decode: int
    blocks_used: int
    blocks_free: int
    running: int
    waiting: int


@dataclass
class EngineResult:
    requests: list
    snapshots: list[IterationSnapshot] = field(default_factory=list)
    num_iterations: int = 0
    wall_time_s: float = 0.0
    peak_blocks_used: int = 0

    def outputs(self, tokenizer: ByteTokenizer) -> dict:
        return {r.request_id: tokenizer.decode(r.output_token_ids) for r in self.requests}


class LLMEngine:
    def __init__(self, model, tokenizer: ByteTokenizer | None = None,
                 engine_config: EngineConfig | None = None,
                 scheduler_config: SchedulerConfig | None = None,
                 enable_prefix_cache: bool = False) -> None:
        self.model = model
        self.cfg: ModelConfig = model.cfg
        self.tok = tokenizer or ByteTokenizer()
        self.ecfg = engine_config or EngineConfig()
        self.allocator = BlockAllocator(self.ecfg.num_blocks, self.ecfg.block_size)
        # 所有序列共享一份 KV 行存储（按物理块寻址），使前缀块可被复用（Lesson 16）。
        self.kv_store: dict = {"k": {}, "v": {}}
        scfg = scheduler_config or SchedulerConfig(
            max_num_seqs=self.ecfg.max_num_seqs,
            max_num_batched_tokens=self.ecfg.max_num_batched_tokens,
            policy=self.ecfg.scheduler_policy,
        )
        self.scheduler = Scheduler(scfg)
        self.prefix_cache = None
        if enable_prefix_cache:
            from ..cache.prefix_cache import PrefixCache
            self.prefix_cache = PrefixCache(self.allocator, self.ecfg.block_size, self.kv_store)
        self._samplers: dict[str, Sampler] = {}

    # ------------------------------------------------------------------ #
    def add_request(self, request_id: str, prompt: str | list[int], max_new_tokens: int,
                    arrival: int = 0, sampling: SamplingParams | None = None,
                    stop_on_eos: bool = True) -> _Req:
        ids = prompt if isinstance(prompt, list) else self.tok.encode(prompt, add_bos=True)
        r = _Req(request_id=request_id, prompt_token_ids=list(ids),
                 max_new_tokens=max_new_tokens, arrival=arrival, stop_on_eos=stop_on_eos)
        # 立刻建 KV 并尝试前缀复用——必须在调度器计算「还需处理多少 prompt token」之前，
        # 否则 num_computed_tokens 与调度的 chunk 大小会错位。
        r.kv = PagedKVCache(self.cfg, self.allocator, store=self.kv_store)
        if self.prefix_cache is not None:
            self.prefix_cache.attach(r)
        self.scheduler.add(r)
        self._samplers[request_id] = Sampler(sampling or SamplingParams())
        return r

    # ------------------------------------------------------------------ #
    def _ensure_kv(self, r: _Req) -> None:
        if r.kv is None:  # 兜底（正常在 add_request 已创建）
            r.kv = PagedKVCache(self.cfg, self.allocator, store=self.kv_store)

    def _run_item(self, item, iteration: int) -> None:
        r = item.request
        self._ensure_kv(r)
        if item.is_prefill:
            start = r.num_computed_tokens
            n = item.num_tokens
            toks = r.prompt_token_ids[start:start + n]
            positions = list(range(start, start + n))
            logits = self.model.forward(toks, positions, r.kv)
            r.num_computed_tokens += n
            if r.prefill_done:
                tok = self._samplers[r.request_id](logits[-1])
                r.output_token_ids.append(tok)
                if r.first_token_iter is None:
                    r.first_token_iter = iteration
        else:
            last = r.output_token_ids[-1]
            pos = r.num_prompt_tokens + r.num_generated - 1
            logits = self.model.forward([last], [pos], r.kv)
            tok = self._samplers[r.request_id](logits[-1])
            r.output_token_ids.append(tok)

    def run(self, max_iterations: int = 100000, tracer=None) -> EngineResult:
        res = EngineResult(requests=[])
        t0 = time.perf_counter()
        it = 0
        seen: dict = {}
        while self.scheduler.has_unfinished:
            it += 1
            if it > max_iterations:
                raise RuntimeError("engine 迭代超限，疑似死循环")
            out = self.scheduler.schedule(it, self.allocator.num_free, self.ecfg.block_size)
            if not out.items:
                if self.scheduler.running:
                    # 有 RUNNING 却排不出工作（预算/块不足以推进）——真实系统会抢占换出。
                    raise RuntimeError(
                        "调度停滞：存在 RUNNING 请求却无法推进（token 预算或 KV 块不足）。"
                        "本教学场景应增大 max_num_batched_tokens / num_blocks，或开启抢占。")
                arrived = any(r.arrival <= it for r in self.scheduler.waiting)
                if self.scheduler.waiting and not arrived:
                    continue   # 新请求尚未到达，空转推进时间
                if self.scheduler.waiting and arrived:
                    stuck = min((r for r in self.scheduler.waiting if r.arrival <= it),
                                key=lambda r: r.request_id)
                    raise RuntimeError(
                        f"调度停滞：请求 {stuck.request_id!r} 的 prompt "
                        f"（{stuck.remaining_prompt} tokens）超过 token 预算 "
                        f"（max_num_batched_tokens={self.scheduler.cfg.max_num_batched_tokens}）"
                        "且未开启 chunked prefill——这正是 chunked prefill（Lesson 15）要解决的问题。")
                break   # 无 running、无 waiting → 结束

            num_prefill = sum(1 for i in out.items if i.is_prefill)
            num_decode = len(out.items) - num_prefill
            if tracer is not None:
                with tracer.section(f"iteration {it}"):
                    tracer.event("Scheduler", scheduled=[i.request.request_id for i in out.items],
                                 scheduled_tokens=out.scheduled_tokens,
                                 prefill=num_prefill, decode=num_decode,
                                 admitted=out.admitted)
            for item in out.items:
                self._run_item(item, it)
                if tracer is not None:
                    r = item.request
                    tracer.detail("ran", request=r.request_id,
                                  phase="prefill" if item.is_prefill else "decode",
                                  tokens=item.num_tokens, kv_len=r.kv.length,
                                  blocks=list(r.kv.block_table))

            finished = self.scheduler.remove_finished()
            for r in finished:
                r.finish_iter = it
                if r.kv is not None:
                    if self.prefix_cache is not None:
                        self.prefix_cache.on_finish(r)
                    else:
                        r.kv.free()
                seen[r.request_id] = r

            res.peak_blocks_used = max(res.peak_blocks_used, self.allocator.num_used)
            res.snapshots.append(IterationSnapshot(
                iteration=it, scheduled=[i.request.request_id for i in out.items],
                scheduled_tokens=out.scheduled_tokens, num_prefill=num_prefill,
                num_decode=num_decode, blocks_used=self.allocator.num_used,
                blocks_free=self.allocator.num_free,
                running=len(self.scheduler.running), waiting=len(self.scheduler.waiting)))

        res.requests = list(seen.values())
        res.requests.sort(key=lambda r: r.request_id)
        res.num_iterations = it
        res.wall_time_s = time.perf_counter() - t0
        # 收尾自检：无 KV 泄漏。开启前缀缓存时，唯一还占用的块应恰好是缓存持有的块，
        # 其余（请求自己的块）必须已释放。缓存本身在 shutdown() 时统一释放，以便跨多次
        # run() 复用共享前缀。
        if self.prefix_cache is None:
            self.allocator.check_no_leak()
        else:
            held = len(set(self.prefix_cache.map.values()))
            assert self.allocator.num_used == held, (
                f"疑似 KV 泄漏：已用块 {self.allocator.num_used} 与缓存持有块 {held} 不符")
        return res

    def shutdown(self) -> None:
        """释放前缀缓存持有的全部块，并做最终无泄漏自检。"""
        if self.prefix_cache is not None:
            self.prefix_cache.flush()
        self.allocator.check_no_leak()
