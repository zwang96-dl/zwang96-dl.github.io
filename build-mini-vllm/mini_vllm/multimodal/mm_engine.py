"""MultiModalEngine —— 完整 Tiny Multimodal Engine（Lesson 29 的 Build 目标）。

在文本引擎的思想上，增加多模态 prefill 与三层缓存 + visual token 预算：

    - 每个请求各持一份 LLM KV Cache；多模态 prefill 时运行 vision encoder（受 encoder 缓存加速），
      合并视觉/文本 embedding，产出首 token；之后是**纯文本 decode**（不再运行 vision encoder）。
    - 迭代级 continuous batching：每步先推进 running 的 decode，再按预算准入并 prefill 新请求。
    - 预算（MultiModalBudget）：max_num_seqs、text/visual token、encoder 工作量。

自检要点（Lesson 30）：placeholder/media 严格对齐、不跨请求串视觉 embedding、
不误命中 stale cache、visual token 计入预算、timestamp metadata 不丢、vision encoder
不在 decode 重复运行。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import VisionConfig
from ..tokenizer import ByteTokenizer
from ..cache.kv_cache import KVCache
from ..sampling import Sampler, SamplingParams
from .runner import MultiModalRunner
from .cache import ProcessorCache, EncoderOutputCache
from .budget import MultiModalBudget


@dataclass
class MMRequest:
    request_id: str
    messages: list
    max_new_tokens: int
    arrival: int = 0
    status: str = "WAITING"
    output_token_ids: list = field(default_factory=list)
    kv: object | None = None
    prefill_done: bool = False
    input_ids: list | None = None
    ranges: list | None = None
    media_meta: list | None = None
    visual_tokens: int = 0
    cur: int = 0
    first_token_iter: int | None = None
    finish_iter: int | None = None

    def finished(self) -> bool:
        return len(self.output_token_ids) >= self.max_new_tokens


@dataclass
class MMSnapshot:
    iteration: int
    admitted: list
    decoded: list
    running: int
    waiting: int
    encoder_runs: int


@dataclass
class MMEngineResult:
    requests: list
    snapshots: list = field(default_factory=list)
    iterations: int = 0
    wall_time_s: float = 0.0

    def outputs(self, tok):
        return {r.request_id: tok.decode(r.output_token_ids) for r in self.requests}


class MultiModalEngine:
    def __init__(self, text_model, vision_cfg: VisionConfig | None = None,
                 tokenizer: ByteTokenizer | None = None,
                 budget: MultiModalBudget | None = None, enable_caches: bool = True) -> None:
        self.model = text_model
        self.tok = tokenizer or ByteTokenizer()
        self.budget = budget or MultiModalBudget()
        self.processor_cache = ProcessorCache() if enable_caches else None
        self.encoder_cache = EncoderOutputCache() if enable_caches else None
        self.runner = MultiModalRunner(text_model, vision_cfg, self.tok,
                                       self.processor_cache, self.encoder_cache)
        self.waiting: list[MMRequest] = []
        self.running: list[MMRequest] = []
        self._samplers: dict = {}

    def add_request(self, request_id, messages, max_new_tokens, arrival=0, sampling=None):
        r = MMRequest(request_id=request_id, messages=messages,
                      max_new_tokens=max_new_tokens, arrival=arrival)
        self.waiting.append(r)
        self._samplers[request_id] = Sampler(sampling or SamplingParams())
        return r

    def _prefill(self, r: MMRequest, iteration: int) -> None:
        r.kv = KVCache(self.model.cfg)
        pre = self.runner.prefill(r.messages, kv_cache=r.kv)
        r.input_ids = pre.input_ids
        r.ranges = pre.placeholder_ranges
        r.media_meta = pre.media_meta
        r.visual_tokens = sum(pre.visual_token_counts)
        tok = self._samplers[r.request_id](pre.logits[-1])
        r.output_token_ids.append(tok)
        r.cur = len(pre.input_ids)
        r.prefill_done = True
        r.first_token_iter = iteration

    def run(self, tracer=None) -> MMEngineResult:
        res = MMEngineResult(requests=[])
        t0 = time.perf_counter()
        it = 0
        done = {}
        while self.waiting or self.running:
            it += 1
            if it > 100000:
                raise RuntimeError("mm-engine 迭代超限")

            # 1) decode 阶段：推进已 prefill 的 running 请求（纯文本，不跑 encoder）
            used_text = 0
            decoded = []
            enc_before = self.runner.encoder_runs
            for r in self.running:
                if r.prefill_done and not r.finished():
                    if used_text + 1 > self.budget.text_token_budget:
                        continue
                    used_text += 1
                    logits = self.model.forward([r.output_token_ids[-1]], [r.cur], kv_cache=r.kv)
                    r.output_token_ids.append(self._samplers[r.request_id](logits[-1]))
                    r.cur += 1
                    decoded.append(r.request_id)
            assert self.runner.encoder_runs == enc_before, "decode 阶段不得运行 vision encoder！"

            # 2) 准入 + prefill 阶段：按预算接纳新请求
            used_visual = 0
            used_encoder = 0
            admitted = []
            for r in sorted([w for w in self.waiting if w.arrival <= it],
                            key=lambda x: (x.arrival, x.request_id)):
                if len(self.running) >= self.budget.max_num_seqs:
                    break
                est = self.runner.estimate(r.messages)
                if not self.budget.fits(0, est["visual_tokens"], est["num_media"],
                                        used_text, used_visual, used_encoder):
                    continue
                self.waiting.remove(r)
                r.status = "RUNNING"
                self.running.append(r)
                self._prefill(r, it)
                used_visual += est["visual_tokens"]
                used_encoder += est["num_media"]
                admitted.append(r.request_id)

            # 3) 回收完成的请求
            still = []
            for r in self.running:
                if r.prefill_done and r.finished():
                    r.status = "FINISHED"
                    r.finish_iter = it
                    done[r.request_id] = r
                else:
                    still.append(r)
            self.running = still

            if tracer is not None:
                tracer.event(f"iter {it}", admitted=admitted, decoded=decoded,
                             running=len(self.running), waiting=len(self.waiting))
            res.snapshots.append(MMSnapshot(it, admitted, decoded, len(self.running),
                                            len(self.waiting), self.runner.encoder_runs))
            # 防御：若一步既没准入也没 decode 且仍有等待（预算过小），推进时间避免死循环
            if not admitted and not decoded and self.running:
                # running 里都在等 decode 预算？不应发生（decode 预算>=1）；跳出以防万一
                raise RuntimeError("mm-engine 停滞：预算过小无法推进")

        res.requests = sorted(done.values(), key=lambda r: r.request_id)
        res.iterations = it
        res.wall_time_s = time.perf_counter() - t0
        return res

    def stats(self) -> dict:
        return {
            "encoder_runs": self.runner.encoder_runs,
            "processor_cache": self.processor_cache.stats() if self.processor_cache else None,
            "encoder_cache": self.encoder_cache.stats() if self.encoder_cache else None,
        }
