"""自回归生成：naive（重算整段）与 cached（KV Cache）两条路径 —— Lesson 5/6/7/8。

- :func:`generate_naive`  —— 每步都把「当前整段序列」重新 forward。正确但浪费：
  第 t 步要处理 prompt_len + t 个 token，累计 O(n²)（Lesson 6 的「重复计算侦探」）。
- :func:`generate_cached` —— 先 prefill 一次，之后每步只 forward **1 个**新 token，
  历史的 K/V 从缓存取（Lesson 7）。两者在贪心解码下产出**完全相同**的序列。

两个函数都记录逐步的处理量与耗时，供 Lesson 6（重复计算）与 Lesson 8（TTFT/TPOT/ITL）使用。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..cache.kv_cache import KVCache
from ..config import ModelConfig
from ..sampling import Sampler, SamplingParams
from ..tokenizer import EOS_ID


@dataclass
class StepRecord:
    step: int
    phase: str            # "prefill" | "decode"
    input_len: int        # 本次 forward 实际喂进去的 token 数
    context_len: int      # 本步之后的上下文长度
    processed_tokens: int # 本步「处理」的 token 数（naive=整段，cached=1 或 prompt_len）
    token: int
    dt: float             # 本步耗时（秒）


@dataclass
class GenResult:
    method: str
    prompt_len: int
    generated: list[int]
    steps: list[StepRecord] = field(default_factory=list)
    total_processed_tokens: int = 0

    def text(self, tokenizer) -> str:
        return tokenizer.decode(self.generated)

    @property
    def ttft(self) -> float:
        """首 Token 延迟（Time To First Token）：第一步（prefill）的耗时。"""
        return self.steps[0].dt if self.steps else 0.0

    @property
    def decode_times(self) -> list[float]:
        return [s.dt for s in self.steps if s.phase == "decode"]

    @property
    def tpot(self) -> float:
        """每输出 Token 时间（Time Per Output Token）：decode 步的平均耗时。"""
        dts = self.decode_times
        return sum(dts) / len(dts) if dts else 0.0

    def summary(self) -> dict:
        return {
            "method": self.method,
            "prompt_len": self.prompt_len,
            "generated_len": len(self.generated),
            "total_processed_tokens": self.total_processed_tokens,
            "ttft_s": self.ttft,
            "tpot_s": self.tpot,
            "steps": [
                {"step": s.step, "phase": s.phase, "input_len": s.input_len,
                 "context_len": s.context_len, "processed_tokens": s.processed_tokens,
                 "dt_s": s.dt}
                for s in self.steps
            ],
        }


def _make_sampler(sampler) -> Sampler:
    if sampler is None:
        return Sampler(SamplingParams(temperature=0.0))  # 默认 greedy
    if isinstance(sampler, Sampler):
        return sampler
    if isinstance(sampler, SamplingParams):
        return Sampler(sampler)
    raise TypeError("sampler 应为 Sampler / SamplingParams / None")


def generate_naive(model, prompt_ids, max_new_tokens, sampler=None,
                   eos_id: int = EOS_ID, stop_on_eos: bool = True, tracer=None) -> GenResult:
    """朴素生成：每步重算整段序列（故意低效，用于对照）。"""
    smp = _make_sampler(sampler)
    ids = list(prompt_ids)
    res = GenResult(method="naive", prompt_len=len(prompt_ids), generated=[])
    for step in range(max_new_tokens):
        t0 = time.perf_counter()
        logits = model.forward(ids, list(range(len(ids))),
                               tracer=tracer if step == 0 else None)
        tok = smp(logits[-1])
        dt = time.perf_counter() - t0
        processed = len(ids)                 # naive：处理「当前整段」
        res.total_processed_tokens += processed
        res.steps.append(StepRecord(step, "prefill" if step == 0 else "decode",
                                    len(ids), len(ids) + 1, processed, tok, dt))
        res.generated.append(tok)
        ids.append(tok)
        if stop_on_eos and tok == eos_id:
            break
    return res


def generate_cached(model, prompt_ids, max_new_tokens, sampler=None,
                    eos_id: int = EOS_ID, stop_on_eos: bool = True, tracer=None) -> GenResult:
    """KV Cache 生成：prefill 一次，之后每步只处理 1 个新 token。"""
    smp = _make_sampler(sampler)
    cfg: ModelConfig = model.cfg
    cache = KVCache(cfg)
    res = GenResult(method="cached", prompt_len=len(prompt_ids), generated=[])

    # prefill：一次吃掉整段 prompt，产出首 token
    t0 = time.perf_counter()
    logits = model.forward(list(prompt_ids), list(range(len(prompt_ids))), cache, tracer)
    tok = smp(logits[-1])
    dt = time.perf_counter() - t0
    res.total_processed_tokens += len(prompt_ids)
    # context_len = 已缓存 token 数 + 1（刚生成、尚未入缓存的那个 token）
    res.steps.append(StepRecord(0, "prefill", len(prompt_ids), cache.length + 1,
                                len(prompt_ids), tok, dt))
    res.generated.append(tok)
    cur = len(prompt_ids)

    if not (stop_on_eos and tok == eos_id):
        for step in range(1, max_new_tokens):
            t0 = time.perf_counter()
            logits = model.forward([tok], [cur], cache)   # 只喂 1 个 token
            nt = smp(logits[-1])
            dt = time.perf_counter() - t0
            res.total_processed_tokens += 1               # cached：每步只处理 1 个
            res.steps.append(StepRecord(step, "decode", 1, cache.length + 1, 1, nt, dt))
            res.generated.append(nt)
            cur += 1
            tok = nt
            if stop_on_eos and tok == eos_id:
                break
    cache.check_consistent()
    return res


def processed_token_curves(prompt_len: int, n_new: int) -> dict:
    """解析地给出 naive 与 cached 的「累计处理 token 数」曲线（Lesson 6 可视化用）。"""
    naive, cached = [], []
    tot_n = tot_c = 0
    for t in range(n_new):
        tot_n += prompt_len + t          # naive 第 t 步处理整段
        tot_c += (prompt_len if t == 0 else 1)  # cached 只在 prefill 处理整段
        naive.append(tot_n)
        cached.append(tot_c)
    return {"naive": naive, "cached": cached,
            "naive_total": tot_n, "cached_total": tot_c}
