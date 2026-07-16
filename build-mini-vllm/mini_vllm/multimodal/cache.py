"""多模态三层缓存（Lesson 27 的 Build 目标）。

三层缓存针对**不同的重复**，绝不能混为一谈：

    ProcessorCache      —— 缓存「预处理结果」：resize/normalize 后的像素、抽帧、grid metadata。
                           key 主要看：媒体内容 + processor 配置 + 采样配置。
    EncoderOutputCache  —— 缓存「视觉编码 + projector 的输出」（投影后的视觉 embedding）。
                           key 还要加：encoder identity + projector identity + dtype + schema 版本。
    LLM KV Cache        —— 文本模型各层的 K/V（前面几课已实现）。多模态 prefill 后的上下文。

**为什么 key 要这么细**：如果换了 encoder/projector 或改了 dtype 却命中旧缓存，就会
悄悄用错的视觉 embedding（stale cache bug）。所以 key 必须覆盖所有会影响结果的因素。
"""

from __future__ import annotations

import hashlib
import json


def content_hash(pixels: list) -> str:
    """对图片像素做稳定哈希（媒体内容指纹）。"""
    h = hashlib.sha256()
    h.update(json.dumps(pixels, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()[:16]


def _key(*parts) -> str:
    return "|".join(str(p) for p in parts)


class _LRU:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.data: dict = {}
        self.order: list = []
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key):
        if key in self.data:
            self.hits += 1
            self.order.remove(key); self.order.append(key)
            return self.data[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key not in self.data and len(self.data) >= self.capacity:
            old = self.order.pop(0); del self.data[old]; self.evictions += 1
        self.data[key] = value
        if key in self.order:
            self.order.remove(key)
        self.order.append(key)

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "size": len(self.data), "evictions": self.evictions}


class ProcessorCache:
    """缓存预处理结果。key = 媒体内容 + processor 配置。"""

    def __init__(self, capacity: int = 64) -> None:
        self._c = _LRU(capacity)

    def key(self, pixels, processor_cfg: dict) -> str:
        return _key("proc", content_hash(pixels), json.dumps(processor_cfg, sort_keys=True))

    def get(self, k): return self._c.get(k)
    def put(self, k, v): self._c.put(k, v)
    def stats(self): return self._c.stats()


class EncoderOutputCache:
    """缓存视觉编码+projector 的输出。key 需覆盖 encoder/projector 身份、dtype、schema 版本。"""

    SCHEMA_VERSION = 1

    def __init__(self, capacity: int = 64) -> None:
        self._c = _LRU(capacity)

    def key(self, pixels, processor_cfg: dict, encoder_id: str, projector_id: str,
            dtype: str = "float") -> str:
        return _key("enc", content_hash(pixels), json.dumps(processor_cfg, sort_keys=True),
                    encoder_id, projector_id, dtype, self.SCHEMA_VERSION)

    def get(self, k): return self._c.get(k)
    def put(self, k, v): self._c.put(k, v)
    def stats(self): return self._c.stats()
