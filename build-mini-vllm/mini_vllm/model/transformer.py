"""TinyTextModel —— 极小 decoder-only Transformer（Lesson 3 的 Build 目标）。

纯 Python 实现，零第三方依赖，CPU 可跑、可离线、可复现。它把前面几课的零件组装
成一个能做前向、能生成的真实模型：

    token ids
      → Embedding 查表                (seq, hidden)
      → 每层 L：
          RMSNorm → Q/K/V 投影 → RoPE → (GQA) causal attention → 输出投影 → 残差
          RMSNorm → SwiGLU 前馈 → 残差
      → 最终 RMSNorm
      → LM head                        (seq, vocab)  = logits

**能力不是重点，数据流与 shape 是真实的。** 权重用固定种子确定性初始化，因此每次
运行结果完全一致（这就是我们的「本地 checkpoint」——只需存 config + seed 即可离线重建）。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config import ModelConfig
from . import matrix as M
from . import attention_ref as A
from .rmsnorm import rms_norm
from .rope import rope_freqs, apply_rope_heads
from .mlp import swiglu


class Rng:
    """确定性伪随机数发生器（splitmix64），用于可复现的权重初始化。

    不用 Python 的 random 是为了保证跨平台、跨版本**逐位一致**。
    """

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return z ^ (z >> 31)

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * (self.next_u64() / 2.0 ** 64)


def _rand_matrix(rng: Rng, rows: int, cols: int, scale: float) -> M.Matrix:
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


class TinyTextModel:
    """一个可前向、可生成的极小 decoder-only Transformer。"""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.cfg = config or ModelConfig()
        cfg = self.cfg
        rng = Rng(cfg.seed)
        H, V, L = cfg.hidden_size, cfg.vocab_size, cfg.num_layers
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim
        I = cfg.intermediate_size
        s = 0.02  # 初始化尺度（toy 模型，够用即可）

        self.embed = _rand_matrix(rng, V, H, s)              # (vocab, hidden)
        self.layers = []
        for _ in range(L):
            self.layers.append({
                "ln1": [1.0] * H,
                "wq": _rand_matrix(rng, H, nh * hd, s),      # (hidden, nh*hd)
                "wk": _rand_matrix(rng, H, nkv * hd, s),     # (hidden, nkv*hd)
                "wv": _rand_matrix(rng, H, nkv * hd, s),
                "wo": _rand_matrix(rng, nh * hd, H, s),      # (nh*hd, hidden)
                "ln2": [1.0] * H,
                "w_gate": _rand_matrix(rng, H, I, s),
                "w_up": _rand_matrix(rng, H, I, s),
                "w_down": _rand_matrix(rng, I, H, s),
            })
        self.final_ln = [1.0] * H
        self.lm_head = _rand_matrix(rng, H, V, s)            # (hidden, vocab)
        self.freqs = rope_freqs(hd, cfg.rope_theta)

    # ------------------------------------------------------------------ #
    def _attention(self, q: M.Matrix, k_all: M.Matrix, v_all: M.Matrix,
                   q_pos: list[int], k_pos: list[int]) -> M.Matrix:
        """多头 + GQA causal attention。q:(seq, nh*hd)，k/v:(Tk, nkv*hd)。"""
        cfg = self.cfg
        nh, nkv, hd, g = (cfg.num_attention_heads, cfg.num_kv_heads,
                          cfg.head_dim, cfg.group_size)
        seq = len(q)
        out = M.zeros(seq, nh * hd)
        for h in range(nh):
            kvh = h // g                                     # GQA：多个 query head 共享一个 kv head
            Qh = [row[h * hd:(h + 1) * hd] for row in q]
            Kh = [row[kvh * hd:(kvh + 1) * hd] for row in k_all]
            Vh = [row[kvh * hd:(kvh + 1) * hd] for row in v_all]
            oh = A.sdpa_positions(Qh, Kh, Vh, q_pos, k_pos)  # (seq, hd)
            for i in range(seq):
                base = h * hd
                oi = oh[i]
                for d in range(hd):
                    out[i][base + d] = oi[d]
        return out

    def forward(self, input_ids: list[int] | None, positions: list[int],
                kv_cache=None, tracer=None, inputs_embeds: M.Matrix | None = None) -> M.Matrix:
        """前向：返回 logits，shape=(seq, vocab)。

        若提供 ``kv_cache``：只为本次输入计算 Q/K/V，把新 K/V 追加进缓存，
        再对「新 Q × 全部缓存 K/V」做 attention（decode 路径）。

        若提供 ``inputs_embeds``（多模态用）：直接使用这批 embedding，跳过 token 查表——
        这样就能把「视觉 embedding」混进文本 embedding 一起做 prefill（Lesson 23/24）。
        """
        cfg = self.cfg
        nh, nkv, hd = cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim

        # 1) 得到 hidden：要么用传入的 embedding，要么按 token id 查表
        if inputs_embeds is not None:
            h = [list(row) for row in inputs_embeds]         # (seq, hidden)
            seq = len(h)
        else:
            seq = len(input_ids)
            h = [list(self.embed[tid]) for tid in input_ids]
        if tracer is not None:
            tracer.detail("embedding", seq_len=seq, hidden=cfg.hidden_size,
                          source=("inputs_embeds" if inputs_embeds is not None else "token_lookup"))

        if kv_cache is not None:
            kv_cache.add_positions(positions)

        for li, layer in enumerate(self.layers):
            hn = rms_norm(h, layer["ln1"], cfg.rms_norm_eps)
            q = M.matmul(hn, layer["wq"])                    # (seq, nh*hd)
            k = M.matmul(hn, layer["wk"])                    # (seq, nkv*hd)
            v = M.matmul(hn, layer["wv"])
            q = apply_rope_heads(q, positions, nh, hd, self.freqs)
            k = apply_rope_heads(k, positions, nkv, hd, self.freqs)

            if kv_cache is not None:
                kv_cache.append(li, k, v)
                k_all, v_all = kv_cache.get(li)
                k_pos = kv_cache.positions
            else:
                k_all, v_all, k_pos = k, v, positions

            attn = self._attention(q, k_all, v_all, positions, k_pos)
            o = M.matmul(attn, layer["wo"])                  # (seq, hidden)
            h = M.add(h, o)                                  # 残差

            hn2 = rms_norm(h, layer["ln2"], cfg.rms_norm_eps)
            mlp = swiglu(hn2, layer["w_gate"], layer["w_up"], layer["w_down"])
            h = M.add(h, mlp)                                # 残差

            if tracer is not None:
                tracer.fine(f"layer {li}", kv_len=(kv_cache.length if kv_cache else seq))

        hf = rms_norm(h, self.final_ln, cfg.rms_norm_eps)
        logits = M.matmul(hf, self.lm_head)                  # (seq, vocab)
        if tracer is not None:
            tracer.detail("logits", shape=[1, seq, cfg.vocab_size])
        return logits


# --------------------------------------------------------------------------- #
# checkpoint：只存 config + seed，加载时确定性重建（离线、可复现、体积极小）
# --------------------------------------------------------------------------- #
def save_checkpoint(model: TinyTextModel, path: str | Path) -> None:
    from dataclasses import asdict
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"config": asdict(model.cfg), "format": "config+seed"},
                            indent=2, ensure_ascii=False), "utf-8")


def load_checkpoint(path: str | Path) -> TinyTextModel:
    data = json.loads(Path(path).read_text("utf-8"))
    return TinyTextModel(ModelConfig(**data["config"]))
