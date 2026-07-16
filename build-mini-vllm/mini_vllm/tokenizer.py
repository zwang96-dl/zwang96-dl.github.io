"""真实的 byte-level tokenizer（Lesson 1 的实现，Phase 1 已提前落地）。

为什么用 byte-level？
---------------------
最简单、最透明、且零依赖的 tokenizer 就是把文本编码成 **UTF-8 字节**。
- 词表天然是固定的 256 个字节值（0..255）。
- 任何 Unicode 文本都能无损编码 / 解码（中文、emoji 都行）。
- 不需要训练、不需要下载任何文件、不需要第三方库。

在 256 个字节之上，我们再加三个 **特殊 token**：

    ==========  ====  ==========================================
    名称        ID    作用
    ==========  ====  ==========================================
    BOS         256   Begin Of Sequence，序列开始
    EOS         257   End Of Sequence，序列结束（生成停止符）
    PAD         258   Padding，把不同长度的序列补齐到同一长度
    ==========  ====  ==========================================

于是 ``vocab_size = 256 + 3 = 259``——正好对应课程推荐的文本配置。

真实 vLLM 用的是 BPE / SentencePiece 等子词 tokenizer；本课用 byte-level
是为了让「文本 → token IDs」这一步完全透明、可手算、可离线。数据流与 shape
是真实的，只是词表更简单。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

BYTE_VOCAB = 256
BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259

_SPECIAL_NAMES = {BOS_ID: "<bos>", EOS_ID: "<eos>", PAD_ID: "<pad>"}


@dataclass
class TokenizerConfig:
    """tokenizer 的可序列化配置（保存在 assets/tokenizer/ 下，离线可读）。"""

    kind: str = "byte-level"
    byte_vocab: int = BYTE_VOCAB
    bos_id: int = BOS_ID
    eos_id: int = EOS_ID
    pad_id: int = PAD_ID
    vocab_size: int = VOCAB_SIZE


class ByteTokenizer:
    """把字符串与 token ID 列表相互转换。

    核心方法：
        - :meth:`encode`  ``str -> list[int]``
        - :meth:`decode`  ``list[int] -> str``
        - :meth:`pad`     把一批序列补齐到同一长度
    """

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()
        self.bos_id = self.config.bos_id
        self.eos_id = self.config.eos_id
        self.pad_id = self.config.pad_id
        self.vocab_size = self.config.vocab_size

    # ------------------------------------------------------------------ #
    # 编码 / 解码
    # ------------------------------------------------------------------ #
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        """把文本编码成 token ID 列表。

        每个字符先按 UTF-8 编码成若干字节，每个字节（0..255）就是一个 token ID。
        可选地在前面加 BOS、在后面加 EOS。
        """
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(text.encode("utf-8"))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """把 token ID 列表解码回文本。

        特殊 token（BOS/EOS/PAD）在 ``skip_special=True`` 时被丢弃，其余
        字节收集起来按 UTF-8 解码。用 ``errors="replace"`` 保证永不抛异常
        （中途被截断的多字节字符会显示成 ``�`` 而不是报错）。
        """
        byte_vals: list[int] = []
        for tid in ids:
            if tid < 0 or tid >= self.vocab_size:
                raise ValueError(f"token id {tid} 超出词表范围 [0, {self.vocab_size})")
            if tid < BYTE_VOCAB:
                byte_vals.append(tid)
            elif not skip_special:
                # 把特殊 token 渲染成可读的占位字符串。
                # 注意：这会打断字节流，因此仅用于调试展示。
                pass
        return bytes(byte_vals).decode("utf-8", errors="replace")

    def id_to_piece(self, tid: int) -> str:
        """把单个 token ID 渲染成人类可读的片段（用于可视化 / Trace）。"""
        if tid in _SPECIAL_NAMES:
            return _SPECIAL_NAMES[tid]
        if 0 <= tid < BYTE_VOCAB:
            ch = bytes([tid])
            try:
                s = ch.decode("utf-8")
                if s.isprintable():
                    return s
            except UnicodeDecodeError:
                pass
            return f"<0x{tid:02X}>"
        raise ValueError(f"token id {tid} 超出词表范围")

    # ------------------------------------------------------------------ #
    # 批处理 / padding
    # ------------------------------------------------------------------ #
    def pad(
        self, batch: list[list[int]], length: int | None = None, side: str = "right"
    ) -> tuple[list[list[int]], list[list[int]]]:
        """把一批序列补齐到同一长度，返回 ``(input_ids, attention_mask)``。

        ``attention_mask`` 中 1 表示真实 token，0 表示 padding——下游 attention
        用它来忽略 padding 位置。这正是 Lesson 9 (static batching) 里「计算浪费」
        的来源。
        """
        if side not in ("left", "right"):
            raise ValueError("side 只能是 'left' 或 'right'")
        target = length if length is not None else max((len(x) for x in batch), default=0)
        out_ids: list[list[int]] = []
        out_mask: list[list[int]] = []
        for seq in batch:
            if len(seq) > target:
                raise ValueError(f"序列长度 {len(seq)} 超过目标 padding 长度 {target}")
            pad_n = target - len(seq)
            padding = [self.pad_id] * pad_n
            mask_pad = [0] * pad_n
            if side == "right":
                out_ids.append(seq + padding)
                out_mask.append([1] * len(seq) + mask_pad)
            else:
                out_ids.append(padding + seq)
                out_mask.append(mask_pad + [1] * len(seq))
        return out_ids, out_mask

    # ------------------------------------------------------------------ #
    # 持久化（离线保存 / 加载）
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self.config), indent=2, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteTokenizer":
        data = json.loads(Path(path).read_text("utf-8"))
        return cls(TokenizerConfig(**data))
