"""MultiModalRunner —— 多模态数据流的编排者（Lesson 24 的 Build 目标）。

完整数据流：

    结构化消息
      → chat template（文本段 + 媒体占位）
      → 媒体预处理（image processor / video sampler）
      → vision encoder + projector  →  视觉 embedding（text_hidden 维）
      → input parser（插入占位 token，记录 PlaceholderRange）
      → 文本 embedding 查表 + 合并视觉 embedding
      → 多模态 prefill（model.forward(inputs_embeds=...)）
      → 文本 decode（vision encoder **不再重复运行**，复用 LLM KV Cache）

要点：vision encoder 只在 prefill 前运行一次；decode 阶段完全是文本自回归，走 KV Cache。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import VisionConfig
from ..model import matrix as M
from ..tokenizer import ByteTokenizer
from ..cache.kv_cache import KVCache
from ..sampling import Sampler, SamplingParams, argmax
from .chat_template import MultiModalChatTemplate
from .image_processor import TinyImageProcessor
from .vision_encoder import TinyVisionEncoder, MultimodalProjector
from .video_sampler import VideoFrameSampler
from .inputs import MultiModalInputParser
from .embedding_merge import merge_multimodal_embeddings
from .media import synth_image, load_image, load_video


@dataclass
class MMPrefillResult:
    input_ids: list
    placeholder_ranges: list
    visual_token_counts: list
    logits: M.Matrix
    merged_embeds: M.Matrix
    media_meta: list = field(default_factory=list)


class MultiModalRunner:
    def __init__(self, text_model, vision_cfg: VisionConfig | None = None,
                 tokenizer: ByteTokenizer | None = None,
                 processor_cache=None, encoder_cache=None) -> None:
        self.model = text_model
        self.vcfg = vision_cfg or VisionConfig()
        self.tok = tokenizer or ByteTokenizer()
        self.template = MultiModalChatTemplate()
        self.processor = TinyImageProcessor(image_size=self.vcfg.image_size)
        self.encoder = TinyVisionEncoder(self.vcfg)
        self.projector = MultimodalProjector(self.vcfg)
        self.sampler_frames = VideoFrameSampler()
        # 可选的三层缓存之二（ProcessorCache / EncoderOutputCache）；LLM KV Cache 另算。
        self.processor_cache = processor_cache
        self.encoder_cache = encoder_cache
        self._proc_cfg = {"image_size": self.vcfg.image_size,
                          "mean": list(self.processor.mean), "std": list(self.processor.std)}
        self.encoder_id = f"tinyvis-s{self.vcfg.seed}-h{self.vcfg.vision_hidden_size}-l{self.vcfg.vision_layers}"
        self.projector_id = f"proj-s{self.vcfg.seed}-t{self.vcfg.text_hidden_size}"
        self.encoder_runs = 0   # 实际运行 vision encoder 的次数（用于验证 decode 不重复编码）

    # ------------------------------------------------------------------ #
    def _load_pixels(self, ref) -> list:
        if isinstance(ref, list):
            return ref                              # 已是像素
        if isinstance(ref, dict) and "synth" in ref:
            s = ref["synth"]
            return synth_image(s.get("h", 16), s.get("w", 16), s.get("seed", 0))
        if isinstance(ref, dict) and "path" in ref:
            return load_image(ref["path"])
        raise ValueError(f"无法解析图片引用：{ref!r}")

    def _encode_image(self, pixels: list) -> M.Matrix:
        # 三层缓存之二：命中 EncoderOutputCache 就直接返回投影后的视觉 embedding，跳过编码。
        if self.encoder_cache is not None:
            ekey = self.encoder_cache.key(pixels, self._proc_cfg, self.encoder_id,
                                          self.projector_id)
            cached = self.encoder_cache.get(ekey)
            if cached is not None:
                return cached
        # ProcessorCache：命中就复用预处理后的像素。
        if self.processor_cache is not None:
            pkey = self.processor_cache.key(pixels, self._proc_cfg)
            proc = self.processor_cache.get(pkey)
            if proc is None:
                proc = self.processor.preprocess(pixels)
                self.processor_cache.put(pkey, proc)
        else:
            proc = self.processor.preprocess(pixels)
        self.encoder_runs += 1
        vis = self.encoder.encode(proc.pixels_norm_chw)     # (num_patches, vision_hidden)
        out = self.projector(vis)                           # (num_patches, text_hidden)
        if self.encoder_cache is not None:
            self.encoder_cache.put(ekey, out)
        return out

    def _encode_video(self, ref, strategy="uniform", num_frames=4):
        # ref: {"path": dir} 或 {"frames": [pixels,...], "fps": f}
        if isinstance(ref, dict) and "path" in ref:
            frames, manifest = load_video(ref["path"])
            src_fps = manifest.get("fps", 1.0)
        else:
            from .media import VideoFrame
            fps = ref.get("fps", 1.0)
            frames = [VideoFrame(pixels=p, frame_index=i, timestamp_seconds=i / fps)
                      for i, p in enumerate(ref["frames"])]
            src_fps = fps
        sampled = self.sampler_frames.sample(frames, strategy=ref.get("strategy", strategy),
                                             num_frames=ref.get("num_frames", num_frames),
                                             fps=ref.get("fps"), source_fps=src_fps)
        embeds = []
        for f in sampled:
            embeds.extend(self._encode_image(f.pixels))     # 逐帧编码并拼接（时间顺序）
        return embeds, self.sampler_frames.timeline(sampled)

    # ------------------------------------------------------------------ #
    def estimate(self, messages: list[dict]) -> dict:
        """在**不做编码**的前提下估算 visual token 数与媒体数（供调度预算 gating）。"""
        segments, _ = self.template.render(messages)
        per_patch = self.vcfg.num_patches
        visual = 0
        num_media = 0
        for kind, payload in segments:
            if kind != "media":
                continue
            num_media += 1
            if payload["modality"] == "image":
                visual += per_patch
            else:  # video
                ref = payload["ref"]
                if isinstance(ref, dict) and "frames" in ref:
                    nframes = min(ref.get("num_frames", 4), len(ref["frames"]))
                else:
                    nframes = ref.get("num_frames", 4) if isinstance(ref, dict) else 4
                visual += nframes * per_patch
        return {"visual_tokens": visual, "num_media": num_media}

    def build_inputs(self, messages: list[dict]):
        """返回 (input_ids, ranges, visual_embeds_list, media_meta)。"""
        segments, _ = self.template.render(messages)
        visual_embeds_list = []
        media_meta = []
        for kind, payload in segments:
            if kind != "media":
                continue
            if payload["modality"] == "image":
                emb = self._encode_image(self._load_pixels(payload["ref"]))
                visual_embeds_list.append(emb)
                media_meta.append({"modality": "image", "visual_tokens": len(emb)})
            else:  # video
                emb, timeline = self._encode_video(payload["ref"])
                visual_embeds_list.append(emb)
                media_meta.append({"modality": "video", "visual_tokens": len(emb),
                                   "timeline": timeline})
        lens = [len(e) for e in visual_embeds_list]
        parser = MultiModalInputParser(self.tok)
        input_ids, ranges = parser.parse(segments, lens)
        return input_ids, ranges, visual_embeds_list, media_meta

    def prefill(self, messages: list[dict], kv_cache=None) -> MMPrefillResult:
        input_ids, ranges, visual_embeds_list, media_meta = self.build_inputs(messages)
        text_embeds = [list(self.model.embed[tid]) for tid in input_ids]
        merged = merge_multimodal_embeddings(text_embeds, ranges, visual_embeds_list)
        positions = list(range(len(input_ids)))
        logits = self.model.forward(None, positions, kv_cache=kv_cache, inputs_embeds=merged)
        return MMPrefillResult(input_ids=input_ids, placeholder_ranges=ranges,
                               visual_token_counts=[len(e) for e in visual_embeds_list],
                               logits=logits, merged_embeds=merged, media_meta=media_meta)

    def generate(self, messages: list[dict], max_new_tokens: int = 8,
                 sampling: SamplingParams | None = None):
        """多模态 prefill + 文本 decode（decode 不再运行 vision encoder）。"""
        smp = Sampler(sampling or SamplingParams())
        cache = KVCache(self.model.cfg)
        pre = self.prefill(messages, kv_cache=cache)
        tok = smp(pre.logits[-1])
        generated = [tok]
        cur = len(pre.input_ids)
        for _ in range(max_new_tokens - 1):
            logits = self.model.forward([tok], [cur], kv_cache=cache)  # 纯文本 decode
            tok = smp(logits[-1])
            generated.append(tok)
            cur += 1
        return {"input_ids": pre.input_ids, "placeholder_ranges": pre.placeholder_ranges,
                "visual_token_counts": pre.visual_token_counts, "media_meta": pre.media_meta,
                "generated": generated, "text": self.tok.decode(generated)}
