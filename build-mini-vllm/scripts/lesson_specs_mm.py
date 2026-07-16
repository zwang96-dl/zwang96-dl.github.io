"""多模态课程页面规格（Lesson 19–30），供 gen_lesson_pages 使用。

统一用通用的 mm_pipeline 动画展示每课的多模态数据流阶段。
"""

SPECS = {}


def _mm(n, aid, aq, stages, concepts, errors, challenge, vllm):
    js = 'MMPipeline.mount("#%s", {stages: %s});' % (aid, _stages_js(stages))
    return {"anim_scripts": ["mm_pipeline.js"], "anim_id": aid, "anim_q": aq,
            "mount_js": js, "concepts": concepts, "errors": errors,
            "challenge": challenge, "vllm": vllm}


def _stages_js(stages):
    parts = []
    for t, d in stages:
        parts.append('{title:%r, detail:%r}' % (t, d))
    return "[" + ",".join(parts).replace("'", '"') + "]"


SPECS[19] = _mm(19, "a19", "一个多模态请求从消息到 token+占位，经历哪些阶段？",
    [("结构化消息", "role + content[text|image|video]"),
     ("Chat Template", "加角色标记、媒体处放占位标记"),
     ("Tokenize 文本 + 插入占位 token", "PAD 作占位，记录 PlaceholderRange"),
     ("媒体单独处理", "processor / sampler（与 tokenizer 分离）"),
     ("视觉 embedding", "encoder + projector（稍后合并）")],
    [("阶段边界", "<p>图片路径/像素<strong>不是</strong> embedding；chat template 不编码像素；"
      "tokenizer 不处理像素；processor 与 template 是不同阶段。占位 token 稍后才被视觉 embedding 替换。</p>")],
    [("占位数≠媒体数", "解析错误", "每段媒体恰好一段占位，长度=该媒体 visual token 数"),
     ("把像素塞进 tokenizer", "阶段混淆", "媒体走 processor/encoder，文本走 tokenizer")],
    ["为什么要用占位 token 而不是直接把像素编码进文本？",
     "chat template、tokenizer、image processor 三者的职责边界是什么？"],
    [("结构化消息 + 占位", "HF chat template + mm processor", "阶段划分一致", "真实模板/占位更复杂，含特殊 token")])

SPECS[20] = _mm(20, "a20", "图片如何从像素变成模型能吃的 Tensor？",
    [("原图 (H,W,3) uint8", "0..255 RGB"),
     ("resize 到正方形", "最近邻到 image_size"),
     ("normalize", "(x/255 − mean) / std"),
     ("布局", "channel-last (H,W,3) / channel-first (3,H,W)")],
    [("布局", "<p>很多视觉模型用 channel-first (C,H,W)。resize/crop/pad 把不同尺寸统一到固定输入；"
      "normalize 把像素拉到稳定范围。dtype 从 uint8 变成浮点。</p>")],
    [("忘了 normalize", "数值范围不稳", "统一 (x/255−mean)/std"),
     ("布局搞反", "C 与 H/W 混淆", "明确 (3,S,S) vs (S,S,3)")],
    ["resize/crop/pad 各在什么场景下用？", "为什么要 normalize？不做会怎样？"],
    [("TinyImageProcessor", "HF ImageProcessor", "resize/normalize/layout 一致", "真实支持多分辨率、动态切分")])

SPECS[21] = _mm(21, "a21", "图片如何切成 patch 并变成 visual token？",
    [("(3,S,S) 图", "channel-first"),
     ("切成 grid×grid 个 patch", "每块 patch_size×patch_size"),
     ("每个 patch 展平", "维度 = 3·patch·patch"),
     ("线性投影", "→ vision_hidden（patch embedding）"),
     ("visual token 数", "= grid×grid")],
    [("手算", "<p>16×16 图、8×8 patch → grid 2×2 = 4 个 patch = 4 个 visual token。每个 patch 展平成 "
      "3×8×8=192 维，再线性投影到 vision_hidden。</p>")],
    [("token 数算错", "grid 算错", "grid = image_size / patch_size"),
     ("展平维度错", "漏了通道", "patch_dim = 3×patch×patch")],
    ["更大分辨率的图会产生更多还是更少 visual token？", "patch_size 变大/变小对 token 数与细节各有何影响？"],
    [("PatchEmbed", "ViT patch embedding", "切块+投影一致", "真实含位置编码、CLS token、多分辨率")])

SPECS[22] = _mm(22, "a22", "vision encoder 与 projector 各做什么？",
    [("patch embedding", "(num_patches, vision_hidden)"),
     ("vision encoder", "非因果自注意力 + 前馈（patch 互相可见）"),
     ("projector", "vision_hidden → text_hidden"),
     ("视觉 embedding", "已在文本 hidden 空间，可与文本合并")],
    [("关键区分", "<p>视觉编码器<strong>不加 causal mask</strong>（图片里 patch 可互相看）。"
      "视觉 embedding 是<strong>连续向量</strong>，不是 token id。projector 负责把维度对齐到 text_hidden。</p>")],
    [("维度不对齐", "projector 输出≠text_hidden", "projector 必须映射到 text_hidden"),
     ("给视觉加了 causal mask", "误用文本的因果掩码", "视觉自注意力是双向的")],
    ["为什么视觉注意力不需要 causal mask，而文本 decode 需要？",
     "projector 为什么必不可少？"],
    [("TinyVisionEncoder+Projector", "ViT + MM projector(MLP)", "编码+投影一致", "真实 encoder 更深、projector 更复杂")])

SPECS[23] = _mm(23, "a23", "视觉 embedding 如何精确替换占位、且不出对齐错误？",
    [("文本 embedding 查表", "占位 token 的 embedding 无所谓"),
     ("校验对齐", "数量/长度/越界/重叠/顺序"),
     ("逐位替换", "占位区间 ← 视觉 embedding"),
     ("合并结果", "(seq, hidden) 供 prefill")],
    [("为什么严格校验", "<p>占位符数量必须等于媒体数；每段长度必须等于该媒体的 visual token 数；"
      "区间不能越界或重叠；顺序要与媒体一致。任何不一致都会让视觉信息错位——多模态最常见的 bug。</p>")],
    [("数量不一致", "占位与媒体不匹配", "validate 会报错并定位"),
     ("长度不符", "visual token 数算错", "长度=该媒体 visual token 数"),
     ("维度不符", "projector 没对齐", "视觉 embedding 维度=text_hidden")],
    ["如果两张图的占位区间重叠会发生什么？为什么必须禁止？",
     "为什么占位 token 本身的 embedding 值无关紧要？"],
    [("merge_multimodal_embeddings", "vLLM embedding merge", "逐位替换一致", "真实按 PlaceholderRange 批量合并")])

SPECS[24] = _mm(24, "a24", "多模态 prefill 与文本 decode 如何衔接？encoder 跑几次？",
    [("媒体预处理", "processor / sampler"),
     ("vision encoder + projector", "视觉 embedding（仅 prefill 前运行一次）"),
     ("合并 embedding", "文本 + 视觉"),
     ("多模态 prefill", "forward(inputs_embeds=...) → 首 token + KV Cache"),
     ("文本 decode", "逐 token，复用 KV Cache，不再跑 encoder")],
    [("三层缓存不同", "<p>vision encoder 通常只在 prefill 前运行一次；decode 阶段是纯文本自回归，"
      "复用 LLM KV Cache。processor cache、encoder output cache、LLM KV cache 是三个不同层级。</p>")],
    [("decode 又跑 encoder", "误在每步重编码", "encoder 只在 prefill；decode 走 KV Cache"),
     ("prefill 没含视觉 token", "合并漏了", "视觉 token 占序列位置、进 KV")],
    ["为什么 decode 不需要重新运行 vision encoder？",
     "多模态 prefill 的序列长度由什么决定？"],
    [("MultiModalRunner", "vLLM mm model runner", "prefill 编码/decode 纯文本一致", "真实分布式、异步、批量编码")])

SPECS[25] = _mm(25, "a25", "为什么不同请求的 visual token 数不同？如何保持对齐？",
    [("解析各请求媒体", "0/1/2 图 或 视频"),
     ("每媒体算 visual token", "image=grid²；video=帧数×grid²"),
     ("动态序列长度", "视觉 token 数不同 → 序列/KV 不同"),
     ("媒体顺序与占位对齐", "media_index 与出现顺序一致")],
    [("动态 visual token", "<p>不同请求含不同数量/类型媒体，视觉 token 数动态变化；真实 VLM 里不同分辨率还会"
      "让单图 token 数不同。padding/packing 与顺序对齐都要小心。</p>")],
    [("多图顺序错", "media_index 乱", "按出现顺序记录 PlaceholderRange"),
     ("视觉 token 数估错", "影响预算", "image=grid²，video=帧数×grid²")],
    ["两张图的请求，KV 需求比一张图的大多少？", "文本-图-文本-图 交错时，如何保证顺序？"],
    [("动态视觉 token", "vLLM 动态 mm 输入", "按媒体动态计数一致", "真实随分辨率/宽高比变化")])

SPECS[26] = _mm(26, "a26", "视频如何抽帧、如何携带时间信息？",
    [("视频/帧序列", "含 fps"),
     ("抽帧", "uniform / fixed_fps / head / tail"),
     ("每帧 metadata", "frame_index + timestamp"),
     ("逐帧图像处理 + patch", "帧数 × grid² 个 visual token"),
     ("按时间顺序拼接", "→ 视频 embedding → prefill")],
    [("时间语义", "<p>抽帧在覆盖度与成本间权衡。每帧带 frame_index 与 presentation timestamp。"
      "<strong>timestamp 是 processor metadata——不显式喂进 prompt，LLM 不会自动理解它。</strong></p>")],
    [("帧序乱", "抽帧未保序", "抽帧后按时间顺序排列"),
     ("以为模型懂时间", "timestamp 未进 prompt", "需显式把时间信息写进文本")],
    ["uniform 与 fixed_fps 在长短视频上各有什么表现？",
     "为什么说 timestamp 不会被 LLM 自动理解？"],
    [("VideoFrameSampler", "vLLM 视频采样", "抽帧策略一致", "真实含关键帧、可变 fps、更多元数据")])

SPECS[27] = _mm(27, "a27", "三层缓存分别缓存什么？如何避免用错缓存？",
    [("ProcessorCache", "预处理像素/抽帧/grid metadata"),
     ("EncoderOutputCache", "视觉编码+projector 的输出"),
     ("LLM KV Cache", "文本各层 K/V"),
     ("Cache Key", "内容hash + 配置 + encoder/projector 身份 + dtype + schema 版本")],
    [("为什么 key 要细", "<p>三层缓存针对不同重复，绝不能混。若换了 encoder/projector 或改了 dtype 却命中旧缓存，"
      "就会用错的视觉 embedding（stale cache bug）。所以 key 必须覆盖所有影响结果的因素。</p>")],
    [("stale cache", "key 太粗", "key 含 encoder/projector 身份、dtype、schema 版本"),
     ("三层混用", "概念混淆", "processor / encoder / KV 各自独立")],
    ["同一张图在两个请求里出现，哪一层缓存能省最多？",
     "换了 projector 后，为什么必须让缓存失效？"],
    [("ProcessorCache/EncoderOutputCache", "vLLM mm 缓存", "分层缓存一致", "真实 key/eviction 更完善")])

SPECS[28] = _mm(28, "a28", "多模态调度如何把视觉工作量纳入预算？",
    [("请求含 text + visual token", "visual 也占序列位置、进 KV"),
     ("估算视觉工作量", "visual token 数 + 待编码媒体数"),
     ("预算 gating", "text/visual token 预算 + encoder 预算 + max_num_seqs"),
     ("准入并 prefill", "在预算内接纳；超出则等待")],
    [("视觉预算", "<p>多模态 prefill 不只有文本 token，还有大量 visual token，且视觉编码本身很贵。"
      "调度要同时约束 text/visual token 与 encoder 工作量。预算只改「何时算」，不改输出。</p>")],
    [("视觉 token 未入预算", "调度只看文本", "visual token 计入序列与预算"),
     ("一次编码太多", "encoder 过载", "encoder 预算限制每步编码媒体数")],
    ["为什么 visual token 也要占调度预算？", "encoder 工作量预算和 KV 预算有何不同？"],
    [("MultiModalBudget", "vLLM mm scheduler", "视觉工作量入调度一致", "真实含 encoder batch、显存估计")])

SPECS[29] = _mm(29, "a29", "完整多模态引擎如何把这一切串起来？",
    [("mixed batch", "text / 单图 / 多图 / 视频"),
     ("多模态 prefill", "编码（命中缓存则跳过）+ 合并 + forward"),
     ("continuous batching", "迭代级准入/推进/退出"),
     ("文本 decode", "复用 KV Cache，encoder 不重复运行"),
     ("报告", "encoder 运行数 / 缓存命中 / 视觉 token / TTFT")],
    [("端到端", "<p>把 processor/encoder/projector/merge/三层缓存/visual 预算/continuous batching 组装成一个"
      "多模态引擎。相同媒体命中 encoder 缓存；不跨请求串视觉 embedding；encoder 不在 decode 重复运行。</p>")],
    [("跨请求串用", "共享了状态", "每请求独立 KV；输出与单独生成对齐"),
     ("decode 重编码", "encoder 调用位置错", "encoder 只在 prefill")],
    ["mixed batch 里，text-only 请求和带视频的请求资源消耗差多少？",
     "如何验证引擎没有跨请求串用视觉 embedding？"],
    [("MultiModalEngine", "vLLM mm engine", "组件与 loop 一致", "真实分布式/异步/更强调度")])

SPECS[30] = _mm(30, "a30", "综合场景下多模态引擎能同时守住哪些正确性？",
    [("综合 workload", "text/多图/视频/共享媒体/有限预算"),
     ("对齐自检", "placeholder 与媒体严格对应"),
     ("无串用自检", "输出与单独生成一致"),
     ("缓存自检", "相同媒体命中、换身份不误命中"),
     ("时间/预算自检", "timestamp 保留、visual token 入预算")],
    [("综合验收", "<p>把多模态所有能力放进一个场景并自检：对齐、无跨请求串用、无 stale cache、"
      "timestamp 保留、visual token 入预算、encoder 不在 decode 重复运行、首次 vs 缓存命中的 TTFT 差异。</p>")],
    [("placeholder bug", "对齐错", "validate 守护 + 测试"),
     ("stale cache", "key 太粗", "key 含身份/版本"),
     ("timestamp 丢失", "metadata 未传递", "media_meta 携带 timeline")],
    ["首次请求与命中 encoder 缓存的请求，TTFT 差异来自哪里？",
     "如果视觉预算过小，会先影响什么？如何缓解？"],
    [("多模态综合自检", "vLLM 生产多模态", "正确性目标一致", "真实含 SLA、抢占、可观测性")])
