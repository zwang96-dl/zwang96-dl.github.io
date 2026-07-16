"""课程元数据——``course.py`` 与 HTML 教材的单一数据源（single source of truth）。

每个 Lesson 的固定信息都集中在这里，避免命令行提示、网页头部、进度追踪
三处各写一份而互相不一致。``status`` 字段诚实地标注每课的实现状态：

    ready    —— 代码、实验、测试、HTML 均已完成，可运行、可测试
    planned  —— 已在课程大纲中，但尚未在当前 Phase 实现（course.py 会明确说明）

这样学习者永远知道「哪里是真的、哪里还没做」，符合本课程的透明原则。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Lesson:
    number: int
    slug: str
    title: str
    phase: str
    track: str  # "text" | "multimodal"
    status: str  # "ready" | "planned"
    goal: str = ""
    why: str = ""
    prereqs: list[int] = field(default_factory=list)
    read: list[str] = field(default_factory=list)
    modify: list[str] = field(default_factory=list)
    focus: list[str] = field(default_factory=list)
    experiment: str = ""  # 底层实验命令（run 会先打印它）
    test: str = ""        # unittest 模块路径（check 会运行它）
    config: str = ""
    workload: str = ""
    outputs: str = ""
    next: int | None = None
    hints: list[str] = field(default_factory=list)  # 四级提示

    @property
    def html(self) -> str:
        return f"docs/lessons/lesson_{self.number:02d}.html"


# --------------------------------------------------------------------------- #
# Lesson 0：完整实现（Phase 1 的纵向切片）
# --------------------------------------------------------------------------- #
_L0 = Lesson(
    number=0,
    slug="airplane-mode",
    title="Airplane Mode — 起飞前的准备",
    phase="Phase 1",
    track="text",
    status="ready",
    goal="在完全离线的环境下跑通课程的第一条主线：检查环境 → 启动网页 → "
    "运行第一个机制模拟器 → 看到真实命令与真实输出 → 获得第一次成功反馈。",
    why="在深入 Attention、KV Cache 之前，你需要一套可信任、可离线、可观察的"
    "工作环境。这一课不讲神经网络，而是让你熟悉「HTML 讲解 / 编辑器改代码 / "
    "终端跑实验」这套贯穿全课的三分工，并第一次亲眼看到一个请求在推理引擎里"
    "的完整生命周期。",
    prereqs=[],
    read=[
        "README.md",
        "mini_vllm/simulator/text_pipeline.py",
        "experiments/lesson_00_intro.py",
    ],
    modify=[],  # Lesson 0 不需要改代码，只需运行与观察
    focus=[
        "mini_vllm/simulator/text_pipeline.py：LifecycleSimulator.run()",
        "mini_vllm/simulator/text_pipeline.py：RequestState 状态机",
    ],
    experiment="python3 -m experiments.lesson_00_intro --config configs/lesson_00_quick.json",
    test="tests.lesson_00.test_intro",
    config="configs/lesson_00_quick.json",
    workload="assets/workloads/lesson_00.json",
    outputs="outputs/lesson_00/",
    next=1,
    hints=[
        # Level 1：方向
        "先跑 `python3 course.py doctor --offline` 确认环境，再跑 "
        "`python3 course.py run 0` 观察模拟器输出。不需要改任何代码。",
        # Level 2：具体位置
        "模拟器的主循环在 mini_vllm/simulator/text_pipeline.py 的 "
        "LifecycleSimulator.run()。它把每个请求从 WAITING 推进到 FINISHED，"
        "先做一次 prefill，再逐 token decode。",
        # Level 3：机制
        "注意观察三个数字：每步的 scheduled tokens（prefill 一次吃掉整段 "
        "prompt，decode 每步只处理 1 个 token）、请求状态、以及累计生成 token "
        "数。加 `--trace` 可以看到每一步的完整状态。",
        # Level 4：接近答案
        "本课只要求「跑通并读懂输出」。运行 "
        "`python3 -m unittest tests.lesson_00.test_intro` 应全部通过；"
        "若失败，多半是没在 build-mini-vllm/ 目录下运行，或用了错误的 Python。",
    ],
)

# --------------------------------------------------------------------------- #
# Lesson 1：元数据已就绪（Phase 2 实现 HTML/实验；tokenizer 代码已提前落地）
# --------------------------------------------------------------------------- #
_L1 = Lesson(
    number=1,
    slug="text-to-token",
    title="文本如何变成 Token",
    phase="Phase 2",
    track="text",
    status="ready",
    goal="理解 character / byte / token 的区别，读懂并使用真实的 byte-level "
    "tokenizer，掌握 encode/decode、BOS/EOS/PAD 与 padding（含 attention mask）。",
    why="token 是模型唯一能看懂的输入。搞不清「文本 → token IDs」，后面所有"
    "关于 shape、KV Cache、调度预算的讨论都会悬空。",
    prereqs=[0],
    read=["mini_vllm/tokenizer.py", "experiments/lesson_01_tokenizer.py"],
    modify=["mini_vllm/tokenizer.py"],
    focus=["ByteTokenizer.encode()", "ByteTokenizer.decode()", "ByteTokenizer.pad()"],
    experiment="python3 -m experiments.lesson_01_tokenizer --config configs/lesson_01_quick.json",
    test="tests.lesson_01.test_tokenizer",
    config="configs/lesson_01_quick.json",
    workload="assets/workloads/lesson_01.json",
    outputs="outputs/lesson_01/",
    next=2,
    hints=[
        "先 `python3 course.py run 1` 看 encode/decode/pad 的完整过程；再打开网页的 "
        "Tokenizer Explorer 输入你自己的文本观察。",
        "关注 mini_vllm/tokenizer.py 的三个方法：encode()（str→ids，可加 BOS/EOS）、"
        "decode()（ids→str，跳过特殊 token）、pad()（一批序列补齐 + 生成 attention mask）。",
        "byte-level 的关键：每个 UTF-8 字节就是一个 token id（0..255），中文一个字占 3 个字节即 "
        "3 个 token。词表 = 256 字节 + BOS(256)/EOS(257)/PAD(258) = 259。",
        "若 check 失败，多看失败信息里的「预期/实际」。常见坑：decode 把多字节字符从中间截断——"
        "本实现用 bytes(...).decode(errors='replace') 一次性解码整段字节，避免此问题。",
    ],
)

_L2 = Lesson(
    number=2,
    slug="tensor-shape",
    title="Tensor Shape 与矩阵乘法",
    phase="Phase 2",
    track="text",
    status="ready",
    goal="从标量到 Tensor，掌握 batch/sequence/hidden 维度、row-column 乘法规则、"
    "shape 相容性与必要的 broadcasting。",
    why="Transformer 的每一层都是矩阵乘法。看不懂 shape，就看不懂数据怎么流、"
    "算力和显存花在哪里。",
    prereqs=[1],
    read=["mini_vllm/model/matrix.py", "experiments/lesson_02_tensor.py"],
    modify=["mini_vllm/model/matrix.py"],
    focus=["matmul()", "transpose()", "add_row_bias()（行广播）"],
    experiment="python3 -m experiments.lesson_02_tensor --config configs/lesson_02_quick.json",
    test="tests.lesson_02.test_matrix",
    config="configs/lesson_02_quick.json",
    workload="assets/workloads/lesson_02.json",
    outputs="outputs/lesson_02/",
    next=3,
    hints=[
        "先 `python3 course.py run 2` 看一次带 shape 检查的矩阵乘法；网页的 Matrix "
        "Multiplication Explorer 可以点击输出格子看乘加过程。",
        "row-column rule：out[i][j] = Σ_k A[i][k]·B[k][j]，即「A 的第 i 行」点乘「B 的第 j 列」。"
        "形状要求 (m,k)@(k,n)->(m,n)。",
        "matmul 在 mini_vllm/model/matrix.py，是最朴素的三重循环。broadcasting 看 add_row_bias："
        "(m,n)+(n,)->(m,n)，给每一行加同一个 bias。",
        "shape 不相容时应抛出带具体数字的 ValueError（A 列数 != B 行数）——这类错误提前暴露最省时间。",
    ],
)

_L4 = Lesson(
    number=4,
    slug="attention-by-hand",
    title="Attention 手算",
    phase="Phase 2",
    track="text",
    status="ready",
    goal="用 2~3 个 token、单 head、小数字，手算并用代码复现 "
    "QKᵀ → scale → causal mask → softmax → weights×V 的每一步。",
    why="Attention 是 Transformer 的心脏，也是 KV Cache 的动机来源。把这五步"
    "彻底算清楚，后面 KV Cache、GQA、PagedAttention 都会顺理成章。",
    prereqs=[2],
    read=["mini_vllm/model/attention_ref.py", "experiments/lesson_04_attention.py"],
    modify=["mini_vllm/model/attention_ref.py"],
    focus=["scaled_dot_product_attention()", "softmax()", "causal_mask_apply()"],
    experiment="python3 -m experiments.lesson_04_attention --config configs/lesson_04_quick.json",
    test="tests.lesson_04.test_attention",
    config="configs/lesson_04_quick.json",
    workload="assets/workloads/lesson_04.json",
    outputs="outputs/lesson_04/",
    next=5,
    hints=[
        "先 `python3 course.py run 4 --trace` 看每个中间量；网页的 Attention Stepper 逐步"
        "高亮 scores→scale→mask→softmax→out。",
        "五步：scores=Q·Kᵀ（相似度）→ /√d（缩放）→ causal mask（未来置 -inf）→ softmax（逐行，和为1）"
        "→ ·V（加权求和）。shape：Q(Tq,d),K(Tk,d),V(Tk,dv)->out(Tq,dv)。",
        "causal mask：位置(i,j) 中 j>i 置 -inf；softmax 对 -inf 取 exp 得 0，等于「看不到未来」。"
        "第 0 个 query 只能看 key0，所以它的权重是 [1,0,...]。",
        "softmax 要数值稳定：先减去每行最大值再 exp。若 out 第 0 行 != V 第 0 行，多半是 mask "
        "或 softmax 写错——用 return_stages=True 打印中间量对照。",
    ],
)


_L3 = Lesson(
    number=3, slug="mini-transformer", title="最小 Decoder-only Transformer",
    phase="Phase 3", track="text", status="ready",
    goal="把 embedding、RMSNorm、Q/K/V、RoPE、attention、SwiGLU、残差、LM head "
    "组装成一个能前向的真实 tiny 模型，理解每一步的 shape。",
    why="这是「你自己的 LLM」的骨架。后面生成、KV Cache、调度都跑在它上面。",
    prereqs=[2], read=["mini_vllm/model/transformer.py", "mini_vllm/model/rmsnorm.py",
                       "mini_vllm/model/rope.py", "mini_vllm/model/mlp.py"],
    modify=["mini_vllm/model/transformer.py"],
    focus=["TinyTextModel.forward()", "rms_norm()", "apply_rope_heads()", "swiglu()"],
    experiment="python3 -m experiments.lesson_03_transformer --config configs/lesson_03_quick.json",
    test="tests.lesson_03.test_transformer", config="configs/lesson_03_quick.json",
    outputs="outputs/lesson_03/", next=4,
    hints=[
        "先 `python3 course.py run 3` 看一次前向的完整数据流与 shape；`inspect model` 看逐层。",
        "forward 顺序：embedding→(每层：RMSNorm→QKV→RoPE→attn→残差→RMSNorm→SwiGLU→残差)→RMSNorm→LM head。",
        "GQA：num_kv_heads<num_attention_heads，多个 query head 共享一个 kv head（省 KV 内存，Lesson 7 用）。",
        "logits.shape=(seq,vocab)。若 shape 不对，多半是某个线性层的权重维度或残差加法对不上。",
    ],
)
_L5 = Lesson(
    number=5, slug="first-token", title="生成第一个 Token",
    phase="Phase 3", track="text", status="ready",
    goal="从 logits 采样：greedy / temperature / top-k / top-p，写出自回归生成循环与停止符——不使用 model.generate()。",
    why="采样策略决定生成的确定性与多样性；生成循环是推理引擎的最内层。",
    prereqs=[3], read=["mini_vllm/sampling.py", "mini_vllm/engine/generate.py"],
    modify=["mini_vllm/sampling.py"],
    focus=["Sampler.__call__()", "top_k_filter()", "top_p_filter()", "generate_cached()"],
    experiment="python3 -m experiments.lesson_05_generation --config configs/lesson_05_quick.json",
    test="tests.lesson_05.test_sampling", config="configs/lesson_05_quick.json",
    outputs="outputs/lesson_05/", next=6,
    hints=[
        "先 `python3 course.py run 5` 看 greedy/temperature/top-k/top-p 四种输出的差别。",
        "greedy=argmax（temperature=0）；temperature 缩放 logits；top-k 只留最大 k 个；top-p 留累计概率达 p 的最小集合。",
        "采样要可复现：给定 seed，温度采样结果应一致（本实现用确定性 splitmix64）。",
        "自回归循环：每步取上一步 logits 的最后一行 → 采样 → 追加 → 直到 max_new_tokens 或遇到 EOS(257)。",
    ],
)
_L6 = Lesson(
    number=6, slug="recompute-detective", title="重复计算侦探",
    phase="Phase 3", track="text", status="ready",
    goal="运行朴素生成循环，逐步观察每个 decode step 的输入长度与处理量——亲眼看到 O(n²) 的浪费。",
    why="不先看到问题，就不会真正理解 KV Cache 为什么必要。",
    prereqs=[5], read=["mini_vllm/engine/generate.py"],
    modify=["mini_vllm/engine/generate.py"],
    focus=["generate_naive()", "processed_token_curves()"],
    experiment="python3 -m experiments.lesson_06_recompute --config configs/lesson_06_quick.json",
    test="tests.lesson_06.test_recompute", config="configs/lesson_06_quick.json",
    outputs="outputs/lesson_06/", next=7,
    hints=[
        "先 `python3 course.py run 6` 看 naive 每步处理量的条形逐步变长。",
        "naive 每步都 forward「当前整段」→ 第 t 步处理 prompt_len+t 个 token，累计 O(n²)。",
        "cached 只在 prefill 处理整段，之后每步处理 1 个 → 累计近似 O(n)。",
        "两者 greedy 输出必须一致——差别只在「算了多少」，不在「算出什么」。",
    ],
)
_L7 = Lesson(
    number=7, slug="kv-cache", title="KV Cache",
    phase="Phase 3", track="text", status="ready",
    goal="实现并验证 KV Cache：缓存每层历史 K/V，decode 只喂 1 个 token，"
    "cached 与 naive 逐值对齐；掌握 cache shape 与内存公式、MHA/MQA/GQA 的影响。",
    why="KV Cache 是 vLLM 一切内存管理（分页、前缀共享）的前提——用内存换计算。",
    prereqs=[6], read=["mini_vllm/cache/kv_cache.py", "mini_vllm/model/transformer.py"],
    modify=["mini_vllm/cache/kv_cache.py"],
    focus=["KVCache.append()", "KVCache.memory_estimate()", "sdpa_positions()"],
    experiment="python3 -m experiments.lesson_07_kv_cache --config configs/lesson_07_quick.json",
    test="tests.lesson_07.test_kv_cache", config="configs/lesson_07_quick.json",
    outputs="outputs/lesson_07/", next=8,
    hints=[
        "先 `python3 course.py run 7` 看正确性对齐（误差应为 0）与内存公式。",
        "缓存 K 和 V（被加权求和的对象），不缓存历史 Q（只用一次）。",
        "内存 = 2·num_kv_heads·head_dim·num_layers·seq_len。GQA 减小 num_kv_heads → 直接省 KV。",
        "对齐失败常因 positions/mask 或 append 时机错。用 sdpa_positions 的位置掩码：k_pos>q_pos 置 -inf。",
    ],
)
_L8 = Lesson(
    number=8, slug="prefill-decode", title="Prefill 与 Decode",
    phase="Phase 3", track="text", status="ready",
    goal="区分 prefill 与 decode，度量 TTFT/TPOT/ITL，理解 prompt-heavy 与 decode-heavy、"
    "compute-bound 与 memory-bound。",
    why="这些指标是评估与优化推理服务的语言；调度与批处理都围绕它们权衡。",
    prereqs=[7], read=["mini_vllm/engine/generate.py"],
    modify=["mini_vllm/engine/generate.py"],
    focus=["GenResult.ttft", "GenResult.tpot", "generate_cached()"],
    experiment="python3 -m experiments.lesson_08_prefill_decode --config configs/lesson_08_quick.json",
    test="tests.lesson_08.test_prefill_decode", config="configs/lesson_08_quick.json",
    outputs="outputs/lesson_08/", next=9,
    hints=[
        "先 `python3 course.py run 8` 对比 prompt-heavy 与 decode-heavy 的 TTFT/TPOT。",
        "TTFT=首 token 延迟(=prefill 耗时)；TPOT=每输出 token 平均耗时；ITL=相邻 token 间隔。",
        "prompt 越长 prefill 越重 → TTFT 越大（compute-bound）；decode 每步只 1 token → 更 memory-bound。",
        "prefill 处理量=prompt_len，decode 处理量=1/步；这决定了两阶段的性能画像。",
    ],
)


def _ready(n, slug, title, exp_mod, test_mod, goal, focus, read, modify, hints, prereq):
    return Lesson(
        number=n, slug=slug, title=title, phase="Phase 3", track="text", status="ready",
        goal=goal, prereqs=[prereq], read=read, modify=modify, focus=focus,
        experiment=f"python3 -m experiments.{exp_mod} --config configs/lesson_{n:02d}_quick.json",
        test=f"tests.lesson_{n:02d}.{test_mod}", config=f"configs/lesson_{n:02d}_quick.json",
        outputs=f"outputs/lesson_{n:02d}/", next=(n + 1), hints=hints)


_TEXT_ENGINE = [
    _ready(9, "static-batching", "Static Batching", "lesson_09_static_batching",
           "test_static_batching",
           "理解静态批处理：padding 到同一长度、一起跑到最长者结束，造成 padding/finished slot 的计算浪费。",
           ["padding 浪费计算", "attention mask"],
           ["experiments/lesson_09_static_batching.py", "mini_vllm/tokenizer.py"],
           ["mini_vllm/tokenizer.py（pad）"],
           ["`run 9` 量化浪费。", "静态成本=批大小×最长长度；连续成本=Σ真实长度。",
            "长度差异越大浪费越多。", "引出 continuous batching（Lesson 10）。"], 8),
    _ready(10, "continuous-batching", "Request 状态机与 Continuous Batching",
           "lesson_10_continuous_batching", "test_continuous_batching",
           "请求状态机 WAITING/RUNNING/FINISHED/ABORTED 与迭代级调度：请求可随时加入/离开运行 batch。",
           ["Request 状态机", "LLMEngine.run 的迭代循环"],
           ["mini_vllm/scheduler/request.py", "mini_vllm/engine/engine.py"],
           ["mini_vllm/scheduler/request.py"],
           ["`run 10` 看逐迭代快照。", "每次迭代动态准入/推进/退出请求。",
            "错峰到达的请求在别人 decode 时加入。", "引擎输出必须与逐请求参考一致。"], 8),
    _ready(11, "scheduler", "Scheduler", "lesson_11_scheduler", "test_scheduler",
           "实现并比较 FIFO / decode-first / SJF / balanced；理解 token 预算与 sequence 预算。",
           ["Scheduler.schedule()", "_order_running / _order_waiting"],
           ["mini_vllm/scheduler/scheduler.py"], ["mini_vllm/scheduler/scheduler.py"],
           ["`run 11` 比较四种策略。", "SJF 优先短作业 → 短请求首 token 更早。",
            "预算约束：max_num_seqs、max_num_batched_tokens。", "策略只改「何时算」，不改「算什么」。"], 10),
    _ready(12, "kv-waste", "连续 KV 分配为何浪费", "lesson_12_kv_waste", "test_kv_waste",
           "max-seq 预留导致 internal/external fragmentation；分页按需分配几乎消除浪费。",
           ["碎片计算"], ["experiments/lesson_12_kv_waste.py"], [],
           ["`run 12` 对比连续预留与分页。", "连续按 max_seq 预留 → 巨大浪费。",
            "分页向上取整到块 → 只剩块内零头。", "引出 Block Allocator（Lesson 13）。"], 11),
    _ready(13, "block-allocator", "Block Allocator", "lesson_13_block_allocator",
           "test_block_allocator",
           "物理块 free list、allocate/free、引用计数、OOM、double free、leak 检测。",
           ["BlockAllocator.allocate/free/incref"],
           ["mini_vllm/cache/block_allocator.py"], ["mini_vllm/cache/block_allocator.py"],
           ["`run 13` 看分配/释放/OOM/double-free。", "用 heap 分配编号最小的空闲块。",
            "引用计数支持共享（前缀缓存）。", "结束必须无泄漏（check_no_leak）。"], 12),
    _ready(14, "paged-kv", "Block Table 与 Paged KV Cache", "lesson_14_paged_kv", "test_paged_kv",
           "logical→physical 的 block table，逻辑连续物理分散；paged KV 与连续 KV 逐值一致。",
           ["PagedKVCache", "block table 映射"],
           ["mini_vllm/cache/block_table.py"], ["mini_vllm/cache/block_table.py"],
           ["`run 14` 看 block table 与对齐误差。", "pos→(logical=pos//bs, offset=pos%bs)。",
            "attention 前 gather 成逻辑顺序 → 结果与连续一致。", "误差必须为 0。"], 13),
    _ready(15, "chunked-prefill", "Token Budget 与 Chunked Prefill", "lesson_15_chunked_prefill",
           "test_chunked_prefill",
           "max_num_batched_tokens 预算；长 prompt 切块与 decode 混排，避免长时间阻塞短请求。",
           ["Scheduler 的 chunked prefill 分支"], ["mini_vllm/scheduler/scheduler.py"],
           ["mini_vllm/scheduler/scheduler.py"],
           ["`run 15` 对比开/关 chunked prefill。", "预算 < prompt 且未开 chunked → 停滞。",
            "切块让系统仍能推进。", "切块不改变输出。"], 14),
    _ready(16, "prefix-cache", "Prefix Caching", "lesson_16_prefix_cache", "test_prefix_cache",
           "相同前缀共享 KV 物理块：block hash、parent hash 链、引用计数、共享、逐出；命中不改变输出。",
           ["PrefixCache.attach/on_finish", "块链式哈希"],
           ["mini_vllm/cache/prefix_cache.py"], ["mini_vllm/cache/prefix_cache.py"],
           ["`run 16` 看命中率与正确性。", "块级链式哈希标识前缀；命中即复用物理块。",
            "引用计数让缓存块在请求结束后存活。", "命中必须不改变输出。"], 15),
    _ready(17, "text-engine", "完整 Text Mini-vLLM", "lesson_17_text_engine", "test_engine",
           "组装 tokenizer/请求/调度/分配/分页 KV/model runner/采样/引擎/benchmark，端到端运行并出报告。",
           ["LLMEngine", "engine loop"],
           ["mini_vllm/engine/engine.py", "benchmarks/report.py"], ["mini_vllm/engine/engine.py"],
           ["`run 17` 端到端 + 报告。", "engine loop：调度→运行→采样→回收。",
            "输出与逐请求参考一致、无泄漏。", "报告含 TTFT/吞吐/KV 利用率/前缀命中。"], 16),
    _ready(18, "text-final", "Text Final Incident Challenge", "lesson_18_final_challenge",
           "test_final",
           "综合场景：短聊天+长 prompt+不同输出+共享前缀+有限 KV 块+动态到达；正确、无泄漏、无 starvation、出报告并对比 naive。",
           ["整套引擎"], ["experiments/lesson_18_final_challenge.py"], [],
           ["`run 18` 跑综合场景。", "组合 continuous batching + 分页 + 前缀 + chunked。",
            "自检：正确/无泄漏/全部完成。", "与 naive 基线对比处理量。"], 17),
]


def _ready_mm(n, slug, title, phase, exp_mod, test_mod, goal, focus, read, modify, hints, prereq):
    return Lesson(
        number=n, slug=slug, title=title, phase=phase, track="multimodal", status="ready",
        goal=goal, prereqs=[prereq], read=read, modify=modify, focus=focus,
        experiment=f"python3 -m experiments.{exp_mod} --config configs/lesson_{n}_quick.json",
        test=f"tests.lesson_{n}.{test_mod}", config=f"configs/lesson_{n}_quick.json",
        outputs=f"outputs/lesson_{n}/", next=(n + 1 if n < 30 else None), hints=hints)


_MM = [
    _ready_mm(19, "mm-request", "多模态请求是什么", "Phase 4",
              "lesson_19_mm_request", "test_mm_request",
              "区分结构化消息 / chat template / token+占位 / processed media / 视觉 embedding 各阶段；"
              "明确图片路径不是 embedding、tokenizer 不处理像素。",
              ["MultiModalInputParser.parse()", "MultiModalChatTemplate.render()"],
              ["mini_vllm/multimodal/messages.py", "mini_vllm/multimodal/chat_template.py",
               "mini_vllm/multimodal/inputs.py"], ["mini_vllm/multimodal/inputs.py"],
              ["`run 19` 看各阶段。", "template 只放占位标记，不编码像素。",
               "占位 token 稍后被视觉 embedding 替换。", "placeholder 数必须与媒体数一致。"], 18),
    _ready_mm(20, "image-tensor", "图片如何变成 Tensor", "Phase 4",
              "lesson_20_image_tensor", "test_image_tensor",
              "RGB、resize/crop/pad、normalize、channel-last vs channel-first、dtype。",
              ["TinyImageProcessor.preprocess()"],
              ["mini_vllm/multimodal/image_processor.py"], ["mini_vllm/multimodal/image_processor.py"],
              ["`run 20` 看 resize/normalize/layout。", "像素用嵌套 list 表示，无需 Pillow。",
               "channel-first=(3,S,S)，channel-last=(S,S,3)。", "normalize：(x/255−mean)/std。"], 19),
    _ready_mm(21, "image-patch", "图片如何变成 Patch Token", "Phase 4",
              "lesson_21_image_patch", "test_patch",
              "patch size、grid、flatten、线性投影、visual token 数（=grid×grid）。",
              ["PatchEmbed.flatten_patches()", "PatchEmbed.__call__()"],
              ["mini_vllm/multimodal/patch_embed.py"], ["mini_vllm/multimodal/patch_embed.py"],
              ["`run 21` 看 patch 数与维度。", "visual token 数 = grid×grid。",
               "patch_dim = 3×patch×patch。", "线性投影到 vision_hidden。"], 20),
    _ready_mm(22, "vision-encoder", "Tiny Vision Encoder 与 Projector", "Phase 4",
              "lesson_22_vision_encoder", "test_vision_encoder",
              "patch embedding、tiny vision encoder（非因果）、projector 把 vision_hidden 投影到 text_hidden。",
              ["TinyVisionEncoder.encode()", "MultimodalProjector.__call__()"],
              ["mini_vllm/multimodal/vision_encoder.py"], ["mini_vllm/multimodal/vision_encoder.py"],
              ["`run 22` 看编码与投影的 shape。", "视觉编码器不加 causal mask（patch 互相可见）。",
               "视觉 embedding 是连续向量，不是 token id。", "projector 让维度对齐 text_hidden。"], 21),
    _ready_mm(23, "placeholder-merge", "Placeholder 与 Embedding Merge", "Phase 4",
              "lesson_23_placeholder_merge", "test_placeholder_merge",
              "PlaceholderRange、merge_multimodal_embeddings、单图/多图/越界/顺序错误的对齐校验。",
              ["merge_multimodal_embeddings()", "validate_placeholders()"],
              ["mini_vllm/multimodal/placeholders.py", "mini_vllm/multimodal/embedding_merge.py"],
              ["mini_vllm/multimodal/embedding_merge.py"],
              ["`run 23` 看合并与错误检测。", "占位处 embedding 被视觉 embedding 逐位替换。",
               "对齐错误（数量/长度/越界/重叠）必须报错。", "视觉 embedding 维度须等于 text_hidden。"], 22),
    _ready_mm(24, "mm-prefill", "Multimodal Prefill 与 Text Decode", "Phase 4",
              "lesson_24_mm_prefill", "test_mm_prefill",
              "媒体预处理→编码→投影→合并→多模态 prefill→文本 decode；vision encoder 只在 prefill 运行。",
              ["MultiModalRunner.prefill()", "MultiModalRunner.generate()"],
              ["mini_vllm/multimodal/runner.py"], ["mini_vllm/multimodal/runner.py"],
              ["`run 24` 看 encoder 只跑一次。", "prefill 用 inputs_embeds 混入视觉 embedding。",
               "decode 是纯文本，复用 LLM KV Cache。", "三层缓存互不相同。"], 23),
    _ready_mm(25, "multi-image", "多图片与动态 Visual Token", "Phase 5",
              "lesson_25_multi_image", "test_multi_image",
              "不同请求含不同数量媒体 → 视觉 token 数动态变化；媒体顺序与 placeholder 对齐。",
              ["MultiModalRunner.build_inputs()"],
              ["mini_vllm/multimodal/runner.py", "mini_vllm/multimodal/inputs.py"],
              ["mini_vllm/multimodal/inputs.py"],
              ["`run 25` 看不同请求的视觉 token 数。", "视觉 token 随媒体数变化。",
               "media_index 顺序须与出现顺序一致。", "真实 VLM 还随分辨率变 per-image token。"], 24),
    _ready_mm(26, "video-frames", "视频如何变成模型输入", "Phase 5",
              "lesson_26_video", "test_video",
              "抽帧（uniform/fixed_fps/head/tail）、FPS、frame index、timestamp、合成时间线。",
              ["VideoFrameSampler.sample()", "VideoFrame"],
              ["mini_vllm/multimodal/video_sampler.py", "mini_vllm/multimodal/media.py"],
              ["mini_vllm/multimodal/video_sampler.py"],
              ["`run 26` 看四种抽帧策略。", "抽帧在覆盖度与成本间权衡。",
               "每帧带 frame_index 与 timestamp。", "timestamp 是 metadata，不会被 LLM 自动理解。"], 25),
    _ready_mm(27, "three-cache", "多模态三层缓存", "Phase 6",
              "lesson_27_three_cache", "test_three_cache",
              "ProcessorCache / EncoderOutputCache / LLM KV Cache 的职责与 key；避免 stale cache。",
              ["ProcessorCache", "EncoderOutputCache"],
              ["mini_vllm/multimodal/cache.py"], ["mini_vllm/multimodal/cache.py"],
              ["`run 27` 看命中/未命中。", "三层缓存针对不同重复，绝不能混。",
               "encoder cache key 要含 encoder/projector 身份、dtype、schema 版本。",
               "换 encoder 身份不应误命中旧缓存。"], 26),
    _ready_mm(28, "mm-scheduler", "Multimodal Scheduler", "Phase 6",
              "lesson_28_mm_scheduler", "test_mm_scheduler",
              "把 visual token / encoder 工作量纳入调度预算；比较不同预算下的准入。",
              ["MultiModalBudget", "MultiModalEngine.run()（准入）"],
              ["mini_vllm/multimodal/budget.py", "mini_vllm/multimodal/mm_engine.py"],
              ["mini_vllm/multimodal/budget.py"],
              ["`run 28` 比较紧/松预算。", "visual token 也占序列位置、进 KV。",
               "encoder 预算限制一次编码多少媒体。", "预算只改「何时算」，不改输出。"], 27),
    _ready_mm(29, "mm-engine", "完整 Tiny Multimodal Engine", "Phase 6",
              "lesson_29_mm_engine", "test_mm_engine",
              "端到端：text/单图/多图/视频/mixed batch/三层缓存/continuous batching/visual budget。",
              ["MultiModalEngine"], ["mini_vllm/multimodal/mm_engine.py"],
              ["mini_vllm/multimodal/mm_engine.py"],
              ["`run 29` 端到端 + 报告。", "多模态 prefill + 文本 decode 的 continuous batching。",
               "相同媒体命中 encoder 缓存；encoder 不在 decode 重复运行。", "不跨请求串视觉 embedding。"], 28),
    _ready_mm(30, "mm-final", "Multimodal Final Incident Challenge", "Phase 6",
              "lesson_30_mm_final", "test_mm_final",
              "综合故障场景 + 自检：对齐、无串用、无 stale cache、timestamp 保留、visual 预算、encoder 不重复。",
              ["整套多模态引擎"], ["mini_vllm/multimodal/mm_engine.py"], [],
              ["`run 30` 跑综合场景。", "组合多图/视频/共享媒体/有限预算。",
               "自检：对齐/无串用/timestamp/缓存命中。", "首次 vs 缓存命中的 TTFT 不同。"], 29),
]


def _planned(n: int, slug: str, title: str, phase: str, track: str, goal: str) -> Lesson:
    return Lesson(
        number=n, slug=slug, title=title, phase=phase, track=track,
        status="planned", goal=goal, next=(n + 1 if n < 30 else None),
        prereqs=[n - 1] if n > 0 else [],
    )


# 完整课程大纲（roadmap）。Phase 1 只有 Lesson 0 是 ready；其余为 planned，
# course.py 会对 planned 课程明确提示「尚未在当前 Phase 实现」。
_REST = []  # 全部课程均已 ready（19–30 见 _MM）

LESSONS: dict[int, Lesson] = {
    les.number: les
    for les in [_L0, _L1, _L2, _L3, _L4, _L5, _L6, _L7, _L8, *_TEXT_ENGINE, *_MM, *_REST]
}

# 为 9–30 课补上「为什么需要这个知识」（_ready/_ready_mm 只填了 goal）。
_WHY = {
    9: "静态批处理的浪费，是理解 continuous batching 价值的前提。",
    10: "迭代级调度是现代推理引擎吞吐的关键，也是后面所有调度/分页的运行框架。",
    11: "调度策略直接决定延迟与吞吐的权衡，是推理服务化最核心的旋钮之一。",
    12: "看清连续预留的浪费，才明白为什么必须引入分页 KV。",
    13: "分配器是分页 KV 与前缀共享的地基；引用计数让「共享」与「安全回收」同时成立。",
    14: "分页把「逻辑连续、物理分散」变成现实——这正是 PagedAttention 的核心思想。",
    15: "token 预算与 chunked prefill 决定长 prompt 会不会拖垮短请求的首 token 延迟。",
    16: "前缀缓存能省下重复 prefill 的大量算力，是多轮对话 / 共享 system prompt 的关键优化。",
    17: "把所有部件组装成引擎，才算真正「造出一个 mini-vLLM」。",
    18: "综合场景检验系统在真实压力下是否仍然正确、稳定、公平。",
    19: "搞清多模态请求的阶段边界，后面图片/视频处理才不会张冠李戴。",
    20: "模型只吃张量；不理解图片如何变成张量，多模态就无从谈起。",
    21: "patch 化决定一张图产生多少 visual token，直接影响序列长度与成本。",
    22: "vision encoder + projector 把像素变成能与文本并肩的向量，是「看懂图」的核心。",
    23: "placeholder 与合并是把视觉信息塞进文本序列的关键，也是最容易出对齐 bug 的地方。",
    24: "理解「encoder 只在 prefill 跑一次、decode 是纯文本」，是多模态推理的性能关键。",
    25: "真实请求的媒体数量千变万化；动态 visual token 是多模态调度的基础。",
    26: "视频是海量帧；抽帧策略与时间元数据决定模型看到什么、以及成本多大。",
    27: "三层缓存针对不同层级的重复；分不清就会用错缓存、拿到错误结果。",
    28: "视觉工作量必须计入调度预算，否则多模态请求会压垮系统。",
    29: "端到端引擎把多模态的一切串起来，是本课程的集大成。",
    30: "综合场景检验多模态系统的对齐、隔离、缓存与时间语义是否都站得住。",
}
for _n, _w in _WHY.items():
    if _n in LESSONS and not LESSONS[_n].why:
        LESSONS[_n].why = _w

# 课程总数与「已就绪」课程数，供 doctor / where-am-i / HTML 使用。
TOTAL_LESSONS = len(LESSONS)
READY_LESSONS = sorted(n for n, l in LESSONS.items() if l.status == "ready")


def get(number: int) -> Lesson:
    if number not in LESSONS:
        raise KeyError(
            f"没有编号为 {number} 的 Lesson（有效范围 0..{max(LESSONS)}）"
        )
    return LESSONS[number]


def phase_of(number: int) -> str:
    return get(number).phase
