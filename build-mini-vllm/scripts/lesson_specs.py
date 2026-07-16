"""每课页面规格（供 scripts/gen_lesson_pages.py 使用）——Lesson 9–18（文本引擎）。

内容驱动自 course_meta 的 goal/why/hints/focus 等；这里补充每课的动画配置、
额外知识小节、常见错误、因果题与真实 vLLM 映射。
"""

# ---- 复用的动画脚本集合 ----
LIFECYCLE = ["request_lifecycle.js"]
BLOCKS = ["blocks_anim.js"]
WASTE = ["waste_anim.js"]

SPECS = {}

SPECS[9] = {
    "anim_scripts": WASTE, "anim_id": "a9",
    "anim_q": "长度不齐的一批请求，静态 padding 浪费了多少计算？",
    "mount_js": 'WasteAnim.mount("#a9", {items:['
                '{label:"r0",a:6,b:24},{label:"r1",a:20,b:24},{label:"r2",a:4,b:24},{label:"r3",a:24,b:24}],'
                'aName:"真实长度", bName:"静态padding到最长", insight:"长度差异越大，浪费越多。"});',
    "concepts": [("padding 与浪费", "<p>静态批处理把所有请求补齐到最长长度、一起跑到最长者结束。"
                  "短请求在 padding / 已完成 slot 上的计算全是浪费。浪费 = 批大小×最长长度 − Σ真实长度。</p>")],
    "errors": [("忘了 mask", "attention 算进了 padding", "用 attention_mask 屏蔽 padding 位置"),
               ("批内长度差异大", "静态批处理浪费高", "改用 continuous batching（Lesson 10）")],
    "challenge": ["若一批里有一个特别长的请求，其余都很短，浪费会怎样？",
                  "为什么 continuous batching 能消除「等待最慢者」的浪费？"],
    "vllm": [("静态成本=批×最长", "早期静态批处理", "padding 概念一致", "vLLM 用 continuous batching 取代它")],
    "summary": "静态批处理因 padding 与「等待最慢者」而浪费；请求长度差异越大越明显。",
}

SPECS[10] = {
    "anim_scripts": LIFECYCLE, "anim_id": "a10",
    "anim_q": "请求经历哪些状态？如何在别人 decode 时动态加入运行 batch？",
    "mount_js": 'RequestLifecycleAnim.mount("#a10", {blockSize:4,numBlocks:12,maxNumSeqs:3,tokenBudget:16,'
                'requests:[{id:"A",prompt:"Hello",maxNew:4,arrival:0},'
                '{id:"B",prompt:"Hi",maxNew:3,arrival:1},{id:"C",prompt:"GPU",maxNew:3,arrival:3}]});',
    "concepts": [("请求状态机", "<table><thead><tr><th>状态</th><th>含义</th></tr></thead><tbody>"
                  "<tr><td><code>WAITING</code></td><td>等待被调度</td></tr>"
                  "<tr><td><code>RUNNING</code></td><td>正在 prefill 或 decode</td></tr>"
                  "<tr><td><code>FINISHED</code></td><td>完成，归还 KV</td></tr>"
                  "<tr><td><code>ABORTED</code></td><td>被中止</td></tr></tbody></table>"
                  "<p>continuous batching = 迭代级调度：每次迭代都可准入新请求、退出完成的请求。</p>")],
    "errors": [("新请求进不来", "只在批空时才调度", "改为每次迭代都尝试准入（iteration-level）"),
               ("完成的请求还占资源", "没及时回收", "完成即移出 running 并 free KV")],
    "challenge": ["为什么 continuous batching 比 static batching 吞吐更高？",
                  "一个请求完成后，它的 KV 块应该怎么处理？"],
    "vllm": [("LLMEngine.run 迭代循环", "vLLM engine loop", "迭代级调度一致", "vLLM 在 GPU 上真正并行整批")],
    "inspect_cmd": "\npython3 course.py inspect scheduler",
    "summary": "请求是状态机；continuous batching 在每次迭代动态准入/推进/退出请求。",
}

SPECS[11] = {
    "anim_scripts": LIFECYCLE, "anim_id": "a11",
    "anim_q": "调度顺序如何影响各请求的首 token 延迟与完成时间？",
    "mount_js": 'RequestLifecycleAnim.mount("#a11", {blockSize:4,numBlocks:16,maxNumSeqs:2,tokenBudget:12,'
                'requests:[{id:"long",prompt:"The quick brown fox",maxNew:4,arrival:0},'
                '{id:"s1",prompt:"Hi",maxNew:3,arrival:0},{id:"s2",prompt:"Yo",maxNew:3,arrival:1}]});',
    "concepts": [("四种策略", "<table><thead><tr><th>策略</th><th>思路</th></tr></thead><tbody>"
                  "<tr><td>FIFO</td><td>先到先服务</td></tr>"
                  "<tr><td>decode-first</td><td>优先推进已在解码的请求（利于吞吐）</td></tr>"
                  "<tr><td>SJF</td><td>最短作业优先（短请求首 token 更早）</td></tr>"
                  "<tr><td>balanced</td><td>decode 优先但给 prefill 留预算，防饥饿</td></tr></tbody></table>"
                  "<p><code>python3 course.py run 11</code> 会打印四种策略的 iterations 与平均 TTFT——"
                  "各不相同，但输出完全一致。</p>")],
    "errors": [("SJF 没生效", "准入仍按 FIFO", "准入顺序也要按策略排序（_order_waiting）"),
               ("长请求饿死短请求", "decode 一直优先", "用 balanced 给 prefill 保留预算")],
    "challenge": ["SJF 为什么能降低平均等待时间？它有什么公平性风险？",
                  "decode-first 对吞吐和对单个新请求的 TTFT 各有什么影响？"],
    "vllm": [("policy 排序", "vLLM Scheduler 策略", "预算+优先级思路一致", "vLLM 还含抢占、优先级、公平性")],
    "inspect_cmd": "\npython3 course.py inspect scheduler",
    "summary": "调度策略决定「何时算」而非「算什么」；SJF 利于短请求，balanced 防饥饿。",
}

SPECS[12] = {
    "anim_scripts": WASTE, "anim_id": "a12",
    "anim_q": "按 max_seq 连续预留 vs 分页按需分配，各浪费多少 KV？",
    "mount_js": 'WasteAnim.mount("#a12", {items:['
                '{label:"s0",a:8,b:64},{label:"s1",a:16,b:64},{label:"s2",a:8,b:64},'
                '{label:"s3",a:24,b:64},{label:"s4",a:8,b:64}],'
                'aName:"分页(向上取整到块)", bName:"连续(按max_seq=64预留)", '
                'insight:"连续预留浪费随 max_seq 与并发暴涨；分页只剩块内零头。"});',
    "concepts": [("两种碎片", "<p><strong>internal fragmentation</strong>：预留多、用得少（块内/段内空着）。"
                  "<strong>external fragmentation</strong>：空闲被切成不连续的小块，凑不出一段连续空间。"
                  "分页用固定大小的块 + 逻辑/物理解耦，几乎消除这两种浪费。</p>")],
    "errors": [("按 max_seq 预留", "internal 碎片巨大", "改为分页按需分配"),
               ("要求连续大段", "external 碎片凑不出", "分页允许物理不连续")],
    "challenge": ["并发请求数翻倍，连续预留的浪费如何变化？分页呢？",
                  "block_size 取大或取小分别有什么权衡？"],
    "vllm": [("按需分页", "PagedAttention", "分页消除碎片", "vLLM 用真实 kernel 直接在分页布局上算")],
    "summary": "连续预留造成大量碎片；分页把浪费压到「不足一块」的零头。",
}

_B = "blocks_anim.js"
SPECS[13] = {
    "anim_scripts": BLOCKS, "anim_id": "a13",
    "anim_q": "物理块如何分配、共享、释放？引用计数怎么防止 double free？",
    "mount_js": 'BlocksAnim.mount("#a13", {frames:['
        '{owners:[null,null,null,null],desc:"4 个物理块，全部空闲。",code:"BlockAllocator(num_blocks=4)"},'
        '{owners:[0,null,null,null],newly:[0],labels:{0:"A"},desc:"请求 A 分配 b0（refcount=1）。",code:"allocate() -> 0"},'
        '{owners:[0,1,null,null],newly:[1],labels:{0:"A",1:"B"},desc:"请求 B 分配 b1。",code:"allocate() -> 1"},'
        '{owners:[0,1,null,null],newly:[0],labels:{0:"A x2",1:"B"},desc:"A 共享 b0：incref，refcount=2。",code:"incref(0)"},'
        '{owners:[0,1,null,null],labels:{0:"A",1:"B"},desc:"free(b0)：refcount 2->1，尚未回收。",code:"free(0)"},'
        '{owners:[null,1,null,null],labels:{1:"B"},desc:"再 free(b0)：1->0，回收进 free list。",code:"free(0) -> 回收"},'
        '{owners:[null,1,null,null],labels:{1:"B"},desc:"耗尽后 allocate 抛 MemoryError；重复 free 抛 double-free；未释放即自检报泄漏。",code:"check_no_leak()"}]});',
    "concepts": [("引用计数", "<p>allocate 令 refcount=1；incref 让另一序列也引用（共享，Lesson 16）；"
                  "free 令 refcount−1，归零才回收。重复 free 到负数即 double free，主动报错。</p>")],
    "errors": [("double free", "释放已空闲的块", "free 前检查 refcount>0"),
               ("leak", "结束时仍有块未释放", "收尾 check_no_leak()"),
               ("OOM", "free list 空", "分配前判断，或触发抢占")],
    "challenge": ["为什么需要引用计数而不是布尔「已用/空闲」？",
                  "block_size 如何影响块数量与碎片？"],
    "vllm": [("BlockAllocator", "BlockSpaceManager", "free list + 引用计数", "vLLM 还有 CoW、换出、跨设备")],
    "inspect_cmd": "\npython3 course.py inspect blocks",
    "summary": "物理块用 free list + 引用计数管理，支持共享与安全回收，并检测 OOM/double-free/leak。",
}

SPECS[14] = {
    "anim_scripts": BLOCKS, "anim_id": "a14",
    "anim_q": "block table 如何把逻辑块映射到物理块？为什么可以物理不连续？",
    "mount_js": 'BlocksAnim.mount("#a14", {frames:['
        '{owners:[null,null,null,null,null,null,null,null],desc:"8 个物理块；一条序列按逻辑块申请物理块。",code:"PagedKVCache"},'
        '{owners:[null,null,null,0,null,null,null,null],newly:[3],labels:{0:"L0"},desc:"逻辑块 0 -> 物理块 3。",code:"block_table=[3]"},'
        '{owners:[null,null,null,0,null,1,null,null],newly:[5],labels:{0:"L0",1:"L1"},desc:"逻辑块 1 -> 物理块 5（物理不连续！）。",code:"block_table=[3,5]"},'
        '{owners:[null,2,null,0,null,1,null,null],newly:[1],labels:{0:"L0",1:"L1",2:"L2"},desc:"逻辑块 2 -> 物理块 1。逻辑连续，物理分散。",code:"block_table=[3,5,1]"},'
        '{owners:[null,2,null,0,null,1,null,null],labels:{0:"L0",1:"L1",2:"L2"},desc:"attention 前按逻辑顺序 gather -> 与连续 KV 逐值一致（误差 0）。",code:"get(layer) gather"}]});',
    "concepts": [("寻址", "<p>token 逻辑位置 <code>pos</code> → <code>logical=pos//block_size</code>、"
                  "<code>offset=pos%block_size</code>；再经 block table 查到物理块。这就是操作系统页表的思路。</p>")],
    "errors": [("gather 顺序错", "block table 索引算错", "logical=pos//bs, offset=pos%bs"),
               ("越界", "块不够", "增长时按需分配新块")],
    "challenge": ["为什么逻辑连续、物理分散不影响 attention 结果？",
                  "block_size 变大/变小对 gather 成本与碎片各有何影响？"],
    "vllm": [("PagedKVCache + block table", "PagedAttention + block table", "逻辑/物理解耦一致",
              "vLLM kernel 直接在分页布局算，免 gather")],
    "summary": "block table 把逻辑块映射到物理块；逻辑连续、物理可分散，gather 后与连续 KV 一致。",
}

SPECS[15] = {
    "anim_scripts": LIFECYCLE, "anim_id": "a15",
    "anim_q": "token 预算小于长 prompt 时，chunked prefill 如何让系统继续推进？",
    "mount_js": 'RequestLifecycleAnim.mount("#a15", {blockSize:4,numBlocks:24,maxNumSeqs:2,tokenBudget:8,'
                'requests:[{id:"long",prompt:"The quick brown fox jumps",maxNew:3,arrival:0},'
                '{id:"short",prompt:"Hi",maxNew:4,arrival:0}]});',
    "concepts": [("预算与切块", "<p><code>max_num_batched_tokens</code> 是一次迭代的 token 预算。"
                  "长 prompt 若一次 prefill 超预算，就切成多块跨迭代处理，并与其它请求的 decode 混排。"
                  "若关闭 chunked prefill 且 prompt 超预算，系统会明确报「调度停滞」——这正是它要解决的问题。</p>")],
    "errors": [("长 prompt 卡住", "预算 < prompt 且未开 chunked", "开启 chunked prefill 或增大预算"),
               ("切块改变结果", "positions/KV 写错", "切块只改「每步算多少」，不改结果")],
    "challenge": ["chunked prefill 对长 prompt 的 TTFT 和对短请求的 TTFT 各有何影响？",
                  "chunk 太小或太大分别有什么代价？"],
    "vllm": [("Scheduler chunked 分支", "vLLM chunked prefill", "切块 + 混排一致", "vLLM 有更精细的预算与优先级")],
    "summary": "token 预算约束一次迭代的处理量；chunked prefill 把长 prompt 切块、与 decode 混排，保持推进且不改变输出。",
}

SPECS[16] = {
    "anim_scripts": BLOCKS, "anim_id": "a16",
    "anim_q": "相同前缀的两个请求如何共享 KV 物理块？命中为何不改变输出？",
    "mount_js": 'BlocksAnim.mount("#a16", {frames:['
        '{owners:[null,null,null,null,null,null,null,null],desc:"两条共享同一段前缀的请求。",code:"PrefixCache"},'
        '{owners:[2,2,0,null,null,null,null,null],newly:[0,1,2],labels:{2:"共享前缀",0:"A尾部"},desc:"请求 A：前缀块 b0,b1 + 自己的尾部块 b2。A 完成后把前缀块登记进缓存。",code:"on_finish: 登记前缀块"},'
        '{owners:[2,2,0,1,null,null,null,null],newly:[3],labels:{2:"共享前缀",0:"A尾部",1:"B尾部"},desc:"请求 B 命中前缀 -> 复用 b0,b1（incref），只新分配尾部块 b3。",code:"attach: 命中，跳过前缀 prefill"},'
        '{owners:[2,2,0,1,null,null,null,null],labels:{2:"共享前缀",0:"A尾部",1:"B尾部"},desc:"命中不改变输出：因果注意力下前缀位置的 K/V 只依赖前缀本身，两请求逐值相同。",code:"输出与不开前缀缓存一致"}]});',
    "concepts": [("为什么正确", "<p>causal attention 中，位置 i 的 K/V 只依赖 token 0..i（且用绝对位置 RoPE）。"
                  "所以相同前缀（同 token、同位置）的 K/V 逐值相同，直接复用物理块即可，"
                  "命中<strong>不改变输出</strong>。缓存用块级链式哈希标识前缀，用引用计数让缓存块存活。</p>")],
    "errors": [("命中改变了输出", "共享了位置不对齐的块", "只共享从位置 0 起、块对齐的前缀"),
               ("缓存块被误回收", "没给缓存持有引用", "登记时 incref，flush 时释放"),
               ("整段命中无 token", "把最后一块也命中了", "至少留 1 个 token 跑 prefill 取首 token logits")],
    "challenge": ["为什么只有「从头开始、块对齐」的前缀才能共享？",
                  "system prompt 很长且被大量请求共享时，前缀缓存能省多少？"],
    "vllm": [("块链式哈希 + 引用计数", "vLLM Prefix Caching", "块级共享一致", "vLLM 有更强的 hash/eviction 与 CoW")],
    "summary": "相同前缀共享 KV 物理块（块哈希 + 引用计数）；因果注意力保证命中不改变输出。",
}

SPECS[17] = {
    "anim_scripts": LIFECYCLE, "anim_id": "a17",
    "anim_q": "完整引擎如何把调度、分页 KV、模型、采样串成一个 loop？",
    "mount_js": 'RequestLifecycleAnim.mount("#a17", {blockSize:4,numBlocks:24,maxNumSeqs:3,tokenBudget:16,'
                'requests:[{id:"chat1",prompt:"Hi",maxNew:5,arrival:0},'
                '{id:"chat2",prompt:"Hello",maxNew:6,arrival:1},'
                '{id:"long",prompt:"The quick brown fox",maxNew:4,arrival:2}]});',
    "concepts": [("engine loop", "<pre><code>while 有未完成请求:\n"
                  "    out = scheduler.schedule(...)      # 谁、各多少 token\n"
                  "    for item in out: 运行模型 → 采样 → 更新请求\n"
                  "    finished = scheduler.remove_finished()\n"
                  "    for r in finished: r.kv.free()     # 归还 KV（无泄漏）</code></pre>"
                  "<p><code>run 17</code> 会输出性能报告：TTFT、吞吐、KV 利用率、前缀命中率，并与 naive 基线对比。</p>")],
    "errors": [("串请求", "共享了不该共享的状态", "每请求独立 KV；测试用逐请求参考对齐"),
               ("KV 泄漏", "完成未 free", "remove_finished 后 free（前缀缓存在 shutdown flush）")],
    "challenge": ["引擎输出为何必须与「逐请求单独生成」逐 token 一致？",
                  "报告里哪些指标关注延迟、哪些关注吞吐？"],
    "vllm": [("LLMEngine", "vLLM LLMEngine", "组件与 loop 一致", "vLLM 有 worker/分布式/异步 API")],
    "inspect_cmd": "\npython3 course.py benchmark 17 --mode quick",
    "summary": "完整引擎 = 调度 + 分页 KV + 模型 + 采样的迭代 loop；输出正确、无泄漏，并可出性能报告。",
}

SPECS[18] = {
    "anim_scripts": LIFECYCLE, "anim_id": "a18",
    "anim_q": "综合场景下，引擎能否同时做到正确、无泄漏、无饥饿？",
    "mount_js": 'RequestLifecycleAnim.mount("#a18", {blockSize:4,numBlocks:20,maxNumSeqs:3,tokenBudget:12,'
                'requests:[{id:"sysA",prompt:"System: What is AI",maxNew:5,arrival:0},'
                '{id:"sysB",prompt:"System: What is ML",maxNew:6,arrival:1},'
                '{id:"chat",prompt:"Hi",maxNew:4,arrival:2},{id:"long",prompt:"Summarize the fox story",maxNew:3,arrival:2}]});',
    "concepts": [("综合验收", "<p>本课把前面所有能力放进一个场景：短聊天 + 长 prompt + 不同输出长度 + "
                  "共享前缀 + 有限 KV 块 + 动态到达。<code>run 18</code> 会自检：正确性（cached==reference，"
                  "无重复执行/无串请求）、无 KV 泄漏、所有请求完成（无 starvation），并与 naive 基线对比处理量。</p>")],
    "errors": [("有请求饿死", "策略不公平", "用 balanced，给 prefill 留预算"),
               ("KV 不够", "块太少", "分页 + 及时回收；真实系统会抢占换出")],
    "challenge": ["首次请求与命中前缀的请求，TTFT 差异来自哪里？",
                  "如果把 num_blocks 调到很小，会先触发什么？如何缓解？"],
    "vllm": [("综合引擎自检", "vLLM 生产引擎", "正确性/无泄漏/公平的目标一致", "vLLM 有抢占、优先级、SLA、可观测性")],
    "inspect_cmd": "\npython3 course.py benchmark 18 --mode quick",
    "summary": "综合场景下引擎保持正确、无泄漏、无明显饥饿，并显著少于 naive 的重复计算。文本主线（0–18）到此完成。",
}
