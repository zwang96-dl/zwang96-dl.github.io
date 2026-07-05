# 现代 GPU 编程 · 面向 ML 系统(中文精读笔记)

> 这是基于 mlc.ai 公开教材《Modern GPU Programming for MLSys》(© 2026 MLC Community)撰写的**中文原创精读笔记 / 学习导读**,仅用于学习目的。
>
> 本笔记是对原书的**转述、归纳与讲解**,**并非原书的逐字翻译**;书中所有结论、代码示例、设计取舍均以原文为准。阅读原文请访问官方站点:
>
> **<https://mlc.ai/modern-gpu-programming-for-mlsys/>**

这份笔记就想带你走通一条路:从"GPU 到底是怎么跑代码的",一直走到"怎么亲手写出真正快的内核"。怎么走?顺着三条主线——先看 **Blackwell 架构**,搞清楚硬件长啥样;再学 **TIRx 编程模型**,知道拿什么语言去写;最后是 **GEMM 和 Flash Attention 实战**,把前面学的东西全拿出来用一遍。末尾还塞了一整套编译器内部和语言参考,你随时翻回来查就行。

> **前置知识**:完全没 GPU 基础?**别直接从第 1 章开始**——先花 20 分钟读 [第 0 章 · 写给纯新手:GPU & ML Infra 极简入门](./ch00_gpu_ml_primer.md)。它用大白话把 warp、内存层级、GEMM、Tensor Core 这些后面满天飞的词先讲一遍,有了画面感,后面会顺很多。

---

## 目录

> 下面表格里,「原文链接」点过去是 mlc.ai 的官方页面;章节标题点过去是本仓库里对应的笔记。

### 🚀 新手必读

| 章节 | 标题 | 一句话简介 |
| :--: | :--- | :--- |
| 00 | [写给纯新手:GPU & ML Infra 极简入门](./ch00_gpu_ml_primer.md) | 零基础总入口:用大白话讲清线程层级 / 内存层级 / GEMM / Tensor Core / 重叠,先建立画面再读正文。 |

### 第一部分 · 硬件、内存与执行模型

| 章节 | 标题 | 一句话简介 | 原文链接 |
| :--: | :--- | :--- | :--: |
| 01 | [GPU 执行模型](./ch01_gpu_execution_model.md) | 从 SIMT、warp / CTA / cluster 到 Tensor Core、TMA、TMEM,建立现代 GPU 的整体心智模型。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_background/index.html) |
| 02 | [什么决定 Kernel 的速度](./ch02_what_makes_kernel_fast.md) | 用屋顶线模型(算术强度定胜负)建立「先诊断、再优化」的工作流,核心杠杆是重叠。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_performance/index.html) |
| 03 | [数据布局与其记号](./ch03_data_layout_notation.md) | 形状 / 步长 / 视图 / 分块 / 命名轴 / swizzle 的统一记号,是后续所有 layout 讨论的语言。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_data_layout/index.html) |
| 04 | [各代 GPU 的 Tensor Core 操作数布局](./ch04_tensorcore_operand_layouts.md) | 从 Ampere 到 Blackwell,MMA 操作数 / 累加器在寄存器与 TMEM 中的逐代布局演变。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_layout_generations/index.html) |
| 05 | [异步数据搬运:TMA](./ch05_async_tma.md) | 张量内存加速器:用张量映射描述符 + tile 坐标 + mbarrier 实现异步批量搬运与多级缓冲。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tma/index.html) |
| 06 | [特殊内存:TMEM(Tensor Memory)](./ch06_tmem.md) | Blackwell 专供 Tensor Core 累加的二维(Lane × Col)片上内存及其使用约束。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tmem/index.html) |
| 07 | [异步协调:mbarrier](./ch07_mbarriers.md) | 到达计数 + 相位位 + 字节计数三机一体的异步屏障,是所有流水线的协调中枢。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_async_barriers/index.html) |
| 08 | [进阶:集群启动控制(CLC)](./ch08_cluster_launch_control.md) | 用持久化核 + 工作窃取式 tile 调度榨干尾部 SM,理解关键路径与尾部行为。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_clc/index.html) |

### 第二部分 · TIRx 编程模型

| 章节 | 标题 | 一句话简介 | 原文链接 |
| :--: | :--- | :--- | :--: |
| 09 | [TIRx 入门](./ch09_intro_tirx.md) | IR 层级的 GPU 编程 DSL,用最小 GEMM 讲透 scope / layout / dispatch 三大要素。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_intro_tirx/index.html) |
| 10 | [TIRx Layout API](./ch10_tirx_layout_api.md) | 把纸面布局记号变成编译器对象:TileLayout / SwizzleLayout / ComposeLayout。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tirx_layout_api/index.html) |

### 第三部分 · GEMM 实战(Step 1→9)

| 章节 | 标题 | 一句话简介 | 原文链接 |
| :--: | :--- | :--- | :--: |
| 11 | [构建分块 GEMM(Step 1–3)](./ch11_gemm_basics.md) | 从最朴素的分块 GEMM 起步,搭好寄存器 / 共享内存 / MMA 的基本骨架。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_gemm_basics/index.html) |
| 12 | [用 TMA 为 GEMM 做流水线(Step 4–6)](./ch12_gemm_async.md) | 引入 TMA + mbarrier + 双缓冲 / 持久化 kernel,让搬运与计算重叠起来。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_gemm_async/index.html) |
| 13 | [用 Warp 专门化与 Cluster 扩展 GEMM(Step 7–9)](./ch13_gemm_advanced.md) | 用 warp 专门化拆分生产者 / 消费者,并借 cluster 跨 CTA 协作冲击峰值。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_gemm_advanced/index.html) |

### 第四部分 · Flash Attention 4

| 章节 | 标题 | 一句话简介 | 原文链接 |
| :--: | :--- | :--- | :--: |
| 14 | [Flash Attention 4](./ch14_flash_attention.md) | 把前述全部机制(TMA / TMEM / mbarrier / warp 专门化)综合到一个完整的 FA4 内核中。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_flash_attention/index.html) |

### 第五部分 · 参考与附录

| 章节 | 标题 | 一句话简介 | 原文链接 |
| :--: | :--- | :--- | :--: |
| 15 | [参考资料(附录)](./ch15_appendix_reference.md) | 一张「需求 → 去处」导航表,告诉你查语言特性 / 看下降流水线 / 调内核时该翻哪一页。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/appendix/index.html) |
| 16 | [调试 Warp 专门化 Kernel](./ch16_debugging_warp_specialized.md) | 把内核看成一组「交接」,用「角色 / 存储 / 交接 / 生命周期」四要素表对照生成的 CUDA 排错。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/appendix/debugging_warp_specialized.html) |
| 17 | [编译器内部(概览)](./ch17_compiler_internals.md) | 面向贡献者:讲清 `tvm.compile(tir_pipeline="tirx")` 内部的下降流水线与关键阶段。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/arch/index.html) |
| 18 | [TIRx Lowering 流水线](./ch18_lowering_pipeline.md) | 高层 TIRx 经 18 个模块级 pass 下降为 host/device 函数,核心是 LowerTIRx。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/arch/lowering_pipeline.html) |
| 19 | [TIRx 语言参考(概览)](./ch19_tirx_language_reference.md) | 语言特性分五块(解析器工具 / 数据类型 / 缓冲区 / 控制流 / 内建)的导读与速查。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/language_reference/index.html) |
| 20 | [解析器工具(Parser utilities)](./ch20_parser_utils.md) | `T.meta_var` / `@T.inline` / `@T.meta_class` / `T.constexpr` 四大解析期元编程工具。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/language_reference/cuda/parser_utils.html) |
| 21 | [数据类型与表达式](./ch21_data_types.md) | dtype / type / PrimExpr / 指针 / 向量化访存的语义与下降到 CUDA 的方式。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/language_reference/cuda/data_types.html) |
| 22 | [缓冲区与内存](./ch22_buffers.md) | buffer = 指针 + 元数据;静态 / 动态共享内存、竞技场分配、TMEM 列偏移。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/language_reference/cuda/buffers.html) |
| 23 | [控制流](./ch23_control_flow.md) | 统一 vs 发散控制流、集体操作、`elect_sync`、各类同步与循环修饰符。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/language_reference/cuda/control_flow.html) |
| 24 | [CUDA C++ / PTX intrinsics](./ch24_intrinsics.md) | 后端内建函数:选举 / 具名屏障 / 栅栏 / 异步代理 / 蝶形归约等逃生口。 | [原文](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/language_reference/cuda/threads_sync.html) |

---

## 建议阅读路径

每个人来这儿的起点都不一样,所以我给你列了几条路。你先看看哪条最像自己,照着走就行:

- **路线 A · 从零开始打基础(最适合新手):** 干脆从 **01 一路读到 24**。第一部分先把硬件和执行模型啃明白,心里有个谱;第二部分学会怎么用 TIRx 把这些写成代码;到了第三、第四部分的实战,再把前面那些机制一个一个串起来。第五部分你不用从头读到尾,当成查字典的工具书,需要了再翻。

- **路线 B · 上来就想写 GEMM:** 你要是 CUDA 和 GPU 架构本来就熟,那别绕弯子,直接跳到 **第 11 章** 动手写分块 GEMM。写着写着碰到不认识的东西——比如 TMA、mbarrier、TMEM、warp 专门化——再回头翻 **第 05 到 08 章** 补一补就好。

- **路线 C · 只想弄懂性能优化是怎么个思路:** 先读 **第 02 章**,把屋顶线模型和"重叠"这两件事吃透,养成"先诊断、再动手"的习惯;然后看 **第 08 章**(持久化核和尾部行为)和 **第 13 章**(warp 专门化、cluster),慢慢就摸清怎么把硬件的性能压榨干净了。

- **路线 D · 想给 TIRx 或编译器贡献代码:** 直接进 **第 17、18 章**,先把编译器内部和下降流水线看懂;**第 19 到 24 章** 那套语言参考放手边,要查语义随时翻。

- **要是你正在抓那种莫名其妙的 bug(数据被写坏、死锁、数据竞争):** 别慌,先翻 **第 16 章**(调试 Warp 专门化 Kernel),照着那张四要素交接表一项一项对过去——这类问题十有八九就是某个交接没对齐。

> **提示:** 每章的核心概念都配了中英对照和一句话解释,想查的话翻本仓库的 **[术语对照表](./术语对照表.md)** 就行。

---

## 版权与免责声明

- 原教材《Modern GPU Programming for MLSys》的版权是 **© 2026 MLC Community** 的。
- 这个仓库是**非官方的中文学习笔记**,东西都是社区里学习的人自己整理的,难免有理解不到位的地方;只要和原文对不上,**就以原文为准**。
- 这份笔记只用来学习和交流,千万别拿去做商业用途。
