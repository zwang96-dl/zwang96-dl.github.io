# 现代 GPU 编程 · 面向 ML 系统(中文精读笔记 · 超级小白版)

> 这是基于 mlc.ai 公开教材《Modern GPU Programming for MLSys》(© 2026 MLC Community)撰写的**中文原创精读笔记 / 学习导读**,仅用于学习目的。
>
> 本笔记是对原书的**转述、归纳与讲解**,**并非原书的逐字翻译**;书中所有结论、代码示例、设计取舍均以原文为准。阅读原文请访问官方站点:
>
> **<https://mlc.ai/modern-gpu-programming-for-mlsys/>**

> **这是"超级小白版"**:同一套内容,有两个版本。一个是写给"已经懂点 GPU、想系统学"的工程师;**你现在看的这个版本更白一档**——它是写给"会写代码,但**从没碰过 GPU、显卡、CUDA、并行**"的程序员的。你会写 Python / C++ / 后端 / 数据处理都行,但只要一提到"线程怎么并行""内存为什么要分好几层""矩阵乘法为什么是主角"就一脸懵——没关系,这个版本就是为你写的。每个 GPU 专有名词第一次出现,我都会停下来用大白话讲清楚"它是什么、为什么需要它",绝不假设你有任何 GPU 背景。

先说说这份笔记想带你去哪。一句话:**从"GPU 到底是怎么跑代码的",一直走到"怎么亲手写出真正快的程序"。**

打个比方,这就像学开车。我们会分三步走:

1. 先看 **Blackwell 架构**——这是 NVIDIA 一代 GPU 芯片的代号(你可以理解成"某个型号的发动机")。这一步是**先掀开引擎盖,看清这台机器内部长什么样**:它有哪些零件、各干什么。
2. 再学 **TIRx 编程模型**——这是一种**专门用来给 GPU 写程序的语言/工具**(后面会细讲)。这一步是**学会握方向盘踩油门**,也就是用什么语言、怎么去指挥这台机器干活。
3. 最后是 **GEMM 和 Flash Attention 实战**。GEMM 是"通用矩阵乘法"(General Matrix Multiply)的缩写,你先记住它就是"两个大表格相乘"——这是 AI 计算里最核心、跑得最多的一种运算;Flash Attention 则是大语言模型里一个关键的计算模块。这一步就是**真正上路开一圈**,把前面学的全都用一遍。

末尾还塞了一整套"编译器内部"和"语言参考",相当于这台机器的**维修手册**——你不用一口气读完,需要时翻回来查就行。

> **前置知识**:完全没 GPU 基础?那你来对地方了,但**别直接从第 1 章开始**——请先花 20 分钟读 [第 0 章 · 写给纯新手:GPU & ML Infra 极简入门](./ch00_gpu_ml_primer.md)。
>
> 为什么非得先读第 0 章?因为后面正文里会**满天飞**地出现一堆词:warp、内存层级、GEMM、Tensor Core……这些词在第 0 章里会被用最直白的大白话先讲一遍,让你**心里先有个画面**。等你对它们有了感觉,再读正文就会顺得多——不然就像没学过单词就去读文章,每句都要查,很容易劝退。

---

## 目录

> 下面就是全书的章节地图。每一行的**章节标题**点过去,是本仓库里对应的中文笔记;**「原文链接」**点过去,是 mlc.ai 的官方英文页面(想对照原书时再点)。
>
> 表格里"一句话简介"这一列,目前还塞了不少专有名词(warp、TMA、Tensor Core……)——**这些词你现在看不懂完全正常,不用慌**。它们都会在第 0 章和对应正文里从头讲起;这张目录只是让你**心里先有个大致地图**,知道每章大概在聊什么、彼此什么关系。

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

每个人来这儿的起点都不一样,所以我给你列了几条路。你先看看哪条最像自己,照着走就行。

> **一句话先理解**:如果你是"从没碰过 GPU 的程序员",那别犹豫,直接走**路线 A**——从头老老实实读下去就好,其他几条路是给已经有点底子的人留的。

- **路线 A · 从零开始打基础(最适合纯新手,也就是大多数读这个版本的你):** 先读 **第 0 章**建立画面感,然后 **01 一路读到 24**。怎么理解这个顺序?第一部分(01–08)先把"硬件长什么样、代码在 GPU 上怎么跑"啃明白,心里有个谱;第二部分(09–10)学会怎么用 TIRx 这门语言把想法写成代码;到了第三、第四部分(11–14)的实战,再把前面学的那些零件一个一个拼起来,造出真正能用的快程序。第五部分(15–24)你**不用从头读到尾**,把它当成查字典的工具书,卡住了再翻。

- **路线 B · 上来就想写 GEMM(给已经懂 GPU 的人):** 你要是 CUDA(NVIDIA 写 GPU 程序的传统语言)和 GPU 架构本来就熟,那别绕弯子,直接跳到 **第 11 章** 动手写"分块 GEMM"。写着写着碰到不认识的东西——比如 TMA、mbarrier、TMEM、warp 专门化(这些词后面都会讲)——再回头翻 **第 05 到 08 章** 补一补就好。

- **路线 C · 只想弄懂"性能优化大概是怎么个思路":** 先读 **第 02 章**,把"屋顶线模型"(一个判断程序到底卡在哪、是算得慢还是搬数据慢的分析工具)和"重叠"(让搬数据和算数据同时进行,别干等)这两件事吃透,养成"先诊断、再动手"的习惯;然后看 **第 08 章** 和 **第 13 章**,慢慢就摸清怎么把硬件的性能压榨干净了。

- **路线 D · 想给 TIRx 或编译器本身贡献代码(进阶):** 直接进 **第 17、18 章**,先把编译器内部和"下降流水线"(把高层代码一步步翻译成 GPU 能执行的底层代码的过程)看懂;**第 19 到 24 章** 那套语言参考放手边,要查语义随时翻。

- **要是你正在抓那种莫名其妙的 bug(数据被写坏、程序卡死不动、两段代码抢着读写同一块内存):** 别慌,先翻 **第 16 章**(调试 Warp 专门化 Kernel),照着那张"四要素交接表"一项一项对过去——这类问题十有八九就是某个"交接"没对齐(谁该先写、谁该等着读,顺序乱了)。

> **提示:** 每章的核心概念都配了中英对照和一句话解释,想查的话翻本仓库的 **[术语对照表](./术语对照表.md)** 就行——读着读着忘了某个词是啥,去那儿一查最快。

---

## 版权与免责声明

- 原教材《Modern GPU Programming for MLSys》的版权是 **© 2026 MLC Community** 的。
- 这个仓库是**非官方的中文学习笔记**,东西都是社区里学习的人自己整理的,难免有理解不到位的地方;只要和原文对不上,**就以原文为准**。
- 这份笔记只用来学习和交流,千万别拿去做商业用途。
