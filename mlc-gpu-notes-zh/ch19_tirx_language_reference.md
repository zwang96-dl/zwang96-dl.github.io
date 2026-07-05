# 第 19 章 · TIRx 语言参考(概览)

> 原文:[TIRx Language Reference](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/language_reference/index.html)

> **本章要点(TL;DR)**
> - 一句话:这一章是「TIRx 语言参考」那一组页面的**导读**。它自己不讲细节,只告诉你一件事——写 TIRx 设备内核 / device kernel 的时候,想查某个特性到底**怎么写、是什么语义**,该翻哪一页。
> - 这些参考页是从第二部分的「TIRx 入门」里单独抠出来的。说白了,它们是**速查手册 / reference**,不是**教程 / tutorial**。
> - TIRx 语言被拆成了五块:解析器工具、数据类型与表达式、缓冲区与内存、控制流、CUDA C++/PTX 内建函数。
> - 这五块刚好对着你写内核时一层层往下走的五个关注点:**元编程 → 标量数据 → 内存对象 → 控制结构 → 直达硬件**。建议先照这个顺序扫一遍,之后写代码时哪儿卡了再回来查。
> - 它属于"参考与附录"这一部分的子页面。主线内容都在 Parts I–IV,这一组只管"把细节列清楚"。

> **前置知识**:读这一章前,最好先懂线程层级(warp / CTA / grid)、共享内存与寄存器这些基础内存概念,以及 TIRx 内核大概长什么样。没把握的话,先翻一下 [第 0 章 · 极简入门](./ch00_gpu_ml_primer.md),以及第 9 章的 TIRx 基础。本章会默认你已经认识这些词。

---

## 19.1 这一页是什么

这一页是一组**语言参考**的入口。你写 TIRx 设备内核会用到的所有语言特性,原文都从第二部分那篇偏教学的「TIRx 入门」里抽了出来,整理成几页速查文档。

为什么要这么拆?因为入门那种一步步讲故事的方式,适合你**第一次理解概念**;可一旦你已经在写内核,只想确认一个很具体的小问题,再去翻教程就太慢了。比如:`T.unroll` 到底是完全展开,还是只是顺序跑一遍循环?动态共享内存 / dynamic shared memory 是不是只能分配一块?`elect_sync()` 最后选中的到底是哪个线程?把这些答案做成手册式的参考页,就是为了让你**遇到问题一查就有**。

> **关键**:这一组页面是拿来"按需查"的,不是拿来"从头读到尾"的。建议先快速扫一遍,心里有个数——每页大概讲什么(本章就帮你干这件事),之后写代码卡住了,再精确翻回对应那页。

---

## 19.2 五个子页面分别讲什么

原文列了五个子页面。下面一个一个说它们讲什么,各配一个最典型的例子。看完你心里就有谱了:**碰到哪类问题,该去翻哪一页**。

| 子页面 | 一句话定位 | 你会在这里查到 |
| --- | --- | --- |
| 解析器工具(Parser utilities) | **解析期(parse-time)的元编程**工具 | 把 Python 值内联进 IR、内联函数、打包解析期状态 |
| 数据类型与表达式 | 每个表达式的**低层 dtype 与高层 type** | dtype 到 CUDA 类型的映射、向量类型、指针(handle) |
| 缓冲区与内存(Buffers & memory) | 内核里所有**内存对象**的声明与语义 | 各种 scope(global/shared/local/tmem)、视图、标量、`let`、张量内存(TMEM) |
| 控制流(Control flow) | `if` / 循环家族 / `while` | 四种循环、统一(uniform)vs 发散(divergent)控制流的坑 |
| CUDA C++/PTX 内建函数 | 直达硬件的**两个逃生口(escape hatches)** | `T.cuda.*` / `T.ptx.*` 内建、内联原始 CUDA、同步语义 |

下面一块块展开说。

### 解析器工具:解析期的元编程

先说这一页要解决什么:有些事你希望在"代码还没真跑、只是被解析"的阶段就办掉,比如提前把循环展开、把一个函数整个铺开。这一页讲的就是干这些事的工具。它们都在 **TVMScript 被解析成 TIRx 的那一刻**起作用,不会冒出任何运行时代码,纯粹是在**解析期 / parse-time**做元编程 / metaprogramming。

- **`T.meta_var(x)`**:你在 **Python** 里先把一个值算好,然后用它把这个值当成编译期的"元值 / meta value",直接塞进 IR。它最妙的一点是:拿这种值去写普通的 Python `for`,循环会在解析期就**自动展开 / unroll**。

```python
n = T.meta_var(4)          # n 是个被内联的 Python int
for j in range(n):         # 解析期就展开成 4 行直线代码
    acc[0] = acc[0] + A[tx, j]
```

- **`@T.inline`**:定义一个函数,让它**在每个调用的地方都被内联**。也就是说,生成的代码里根本看不到真正的函数调用,而是把函数体直接摊开。它走的是 Python 的 LEGB 词法作用域和延迟绑定那一套。
- **`@T.meta_class`**:给一个普通 Python 类打个标记,意思是"它的实例就是一个解析期元值"。这个类的字段里可以放缓冲区和标量。用它你能把内核流水线(pipeline,把多步操作重叠起来跑,见第 0 章)的状态(屏障、累加器(accumulator,一边算一边把结果累加进去的寄存器)、scratch 视图)**打包成一个对象**,免得在内核里到处撒一堆零散的局部变量。
- **`T.constexpr`**:标记编译期的内核参数,然后由 `@T.jit` 的 `.specialize(...)` 把它烘焙进去(细节看「TIRx 入门」)。

> **注意**:这一整页就一个关键词——**解析期**。这些工具弄出来的东西,你在最终的内核里是看不见的:要么被内联了,要么被展开了,要么被当成常量烘进去了。

### 数据类型与表达式:dtype 与 type 两套体系

先记住一句话:TIRx 里每个表达式都同时背着**两层类型信息**。一层管"底下到底是哪些二进制位",另一层管"逻辑上这是个什么玩意儿"。

- **低层 dtype**:就是 `PrimExpr.dtype`,管的是"是什么位 / what bits"。比如 `float32`、`float16`、`bfloat16`、`int32`、`bool`,低精度的 `float8_e4m3fn` / `float4_e2m1fn`,向量形式的 `float32x4`,还有指针 `handle`。每个 dtype 都对应一种 CUDA 类型。
- **高层 type**:标量写成 `PrimType(dtype)`,指针写成 `PointerType(PrimType(dtype), scope)`。这一层主要在你**跟指针打交道**的时候才用得上,平时基本不用管它。

这一页最该记的,是这张 **dtype → CUDA 类型映射表**(用中文重排如下):

| TIRx dtype | CUDA 类型 |
| --- | --- |
| `float32` | `float` |
| `float16` | `half` |
| `bfloat16` | `nv_bfloat16` |
| `int32` | `int` |
| `uint8` | `uchar` |
| `bool` | `signed char` |
| `float32x4` | `float4`(向量) |
| `handle` | `T*`(指针) |

向量 dtype(比如 `float32x4`)好在哪?它有两种玩法。一种是直接声明一个 `float4` 寄存器(register,每个线程私有的最快存储,见第 0 章)(`T.alloc_local((1,), "float32x4")`,然后用 `v[0]` 索引);另一种是配合 `vload`/`vstore`,把 16 字节当成**一次宽访问**一口气搬过去。这里有个点:向量 dtype 不是只能跟 `vload` 搭配——随便哪个缓冲区或标量都能带上它。

再说指针(`handle`)。缓冲区的 `data` 指针是个**不可变 / immutable**的 `Var`,所以怎么拿到它是有讲究的。常规两种办法:要么让 `alloc_buffer` 在分配的时候顺手把它定义好,要么用 `decl_buffer(data=ptr)` 复用一个现成的指针。但有一种情况得特别处理:如果你想让缓冲区指向一个**指针表达式**(比如 `T.ptx.map_shared_rank` 算出来的、另一个 cluster(一簇协作的 CTA,见第 0 章)里 CTA(线程块,见第 0 章)的共享地址),那就得先用 `T.let` 把这个表达式绑成一个指针 `Var`,再拿去用。

### 缓冲区与内存:内核里所有"内存对象"的字典

这是五页里**最厚的一页**,因为内核里几乎啥都是缓冲区。它讲清楚三件事:两套声明 API、各个内存空间(scope),还有一堆"看着像数据、其实只是元数据"的视图和标量概念。

**两个根 API**:

| API | 干什么 |
| --- | --- |
| `T.alloc_buffer(shape, dtype, scope=…)` | **真正分配新存储**(发出 `AllocBuffer` 节点);`T.alloc_shared` / `T.alloc_local` 是它带 `scope="shared"` / `"local"` 的简写 |
| `T.decl_buffer(shape, dtype, data=…)` | 在**已有指针上声明一个视图**(不分配),用来别名/重解释存储 |

**内存空间(scope)**:

| scope | 简写 | 对应内存 |
| --- | --- | --- |
| `"global"` | (默认) | 设备全局内存 |
| `"shared"` | `T.alloc_shared` | 静态共享内存(`__shared__`) |
| `"shared.dyn"` | (用 pool) | 动态共享内存(按启动参数定大小) |
| `"local"` | `T.alloc_local` | 每线程寄存器 |
| `"tmem"` | (TMEM pool) | Blackwell(NVIDIA 最新一代 GPU 架构)张量内存(TMEM,Tensor Core 专用的片上内存) |

这一页有几个**反复出现的要点**,先记在心里:

- **缓冲区说白了就是"挂在指针上的一层元数据"。** 同样逻辑上去访问 `B[i,j]`,你只要改一下 `B` 的 layout 或 `elem_offset`,生成的地址算术就完全变了(行主序是 `i*8+j`,列主序是 `j*4+i`,带偏移是 `i*8+j+64`……)。可底下那个源数据指针从头到尾压根没动过。
- **动态共享内存,整个内核只能分配一块。** 这一块你就当成一块"竞技场 / arena"。剩下所有想用动态共享内存的缓冲区,都得用 `decl_buffer(data=arena.data, elem_offset=…)` 在这块地里开视图。换句话说,写两遍 `alloc_buffer(scope="shared.dyn")` 是错的。`T.SMEMPool` 就帮你把这套"竞技场记账"自动化成了一个 bump 分配器。
- **标量 / scalar 其实就是"单元素 local 寄存器缓冲区"的语法糖。** 你写 `phase: T.int32 = 0`,跟你老老实实写 `T.alloc_local((1,), "int32")` 再用 `[0]` 索引,解析完得到的 TIRx **结构一模一样**。前者无非是让你少写一个 `[0]`。
- **`let` 和标量是两码事,别搞混。** `T.let` 是个**不可变**绑定(降下去就是一个普通 C 变量,不是数组)。正因为它不可变,算术分析器才能把它的常量界、可整除性、取值范围这些事实**带到每一个用到它的地方**,顺手帮你化简索引、砍掉越界检查、决定要不要向量化。而可变标量呢,本质上是一次内存 load,上面这些好处一样都带不过去。
- **张量内存 / TMEM 是个特例。** 它得用 warp-uniform 的 `T.ptx.tcgen05.alloc/dealloc` 显式申请和释放,而且不能直接寻址——读写只能走 `tcgen05` 的 `mma`/`ld`/`st`/`cp`。每个张量都用 `decl_buffer(scope="tmem", allocated_addr=<列偏移>, layout=…)` 开一个视图。这一整套流程,`T.TMEMPool` 都替你封装好了。

这一页末尾还有一张 **Buffer 方法表**(`data` / `ptr_to` / `vload` / `vstore` / `view` / `local` / `permute` / `access_ptr`)。这些方法基本都是**编译期**的重排或重解释:要么就改改索引算术,要么递给你一个指针,本身不会产生任何运行时操作。

### 控制流:`if`、循环家族、`while`

这一页很短,因为 TIRx 的控制流"长得跟 CUDA 差不多":Python 的 `if/else` 直接变成 CUDA 的 `if/else`,`break`/`continue` 在循环里照样能用。真正值得专门查的只有两件事:

**循环有四种风味**:

| 写法 | 含义 |
| --- | --- |
| `T.serial(n)` | 顺序循环(普通 `range` 就降成它;ptxas 仍可能自行展开) |
| `T.unroll(n)` | **完全展开**成直线语句,生成代码里没有循环 |
| `T.vectorized(n)` | 向量化循环 |
| `T.grid(*extents)` | 嵌套循环巢 |

**统一 / uniform 控制流 vs 发散 / divergent 控制流**——这是最容易踩的坑,也是这一页真正想敲打你的地方。

> **关键**:先记住一句话——**集合操作 / collective op,必须让它要同步的每一个线程"齐刷刷地"都执行到。** 比如 `T.cuda.cta_sync()` 会映射成 `__syncthreads()`,它要求线程块里**所有**线程都走到这一行。要是你把它塞进 `if wg_id == 0:` 这种只有部分 warpgroup 才进得去的分支里,那别的 warpgroup(一组 warp 的集合,常见是 4 个 warp = 128 线程)永远到不了这一行,内核就**死锁 / deadlock**(大家互相干等、谁也走不下去)了。如果你只是想让某一个 warpgroup 内部同步一下,那就改用 warpgroup 作用域的 `T.cuda.warpgroup_sync(id)`。

同样的话也适用于屏障初始化。`mbarrier`(memory barrier,放在共享内存里、让线程和异步引擎互相等待的硬件屏障)的 `.init()` 会降成一个单线程守卫(`if (threadIdx.x < 1)`)。你要是再把它套进另一个发散分支里,屏障就可能**压根没被初始化**,结果就是莫名其妙的启动失败。

还有个小技巧:如果你只是想做**表达式层面的"二选一"**(而不是真的要走某一条分支),那就用 `T.if_then_else(cond, a, b)`。它降下去就是 C 的三元运算符,**不会引入控制流发散**。

### CUDA C++/PTX 内建函数:两个直达硬件的逃生口

有时候现成的 tile(把大矩阵切出来的小方块)原语满足不了你的需求,这时候这一页给了你两条"逃生口 / escape hatch",让你直接够得着硬件:

1. **调后端内建。** 用 `T.cuda.*` / `T.ptx.*` 这两个命名空间(来自 `tvm.backend.cuda`),直接把设备内建暴露给你——同步、mbarrier、归约都有,还有 PTX 那一大家子数据搬运 / MMA(matrix multiply-accumulate,Tensor Core 上的矩阵乘加指令)(`cp_async`、`cp_async.bulk.tensor` 也就是 TMA(Tensor Memory Accelerator,专门成块搬运张量的硬件引擎)、`ldmatrix`/`stmatrix`、Blackwell 的 `tcgen05.*`、`atomic_add`、`fence` 等等)。
2. **内联原始 CUDA。** 那些连内建都没有的东西,就用 `T.cuda.func_call(name, *args, source_code=…, return_type=…)`,把一段 `__device__` 源码原样注进去,再把调用接上。

举个例子,用 warp shuffle 在 warp(一个 warp = 32 个线程的小班,锁步执行,见第 0 章)内做一次全归约,核心就一行:

```python
# __shfl_xor_sync 的 TIRx 写法,蝶式归约里逐步折半
v[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, v[0], i[0], 32, 32)
```

它会直接降成 `__shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32)`。

这一页还专门讲了**四类同步语义**。它们反复出现,而且一旦用错,通常不是给你报个错,而是静默崩坏或者死锁,所以值得单独记一记:

| 机制 | 关键语义 / 坑 |
| --- | --- |
| **Mbarrier 相位 / phase** | `try_wait(bar, phase)` 会一直卡着,直到屏障内部的相位**不等于**你传进去的 `phase`。要是你跨循环复用同一个屏障,那每次等完都得把你本地的相位跟踪器翻一下(`phase ^= 1`);不然后面的等待会立刻返回,引擎可能读到只写了一半的内存 |
| **选举 / election** | `T.ptx.elect_sync()` 选中的是**一个 warp 里的某个活跃 lane(通道,即线程在 warp 内的编号 0–31)**——它既不是 lane 0,也不是"每 CTA 一个线程"。你要是想精确收到"恰好一个线程",还得再加一道 `if warp_id == 0:` 这样的 warp 级守卫 |
| **具名 warpgroup 屏障** | `cta_sync()` 要求整个 CTA 都到齐;做了 warpgroup 特化之后,就改用 `warpgroup_sync(id)`。硬件一共给了 16 个具名屏障(ID 0–15),不同 warpgroup 得用不同的 ID(比如 `warpgroup_sync(wg_id + 10)`),免得撞到同一个硬件屏障 |
| **栅栏 / fence** | 用来给"生产者写、消费者(往往是异步引擎)读"这件事排个先后:`fence.proxy_async("shared::cta")`、`fence.mbarrier_init()`、`tcgen05.fence.after_thread_sync()` 各管一条排序边 |

> **注意**:这四类同步是 GEMM(通用矩阵乘法,GPU 上最核心的计算)和 FlashAttention 内核里的"常客"。它们管的是**异步引擎和并行线程组**,所以一旦用错,通常不会给你报错,而是直接**静默数据损坏或死锁**。这也正是作者为什么要单独写一章「Warp-Specialized 内核调试」。

---

## 19.3 建议的阅读顺序

这五页虽说是速查手册,但作者排它们的顺序其实藏着一条线索:**写内核的时候,你的关注点会一层一层往下走**。所以第一次速览,照下面这个顺序过一遍最顺:

```mermaid
flowchart LR
    P1["1. 解析器工具<br/>元编程 / 解析期"] --> P2["2. 数据类型与表达式<br/>标量 / dtype / 指针"]
    P2 --> P3["3. 缓冲区与内存<br/>内存对象 / scope / 视图"]
    P3 --> P4["4. 控制流<br/>if / 循环 / 同步守卫"]
    P4 --> P5["5. CUDA/PTX 内建<br/>直达硬件 + 同步语义"]
```

1. **解析器工具**:先搞明白"解析期能做哪些元编程",因为后面好多代码都靠它来展开和内联。
2. **数据类型与表达式**:把"标量数据"的底子打牢——dtype、向量类型、指针。
3. **缓冲区与内存**:进到"内存对象"这一层。这页分量最重,也是你回头查得最勤的一页。
4. **控制流**:有了数据和内存,再来看怎么把执行流程组织起来,尤其得记牢"集合操作必须齐刷刷到达"这个坑。
5. **CUDA C++/PTX 内建**:最后是直达硬件的逃生口和四类同步语义——写 GEMM/FA 的时候你也最常翻回这一页。

> **关键**:速览过一遍以后,就别再"从头读到尾"了。它真正的用法是:代码写到哪儿卡住了,就跳回对应那页查精确语义。

---

## 小结

本章是「TIRx 语言参考」这组页面的导读,核心就三点:

- 它是从「TIRx 入门」里抽出来的**语言速查手册**,作用是"查精确写法和语义",不是再把教程重讲一遍。
- 它把 TIRx 语言切成五块,刚好对着写内核的五个层面:**解析器工具(解析期元编程)→ 数据类型与表达式(标量/dtype/指针)→ 缓冲区与内存(内存对象/scope/视图)→ 控制流(if/循环/同步守卫)→ CUDA/PTX 内建(直达硬件 + 四类同步语义)**。
- 它的用法是**按需查**:先按上面的顺序速览一遍,在脑子里画一张地图;之后写代码碰到具体问题,再跳回对应那页。其中你回头翻得最多的是两页——「缓冲区与内存」(最厚)和「CUDA/PTX 内建」(同步语义的坑最多)。

## 延伸阅读

- 原文页面:[TIRx Language Reference — Modern GPU Programming for MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/language_reference/index.html)
- 前置教程:第二部分的「TIRx 入门 / Introduction to TIRx」和「TIRx 布局 API / TIRx Layout API」。
- 完整的 `tvm.tirx` Python API:见上游 TVM / Apache TVM 的官方文档。
- 相关实践:想看四类同步语义的实际用法,可以读 Part III 的 GEMM 内核,以及「Warp-Specialized 内核调试」那一章。

## 术语对照

| 中文 | English |
| --- | --- |
| 设备内核 | device kernel |
| 解析器工具 | parser utilities |
| 解析期 | parse-time |
| 元编程 | metaprogramming |
| 元值 | meta value |
| 内联函数 | inline function |
| 数据类型 / 标量类型 | dtype |
| 向量类型 | vector dtype |
| 指针 | handle / pointer |
| 缓冲区 | buffer |
| 内存空间 | scope |
| 视图 | view |
| 静态共享内存 | static shared memory |
| 动态共享内存 | dynamic shared memory |
| 竞技场 | arena |
| 寄存器 | register |
| 标量 | scalar |
| 不可变绑定 | let / immutable binding |
| 张量内存 | tensor memory (TMEM) |
| 控制流 | control flow |
| 统一控制流 | uniform control flow |
| 发散控制流 | divergent control flow |
| 集合操作 | collective op |
| 死锁 | deadlock |
| 内建函数 | intrinsic |
| 逃生口 | escape hatch |
| 相位 | phase |
| 选举 | election |
| 栅栏 | fence |
