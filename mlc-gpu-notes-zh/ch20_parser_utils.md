# 第 20 章 · 解析器工具(Parser utilities)

> 原文:[Parser utilities](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/language_reference/cuda/parser_utils.html)

> **本章要点(TL;DR)**
>
> - 本章讲四个工具:`T.meta_var`、`@T.inline`、`@T.meta_class`、`T.constexpr`。它们有个共同点:都在**解析时 / parse time** 干活。也就是说,它们出力的那一刻,是「TVMScript 翻译成 TIRx IR」的瞬间,而不是 GPU 真正跑代码的时候。说白了,它们就是 TIRx 的「元编程 / metaprogramming」入口。
> - **`T.meta_var(x)`**:把一个**你用 Python 当场算好的值**直接塞进 IR,当编译期常量使。最典型的玩法,是让一个普通的 Python `for` 循环在解析阶段就被**展开 / unroll**。
> - **`@T.inline`**:定义一个特殊的函数,它会在**每个调用点就地展开**。换句话说,生成的 IR 里压根看不到函数调用,只剩替换进去的函数体。它遵循 Python 的 LEGB 作用域和延迟绑定 / late binding 规则。
> - **`@T.meta_class`**:给一个普通 Python 类贴个「解析期元对象」的标记。它的实例字段能装下 buffer、标量这些 IR 对象,于是你就能把一组相关的分配和状态**打包成一个对象**,不用再拎着一堆零散的局部变量到处传。
> - **`T.constexpr`**:标记一个**编译期的 kernel 参数**,由 `@T.jit` 的 `.specialize(...)` 在编译时把它「烤」进 kernel。本章只点个名,细节看《TIRx 入门》。

> **前置知识**:读这一章前,最好先懂 TIRx / TVMScript 的基本写法,以及 buffer、SMEM(共享内存)、scope(内存作用域)这些概念。没把握的话,先翻一下 [第 0 章 · 极简入门](./ch00_gpu_ml_primer.md),以及第 9 章 TIRx 基础。本章会默认你已经认识这些词。

---

## 一、先搞清楚:什么叫「解析期工具」

要听懂这一章,先得在脑子里摆一条时间线。你写好一段 TIRx kernel,到它真正在 GPU 上跑起来,中间得过三道关:

```mermaid
flowchart LR
    A["你写的 TVMScript<br/>(带 @T 装饰的 Python)"] -- "① 解析 (parse)" --> B["TIRx IR<br/>(结构化中间表示)"]
    B -- "② 下降 (lower) / 编译" --> C["GPU 上执行的机器码<br/>(SASS / PTX)"]
    C -- "③ 运行 (runtime)" --> C
```

- **① 解析阶段 / parse time**:解析器读你写的那段 Python(也就是 TVMScript),把它**翻译成 TIRx IR**。注意,这一步是在 Python 进程里跑的——你的代码这时候是被当成数据来处理的。
- **② 下降 / 编译阶段**:`LowerTIRx` 这类 pass 上场,把 IR 一步步落实成具体的 GPU 指令。
- **③ 运行阶段**:代码真正在 GPU 上跑。

> **关键**:本章这四个工具,**全都活在阶段 ①**。它们不会变成 GPU 上的任何一条指令。它们只在「Python 翻译成 IR」这一瞬间动手——要么把一个 Python 值塞进 IR,要么趁翻译的时候提前把循环或函数展开掉,要么把一堆状态打成一个包。

说白了,这就是「元编程」:**拿 Python 这门宿主语言,去操控、去生成另一门语言(TIRx IR)的代码**。你只要记住一句话——「这些事全发生在解析器里」,后面四节就都顺了。

那它们各自治什么病?主要是下面这三类:

| 工具 | 解决的痛点 |
| --- | --- |
| `T.meta_var` | 想把 Python 算出来的常量直接用进 IR,又不想多写一个没用的临时变量;想让循环在解析期展开 |
| `@T.inline` | 想把一段重复逻辑抽成函数复用,又不希望 IR 里真出现一次函数调用 |
| `@T.meta_class` | kernel 里状态太多(屏障、累加器、scratch view……),不想把十几个局部变量一路穿下去 |
| `T.constexpr` | 想要一个编译期就固定下来的 kernel 参数(如 tile 尺寸),让编译器据此特化 |

---

## 二、`T.meta_var` — 把 Python 值 inline 进 IR

### 2.1 它在做什么

`T.meta_var(x)` 其实就是在跟解析器交代一句:**「`x` 是个编译期的 meta 值,你直接把它 inline 进 IR 就行,别当成普通的『脚本变量』去解析。」**

这里的 `x`,是你在 **Python 这一层**就已经算好的值,比如一个 Python `int`。那加不加 `T.meta_var`,到底差在哪?对比一下你就懂了:

- **不加**:解析器可能把 `n = 4` 看成 IR 里的一次变量赋值,于是真给你生成一个 IR 变量节点。
- **加了** `T.meta_var(4)`:这下 `n` 就只是个 Python 整数 `4`。后面哪儿用到 `n`,解析器就把 `4` 这个字面值直接填进去——IR 里根本没有 `n` 这么个变量。

```python
n = T.meta_var(4)              # n 就是 Python int 4,会被直接 inline
for j in range(n):             # 上界是 meta 值 -> 在解析期被「展开」
    acc[0] = acc[0] + A[tx, j]
```

### 2.2 最重要的副作用:驱动循环展开

`T.meta_var` 最值钱的本事,就是能让循环在**解析期直接被展开**。

为什么?关键全在 `range(n)` 里那个 `n` 身上。`n` 是个**实打实的 Python 值**(不是 IR 变量),所以 `for j in range(n)` 不过是一个最普通的 **Python `for` 循环**。解析器翻译的时候,会老老实实把这个 Python 循环**跑一遍**:转一圈,就吐一条对应的 IR 语句。这么一圈一条,循环就被摊平了。

还拿上面那段代码举例,它等价于解析器生成了这么一段 IR(意思上是这样):

```python
acc[0] = acc[0] + A[tx, 0]
acc[0] = acc[0] + A[tx, 1]
acc[0] = acc[0] + A[tx, 2]
acc[0] = acc[0] + A[tx, 3]
```

下面这张图,把「普通的运行期循环」和「meta 值驱动的解析期展开」并排一放,差别一眼就出来了:

```mermaid
flowchart LR
    subgraph A["普通 IR 循环变量"]
        A1["for j in range(N)<br/>N 是 IR 变量"] --> A2["IR 里保留一个<br/>真实的 for 循环节点"]
        A2 --> A3["运行期在 GPU 上<br/>循环执行 N 次"]
    end
    subgraph B["meta_var 驱动展开"]
        B1["n = T.meta_var(4)<br/>n 是 Python int"] --> B2["for j in range(n)<br/>就是 Python for"]
        B2 --> B3["解析器跑一遍 Python 循环<br/>逐圈生成 IR 语句"]
        B3 --> B4["IR 里没有循环,<br/>只剩 4 条展开后的语句"]
    end
```

> **关键**:一个 `for` 会不会在解析期展开?判断起来特别简单——就看它的**循环上界(还有循环里用到的那些值)是不是 meta 值**。而 `T.meta_var` 干的,正是把一个 Python 值「升格」成 meta 值的那个开关。上界一旦是 meta 值,这个 `for` 就退回成一次普通的 Python 迭代,解析器照着它一条一条把语句铺出来就完事了。

### 2.3 为什么这么设计

- **省掉那种「用一次就扔」的局部变量**:很多时候你不过是想用一个 Python 算出来的常量(比如 `tile_k = 64`,这里的 tile 就是大矩阵切出来的小方块,见第 0 章),犯不着在 IR 里专门给它立一个变量节点。`T.meta_var` 能让这个值不留痕迹地融进 IR。
- **把 Python 和 IR 的元编程打通**:它把 Python 算出来的结果,干干净净地注入到 IR 的生成过程里。循环展开只是最常见的一种用法而已。说到底,只要你想让「解析器把某个值当成已知常量来处理」,都能用它。

---

## 三、`@T.inline` — 内联函数

### 3.1 它在做什么

先给个直觉:`@T.inline` 让一个函数「写的时候像函数,出来的时候像手写」。它修饰的函数,**函数体会在每个调用点就地铺开(也就是 inline)**。换句话说,**生成的 IR 里一个函数调用都看不到**,看到的只是把函数体「抄」过去、再把形参换成实参之后的那几行语句。

```python
@T.inline
def add_into(acc, x):
    acc[0] = acc[0] + x

add_into(acc, A[tx, j])        # 内联后 -> acc[0] = acc[0] + A[tx, j]
```

调一次 `add_into(acc, A[tx, j])`,效果跟你亲手写 `acc[0] = acc[0] + A[tx, j]` 一模一样:形参 `acc` 换成实参 `acc`,形参 `x` 换成实参 `A[tx, j]`,然后整个函数体铺到调用点上,就这么回事。

### 3.2 作用域规则:LEGB + 延迟绑定

原文专门强调:`@T.inline` **遵循 Python 的 LEGB 词法作用域,而且采用延迟绑定 / late binding**。这俩词听着唬人,其实都是 Python 里现成的规矩,一个个拆开看就明白了:

- **LEGB**:就是 Python 找名字的标准顺序——Local(局部)→ Enclosing(闭包外层)→ Global(模块全局)→ Built-in(内建)。内联函数体里用到的名字,就照这个顺序一层层往外找。
- **延迟绑定**:名字是**真正用到它的那一刻**才去解析,不是定义的时候就钉死。
- **形参会盖住外层的同名变量(shadowing)**:正因为走 LEGB,内联函数的**形参名**要是碰巧跟外层某个变量撞名了,那在函数体里,这个名字指的就是**形参**(Local 这一层级优先级最高),外层那个同名的就被挡在外面了。

> **注意**:这条规矩在实战里很要紧。举个例子:外层有个变量叫 `x`,你的内联函数形参也叫 `x`,那函数体里的 `x` 用的是**传进来的实参**,不是外层那个 `x`。这正是它和朴素宏(macro)的区别——宏只是把函数体原封不动粘到调用点,一不留神就会误碰外层的同名变量。`@T.inline` 有作用域规则兜底,行为更像一个真函数,只是省掉了运行期的调用开销而已。

下面这张图,把「`@T.inline`」和「真实的函数调用」摆一块儿看:

```mermaid
flowchart TD
    C["写法相同,都长得像函数<br/>add_into(acc, A[tx, j])"]
    C --> I["@T.inline 版<br/>解析期就地展开"]
    C --> R["真实函数调用版<br/>IR 里保留一个 call 节点"]
    I --> IO["IR: acc[0] = acc[0] + A[..]<br/>(无 call,无栈帧)"]
    R --> RO["IR: call add_into(acc, A[..])<br/>(运行期有调用开销)"]
```

### 3.3 为什么这么设计

- **既能复用,又零开销**:你想把重复的逻辑抽成函数,让代码好读、好复用;可又不想在底层 kernel 里背上函数调用的开销(在 GPU kernel 里,函数调用往往不便宜,还碍着编译器优化)。`@T.inline` 正好两头都顾上——写着像函数,出来像手写。
- **比宏靠谱**:它不是死板的文本替换,而是带着 Python 作用域语义的展开。所以同名遮蔽它能处理对,宏式展开里那些经典的坑也就绕过去了。

---

## 四、`@T.meta_class` — 解析期状态对象

### 4.1 它在做什么

先说它治什么病:kernel 里东西一多,你就得拎着一大把零散的局部变量满世界跑。`@T.meta_class` 就是来帮你「装箱」的。

具体来说,它给一个**普通 Python 类**贴个标记:这个类的**实例是解析期的 meta 值**。它最关键的能耐是——**实例的字段(field)能装下 buffer(一块带类型和形状的连续内存区,见第 0 章)、标量这些 IR 对象**。这么一来,你就能把一组相关的分配(allocation)和状态**打包进一个对象**,然后在 kernel 体里像用普通 buffer 那样,直接用它的字段。

```python
@T.meta_class
class State:
    def __init__(self, smem):
        self.acc = T.alloc_local([1], "float32")                      # 字段持有一块 local buffer
        self.buf = T.decl_buffer([64], "float16", smem, scope="shared.dyn")  # 字段持有一个 SMEM 视图

s = State(smem.data)
s.acc[0] = T.float32(0.0)      # 像普通 buffer 一样直接用它的字段
# ... s.buf[i] ...             # 同理
```

仔细看 `State`,它就是个再普通不过的 Python 类:`__init__` 里调用 TIRx 的分配 API(`T.alloc_local`、`T.decl_buffer`),把结果存进 `self.acc`、`self.buf`。那 `@T.meta_class` 这个装饰器干了啥?它让解析器把 `State` 的实例当成「解析期的状态容器」。于是实例的字段就能装得下 IR 对象,在 kernel 体里也能被正确引用。

### 4.2 为什么这么设计:别再「穿线」一堆局部变量

一个真正快的 kernel,流水线(把搬数据和算数据错开重叠起来跑,见第 0 章)上的状态多得吓人:

- 好几个 **mbarrier(屏障,让线程互相等齐、对上步调的同步原语)**;
- 一堆**累加器(accumulator,边算边把结果累加进去的那块寄存器/buffer)**;
- 各种 **scratch 视图 / 中间 buffer(临时用来暂存中间结果的草稿空间)**;
- 还有阶段计数、phase 标志这类标量。

不打包的话,你就得把这十几样东西全当局部变量,一路从外层「穿」到内层,再从一个内联函数传给下一个内联函数。结果呢?参数列表越拖越长,又容易出错,又难读。`@T.meta_class` 的思路是:这些**逻辑上本来就是一伙的**状态,干脆全收进一个对象里。

```mermaid
flowchart TD
    subgraph Before["不打包:穿线一堆局部变量"]
        b0["kernel 体"]
        b0 --> b1["acc"]
        b0 --> b2["buf"]
        b0 --> b3["barrier0"]
        b0 --> b4["barrier1"]
        b0 --> b5["phase"]
        b1 & b2 & b3 & b4 & b5 -.->|"逐个传进各内联函数"| bx["参数列表越来越长"]
    end
    subgraph After["打包:一个 meta_class 实例"]
        a0["kernel 体"] --> s["s = State(...)"]
        s --> sa["s.acc"]
        s --> sb["s.buf"]
        s --> sc["s.barriers"]
        s --> sp["s.phase"]
        s -.->|"只传一个对象"| ay["接口干净"]
    end
```

> **关键**:`@T.meta_class` 给你的是**条理**,不是性能。它本身不会冒出任何运行期对象——实例只活在解析期。它治的是「kernel 代码怎么组织」这个病:把散落一地的流水线状态拢成一个有名字的整体,让 kernel 体读起来像在操作一台清清楚楚的状态机,而不是在一堆裸 buffer 里转晕。

### 4.3 和 `@T.inline` 的搭配

`@T.meta_class` 和 `@T.inline` 是天生一对。你可以写一组 `@T.inline` 的「方法式」辅助函数,让它们都接收同一个 `State` 实例(比如 `add_into(s, ...)`、`advance(s)`)。内联展开之后,这些函数既复用了逻辑,又共享了同一份状态,接口还特别干净。

---

## 五、`T.constexpr` — 编译期 kernel 参数

`T.constexpr` 用来标记一个**编译期的 kernel 参数**。这个参数会被 `@T.jit` 的 `.specialize(...)` 在编译时「烤」进(bake in)kernel。

原文在本章只点了个名,扔下一句话:**完整细节看《TIRx 入门(Introduction to TIRx)》**。不过照着本书一贯的路子,我们先把它的定位讲清楚:

- 普通的 kernel 参数,是**运行期**才定的——每次调用都能传不同的值。
- 而打了 `T.constexpr` 标记的参数,在 **`.specialize(...)` 那一刻就定死了**。这相当于跟编译器说:「在这个特化(specialize)出来的 kernel 版本里,这个值就是个常量。」
- 既然成了编译期常量,编译器就敢放手做更狠的优化:常量折叠、循环展开、砍掉分支,样样都行;还能拿它去驱动布局和调度上的决策。这也正是它跟本章前几个工具气味相投的地方——都是「把某个值在更早的阶段就钉死,好让后面生成的代码更优」。

> **注意**:`T.constexpr` 起作用的时机,跟前三个工具不太一样。`T.meta_var`、`@T.inline`、`@T.meta_class` 都在**解析期**动手;而 `T.constexpr` 挂在 `@T.jit` 的 **`.specialize(...)`** 这套编译期特化机制上。但骨子里的精神是一回事——「决策能提前就提前」,只是落地的位置不同罢了。具体语义以《TIRx 入门》为准,本章只是把它收进「解析器 / 编译期工具」这一家子里点个名。

---

## 六、四个工具的横向对比

把这四件工具塞进同一张表,谁负责干啥就一目了然了:

| 工具 | 作用对象 | 生效时机 | 核心效果 | 典型场景 |
| --- | --- | --- | --- | --- |
| `T.meta_var(x)` | 一个 Python 值 | 解析期 | 把值 inline 进 IR;让 `for` 在解析期展开 | 用 Python 常量驱动循环展开 |
| `@T.inline` | 一个函数 | 解析期 | 函数体在每个调用点就地展开,IR 里无 call | 抽取可复用逻辑且零调用开销 |
| `@T.meta_class` | 一个 Python 类 | 解析期 | 实例字段可装 buffer/标量,打包状态 | 聚合屏障/累加器/scratch 等流水线状态 |
| `T.constexpr` | 一个 kernel 参数 | 编译期(`.specialize`) | 把参数烤成编译期常量 | tile 尺寸等编译期固定的配置 |

这四个工具其实共用一条思路,一句话就能串起来:

> **能提前确定的东西,就趁早定下来**——不管是值(`meta_var` / `constexpr`)、函数展开(`inline`),还是状态怎么组织(`meta_class`)。这样既让 IR 更干净、生成的 GPU 代码更快,又不用丢掉 Python 这门宿主语言的元编程表达力。

---

## 小结

- 本章这四个工具,都是 TIRx 的**解析期 / 编译期元编程入口**。它们干的都是「在 TVMScript → TIRx IR 这条翻译链上做文章」,本身不对应 GPU 上的任何一条运行期指令。
- **`T.meta_var(x)`**:把 Python 算出来的值升格成编译期 meta 值,再 inline 进 IR。最常见的效果,是让 `for j in range(n)` 这类循环在解析期就被**展开**成一条条铺平的语句。
- **`@T.inline`**:函数体在每个调用点就地展开,IR 里看不到函数调用。它遵循 Python 的 **LEGB 词法作用域和延迟绑定**,形参能正确盖住外层的同名变量,所以比朴素的宏更靠谱。
- **`@T.meta_class`**:把普通 Python 类的实例变成「解析期状态容器」,字段能装下 buffer / 标量。你可以拿它**打包**屏障、累加器、scratch 视图这些流水线状态,省得在 kernel 体里拎着一堆零散局部变量到处穿。它和 `@T.inline` 一搭,就能写出特别干净的「带状态辅助函数」。
- **`T.constexpr`**:标记编译期的 kernel 参数,由 `@T.jit` 的 `.specialize(...)` 烤进 kernel,细节看《TIRx 入门》。
- 全章就一条主心骨:**能提前定的值、展开、状态组织,就趁早定下来**。这样既留住了 Python 的元编程能力,又让生成的 IR 和 GPU 代码更精简、更快。

## 延伸阅读

- 原文页面:[Parser utilities — Modern GPU Programming for MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/tirx_guide/language_reference/cuda/parser_utils.html)
- `T.constexpr` 与 `@T.jit` 的 `.specialize(...)` 细节:见本书第二部分「TIRx 入门(Introduction to TIRx)」。
- 元编程与循环展开的更广背景:可结合本书 TIRx 语言参考的其余章节一起看。

## 术语对照

| 中文 | English |
| --- | --- |
| 解析期 / 解析时 | parse time |
| 元编程 | metaprogramming |
| 元值 / meta 值 | meta value |
| 内联(就地展开) | inline |
| 循环展开 | unroll |
| 词法作用域 | lexical (LEGB) scoping |
| 延迟绑定 | late binding |
| 遮蔽(同名覆盖) | shadowing |
| 编译期常量 | compile-time constant |
| 特化 | specialize |
| 累加器 | accumulator |
| 屏障 | mbarrier / barrier |
| 中间暂存(视图) | scratch (view) |
