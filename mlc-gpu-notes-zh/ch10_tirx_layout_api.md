# 第 10 章 · TIRx Layout API

> 原文:[TIRx Layout API](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tirx_layout_api/index.html)

> **本章要点(TL;DR)**
>
> - 上一章我们在纸上发明了一套记号来描述布局。这一章干的事说白了就一句话:把那套纸面记号原样搬进编译器,变成你能 new 出来的对象。这样一来,你在纸上画的布局,跟内核里敲的代码,几乎就是同一行字。
> - 要记的东西不多,就三个:`TileLayout`(仿射布局)、`SwizzleLayout`(拿 XOR 给共享内存换序)、`ComposeLayout`(把 swizzle 叠到 tile 上面)。
> - 整套设计的魂就一句话:**一个布局,把一个逻辑坐标,映射到一个甚至好几个物理坐标**。方向记牢——是「逻辑 → 物理」,而且物理那头是一个集合。为什么非得是集合?因为数据要广播(broadcast)的时候,只有"一对多"才说得通。
> - `TileLayout` 分三步走,有点像填快递单:**shard / 分片**(写作 `S[...]`,公式里记成 `D`)先把基础位置定下来;**replica / 复制**(写作 `R[...]`,记成 `R`)再把它抄送到别的地方;**offset / 偏移**(记成 `O`)最后整体挪个位置。一句话公式:`L(x) = { D(x) + r + O | r ∈ R }`。
> - 布局里的轴(axis)可不是匿名的维度编号,每一条都是有名有姓的真实硬件坐标,比如 `laneid`、`warpid`、`TLane`、`TCol`、`m`、`Bank`。另外配了几个现成的构造器(`tmem_datapath_layout`、`tcgen05_atom_layout`、`wg_local_layout`),把高频的硬件布局打包好让你直接拿来用——但别紧张,它们吐出来的还是普通 `TileLayout`,没藏什么黑魔法。

> **前置知识**:读这一章前,最好先懂第 3 章的布局记号(shape/stride 那一套)、第 9 章的 TIRx 基础,以及 warp / lane、TMEM、swizzle 这几个词。没把握的话,先翻一下 [第 0 章 · 极简入门](./ch00_gpu_ml_primer.md),以及第 3、9 章。本章会默认你已经认识这些词。

---

## 10.1 为什么需要 Layout API:让记号变成对象

先把上一章那套记号在脑子里过一遍。它就三块拼起来:一个 tile(大矩阵切出来的小方块)形状,一组挂在具名轴上的步长(stride),外加一个可选的复制项。那个复制项是专门留给"被拷贝、而不是被切开"的数据用的。这一章要干的,就是把这套记号坐实成编译器真能跑的 API。

好处一句话就能说明白:**纸上写的,跟代码里写的,一模一样**。比方说你在文档里写下这么一行:

```python
S[(128, 256) : (1@TLane, 1@TCol)]
```

这一行可不是写给人看的说明,它真的会**构造出一个 `TileLayout` 对象**。这个对象能挂(attach)到 buffer 身上。一旦挂上去,凡是碰这个 buffer 的 tile 操作,都能直接从布局里读出"这块数据该摆在硬件的哪个角落"。

好处在哪?**摆放规则只写一次、只检查一次,后面编译器反反复复地用。**你不用在每个算子里都把"每个元素住哪儿"重抄一遍。

那布局是啥时候挂上去的?两个时机:要么从内存池里分配的时候,要么声明 buffer 的时候。

```python
pool.alloc(shape, dtype, layout=layout)            # 从内存池分配时附带布局
T.decl_buffer(shape, dtype, scope=scope, layout=layout)  # 声明 buffer 时附带布局
```

从这一刻起,buffer 就自己"背着"物理摆放信息到处走了。tile 操作再也不用一遍遍啰嗦每个元素在哪。

所有布局对象都住在同一个模块里:

```python
from tvm.tirx.layout import (
    TileLayout, SwizzleLayout, ComposeLayout,   # 三个核心布局类
    S, R,                                        # shard / replica 记号构造器
    laneid, warpid, tid_in_wg,                   # 常用具名硬件轴
    TLane, TCol, m,                              # TMEM 轴 与 默认内存轴
    tcgen05_atom_layout, tmem_datapath_layout,   # 现成布局构造器
)
```

> **关键**:整套 API 就围着一个念头转——**一个逻辑下标,对应的未必是单个物理地址,而是一组挂在具名轴上的物理坐标,也就是一个集合**。大多数时候这集合里就一个元素。可一旦用上复制,同一个逻辑元素就会在好几个物理位置同时露面。布局之所以拆成 shard / replica / offset 三块,正是为了把这件事支撑起来:shard 把元素摆好,replica 把它拷到别处,offset 再整体平移一下。

---

## 10.2 先看几个例子(Layouts by Example)

别急着抠机制,先看几个典型例子,把手感找到再说。

**例 1:TMEM(Blackwell 上专给张量核用的特种内存,见第 0 章)里的累加器(accumulator,矩阵乘里不断累加部分和的那块寄存器/内存)——只摆放,不复制。**

```python
acc = TileLayout(S[(128, 256) : (1@TLane, 1@TCol)])
```

这行怎么读?很直白:逻辑的行扔到 `TLane`,逻辑的列扔到 `TCol`。顺嘴提一句命名:TMEM 那一章里硬件坐标叫 Lane 和 Col,到了 TIRx 记号里就改写成 `TLane`、`TCol`,其实是同一个东西。

**例 2:块缩放(block-scaled)MMA(matrix-multiply-accumulate,张量核上的矩阵乘累加指令)的缩放因子布局——这回用上了复制。**

```python
scale_factor_layout = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)] + R[4 : 32@TLane]
)
```

怎么读?shard 先在 TMEM 里放下一个 32 行的组。replica 接着以 32 个 lane(通道,即 warp 内的线程编号 0–31,见第 0 章)为间隔,把这个组重复 4 份。这么一来,这个 32 行的组就把 128 个 lane 的整片 TMEM 铺满了。

**例 3:张量核(Tensor Core,GPU 上专做矩阵乘的硬件单元)寄存器片段(fragment,一个 tile 分散存到各线程寄存器里的那一小片)——散落在 lane 和 warp(线程束,一个 warp = 32 个线程的小班,见第 0 章)上。**

```python
frag = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
)
```

这里有两点得留个心。第一,**同一条物理轴可以出现不止一次**——你看,这里就有两个不同的 iter 都往 `laneid` 上塞数据。第二,**要是你没写明往哪条轴放**(例子里最后那个光秃秃的 `1`),那它默认就落到内存轴 `m` 上。

**例 4:真实内核里,常见的硬件布局一般直接调构造器拿。**

```python
acc = tmem_datapath_layout("D", 128, 256)
ld  = tcgen05_atom_layout("32x32b", (128, 64), "float32")
```

这些构造器返回的依然是**普通的 `TileLayout` 对象**。它们不过是替你把常用布局打了个包,不是另起炉灶搞出来的新东西。所以你想干啥都行:检视(inspect)它返回的布局、把它跟别的布局拼起来,或者碰上特别古怪的形状,索性退回去手写底层的 `S[...]` / `R[...]`。

---

## 10.3 交互演示(Interactive Demo)要点

原文在这儿放了个交互式演示。你可以挑一个预置布局,改改逻辑形状,改改 `S` / `R` 项,再选个 dtype 和 swizzle 模式,然后**点中一个元素,看它到底落到了哪个、或者哪几个物理坐标上**。

可惜这是个交互组件,静态笔记里搬不过来。不过没关系,下面这张"数据流"图就把它在背后干的活儿概括清楚了——而且这张图基本就是后面整套 API 的精确版,值得你记住:

```mermaid
flowchart LR
    A["逻辑元素 x<br/>(进入布局)"] --> B["1. 展平 flatten<br/>(行主序 row-major)"]
    B --> C["2. 按 shard extents 切分<br/>得到 c0, c1, ..."]
    C --> D["3. 各分量乘步长累加到具名轴<br/>ck*sk @ ak,再加 offset"]
    D --> E{"有 replica?"}
    E -- "无" --> F["单一物理坐标<br/>{...}"]
    E -- "有" --> G["枚举所有副本组合<br/>得到坐标集合 {...}, {...}"]
```

> **注意**:这个演示最值钱的地方,就是它把一件事**画给你看了**:同一列访问,不加 swizzle 的时候全挤在一个 bank 里;一旦配上对路的 swizzle,立马就散到不同 bank 上去了。有条件的话,强烈建议去原书亲手玩一玩。

---

## 10.4 `TileLayout`:主力仿射布局

`TileLayout` 是你最常打交道的布局对象,它属于**仿射(affine)**那一类——说白了就是"乘个步长再加起来"这种规规矩矩的变换。它的写法跟正文记号完全是一套:

```python
TileLayout(S[shape : strides])                                   # 仅 shard
TileLayout(S[shape : strides] + R[replica_shape : replica_stride])  # shard + replica
TileLayout(S[shape : strides] + R[...] + offset)                 # shard + replica + offset
```

- `S` 项就是 **shard spec**(分片规格),你可以一句话读出来:"拿一个这种形状的逻辑 tile,按这些步长,把它摊到具名轴上去"。
- 要是某个值得同时出现在好几个地方,就拿 `R` 项把 shard 扩一扩。
- 最后还能再叠一个可选的 offset。

再往下挖一层:这几个部件底层都靠一种叫 **iter** 的东西来表示。一个 iter 就是一个三元组:

```
(extent, stride, axis)
```

你可以把一个 iter 想象成"在某条具名轴上,按固定步长走一趟"。这三个数:`extent` 是走几步(也就是有多少个位置),`stride` 是每步迈多远,`axis` 是在哪条硬件坐标上走。下面咱们把 shard / replica / offset 三块挨个拆开看。

### 10.4.1 Shard(分片,记作 `D`)

shard 就是 `S[...]` 写出来的那一块。它干的活儿是:把逻辑下标摊到一个或几个 iter 上**切开**,再算出**基础物理坐标**。看个例子:

```python
S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
```

这里有 4 个 shard iter,extent 依次是 `8, 2, 4, 2`。它们的步长把数据分别送到 `laneid`、`warpid`、又一次 `laneid`、还有默认内存轴 `m` 上。

别觉得陌生,这其实就是你平时那套**"形状-步长"规则的加强版**。唯一不一样的地方是:步长现在挂到了**具名硬件轴**上,而不再是挂在一个扁平的内存地址上。

### 10.4.2 Replica(复制,记作 `R`)

replica 说的是同一个逻辑元素的**额外物理副本**。它有个要命的特点:**replica iter 跟逻辑下标一点关系都没有**。换句话说,不管你访问的是哪个逻辑元素,replica 都照着同一个套路在硬件空间里多铺几份。比如:

```python
R[2 : 4@warpid]   # 在 warpid 轴上,生成两个相隔 4 个 warp 的副本
```

> **关键**:别把复制看成图省事的小聪明,它是在**老老实实地照搬硬件本来的样子**。有些数据天生就得在 warp、lane 或者内存区域之间广播(broadcast)出去。而"逻辑→物理"这个方向,恰好天生就能把这事说清楚——因为它本来就允许一个逻辑元素对应一组物理坐标。

### 10.4.3 Offset(偏移,记作 `O`)

offset 就是一个固定坐标,它会加到**每一个**结果上。比如:

```python
5@warpid   # 把整个摆放在 warpid 轴上整体平移 5
```

offset 通常拿来干这么几件事:把 tile 挪到指定的起点;给某个独占用途圈一块地方;或者描述一个"紧贴在另一个 tile 后头"的 tile(俩人共用同一块资源)。

### 10.4.4 三部分如何协同

一个布局会**按固定的顺序**走这三步,次序乱不得:

1. **shard** 先把基础坐标算出来;
2. **replica** 再把这个坐标抄成零个或多个额外副本;
3. **offset** 最后把每个坐标都挪一挪。

对逻辑坐标 `x`,最终结果就是:

```
L(x) = { D(x) + r + O | r ∈ R }
```

这公式不用背,记住"定位 → 抄送 → 平移"这三步就够了。要是没有 replica,`R` 里就只剩一个零偏移,结果是个**只装一个元素的集合**;要是有 replica,那每个副本位置都会摊上一个坐标。一个写全乎的布局长这样:

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]  # shard:摆放逻辑 tile
    + R[2 : 4@warpid]                                     # replica:在 4 个 warp 外建第二份副本
    + 5@warpid                                            # offset:整体平移到 warpid=5 起步
)
```

从左往右读就行:shard 把逻辑 tile 摆好,replica 在隔着 4 个 warp ID 的地方又放一份副本,offset 把整体挪到从 `warpid = 5` 起步。要是你手头已经有现成的 iter 对象,也可以直接拼:

```python
TileLayout.from_iters(shard, replica, offset)
```

不过大多数代码还是爱用 `S[...]` / `R[...]` 记号,因为它读起来更像数学公式,一眼就懂。

---

## 10.5 具名轴(Named Axes):名字本身就是语义

前面一直在念叨"具名轴",这里专门掰开讲讲。布局里的轴**不是匿名的维度编号**,每一条都对应一个真实的硬件坐标,或者编译器层面的某个摆放坐标。常见的轴有这么几类:

| 类别 | 轴名 | 含义 |
| --- | --- | --- |
| 网格轴(grid) | `bx`, `by`, `bz` | 把工作分布到不同 CTA(cooperative thread array,即一个 block,见第 0 章)上 |
| 集群轴(cluster) | `cbx`, `cby`, `cbz` | 在一个 CTA cluster(一组能协同的 block 集群)内部摆放工作 |
| 线程轴(thread) | `tx`, `warpid`, `laneid`, `wgid`, `tid_in_wg`, `wid_in_wg` | 描述在 CTA 或 warpgroup(几个 warp 凑成的协作组)内部的归属 |
| 内存轴 | `m` | 默认的线性内存轴(寄存器槽常用它) |
| 二维便签 | `P`, `F` | 用于二维 scratchpad 式摆放 |
| 共享内存 | `Bank` | 命名共享内存的 bank |
| TMEM 轴 | `TLane`, `TCol` | TIRx 对 TMEM 的 Lane / Col 坐标的命名 |

> **关键**:**轴的名字本身就是布局的一部分**。为啥这么强调?因为**两个数值一样的坐标,硬件含义可能差着十万八千里**:`1@tx` 不等于 `1@tid_in_wg`,`1@laneid` 也不等于 `1@TLane`。布局把这层意思明明白白写死,你就不用再靠上下文去猜"这一维到底指啥"了。

---

## 10.6 前向映射(Forward Mapping):`layout.apply()`

所谓"求值一个布局",意思就是:你扔给它一个逻辑坐标,它给你算出对应的物理落点。要用的 API 是:

```python
layout.apply(*coord)
```

- 没复制的时候,你拿到**一个坐标字典**;
- 有复制的时候,你拿到**一组坐标字典**。

坐标字典说白了就是把轴名映到一个整数位置,长这样:

```python
{"laneid": 7, "warpid": 2, "m": 1}
```

整个求值就四步,跟前面那张数据流图一一对得上:

1. **展平(flatten)**:先把逻辑坐标按**行主序**压成一个一维下标。对形状 `(S0, S1, ..., Sr-1)` 里的坐标 `(x0, ..., xr-1)`,公式是:

   ```
   flat = x0*S1*S2*...*Sr-1 + x1*S2*...*Sr-1 + ... + xr-2*Sr-1 + xr-1
   ```

2. **切分(split)**:把上一步那个一维下标,按 shard 的 extents `(e0, ..., en-1)` 重新切成几段,得到分量 `c0, ..., cn-1`(依然是行主序)。

3. **累加(accumulate)**:每个分量乘上自己的步长,加到对应的轴上去。具体讲,要是第 `k` 个 shard iter 是三元组 `(ek, sk, ak)`,那分量 `ck` 就往轴 `ak` 上贡献 `ck*sk`。同一条轴上的贡献全加起来,最后再补上 offset。

4. **复制(replica)**:把 replica iter 施加上去。每个 replica iter 都贡献一个跟逻辑坐标无关的额外偏移;要是有好几个 replica iter,布局就把它们所有的组合都枚举出来。

> **注意**:这里藏着一个特别好用的推论——**布局压根不用把输入形状写死**。它只认一件事:逻辑 tile 的元素总数,得等于 shard extents 的乘积。这个数一对上,光靠"展平 + 切分"就能把整个映射定义得清清楚楚。正是这一点,让布局变得相当通用。

---

## 10.7 案例 A:张量核寄存器 tile

光说不练假把式,咱们拿前面的公式跑一个完整的例子。假设有个逻辑 `(8, 16)` 的 tile,要分到 2 个 warp(每个 warp 32 个 lane)上,每个 lane 自己拿一小段寄存器(register,每个线程私有、最快的存储,见第 0 章)片段。寄存器槽就用默认内存轴 `m` 来表示。布局长这样:

```python
layout = TileLayout(
    S[(8, 2, 4, 2) : (4@laneid, 1@warpid, 1@laneid, 1)]
    + R[2 : 4@warpid]
    + 5@warpid
)
```

随便挑个逻辑元素 `(i, j)`,咱们按四步走一遍:

**第 1 步 行主序展平**:`flat = 16*i + j`

**第 2 步 按 shard extents `(8, 2, 4, 2)` 切分**:

```
c0 = i
c1 = floor(j / 8)
c2 = floor(j / 2) mod 4
c3 = j mod 2
```

**第 3 步 累加到各轴**(`4@laneid, 1@warpid, 1@laneid, 1` 对应 `m`):

```
laneid = 4*c0 + c2
warpid = c1
m      = c3
```

**加上 offset `5@warpid`**:

```
laneid = 4*i + floor(j/2) mod 4
warpid = floor(j/8) + 5
m      = j mod 2
```

**第 4 步 施加 replica `R[2 : 4@warpid]`**(在 `warpid` 上加 0 或 4):

```
laneid = 4*i + floor(j/2) mod 4
warpid = floor(j/8) + 5 + 4*r,   r ∈ {0, 1}
m      = j mod 2
```

捋一捋结果:shard 把 tile 放到了 warp 5 和 6 上,replica 又把它拷到 warp 9 和 10。所以同一个逻辑元素,最后落在了**两组 warp 位置**上:

```mermaid
flowchart TD
    A["shard 落点<br/>warpid = 5, 6"] -- "+4 (replica)" --> B["replica 副本<br/>warpid = 9, 10"]
```

> **关键**:这个例子正好讲明白了**为啥这套模型非得用"物理坐标的集合"不可**。倒过来想一下:假如你用"物理坐标 → 逻辑坐标"的函数,复制根本就写不出来——因为好几个物理点都会映回同一个逻辑点,这压根就不是一个合法的函数。反过来,用"一个逻辑坐标 → 多个物理坐标",一点障碍都没有。这就是 API 死活要选"逻辑→物理"这个方向的根本原因。

---

## 10.8 案例 B:Blackwell 张量内存(TMEM)

上个例子用的是线程轴。但同一套布局模型,其实拿来描述**内存摆放**也照样行——轴不一定非得是线程轴,换成内存轴一样玩得转。TMEM 靠硬件 Lane 和 Col 坐标来寻址,在 TIRx 记号里就写成 `TLane`、`TCol`。看这个布局:

```python
layout = TileLayout(
    S[(2, 128, 112) : (112@TCol, 1@TLane, 1@TCol)]
)
```

要是逻辑形状正好就是 `(2, 128, 112)`,那就省事了:切出来的分量恰好就是逻辑坐标本身,不用再折腾换算。对元素 `(a, l, c)`:

```
TLane = l
TCol  = 112*a + c
```

- extent 是 128、步长 `1@TLane` 的那个 iter,负责把 128 行 TMEM Lane 填满;
- extent 是 2(步长 `112@TCol`)和 extent 是 112(步长 `1@TCol`)这两个 iter 凑一块儿,一共盖住 224 列,也就是 `TCol ∈ [0, 224)`。

| TLane \ TCol | `[0, 111]` | `[112, 223]` |
| --- | --- | --- |
| `TLane=0` | a=0 段(112 列) | a=1 段(112 列) |
| `TLane=1` | a=0 段(112 列) | a=1 段(112 列) |
| ... | ... | ... |
| `TLane=127` | a=0 段(112 列) | a=1 段(112 列) |

图注:128 行 TMEM Lane 全部填满;`TCol` 共 224 列,左半 `[0, 111]` 是 `a=0` 段,右半 `[112, 223]` 是 `a=1` 段。

> **注意**:这 224 列可不是手滑写错了,是**故意的**——**TMEM 布局不一定非得凑成 2 的幂**。举个例子,一个块缩放 FP8 GEMM(general matrix multiply,通用矩阵乘法)就可能专门挑 224 列的累加器:因为要是塞满 256 列,TMEM 就装不下了,没法把两个累加器阶段再加上缩放因子全放进去。这种"不规整"的形状,Layout API **张口就能写**——这正是它"支持通用形状"这个设计目标在发挥作用。

---

## 10.9 缩放因子布局(Scale Factor Layouts):复制的真正用武之地

前面那个累加器是**纯摆放**:一个逻辑累加器元素,只对一个 TMEM 坐标,一对一。可块缩放 MMA 里的**缩放因子(scale factor)**就两码事了——同一个物理组,经常得让**好几个 warp 窗口**同时都能看见。这下复制可就大显身手了:

```python
scale = TileLayout(
    S[(32, sf_per_mma) : (1@TLane, 1@TCol)]   # shard:在 TMEM 放一个 32 行的缩放因子组
    + R[4 : 32@TLane]                          # replica:相隔 32 lane,复制 4 份
)
```

对逻辑缩放坐标 `(r, s)`:

```
shard:           TLane = r ,            TCol = s
replica 之后:     TLane = r + 32*q ,     TCol = s ,   q ∈ {0,1,2,3}
```

这么一来,这个 32 行组就在 TMEM lane 的 0–31、32–63、64–95、96–127 这四个窗口里同时现身了:

| lane 窗口 | `TLane [0, 31]` | `TLane [32, 63]` | `TLane [64, 95]` | `TLane [96, 127]` |
| --- | --- | --- | --- | --- |
| 内容 | SF | SF | SF | SF |
| replica 偏移 | 原本 (+0) | +32 | +64 | +96 |

图注:四个 warp 大小的 lane 窗口看到的是同一份缩放因子组(SF),通过 replica 在 `TLane` 上分别偏移 0 / 32 / 64 / 96。

这其实就是「跨 GPU 世代的张量核操作数布局」那一章讲的 **`warpx4` 广播模式**:四个 warp 大小的 TMEM lane 窗口,看到的都是同一份缩放因子组。

当然,真实场景里这个布局还要再复杂一些。这个"原子"会跟外层 iter(沿 M 行和 K 缩放因子组铺开)拼到一起。另外,看缩放因子是什么 dtype,**好几个缩放因子还可能挤进同一个 32 位 `TCol` 单元**——比方说 fp8 缩放因子,就能把 4 个值打包塞进一个 32 位列单元。除此之外还有两个可选的 iter:一个"步长为 0 的复用"iter,一个"流水深度"iter,分别用来描述跨多次 MMA 的缩放复用,以及双缓冲。

> **关键**:这里最该记住的一点是——**同一个 `TileLayout` 模型,把两种压根不一样的情况一锅端了**。累加器是 TMEM 里的"单点摆放";缩放因子是同一片 TMEM 地址空间里的"复制摆放"。就一套模型,两类需求全搞定。

---

## 10.10 现成布局(Ready-Made Layouts)

写内核的时候,谁也不乐意每个硬件布局都从头手敲。所以 TIRx 替那些天天用到的布局备好了现成的构造器,直接调就完事:

| 构造器 | 返回内容 | 关键参数说明 |
| --- | --- | --- |
| `tmem_datapath_layout(datapath, rows, cols)` | `tcgen05.mma` 写出的 TMEM 累加器布局 | `datapath` 选择行摆放模式:`"D"` 对应 `M=128` 的恒等式摆放,`"F"` 对应 `M=64` 的散布式摆放 |
| `tcgen05_atom_layout(instr_shape, tensor_shape, dtype)` | 由一个 `tcgen05.ld` / `tcgen05.st` 原子搬运的寄存器 tile 布局 | `instr_shape` 如 `.32x32b`、`.16x64b`、`.16x128b` 等 |
| `wg_local_layout(cols, rows=128)` | warpgroup 本地寄存器 tile | 通常每个线程一行,落在 `tid_in_wg` 上 |

这里有个关于 `tcgen05_atom_layout` 的小门道值得留意:它在"DSL 层"和"降级(lowering)层"长得不一个样。在 DSL 这一层,它就是个 **warpgroup 分布式 tile**;可一到降级阶段,它就被拆成**四条 warp 协同(warp-collective)的 `tcgen05.ld` / `tcgen05.st` 指令,一个 warp 摊一条**,每条管自己那 32 个 TMEM lane。

```mermaid
flowchart TD
    A["DSL 层:tcgen05_atom_layout(...)<br/>一个 warpgroup 分布式 tile"] -->|"降级 lowering"| B["warp 0: tcgen05.ld/st (TMEM lane 0-31)"]
    A --> C["warp 1: tcgen05.ld/st (lane 32-63)"]
    A --> D["warp 2: tcgen05.ld/st (lane 64-95)"]
    A --> E["warp 3: tcgen05.ld/st (lane 96-127)"]
```

> **注意**:这些 helper 不过是替你省掉手写常见硬件映射的功夫,它们**并没有把模型藏起来**——每个 helper 吐出来的,都是拿前面那套 `S` / `R` 部件拼成的普通 `TileLayout`。

---

## 10.11 `SwizzleLayout` 与 `ComposeLayout`:仿射之外的换序

`TileLayout` 是**仿射**的:它能描述具名轴上的步长、复制和偏移。线程片段、TMEM tile、紧凑缩放因子布局这一大类摆放,它都拿得下。

可有一样它就是搞不定,那就是**共享内存(SMEM,一个 block 内线程共享的片上内存,见第 0 章)的 swizzle(换序,把数据在存储体间打乱摆放以避开冲突)**。swizzle 是用来躲开 bank 冲突(bank conflict,多个 lane 同时挤一个存储体的不同地址,被迫排队串行)的,它**根本不是仿射那种步长模式**,而是对线性共享内存地址来一次**基于 XOR 的置换(permutation)**。这种重排硬塞进仿射模型里,别扭得很。

所以 TIRx 索性把 swizzle 单拎出来做成一个布局对象,再让它跟 tile 布局组合着用:

```python
swizzle = SwizzleLayout(...)              # 单独的换序对象
layout  = ComposeLayout(swizzle, tile)    # 把 swizzle 叠在 tile 之上
```

执行顺序一目了然:**先让 tile 布局把线性内存地址算出来,再让 swizzle 把这个地址重排一遍**。把这两层拆开,比硬把 XOR 置换往仿射模型里塞,要干净利落得多。

```mermaid
flowchart LR
    A["逻辑坐标"] -->|"TileLayout(仿射)"| B["线性内存地址 m"]
    B -->|"SwizzleLayout(XOR 置换)"| C["换序后地址 addr"]
    C --> D["实际共享内存落点<br/>(bank 被打散)"]
```

---

## 10.12 为什么要 swizzle:bank 冲突的来由

得先把 bank 冲突是怎么来的讲清楚,你才明白 swizzle 到底在治啥病。

共享内存被切成 **32 个 bank**,每个 bank 的字(word)是 4 字节。规矩是这样:一次访问里,要是好几个 lane 同时去碰**同一个 bank 里的不同地址**,硬件就只能排着队一个一个来,这次访问就被**串行化(serialized)**了,慢一大截。

麻烦就麻烦在:一个老老实实的行主序 tile,**从结构上**天生就会撞上这种冲突。咱们拿一个 `(8, 64)` 的 float16 tile、行主序布局来瞧瞧:

```python
TileLayout(S[(8, 64) : (64@m, 1@m)])
```

逻辑元素 `(i, j)` 的线性地址是 `m = 64*i + j`。一行有 64 个 float16,也就是 128 字节——**正好是一整条共享内存 bank line**。问题就卡在这儿:假设一个 warp 沿着某一列(`j` 固定)往下读,每往下挪一行,地址就整整跨过一条 128 字节的 line。跨完一条 line,bank 索引又绕回原处,于是这一列读下来,**所有行全挤在同一个 bank 上**,这冲突是跑不掉了。

那 swizzle 怎么治?思路是:**让地址的低位去依赖更高的行位**。这么一搞,原本死活落在同一个 bank 的那一列,就被打散到不同 bank 上去了。

---

## 10.13 swizzle 变换的数学形式

思路讲明白了,咱们把 swizzle 的数学形式摆出来。别一看位运算就发怵,先弄清每个参数是干啥的。一个 `SwizzleLayout` 由三个整数管着:

```
per_element = M   # 保持连续的低位个数
swizzle_len = B   # 参与 XOR 的位组宽度
atom_len    = S   # 被取出去 XOR 的高位组起点
```

输入是线性元素地址 `m`,变换分几步走。先拿大白话过一遍:**`m` 最低的 `M` 位原封不动**(这是为了让一小撮连续元素继续挨在一起);剩下的高位拽下来当临时值 `x = m >> M`;接着把 `x` 里 `[S, S+B)` 那段位组**异或(XOR)**到 `[0, B)` 这段上;最后再把刚才没动的低 `M` 位拼回去。写成伪代码就是:

```python
mask = (1 << B) - 1
low  = m & ((1 << M) - 1)     # 取低 M 位,保持连续
x    = m >> M                  # 高位部分
x2   = x ^ ((x >> S) & mask)   # 把 [S, S+B) 位组 XOR 到 [0, B)
addr = (x2 << M) | low         # 把低 M 位拼回去
```

> **注意**:这套变换要成立,前提是 `S ≥ B`。还有一点务必记牢:**swizzle 不会动 tile 里到底有哪些逻辑元素**,它只动这些元素**落到共享内存的哪个位置**。MMA 读到的照样是同一个逻辑 tile,swizzle 不过是把背后那套物理 bank 排布优化了一下罢了。

---

## 10.14 如何选择 swizzle 参数

好消息是:平时你基本不用亲自去定这三个参数——它们会**根据 dtype 和共享内存 swizzle 模式自动推出来**。常见的 swizzle 模式就 32 字节、64 字节、128 字节这三种。

不过道理还是值得搞明白。`per_element`(也就是 `M`)怎么选?原则是:**让一个向量大小的小组保持连续**。拿 float16 说,一个 16 字节向量能装 8 个元素,所以 `M = log2(8) = 3`。再配上 128 字节 swizzle,布局就这么写:

```python
SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
```

这套参数一箭双雕:既保住了 16 字节向量组不被拆散,又足够把更大范围的共享内存地址模式搅乱,从而把列方向的 bank 冲突消掉。于是,一个带 swizzle 的共享内存分配就长这样:

```python
tile    = TileLayout(S[(8, 64) : (64@m, 1@m)])
swizzle = SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)
layout  = ComposeLayout(swizzle, tile)   # 组合后的布局才是挂到 SMEM buffer 上的那个
```

> **关键**:绝大多数代码**根本不该**自己去手算这些参数——dtype 和 descriptor 模式一般早就把它们定死了。你真正得操心的就一件事:**TIRx 布局里的 swizzle、TMA descriptor、还有 MMA 的预期,这三家必须对得上、保持一致**。

---

## 10.15 元素的 bank 与 line

怎么验证一个 swizzle 是不是真管用?办法挺简单:把 swizzle 之后的元素地址换算回它落在哪个 bank,看看是不是散开了。设 `addr` 是 swizzle 后的元素地址,`b` 是每个元素的字节数:

```
byte = addr * b               # 字节地址
bank = floor(byte / 4) mod 32 # 32 个 bank,每 bank 字 4 字节
line = floor(byte / 128)      # 128 字节的 bank line
```

对 float16,`b = 2`,bank 公式就简化成 `bank = floor(addr / 2) mod 32`。下面那段演算用的就是这个。

---

## 10.16 完整演算:对 `(8, 64)` float16 tile 施加 128B swizzle

理论讲完了,咱们把整件事从头到尾算一遍,看看冲突是不是真没了。还是那个行主序 float16 tile:`m = 64*i + j`,套上 `SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)`。把它代进变换公式:

```
x    = m >> 3
addr = ((x ^ ((x >> 3) & 7)) << 3) | (m & 7)
```

令 `q = floor(j/8)`、`r = j mod 8`,可化简为:

```
addr = 64*i + 8*(q XOR i) + r
```

**盯住列 `j = 0` 看**(这时 `q = 0`、`r = 0`),得到 `addr = 72*i`。咱们一行一行地算 bank(用 `bank = floor(addr/2) mod 32`):

| 行 `i` | `addr = 72*i` | `bank = floor(addr/2) mod 32` |
| :---: | :---: | :---: |
| 0 | 0   | **0**  |
| 1 | 72  | **4**  |
| 2 | 144 | **8**  |
| 3 | 216 | **12** |
| 4 | 288 | **16** |
| 5 | 360 | **20** |
| 6 | 432 | **24** |
| 7 | 504 | **28** |

看结果:这一列现在散在 **8 个互不相同的 bank** 上,冲突没了。

**再对比一下不加 swizzle 的情形**:同一列的地址是 `m = 64*i`,于是 `bank = floor(64*i / 2) mod 32 = 0`——**每一行全挤在 bank 0**,整次访问被彻底串行化。差距一眼就看出来了。

| 情况(列 `j=0`) | 8 行 `i` 落到的 bank | 结果 |
| --- | --- | --- |
| 无 swizzle | 全部塞进 bank 0 | 串行化,约 8x 慢 |
| 有 128B swizzle | 散到 bank 0, 4, 8, 12, 16, 20, 24, 28 | 无冲突 |

> **注意**:可别把这个"无冲突"当成包治百病的护身符,它**是有前提的**:你得照着 swizzle 当初的设计意图来用。dtype、swizzle 宽度、访问形状,统统得跟 TMA、MMA 的 descriptor 模式对上。一个 128 字节 float16 swizzle,是**专门冲着特定的 16 字节行块和张量核访问模式**设计的——它**可不保证**你随便来个什么共享内存访问都能无冲突。

---

## 10.17 设计原理(Design Rationale)

学到这儿,咱们回头看看整套 Layout API 背后那三条设计取舍。你会发现,前面每个例子其实都在替它们做注脚:

1. **支持通用形状**。硬件 tile 可不总是 2 的幂。全局张量、共享内存阶段、TMEM 累加器、缩放因子缓冲,常常因为容量见顶或者算法上的取舍,长成"不规整"的形状。布局模型把这种形状当**家常便饭**来对待(还记得 10.8 那个 224 列的累加器吧)。

2. **映射方向是「逻辑坐标 → 物理坐标」**。这个方向为啥要紧?因为复制太常见了:一个逻辑元素可能同时住在好几个物理位置。"逻辑→物理"这个方向,正好能把这点直接写成一个坐标集合(回想一下 10.7 那个 warp 复制)。

3. **硬件轴明着写出来**。布局**不玩**匿名维度,也不靠"回头看上下文"去猜每一维是啥意思。`tx`、`tid_in_wg`、`laneid`、`warpid`、`TLane`、`TCol` 彼此的区别,统统**白纸黑字写进布局本身**。

最后还有一条很要紧的分工:**布局不管合法性、也不管可行性,那不是它一个人扛的活儿**。布局就管一件事——"数据搁哪儿";至于"某个操作能不能合法、又高效地用上这种摆放",那是上层 tile 原语(primitive)拍板的事。这么一分工,Layout API 既保持了小巧,又把足够的信息喂给编译器,让它去派发(dispatch)真正的硬件操作。

---

## 小结

- TIRx Layout API 把上一章那套纸面布局记号**坐实成了对象**,核心就三个类:`TileLayout`、`SwizzleLayout`、`ComposeLayout`。
- `TileLayout` 是仿射布局,分三步走——**shard(`S`,定位)+ replica(`R`,抄送)+ offset(`O`,平移)**,底层全靠 `(extent, stride, axis)` 这种 **iter** 三元组来表示。一句话公式 `L(x) = { D(x) + r + O | r ∈ R }` 抓住了魂:一个逻辑坐标 → 一组物理坐标。
- `layout.apply()` 的求值就四步:**展平 → 按 shard extents 切分 → 乘步长累加到具名轴再加 offset → 枚举 replica 组合**。它不把输入形状写死,只要"逻辑元素数 = shard extents 的乘积"就行。
- 轴是**具名硬件坐标**,名字本身就带着意思(记牢 `1@laneid ≠ 1@TLane`)。两个案例(张量核寄存器 tile、Blackwell TMEM)说明同一个模型既能描述线程摆放、也能描述内存摆放;缩放因子布局则演示了 replica 怎么自然地把 `warpx4` 广播表达出来。
- swizzle 是 **XOR 置换、不属于仿射**,所以单拎出来做成 `SwizzleLayout`,再用 `ComposeLayout` 叠到 tile 上。它只动物理落点、不动逻辑内容,靠"低位依赖高位"把 bank 打散,从而消掉列访问的 bank 冲突——但前提是得跟 TMA / MMA 的 descriptor 模式严丝合缝地对上,否则白搭。
- 三条设计哲学串起来:**支持通用(非 2 的幂)形状、逻辑→物理的集合映射、把具名硬件轴明着写出来**。布局只管"搁哪儿",合法性交给上层 tile 原语去判断。

## 延伸阅读

- 原文:[TIRx Layout API — Modern GPU Programming for MLSys](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tirx_layout_api/index.html)
- 前置章节:《数据布局及其记号(Data Layout and Its Notation)》——本章记号的来源
- 相关章节:《特种内存:TMEM(Special Memory: TMEM)》《跨 GPU 世代的张量核操作数布局(Tensor Core Operand Layouts Across GPU Generations)》

## 术语对照

| 中文 | English | 说明 |
| --- | --- | --- |
| 仿射布局 | affine layout | `TileLayout` 表达的步长/复制/偏移类布局 |
| 分片(规格) | shard / shard spec | 由 `S[...]` 构建,产出基础物理坐标 |
| 复制(规格) | replica / replica spec | 由 `R[...]` 构建,生成与逻辑无关的额外副本 |
| 偏移 | offset | 加到每个结果上的固定坐标 |
| 迭代器(三元组) | iter | `(extent, stride, axis)`,描述一条具名轴上的带步长行走 |
| 具名轴 | named axis | 真实硬件坐标或编译器摆放坐标,名字即语义 |
| 前向映射 | forward mapping | 逻辑坐标 → 物理坐标(集合)的求值 |
| 换序 / 拌序 | swizzle | 基于 XOR 的共享内存地址置换,用于避开 bank 冲突 |
| bank 冲突 | bank conflict | 多 lane 访问同一 bank 不同地址导致的串行化 |
| 块缩放(MMA) | block-scaled (MMA) | 带缩放因子的矩阵乘累加 |
| 缩放因子 | scale factor | 块缩放 MMA 中的缩放系数 |
| 广播 | broadcast | 同一数据在 warp/lane/内存区域间被复制 |
| 降级 | lowering | DSL 层布局编译为具体硬件指令的过程 |
| 张量内存 | TMEM (Tensor Memory) | Blackwell 上由 Lane/Col 寻址的特种内存 |
| 线程束 / warp | warp | 32 个 lane 的执行单元 |
| warp 组 | warpgroup | 多个 warp 组成的协作单元 |
