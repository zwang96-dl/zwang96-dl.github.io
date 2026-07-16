# Build Your Own Mini-vLLM

**从零构建支持文本、图片和视频的大模型推理引擎** —— 一套完全离线、交互式的中文课程。

> 在一台 8GB 的 Apple Silicon MacBook Air 上、在没有网络的飞机上，你也能一步步搭出一个
> 能讲清楚 vLLM 核心思想的教学型推理引擎。

---

## 新用户主线（照着做即可）

### 1) 支持环境

| 项目   | 要求                                   |
|--------|----------------------------------------|
| 系统   | macOS 14+                              |
| 架构   | Apple Silicon（arm64）                 |
| Python | 3.11+                                  |
| 内存   | 8 GB 起                                |
| 计算   | 仅 CPU 即可；MPS 可选加速               |
| 依赖   | **核心为纯标准库，无需任何第三方包**    |

### 2) 准备环境

```bash
cd build-mini-vllm

# 可选：建一个隔离的虚拟环境（核心不装任何包也能跑）
python3.11 -m venv .venv
source .venv/bin/activate
```

> 核心课程（机制模拟器 + 纯 Python tiny model）不依赖 PyTorch/numpy/Pillow，
> 因此**这一步即使跳过 `pip install` 也能运行**。完整的离线安装流程见
> [`offline/INSTALL_OFFLINE.md`](offline/INSTALL_OFFLINE.md)。

### 3) 校验离线资源

```bash
python3 course.py verify-offline-bundle
```
> 检查所有 HTML/CSS/JS、tokenizer、配置、workload、manifest 是否齐全，网页是否零远程依赖，
> 代码是否无运行时联网。**任何必需文件缺失都会明确报错**（不是 warning）。

### 4) 启动 HTML 教程

```bash
python3 course.py serve
# 浏览器打开 http://127.0.0.1:8000/index.html
# 也可直接双击 docs/index.html 离线打开
```

### 5) 开始 Lesson 0，并运行第一个模拟器

```bash
python3 course.py run 0        # 运行请求生命周期机制模拟器（先打印等价的底层命令）
python3 course.py check 0      # 跑通测试，获得第一次成功反馈
python3 course.py where-am-i   # 我在哪、下一步做什么
```

看到 `Lesson 0 检查通过！` 就说明主线跑通了 —— 去网页的
[Lesson 0](docs/lessons/lesson_00.html) 用动画单步观察同一套过程。

---

<details>
<summary><strong>更多命令（先跑通主线，再看这里）</strong></summary>

```bash
python3 course.py                      # 终端版首页：当前进度 & 唯一推荐的下一步
python3 course.py doctor [--offline]   # 环境自检
python3 course.py lesson 0             # 某课的固定头部信息
python3 course.py run 0 [--trace]      # 运行实验（可加 --trace / --verbose / --mode）
python3 course.py check 0              # 运行测试并更新进度
python3 course.py hint 0 --level 1     # 四级提示
python3 course.py inspect scheduler    # 内部状态：scheduler | blocks | kv-cache

python3 -m unittest discover -s tests  # 直接运行全部测试（绕过 course.py）
python3 scripts/build_offline_release.py   # 构建离线发布包到 dist/
```

**透明原则**：每个 `course.py` 封装命令在执行前都会打印它等价的底层命令、输入、输出、
以及「是否修改源代码」。你随时可以绕过 `course.py` 直接运行底层 `python3 -m ...`。
没有任何命令会在后台自动改你的代码或做 git 恢复。

</details>

---

## 这套课程长什么样

```
离线 HTML 知识教材   +   真实 Python mini-vLLM 项目   +   测试 / Trace / Inspector / Benchmark
   （负责理解）              （负责实现）                      （负责验证）
```

- **HTML** 负责讲解、动画、手算、Tensor shape、数据流、代码导读；
- **编辑器** 负责阅读和修改真实工程代码；
- **终端** 负责运行实验、跑测试、看 Trace、跑 Benchmark。

网页帮助你理解代码，但**不替你运行、也绝不修改你的代码**。

## 双轨教学

- **轨道 A · 机制模拟器**：用确定性整数与数据结构模拟请求、调度、KV block、碎片、OOM……
  快、可重复、可单步、无需任何第三方库。
- **轨道 B · 真实 Tiny Model**：极小 decoder-only Transformer 与 tiny multimodal model，
  纯 Python 实现，CPU 可跑，MPS 可选。

## 目录结构（节选）

```
build-mini-vllm/
├── course.py              # 统一、透明的入口
├── mini_vllm/             # 引擎源码（simulator/ 已就绪；model/cache/scheduler/… 逐 Phase 落地）
├── experiments/           # 每课的可运行实验
├── tests/                 # unittest 测试
├── configs/ · assets/     # 配置与数据资产（tokenizer/workload/…）
├── docs/                  # 离线 HTML 教材（index + lessons + css/js/anim）
├── offline/               # 离线交付：requirements-lock / manifests / wheels / 安装说明
└── scripts/               # 构建离线发布包等
```

## 当前进度

见 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)。**全部 7 个 Phase 已交付：Lesson 0–30 全部就绪，148 项测试通过，完全离线。**

- **文本主线（0–18）**：Tokenizer → Tensor/矩阵 → Transformer → 生成 → **KV Cache** → 批处理 → 调度 → **Paged KV** → chunked prefill → **前缀缓存** → 完整引擎 → 综合挑战。
- **多模态主线（19–30）**：多模态消息 → 图片 Tensor → patch → vision encoder/projector → placeholder/合并 → 多模态 prefill → 多图/视频 → **三层缓存** → 多模态调度 → 完整多模态引擎 → 综合挑战。

```bash
# 文本引擎
python3 course.py run 7 && python3 course.py check 7    # KV Cache（cached==naive，误差 0）
python3 course.py run 14 && python3 course.py check 14  # Paged KV（paged==连续，误差 0）
python3 course.py run 16                                # 前缀缓存（命中不改变输出）
python3 course.py benchmark 18                          # 综合挑战性能报告
# 多模态
python3 course.py mm-demo image                        # 图片 prefill + 文本 decode
python3 course.py mm-demo video                        # 视频抽帧
python3 course.py mm-inspect placeholders              # placeholder 对齐
python3 course.py mm-benchmark                         # 多模态端到端报告
```

**关键正确性性质（均有测试守护）**：cached==naive、paged==连续（误差 0）、前缀/多模态命中不改变输出、
vision encoder 不在 decode 重复运行、不跨请求串视觉 embedding、placeholder 严格对齐、timestamp 保留、无 KV 泄漏。

## 用 Git 查看与恢复代码（透明、由你掌控）

```bash
git status            # 看改了哪些文件
git diff              # 看具体改动
git log --oneline     # 看提交历史
git restore <file>    # 放弃某文件的改动（由你手动执行，课程不会自动做）
```
