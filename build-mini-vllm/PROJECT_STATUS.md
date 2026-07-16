# PROJECT_STATUS

Build Your Own Mini-vLLM 的分阶段实施状态。诚实标注「已完成 / 进行中 / 计划中」，
不把未完成的东西说成完成。

**最后更新：** 2026-07-14 · **当前里程碑：** **全部 7 个 Phase 完成**——文本主线（0–18）
与多模态主线（19–30）全部就绪，共 **31 课、148 项测试通过**，完全离线可运行。

## 关键工程决策（Key Decisions）

| 决策 | 原因 |
|------|------|
| **核心纯 Python 标准库** | 本机无 pip/torch/numpy/Pillow；纯标准库让 Level A（零依赖离线）与机制模拟器天然成立，8GB / 仅 CPU 也能跑、可测试、可复现。 |
| **轨道 B tiny model 计划用纯 Python 实现** | 避免大体积 torch wheel 进离线仓库；仍是「真实前向 + 真实 shape」，只是数值用纯 Python。PyTorch 作为 optional 加速。 |
| **`offline/requirements-lock.txt` 刻意为空** | 核心无第三方必需依赖；`--no-index --require-hashes` 安装会立即成功且绝不联网。 |
| **course.py 是透明薄封装** | 每个命令先打印等价底层命令、输入/输出、是否改源码；可随时绕过直接跑 `python3 -m ...`。 |

## 进度总览

| Phase | 内容 | 状态 |
|-------|------|------|
| **Phase 1** | 安装/doctor → Lesson 0 → 第一个 simulator → HTML → 真实命令 → 输出 → 成功反馈 | ✅ **已完成** |
| **Phase 2** | 第一条完整教学闭环：Lesson 1(Tokenizer)/2(Tensor)/4(Attention) 的网页动画+实验+测试+Inspector | ✅ **已完成** |
| **Phase 3** | Text Engine（Lesson 3,5–18）：tiny Transformer、generation、KV Cache、batching、scheduler、block allocator、paged KV、chunked prefill、prefix cache、完整引擎与综合挑战 | ✅ **已完成** |
| **Phase 4** | Single Image（Lesson 19–24）：message/template/image processor/patch/vision encoder/projector/placeholder/merge/mm-prefill | ✅ **已完成** |
| **Phase 5** | Multi-image & Video（Lesson 25–26）：动态 visual token、多图、mixed batch、抽帧、timestamp | ✅ **已完成** |
| **Phase 6** | Multimodal Cache & Scheduler（Lesson 27–30）：三层缓存、visual budget、mm scheduler、综合挑战 | ✅ **已完成** |
| **Phase 7** | 离线打包与打磨：全部课程就绪、无占位、离线 release、离线自检、accessibility、Benchmark、最终报告 | ✅ **已完成** |

## Phase 1 已交付清单（可运行、可测试）

**引擎 / 核心**
- `mini_vllm/trace.py` — 三层可观测性 Tracer（normal/verbose/trace）
- `mini_vllm/tokenizer.py` — 真实 byte-level tokenizer（vocab 259，encode/decode/pad）
- `mini_vllm/config.py` — Model/Vision/Engine 配置 + Quick/Normal/Stress 规模
- `mini_vllm/simulator/text_pipeline.py` — **请求生命周期机制模拟器**（状态机、prefill/decode、token 预算、continuous batching、KV block 分配与释放、OOM）
- `mini_vllm/course_meta.py` — 31 课元数据（单一数据源）
- `mini_vllm/offline_check.py` — 离线自检逻辑（course.py 与测试共用）

**Course Runner**
- `course.py` — `serve / doctor[--offline] / verify-offline-bundle / where-am-i / lesson / run / check / hint / inspect`；`benchmark / mm-*` 为透明占位（明确标注计划中，不伪装成功）

**实验 / 配置 / 资产**
- `experiments/lesson_00_intro.py`（支持 `--mode/--trace/--verbose`，写 `outputs/lesson_00/result.json`）
- `configs/lesson_00_quick.json`、`assets/workloads/lesson_00.json`、`assets/tokenizer/byte_tokenizer.json`

**测试（全部通过）**
- `tests/test_tokenizer.py`（9）、`tests/lesson_00/test_intro.py`（8）、`tests/test_system.py`、`tests/test_offline.py`

**网页（离线、零远程依赖、主题感知、无障碍）**
- `docs/index.html`（首页 + 进度 + 搜索 + 路线图）
- `docs/lessons/lesson_00.html`（固定头部、Learn/Build/Challenge/Deep Dive、交互式动画、真实命令、四级提示、常见错误、Trace、vLLM 映射）
- `docs/js/anim/request_lifecycle.js`（Play/Pause/Step/Reset/速度/键盘/reduced-motion；Python 模拟器的忠实镜像）
- `docs/css/course.css`、`docs/js/course.js`、`docs/references/`、`docs/glossary/`

**离线交付**
- `offline/requirements-lock.txt`、`offline/manifests/{wheels,assets}.json`、`offline/INSTALL_OFFLINE.md`
- `scripts/build_offline_release.py`（生成 `dist/build-mini-vllm-offline-macos-arm64/` + 全树 SHA-256 manifest）

## Phase 2 已交付清单（可运行、可测试）

**引擎 / 核心（纯 Python，零依赖）**
- `mini_vllm/model/matrix.py` — 张量/矩阵工具：shape 校验、matmul（row-column rule）、transpose、
  batched_matmul、行广播、误差比较（Lesson 2 的 Build 目标）
- `mini_vllm/model/attention_ref.py` — 单头 causal scaled-dot-product attention：softmax（数值稳定）、
  causal mask、五步 + return_stages（Lesson 4 的 Build 目标）
- `mini_vllm/tokenizer.py` — byte-level tokenizer（Phase 1 已落地，Lesson 1 正式讲解）

**实验 / 配置 / 资产**
- `experiments/lesson_01_tokenizer.py`、`lesson_02_tensor.py`、`lesson_04_attention.py`（均支持 --mode/--trace/--verbose）
- `configs/lesson_0{1,2,4}_quick.json` + `assets/workloads/lesson_0{1,2,4}.json`

**测试（全部通过，共 55 项）**
- `tests/lesson_01/test_tokenizer.py`（5）、`tests/lesson_02/test_matrix.py`（9）、`tests/lesson_04/test_attention.py`（9）
- `tests/test_system.py` 升级为「遍历所有已就绪课程」检查网页契约

**网页（离线、零远程依赖、含交互式动画）**
- `docs/lessons/lesson_01.html`（Tokenizer Explorer，可实时输入文本）
- `docs/lessons/lesson_02.html`（Matrix Multiplication Explorer，逐格乘加）
- `docs/lessons/lesson_04.html`（Attention Stepper，五步逐步）
- `docs/js/anim/stepper_core.js`（复用的单步动画内核 + 矩阵/chip 渲染）
- `docs/js/anim/tokenizer_explorer.js`、`matrix_mul.js`、`attention_stepper.js`
- 三个动画均经 node 实机执行验证，输出与 Python 引擎逐值一致

**Inspector / Course Runner**
- `course.py inspect` 新增 `tokenizer` / `matmul` / `attention` 三个检查器
- `mini_vllm/course_meta.py`：Lesson 1/2/4 标为 ready，含完整元数据与四级提示
- 首页与离线自检（REQUIRED_FILES）已同步覆盖新资产

## Phase 3 Part A 已交付清单（可运行、可测试，纯 Python 零依赖）

**模型 / 引擎**
- `mini_vllm/model/{rmsnorm,rope,mlp,transformer}.py` — RMSNorm / RoPE / SwiGLU / TinyTextModel（含 GQA、确定性初始化）
- `mini_vllm/model/attention_ref.py` — 新增 `sdpa_positions()`（位置感知 causal attention，支持 KV Cache）
- `mini_vllm/cache/kv_cache.py` — KVCache（append / 一致性自检 / 内存公式）
- `mini_vllm/sampling.py` — greedy/temperature/top-k/top-p，确定性可复现
- `mini_vllm/engine/generate.py` — `generate_naive` / `generate_cached` / `processed_token_curves` + TTFT/TPOT
- `assets/checkpoints/tiny_text.json` — config+seed 形式的本地 checkpoint（离线确定性重建，diff=0）

**实验 / 测试（Lesson 3/5/6/7/8，新增 29 项测试，全绿）**
- `experiments/lesson_0{3,5,6,7,8}_*.py` + `configs/lesson_0{3,5,6,7,8}_quick.json`
- `tests/lesson_0{3,5,6,7,8}/…` —— 含 **KV Cache 正确性对齐**（cached 与 naive logits 逐值一致，误差 0）

**网页动画（离线，均经 node 实机执行验证，与 Python 引擎一致）**
- `docs/lessons/lesson_0{3,5,6,7,8}.html`
- `docs/js/anim/{transformer_pipeline,sampling_explorer,recompute_timeline,kv_cache_stepper,prefill_decode_timeline}.js`

**Inspector**：`course.py inspect model | generate`（真实前向 / 生成）

## Phase 3B / 4 / 5 / 6 已交付清单（纯 Python 零依赖）

**Phase 3B · 调度与分页（Lesson 9–18）**
- `mini_vllm/cache/{block_allocator,block_table,prefix_cache}.py` — 引用计数分配器、Paged KV（paged==连续，误差 0）、前缀共享
- `mini_vllm/scheduler/{request,scheduler}.py` — 状态机、FIFO/decode-first/SJF/balanced、token 预算、chunked prefill
- `mini_vllm/engine/engine.py` + `benchmarks/report.py` — continuous batching 引擎、性能报告、停滞检测
- `course.py benchmark 8|17|18`；`inspect blocks|scheduler`

**Phase 4–6 · 多模态（Lesson 19–30）**
- `mini_vllm/multimodal/`：messages / chat_template / inputs / placeholders / media / image_processor /
  video_sampler / patch_embed / vision_encoder(+projector) / embedding_merge / cache（三层）/ budget / runner / mm_engine
- 本地素材：`assets/images/*.json`、`assets/videos/demo/`（合成、离线、确定性）
- `course.py mm-demo image|video`、`mm-inspect placeholders`、`mm-benchmark`
- 关键性质（均有测试）：paged==连续（误差 0）、cached==naive（误差 0）、前缀/多模态命中不改变输出、
  encoder 不在 decode 重复运行、不跨请求串视觉 embedding、placeholder 严格对齐、timestamp 保留、无 KV 泄漏

## 已知限制（Known Limitations）

- 轨道 B 的 tiny model 用纯 Python（hidden=32 等极小配置）以保证 CPU 速度与零依赖；**能力不是重点，数据流与 shape 真实**。
- `offline/wheels/macos-arm64/` 不含 torch wheel（核心不需要；可选加速需用户按 `offline/INSTALL_OFFLINE.md` 自行下载）。
- 本沙盒无 pip，故 Level B 的 `pip install` 未实机执行；但 `requirements-lock.txt` 为空，该命令设计上会立即成功且绝不联网。Level A 已完整验收。
- 视频/图片为确定性合成素材（教学用），非真实拍摄内容。

## 下一步（可选增强）

课程已完整交付。可选的后续增强：真实 BPE tokenizer、可选 torch/MPS 后端、更大的 tiny checkpoint、
更多 workload 与 benchmark 场景、抢占式调度、真实数据集样例。
