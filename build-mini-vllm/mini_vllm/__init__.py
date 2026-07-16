"""mini_vllm — a teaching-oriented, from-scratch large-model inference engine.

设计原则（Design principles）
-----------------------------
1. **纯标准库核心（pure-stdlib core）**：核心代码只依赖 Python 标准库，
   在任何 Apple Silicon MacBook Air（甚至 8GB 内存 / 仅 CPU）上都能离线运行，
   无需 PyTorch、numpy 或 Pillow。这样 Level A（零第三方依赖离线学习）和
   机制模拟器（Track A）随时可运行、可测试。
2. **透明（transparent）**：所有状态、shape、数据流都可以被打印、追踪、检查。
   没有隐藏的自动化，没有静默的联网。
3. **可演化（single evolving codebase）**：每个 Lesson 在同一套真实系统上
   增加能力，而不是创建一份份近似副本。

本包目前包含（Phase 1）：
    trace        —— 统一的可观测性 / Trace 基础设施
    tokenizer    —— 真实的 byte-level tokenizer
    config       —— 模型 / 引擎配置 dataclass
    simulator/   —— 轨道 A：确定性机制模拟器（无神经网络）
    course_meta  —— 课程元数据（course.py 与 HTML 的单一数据源）

后续 Phase 会在 model/ cache/ scheduler/ engine/ multimodal/ 下逐步填充。
"""

__version__ = "0.1.0"

# 课程支持的运行时下限（用于 doctor 检查）。
MIN_PYTHON = (3, 11)
