"""轨道 A：机制模拟器（mechanism simulators）。

这些模拟器**不使用神经网络**，而是用确定性的整数与数据结构，模拟推理引擎
里真正重要的机制：请求状态机、prefill/decode、token 预算、KV block 分配、
碎片、调度……

特点：
    - 快（毫秒级）
    - 可重复（固定的确定性规则，不用随机数）
    - 可单步（每次迭代产出一个快照）
    - 易 Trace / 易可视化
    - 零第三方依赖（纯标准库，8GB MacBook Air / 仅 CPU 也能跑）

它们是 HTML 动画在 Python 侧的「真身」——网页动画只是把这里的每一步画出来。
"""

from .text_pipeline import (  # noqa: F401
    LifecycleSimulator,
    RequestState,
    SimRequest,
    SimResult,
    IterationRecord,
)
