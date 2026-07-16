"""scheduler/ —— 请求状态机与迭代级调度（轨道 B / 机制）。

计划在 Phase 3 落地：
    request.py    —— 请求对象与状态机 WAITING/RUNNING/FINISHED/ABORTED（Lesson 10）
    policies.py   —— FIFO / decode-first / SJF / balanced（Lesson 11）
    scheduler.py  —— token 预算、sequence 预算、chunked prefill（Lesson 15）

Phase 1 的调度预览在 ``mini_vllm/simulator/text_pipeline.py`` 的 LifecycleSimulator。
"""
