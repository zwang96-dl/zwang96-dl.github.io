"""model/ —— 极小 decoder-only Transformer（轨道 B）。

计划在 Phase 3 落地，纯标准库实现（不依赖 PyTorch），可在仅 CPU 的
MacBook Air 上离线运行：
    embedding.py   —— token embedding
    rmsnorm.py     —— RMSNorm
    rope.py        —— 旋转位置编码 RoPE
    attention.py   —— causal multi-head / GQA attention
    mlp.py         —— SwiGLU
    transformer.py —— 组装成 forward，含 LM head

Phase 1 仅提供机制模拟器（见 ``mini_vllm/simulator/``），此目录暂为占位。
"""
