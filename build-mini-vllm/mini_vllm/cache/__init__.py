"""cache/ —— KV Cache 与分页内存管理（轨道 B / 机制）。

计划在 Phase 3 落地：
    kv_cache.py        —— 逐层 K/V 缓存与 append（Lesson 7）
    block_allocator.py —— physical block 池、free list、OOM/leak 检测（Lesson 13）
    block_table.py     —— logical→physical 映射、Paged KV（Lesson 14）
    prefix_cache.py    —— block hash、引用计数、前缀共享（Lesson 16）

Phase 1 的机制预览在 ``mini_vllm/simulator/text_pipeline.py`` 的 _BlockPool。
"""
