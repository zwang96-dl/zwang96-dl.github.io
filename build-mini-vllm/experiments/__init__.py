"""experiments/ —— 每课对应的、可从终端直接运行的真实实验。

设计约定：
    - 每个实验都是 ``python3 -m experiments.lesson_XX_name`` 可直接运行的模块。
    - 支持 ``--config`` 指定配置、``--mode {quick,normal,stress}`` 选择规模、
      ``--verbose`` / ``--trace`` 控制可观测层级、``--out`` 指定输出目录。
    - 实验只**读**配置与 workload，只**写** outputs/ 下的结果；**绝不修改源代码**。
    - ``course.py run N`` 只是这些命令的透明薄封装，运行前会打印等价的底层命令。
"""
