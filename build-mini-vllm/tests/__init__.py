"""tests/ —— 使用标准库 unittest。

约定：测试失败信息要**有教学价值**——不仅说「不相等」，还要指出预期、实际、
可能原因、对应文件与函数。运行全部测试：

    python3 -m unittest discover -s tests
"""

import sys
from pathlib import Path

# 让 `python3 -m unittest discover -s tests` 无论从哪运行都能 import mini_vllm。
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
