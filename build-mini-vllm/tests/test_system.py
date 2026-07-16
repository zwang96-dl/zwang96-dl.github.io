"""系统级测试：命令注册、网页结构、包可导入性。

这些测试守护「新用户第一条学习路径」不被破坏：course.py 的核心命令都在、
Lesson 0 网页含有可用的动画控件与固定头部、mini_vllm 能干净导入。
"""

import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


class TestCommandRegistry(unittest.TestCase):
    def test_course_module_imports(self):
        import course  # noqa: F401  —— course.py 应可被 import 而不执行 main

    def test_core_commands_registered(self):
        import course
        for name in [
            "serve", "doctor", "verify-offline-bundle", "where-am-i",
            "lesson", "run", "check", "hint", "inspect",
        ]:
            self.assertIn(
                name, course.COMMANDS,
                msg=f"course.py 缺少核心命令 {name!r}；检查 course.py 的 COMMANDS 注册表。",
            )


class TestPackageImports(unittest.TestCase):
    def test_import_mini_vllm(self):
        import mini_vllm
        self.assertTrue(hasattr(mini_vllm, "__version__"))

    def test_import_simulator_and_tokenizer(self):
        from mini_vllm.simulator import LifecycleSimulator  # noqa: F401
        from mini_vllm.tokenizer import ByteTokenizer  # noqa: F401


class TestReadyLessonPages(unittest.TestCase):
    """对每个「已就绪」课程的网页施加统一契约：动画控件、固定头部、真实命令。"""

    def _pages(self):
        import course  # noqa: F401 —— 触发 sys.path 设置
        from mini_vllm import course_meta
        for n in course_meta.READY_LESSONS:
            path = _ROOT / f"docs/lessons/lesson_{n:02d}.html"
            yield n, path

    def test_ready_pages_exist(self):
        for n, path in self._pages():
            with self.subTest(lesson=n):
                self.assertTrue(path.exists(),
                                msg=f"已就绪的 Lesson {n} 缺少网页 {path.name}。")

    def test_animation_controls(self):
        for n, path in self._pages():
            html = path.read_text("utf-8")
            for control in ["play", "pause", "next", "prev", "reset"]:
                with self.subTest(lesson=n, control=control):
                    self.assertRegex(
                        html, rf'data-anim-control=["\']{control}["\']',
                        msg=f"Lesson {n} 动画缺少 {control!r} 控件（data-anim-control）。",
                    )

    def test_fixed_header_fields(self):
        for n, path in self._pages():
            html = path.read_text("utf-8")
            for field in ["当前目标", "需要阅读的文件", "实验命令", "测试命令", "完成后下一步"]:
                with self.subTest(lesson=n, field=field):
                    self.assertIn(field, html, msg=f"Lesson {n} 缺少固定头部字段「{field}」。")

    def test_declares_animation_question(self):
        for n, path in self._pages():
            with self.subTest(lesson=n):
                self.assertIn("这个动画要回答的问题", path.read_text("utf-8"),
                              msg=f"Lesson {n} 动画未写明「这个动画要回答的问题」。")

    def test_shows_real_run_command(self):
        for n, path in self._pages():
            with self.subTest(lesson=n):
                self.assertIn(f"course.py run {n}", path.read_text("utf-8"),
                              msg=f"Lesson {n} 网页应展示真实运行命令 `course.py run {n}`。")

    def test_animation_is_wired_to_defined_module(self):
        """每页的动画挂载 `X.mount(...)` 必须指向一个在其引用脚本里 `window.X=...` 定义过的模块。

        这守护「动画未接线 / 名字写错 / 引用了不存在的动画」这一类 bug——它们不会被
        控件/文本类检查发现，却会让动画空白。"""
        import re
        for n, path in self._pages():
            html = path.read_text("utf-8")
            mount = re.search(r'(\w+)\.mount\(', html)
            with self.subTest(lesson=n):
                self.assertIsNotNone(mount, msg=f"Lesson {n} 未发现动画挂载调用 X.mount(...)。")
                name = mount.group(1)
                srcs = re.findall(r'<script src="\.\./js/anim/([^"]+)">', html)
                self.assertTrue(srcs, msg=f"Lesson {n} 未引用任何 docs/js/anim 脚本。")
                defined = False
                for s in srcs:
                    js = (_ROOT / "docs/js/anim" / s).read_text("utf-8")
                    if re.search(rf'window\.{name}\s*=', js):
                        defined = True
                        break
                self.assertTrue(
                    defined,
                    msg=f"Lesson {n} 挂载了 {name}.mount()，但其引用的动画脚本 {srcs} "
                        f"中没有 window.{name} 的定义——动画会空白。")

    def test_reduced_motion_supported(self):
        # 无障碍：动画应尊重 prefers-reduced-motion（在共享 CSS 中）。
        css = (_ROOT / "docs/css/course.css").read_text("utf-8")
        self.assertIn("prefers-reduced-motion", css,
                      msg="应支持 prefers-reduced-motion（无障碍要求）。")


class TestHomePage(unittest.TestCase):
    def test_home_lists_lesson_0(self):
        html = (_ROOT / "docs/index.html").read_text("utf-8")
        self.assertIn("Lesson 0", html)
        self.assertIn("Build Your Own Mini-vLLM", html)


if __name__ == "__main__":
    unittest.main()
