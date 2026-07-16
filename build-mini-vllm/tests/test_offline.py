"""离线交付测试：verify-offline-bundle 的逻辑必须真实地发现问题。

不仅测「通过」，还测「该失败时会失败」——用一个临时缺文件的副本目录验证
verifier 会明确报错，而不是静默放过（Prompt 要求：必需文件缺失必须失败）。
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mini_vllm.offline_check import verify_bundle, REQUIRED_FILES

_ROOT = Path(__file__).resolve().parent.parent


class TestOfflineBundle(unittest.TestCase):
    def test_real_repo_passes(self):
        report = verify_bundle(_ROOT, check_wheels=True)
        failed = [c.name for c in report.checks if not c.ok]
        self.assertTrue(
            report.ok,
            msg="真实仓库应通过离线自检；未通过的检查："
                + str(failed)
                + "。逐项细节见 `python3 course.py verify-offline-bundle`。",
        )

    def test_required_files_all_present(self):
        missing = [f for f in REQUIRED_FILES if not (_ROOT / f).exists()]
        self.assertEqual(
            missing, [],
            msg="以下运行时必需文件缺失：" + str(missing),
        )

    def test_core_requires_zero_wheels(self):
        # 核心为纯标准库：required wheel 集合必须为空，Level A 才能天然满足。
        data = json.loads((_ROOT / "offline/manifests/wheels.json").read_text("utf-8"))
        self.assertEqual(
            data.get("required", None), [],
            msg="核心不应依赖任何 wheel（required 应为 []）；"
                "torch 等属于 optional bundle。检查 offline/manifests/wheels.json。",
        )

    def test_verifier_fails_when_required_file_missing(self):
        # 复制一份仓库到临时目录，删掉一个必需文件，verifier 必须失败。
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "repo"
            shutil.copytree(
                _ROOT, dst,
                ignore=shutil.ignore_patterns(
                    ".git", "__pycache__", "outputs", ".venv*", "dist", "*.pyc"
                ),
            )
            victim = dst / "docs/lessons/lesson_00.html"
            self.assertTrue(victim.exists())
            victim.unlink()
            report = verify_bundle(dst, check_wheels=True)
            self.assertFalse(
                report.ok,
                msg="删除必需文件后 verifier 仍报通过——这会让「完全离线可用」的"
                    "承诺变成谎言。检查 mini_vllm/offline_check.py 的必需文件检查。",
            )


if __name__ == "__main__":
    unittest.main()
