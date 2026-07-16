"""Lesson 18 检查：综合故障场景——正确、无泄漏、全部完成、无重复执行。"""
import unittest
from pathlib import Path
from experiments.lesson_18_final_challenge import run_experiment
from experiments._common import read_config
from mini_vllm.trace import Tracer

_ROOT = Path(__file__).resolve().parent.parent.parent


class TestFinalChallenge(unittest.TestCase):
    def test_all_checks_pass(self):
        cfg = read_config("configs/lesson_18_quick.json")
        rep = run_experiment(cfg, Tracer("quiet"))
        c = rep["checks"]
        self.assertTrue(c["correct_vs_reference"],
                        msg="综合场景输出必须与逐请求参考一致（无重复执行/无跨请求串用）。")
        self.assertTrue(c["all_requests_finished"], msg="所有请求都应完成（无 starvation）。")
        self.assertTrue(c["no_kv_leak"], msg="综合场景运行后不得有 KV 泄漏。")

    def test_report_beats_naive_processing(self):
        cfg = read_config("configs/lesson_18_quick.json")
        rep = run_experiment(cfg, Tracer("quiet"))
        # 引擎实际处理的 output token 数远小于 naive 基线重算的 token 数
        self.assertLess(rep["output_tokens"], rep["baseline_naive_processed_tokens"],
                        msg="引擎（KV Cache）应显著少于 naive 重算的处理量。")


if __name__ == "__main__":
    unittest.main()
