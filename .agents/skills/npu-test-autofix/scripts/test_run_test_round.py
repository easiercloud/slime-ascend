#!/usr/bin/env python3
"""Tests for NPU test discovery and ST/UT classification."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_test_round.py")
SPEC = importlib.util.spec_from_file_location("run_test_round", SCRIPT)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class DiscoveryAndClassificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[4]

    def test_complete_tests_root_includes_utils_cases(self) -> None:
        discovered = RUNNER.discover(self.repo, ["slime/tests", "tests"], [], [])
        ids = {RUNNER.test_id(self.repo, path) for path in discovered}
        self.assertIn("slime/tests/utils/test_hf_checkpoint_saver.py", ids)
        self.assertIn(
            "tests/tests_npu/st/test_qwen2.5_0.5B_gsm8k_async_short_npu.py",
            ids,
        )

    def test_classifies_utils_as_ut_and_training_as_st(self) -> None:
        utility = self.repo / "slime/tests/utils/test_hf_checkpoint_saver.py"
        training = (
            self.repo
            / "tests/tests_npu/st/test_qwen2.5_0.5B_gsm8k_async_short_npu.py"
        )
        self.assertEqual(RUNNER.classify_test(self.repo, utility), "UT")
        self.assertEqual(RUNNER.classify_test(self.repo, training), "ST")

    def test_ut_uses_pytest_and_st_executes_directly(self) -> None:
        path = self.repo / "slime/tests/utils/test_hf_checkpoint_saver.py"
        self.assertEqual(
            RUNNER.command_for(path, "python3", "UT")[:3],
            ["python3", "-m", "pytest"],
        )
        self.assertEqual(
            RUNNER.command_for(path, "python3", "ST"),
            ["python3", str(path)],
        )

    def test_default_npu_environment_sets_coredump_signal_without_override(self) -> None:
        previous = os.environ.get("ASCEND_COREDUMP_SIGNAL")
        try:
            os.environ.pop("ASCEND_COREDUMP_SIGNAL", None)
            RUNNER.configure_default_npu_environment()
            self.assertEqual(os.environ["ASCEND_COREDUMP_SIGNAL"], "none")
            os.environ["ASCEND_COREDUMP_SIGNAL"] = "custom"
            RUNNER.configure_default_npu_environment()
            self.assertEqual(os.environ["ASCEND_COREDUMP_SIGNAL"], "custom")
        finally:
            if previous is None:
                os.environ.pop("ASCEND_COREDUMP_SIGNAL", None)
            else:
                os.environ["ASCEND_COREDUMP_SIGNAL"] = previous

    def test_failed_or_stopped_child_job_is_fail_closed(self) -> None:
        self.assertIsNotNone(
            RUNNER.FAILED_CHILD_JOB_RE.search("Job 'raysubmit_abc' failed")
        )
        self.assertIsNotNone(
            RUNNER.FAILED_CHILD_JOB_RE.search("Job 'raysubmit_abc' was stopped")
        )
        self.assertIsNone(
            RUNNER.FAILED_CHILD_JOB_RE.search("Job 'raysubmit_abc' succeeded")
        )



if __name__ == "__main__":
    unittest.main()
