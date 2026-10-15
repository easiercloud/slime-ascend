#!/usr/bin/env python3
"""Tests for final issue-record coverage auditing."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("audit_issue_records.py")
SPEC = importlib.util.spec_from_file_location("audit_issue_records", SCRIPT)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


ISSUE = """# ISSUE-001: Example

- Category: TEST_SCRIPT
- Status: RESOLVED
- Cases: tests/utils/test_example.py

## Attempts

- Hypothesis: fixture mismatch
- Minimal change scope: one fixture
- Commands run: pytest
- Result: PASS

## Final Status

- Outcome: resolved
"""


class IssueAuditTest(unittest.TestCase):
    def create_state(self, root: Path) -> Path:
        state = root / ".npu-test-autofix"
        (state / "rounds").mkdir(parents=True)
        (state / "issues").mkdir()
        (state / "rounds" / "round-1.json").write_text(
            json.dumps(
                {
                    "latest_results": [
                        {
                            "test_id": "tests/utils/test_example.py",
                            "status": "NG",
                        },
                        {
                            "test_id": "tests/utils/test_pass.py",
                            "status": "PASS",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        return state

    def test_complete_final_record_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = self.create_state(Path(temporary))
            (state / "issues" / "ISSUE-001.md").write_text(ISSUE, encoding="utf-8")
            result = AUDIT.audit(state)
            self.assertTrue(result["passed"])
            self.assertEqual(result["counts_by_category"], {"TEST_SCRIPT": 1})

    def test_missing_issue_fails_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = self.create_state(Path(temporary))
            result = AUDIT.audit(state)
            self.assertFalse(result["passed"])
            self.assertEqual(
                result["uncovered_baseline_ng_cases"],
                ["tests/utils/test_example.py"],
            )


if __name__ == "__main__":
    unittest.main()
