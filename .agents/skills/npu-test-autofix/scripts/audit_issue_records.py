#!/usr/bin/env python3
"""Audit issue records against every NG case from the baseline round."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


CATEGORIES = {
    "PRODUCT_CODE",
    "NPU_ADAPTATION",
    "TEST_SCRIPT",
    "WORKFLOW_SCRIPT",
    "ENVIRONMENT",
    "DEPENDENCY",
    "ASSET",
    "RESOURCE",
    "TIMEOUT_FLAKY",
}
FINAL_STATUSES = {
    "RESOLVED",
    "UNRESOLVED",
    "EXTERNAL_BLOCKED",
    "SKIPPED_AFTER_MAX_ROUNDS",
}
REQUIRED_MARKERS = (
    ("## 尝试记录", "## Attempts"),
    ("- 假设：", "- Hypothesis:"),
    ("- 最小修改范围：", "- Minimal change scope:"),
    ("- 执行命令：", "- Commands run:"),
    ("- 结果：", "- Result:"),
    ("## 最终状态", "## Final Status"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--state-dir", default=".npu-test-autofix")
    parser.add_argument("--output", default=".npu-test-autofix/issue-audit.json")
    return parser.parse_args()


def field(text: str, *names: str) -> str | None:
    alternatives = "|".join(re.escape(name) for name in names)
    match = re.search(
        rf"^- (?:{alternatives})[：:]\s*`?([A-Z_]+)`?\s*$",
        text,
        flags=re.MULTILINE,
    )
    return match.group(1) if match else None


def audit(state_dir: Path) -> dict:
    round_one = state_dir / "rounds" / "round-1.json"
    if not round_one.is_file():
        raise FileNotFoundError(f"Baseline result is missing: {round_one}")
    baseline = json.loads(round_one.read_text(encoding="utf-8"))
    baseline_ng = sorted(
        item["test_id"]
        for item in baseline["latest_results"]
        if item["status"] != "PASS"
    )
    issue_dir = state_dir / "issues"
    records = []
    errors: list[str] = []
    covered: set[str] = set()
    categories: Counter[str] = Counter()
    statuses: Counter[str] = Counter()

    for path in sorted(issue_dir.glob("ISSUE-*.md")) if issue_dir.is_dir() else []:
        text = path.read_text(encoding="utf-8")
        category = field(text, "类别", "Category")
        status = field(text, "状态", "Status")
        if category not in CATEGORIES:
            errors.append(f"{path.name}: invalid or missing Category")
        else:
            categories[category] += 1
        if status not in FINAL_STATUSES:
            errors.append(f"{path.name}: Status must be final, got {status!r}")
        else:
            statuses[status] += 1
        for markers in REQUIRED_MARKERS:
            if not any(marker in text for marker in markers):
                errors.append(f"{path.name}: missing marker {markers[0]!r}")
        cases = [case_id for case_id in baseline_ng if case_id in text]
        covered.update(cases)
        records.append(
            {
                "issue_id": path.stem,
                "category": category,
                "status": status,
                "baseline_ng_cases": cases,
                "path": path.as_posix(),
            }
        )

    uncovered = sorted(set(baseline_ng) - covered)
    if uncovered:
        errors.append("Baseline NG cases without an issue record: " + ", ".join(uncovered))
    if not baseline_ng and records:
        errors.append("Issue records exist although the baseline has no NG cases")

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_ng_count": len(baseline_ng),
        "issue_count": len(records),
        "covered_baseline_ng_count": len(covered),
        "uncovered_baseline_ng_cases": uncovered,
        "counts_by_category": dict(sorted(categories.items())),
        "counts_by_status": dict(sorted(statuses.items())),
        "issues": records,
        "errors": errors,
        "passed": not errors,
    }


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    state_dir = (repo / args.state_dir).resolve()
    output = (repo / args.output).resolve()
    if repo not in state_dir.parents or repo not in output.parents:
        print("error: state-dir and output must be inside repo-root", file=sys.stderr)
        return 2
    try:
        result = audit(state_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"Issue audit: baseline NG={result['baseline_ng_count']}, "
        f"issues={result['issue_count']}, passed={result['passed']}"
    )
    for error in result["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
