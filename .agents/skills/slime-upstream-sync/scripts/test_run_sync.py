from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_sync import (
    SyncError,
    changed_paths,
    load_agent_result,
    load_success_sha,
)


SHA = "a" * 40


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def valid_result() -> dict:
    return {
        "schema_version": 2,
        "status": "PASS",
        "mode": "full",
        "upstream_sha": SHA,
        "baseline_discovered": 3,
        "executed": 3,
        "pass": 3,
        "ng": 0,
        "unexecuted": 0,
        "missing_resources": 0,
        "timeout": 0,
        "replay_verified": True,
        "report_path": ".upstream-sync/audit-report.zh-CN.md",
        "conversion_plan_path": ".upstream-sync/conversion-plan.json",
        "validation_result_path": ".upstream-sync/validation-result.json",
        "environment_path": ".upstream-sync/environment.json",
        "change_assessment_path": ".upstream-sync/change-assessment.json",
    }


def test_load_agent_result_accepts_only_complete_pass(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_json(path, valid_result())
    assert load_agent_result(path, SHA)["pass"] == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "NG"),
        ("baseline_discovered", 0),
        ("timeout", 1),
        ("replay_verified", False),
    ],
)
def test_load_agent_result_rejects_non_pass_gates(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = valid_result()
    payload[field] = value
    path = tmp_path / "result.json"
    write_json(path, payload)
    with pytest.raises(SyncError, match="全量 PASS|baseline_discovered"):
        load_agent_result(path, SHA)


def test_load_agent_result_rejects_wrong_sha(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_json(path, valid_result())
    with pytest.raises(SyncError, match="upstream_sha"):
        load_agent_result(path, "b" * 40)


def test_load_agent_result_accepts_complete_trial(tmp_path: Path) -> None:
    payload = valid_result()
    payload.update(
        {
            "status": "TRIAL_COMPLETE",
            "mode": "trial_existing_assets",
            "baseline_discovered": 5,
            "executed": 3,
            "pass": 3,
            "ng": 0,
            "unexecuted": 2,
            "missing_resources": 2,
        }
    )
    path = tmp_path / "result.json"
    write_json(path, payload)
    assert load_agent_result(path, SHA, trial_mode=True)["unexecuted"] == 2


def test_load_agent_result_rejects_incomplete_trial(tmp_path: Path) -> None:
    payload = valid_result()
    payload.update(
        {
            "status": "TRIAL_COMPLETE",
            "mode": "trial_existing_assets",
            "baseline_discovered": 5,
            "executed": 3,
            "pass": 3,
            "ng": 0,
            "unexecuted": 2,
            "missing_resources": 1,
            "replay_verified": False,
        }
    )
    path = tmp_path / "result.json"
    write_json(path, payload)
    with pytest.raises(SyncError, match="试运行结果不完整"):
        load_agent_result(path, SHA, trial_mode=True)


def test_load_agent_result_allows_pre_replay_gate(tmp_path: Path) -> None:
    payload = valid_result()
    payload["replay_verified"] = False
    path = tmp_path / "result.json"
    write_json(path, payload)
    assert load_agent_result(path, SHA, require_replay=False)["pass"] == 3


def test_load_agent_result_rejects_inconsistent_counts(tmp_path: Path) -> None:
    payload = valid_result()
    payload["executed"] = 2
    path = tmp_path / "result.json"
    write_json(path, payload)
    with pytest.raises(SyncError, match="executed"):
        load_agent_result(path, SHA)


def test_load_success_sha_rejects_corrupt_marker(tmp_path: Path) -> None:
    marker = tmp_path / "last-success.json"
    write_json(marker, {"upstream_sha": "short"})
    with pytest.raises(SyncError, match="成功标记"):
        load_success_sha(marker)


def test_changed_paths_handles_spaces_and_renames(tmp_path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    original = tmp_path / "old name.txt"
    original.write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    original.rename(tmp_path / "new name.txt")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    assert changed_paths(tmp_path) == ["new name.txt"]
