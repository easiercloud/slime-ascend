#!/usr/bin/env python3
"""Collect the immutable upstream delta inspected at the start of a sync run."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


FEATURE_AREAS = {
    "rollout_and_sampling": ("slime/rollout/", "slime/backends/sglang", "slime/ray/rollout"),
    "training_and_optimization": ("slime/backends/megatron", "slime/ray/train", "slime/utils/ppo"),
    "models_and_checkpoints": ("models/", "checkpoint", "convert_hf", "weight"),
    "distributed_and_parallelism": ("distributed", "parallel", "placement_group", "actor_group"),
    "configuration_and_cli": ("arguments.py", "config", "cli", "examples/"),
    "dependencies_and_build": ("docker/", "requirements", "pyproject.toml", "setup.py"),
    "tests_and_ci": ("tests/", ".github/workflows/"),
    "documentation": ("docs/", "README"),
}
DEPENDENCY_PATTERN = re.compile(
    r"SGLANG|MEGATRON|ROUTER|torch|transformers|ray|flash|kernel|memfabric|nixl|mooncake|"
    r"mbridge|modelopt|requirements|pip install|git checkout",
    re.IGNORECASE,
)
DEPENDENCY_FILES = (
    "docker/Dockerfile",
    "pyproject.toml",
    "requirements.txt",
    "requirements",
    "docker/requirements",
)


def git(repo: Path, *args: str, check: bool = True) -> str:
    process = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if check and process.returncode:
        raise RuntimeError(process.stderr.strip() or "git command failed")
    return process.stdout


def ref_file(repo: Path, ref: str, path: str) -> str | None:
    process = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    return process.stdout if process.returncode == 0 else None


def dependency_snapshot(repo: Path, ref: str) -> dict[str, list[str]]:
    paths = git(repo, "ls-tree", "-r", "--name-only", ref).splitlines()
    selected = [
        path
        for path in paths
        if any(path == prefix or path.startswith(prefix + "/") for prefix in DEPENDENCY_FILES)
    ]
    snapshot: dict[str, list[str]] = {}
    for path in selected:
        text = ref_file(repo, ref, path)
        if text is None:
            continue
        lines = [line.strip() for line in text.splitlines() if DEPENDENCY_PATTERN.search(line)]
        if lines:
            snapshot[path] = lines
    return snapshot


def changed_entries(repo: Path, base: str, target: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in git(repo, "diff", "--name-status", "--find-renames", base, target).splitlines():
        fields = line.split("\t")
        status = fields[0]
        if status.startswith(("R", "C")) and len(fields) == 3:
            entries.append({"status": status, "old_path": fields[1], "path": fields[2]})
        elif len(fields) == 2:
            entries.append({"status": status, "path": fields[1]})
    return entries


def classify_features(entries: list[dict[str, str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for area, markers in FEATURE_AREAS.items():
        paths = sorted(
            {
                item["path"]
                for item in entries
                if any(marker in item["path"] for marker in markers)
            }
        )
        if paths:
            result[area] = paths
    return result


def test_delta(entries: list[dict[str, str]], repo: Path, target: str) -> dict:
    changed = [item for item in entries if item["path"].startswith("tests/")]
    by_status = {
        "added": [item["path"] for item in changed if item["status"] == "A"],
        "modified": [item["path"] for item in changed if item["status"] == "M"],
        "deleted": [item["path"] for item in changed if item["status"] == "D"],
        "renamed": [item for item in changed if item["status"].startswith("R")],
    }
    target_tests = [
        path
        for path in git(repo, "ls-tree", "-r", "--name-only", target, "tests").splitlines()
        if Path(path).name.startswith("test_") and path.endswith(".py")
    ]
    ci = [item for item in entries if item["path"].startswith(".github/workflows/")]
    return {
        "changed": bool(changed or ci),
        **by_status,
        "ci_changes": ci,
        "target_test_count": len(target_tests),
        "target_tests": target_tests,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    base = git(repo, "rev-parse", f"{args.base}^{{commit}}").strip()
    target = git(repo, "rev-parse", f"{args.target}^{{commit}}").strip()
    entries = changed_entries(repo, base, target)
    commits = []
    for line in git(repo, "log", "--format=%H%x09%s", f"{base}..{target}").splitlines():
        sha, _, subject = line.partition("\t")
        commits.append({"sha": sha, "subject": subject})
    before_dependencies = dependency_snapshot(repo, base)
    after_dependencies = dependency_snapshot(repo, target)
    dependency_paths = sorted(
        item["path"]
        for item in entries
        if item["path"].startswith(("docker/", "requirements", "pyproject.toml"))
    )
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_sha": base,
        "target_sha": target,
        "upstream_changed": base != target,
        "commit_count": len(commits),
        "commits": commits,
        "source_changes": {
            "changed_file_count": len(entries),
            "files": entries,
            "feature_candidates": classify_features(entries),
            "feature_assessment": [],
        },
        "dependency_changes": {
            "changed": before_dependencies != after_dependencies or bool(dependency_paths),
            "changed_files": dependency_paths,
            "before": before_dependencies,
            "after": after_dependencies,
        },
        "test_changes": test_delta(entries, repo, target),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"base={base} target={target} commits={len(commits)} "
        f"files={len(entries)} tests_changed={payload['test_changes']['changed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
