#!/usr/bin/env python3
"""Fail-closed upstream synchronization orchestrator for slime-ascend."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = 2
SUCCESS_STATUSES = {"PASS"}


class SyncError(RuntimeError):
    """An expected workflow failure that must not update the success marker."""


@dataclass(frozen=True)
class Config:
    ascend_repo_url: str
    ascend_branch: str
    upstream_repo_url: str
    upstream_branch: str
    submodule_path: str
    state_dir: Path
    work_root: Path
    prompt_template: Path
    prepare_command: str
    finalize_command: str
    agent_command: tuple[str, ...]
    push_remote: str
    push_branch_prefix: str
    success_tag_prefix: str
    keep_workspace: bool

    @classmethod
    def from_env(cls, script_root: Path) -> "Config":
        repo_url = os.environ.get("SLIME_ASCEND_REPO_URL", "").strip()
        if not repo_url:
            raise SyncError("缺少必填环境变量 SLIME_ASCEND_REPO_URL")

        agent_command = tuple(
            shlex.split(os.environ.get("SLIME_SYNC_AGENT_COMMAND", "codex exec"))
        )
        if not agent_command:
            raise SyncError("SLIME_SYNC_AGENT_COMMAND 不能为空")

        return cls(
            ascend_repo_url=repo_url,
            ascend_branch=os.environ.get(
                "SLIME_ASCEND_BASE_BRANCH", "refactor"
            ).strip(),
            upstream_repo_url=os.environ.get(
                "SLIME_UPSTREAM_REPO_URL", "https://github.com/THUDM/slime.git"
            ).strip(),
            upstream_branch=os.environ.get(
                "SLIME_UPSTREAM_BRANCH", "main"
            ).strip(),
            submodule_path=os.environ.get(
                "SLIME_UPSTREAM_SUBMODULE_PATH", "slime"
            ).strip(),
            state_dir=Path(
                os.environ.get(
                    "SLIME_SYNC_STATE_DIR", "/var/lib/slime-upstream-sync"
                )
            ),
            work_root=Path(
                os.environ.get(
                    "SLIME_SYNC_WORK_ROOT", "/var/tmp/slime-upstream-sync"
                )
            ),
            prompt_template=Path(
                os.environ.get(
                    "SLIME_SYNC_PROMPT_TEMPLATE",
                    str(script_root.parent / "references" / "agent_prompt.md"),
                )
            ),
            prepare_command=os.environ.get(
                "SLIME_SYNC_PREPARE_COMMAND",
                "bash .agents/skills/slime-upstream-sync/scripts/materialize_overlay.sh",
            ).strip(),
            finalize_command=os.environ.get(
                "SLIME_SYNC_FINALIZE_COMMAND",
                "bash .agents/skills/slime-upstream-sync/scripts/finalize_overlay.sh",
            ).strip(),
            agent_command=agent_command,
            push_remote=os.environ.get("SLIME_SYNC_PUSH_REMOTE", "origin").strip(),
            push_branch_prefix=os.environ.get(
                "SLIME_SYNC_PUSH_BRANCH_PREFIX", "sync/upstream-slime"
            ).strip(),
            success_tag_prefix=os.environ.get(
                "SLIME_SYNC_SUCCESS_TAG_PREFIX", "sync-success/slime-main"
            ).strip(),
            keep_workspace=os.environ.get(
                "SLIME_SYNC_KEEP_WORKSPACE", "0"
            ).lower()
            in {"1", "true", "yes"},
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> str:
    process = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if process.returncode:
        detail = (process.stderr or process.stdout or "").strip()
        raise SyncError(
            f"命令失败({process.returncode}): {shlex.join(argv)}"
            + (f"\n{detail}" if detail else "")
        )
    return (process.stdout or "").strip()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def remote_sha(repo_url: str, branch: str) -> str:
    output = run(
        ["git", "ls-remote", "--heads", repo_url, f"refs/heads/{branch}"],
        capture=True,
    )
    fields = output.split()
    if len(fields) != 2 or fields[1] != f"refs/heads/{branch}":
        raise SyncError(f"远端分支不存在或结果异常: {repo_url}#{branch}")
    return fields[0]


def remote_tag_exists(repo_url: str, tag: str) -> bool:
    output = run(
        ["git", "ls-remote", "--tags", repo_url, f"refs/tags/{tag}"],
        capture=True,
    )
    return bool(output.strip())


def load_success_sha(marker: Path) -> str | None:
    if not marker.exists():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError(f"成功标记损坏: {marker}: {exc}") from exc
    sha = payload.get("upstream_sha")
    if not isinstance(sha, str) or len(sha) != 40:
        raise SyncError(f"成功标记缺少合法 upstream_sha: {marker}")
    return sha


def load_agent_result(
    path: Path,
    expected_sha: str,
    *,
    trial_mode: bool = False,
    require_replay: bool = True,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SyncError(f"代理未生成机器结果: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SyncError(f"代理机器结果不是合法 JSON: {path}: {exc}") from exc

    required = {
        "schema_version",
        "status",
        "mode",
        "upstream_sha",
        "baseline_discovered",
        "executed",
        "pass",
        "ng",
        "unexecuted",
        "missing_resources",
        "timeout",
        "replay_verified",
        "report_path",
        "conversion_plan_path",
        "validation_result_path",
        "environment_path",
        "change_assessment_path",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise SyncError(f"代理机器结果缺少字段: {', '.join(missing)}")
    if payload["schema_version"] != RESULT_SCHEMA_VERSION:
        raise SyncError("代理机器结果 schema_version 不兼容")
    if payload["upstream_sha"] != expected_sha:
        raise SyncError("代理验证的 upstream_sha 与本轮目标不一致")
    expected_mode = "trial_existing_assets" if trial_mode else "full"
    if payload["mode"] != expected_mode:
        raise SyncError(f"代理机器结果 mode 必须是 {expected_mode}")
    for name in (
        "status",
        "mode",
        "upstream_sha",
        "report_path",
        "conversion_plan_path",
        "validation_result_path",
        "environment_path",
        "change_assessment_path",
    ):
        if not isinstance(payload[name], str) or not payload[name]:
            raise SyncError(f"代理机器结果字段 {name} 必须是非空字符串")
    if type(payload["replay_verified"]) is not bool:
        raise SyncError("代理机器结果 replay_verified 必须是布尔值")

    counts = (
        "baseline_discovered",
        "executed",
        "pass",
        "ng",
        "unexecuted",
        "missing_resources",
        "timeout",
    )
    if any(type(payload[name]) is not int or payload[name] < 0 for name in counts):
        raise SyncError("代理机器结果的计数字段必须是非负整数")
    if payload["executed"] != payload["pass"] + payload["ng"]:
        raise SyncError("代理机器结果 executed 必须等于 pass + ng")
    if (
        payload["pass"] + payload["ng"] + payload["unexecuted"]
        != payload["baseline_discovered"]
    ):
        raise SyncError("代理机器结果各状态计数与 baseline_discovered 不一致")
    if payload["missing_resources"] > payload["ng"] + payload["unexecuted"]:
        raise SyncError("missing_resources 不能大于 ng + unexecuted")
    if trial_mode:
        complete_trial = (
            payload["status"] == "TRIAL_COMPLETE"
            and payload["baseline_discovered"] > 0
            and (not require_replay or payload["replay_verified"] is True)
        )
        if not complete_trial:
            raise SyncError("试运行结果不完整，禁止将其记录为完成")
    else:
        all_pass = (
            payload["status"] in SUCCESS_STATUSES
            and payload["baseline_discovered"] > 0
            and payload["pass"] == payload["baseline_discovered"]
            and payload["executed"] == payload["baseline_discovered"]
            and payload["ng"] == 0
            and payload["unexecuted"] == 0
            and payload["missing_resources"] == 0
            and payload["timeout"] == 0
            and (not require_replay or payload["replay_verified"] is True)
        )
        if not all_pass:
            raise SyncError("全量 PASS 门禁未满足，禁止推送和更新成功标记")
    return payload


def ensure_report(workspace: Path, result: dict[str, Any]) -> Path:
    if not isinstance(result.get("report_path"), str):
        raise SyncError("report_path 必须是字符串")
    report = (workspace / result["report_path"]).resolve()
    if workspace.resolve() not in report.parents:
        raise SyncError("report_path 必须位于独立工作区内")
    if not report.is_file() or report.stat().st_size == 0:
        raise SyncError(f"中文审计报告不存在或为空: {report}")
    text = report.read_text(encoding="utf-8")
    if not any("\u4e00" <= char <= "\u9fff" for char in text):
        raise SyncError("审计报告未包含中文内容")
    return report


def ensure_machine_artifacts(workspace: Path, result: dict[str, Any]) -> None:
    workspace = workspace.resolve()
    for field in (
        "conversion_plan_path",
        "validation_result_path",
        "environment_path",
        "change_assessment_path",
    ):
        value = result.get(field)
        if not isinstance(value, str) or not value:
            raise SyncError(f"{field} 必须是非空字符串")
        path = (workspace / value).resolve()
        if workspace not in path.parents:
            raise SyncError(f"{field} 必须位于独立工作区内")
        if not path.is_file() or path.stat().st_size == 0:
            raise SyncError(f"机器审计产物不存在或为空: {path}")


def render_prompt(
    template: Path,
    *,
    workspace: Path,
    previous_sha: str | None,
    upstream_sha: str,
    submodule_path: str,
    run_mode: str,
) -> str:
    text = template.read_text(encoding="utf-8")
    replacements = {
        "{{WORKSPACE}}": str(workspace),
        "{{PREVIOUS_SHA}}": previous_sha or "NONE",
        "{{UPSTREAM_SHA}}": upstream_sha,
        "{{SUBMODULE_PATH}}": submodule_path,
        "{{RUN_MODE}}": run_mode,
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def changed_paths(workspace: Path) -> list[str]:
    output = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--ignore-submodules=none"],
        cwd=workspace,
        check=True,
        capture_output=True,
    ).stdout
    entries = output.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        status = entry[:2]
        path = entry[3:].decode("utf-8", errors="surrogateescape")
        paths.append(path)
        if status[:1] in {b"R", b"C"}:
            index += 1
    return paths


def git_change_gate(
    workspace: Path, submodule_path: str, expected_upstream_sha: str
) -> None:
    paths = changed_paths(workspace)
    if not paths:
        raise SyncError("验证通过但没有可提交变更，拒绝创建空同步分支")
    allowed_exact = {
        submodule_path,
        ".upstream-sync/audit-report.zh-CN.md",
        ".upstream-sync/result.json",
        ".upstream-sync/change-assessment.json",
    }
    allowed_prefixes = ("patches/", "overlay/", ".upstream-sync/replay/")
    unexpected = sorted(
        path
        for path in paths
        if path not in allowed_exact
        and not any(path.startswith(prefix) for prefix in allowed_prefixes)
    )
    if unexpected:
        raise SyncError(
            "发现允许范围外的改动，拒绝推送: " + ", ".join(unexpected)
        )
    submodule_status = run(
        ["git", "status", "--porcelain=v1"],
        cwd=workspace / submodule_path,
        capture=True,
    )
    if submodule_status:
        raise SyncError("finalize 后 slime 子模块仍有未提交修改，拒绝推送悬空状态")
    actual_sha = run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace / submodule_path,
        capture=True,
    )
    if actual_sha != expected_upstream_sha:
        raise SyncError("finalize 后 slime 子模块未指向本轮目标 upstream SHA")


def preserve_artifacts(
    workspace: Path | None, state_dir: Path, run_id: str
) -> None:
    if workspace is None:
        return
    source_dir = workspace / ".upstream-sync"
    if not source_dir.is_dir():
        return
    destination = state_dir / "artifacts" / run_id
    for name in (
        "result.json",
        "audit-report.zh-CN.md",
        "conversion-plan.json",
        "validation-result.json",
        "environment.json",
        "change-assessment.json",
    ):
        source = source_dir / name
        if source.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / name)
    replay = source_dir / "replay"
    if replay.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(replay, destination / "replay", dirs_exist_ok=True)


def execute_hook(command: str, workspace: Path, env: dict[str, str]) -> None:
    if not command:
        raise SyncError("准备/收尾命令不能为空")
    run(["bash", "-euo", "pipefail", "-c", command], cwd=workspace, env=env)


def write_run_audit(
    audit_path: Path,
    *,
    run_id: str,
    status: str,
    upstream_sha: str | None,
    previous_sha: str | None,
    detail: str,
    workspace: Path | None = None,
    pushed_branch: str | None = None,
) -> None:
    atomic_json(
        audit_path,
        {
            "schema_version": 1,
            "run_id": run_id,
            "status": status,
            "upstream_sha": upstream_sha,
            "previous_success_sha": previous_sha,
            "detail": detail,
            "workspace": str(workspace) if workspace else None,
            "pushed_branch": pushed_branch,
            "updated_at_utc": utc_now().isoformat(),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="只检查命令、模板、目录和远端分支，不克隆、不运行代理、不推送",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="完成准备和代理验证，但不提交、不推送、不更新成功标记",
    )
    parser.add_argument(
        "--trial-existing-assets",
        action="store_true",
        help=(
            "只执行本机现有资产覆盖的用例；缺资源项记为未执行；"
            "无论结果如何都不推送、不更新成功标记"
        ),
    )
    return parser.parse_args()


def preflight(config: Config) -> tuple[str, str]:
    for command in ("git", config.agent_command[0], "bash"):
        if shutil.which(command) is None:
            raise SyncError(f"缺少命令: {command}")
    if not config.prompt_template.is_file():
        raise SyncError(f"代理提示模板不存在: {config.prompt_template}")
    run(
        [
            "git",
            "check-ref-format",
            f"refs/tags/{config.success_tag_prefix}-{'0' * 40}",
        ]
    )
    ascend_sha = remote_sha(config.ascend_repo_url, config.ascend_branch)
    upstream_sha = remote_sha(config.upstream_repo_url, config.upstream_branch)
    return ascend_sha, upstream_sha


def main() -> int:
    args = parse_args()
    script_root = Path(__file__).resolve().parent
    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ")
    workspace: Path | None = None
    upstream_sha: str | None = None
    previous_sha: str | None = None
    config: Config | None = None
    run_succeeded = False

    try:
        config = Config.from_env(script_root)
        config.state_dir.mkdir(parents=True, exist_ok=True)
        config.work_root.mkdir(parents=True, exist_ok=True)
        audit_path = config.state_dir / "runs" / f"{run_id}.json"
        latest_audit = config.state_dir / "latest-run.json"
        marker = config.state_dir / "last-success.json"

        lock_path = config.state_dir / "sync.lock"
        with lock_path.open("a+", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SyncError("已有同步任务运行，本轮跳过") from exc

            ascend_sha, upstream_sha = preflight(config)
            previous_sha = load_success_sha(marker)
            success_tag = f"{config.success_tag_prefix}-{upstream_sha}"
            if args.check_config:
                detail = (
                    f"配置检查通过；ascend={ascend_sha}；upstream={upstream_sha}"
                )
                write_run_audit(
                    audit_path,
                    run_id=run_id,
                    status="CHECKED",
                    upstream_sha=upstream_sha,
                    previous_sha=previous_sha,
                    detail=detail,
                )
                shutil.copy2(audit_path, latest_audit)
                print(detail)
                return 0

            if remote_tag_exists(config.ascend_repo_url, success_tag):
                detail = "上游 SHA 与远端成功标签相同，已跳过"
                if previous_sha != upstream_sha:
                    atomic_json(
                        marker,
                        {
                            "schema_version": 1,
                            "upstream_repo": config.upstream_repo_url,
                            "upstream_branch": config.upstream_branch,
                            "upstream_sha": upstream_sha,
                            "success_tag": success_tag,
                            "recovered_from_remote": True,
                            "validated_at_utc": utc_now().isoformat(),
                        },
                    )
                write_run_audit(
                    audit_path,
                    run_id=run_id,
                    status="SKIPPED_UNCHANGED",
                    upstream_sha=upstream_sha,
                    previous_sha=previous_sha,
                    detail=detail,
                )
                shutil.copy2(audit_path, latest_audit)
                print(detail)
                return 0

            workspace = Path(
                tempfile.mkdtemp(prefix=f"run-{run_id}-", dir=config.work_root)
            )
            run(
                [
                    "git",
                    "clone",
                    "--branch",
                    config.ascend_branch,
                    "--single-branch",
                    "--recurse-submodules",
                    config.ascend_repo_url,
                    str(workspace),
                ]
            )
            submodule = workspace / config.submodule_path
            if not submodule.is_dir():
                raise SyncError(f"slime 子模块目录不存在: {submodule}")
            run(
                ["git", "submodule", "update", "--init", "--recursive"],
                cwd=workspace,
            )
            base_submodule_sha = run(
                ["git", "rev-parse", "HEAD"], cwd=submodule, capture=True
            )
            run(["git", "fetch", config.upstream_repo_url, upstream_sha], cwd=submodule)
            assessment_base_sha = previous_sha or base_submodule_sha
            if previous_sha and previous_sha != base_submodule_sha:
                run(["git", "fetch", config.upstream_repo_url, previous_sha], cwd=submodule)
            run(["git", "checkout", "--detach", upstream_sha], cwd=submodule)

            change_assessment = workspace / ".upstream-sync" / "change-assessment.json"
            run(
                [
                    sys.executable,
                    str(script_root / "collect_change_assessment.py"),
                    "--repo",
                    str(submodule),
                    "--base",
                    assessment_base_sha,
                    "--target",
                    upstream_sha,
                    "--output",
                    str(change_assessment),
                ],
                cwd=workspace,
            )

            hook_env = os.environ.copy()
            hook_env.update(
                {
                    "SLIME_SYNC_WORKSPACE": str(workspace),
                    "SLIME_SYNC_SUBMODULE": str(submodule),
                    "SLIME_SYNC_PREVIOUS_SHA": previous_sha or "",
                    "SLIME_SYNC_BASE_SHA": base_submodule_sha,
                    "SLIME_SYNC_UPSTREAM_SHA": upstream_sha,
                    "SLIME_SYNC_RUN_ID": run_id,
                }
            )
            execute_hook(config.prepare_command, workspace, hook_env)

            result_path = workspace / ".upstream-sync" / "result.json"
            report_path = workspace / ".upstream-sync" / "audit-report.zh-CN.md"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            prompt = render_prompt(
                config.prompt_template,
                workspace=workspace,
                previous_sha=previous_sha,
                upstream_sha=upstream_sha,
                submodule_path=config.submodule_path,
                run_mode=(
                    "trial_existing_assets"
                    if args.trial_existing_assets
                    else "full"
                ),
            )
            agent_env = hook_env.copy()
            agent_env.update(
                {
                    "SLIME_SYNC_RESULT_PATH": str(result_path),
                    "SLIME_SYNC_REPORT_PATH": str(report_path),
                }
            )
            run(
                [
                    *config.agent_command,
                    "--cd",
                    str(workspace),
                    "--config",
                    'approval_policy="never"',
                    "--sandbox",
                    "danger-full-access",
                    "--ephemeral",
                    "--color",
                    "never",
                    prompt,
                ],
                cwd=workspace,
                env=agent_env,
            )

            result = load_agent_result(
                result_path,
                upstream_sha,
                trial_mode=args.trial_existing_assets,
                require_replay=False,
            )
            ensure_report(workspace, result)
            ensure_machine_artifacts(workspace, result)
            execute_hook(config.finalize_command, workspace, hook_env)
            result = load_agent_result(
                result_path,
                upstream_sha,
                trial_mode=args.trial_existing_assets,
                require_replay=True,
            )
            git_change_gate(workspace, config.submodule_path, upstream_sha)

            if args.trial_existing_assets:
                detail = (
                    "已有资源试运行完成；未推送、未更新远端成功标签；"
                    f"PASS={result['pass']} NG={result['ng']} "
                    f"UNEXECUTED={result['unexecuted']}"
                )
                write_run_audit(
                    audit_path,
                    run_id=run_id,
                    status="TRIAL_COMPLETE_NO_PUSH",
                    upstream_sha=upstream_sha,
                    previous_sha=previous_sha,
                    detail=detail,
                    workspace=workspace,
                )
                shutil.copy2(audit_path, latest_audit)
                print(detail)
                run_succeeded = True
                return 0

            if args.no_push:
                detail = "全量验证通过；因 --no-push 未提交、未推送、未更新成功标记"
                write_run_audit(
                    audit_path,
                    run_id=run_id,
                    status="PASS_NO_PUSH",
                    upstream_sha=upstream_sha,
                    previous_sha=previous_sha,
                    detail=detail,
                    workspace=workspace,
                )
                shutil.copy2(audit_path, latest_audit)
                print(detail)
                run_succeeded = True
                return 0

            branch = f"{config.push_branch_prefix}-{upstream_sha[:12]}"
            run(["git", "switch", "-c", branch], cwd=workspace)
            run(["git", "add", "-A"], cwd=workspace)
            run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"chore: sync upstream slime {upstream_sha[:12]}",
                ],
                cwd=workspace,
            )
            run(
                ["git", "tag", success_tag],
                cwd=workspace,
            )
            run(
                [
                    "git",
                    "push",
                    "--atomic",
                    config.push_remote,
                    f"HEAD:refs/heads/{branch}",
                    f"refs/tags/{success_tag}",
                ],
                cwd=workspace,
            )
            atomic_json(
                marker,
                {
                    "schema_version": 1,
                    "upstream_repo": config.upstream_repo_url,
                    "upstream_branch": config.upstream_branch,
                    "upstream_sha": upstream_sha,
                    "ascend_base_branch": config.ascend_branch,
                    "pushed_branch": branch,
                    "success_tag": success_tag,
                    "validated_at_utc": utc_now().isoformat(),
                    "result": result,
                },
            )
            detail = "全量 PASS，已推送新分支并更新成功标记"
            write_run_audit(
                audit_path,
                run_id=run_id,
                status="PUSHED",
                upstream_sha=upstream_sha,
                previous_sha=previous_sha,
                detail=detail,
                workspace=workspace,
                pushed_branch=branch,
            )
            shutil.copy2(audit_path, latest_audit)
            print(f"{detail}: {branch}")
            run_succeeded = True
            return 0
    except (OSError, SyncError) as exc:
        detail = str(exc)
        print(f"同步失败（未推送、未更新成功标记）: {detail}", file=sys.stderr)
        if config is not None:
            audit_path = config.state_dir / "runs" / f"{run_id}.json"
            write_run_audit(
                audit_path,
                run_id=run_id,
                status="FAILED",
                upstream_sha=upstream_sha,
                previous_sha=previous_sha,
                detail=detail,
                workspace=workspace,
            )
            shutil.copy2(audit_path, config.state_dir / "latest-run.json")
        return 1
    finally:
        if config is not None:
            try:
                preserve_artifacts(workspace, config.state_dir, run_id)
            except OSError as exc:
                print(f"保留审计产物失败: {exc}", file=sys.stderr)
        if (
            workspace is not None
            and workspace.exists()
            and config is not None
            and not config.keep_workspace
            and run_succeeded
        ):
            shutil.rmtree(workspace)


if __name__ == "__main__":
    raise SystemExit(main())
