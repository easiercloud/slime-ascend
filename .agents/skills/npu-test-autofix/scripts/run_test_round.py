#!/usr/bin/env python3
"""Run one deterministic round of NPU tests and persist structured results."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

MAX_ROUNDS = 5
# The v0.3.1 delivery layout keeps generic upstream tests in the pristine
# Slime submodule and converted NPU tests in the outer repository.
DEFAULT_ROOTS = ("slime/tests", "tests")
SKIP_NAMES = {"run_all_npu_tests.py", "run_all_ut.py", "ci_check.py", "convert_weights.py"}
TEST_TYPES = ("ST", "UT", "UNCLASSIFIED")
FAILED_CHILD_JOB_RE = re.compile(
    r"Job ['\"]?[^'\"\n]+['\"]? (?:failed|was stopped)", re.IGNORECASE
)
DEFAULT_NPU_ENV = {
    "ASCEND_COREDUMP_SIGNAL": "none",
    # Avoid HCCL's default fixed listener port (16666): stale or concurrent
    # rollout workers otherwise make an unrelated test fail during startup.
    "HCCL_HOST_SOCKET_PORT_RANGE": "60000-60050",
    "HCCL_NPU_SOCKET_PORT_RANGE": "61000-61050",
}


@dataclass
class Result:
    test_id: str
    test_type: str
    command: list[str]
    status: str
    return_code: int | None
    duration_seconds: float
    log_path: str
    round: int
    timed_out: bool = False
    idle_timed_out: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--test-root", action="append", default=[])
    parser.add_argument("--round", type=int, choices=range(1, MAX_ROUNDS + 1), required=True)
    parser.add_argument("--rerun-ng", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--idle-log-timeout", type=int, default=600)
    parser.add_argument("--ut-timeout", type=int, default=600)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--test-type", action="append", choices=TEST_TYPES, default=[])
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--state-dir", default=".npu-test-autofix")
    parser.add_argument('--env-file', default='.npu-test-autofix/test-assets.env')
    return parser.parse_args()


def discover(repo: Path, roots: list[str], includes: list[str], excludes: list[str]) -> list[Path]:
    tests: set[Path] = set()
    patterns = includes or ["test_*.py", "test_*.sh"]
    for root_name in roots:
        root = (repo / root_name).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Test root does not exist: {root}")
        if repo not in root.parents and root != repo:
            raise ValueError(f"Test root escapes repository: {root}")
        for path in root.rglob("test_*"):
            if not path.is_file() or path.name in SKIP_NAMES:
                continue
            relative = path.relative_to(repo).as_posix()
            lowered_parts = {part.lower() for part in path.parts}
            if "__pycache__" in lowered_parts or "logs" in lowered_parts:
                continue
            if not any(fnmatch.fnmatch(path.name, pattern) for pattern in patterns):
                continue
            if any(fnmatch.fnmatch(relative, pattern) for pattern in excludes):
                continue
            tests.add(path)
    return sorted(tests, key=lambda path: path.relative_to(repo).as_posix())


def test_id(repo: Path, path: Path) -> str:
    return path.relative_to(repo).as_posix()


def classify_test(repo: Path, path: Path) -> str:
    relative = path.relative_to(repo)
    directory_parts = {part.lower() for part in relative.parts[:-1]}
    if "ut" in directory_parts or "utils" in directory_parts or "utiles" in directory_parts:
        return "UT"
    if "st" in directory_parts or "nightly_ci" in directory_parts or any(
        part.startswith("test_npu_cann") for part in directory_parts
    ):
        return "ST"
    if path.suffix == ".sh":
        return "ST"

    source = path.read_text(encoding="utf-8", errors="replace")
    lowered = source.lower()
    st_markers = ("execute_train(", "ray job submit", "train.py", "torchrun")
    if any(marker in lowered for marker in st_markers):
        return "ST"
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return "UNCLASSIFIED"
    has_test_function = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(tree)
    )
    imports_test_framework = any(
        (
            isinstance(node, ast.Import)
            and any(alias.name.split(".", 1)[0] in {"pytest", "unittest"} for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".", 1)[0] in {"pytest", "unittest"}
        )
        for node in ast.walk(tree)
    )
    if has_test_function or imports_test_framework:
        return "UT"
    return "UNCLASSIFIED"


def command_for(path: Path, python: str, test_type: str) -> list[str]:
    if path.suffix == ".py":
        if test_type == "UT":
            return [python, "-m", "pytest", str(path)]
        return [python, str(path)]
    if path.suffix == ".sh":
        bash = shutil.which("bash")
        if not bash:
            raise RuntimeError("bash is required to execute shell tests")
        return [bash, str(path)]
    raise ValueError(f"Unsupported test type: {path}")


def write_inventory(
    state_dir: Path,
    repo: Path,
    tests: list[Path],
    classified: dict[str, str],
) -> None:
    counts = {
        test_type: sum(
            classified[test_id(repo, path)] == test_type for path in tests
        )
        for test_type in TEST_TYPES
    }
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total": len(tests),
        "counts_by_type": counts,
        "tests": [
            {
                "test_id": test_id(repo, path),
                "test_type": classified[test_id(repo, path)],
            }
            for path in tests
        ],
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "test-inventory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_latest(state_dir: Path) -> dict:
    latest = state_dir / "latest.json"
    if not latest.exists():
        raise FileNotFoundError("--rerun-ng requires an existing latest.json")
    return json.loads(latest.read_text(encoding="utf-8"))


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line_number, raw_line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        parts = shlex.split(line, comments=True)
        if len(parts) != 1 or '=' not in parts[0]:
            raise ValueError(f'Invalid env file line {line_number}: {raw_line!r}')
        key, value = parts[0].split('=', 1)
        if not key:
            raise ValueError(f'Empty env name on line {line_number}')
        os.environ[key] = value


def configure_default_npu_environment() -> None:
    for key, value in DEFAULT_NPU_ENV.items():
        os.environ.setdefault(key, value)


def select_tests(
    repo: Path, discovered: list[Path], state_dir: Path, rerun_ng: bool
) -> tuple[list[Path], list[str]]:
    discovered_ids = [test_id(repo, path) for path in discovered]
    if not rerun_ng:
        return discovered, discovered_ids
    latest = load_latest(state_dir)
    prior_results = {item["test_id"]: item for item in latest["latest_results"]}
    ng_ids = {case_id for case_id, item in prior_results.items() if item["status"] != "PASS"}
    by_id = {test_id(repo, path): path for path in discovered}
    missing = sorted(ng_ids - set(by_id))
    if missing:
        raise RuntimeError("Previously failing tests disappeared: " + ", ".join(missing))
    return [by_id[case_id] for case_id in sorted(ng_ids)], latest["baseline_tests"]


def terminate_process(process: subprocess.Popen) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    else:
        process.kill()


def stop_logged_ray_job(log_path: Path) -> str | None:
    """Stop the Ray submission recorded by a stalled test, if one exists."""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"raysubmit_[A-Za-z0-9]+", text)
    if not matches:
        return None
    submission_id = matches[-1]
    try:
        subprocess.run(
            ["ray", "job", "stop", submission_id, "--address=http://127.0.0.1:8265"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        # The test process is terminated by the caller regardless.  A stalled
        # Ray CLI must not prevent the case from being recorded as NG.
        pass
    return submission_id


def run_case(
    repo: Path,
    path: Path,
    test_type: str,
    python: str,
    timeout: int,
    idle_log_timeout: int,
    round_number: int,
    log_dir: Path,
) -> Result:
    case_id = test_id(repo, path)
    safe_name = case_id.replace("/", "__").replace("\\", "__")
    log_path = log_dir / f"{safe_name}.log"
    command = command_for(path, python, test_type)
    case_env = os.environ.copy()
    tests_root = str(repo / "tests")
    existing_pythonpath = case_env.get("PYTHONPATH")
    case_env["PYTHONPATH"] = (
        tests_root if not existing_pythonpath else os.pathsep.join((tests_root, existing_pythonpath))
    )
    started = time.monotonic()
    timed_out = False
    idle_timed_out = False
    return_code: int | None = None
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"CASE: {case_id}\nCOMMAND: {command!r}\nSTART: {datetime.now().isoformat()}\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=case_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= timeout:
                timed_out = True
                terminate_process(process)
                return_code = process.poll()
                log.write(f"\nTIMEOUT after {timeout} seconds\n")
                break
            if time.time() - log_path.stat().st_mtime >= idle_log_timeout:
                idle_timed_out = True
                submission_id = stop_logged_ray_job(log_path)
                terminate_process(process)
                return_code = process.poll()
                log.write(
                    f"\nIDLE_LOG_TIMEOUT after {idle_log_timeout} seconds"
                    f"; stopped Ray job {submission_id or 'not-found'}\n"
                )
                break
            time.sleep(5)
        if return_code is None:
            return_code = process.poll()
    duration = round(time.monotonic() - started, 3)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    failed_child_job = bool(FAILED_CHILD_JOB_RE.search(log_text))
    return Result(
        test_id=case_id,
        test_type=test_type,
        command=command,
        status=(
            "PASS"
            if return_code == 0
            and not timed_out
            and not idle_timed_out
            and not failed_child_job
            else "NG"
        ),
        return_code=return_code,
        duration_seconds=duration,
        log_path=log_path.relative_to(repo).as_posix(),
        round=round_number,
        timed_out=timed_out,
        idle_timed_out=idle_timed_out,
    )


def merge_latest(state_dir: Path, baseline_ids: list[str], current: list[Result]) -> list[dict]:
    latest_path = state_dir / "latest.json"
    previous: dict[str, dict] = {}
    if latest_path.exists():
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        previous = {item["test_id"]: item for item in payload["latest_results"]}
    for result in current:
        previous[result.test_id] = asdict(result)
    missing = set(baseline_ids) - set(previous)
    if missing:
        raise RuntimeError("No result for baseline tests: " + ", ".join(sorted(missing)))
    return [previous[case_id] for case_id in baseline_ids]


def write_reports(
    state_dir: Path,
    round_number: int,
    baseline_ids: list[str],
    current: list[Result],
    latest: list[dict],
) -> None:
    rounds_dir = state_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    passed = sum(item["status"] == "PASS" for item in latest)
    ng = len(latest) - passed
    timeouts = sum(bool(item.get("timed_out")) for item in latest)
    pass_rate = round(passed / len(baseline_ids) * 100, 2) if baseline_ids else 0.0
    summary_by_type = {}
    for test_type in TEST_TYPES:
        typed = [item for item in latest if item.get("test_type") == test_type]
        typed_pass = sum(item["status"] == "PASS" for item in typed)
        typed_total = len(typed)
        summary_by_type[test_type] = {
            "total": typed_total,
            "pass": typed_pass,
            "ng": typed_total - typed_pass,
            "timeouts": sum(bool(item.get("timed_out")) for item in typed),
            "pass_rate_percent": (
                round(typed_pass / typed_total * 100, 2) if typed_total else 0.0
            ),
        }
    payload = {
        "round": round_number,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "baseline_tests": baseline_ids,
        "executed_this_round": [asdict(item) for item in current],
        "latest_results": latest,
        "summary": {
            "total": len(baseline_ids),
            "pass": passed,
            "ng": ng,
            "timeouts": timeouts,
            "pass_rate_percent": pass_rate,
        },
        "summary_by_type": summary_by_type,
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    (rounds_dir / f"round-{round_number}.json").write_text(serialized, encoding="utf-8")
    (state_dir / "latest.json").write_text(serialized, encoding="utf-8")
    lines = [
        "# NPU Test Autofix Summary",
        "",
        f"- Latest round: {round_number}",
        f"- Total baseline tests: {len(baseline_ids)}",
        f"- PASS: {passed}",
        f"- NG: {ng}",
        f"- Timeouts: {timeouts}",
        f"- Pass rate: {pass_rate:.2f}%",
        "",
        "## Results By Type",
        "",
        "| Type | Total | PASS | NG | Timeouts | Pass rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for test_type in TEST_TYPES:
        item = summary_by_type[test_type]
        lines.append(
            f"| {test_type} | {item['total']} | {item['pass']} | {item['ng']} | "
            f"{item['timeouts']} | {item['pass_rate_percent']:.2f}% |"
        )
    lines.extend([
        "",
        "## Cases",
        "",
        "| Type | Status | Test | Round | Duration(s) | Log |",
        "|---|---|---|---:|---:|---|",
    ])
    for item in latest:
        lines.append(
            f"| {item.get('test_type', 'UNCLASSIFIED')} | {item['status']} | "
            f"`{item['test_id']}` | {item['round']} | "
            f"{item['duration_seconds']} | `{item['log_path']}` |"
        )
    (state_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    state_dir = (repo / args.state_dir).resolve()
    if repo not in state_dir.parents:
        print("error: state-dir must be inside the repository", file=sys.stderr)
        return 2
    if args.round == 1 and args.rerun_ng:
        print("error: round 1 must be a full baseline", file=sys.stderr)
        return 2
    if args.inventory_only and (args.round != 1 or args.rerun_ng):
        print("error: --inventory-only requires --round 1 without --rerun-ng", file=sys.stderr)
        return 2
    if args.round > 1 and not args.rerun_ng:
        print("error: rounds 2-5 must use --rerun-ng", file=sys.stderr)
        return 2
    roots = args.test_root or list(DEFAULT_ROOTS)
    try:
        configure_default_npu_environment()
        env_file = (repo / args.env_file).resolve()
        if repo not in env_file.parents:
            raise ValueError('env-file must be inside the repository')
        load_env_file(env_file)
        discovered = discover(repo, roots, args.include, args.exclude)
        classified = {test_id(repo, path): classify_test(repo, path) for path in discovered}
        if args.test_type:
            allowed_types = set(args.test_type)
            discovered = [
                path
                for path in discovered
                if classified[test_id(repo, path)] in allowed_types
            ]
        if not discovered:
            raise RuntimeError("No tests discovered")
        type_counts = {
            test_type: sum(
                classified[test_id(repo, path)] == test_type for path in discovered
            )
            for test_type in TEST_TYPES
        }
        print(
            "Discovered "
            + ", ".join(f"{name}={count}" for name, count in type_counts.items()),
            flush=True,
        )
        write_inventory(state_dir, repo, discovered, classified)
        if args.inventory_only:
            print(
                f"Inventory: {(state_dir / 'test-inventory.json').relative_to(repo)}",
                flush=True,
            )
            return 0
        selected, baseline_ids = select_tests(repo, discovered, state_dir, args.rerun_ng)
        if not selected:
            print("All baseline tests already PASS; nothing to rerun.")
            return 0
        log_dir = state_dir / "logs" / f"round-{args.round}"
        log_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for index, path in enumerate(selected, 1):
            case_id = test_id(repo, path)
            test_type = classified[case_id]
            timeout = args.ut_timeout if test_type == "UT" else args.timeout
            print(f"[{index}/{len(selected)}] RUN {test_type} {case_id}", flush=True)
            result = run_case(
                repo,
                path,
                test_type,
                args.python,
                timeout,
                args.idle_log_timeout,
                args.round,
                log_dir,
            )
            print(f"[{index}/{len(selected)}] {result.status} {case_id} ({result.duration_seconds:.1f}s)", flush=True)
            results.append(result)
        latest = merge_latest(state_dir, baseline_ids, results)
        write_reports(state_dir, args.round, baseline_ids, results, latest)
    except (FileNotFoundError, RuntimeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 1 if any(result.status == "NG" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
