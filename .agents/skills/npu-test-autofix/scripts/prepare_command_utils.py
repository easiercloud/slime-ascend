#!/usr/bin/env python3
"""Prepare command_utils.py with required Slime Ascend paths and env vars."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_TARGET = "slime/utils/external_utils/command_utils.py"
ENV_TEMPLATE = """\
"PYTHONPATH": f"__RUNTIME_PYTHONPATH__:{os.environ.get('PYTHONPATH', '')}",
"SLIME_ENABLE_PROFILING": "False",
"RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
"RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
"CUDA_DEVICE_MAX_CONNECTIONS": "1",
"NCCL_NVLS_ENABLE": str(int(check_has_nvlink())),
"no_proxy": f"127.0.0.1,{master_addr}",
# This is needed by megatron / torch distributed in multi-node setup
"MASTER_ADDR": master_addr,
"HCCL_DETERMINISTIC": "True",
"LCCL_DETERMINISTIC": "1",
"CLOSE_MATMUL_K_SHIFT": "1",
"ATB_LLM_LCOC_ENABLE": "0",
"ASCEND_COREDUMP_SIGNAL": os.environ.get("ASCEND_COREDUMP_SIGNAL", "none"),
"""


@dataclass(frozen=True)
class DependencyResult:
    name: str
    status: str
    path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--workspace-root", default="/workspace")
    parser.add_argument("--megatron-root")
    parser.add_argument("--bridge-root")
    parser.add_argument("--megatron-adaptor-root")
    parser.add_argument("--transformer-engine-root")
    parser.add_argument("--mindspeed-root")
    parser.add_argument("--extra-pythonpath", action="append", default=[])
    parser.add_argument("--report", default=".npu-test-autofix/preparation.md")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-paths", action="store_true")
    return parser.parse_args()


def imported_top_level_modules(repo: Path) -> set[str]:
    modules: set[str] = set()
    for base in (repo / "slime", repo / "tests"):
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts or "logs" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.add(node.module.split(".", 1)[0])
    return modules


def candidate_python_roots(workspace: Path) -> list[Path]:
    candidates = [workspace]
    if workspace.is_dir():
        for child in sorted(workspace.iterdir()):
            if not child.is_dir():
                continue
            candidates.extend((child, child / "src", child / "python"))
    return candidates


def provides_module(root: Path, module: str) -> bool:
    relative = Path(*module.split("."))
    return (root / relative).is_dir() or (root / relative).with_suffix(".py").is_file()


def find_module_root(workspace: Path, module: str, preferred: list[Path]) -> Path | None:
    seen: set[Path] = set()
    for candidate in [*preferred, *candidate_python_roots(workspace)]:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_dir() and provides_module(candidate, module):
            return candidate
    return None


def add_unique(paths: list[str], value: Path | str) -> None:
    normalized = str(Path(value))
    if normalized not in paths:
        paths.append(normalized)


def discover_python_paths(
    repo: Path,
    workspace: Path,
    args: argparse.Namespace,
) -> tuple[list[str], list[DependencyResult], list[str]]:
    paths: list[str] = []
    results: list[DependencyResult] = []
    missing: list[str] = []
    imports = imported_top_level_modules(repo)
    # The delivery tree intentionally keeps the Slime submodule at pristine
    # upstream v0.3.1.  MegatronAdaptor imports are introduced by the
    # manifest-managed NPU patch at build time, so source scanning alone must
    # not classify that pinned runtime dependency as "not referenced".
    manifest_required_modules = {"megatron_adaptor", "transformer_engine"}

    core_specs = (
        (
            "Megatron-LM",
            "megatron",
            args.megatron_root,
            [workspace / "Megatron-LM"],
        ),
        (
            "Megatron-Bridge",
            None,
            args.bridge_root,
            [workspace / "Megatron-Bridge" / "src"],
        ),
    )
    for name, module, override, preferred in core_specs:
        if override:
            root = Path(override).resolve()
        elif module:
            root = find_module_root(workspace, module, preferred)
        else:
            root = next((path.resolve() for path in preferred if path.is_dir()), None)
        if root and root.is_dir():
            add_unique(paths, root)
            results.append(DependencyResult(name, "workspace", str(root)))
        else:
            results.append(DependencyResult(name, "missing", None))
            missing.append(name)

    optional_specs = (
        (
            "MegatronAdaptor",
            "megatron_adaptor",
            args.megatron_adaptor_root,
            [workspace / "MegatronAdaptor", workspace / "megatron_adaptor"],
        ),
        (
            "TransformerEngine",
            "transformer_engine",
            args.transformer_engine_root,
            [workspace / "TransformerEngine", workspace / "transformer_engine"],
        ),
        (
            "mbridge",
            "mbridge",
            None,
            [workspace / "mbridge"],
        ),
        (
            "MindSpeed",
            "mindspeed",
            args.mindspeed_root,
            [workspace / "MindSpeed"],
        ),
    )
    for name, module, override, preferred in optional_specs:
        if module not in imports and module not in manifest_required_modules and not override:
            results.append(DependencyResult(name, "not referenced", None))
            continue
        root = (
            Path(override).resolve()
            if override
            else find_module_root(workspace, module, preferred)
        )
        if root and root.is_dir() and provides_module(root, module):
            add_unique(paths, root)
            results.append(DependencyResult(name, "workspace", str(root)))
        elif importlib.util.find_spec(module) is not None:
            results.append(DependencyResult(name, "installed package", None))
        else:
            results.append(DependencyResult(name, "missing", None))
            missing.append(name)

    for extra in args.extra_pythonpath:
        root = Path(extra).resolve()
        if not root.is_dir():
            missing.append(f"extra PYTHONPATH {root}")
            results.append(DependencyResult("extra", "missing", str(root)))
            continue
        add_unique(paths, root)
        results.append(DependencyResult("extra", "workspace", str(root)))
    return paths, results, missing


def find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"Function not found: {name}")


def prepare_source(
    source: str,
    python_paths: list[str],
) -> tuple[str, list[str]]:
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    newline = "\r\n" if "\r\n" in source else "\n"
    changes: list[str] = []

    convert_function = find_function(tree, "convert_checkpoint")
    indexes = [
        index
        for index in range(convert_function.lineno - 1, convert_function.end_lineno)
        if "PYTHONPATH=" in lines[index]
    ]
    if len(indexes) != 1:
        raise RuntimeError(
            "Expected one conversion PYTHONPATH in convert_checkpoint, "
            f"found {len(indexes)}"
        )
    index = indexes[0]
    indent = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
    discovered_pythonpath = ":".join(python_paths)
    desired = (
        f'{indent}f"PYTHONPATH={discovered_pythonpath}:$PYTHONPATH "{newline}'
    )
    if lines[index] != desired:
        lines[index] = desired
        changes.append("convert_checkpoint.PYTHONPATH")

    intermediate = "".join(lines)
    tree = ast.parse(intermediate)
    execute_function = find_function(tree, "execute_train")
    function_lines = range(execute_function.lineno - 1, execute_function.end_lineno)
    env_indexes = [
        index for index in function_lines if lines[index].strip() == '"env_vars": {'
    ]
    if len(env_indexes) != 1:
        raise RuntimeError(f"Expected one runtime env_vars block, found {len(env_indexes)}")
    env_index = env_indexes[0]
    static_start = next(
        (
            index
            for index in range(env_index + 1, execute_function.end_lineno)
            if lines[index].lstrip().startswith('"PYTHONPATH":')
        ),
        None,
    )
    static_end = next(
        (
            index
            for index in range((static_start or env_index) + 1, execute_function.end_lineno)
            if lines[index].strip() == "**("
        ),
        None,
    )
    if static_start is None or static_end is None:
        raise RuntimeError("Could not locate static runtime env_vars region")

    static_indent = lines[static_start][
        : len(lines[static_start]) - len(lines[static_start].lstrip())
    ]
    desired_block = [
        f"{static_indent}{line}{newline}"
        for line in ENV_TEMPLATE.replace(
            "__RUNTIME_PYTHONPATH__", discovered_pythonpath
        ).splitlines()
    ]
    if lines[static_start:static_end] != desired_block:
        lines[static_start:static_end] = desired_block
        changes.append("execute_train.runtime_env_json.env_vars")

    prepared = "".join(lines)
    ast.parse(prepared)
    return prepared, changes


def verify_paths(missing: list[str]) -> None:
    if missing:
        raise FileNotFoundError(
            "Required NPU dependencies are unavailable: " + ", ".join(missing)
        )


def write_report(
    report: Path,
    target: Path,
    changes: list[str],
    python_paths: list[str],
    dependencies: list[DependencyResult],
    check_only: bool,
) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# NPU Test Preparation",
        "",
        f"- Time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Target: `{target.as_posix()}`",
        f"- Mode: {'check' if check_only else 'apply'}",
        f"- Result: {'changes required' if check_only and changes else 'prepared'}",
        f"- PYTHONPATH prefix: `{':'.join(python_paths)}`",
        "",
        "## Dependency Discovery",
        "",
    ]
    lines.extend(
        f"- {item.name}: {item.status}"
        + (f" (`{item.path}`)" if item.path else "")
        for item in dependencies
    )
    lines.extend([
        "",
        "## Modified Regions",
        "",
    ])
    lines.extend(f"- `{change}`" for change in changes)
    if not changes:
        lines.append("- No changes; file was already prepared.")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    target = (repo / args.target).resolve()
    report = (repo / args.report).resolve()
    if repo not in target.parents or repo not in report.parents:
        print("error: target and report must be inside repo-root", file=sys.stderr)
        return 2
    try:
        python_paths, dependencies, missing = discover_python_paths(
            repo, Path(args.workspace_root).resolve(), args
        )
        if args.verify_paths:
            verify_paths(missing)
        if not python_paths:
            raise RuntimeError("No workspace Python paths were discovered")
        source = target.read_text(encoding="utf-8")
        prepared, changes = prepare_source(source, python_paths)
        if changes and not args.check:
            temporary = target.with_suffix(target.suffix + ".npu-prepare.tmp")
            temporary.write_text(prepared, encoding="utf-8", newline="")
            os.replace(temporary, target)
        if not args.check:
            write_report(
                report,
                target.relative_to(repo),
                changes,
                python_paths,
                dependencies,
                args.check,
            )
    except (FileNotFoundError, RuntimeError, SyntaxError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.check and changes:
        print("NPU preparation required: " + ", ".join(changes))
        return 1
    print(
        "NPU preparation applied: " + ", ".join(changes)
        if changes
        else "NPU preparation already satisfied"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
