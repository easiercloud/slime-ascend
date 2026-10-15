#!/usr/bin/env python3
"""Convert Slime GPU tests while preserving the upstream source layout.

This intentionally does not emit ``ast.unparse`` output.  The AST is used only
to locate statements safely; the resulting NPU file remains a lightly edited
copy of its upstream counterpart so that review diffs retain comments, blank
lines, parenthesised argument groups, and line wrapping.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


ROOT_LINES = (
    'HF_WEIGHTS_ROOT = os.environ["SLIME_TEST_HF_WEIGHTS_ROOT"]\n'
    'DATASETS_ROOT = os.environ["SLIME_TEST_DATASETS_ROOT"]\n'
    'MEGATRON_WEIGHTS_ROOT = os.environ["SLIME_TEST_MEGATRON_WEIGHTS_ROOT"]\n'
)


def _call_name(node: ast.Call) -> str:
    return ast.unparse(node.func)


def _static_num_gpus(tree: ast.Module) -> int | None:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "NUM_GPUS" for target in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, int) else None
    return None


def _is_download_or_unsafe_prepare(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    name = _call_name(call)
    rendered = ast.unparse(call)
    if name in {"U.hf_download_dataset", "U.convert_checkpoint"}:
        return True
    return name == "U.exec_command" and any(
        marker in rendered for marker in ("hf download", "mkdir -p /root", "pkill", "ray stop")
    )


def _replace_line_range(lines: list[str], start: int, end: int, replacement: list[str]) -> None:
    """Replace one-based inclusive source lines, processing ranges bottom-up."""
    lines[start - 1 : end] = replacement


def _strip_prepare_calls(text: str) -> str:
    """Remove downloads/conversion calls but retain valid, readable functions."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    changes: list[tuple[int, int, list[str]]] = []

    for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        removable = [statement for statement in function.body if _is_download_or_unsafe_prepare(statement)]
        if not removable:
            continue
        if len(removable) == len(function.body):
            indent = " " * (function.body[0].col_offset if function.body else function.col_offset + 4)
            changes.append((function.body[0].lineno, function.body[-1].end_lineno, [f"{indent}pass\n"]))
        else:
            changes.extend((statement.lineno, statement.end_lineno, []) for statement in removable)

    for start, end, replacement in sorted(changes, reverse=True):
        _replace_line_range(lines, start, end, replacement)
    text = "".join(lines)
    text = re.sub(r"\ndef prepare\(\):\n    pass\n", "\n", text)
    text = text.replace("    prepare()\n", "")
    return text


def _inject_roots(text: str) -> str:
    tree = ast.parse(text)
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    lines = text.splitlines(keepends=True)
    if not any(isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names) for node in imports):
        lines.insert(0, "import os\n")
        text = "".join(lines)
        tree = ast.parse(text)
        imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        lines = text.splitlines(keepends=True)
    if "SLIME_TEST_HF_WEIGHTS_ROOT" not in text:
        insert_at = max((node.end_lineno for node in imports), default=0)
        lines[insert_at:insert_at] = [ROOT_LINES]
    return "".join(lines)


def _model_suffix_to_megatron(suffix: str) -> str:
    """Map a model-relative HF suffix to its canonical torch-dist checkpoint."""
    suffix = suffix.rstrip("/")
    if suffix.endswith("_torch_dist"):
        return suffix
    if suffix.endswith("{MODEL_NAME}"):
        return suffix + "_torch_dist"
    return suffix


def _replace_assets(text: str) -> str:
    text = text.replace("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")
    text = text.replace("cuda:", "npu:")
    text = text.replace("/root/datasets", "{DATASETS_ROOT}")
    text = text.replace(
        'f"/root/models/{MODEL_NAME}_torch_dist"',
        'f"{MEGATRON_WEIGHTS_ROOT}/{MODEL_NAME}_torch_dist"',
    )
    text = text.replace(
        'f"/dev/shm/{MODEL_NAME}_torch_dist"',
        'f"{MEGATRON_WEIGHTS_ROOT}/{MODEL_NAME}_torch_dist"',
    )
    text = text.replace('f"/root/models/{MODEL_NAME}"', 'f"{HF_WEIGHTS_ROOT}/{MODEL_NAME}"')
    text = re.sub(
        r'return f"/root/models/(?P<name>\{MODEL_NAME\}_slime_[^"]+)"',
        r'return os.path.join(os.environ.get("SLIME_TEST_WORK_ROOT", "/tmp"), f"\g<name>")',
        text,
    )

    def hf(match: re.Match[str]) -> str:
        return match.group("prefix") + "{HF_WEIGHTS_ROOT}" + match.group("suffix")

    def ref(match: re.Match[str]) -> str:
        suffix = _model_suffix_to_megatron(match.group("suffix"))
        return match.group("prefix") + "{MEGATRON_WEIGHTS_ROOT}" + suffix

    # The source uses f-strings for all model paths.  Map each argument
    # separately: --hf-checkpoint must never be redirected to weights-MA.
    text = re.sub(r"(?P<prefix>--hf-checkpoint\s+)/root/models(?P<suffix>[^\s'\"]*)", hf, text)
    text = re.sub(r"(?P<prefix>--ref-load\s+)/root/models(?P<suffix>[^\s'\"]*)", ref, text)
    text = re.sub(r"(?P<prefix>--ref-load\s+)/root(?P<suffix>[^\s'\"]*)", ref, text)

    # Dataset paths often live in ordinary concatenated string literals rather
    # than f-strings.  Make only those literals f-strings, preserving their
    # original grouping and line wrapping.
    lines = []
    for line in text.splitlines(keepends=True):
        if "{DATASETS_ROOT}" in line:
            match = re.search(r"(?<![A-Za-z0-9_])(['\"])", line)
            if match and (match.start() == 0 or line[match.start() - 1] != "f"):
                line = line[: match.start()] + "f" + line[match.start() :]
        lines.append(line)
    text = "".join(lines)
    return text


def _inject_npu_runtime(text: str) -> str:
    if "--sglang-device" not in text:
        tree = ast.parse(text)
        lines = text.splitlines(keepends=True)
        inserts: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == "sglang_args" for target in node.targets):
                indent = " " * node.col_offset
                inserts.append((node.end_lineno, f'{indent}sglang_args += "--sglang-device npu "\n'))
        for line_number, line in sorted(inserts, reverse=True):
            lines.insert(line_number, line)
        text = "".join(lines)

    if "torch.cuda" in text:
        text = text.replace("torch.cuda", "torch.npu")
        if "import torch_npu" not in text:
            tree = ast.parse(text)
            imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
            lines = text.splitlines(keepends=True)
            insert_at = max((node.end_lineno for node in imports), default=0)
            lines.insert(insert_at, "import torch_npu\n")
            text = "".join(lines)
    return text


def _format_generated_blocks(text: str) -> str:
    """Apply the small, stable layout conventions used by maintained NPU tests."""
    text = re.sub(
        r"(?m)^(?P<indent>[ \t]*)ckpt_args = (?P<hf>f?['\"][^\n]*--hf-checkpoint[^\n]*['\"]) (?P<ref>f?['\"][^\n]*--ref-load[^\n]*['\"])$",
        lambda match: (
            f"{match.group('indent')}ckpt_args = (\n"
            f"{match.group('indent')}    {match.group('hf')}\n"
            f"{match.group('indent')}    {match.group('ref')}\n"
            f"{match.group('indent')})"
        ),
        text,
    )
    return re.sub(r"\n{3,}", "\n\n", text)


def convert(source: Path, target: Path) -> dict[str, object]:
    original = source.read_text(encoding="utf-8")
    num_gpus = _static_num_gpus(ast.parse(original))
    if not num_gpus:
        return {"source": str(source), "target": None, "status": "REUSE_UNCHANGED", "num_devices": 0}

    rendered = _strip_prepare_calls(original)
    rendered = _inject_roots(rendered)
    rendered = _replace_assets(rendered)
    rendered = _inject_npu_runtime(rendered)
    rendered = _format_generated_blocks(rendered)
    ast.parse(rendered)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(rendered, encoding="utf-8")
    return {"source": str(source), "target": str(target), "status": "STATIC_READY", "num_devices": num_gpus}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    tests_root = args.repo_root.resolve() / "tests"
    output_root = args.output_root.resolve()
    existing_map = {
        "test_qwen2.5_0.5B_async_short.py": "test_qwen2.5_0.5B_gsm8k_async_short_npu.py",
        "test_qwen2.5_0.5B_debug_rollout_then_train.py": "test_qwen2.5_0.5B_debug_rollout_then_train_npu.py",
        "test_qwen2.5_0.5B_opd_sglang.py": "test_qwen2.5_0.5B_opd_sglang_npu.py",
        "test_qwen3_0.6B_parallel_check.py": "test_qwen3_0.6B_parallel_check_npu.py",
        "test_qwen3_4B_ckpt.py": "test_qwen3_4B_ckpt_npu.py",
    }
    nightly = {"test_qwen3_0.6B_parallel_check.py", "test_qwen3_4B_ckpt.py"}
    rows = []
    for source in sorted(tests_root.rglob("test_*.py")):
        relative = source.relative_to(tests_root)
        if relative.parts and relative.parts[0] == "tests_npu":
            continue
        num_gpus = _static_num_gpus(ast.parse(source.read_text(encoding="utf-8")))
        if not num_gpus:
            rows.append({"source": str(relative), "target": str(relative), "status": "REUSE_UNCHANGED", "num_devices": 0})
            continue
        target = output_root / ("nightly_CI" if source.name in nightly else "st") / existing_map.get(source.name, source.stem + "_npu.py")
        row = convert(source, target) if args.overwrite or not target.exists() else {"status": "EXISTING_NPU", "num_devices": num_gpus}
        row.update(source=str(relative), target=str(target.relative_to(output_root.parent.parent)))
        rows.append(row)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"schema_version": 1, "source_count": len(rows), "cases": rows}, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
