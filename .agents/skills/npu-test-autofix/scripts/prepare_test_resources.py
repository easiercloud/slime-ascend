#!/usr/bin/env python3
"""Validate local test assets and convert each HF model to Megatron once."""

from __future__ import annotations

import argparse
import ast
import contextlib
import json
import os
import re
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_TEST_ROOT = "tests"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: str
    num_gpus: int
    sources: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--test-root", action="append", default=[])
    parser.add_argument(
        "--hf-weights-root",
        default=os.environ.get("SLIME_TEST_HF_WEIGHTS_ROOT"),
    )
    parser.add_argument(
        "--datasets-root",
        default=os.environ.get("SLIME_TEST_DATASETS_ROOT"),
    )
    parser.add_argument(
        "--converted-root",
        default=os.environ.get("SLIME_TEST_MEGATRON_WEIGHTS_ROOT"),
    )
    parser.add_argument("--env-file", default=".npu-test-autofix/test-assets.env")
    parser.add_argument("--manifest", default=".npu-test-autofix/test-assets.json")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument(
        '--requirements',
        default='.npu-test-autofix/asset-requirements.json',
    )
    parser.add_argument('--inventory', action='store_true')
    return parser.parse_args()


def literal_constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return values


def is_st_test(path: Path) -> bool:
    directory_parts = {part.lower() for part in path.parts[:-1]}
    if 'ut' in directory_parts or 'utils' in directory_parts or 'utiles' in directory_parts:
        return False
    if 'st' in directory_parts or any(
        part.startswith('test_npu_cann') for part in directory_parts
    ):
        return True
    if path.suffix == '.sh':
        return True
    source = path.read_text(encoding='utf-8', errors='replace').lower()
    return any(
        marker in source
        for marker in ('execute_train(', 'ray job submit', 'train.py', 'torchrun')
    )


def discover_models(repo: Path, roots: list[Path]) -> tuple[list[ModelSpec], list[str]]:
    grouped: dict[tuple[str, str], dict[str, object]] = {}
    skipped: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("test_*.py")):
            if "__pycache__" in path.parts or "logs" in path.parts:
                continue
            if not is_st_test(path):
                continue
            constants = literal_constants(path)
            name = constants.get("MODEL_NAME")
            model_type = constants.get("MODEL_TYPE")
            num_gpus = constants.get("NUM_GPUS")
            relative = path.relative_to(repo).as_posix()
            if not isinstance(name, str) or not isinstance(model_type, str):
                skipped.append(relative)
                continue
            key = (name, model_type)
            item = grouped.setdefault(
                key,
                {"num_gpus": 1, "sources": []},
            )
            if isinstance(num_gpus, int) and num_gpus > 0:
                item["num_gpus"] = max(int(item["num_gpus"]), num_gpus)
            item["sources"].append(relative)

    by_name: dict[str, str] = {}
    specs: list[ModelSpec] = []
    for (name, model_type), item in sorted(grouped.items()):
        previous_type = by_name.setdefault(name, model_type)
        if previous_type != model_type:
            raise RuntimeError(
                f"Model {name!r} has conflicting types: {previous_type!r}, {model_type!r}"
            )
        specs.append(
            ModelSpec(
                name=name,
                model_type=model_type,
                num_gpus=int(item["num_gpus"]),
                sources=tuple(item["sources"]),
            )
        )
    return specs, skipped


def discover_datasets(repo: Path, roots: list[Path]) -> list[dict]:
    grouped: dict[str, set[str]] = {}
    for root in roots:
        for path in sorted(root.rglob('test_*.py')):
            if '__pycache__' in path.parts or 'logs' in path.parts:
                continue
            if not is_st_test(path):
                continue
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            relative = path.relative_to(repo).as_posix()
            source = path.read_text(encoding='utf-8')
            for logical_path in re.findall(
                r'(?:DATASETS_ROOT|\{DATASETS_ROOT\}|/root/datasets)/([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)',
                source,
            ):
                grouped.setdefault(logical_path, set()).add(relative)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                words = node.value.split()
                for index, word in enumerate(words):
                    asset_path = None
                    if word == '--prompt-data' and index + 1 < len(words):
                        asset_path = words[index + 1]
                    elif word == '--eval-prompt-data' and index + 2 < len(words):
                        asset_path = words[index + 2]
                    if not asset_path:
                        continue
                    normalized = asset_path.replace('\\', '/')
                    if '/datasets/' in normalized:
                        logical_path = normalized.split('/datasets/', 1)[1]
                    else:
                        logical_path = normalized.lstrip('/')
                    grouped.setdefault(logical_path, set()).add(relative)
    return [
        {'logical_path': path, 'sources': sorted(sources), 'local_path': None}
        for path, sources in sorted(grouped.items())
    ]


def write_requirements(
    path: Path,
    roots: list[Path],
    specs: list[ModelSpec],
    datasets: list[dict],
    skipped: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                'generated_at': datetime.now().astimezone().isoformat(timespec='seconds'),
                'test_roots': [str(root) for root in roots],
                'models': [
                    {
                        **asdict(spec),
                        'expected_hf_relative_path': spec.name,
                        'expected_megatron_relative_path': f'{spec.name}_torch_dist',
                        'local_hf_path': None,
                    }
                    for spec in specs
                ],
                'datasets': datasets,
                'tests_without_static_model_metadata': skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )


def resolve_inside(repo: Path, value: str, label: str) -> Path:
    path = (repo / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def acquire_lock(converted_root: Path, model_name: str) -> tuple[int, Path]:
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    lock = converted_root / f".{safe_name}.conversion.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"Conversion lock exists for {model_name}: {lock}. "
            "Check whether another conversion is running."
        ) from error
    os.write(descriptor, f"pid={os.getpid()}\n".encode())
    return descriptor, lock


def validate_converted_checkpoint(path: Path) -> None:
    """Reject empty/partial cache directories before a test reaches Ray."""
    tracker = path / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        raise RuntimeError(
            f"Converted checkpoint is incomplete: {path} lacks {tracker.name}. "
            "Remove or repair the partial directory, then rerun conversion."
        )
    marker = tracker.read_text(encoding="utf-8").strip()
    if marker != "release" and not marker.isdigit():
        raise RuntimeError(
            f"Converted checkpoint has invalid tracker {tracker}: {marker!r}."
        )


def convert_models(
    repo: Path,
    specs: list[ModelSpec],
    hf_root: Path,
    converted_root: Path,
) -> list[dict]:
    sys.path.insert(0, str(repo))
    import slime.utils.external_utils.command_utils as command_utils

    results = []
    for spec in specs:
        hf_path = hf_root / spec.name
        output = converted_root / f"{spec.name}_torch_dist"
        if output.exists():
            validate_converted_checkpoint(output)
            print(f"REUSE {spec.name}: {output}")
            results.append({"model": spec.name, "status": "reused", "output": str(output)})
            continue
        if not hf_path.is_dir():
            raise FileNotFoundError(
                f"Local HF weight is missing for {spec.name}: {hf_path}"
            )
        descriptor, lock = acquire_lock(converted_root, spec.name)
        try:
            print(f"CONVERT {spec.name}: {hf_path} -> {output}")
            # command_utils invokes repository-relative model profiles and
            # tools/convert_hf_to_torch_dist.py.  Keep callers free to launch
            # this helper from an outer sync workspace by anchoring conversion
            # to the selected framework repository.
            with contextlib.chdir(repo):
                command_utils.convert_checkpoint(
                    model_name=spec.name,
                    megatron_model_type=spec.model_type,
                    num_gpus_per_node=spec.num_gpus,
                    dir_dst=str(converted_root),
                    hf_checkpoint=str(hf_path),
                )
            if not output.exists():
                raise RuntimeError(
                    f"Conversion returned without creating expected output: {output}"
                )
            results.append(
                {"model": spec.name, "status": "converted", "output": str(output)}
            )
        finally:
            os.close(descriptor)
            lock.unlink(missing_ok=True)
    return results


def write_env_file(
    path: Path,
    hf_root: Path,
    datasets_root: Path,
    converted_root: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "SLIME_TEST_HF_WEIGHTS_ROOT": str(hf_root),
        "SLIME_TEST_DATASETS_ROOT": str(datasets_root),
        "SLIME_TEST_MEGATRON_WEIGHTS_ROOT": str(converted_root),
    }
    path.write_text(
        "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    try:
        roots = [
            resolve_inside(repo, value, "test root")
            for value in (args.test_root or [DEFAULT_TEST_ROOT])
        ]
        specs, skipped = discover_models(repo, roots)
        datasets = discover_datasets(repo, roots)
        if not specs:
            raise RuntimeError("No tests with MODEL_NAME and MODEL_TYPE were discovered")

        requirements = (repo / args.requirements).resolve()
        if requirements != repo and repo not in requirements.parents:
            raise ValueError("requirements must be inside repo-root")
        write_requirements(requirements, roots, specs, datasets, skipped)
        if args.inventory:
            print(f"Required models ({len(specs)}):")
            for spec in specs:
                print(f"  - {spec.name} [{spec.model_type}]")
            print(f"Required datasets ({len(datasets)}):")
            for dataset in datasets:
                print(f"  - {dataset['logical_path']}")
            print(f"Requirements: {requirements}")
            return 0

        if not args.hf_weights_root or not args.datasets_root:
            raise ValueError(
                "set SLIME_TEST_HF_WEIGHTS_ROOT and SLIME_TEST_DATASETS_ROOT "
                "or pass --hf-weights-root and --datasets-root"
            )
        hf_root = resolve_inside(repo, args.hf_weights_root, "HF weights root")
        datasets_root = resolve_inside(repo, args.datasets_root, "datasets root")
        if args.converted_root:
            converted_root = Path(args.converted_root).resolve()
        elif roots[0] == repo / 'tests':
            converted_root = roots[0] / 'weights-MA'
        else:
            converted_root = roots[0].parent / 'weights-MA'
        converted_root.mkdir(parents=True, exist_ok=True)
        env_file = (repo / args.env_file).resolve()
        manifest = (repo / args.manifest).resolve()
        if repo not in env_file.parents or repo not in manifest.parents:
            raise ValueError("env-file and manifest must be inside repo-root")
        write_env_file(env_file, hf_root, datasets_root, converted_root)
        conversion_results = (
            [] if args.scan_only else convert_models(repo, specs, hf_root, converted_root)
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "test_roots": [str(root) for root in roots],
                    "hf_weights_root": str(hf_root),
                    "datasets_root": str(datasets_root),
                    "converted_root": str(converted_root),
                    "requirements": str(requirements),
                    "models": [asdict(spec) for spec in specs],
                    "datasets": datasets,
                    "skipped_tests_without_static_model_metadata": skipped,
                    "conversions": conversion_results,
                    "scan_only": args.scan_only,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, RuntimeError, ValueError, OSError, SyntaxError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Prepared {len(specs)} unique model(s)")
    print(f"Converted cache: {converted_root}")
    print(f"Environment file: {env_file}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
