#!/usr/bin/env python3
"""Create and compare deterministic content manifests for replay verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath


DEFAULT_EXCLUDES = (
    ".git",
    ".upstream-sync",
    ".npu-test-autofix*",
    ".pytest_cache",
    "**/__pycache__",
    "**/*.pyc",
)


def excluded(relative: str, patterns: tuple[str, ...]) -> bool:
    path = PurePosixPath(relative)
    return any(path.match(pattern) or relative == pattern for pattern in patterns)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_manifest(
    root: Path,
    extra_excludes: list[str] | None = None,
    *,
    use_default_excludes: bool = True,
) -> dict:
    root = root.resolve()
    patterns = (DEFAULT_EXCLUDES if use_default_excludes else ()) + tuple(
        extra_excludes or []
    )
    files: dict[str, dict[str, str | int]] = {}
    for directory, dir_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        kept_dirs = []
        for name in sorted(dir_names):
            path = current / name
            relative = path.relative_to(root).as_posix()
            if excluded(relative, patterns):
                continue
            if path.is_symlink():
                info = path.lstat()
                files[relative] = {
                    "type": "symlink",
                    "target": os.readlink(path),
                    "mode": stat.S_IMODE(info.st_mode),
                }
            else:
                kept_dirs.append(name)
        dir_names[:] = kept_dirs
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(root).as_posix()
            if excluded(relative, patterns):
                continue
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if path.is_symlink():
                files[relative] = {
                    "type": "symlink",
                    "target": os.readlink(path),
                    "mode": mode,
                }
            elif path.is_file():
                files[relative] = {
                    "type": "file",
                    "sha256": hash_file(path),
                    "mode": mode,
                }
    serialized = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "root": str(root),
        "excludes": list(patterns),
        "file_count": len(files),
        "tree_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "files": files,
    }


def compare(left: dict, right: dict) -> list[str]:
    left_files = left.get("files", {})
    right_files = right.get("files", {})
    differences = []
    for path in sorted(set(left_files) | set(right_files)):
        if path not in left_files:
            differences.append(f"ONLY_REPLAY\t{path}")
        elif path not in right_files:
            differences.append(f"ONLY_VALIDATED\t{path}")
        elif left_files[path] != right_files[path]:
            differences.append(f"DIFFERENT\t{path}")
    return differences


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--exclude", action="append", default=[])
    create.add_argument("--no-default-excludes", action="store_true")
    check = subparsers.add_parser("compare")
    check.add_argument("--validated", type=Path, required=True)
    check.add_argument("--replay", type=Path, required=True)
    check.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "create":
            payload = create_manifest(
                args.root,
                args.exclude,
                use_default_excludes=not args.no_default_excludes,
            )
            write_json(args.output, payload)
            print(payload["tree_sha256"])
            return 0
        left = json.loads(args.validated.read_text(encoding="utf-8"))
        right = json.loads(args.replay.read_text(encoding="utf-8"))
        differences = compare(left, right)
        result = {
            "schema_version": 1,
            "replay_verified": not differences,
            "validated_tree_sha256": left.get("tree_sha256"),
            "replay_tree_sha256": right.get("tree_sha256"),
            "differences": differences,
        }
        if args.output:
            write_json(args.output, result)
        if differences:
            print("\n".join(differences), file=sys.stderr)
            return 1
        print("replay verified")
        return 0
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
