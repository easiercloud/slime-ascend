#!/usr/bin/env python3
"""Validate and apply the authoritative slime-ascend patch manifest."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


CATEGORY_ORDER = ("npu-compat", "upstream-fixes", "features", "third_party")
VALID_TARGETS = {
    "slime",
    "sglang",
    "megatron",
    "megatron-bridge",
    "mbridge",
    "sgl-kernel-npu",
    "transformer-engine-npu",
}


class ManifestError(RuntimeError):
    pass


@dataclass(frozen=True)
class Patch:
    path: Path
    category: str
    identifier: str
    target: str
    expected_sha256: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(manifest: Path, patches_root: Path) -> list[Patch]:
    try:
        payload = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"patch manifest does not exist: {manifest}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(f"invalid patch manifest: {exc}") from exc

    entries = payload.get("patches")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("manifest must contain at least one [[patches]] entry")

    patches: list[Patch] = []
    identifiers: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"patch entry {index} must be a table")
        category = entry.get("category")
        if category not in CATEGORY_ORDER:
            raise ManifestError(f"patch entry {index} has invalid category: {category!r}")
        target = entry.get("target", "slime")
        if target not in VALID_TARGETS:
            raise ManifestError(f"patch entry {index} has invalid target: {target!r}")
        if category != "third_party" and target != "slime":
            raise ManifestError(
                f"patch entry {index} in {category!r} must target 'slime', got {target!r}"
            )

        # v2 is one patch per table. The grouped `files` form remains readable
        # during migration, while still becoming authoritative.
        values: list[tuple[str, str, str | None]] = []
        if isinstance(entry.get("file"), str):
            file_name = entry["file"]
            identifier = entry.get("id", f"{category}:{file_name}")
            values.append((file_name, identifier, entry.get("sha256")))
        elif isinstance(entry.get("files"), list):
            for file_name in entry["files"]:
                if not isinstance(file_name, str):
                    raise ManifestError(f"patch entry {index} contains a non-string file")
                values.append((file_name, f"{category}:{file_name}", None))
        else:
            raise ManifestError(f"patch entry {index} requires file or files")

        for file_name, identifier, expected_hash in values:
            if not isinstance(identifier, str) or not identifier:
                raise ManifestError(f"patch entry {index} has an invalid id")
            if identifier in identifiers:
                raise ManifestError(f"duplicate patch id: {identifier}")
            identifiers.add(identifier)
            relative = Path(file_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ManifestError(f"unsafe patch path: {file_name}")
            if relative.parts and relative.parts[0] in CATEGORY_ORDER:
                patch_path = patches_root / relative
            else:
                patch_path = patches_root / category / relative
            patches.append(Patch(patch_path.resolve(), category, identifier, target, expected_hash))

    return patches


def validate(manifest: Path, patches_root: Path) -> list[Patch]:
    patches_root = patches_root.resolve()
    patches = load_manifest(manifest.resolve(), patches_root)
    declared = {patch.path for patch in patches}
    actual = {
        path.resolve()
        for category in CATEGORY_ORDER
        for path in (patches_root / category).rglob("*.patch")
        if path.is_file()
    }
    missing = sorted(declared - actual)
    undeclared = sorted(actual - declared)
    if missing:
        raise ManifestError("manifest references missing patches: " + ", ".join(map(str, missing)))
    if undeclared:
        raise ManifestError("patch directory contains undeclared files: " + ", ".join(map(str, undeclared)))
    for patch in patches:
        if patch.expected_sha256:
            actual_hash = sha256_file(patch.path)
            if actual_hash != patch.expected_sha256:
                raise ManifestError(
                    f"sha256 mismatch for {patch.identifier}: expected "
                    f"{patch.expected_sha256}, got {actual_hash}"
                )
    return patches


def apply_patches(repo: Path, patches: list[Patch], target: str = "slime") -> None:
    repo = repo.resolve()
    for patch in patches:
        if patch.target != target:
            continue
        check = subprocess.run(
            ["git", "-C", str(repo), "apply", "--check", str(patch.path)],
            text=True,
            capture_output=True,
        )
        if check.returncode:
            detail = (check.stderr or check.stdout).strip()
            raise ManifestError(f"patch check failed for {patch.identifier}: {detail}")
        subprocess.run(
            ["git", "-C", str(repo), "apply", str(patch.path)], check=True
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "apply"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--patches-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        patches = validate(args.manifest, args.patches_root)
        if args.command == "apply":
            if args.repo is None:
                raise ManifestError("--repo is required for apply")
            apply_patches(args.repo, patches)
        for patch in patches:
            print(f"{patch.category}\t{patch.identifier}\t{patch.path}\t{patch.target}")
        return 0
    except (ManifestError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
