#!/usr/bin/env python3
"""Rebuild patches and overlay in a second clean tree and compare content."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from patch_manifest import ManifestError, apply_patches, validate
from tree_manifest import compare, create_manifest, write_json


class ReplayError(RuntimeError):
    pass


def run(argv: list[str], cwd: Path | None = None) -> str:
    process = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    if process.returncode:
        detail = (process.stderr or process.stdout).strip()
        raise ReplayError(f"command failed: {' '.join(argv)}: {detail}")
    return process.stdout.strip()


def copy_overlay(overlay: Path, replay: Path, allow_overwrite: bool) -> None:
    for source in sorted(overlay.rglob("*")):
        relative = source.relative_to(overlay)
        target = replay / relative
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists() and not allow_overwrite:
            raise ReplayError(
                f"overlay must contain new files only; target already exists: {relative}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(source.readlink())
        else:
            shutil.copy2(source, target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validated-repo", type=Path, required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--patches-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--allow-overlay-overwrite", action="store_true")
    parser.add_argument("--keep-replay-tree", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validated = args.validated_repo.resolve()
    overlay = args.overlay.resolve()
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay = Path(tempfile.mkdtemp(prefix="slime-replay-", dir=args.work_root)).resolve()
    cleanup = not args.keep_replay_tree
    try:
        actual_sha = run(["git", "rev-parse", "HEAD"], cwd=validated)
        if actual_sha != args.target_sha:
            raise ReplayError(
                f"validated repo HEAD {actual_sha} does not match target {args.target_sha}"
            )
        patches = validate(args.manifest, args.patches_root)
        run(["git", "clone", "--shared", "--no-checkout", str(validated), str(replay)])
        run(["git", "checkout", "--detach", args.target_sha], cwd=replay)
        apply_patches(replay, patches, target="slime")
        copy_overlay(overlay, replay, args.allow_overlay_overwrite)

        validated_manifest = create_manifest(validated, args.exclude)
        replay_manifest = create_manifest(replay, args.exclude)
        differences = compare(validated_manifest, replay_manifest)
        write_json(args.output_dir / "validated-tree.json", validated_manifest)
        write_json(args.output_dir / "replay-tree.json", replay_manifest)
        result = {
            "schema_version": 1,
            "replay_verified": not differences,
            "target_sha": args.target_sha,
            "validated_tree_sha256": validated_manifest["tree_sha256"],
            "replay_tree_sha256": replay_manifest["tree_sha256"],
            "differences": differences,
            "replay_tree": str(replay) if args.keep_replay_tree else None,
        }
        write_json(args.output_dir / "replay-result.json", result)
        if differences:
            print("replay mismatch:\n" + "\n".join(differences), file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ManifestError, ReplayError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        if cleanup and replay.exists():
            shutil.rmtree(replay)


if __name__ == "__main__":
    raise SystemExit(main())
