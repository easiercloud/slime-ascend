#!/usr/bin/env python3
"""Detect a supported RL-framework profile without importing project code."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SUPPORTED = ("auto", "slime", "verl", "generic")


def project_text(repo: Path) -> str:
    for name in ("pyproject.toml", "setup.py", "setup.cfg"):
        path = repo / name
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace").lower()
    return ""


def detect(repo: Path, requested: str = "auto") -> dict:
    repo = repo.resolve()
    if not repo.is_dir():
        raise ValueError(f"repo-root does not exist: {repo}")
    text = project_text(repo)
    matches: list[str] = []
    slime_markers = [
        repo / "slime",
        repo / "slime_plugins",
        repo / "train_async.py",
    ]
    if sum(marker.exists() for marker in slime_markers) >= 2 or "slime" in text:
        matches.append("slime")
    verl_markers = [repo / "verl", repo / "recipe"]
    if (repo / "verl").is_dir() or "name = \"verl\"" in text or "name='verl'" in text:
        matches.append("verl")

    if requested != "auto":
        profile = requested
        detection = "explicit"
    elif len(matches) == 1:
        profile = matches[0]
        detection = "auto"
    elif len(matches) == 0:
        profile = "generic"
        detection = "unknown"
    else:
        profile = "generic"
        detection = "ambiguous"

    roots = [name for name in ("tests", "test", "verl/tests") if (repo / name).is_dir()]
    return {
        "schema_version": 1,
        "framework": profile,
        "detection": detection,
        "matches": matches,
        "repo_root": str(repo),
        "candidate_test_roots": roots,
        "requires_profile_confirmation": detection in {"unknown", "ambiguous"},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--framework", choices=SUPPORTED, default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = detect(args.repo_root, args.framework)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False))
    return 2 if args.strict and payload["requires_profile_confirmation"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
