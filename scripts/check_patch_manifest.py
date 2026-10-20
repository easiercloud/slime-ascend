#!/usr/bin/env python3
"""Verify every registered patch before installation or submodule reset."""

from __future__ import annotations

import hashlib
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "patches" / "MANIFEST.toml"


def main() -> int:
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    problems: list[str] = []
    for item in data["patches"]:
        path = REPO_ROOT / "patches" / item["category"] / item["file"]
        if not path.is_file():
            problems.append(f"missing: {path.relative_to(REPO_ROOT)}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            problems.append(f"sha256 mismatch: {path.relative_to(REPO_ROOT)}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"Verified {len(data['patches'])} registered patches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
