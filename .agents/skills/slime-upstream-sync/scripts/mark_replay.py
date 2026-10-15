#!/usr/bin/env python3
"""Atomically attach a successful deterministic replay to result.json."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--replay-result", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = json.loads(args.result.read_text(encoding="utf-8"))
        replay = json.loads(args.replay_result.read_text(encoding="utf-8"))
        if replay.get("replay_verified") is not True:
            raise ValueError("replay result is not verified")
        result["replay_verified"] = True
        result["replay_result_path"] = ".upstream-sync/replay/replay-result.json"
        temporary = args.result.with_suffix(args.result.suffix + ".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.result)
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
