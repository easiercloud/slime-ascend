#!/usr/bin/env python3
"""Inventory changed upstream tests against existing Ascend coverage."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIME = ROOT / "slime"
MAPPINGS = {
    "test_ppo_logprob_entropy_gpu.py": "nightly_CI/test_ppo_logprob_entropy_gpu_npu.py",
    "test_qwen2.5_0.5B_debug_rollout_then_train.py": "st/test_qwen2.5_0.5B_debug_rollout_then_train_npu.py",
    "test_qwen2.5_0.5B_fanout_short.py": "nightly_CI/test_qwen2.5_0.5B_fanout_short_npu.py",
    "test_fully_async_rollout.py": "nightly_CI/test_qwen2.5_0.5B_fully_async_short_npu.py",
    "test_sglang_runtime.py": "st/test_qwen2.5_0.5B_gsm8k_async_short_npu.py",
    "test_model_provider_critic.py": "nightly_CI/test_qwen3_4B_ppo_train_critic_only_npu.py",
    "test_sglang_weight_update_compat.py": "st/test_qwen2.5_0.5B_gsm8k_async_short_npu.py",
    "test_training_recovery.py": "nightly_CI/test_qwen2.5_0.5B_fully_async_short_npu.py",
}


def changed_tests(base: str, target: str) -> list[tuple[str, str]]:
    raw = subprocess.check_output(
        ["git", "-C", str(SLIME), "diff", "--name-status", "-z", base, target, "--", "tests"]
    )
    fields = raw.decode().strip(chr(0)).split(chr(0)) if raw else []
    rows = []
    index = 0
    while index < len(fields):
        status = fields[index]
        if status.startswith(("R", "C")):
            path = fields[index + 2]
            index += 3
        else:
            path = fields[index + 1]
            index += 2
        if path.endswith(".py") and Path(path).name.startswith("test_"):
            rows.append((status, path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = changed_tests(args.base, args.target)
    lines = [
        "# Upstream test mapping",
        "",
        f"Base: {args.base}",
        f"Target: {args.target}",
        "",
        "Generated from the exact upstream Git diff. CPU portable cases run against the patched Slime tree in the NPU container. A mapped device case still requires resource and dependency validation.",
        "",
        "| Change | Upstream test | Declared devices | Ascend coverage | Next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    counts = {"CPU": 0, "GPU": 0, "unknown": 0}
    for status, path in rows:
        source = SLIME / path
        body = source.read_text(errors="replace") if source.exists() else ""
        match = re.search(r"^NUM_GPUS\s*=\s*(\d+)\b", body, flags=re.MULTILINE)
        devices = int(match.group(1)) if match else None
        if devices is None and Path(path).name == "test_trajectory_manager_branching.py":
            devices = 0  # Reviewed: trajectory-only CPU contract; no device fixture.
        kind = "unknown" if devices is None else ("CPU" if devices == 0 else "GPU")
        counts[kind] += 1
        mapped = MAPPINGS.get(Path(path).name)
        if mapped:
            target_path = ROOT / "tests/tests_npu" / mapped
            coverage = f"tests/tests_npu/{mapped}" if target_path.is_file() else "MAPPING_MISSING"
            action = "Run upstream CPU UT if device count is zero, then mapped ST/Nightly; inspect changed assertions" if kind == "CPU" else "Run mapped ST/Nightly after UT; inspect changed assertions"
        elif kind == "CPU":
            coverage = "upstream CPU UT in NPU container"
            action = "Run directly; do not duplicate"
        elif kind == "GPU":
            coverage = "open NPU device gap"
            action = "Review model assets, kernels, and device API; migrate if supported"
        else:
            coverage = "unclassified"
            action = "Inspect markers and fixture before mapping"
        lines.append(f"| {status} | {path} | {kind if devices is None else devices} | {coverage} | {action} |")
    lines += ["", f"Total: {len(rows)} changed Python tests; declared CPU {counts['CPU']}, GPU {counts['GPU']}, unclassified {counts['unknown']}.", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(chr(10).join(lines))
    print(lines[-2])


if __name__ == "__main__":
    main()
