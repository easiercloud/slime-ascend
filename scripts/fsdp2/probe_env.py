#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Collect a host/container inventory before choosing an FSDP2 runtime.

This script does not import training frameworks, initialize NPU/HCCL, install
packages, download models, or claim that an adapter is supported.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


DISTRIBUTIONS = (
    "torch", "torch-npu", "transformers", "ray", "mindspeed", "mindspeed-mm",
    "fsdp-turbo", "triton", "triton-ascend", "flash-linear-attention",
    "fla-core", "fla-linear-attention", "tiktoken",
)
MODULES = ("torch", "torch_npu", "mindspeed_mm", "fsdp_turbo", "ray",
           "fla", "fla_npu", "triton_ascend_kernels", "cann_ops_transformer")
VERSION_FILES = (
    "/usr/local/Ascend/driver/version.info",
    "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
    "/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info",
)


def run_readonly(argv):
    if shutil.which(argv[0]) is None:
        return {"available": False, "command": argv}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {"available": True, "command": argv, "exit_code": result.returncode,
                "stdout": result.stdout[:30000], "stderr": result.stderr[:3000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "command": argv, "error": str(exc)}


def collect():
    packages = {}
    for name in DISTRIBUTIONS:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    modules = {}
    for name in MODULES:
        try:
            spec = importlib.util.find_spec(name)
            if spec is None:
                modules[name] = None
            elif spec.origin is not None:
                modules[name] = spec.origin
            else:
                modules[name] = {"namespace_paths": list(spec.submodule_search_locations or [])}
        except (ImportError, ValueError, OSError) as exc:
            modules[name] = {"error": str(exc)}
    files = {}
    for name in VERSION_FILES:
        try:
            files[name] = Path(name).read_text(errors="replace")[:16000]
        except FileNotFoundError:
            files[name] = None
        except OSError as exc:
            files[name] = {"error": str(exc)}
    revisions = {}
    origin = modules.get("fsdp_turbo")
    if isinstance(origin, str):
        source_root = Path(origin).resolve().parent.parent
        if (source_root / ".git").exists():
            result = run_readonly(["git", "-C", str(source_root), "rev-parse", "HEAD"])
            if result.get("exit_code") == 0:
                revisions["fsdp_turbo"] = result["stdout"].strip()
    return {
        "schema_version": 2,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"hostname": platform.node(), "machine": platform.machine(),
                 "system": platform.system(), "release": platform.release(),
                 "python": sys.version, "python_version": platform.python_version(),
                 "python_executable": sys.executable},
        "packages": packages,
        "module_locations": modules,
        "source_revisions": revisions,
        "ascend_version_files": files,
        "visible_devices": {key: os.environ.get(key) for key in (
            "ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "LOCAL_RANK", "WORLD_SIZE")},
        "npu_smi": run_readonly(["npu-smi", "info"]),
        "assessment": {
            "training_tested": False,
            "adapter_selected": None,
            "note": "Inventory only. Confirm card type, image digest, operators, real BF16 weights and EP step separately.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write a new JSON report; refuses to overwrite an existing file")
    args = parser.parse_args()
    report = json.dumps(collect(), ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents accidentally replacing a previous inventory.
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report)
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
