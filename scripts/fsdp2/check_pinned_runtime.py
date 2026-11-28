"""Check the Slime core training pins and the small Turbo source backport."""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    lock = json.loads(Path(__file__).with_name("pinned-runtime.json").read_text())
    errors = []
    actual = {"python": platform.python_version(), "architecture": platform.machine(), "packages": {}}
    if platform.machine() != "aarch64" or platform.python_version_tuple()[:2] != ("3", "12"):
        errors.append("Expected Linux aarch64 Python 3.12")
    for name, expected in lock["packages"].items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = "missing"
        actual["packages"][name] = version
        if version.split("+")[0] != expected:
            errors.append(f"{name}: expected {expected}, got {version}")
    info = Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info")
    if not info.is_file() or f"version={lock['cann']}" not in info.read_text().splitlines():
        errors.append("CANN version does not match the Slime 9.1.0 pin")
    spec = importlib.util.find_spec("fsdp_turbo")
    if spec is None or spec.origin is None:
        errors.append("FSDPTurbo is not installed")
    else:
        source = Path(spec.origin).resolve().parent
        actual["turbo_source"] = str(source)
        actual["turbo_device_sha256"] = hashlib.sha256((source / "utils/device.py").read_bytes()).hexdigest()
        if actual["turbo_device_sha256"] != lock["turbo"]["device_sha256"]:
            errors.append("Turbo accelerator source does not match the upstream backport")
        revision = subprocess.run(["git", "-C", str(source.parent), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=10)
        actual["turbo_commit"] = revision.stdout.strip()
        ancestry = subprocess.run(["git", "-C", str(source.parent), "merge-base", "--is-ancestor",
                                   lock["turbo"]["base_commit"], "HEAD"], capture_output=True, timeout=10)
        changed = subprocess.run(["git", "-C", str(source.parent), "diff", "--name-only",
                                  lock["turbo"]["base_commit"], "--"], capture_output=True, text=True, timeout=10)
        if revision.returncode or ancestry.returncode or changed.returncode or changed.stdout.splitlines() != ["fsdp_turbo/utils/device.py"]:
            errors.append("Turbo checkout must match the pinned base plus the accelerator backport")
    print(json.dumps({"actual": actual, "errors": errors, "runtime_match": not errors}, indent=2))
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
