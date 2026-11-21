#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare an inventory with the documented Turbo GLM candidate profile.

Exit 0 means metadata matches the profile. It does not validate imports,
operator execution, NPU availability, free devices, weights, or training.
"""

import argparse
import json
from pathlib import Path
import re


def _stable_version(value):
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\.post(\d+))?(?:\+[A-Za-z0-9_.-]+)?", value or "")
    return tuple(int(part or 0) for part in match.groups()) if match else None


def compare_profile(inventory: dict, profile: dict) -> dict:
    """Return metadata mismatches without importing or modifying the runtime."""
    checks = []

    def add(component, observed, expected, matches):
        checks.append({"component": component, "observed": observed,
                       "expected": expected, "matches": bool(matches)})

    host = inventory["host"]
    python_version = host.get("python_version") or host.get("python", "").split(" ")[0]
    parsed = _stable_version(python_version)
    series = profile["python_series"]
    add("python", python_version, series, parsed and list(parsed[:2]) == [int(v) for v in series.split(".")])

    for name, rule in profile["packages"].items():
        observed = inventory["packages"].get(name)
        version = _stable_version(observed)
        expected = rule["release"]
        matches = version and version[:3] == _stable_version(expected)[:3]
        if "minimum_post" in rule:
            matches = matches and version[3] >= rule["minimum_post"]
            expected += ".post" + str(rule["minimum_post"]) + "+"
        add(name, observed, expected, matches)

    # Driver version files can also contain 'version'; only toolkit files count.
    cann_versions = []
    for path, content in inventory["ascend_version_files"].items():
        if "driver" not in path and isinstance(content, str):
            cann_versions.extend(re.findall(r"(?m)^version\s*=\s*([^\s]+)", content))
    minimum = _stable_version(profile["cann_minimum"])
    add("cann", cann_versions, ">=" + profile["cann_minimum"], cann_versions and all(
        _stable_version(v) and _stable_version(v) >= minimum for v in cann_versions))

    for name in profile["required_modules"]:
        location = inventory["module_locations"].get(name)
        discoverable = isinstance(location, str) or (
            isinstance(location, dict) and bool(location.get("namespace_paths")))
        add("module:" + name, location, "discoverable (import not tested)", discoverable)

    revision = inventory.get("source_revisions", {}).get("fsdp_turbo")
    add("fsdp_turbo_sha", revision, profile["candidate_sha"], revision == profile["candidate_sha"])
    return {"profile": profile["profile"], "matches_documented_profile": all(c["matches"] for c in checks),
            "training_tested": False, "kernel_execution_tested": False, "checks": checks,
            "source_url": profile["source_url"], "note": profile["note"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path, help="JSON produced by probe_env.py")
    args = parser.parse_args()
    try:
        inventory = json.loads(args.inventory.read_text())
        profile = json.loads(Path(__file__).with_name("fsdp_turbo_glm53_profile.json").read_text())
        result = compare_profile(inventory, profile)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(2, "Invalid inventory or profile: " + str(exc) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["matches_documented_profile"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
