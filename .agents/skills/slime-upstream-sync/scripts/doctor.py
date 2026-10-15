#!/usr/bin/env python3
"""Portable fail-closed preflight for slime upstream synchronization hosts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SGLANG_IMAGE_RE = re.compile(r"^ARG\s+SGLANG_IMAGE_TAG=(?:v)?([^\s-]+)", re.MULTILINE)
MEGATRON_COMMIT_RE = re.compile(r"^ARG\s+MEGATRON_COMMIT=([0-9a-f]{7,40})", re.MULTILINE)
ROUTER_WHEEL_RE = re.compile(r"pip install (https://\S+/sglang_router-[^\s]+\.whl)")
MEMFABRIC_HYBRID_RE = re.compile(r"memfabric-hybrid==([0-9]+(?:\.[0-9]+)+)")
SHELL_ASSIGNMENT_RE = re.compile(
    r"^(SGLANG_COMMIT|MEGATRON_COMMIT|SGLANG_ROUTER_WHEEL|"
    r"SGL_KERNEL_NPU_VERSION|SGL_KERNEL_NPU_WHEEL_VERSION|SGL_KERNEL_NPU_FIX_COMMIT|"
    r"MEMFABRIC_HYBRID_VERSION|CANN_VERSION|MBRIDGE_COMMIT|"
    r"MEGATRON_BRIDGE_COMMIT|MEGATRON_ADAPTOR_COMMIT|"
    r"TRANSFORMER_ENGINE_NPU_COMMIT)=([^\s#]+)",
    re.MULTILINE,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def writable_parent(path: Path) -> bool:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK | os.X_OK)


def command_version(command: list[str]) -> dict:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "path": None, "version": None, "timed_out": False}
    try:
        process = subprocess.run(command, text=True, capture_output=True, timeout=20)
    except subprocess.TimeoutExpired:
        return {
            "available": False,
            "path": executable,
            "version": "probe timed out after 20 seconds",
            "timed_out": True,
        }
    output = (process.stdout or process.stderr).strip().splitlines()
    return {
        "available": process.returncode == 0,
        "path": executable,
        "version": output[0] if output else None,
        "timed_out": False,
    }


def add_check(checks: list[dict], name: str, passed: bool, detail: str, required: bool = True) -> None:
    checks.append(
        {"name": name, "passed": bool(passed), "required": required, "detail": detail}
    )


def upstream_dependency_contract(slime_root: Path) -> dict[str, str]:
    dockerfile = slime_root / "docker" / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    sglang = SGLANG_IMAGE_RE.search(text)
    megatron = MEGATRON_COMMIT_RE.search(text)
    router = ROUTER_WHEEL_RE.search(text)
    if not sglang or not megatron or not router:
        raise ValueError(f"cannot resolve SGLang/Megatron/router pins from {dockerfile}")
    return {
        "sglang": sglang.group(1),
        "megatron": megatron.group(1),
        "sglang_router_wheel": router.group(1),
        "sglang_router_marker": "slime",
    }


def installer_dependency_contract(installer: Path) -> dict[str, str]:
    values = dict(SHELL_ASSIGNMENT_RE.findall(installer.read_text(encoding="utf-8")))
    missing = {
        "SGLANG_COMMIT",
        "MEGATRON_COMMIT",
        "SGLANG_ROUTER_WHEEL",
        "SGL_KERNEL_NPU_VERSION",
        "SGL_KERNEL_NPU_WHEEL_VERSION",
        "SGL_KERNEL_NPU_FIX_COMMIT",
        "MEMFABRIC_HYBRID_VERSION",
        "CANN_VERSION",
        "MBRIDGE_COMMIT",
        "MEGATRON_BRIDGE_COMMIT",
        "MEGATRON_ADAPTOR_COMMIT",
        "TRANSFORMER_ENGINE_NPU_COMMIT",
    } - values.keys()
    if missing:
        raise ValueError(f"missing assignments in {installer}: {', '.join(sorted(missing))}")
    return {
        "sglang": values["SGLANG_COMMIT"].removeprefix("v"),
        "megatron": values["MEGATRON_COMMIT"],
        "sglang_router_wheel": values["SGLANG_ROUTER_WHEEL"],
        "sgl_kernel_npu_version": values["SGL_KERNEL_NPU_VERSION"],
        "sgl_kernel_npu_wheel_version": values["SGL_KERNEL_NPU_WHEEL_VERSION"],
        "sgl_kernel_npu_fix_commit": values["SGL_KERNEL_NPU_FIX_COMMIT"],
        "memfabric_hybrid_version": values["MEMFABRIC_HYBRID_VERSION"],
        "cann_version": values["CANN_VERSION"],
        "mbridge": values["MBRIDGE_COMMIT"],
        "megatron_bridge": values["MEGATRON_BRIDGE_COMMIT"],
        "megatron_adaptor": values["MEGATRON_ADAPTOR_COMMIT"],
        "transformer_engine_npu": values["TRANSFORMER_ENGINE_NPU_COMMIT"],
    }


def sglang_npu_dependency_contract(sglang_root: Path) -> dict[str, str]:
    dockerfile = sglang_root / "docker" / "npu.Dockerfile"
    match = MEMFABRIC_HYBRID_RE.search(dockerfile.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"cannot resolve MemFabric pin from {dockerfile}")
    return {"memfabric_hybrid_version": match.group(1)}


def router_runtime_state(marker: str) -> dict:
    probe = subprocess.run(
        [sys.executable, "-c", "import sglang_router; print(sglang_router.__version__)"],
        text=True,
        capture_output=True,
    )
    version = probe.stdout.strip() if probe.returncode == 0 else None
    detail = f"version={version!r} required_marker={marker!r}"
    return {"version": version, "matches": bool(version and marker in version), "detail": detail}


def sgl_kernel_npu_runtime_state(expected_version: str, fix_commit: str) -> dict:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.metadata, inspect; "
            "from sgl_kernel_npu.norm import fused_split_qk_norm as module; "
            "print(importlib.metadata.version('sgl-kernel-npu')); "
            "print('has_bias_guard=' + str('hasattr(q_a_layernorm' in inspect.getsource(module)))",
        ],
        text=True,
        capture_output=True,
    )
    lines = probe.stdout.strip().splitlines()
    version = lines[0] if lines else None
    has_bias_guard = "has_bias_guard=True" in lines[1:]
    matches = probe.returncode == 0 and version == expected_version and has_bias_guard
    detail = (
        f"version={version!r} expected={expected_version!r} "
        f"official_fix_commit={fix_commit} has_bias_guard={has_bias_guard}"
    )
    if probe.returncode != 0:
        detail += f" probe_error={probe.stderr.strip()!r}"
    return {
        "version": version,
        "fix_commit": fix_commit,
        "has_bias_guard": has_bias_guard,
        "matches": matches,
        "detail": detail,
    }


def memfabric_hybrid_runtime_state(expected_version: str) -> dict:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.metadata; import memfabric_hybrid; "
            "print(importlib.metadata.version('memfabric-hybrid'))",
        ],
        text=True,
        capture_output=True,
    )
    version = probe.stdout.strip() if probe.returncode == 0 else None
    matches = probe.returncode == 0 and version == expected_version
    detail = f"version={version!r} expected={expected_version!r} importable={probe.returncode == 0}"
    if probe.returncode != 0:
        detail += f" probe_error={probe.stderr.strip()!r}"
    return {"version": version, "matches": matches, "detail": detail}


def dependency_pin_matches(name: str, declared: str, expected: str) -> bool:
    if name == "megatron":
        return len(declared) >= 7 and expected.startswith(declared)
    return declared == expected


def git_dependency_state(repo: Path, expected_ref: str) -> dict:
    if not (repo / ".git").exists():
        return {"path": str(repo), "head": None, "expected_commit": None, "matches": False, "detail": "not a git worktree"}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.STDOUT).strip()
        expected = subprocess.check_output(
            ["git", "rev-parse", f"{expected_ref}^{{commit}}"], cwd=repo, text=True, stderr=subprocess.STDOUT
        ).strip()
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", expected, head], cwd=repo, capture_output=True, text=True
        ).returncode == 0
        detail = f"HEAD={head} expected_base={expected}"
        return {"path": str(repo), "head": head, "expected_commit": expected, "matches": ancestor, "detail": detail}
    except subprocess.CalledProcessError as error:
        detail = (error.stdout or str(error)).strip()
        return {"path": str(repo), "head": None, "expected_commit": None, "matches": False, "detail": detail}


def git_component_inventory(repo: Path) -> dict:
    if not (repo / ".git").exists():
        return {"kind": "git", "path": str(repo), "present": False, "commit": None}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=repo, text=True, capture_output=True
        )
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=repo, text=True
            ).strip()
        )
        return {
            "kind": "git",
            "path": str(repo),
            "present": True,
            "commit": head,
            "remote": remote.stdout.strip() if remote.returncode == 0 else None,
            "dirty": dirty,
        }
    except subprocess.CalledProcessError as error:
        return {
            "kind": "git",
            "path": str(repo),
            "present": False,
            "commit": None,
            "error": (error.stdout or str(error)).strip(),
        }


def package_component_inventory(distribution: str) -> dict:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.metadata as m, json; d=m.distribution(" + repr(distribution) + "); "
            "print(json.dumps({'version': d.version, 'path': str(d.locate_file('')), "
            "'direct_url': d.read_text('direct_url.json')}))",
        ],
        text=True,
        capture_output=True,
    )
    if probe.returncode:
        return {"kind": "python_distribution", "present": False, "version": None}
    payload = json.loads(probe.stdout)
    direct_url = payload.pop("direct_url", None)
    payload.update({"kind": "python_distribution", "present": True})
    if direct_url:
        try:
            payload["direct_url"] = json.loads(direct_url)
        except json.JSONDecodeError:
            payload["direct_url"] = direct_url
    return payload


def cann_runtime_state(root: Path, expected_version: str) -> dict:
    candidates = (
        root / f"cann-{expected_version}" / "share" / "info" / "runtime" / "version.info",
        root / "ascend-toolkit" / "latest" / "version.cfg",
    )
    version_file = next((path for path in candidates if path.is_file()), None)
    text = version_file.read_text(encoding="utf-8", errors="replace") if version_file else ""
    match = re.search(r"^(?:Version|version)\s*=\s*([^\s]+)", text, re.MULTILINE)
    version = match.group(1) if match else None
    atb_files = sorted((root / "nnal" / "atb").glob("*/version.info"))
    atb_text = atb_files[-1].read_text(encoding="utf-8", errors="replace") if atb_files else ""
    atb_version = re.search(r"Ascend-cann-atb Version\s*:\s*(\S+)", atb_text)
    atb_commit = re.search(r"commit id\s*:\s*([0-9a-f]{7,40})", atb_text)
    matches = version == expected_version
    detail = (
        f"version={version!r} expected={expected_version!r} version_file={str(version_file)!r} "
        f"atb_version={atb_version.group(1) if atb_version else None!r} "
        f"atb_commit={atb_commit.group(1) if atb_commit else None!r}"
    )
    return {
        "kind": "binary_stack",
        "path": str(root),
        "version": version,
        "commit": None,
        "build_identity": text.strip() or None,
        "atb_version": atb_version.group(1) if atb_version else None,
        "atb_commit": atb_commit.group(1) if atb_commit else None,
        "matches": matches,
        "detail": detail,
    }


def sglang_python_runtime_state(repo: Path, git_state: dict) -> dict:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.metadata as m, json, sglang; "
            "d=m.distribution('sglang'); "
            "print(json.dumps({'metadata_version': d.version, "
            "'module_version': sglang.__version__, 'module_file': sglang.__file__, "
            "'direct_url': json.loads(d.read_text('direct_url.json') or '{}')}))",
        ],
        text=True,
        capture_output=True,
    )
    try:
        payload = json.loads(probe.stdout) if probe.returncode == 0 else {}
    except json.JSONDecodeError:
        payload = {}

    python_root = (repo / "python").resolve()
    module_file = Path(payload.get("module_file", "/invalid")).resolve()
    direct_url = payload.get("direct_url", {})
    editable = direct_url.get("dir_info", {}).get("editable") is True
    source_url = direct_url.get("url")
    expected_url = python_root.as_uri()
    versions_match = bool(payload.get("metadata_version")) and (
        payload.get("metadata_version") == payload.get("module_version")
    )
    head = git_state.get("head") or ""
    marker_match = re.search(r"\+g([0-9a-f]{7,40})(?:\.|$)", payload.get("metadata_version", ""))
    head_marker = marker_match.group(1) if marker_match else ""
    metadata_fresh = bool(head_marker and head.startswith(head_marker))
    source_matches = module_file.is_relative_to(python_root) and source_url == expected_url
    matches = (
        probe.returncode == 0
        and git_state.get("matches") is True
        and editable
        and source_matches
        and versions_match
        and metadata_fresh
    )
    detail = (
        f"{git_state.get('detail')} metadata_version={payload.get('metadata_version')!r} "
        f"module_version={payload.get('module_version')!r} module_file={str(module_file)!r} "
        f"editable={editable} source_url={source_url!r} current_head_marker={head_marker!r}"
    )
    if probe.returncode != 0:
        detail += f" probe_error={probe.stderr.strip()!r}"
    return {
        **git_state,
        **payload,
        "editable": editable,
        "source_matches": source_matches,
        "versions_match": versions_match,
        "metadata_fresh": metadata_fresh,
        "matches": matches,
        "detail": detail,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--check-sync-layout", action="store_true")
    parser.add_argument("--require-npu", action="store_true")
    parser.add_argument("--require-assets", action="store_true")
    parser.add_argument("--check-upstream-dependencies", action="store_true")
    parser.add_argument("--dependency-root", type=Path, default=Path("/workspace"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    skill_root = Path(__file__).resolve().parent.parent
    skills_root = skill_root.parent
    checks: list[dict] = []

    add_check(checks, "platform_linux", platform.system() == "Linux", platform.platform())
    commands = {
        "python": command_version([sys.executable, "--version"]),
        "git": command_version(["git", "--version"]),
        "bash": command_version(["bash", "--version"]),
        "codex": command_version(["codex", "--version"]),
    }
    for name, result in commands.items():
        add_check(
            checks,
            f"command_{name}",
            result["available"],
            result["version"] or "not available",
        )

    add_check(checks, "repo_root", repo.is_dir(), str(repo))
    add_check(checks, "state_dir_writable", writable_parent(args.state_dir), str(args.state_dir))
    add_check(checks, "work_root_writable", writable_parent(args.work_root), str(args.work_root))

    child_skills = {}
    for name in ("gpu-to-npu-test-converter", "npu-test-autofix"):
        path = skills_root / name / "SKILL.md"
        child_skills[name] = {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        add_check(checks, f"skill_{name}", path.is_file(), str(path))

    runtime_paths = (
        skill_root / "scripts" / "run_sync.py",
        skill_root / "scripts" / "collect_change_assessment.py",
        skill_root / "scripts" / "materialize_overlay.sh",
        skill_root / "scripts" / "finalize_overlay.sh",
        skill_root / "references" / "agent_prompt.md",
    )
    for path in runtime_paths:
        add_check(checks, f"runtime_{path.name}", path.is_file(), str(path))

    if args.check_sync_layout:
        for relative in ("patches/MANIFEST.toml", "overlay", "slime"):
            path = repo / relative
            add_check(checks, f"layout_{relative}", path.exists(), str(path))

    dependencies = {}
    component_inventory = {}
    if args.check_upstream_dependencies:
        try:
            expected = upstream_dependency_contract(repo / "slime")
            expected.update(sglang_npu_dependency_contract(args.dependency_root / "sglang"))
            declared = installer_dependency_contract(repo / "scripts" / "quick_install.sh")
            dependencies = {"expected": expected, "installer": declared, "runtime": {}}
            for name in ("sglang", "megatron"):
                add_check(
                    checks,
                    f"dependency_installer_{name}",
                    dependency_pin_matches(name, declared[name], expected[name]),
                    f"expected={expected[name]} installer={declared[name]}",
                )
            add_check(
                checks,
                "dependency_installer_sglang_router",
                declared["sglang_router_wheel"] == expected["sglang_router_wheel"],
                f"expected={expected['sglang_router_wheel']} installer={declared['sglang_router_wheel']}",
            )
            kernel_installer_valid = (
                bool(re.fullmatch(r"\d{4}\.\d+\.\d+", declared["sgl_kernel_npu_version"]))
                and bool(re.fullmatch(r"\d{4}\.\d+\.\d+", declared["sgl_kernel_npu_wheel_version"]))
                and bool(re.fullmatch(r"[0-9a-f]{40}", declared["sgl_kernel_npu_fix_commit"]))
            )
            add_check(
                checks,
                "dependency_installer_sgl_kernel_npu",
                kernel_installer_valid,
                "classification=NPU_EQUIVALENT "
                f"release={declared['sgl_kernel_npu_version']} "
                f"wheel_metadata={declared['sgl_kernel_npu_wheel_version']} "
                f"official_fix_commit={declared['sgl_kernel_npu_fix_commit']}",
            )
            add_check(
                checks,
                "dependency_installer_memfabric_hybrid",
                declared["memfabric_hybrid_version"] == expected["memfabric_hybrid_version"],
                "classification=NPU_EQUIVALENT "
                f"expected={expected['memfabric_hybrid_version']} "
                f"installer={declared['memfabric_hybrid_version']}",
            )
            for name in (
                "mbridge",
                "megatron_bridge",
                "megatron_adaptor",
                "transformer_engine_npu",
            ):
                valid = bool(re.fullmatch(r"[0-9a-f]{40}", declared[name]))
                add_check(
                    checks,
                    f"dependency_installer_{name}",
                    valid,
                    f"classification=NPU_EQUIVALENT commit={declared[name]}",
                )
            add_check(
                checks,
                "dependency_installer_cann",
                bool(re.fullmatch(r"\d+(?:\.\d+)+", declared["cann_version"])),
                f"classification=NPU_EQUIVALENT version={declared['cann_version']}",
            )
            runtime_specs = {
                "sglang": (args.dependency_root / "sglang", f"v{expected['sglang']}"),
                "megatron": (args.dependency_root / "Megatron-LM", expected["megatron"]),
                "mbridge": (args.dependency_root / "mbridge", declared["mbridge"]),
                "megatron_bridge": (
                    args.dependency_root / "Megatron-Bridge",
                    declared["megatron_bridge"],
                ),
                "megatron_adaptor": (
                    args.dependency_root / "MegatronAdaptor",
                    declared["megatron_adaptor"],
                ),
                "transformer_engine_npu": (
                    args.dependency_root / "TransformerEngineNPU",
                    declared["transformer_engine_npu"],
                ),
            }
            for name, (path, expected_ref) in runtime_specs.items():
                state = git_dependency_state(path, expected_ref)
                if name == "sglang":
                    state = sglang_python_runtime_state(path, state)
                dependencies["runtime"][name] = state
                add_check(checks, f"dependency_runtime_{name}", state["matches"], state["detail"])
            router = router_runtime_state(expected["sglang_router_marker"])
            dependencies["runtime"]["sglang_router"] = router
            add_check(checks, "dependency_runtime_sglang_router", router["matches"], router["detail"])
            kernel = sgl_kernel_npu_runtime_state(
                declared["sgl_kernel_npu_wheel_version"],
                declared["sgl_kernel_npu_fix_commit"],
            )
            kernel["release_version"] = declared["sgl_kernel_npu_version"]
            dependencies["runtime"]["sgl_kernel_npu"] = kernel
            add_check(
                checks,
                "dependency_runtime_sgl_kernel_npu",
                kernel["matches"],
                kernel["detail"],
            )
            memfabric = memfabric_hybrid_runtime_state(expected["memfabric_hybrid_version"])
            dependencies["runtime"]["memfabric_hybrid"] = memfabric
            add_check(
                checks,
                "dependency_runtime_memfabric_hybrid",
                memfabric["matches"],
                memfabric["detail"],
            )
            cann = cann_runtime_state(
                Path(os.environ.get("CANN_INSTALL_PATH", "/usr/local/Ascend")),
                declared["cann_version"],
            )
            dependencies["runtime"]["cann"] = cann
            add_check(checks, "dependency_runtime_cann", cann["matches"], cann["detail"])

            git_components = {
                "slime": repo / "slime",
                "sglang": args.dependency_root / "sglang",
                "megatron": args.dependency_root / "Megatron-LM",
                "megatron_adaptor": args.dependency_root / "MegatronAdaptor",
                "mbridge": args.dependency_root / "mbridge",
                "megatron_bridge": args.dependency_root / "Megatron-Bridge",
                "transformer_engine_npu": args.dependency_root / "TransformerEngineNPU",
            }
            component_inventory = {
                name: git_component_inventory(path) for name, path in git_components.items()
            }
            slime_head = component_inventory["slime"].get("commit")
            component_inventory["slime"].update(
                {"required_base_commit": slime_head, "patched_head": slime_head}
            )
            for name in (
                "sglang",
                "megatron",
                "mbridge",
                "megatron_bridge",
                "megatron_adaptor",
                "transformer_engine_npu",
            ):
                runtime = dependencies["runtime"][name]
                component_inventory[name].update(
                    {
                        "required_base_commit": runtime.get("expected_commit"),
                        "patched_head": runtime.get("head"),
                        "base_is_ancestor": runtime.get("matches"),
                    }
                )
            component_inventory["cann"] = cann
            for name, distribution in {
                "sglang_router": "sglang-router",
                "sgl_kernel_npu": "sgl-kernel-npu",
                "torch": "torch",
                "torch_npu": "torch-npu",
                "deep_ep": "deep-ep",
                "torch_memory_saver": "torch-memory-saver",
                "memfabric_hybrid": "memfabric-hybrid",
                "triton_ascend": "triton-ascend",
                "megatron_core": "megatron-core",
                "megatron_bridge_package": "megatron-bridge",
                "mbridge_package": "mbridge",
                "transformer_engine": "transformer-engine",
                "mindspeed": "mindspeed",
                "nvidia_modelopt": "nvidia-modelopt",
                "transformers": "transformers",
                "ray": "ray",
                "fastapi": "fastapi",
                "mooncake_transfer_engine": "mooncake-transfer-engine",
                "nixl": "nixl",
            }.items():
                component_inventory[name] = package_component_inventory(distribution)
        except (OSError, ValueError) as error:
            dependencies = {"error": str(error)}
            add_check(checks, "dependency_contract", False, str(error))

    npu = command_version(["npu-smi", "info"])
    add_check(
        checks,
        "npu_smi",
        npu["available"],
        npu["version"] or "not available",
        required=args.require_npu,
    )
    torch_npu_available = importlib.util.find_spec("torch_npu") is not None
    add_check(
        checks,
        "python_torch_npu",
        torch_npu_available,
        "importable" if torch_npu_available else "not importable",
        required=args.require_npu,
    )

    asset_variables = (
        "SLIME_TEST_HF_WEIGHTS_ROOT",
        "SLIME_TEST_DATASETS_ROOT",
        "SLIME_TEST_MEGATRON_WEIGHTS_ROOT",
    )
    assets = {}
    for variable in asset_variables:
        raw = os.environ.get(variable)
        path = Path(raw).expanduser().resolve() if raw else None
        valid = path is not None and path.is_dir() and os.access(path, os.R_OK | os.X_OK)
        assets[variable] = {"configured": raw is not None, "path": str(path) if path else None}
        add_check(
            checks,
            f"asset_{variable}",
            valid,
            str(path) if path else "not configured",
            required=args.require_assets,
        )

    failures = [item for item in checks if item["required"] and not item["passed"]]
    payload = {
        "schema_version": 1,
        "status": "PASS" if not failures else "NG",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo),
        "skill_root": str(skill_root),
        "platform": {"system": platform.system(), "release": platform.release()},
        "commands": commands,
        "child_skills": child_skills,
        "assets": assets,
        "dependencies": dependencies,
        "component_inventory": component_inventory,
        "environment_presence": {
            name: bool(os.environ.get(name))
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE")
        },
        "checks": checks,
        "failure_count": len(failures),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for item in checks:
        label = "PASS" if item["passed"] else ("NG" if item["required"] else "WARN")
        print(f"{label}\t{item['name']}\t{item['detail']}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
