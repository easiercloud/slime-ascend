#!/usr/bin/env python3
"""Validate synchronization conversion plans and final machine results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SHA_LENGTH = 40
CASE_STATUSES = {
    "PLANNED",
    "STATIC_READY",
    "BLOCKED",
    "VALIDATED_PASS",
    "VALIDATED_NG",
    "UNEXECUTED_MISSING_ASSET",
}


class ContractError(RuntimeError):
    pass


def read_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("top-level JSON value must be an object")
    return payload


def require_sha(value: object, label: str) -> None:
    if not isinstance(value, str) or len(value) != SHA_LENGTH or any(
        char not in "0123456789abcdefABCDEF" for char in value
    ):
        raise ContractError(f"{label} must be a 40-character hexadecimal SHA")


def validate_conversion(payload: dict) -> None:
    if payload.get("schema_version") != 1:
        raise ContractError("conversion plan schema_version must be 1")
    require_sha(payload.get("base_sha"), "base_sha")
    require_sha(payload.get("target_sha"), "target_sha")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ContractError("cases must be an array")
    targets: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ContractError(f"case {index} must be an object")
        for field in ("source", "target", "change_type", "test_type", "status"):
            if not isinstance(case.get(field), str) or not case[field]:
                raise ContractError(f"case {index} has invalid {field}")
        if case["target"] in targets:
            raise ContractError(f"duplicate conversion target: {case['target']}")
        targets.add(case["target"])
        if case["status"] not in CASE_STATUSES:
            raise ContractError(f"case {index} has invalid status: {case['status']}")
        contract = case.get("behavior_contract")
        if not isinstance(contract, dict) or not isinstance(contract.get("capability"), str):
            raise ContractError(f"case {index} lacks a behavior contract")
        if not isinstance(contract.get("phases"), list) or not isinstance(
            contract.get("success_criteria"), list
        ):
            raise ContractError(f"case {index} behavior contract is incomplete")


def safe_report(workspace: Path, report_path: object) -> Path:
    if not isinstance(report_path, str) or not report_path:
        raise ContractError("report_path must be a non-empty string")
    workspace = workspace.resolve()
    report = (workspace / report_path).resolve()
    if workspace != report and workspace not in report.parents:
        raise ContractError("report_path escapes the workspace")
    if not report.is_file() or report.stat().st_size == 0:
        raise ContractError(f"report does not exist or is empty: {report}")
    text = report.read_text(encoding="utf-8")
    if sum("\u4e00" <= char <= "\u9fff" for char in text) < 4:
        raise ContractError("report does not contain enough Chinese content")
    return report


def validate_report_content(report: Path) -> None:
    text = report.read_text(encoding="utf-8")
    required_terms = (
        "上游变化",
        "测试变化",
        "SGLang",
        "Megatron",
        "MegatronAdaptor",
        "mbridge",
        "Megatron-Bridge",
        "TransformerEngineNPU",
        "CANN",
        "SGL Kernel NPU",
        "torch-npu",
        "DeepEP",
        "TorchMemorySaver",
        "MemFabric",
        "Triton Ascend",
        "ATB",
        "dev 分支集成",
    )
    missing = [term for term in required_terms if term not in text]
    if missing:
        raise ContractError("report lacks change/dependency sections: " + ", ".join(missing))


def safe_workspace_file(workspace: Path, path_value: object, label: str) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ContractError(f"{label} must be a non-empty string")
    workspace = workspace.resolve()
    resolved = (workspace / path_value).resolve()
    if workspace != resolved and workspace not in resolved.parents:
        raise ContractError(f"{label} escapes the workspace")
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ContractError(f"{label} does not exist or is empty: {resolved}")
    return resolved


def validate_environment(path: Path) -> None:
    payload = read_object(path)
    if payload.get("status") != "PASS":
        raise ContractError("environment preflight must be PASS")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise ContractError("environment checks must be an array")
    by_name = {
        item.get("name"): item
        for item in checks
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    required = {
        "dependency_installer_sglang",
        "dependency_installer_megatron",
        "dependency_runtime_sglang",
        "dependency_runtime_megatron",
        "dependency_installer_sglang_router",
        "dependency_runtime_sglang_router",
        "dependency_installer_sgl_kernel_npu",
        "dependency_runtime_sgl_kernel_npu",
        "dependency_installer_memfabric_hybrid",
        "dependency_runtime_memfabric_hybrid",
        "dependency_installer_mbridge",
        "dependency_runtime_mbridge",
        "dependency_installer_megatron_bridge",
        "dependency_runtime_megatron_bridge",
        "dependency_installer_megatron_adaptor",
        "dependency_runtime_megatron_adaptor",
        "dependency_installer_transformer_engine_npu",
        "dependency_runtime_transformer_engine_npu",
        "dependency_installer_cann",
        "dependency_runtime_cann",
    }
    missing = sorted(required - by_name.keys())
    if missing:
        raise ContractError(f"environment lacks dependency checks: {', '.join(missing)}")
    failed = sorted(name for name in required if by_name[name].get("passed") is not True)
    if failed:
        raise ContractError(f"environment dependency checks are not PASS: {', '.join(failed)}")
    inventory = payload.get("component_inventory")
    if not isinstance(inventory, dict):
        raise ContractError("environment lacks component_inventory")
    git_components = {
        "slime",
        "sglang",
        "megatron",
        "megatron_adaptor",
        "mbridge",
        "megatron_bridge",
        "transformer_engine_npu",
    }
    missing_components = sorted((git_components | {"cann"}) - inventory.keys())
    if missing_components:
        raise ContractError("component_inventory lacks: " + ", ".join(missing_components))
    for name in git_components:
        require_sha(inventory[name].get("commit"), f"component_inventory.{name}.commit")
        require_sha(
            inventory[name].get("required_base_commit"),
            f"component_inventory.{name}.required_base_commit",
        )
        require_sha(
            inventory[name].get("patched_head"),
            f"component_inventory.{name}.patched_head",
        )
    if not isinstance(inventory["cann"].get("version"), str) or not inventory["cann"]["version"]:
        raise ContractError("component_inventory.cann.version must be recorded")


def validate_change_assessment(path: Path, upstream_sha: str) -> None:
    payload = read_object(path)
    if payload.get("schema_version") != 1:
        raise ContractError("change assessment schema_version must be 1")
    require_sha(payload.get("base_sha"), "change_assessment.base_sha")
    require_sha(payload.get("target_sha"), "change_assessment.target_sha")
    if payload["target_sha"] != upstream_sha:
        raise ContractError("change assessment target_sha does not match result upstream_sha")
    for section in ("source_changes", "dependency_changes", "test_changes"):
        if not isinstance(payload.get(section), dict):
            raise ContractError(f"change assessment lacks {section}")
    tests = payload["test_changes"]
    for field in ("added", "modified", "deleted", "renamed", "ci_changes", "target_tests"):
        if not isinstance(tests.get(field), list):
            raise ContractError(f"change assessment test_changes.{field} must be an array")
    if payload.get("upstream_changed") is True:
        assessment = payload["source_changes"].get("feature_assessment")
        if not isinstance(assessment, list) or not assessment:
            raise ContractError("changed upstream requires a completed feature_assessment")
        for index, item in enumerate(assessment):
            if not isinstance(item, dict):
                raise ContractError(f"feature_assessment {index} must be an object")
            for field in ("capability", "change_type", "summary"):
                if not isinstance(item.get(field), str) or not item[field]:
                    raise ContractError(f"feature_assessment {index} lacks {field}")
            if not isinstance(item.get("evidence"), list) or not item["evidence"]:
                raise ContractError(f"feature_assessment {index} lacks evidence")


def validate_result(payload: dict, workspace: Path) -> None:
    if payload.get("schema_version") != 2:
        raise ContractError("result schema_version must be 2")
    require_sha(payload.get("upstream_sha"), "upstream_sha")
    mode = payload.get("mode")
    if mode not in {"full", "trial_existing_assets"}:
        raise ContractError(f"invalid mode: {mode!r}")
    count_fields = (
        "baseline_discovered",
        "executed",
        "pass",
        "ng",
        "unexecuted",
        "missing_resources",
        "timeout",
    )
    for field in count_fields:
        if type(payload.get(field)) is not int or payload[field] < 0:
            raise ContractError(f"{field} must be a non-negative integer")
    if payload["executed"] != payload["pass"] + payload["ng"]:
        raise ContractError("executed must equal pass + ng")
    if payload["pass"] + payload["ng"] + payload["unexecuted"] != payload["baseline_discovered"]:
        raise ContractError("case counts do not equal baseline_discovered")
    if payload["missing_resources"] > payload["ng"] + payload["unexecuted"]:
        raise ContractError("missing_resources cannot exceed ng + unexecuted")
    report = safe_report(workspace, payload.get("report_path"))
    validate_report_content(report)
    safe_workspace_file(workspace, payload.get("conversion_plan_path"), "conversion_plan_path")
    safe_workspace_file(workspace, payload.get("validation_result_path"), "validation_result_path")
    environment = safe_workspace_file(workspace, payload.get("environment_path"), "environment_path")
    validate_environment(environment)
    assessment = safe_workspace_file(
        workspace, payload.get("change_assessment_path"), "change_assessment_path"
    )
    validate_change_assessment(assessment, payload["upstream_sha"])

    if mode == "full":
        status = payload.get("status")
        if status not in {"PASS", "NG"}:
            raise ContractError("full mode requires status PASS or NG")
        pass_gate = (
            payload["baseline_discovered"] > 0
            and payload["executed"] == payload["baseline_discovered"]
            and payload["pass"] == payload["baseline_discovered"]
            and payload["ng"] == payload["unexecuted"] == 0
            and payload["missing_resources"] == payload["timeout"] == 0
            and payload.get("replay_verified") is True
        )
        if status == "PASS" and not pass_gate:
            raise ContractError("full-mode PASS gate is not satisfied")
        if status == "NG" and pass_gate:
            raise ContractError("full-mode NG contradicts satisfied PASS gate")
    elif payload.get("status") != "TRIAL_COMPLETE":
        raise ContractError("trial mode requires TRIAL_COMPLETE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("conversion", "result"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--workspace", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = read_object(args.path)
        if args.kind == "conversion":
            validate_conversion(payload)
        else:
            if args.workspace is None:
                raise ContractError("--workspace is required for result validation")
            validate_result(payload, args.workspace)
        print("contract valid")
        return 0
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
