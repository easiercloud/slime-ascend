from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import doctor
from patch_manifest import ManifestError, validate
from tree_manifest import compare, create_manifest
from validate_contracts import ContractError, validate_result


SHA = "a" * 40


def test_doctor_converts_probe_timeout_to_structured_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "/usr/bin/fake")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 20)

    monkeypatch.setattr(doctor.subprocess, "run", timeout)
    result = doctor.command_version(["fake", "--version"])
    assert result["available"] is False
    assert result["timed_out"] is True


def test_dependency_contract_uses_upstream_dockerfile(tmp_path: Path) -> None:
    slime = tmp_path / "slime"
    (slime / "docker").mkdir(parents=True)
    (slime / "docker" / "Dockerfile").write_text(
        "ARG SGLANG_IMAGE_TAG=v0.5.15.post1-cu129\n"
        "ARG MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087\n"
        "RUN pip install https://example.test/sglang_router-0.3.2-py3-none-any.whl\n",
        encoding="utf-8",
    )
    assert doctor.upstream_dependency_contract(slime) == {
        "sglang": "0.5.15.post1",
        "megatron": "1dcf0dafa884ad52ffb243625717a3471643e087",
        "sglang_router_wheel": "https://example.test/sglang_router-0.3.2-py3-none-any.whl",
        "sglang_router_marker": "slime",
    }


def test_installer_dependency_contract_requires_explicit_pins(tmp_path: Path) -> None:
    installer = tmp_path / "quick_install.sh"
    installer.write_text(
        "SGLANG_COMMIT=0.5.15.post1\nMEGATRON_COMMIT=1dcf0dafa\n"
        "SGLANG_ROUTER_WHEEL=https://example.test/router.whl\n"
        "SGL_KERNEL_NPU_VERSION=2026.6.2\n"
        "SGL_KERNEL_NPU_WHEEL_VERSION=2026.6.1\n"
        f"SGL_KERNEL_NPU_FIX_COMMIT={SHA}\n"
        "MEMFABRIC_HYBRID_VERSION=1.0.8\n"
        "CANN_VERSION=9.0.0\n"
        f"MBRIDGE_COMMIT={SHA}\n"
        f"MEGATRON_BRIDGE_COMMIT={SHA}\n"
        f"MEGATRON_ADAPTOR_COMMIT={SHA}\n"
        f"TRANSFORMER_ENGINE_NPU_COMMIT={SHA}\n",
        encoding="utf-8",
    )
    assert doctor.installer_dependency_contract(installer) == {
        "sglang": "0.5.15.post1",
        "megatron": "1dcf0dafa",
        "sglang_router_wheel": "https://example.test/router.whl",
        "sgl_kernel_npu_version": "2026.6.2",
        "sgl_kernel_npu_wheel_version": "2026.6.1",
        "sgl_kernel_npu_fix_commit": SHA,
        "memfabric_hybrid_version": "1.0.8",
        "cann_version": "9.0.0",
        "mbridge": SHA,
        "megatron_bridge": SHA,
        "megatron_adaptor": SHA,
        "transformer_engine_npu": SHA,
    }
    assert doctor.dependency_pin_matches(
        "megatron", "1dcf0dafa", "1dcf0dafa884ad52ffb243625717a3471643e087"
    )


def test_sglang_npu_contract_uses_npu_dockerfile(tmp_path: Path) -> None:
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "npu.Dockerfile").write_text(
        "RUN pip install memfabric-hybrid==1.0.8\n", encoding="utf-8"
    )
    assert doctor.sglang_npu_dependency_contract(tmp_path) == {
        "memfabric_hybrid_version": "1.0.8"
    }


def test_sglang_runtime_requires_current_editable_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "sglang"
    module = repo / "python" / "sglang" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.touch()
    head = "74f6eca3d4fa3f97c81b128cccd6059c93951723"
    payload = {
        "metadata_version": "0.5.17.dev31+g74f6eca3d",
        "module_version": "0.5.17.dev31+g74f6eca3d",
        "module_file": str(module),
        "direct_url": {"dir_info": {"editable": True}, "url": (repo / "python").as_uri()},
    }
    monkeypatch.setattr(
        doctor.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, json.dumps(payload), ""),
    )
    state = doctor.sglang_python_runtime_state(
        repo,
        {"head": head, "matches": True, "detail": "base matched"},
    )
    assert state["matches"] is True
    assert state["metadata_fresh"] is True

    payload["metadata_version"] = "0.5.17.dev30+gae517ec08"
    payload["module_version"] = payload["metadata_version"]
    stale = doctor.sglang_python_runtime_state(
        repo,
        {"head": head, "matches": True, "detail": "base matched"},
    )
    assert stale["matches"] is False
    assert stale["metadata_fresh"] is False


def test_change_assessment_detects_dependency_feature_and_test_delta(tmp_path: Path) -> None:
    repo = tmp_path / "slime"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    (repo / "docker").mkdir()
    (repo / "tests").mkdir()
    (repo / "slime" / "rollout").mkdir(parents=True)
    (repo / "docker" / "Dockerfile").write_text("ARG SGLANG_IMAGE_TAG=v0.5.13\n", encoding="utf-8")
    (repo / "tests" / "test_old.py").write_text("def test_old(): pass\n", encoding="utf-8")
    (repo / "slime" / "rollout" / "worker.py").write_text("old = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    (repo / "docker" / "Dockerfile").write_text("ARG SGLANG_IMAGE_TAG=v0.5.15.post1\n", encoding="utf-8")
    (repo / "tests" / "test_new.py").write_text("def test_new(): pass\n", encoding="utf-8")
    (repo / "slime" / "rollout" / "worker.py").write_text("new = True\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "target"], cwd=repo, check=True)
    target = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    output = tmp_path / "assessment.json"
    script = Path(__file__).with_name("collect_change_assessment.py")
    subprocess.run(
        ["python3", str(script), "--repo", str(repo), "--base", base, "--target", target, "--output", str(output)],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["upstream_changed"] is True
    assert payload["dependency_changes"]["changed"] is True
    assert payload["test_changes"]["added"] == ["tests/test_new.py"]
    assert "rollout_and_sampling" in payload["source_changes"]["feature_candidates"]


def test_manifest_is_authoritative(tmp_path: Path) -> None:
    patches = tmp_path / "patches"
    category = patches / "npu-compat"
    category.mkdir(parents=True)
    (category / "one.patch").write_text("patch one\n", encoding="utf-8")
    manifest = patches / "MANIFEST.toml"
    manifest.write_text(
        '[[patches]]\ncategory = "npu-compat"\nfiles = ["one.patch"]\n',
        encoding="utf-8",
    )
    assert [item.identifier for item in validate(manifest, patches)] == [
        "npu-compat:one.patch"
    ]
    assert validate(manifest, patches)[0].target == "slime"
    (category / "extra.patch").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="undeclared"):
        validate(manifest, patches)


def test_manifest_routes_external_patches_without_applying_to_slime(tmp_path: Path) -> None:
    patches = tmp_path / "patches"
    category = patches / "third_party" / "sglang"
    category.mkdir(parents=True)
    (category / "one.patch").write_text("external patch\n", encoding="utf-8")
    manifest = patches / "MANIFEST.toml"
    manifest.write_text(
        '[[patches]]\ncategory = "third_party"\ntarget = "sglang"\n'
        'files = ["sglang/one.patch"]\n',
        encoding="utf-8",
    )
    items = validate(manifest, patches)
    assert [(item.target, item.path.name) for item in items] == [("sglang", "one.patch")]


def test_tree_manifest_detects_untracked_and_mode_changes(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "new.py").write_text("x = 1\n", encoding="utf-8")
    (right / "new.py").write_text("x = 2\n", encoding="utf-8")
    (left / "only-left.txt").write_text("left\n", encoding="utf-8")
    differences = compare(create_manifest(left), create_manifest(right))
    assert "DIFFERENT\tnew.py" in differences
    assert "ONLY_VALIDATED\tonly-left.txt" in differences


def test_tree_manifest_records_directory_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    target = root / "target"
    target.mkdir(parents=True)
    (root / "linked-directory").symlink_to("target", target_is_directory=True)
    manifest = create_manifest(root)
    assert manifest["files"]["linked-directory"]["type"] == "symlink"
    assert manifest["files"]["linked-directory"]["target"] == "target"


def test_tree_manifest_can_include_runtime_artifacts_for_overlay_guard(
    tmp_path: Path,
) -> None:
    root = tmp_path / "overlay"
    cache = root / "tests" / "__pycache__"
    cache.mkdir(parents=True)
    artifact = cache / "test_case.pyc"
    artifact.write_bytes(b"historical")

    default_manifest = create_manifest(root)
    strict_manifest = create_manifest(root, use_default_excludes=False)

    assert "tests/__pycache__/test_case.pyc" not in default_manifest["files"]
    assert "tests/__pycache__/test_case.pyc" in strict_manifest["files"]


def valid_result(workspace: Path) -> dict:
    report = workspace / ".upstream-sync" / "audit-report.zh-CN.md"
    report.parent.mkdir()
    report.write_text(
        "# 审计报告\n\n上游变化与测试变化。\n"
        "SGLang Megatron MegatronAdaptor mbridge Megatron-Bridge "
        "TransformerEngineNPU CANN ATB。\n"
        "SGL Kernel NPU、torch-npu、DeepEP、TorchMemorySaver、MemFabric、"
        "Triton Ascend。dev 分支集成。\n",
        encoding="utf-8",
    )
    (workspace / ".upstream-sync" / "conversion-plan.json").write_text("{}", encoding="utf-8")
    (workspace / ".upstream-sync" / "validation-result.json").write_text("{}", encoding="utf-8")
    dependency_checks = [
        {"name": name, "passed": True, "required": True, "detail": "matched"}
        for name in (
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
        )
    ]
    (workspace / ".upstream-sync" / "environment.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "checks": dependency_checks,
                "component_inventory": {
                    name: {
                        "commit": SHA,
                        "required_base_commit": SHA,
                        "patched_head": SHA,
                    }
                    for name in (
                        "slime",
                        "sglang",
                        "megatron",
                        "megatron_adaptor",
                        "mbridge",
                        "megatron_bridge",
                        "transformer_engine_npu",
                    )
                }
                | {"cann": {"version": "9.0.0", "commit": None}},
            }
        ),
        encoding="utf-8",
    )
    (workspace / ".upstream-sync" / "change-assessment.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_sha": SHA,
                "target_sha": SHA,
                "upstream_changed": False,
                "source_changes": {"feature_assessment": []},
                "dependency_changes": {},
                "test_changes": {
                    "added": [],
                    "modified": [],
                    "deleted": [],
                    "renamed": [],
                    "ci_changes": [],
                    "target_tests": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "schema_version": 2,
        "status": "PASS",
        "mode": "full",
        "upstream_sha": SHA,
        "baseline_discovered": 2,
        "executed": 2,
        "pass": 2,
        "ng": 0,
        "unexecuted": 0,
        "missing_resources": 0,
        "timeout": 0,
        "replay_verified": True,
        "report_path": ".upstream-sync/audit-report.zh-CN.md",
        "conversion_plan_path": ".upstream-sync/conversion-plan.json",
        "validation_result_path": ".upstream-sync/validation-result.json",
        "environment_path": ".upstream-sync/environment.json",
        "change_assessment_path": ".upstream-sync/change-assessment.json",
    }


def test_full_result_requires_replay(tmp_path: Path) -> None:
    payload = valid_result(tmp_path)
    validate_result(payload, tmp_path)
    payload["replay_verified"] = False
    with pytest.raises(ContractError, match="PASS gate"):
        validate_result(payload, tmp_path)


def test_result_requires_dependency_preflight_pass(tmp_path: Path) -> None:
    payload = valid_result(tmp_path)
    environment = tmp_path / payload["environment_path"]
    environment.write_text(json.dumps({"status": "NG", "checks": []}), encoding="utf-8")
    with pytest.raises(ContractError, match="environment preflight must be PASS"):
        validate_result(payload, tmp_path)


def test_result_requires_feature_assessment_for_changed_upstream(tmp_path: Path) -> None:
    payload = valid_result(tmp_path)
    assessment = tmp_path / payload["change_assessment_path"]
    change = json.loads(assessment.read_text(encoding="utf-8"))
    change["base_sha"] = "b" * 40
    change["upstream_changed"] = True
    assessment.write_text(json.dumps(change), encoding="utf-8")
    with pytest.raises(ContractError, match="feature_assessment"):
        validate_result(payload, tmp_path)


def test_result_requires_complete_component_inventory(tmp_path: Path) -> None:
    payload = valid_result(tmp_path)
    environment = tmp_path / payload["environment_path"]
    state = json.loads(environment.read_text(encoding="utf-8"))
    del state["component_inventory"]["transformer_engine_npu"]
    environment.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ContractError, match="component_inventory lacks"):
        validate_result(payload, tmp_path)


def test_result_rejects_report_escape(tmp_path: Path) -> None:
    payload = valid_result(tmp_path)
    outside = tmp_path.parent / "outside-report.md"
    outside.write_text("中文审计报告内容\n", encoding="utf-8")
    payload["report_path"] = "../outside-report.md"
    with pytest.raises(ContractError, match="escapes"):
        validate_result(payload, tmp_path)


def test_replay_verifier_detects_unpersisted_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    (repo / "base.py").write_text("base = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    (repo / "forgotten.py").write_text("forgotten = True\n", encoding="utf-8")

    patches = tmp_path / "patches"
    (patches / "npu-compat").mkdir(parents=True)
    patch = patches / "npu-compat" / "noop.patch"
    (repo / "base.py").write_text("base = 2\n", encoding="utf-8")
    diff = subprocess.check_output(["git", "diff", "--", "base.py"], cwd=repo, text=True)
    patch.write_text(diff, encoding="utf-8")
    manifest = patches / "MANIFEST.toml"
    manifest.write_text(
        '[[patches]]\ncategory = "npu-compat"\nfiles = ["noop.patch"]\n',
        encoding="utf-8",
    )
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    output = tmp_path / "output"
    script = Path(__file__).with_name("replay_verify.py")
    process = subprocess.run(
        [
            "python3",
            str(script),
            "--validated-repo",
            str(repo),
            "--target-sha",
            sha,
            "--patches-root",
            str(patches),
            "--manifest",
            str(manifest),
            "--overlay",
            str(overlay),
            "--work-root",
            str(tmp_path / "work"),
            "--output-dir",
            str(output),
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode == 1
    result = json.loads((output / "replay-result.json").read_text(encoding="utf-8"))
    assert "ONLY_VALIDATED\tforgotten.py" in result["differences"]
