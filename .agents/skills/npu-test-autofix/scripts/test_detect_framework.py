from __future__ import annotations

from pathlib import Path

from detect_framework import detect


def test_detects_slime_from_multiple_markers(tmp_path: Path) -> None:
    (tmp_path / "slime").mkdir()
    (tmp_path / "slime_plugins").mkdir()
    payload = detect(tmp_path)
    assert payload["framework"] == "slime"
    assert payload["detection"] == "auto"


def test_detects_verl_package(tmp_path: Path) -> None:
    (tmp_path / "verl").mkdir()
    (tmp_path / "tests").mkdir()
    payload = detect(tmp_path)
    assert payload["framework"] == "verl"
    assert payload["candidate_test_roots"] == ["tests"]


def test_unknown_framework_requires_confirmation(tmp_path: Path) -> None:
    payload = detect(tmp_path)
    assert payload["framework"] == "generic"
    assert payload["requires_profile_confirmation"] is True


def test_explicit_profile_overrides_detection(tmp_path: Path) -> None:
    (tmp_path / "slime").mkdir()
    (tmp_path / "slime_plugins").mkdir()
    payload = detect(tmp_path, "verl")
    assert payload["framework"] == "verl"
    assert payload["detection"] == "explicit"
