#!/usr/bin/env python3
"""Tests for workspace dependency discovery."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_command_utils.py")
SPEC = importlib.util.spec_from_file_location("prepare_command_utils", SCRIPT)
assert SPEC and SPEC.loader
PREPARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE
SPEC.loader.exec_module(PREPARE)


class DependencyDiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        # Skills are installed at <repo>/.agents/skills/<name>/scripts.
        self.repo = Path(__file__).resolve().parents[4] / "slime"

    def test_adds_only_referenced_workspace_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            layouts = {
                "Megatron-LM": ("", "megatron"),
                "Megatron-Bridge": ("src", "bridge_marker"),
                "MegatronAdaptor": ("", "megatron_adaptor"),
                "TransformerEngine": ("", "transformer_engine"),
                "mbridge": ("", "mbridge"),
                "MindSpeed": ("", "mindspeed"),
            }
            for project, (subdirectory, package) in layouts.items():
                (workspace / project / subdirectory / package).mkdir(parents=True)

            args = argparse.Namespace(
                megatron_root=None,
                bridge_root=None,
                megatron_adaptor_root=None,
                transformer_engine_root=None,
                mindspeed_root=None,
                extra_pythonpath=[],
            )
            paths, results, missing = PREPARE.discover_python_paths(
                self.repo, workspace, args
            )
            statuses = {item.name: item.status for item in results}

            self.assertEqual(statuses["Megatron-LM"], "workspace")
            self.assertEqual(statuses["Megatron-Bridge"], "workspace")
            self.assertEqual(statuses["MegatronAdaptor"], "workspace")
            self.assertEqual(statuses["TransformerEngine"], "workspace")
            self.assertEqual(statuses["mbridge"], "not referenced")
            self.assertEqual(statuses["MindSpeed"], "not referenced")
            self.assertFalse(any("MindSpeed" in path for path in paths))
            self.assertEqual(missing, [])

    def test_runtime_template_propagates_coredump_signal_setting(self) -> None:
        source = (
            self.repo
            / "slime"
            / "utils"
            / "external_utils"
            / "command_utils.py"
        ).read_text(encoding="utf-8")
        prepared, _ = PREPARE.prepare_source(
            source,
            ["/workspace/Megatron-LM", "/workspace/Megatron-Bridge/src"],
        )
        self.assertIn(
            '"ASCEND_COREDUMP_SIGNAL": os.environ.get('
            '"ASCEND_COREDUMP_SIGNAL", "none")',
            prepared,
        )



if __name__ == "__main__":
    unittest.main()
