# SPDX-License-Identifier: Apache-2.0
"""Test metadata gates without importing training frameworks or using NPUs."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "fsdp2"
SPEC = importlib.util.spec_from_file_location("turbo_env", SCRIPTS / "check_fsdp_turbo_env.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)
PROFILE = json.loads((SCRIPTS / "fsdp_turbo_glm53_profile.json").read_text())


def candidate_inventory():
    return {
        "host": {"python_version": "3.12.13"},
        "packages": {"torch": "2.10.0+cpu", "torch-npu": "2.10.0.post4",
                     "transformers": "5.17.0", "triton": "3.6.0", "triton-ascend": "3.6.0",
                     "tiktoken": "0.13.0"},
        "ascend_version_files": {"/usr/local/Ascend/toolkit/install.info": "version=9.2.0\n",
                                 "/usr/local/Ascend/driver/version.info": "version=25.3.0\n"},
        "module_locations": {"fla": "/packages/fla/__init__.py", "fla_npu": "/packages/fla_npu/__init__.py",
                             "triton_ascend_kernels": "/packages/triton_ascend_kernels/__init__.py",
                             "cann_ops_transformer": {"namespace_paths": ["/packages/cann_ops_transformer"]}},
        "source_revisions": {"fsdp_turbo": "79b9ae5f6db0e3c6753ad7ff6c0f1ed20d08b09a"},
    }


class TurboEnvironmentTests(unittest.TestCase):
    def test_matching_metadata_does_not_claim_training_or_kernels(self):
        result = CHECKER.compare_profile(candidate_inventory(), PROFILE)
        self.assertTrue(result["matches_documented_profile"])
        self.assertFalse(result["training_tested"])
        self.assertFalse(result["kernel_execution_tested"])

    def test_observed_old_environment_reports_specific_mismatches(self):
        inventory = candidate_inventory()
        inventory["packages"].update(torch="2.7.1", **{"torch-npu": "2.7.1.post8", "transformers": "5.2.0"})
        inventory["ascend_version_files"]["/usr/local/Ascend/toolkit/install.info"] = "version=9.1.0\n"
        inventory["source_revisions"]["fsdp_turbo"] = "262f5ba4ae266d62b01c7e8a2e2bc46d46d1a0c3"
        result = CHECKER.compare_profile(inventory, PROFILE)
        failures = {c["component"] for c in result["checks"] if not c["matches"]}
        self.assertEqual(failures, {"torch", "torch-npu", "transformers", "cann", "fsdp_turbo_sha"})

    def test_driver_version_cannot_substitute_for_missing_cann(self):
        inventory = candidate_inventory()
        del inventory["ascend_version_files"]["/usr/local/Ascend/toolkit/install.info"]
        result = CHECKER.compare_profile(inventory, PROFILE)
        self.assertFalse(next(c for c in result["checks"] if c["component"] == "cann")["matches"])

    def test_prerelease_old_post_and_missing_kernel_are_rejected(self):
        inventory = candidate_inventory()
        inventory["packages"]["torch"] = "2.10.0rc1"
        inventory["packages"]["torch-npu"] = "2.10.0.post3"
        inventory["module_locations"]["fla_npu"] = None
        result = CHECKER.compare_profile(inventory, PROFILE)
        self.assertEqual({c["component"] for c in result["checks"] if not c["matches"]},
                         {"torch", "torch-npu", "module:fla_npu"})

    def test_input_is_not_mutated(self):
        inventory = candidate_inventory()
        before = copy.deepcopy(inventory)
        CHECKER.compare_profile(inventory, PROFILE)
        self.assertEqual(inventory, before)


if __name__ == "__main__":
    unittest.main()
