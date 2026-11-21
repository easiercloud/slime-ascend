# SPDX-License-Identifier: Apache-2.0
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/fsdp2/smoke_qwen3_random.py"
SPEC = importlib.util.spec_from_file_location("qwen3_smoke", SCRIPT)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class SmokePlanTests(unittest.TestCase):
    def test_busy_or_partial_devices_do_not_pass(self):
        idle = "\n".join("No running processes found in NPU %d" % device for device in range(2, 8))
        self.assertNotEqual(SMOKE.idle_device_ids(idle), set(range(8)))
        self.assertEqual(SMOKE.idle_device_ids("unrecognized device output"), set())
        self.assertEqual(SMOKE.idle_device_ids(idle + "\nNo running processes found in NPU 0\nNo running processes found in NPU 1"), set(range(8)))

    def test_inspect_preserves_eight_billion_model_dimensions(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--inspect"], capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertEqual(plan["world_size"], 8)
        self.assertEqual(plan["model"]["num_hidden_layers"], 2)
        self.assertEqual(plan["model"]["hidden_size"], 4096)
        self.assertEqual(plan["model"]["vocab_size"], 151936)
        self.assertFalse(plan["weights_required"])

    def test_direct_non_eight_rank_training_is_rejected(self):
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("eight torchrun ranks", result.stderr)


if __name__ == "__main__":
    unittest.main()
