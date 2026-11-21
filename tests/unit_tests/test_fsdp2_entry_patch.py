"""Apply the delivered patch to pinned Slime, then test its actual entry functions.

These are CPU argument/handoff regressions, not an NPU training acceptance.
SLIME_ENTRY_UPSTREAM may point to any local clone containing the pinned commit.
"""

import argparse
import ast
import contextlib
import io
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import typing
import unittest
from unittest.mock import patch
import tomllib

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parents[1]
UPSTREAM = Path(os.environ.get("SLIME_ENTRY_UPSTREAM", WORKSPACE / "work/slime-upstream-readonly"))
PIN = "18868a79b0b083a093798a94ed0f8cb4f71b9c14"


def extract_functions(path, names, namespace=None):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    result = {} if namespace is None else namespace
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), result)
    return result


class FSDP2EntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not UPSTREAM.exists():
            raise unittest.SkipTest("A local pinned Slime clone is required; set SLIME_ENTRY_UPSTREAM")
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.stage = Path(cls.temporary.name) / "slime"
        cls.stage.mkdir()
        archive = subprocess.run(["git", "-C", str(UPSTREAM), "archive", PIN], check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as source:
            source.extractall(cls.stage, filter="data")
        manifest = tomllib.loads((REPO / "patches/MANIFEST.toml").read_text())
        for item in manifest["patches"]:
            if item["target"] == "slime" and item["category"] in ("npu-compat", "upstream-fixes"):
                subprocess.run(["git", "apply", str(REPO / "patches" / item["category"] / item["file"])],
                               cwd=cls.stage, check=True, capture_output=True)
        patch = REPO / "patches/features/fsdp2-turbo-entry.patch"
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=cls.stage, check=True, capture_output=True)
        subprocess.run(["git", "apply", str(patch)], cwd=cls.stage, check=True, capture_output=True)
        cls.functions = extract_functions(
            cls.stage / "slime/utils/arguments.py", {"reset_arg", "get_slime_extra_args_provider"},
            {"argparse": argparse, "json": json, "Any": typing.Any, "logger": logging.getLogger(__name__)})
        cls.parser_ns = extract_functions(cls.stage / "slime/utils/fsdp2_arguments.py", {"parse_fsdp2_args"},
                                         {"argparse": argparse, "json": json, "os": os, "Path": Path})
        cls.publication_ns = extract_functions(cls.stage / "slime/ray/actor_group.py", {"validate_fsdp2_publication"},
                                              {"Path": Path})
        tree = ast.parse((cls.stage / "slime/ray/rollout.py").read_text())
        manager = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RolloutManager")
        method = next(node for node in manager.body if isinstance(node, ast.FunctionDef) and node.name == "accept_fsdp2_weights")
        cls.receiver_namespace = {"Path": Path}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "receiver", "exec"), cls.receiver_namespace)
        cls.config = Path(cls.temporary.name) / "config.json"
        cls.config.write_text('{"model_type":"qwen3","num_hidden_layers":2}')

    def argv(self):
        return ["--train-backend", "fsdp2", "--fsdp-fixed-rollout", "--fsdp-random-init",
                "--fsdp-model-config", str(self.config), "--rollout-batch-size", "4",
                "--n-samples-per-prompt", "2", "--num-rollout", "2", "--update-weight-disk-dir",
                str(Path(self.temporary.name) / "exports")]

    def parse(self, extra=()):
        with contextlib.redirect_stderr(io.StringIO()):
            return self.parser_ns["parse_fsdp2_args"](self.functions["get_slime_extra_args_provider"](), self.argv() + list(extra))

    def test_parser_uses_global_eight_sample_batch(self):
        args = self.parse()
        self.assertEqual((args.global_batch_size, args.micro_batch_size, args.num_steps_per_rollout), (8, 1, 1))
        self.assertEqual((args.rollout_num_gpus, args.update_weight_start_version, args.fsdp_weight_version), (0, -1, -1))
        self.assertTrue(args.debug_train_only)
        self.assertEqual(args.distributed_backend, "hccl")
        self.assertEqual(args.data_source_path, "fixed_rollout.FixedDataSource")

    def test_megatron_remains_default_and_does_not_use_fsdp_parser(self):
        text = (self.stage / "slime/utils/arguments.py").read_text()
        self.assertIn('choices=["megatron", "fsdp2"], default="megatron"', text)
        self.assertLess(text.index('if pre.train_backend == "fsdp2"'), text.index('from slime.backends.megatron_utils.arguments import megatron_parse_args'))

    def test_reject_unsupported_algorithm_switches(self):
        for switch in ("--normalize-advantages", "--use-tis", "--use-opd", "--use-critic", "--use-kl-loss",
                       "--use-score-centering", "--calculate-per-token-loss", "--use-dynamic-batch-size"):
            with self.subTest(switch=switch), self.assertRaises(SystemExit):
                self.parse([switch])

    def test_reject_sft_and_custom_loss(self):
        for value in ("sft_loss", "custom_loss"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                self.parse(["--loss-type", value])

    def test_reject_incompatible_schedule(self):
        for extra in (["--global-batch-size", "16"], ["--micro-batch-size", "2"],
                      ["--num-steps-per-rollout", "2"], ["--n-samples-per-prompt", "1"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.parse(extra)

    def test_reject_parallel_offload_recovery_and_dcp(self):
        for extra in (["--context-parallel-size", "2"], ["--actor-num-gpus-per-node", "4"],
                      ["--offload-train"], ["--use-fault-tolerance"], ["--start-rollout-id", "1"],
                      ["--save", "/tmp/checkpoint"], ["--load", "/tmp/checkpoint"], ["--eval-interval", "1"]):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.parse(extra)

    def test_actor_environment_inherits_hccl_with_explicit_override(self):
        with patch.dict(os.environ, {"HCCL_NPU_SOCKET_PORT_RANGE": "auto", "HCCL_HOST_SOCKET_PORT_RANGE": "auto"}):
            args = self.parse(["--train-env-vars", '{"HCCL_NPU_SOCKET_PORT_RANGE":"30000-30031"}'])
        self.assertEqual(args.train_env_vars["HCCL_NPU_SOCKET_PORT_RANGE"], "30000-30031")
        self.assertEqual(args.train_env_vars["HCCL_HOST_SOCKET_PORT_RANGE"], "auto")
        self.assertEqual(args.train_env_vars["TORCH_DEVICE_BACKEND_AUTOLOAD"], "0")

    def test_unknown_flags_are_not_silently_ignored(self):
        with self.assertRaises(SystemExit):
            self.parse(["--not-a-real-option"])

    def test_requires_explicit_fixed_random_scope(self):
        for flag in ("--fsdp-fixed-rollout", "--fsdp-random-init"):
            argv = self.argv()
            argv.remove(flag)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.parser_ns["parse_fsdp2_args"](self.functions["get_slime_extra_args_provider"](), argv)

    def descriptor(self, version=0):
        directory = (Path(self.temporary.name) / "exports" / f"weight_v{version:06d}").resolve()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text("{}")
        (directory / "COMPLETE").write_text(str(version))
        return {"weight_version": version, "model_path": str(directory), "manifest_path": str(directory / "manifest.json")}

    def test_publication_requires_eight_matching_completed_ranks(self):
        result = self.descriptor()
        verified = self.publication_ns["validate_fsdp2_publication"]([result] * 8, result["model_path"], 0)
        self.assertEqual(verified, result)
        with self.assertRaises(RuntimeError):
            self.publication_ns["validate_fsdp2_publication"]([result] * 7, result["model_path"], 0)
        bad = dict(result, weight_version=1)
        with self.assertRaises(RuntimeError):
            self.publication_ns["validate_fsdp2_publication"]([result] * 7 + [bad], result["model_path"], 0)
        (Path(result["model_path"]) / "COMPLETE").unlink()
        with self.assertRaises(RuntimeError):
            self.publication_ns["validate_fsdp2_publication"]([result] * 8, result["model_path"], 0)

    def test_manager_advances_version_only_after_verified_receiver(self):
        args = self.parse()
        manager = SimpleNamespace(args=args, training_weight_version=-1)
        calls = []
        def receiver(**publication):
            calls.append(publication)
            return {"verified": True, "weight_version": publication["weight_version"], "model_path": publication["model_path"]}
        self.receiver_namespace["load_function"] = lambda _: receiver
        accept = self.receiver_namespace["accept_fsdp2_weights"]
        for version in (0, 1, 2):
            accept(manager, **self.descriptor(version))
            self.assertEqual(manager.training_weight_version, version)
            self.assertEqual(args.fsdp_weight_version, version)
        self.assertEqual(len(calls), 3)
        with self.assertRaises(RuntimeError):
            accept(manager, **self.descriptor(2))
        self.assertEqual(len(calls), 3)

    def test_receiver_failure_does_not_claim_acceptance(self):
        args = self.parse()
        manager = SimpleNamespace(args=args, training_weight_version=-1)
        self.receiver_namespace["load_function"] = lambda _: lambda **_: {"verified": False}
        with self.assertRaises(RuntimeError):
            self.receiver_namespace["accept_fsdp2_weights"](manager, **self.descriptor())
        self.assertEqual(args.fsdp_weight_version, -1)
        self.assertEqual(manager.training_weight_version, -1)

    def test_real_cpu_rollout_data_path_without_serving_packages(self):
        dependencies = ("torch", "ray", "yaml", "wandb", "numpy", "httpx", "requests", "psutil")
        if any(importlib.util.find_spec(name) is None for name in dependencies):
            self.skipTest("Optional CPU runtime dependencies are not installed")
        fixture = REPO / "scripts/fsdp2/fixed_rollout.py"
        if not fixture.is_file():
            self.skipTest("The integration fixture is provided with the Slime launcher")
        script = r"""
import sys,tempfile
from types import SimpleNamespace
from slime.utils.arguments import parse_args
from slime.ray.rollout import RolloutManager
from slime.utils.dp_schedule import build_dp_schedule
with tempfile.TemporaryDirectory() as output:
    sys.argv=['train.py','--train-backend','fsdp2','--fsdp-fixed-rollout','--fsdp-random-init',
              '--fsdp-model-config',sys.argv[1],'--rollout-batch-size','4',
              '--n-samples-per-prompt','2','--num-rollout','2','--update-weight-disk-dir',output]
    args,plan=parse_args(return_restore_plan=True)
    cls=RolloutManager.__ray_metadata__.modified_class.__mro__[1]
    deployment=SimpleNamespace(controller=None,servers={},engine_lock=object(),
                               placements={'rollout':None},reused=False,restore_plan=None)
    manager=cls(args,None,deployment=deployment)
    manager.attach_training(args,deployment)
    manager.args.fsdp_weight_version=0
    data,_=manager._get_rollout_data(0)
    converted=manager.batch_builder.convert(data)
    parallel={'dp_size':8,'cp_size':1,'vpp_size':1,'microbatch_group_size_per_vp_stage':1}
    partitions,mb,nmb,gbs=build_dp_schedule(args,parallel,[len(t) for t in converted['tokens']],
                                         global_batch_size=8,rollout_indices=converted['rollout_ids'])
    assert len(data)==8 and len(set(converted['rollout_ids']))==8
    assert converted['rewards'][0]<0<converted['rewards'][1]
    assert nmb==[1] and gbs==[8] and all(len(p)==1 for p in partitions)
    assert converted['rollout_mask_sums']==[32,30]*4
    assert not any(m.startswith(('sglang','megatron','straw')) for m in sys.modules)
    print('REAL_CPU_DATA_PATH_OK')
"""
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(self.stage), str(fixture.parent)]))
        result = subprocess.run([os.sys.executable, "-c", script, str(fixture.with_name("qwen3_8b_2layer.json"))],
                                env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("REAL_CPU_DATA_PATH_OK", result.stdout)

    def test_changed_sources_compile(self):
        for relative in ("slime/utils/arguments.py", "slime/utils/fsdp2_arguments.py", "slime/ray/placement_group.py",
                         "slime/ray/actor_group.py", "slime/ray/rollout.py", "slime/ray/serving.py"):
            compile((self.stage / relative).read_text(), relative, "exec")


if __name__ == "__main__":
    unittest.main()
