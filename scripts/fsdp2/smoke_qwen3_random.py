#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a random-initialized Qwen3 FSDP8 runtime smoke test, without weights."""

import argparse
from datetime import timedelta
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess


def idle_device_ids(report: str) -> set:
    """Parse explicit idle-device statements; unknown formats do not pass."""
    return {int(value) for value in re.findall(r"No running processes found in NPU\s+(\d+)", report)}


def train(args, model_config):
    import torch
    import torch.distributed as dist
    import fsdp_turbo
    from torch.distributed.tensor import DTensor
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from fsdp_turbo.fsdp_turbo import FSDPTurbo
    from fsdp_turbo.fsdp_turbo_config import DistributedConfig, FSDPPlanConfig, FSDPTurboConfig, MemoryConfig
    from fsdp_turbo.utils.device import get_accelerator_name, set_accelerator_compatible

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    backend = set_accelerator_compatible()
    if backend != "hccl":
        raise RuntimeError("This smoke test requires eight Ascend NPUs.")
    torch.accelerator.set_device(local_rank)
    device = torch.device(get_accelerator_name(), local_rank)
    dist.init_process_group(backend="cpu:gloo,npu:hccl", timeout=timedelta(seconds=180))
    try:
        torch.manual_seed(42)
        config = Qwen3Config.from_dict(model_config)
        config._attn_implementation = "eager"
        model = Qwen3ForCausalLM(config)
        global_parameters = sum(parameter.numel() for parameter in model.parameters())
        runtime_config = FSDPTurboConfig(
            distributed=DistributedConfig(
                fully_shard_parallel_size=8, tensor_parallel_size=1, expert_parallel_size=1,
                fsdp_plan=FSDPPlanConfig(
                    apply_modules={"model.layers.{*}": {}, "model.embed_tokens": {}, "lm_head": {}},
                    param_dtype="bf16", reduce_dtype="fp32", output_dtype="bf16",
                    fsdp_implementation="native", num_to_forward_prefetch=0, num_to_backward_prefetch=0)),
            memory=MemoryConfig(recompute=False))
        model = FSDPTurbo(runtime_config, model)
        model.train()
        watched = next(parameter for name, parameter in model.named_parameters()
                       if name.endswith("layers.0.self_attn.q_proj.weight"))
        if not isinstance(watched, DTensor) or watched.to_local().numel() * 8 != watched.numel():
            raise RuntimeError("Expected an eight-way sharded Q-projection parameter.")
        metadata = {"rank": rank, "local_rank": local_rank, "world_size": dist.get_world_size(),
                    "device": str(device), "global_parameters": global_parameters,
                    "mesh_shape": list(watched.device_mesh.shape),
                    "placements": [str(p) for p in watched.placements],
                    "global_shape": list(watched.shape), "local_shape": list(watched.to_local().shape)}
        source_root = Path(fsdp_turbo.__file__).resolve().parent.parent
        revision = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, timeout=10)
        runtime = {"torch": torch.__version__, "torch_npu": importlib.metadata.version("torch-npu"),
                   "transformers": importlib.metadata.version("transformers"),
                   "turbo_module": str(Path(fsdp_turbo.__file__).resolve()),
                   "turbo_sha": revision.stdout.strip() if revision.returncode == 0 else None,
                   "parameter_dtype": str(watched.dtype), "param_cast_dtype": "torch.bfloat16",
                   "reduce_dtype": "torch.float32"}
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
        records = []
        for step in range(args.steps):
            generator = torch.Generator(device="cpu").manual_seed(1000 + rank * args.steps + step)
            tokens = torch.randint(0, config.vocab_size, (1, args.seq_length), generator=generator).to(device)
            optimizer.zero_grad(set_to_none=True)
            before = watched.to_local().detach().clone()
            loss = model(input_ids=tokens, labels=tokens, use_cache=False).loss
            loss.backward()
            grad_squared = torch.zeros((), device=device, dtype=torch.float32)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    gradient = parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
                    grad_squared += gradient.detach().float().square().sum()
            dist.all_reduce(grad_squared)
            grad_norm = math.sqrt(grad_squared.item())
            if not math.isfinite(grad_norm) or grad_norm == 0:
                raise RuntimeError("Gradient norm is nonfinite or zero.")
            optimizer.step()
            delta_squared = (watched.to_local().detach() - before).float().square().sum()
            dist.all_reduce(delta_squared)
            delta = math.sqrt(delta_squared.item())
            mean_loss = loss.detach().float().clone()
            dist.all_reduce(mean_loss)
            mean_loss = mean_loss.item() / 8
            if not math.isfinite(mean_loss) or not math.isfinite(delta) or delta == 0:
                raise RuntimeError("Loss or optimizer update failed validation.")
            record = {"step": step + 1, "mean_loss": mean_loss, "global_grad_norm": grad_norm,
                      "q_projection_update_l2": delta}
            records.append(record)
            if rank == 0:
                print(json.dumps(record), flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / ("rank%02d.json" % rank)).write_text(json.dumps(
            {**metadata, "runtime": runtime, "steps": records}, indent=2) + "\n")
        finished = torch.ones((), device=device)
        dist.all_reduce(finished)
        if rank == 0:
            (args.output_dir / "summary.json").write_text(json.dumps({
                "scope": "FSDPTurbo native FSDP8 runtime smoke; not Slime GRPO",
                "initialization": "random", "data": "fixed synthetic tokens", "completed_ranks": int(finished.item()),
                "model": model_config, "runtime": runtime, "steps": records}, indent=2) + "\n")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("qwen3_8b_2layer.json"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--output-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--inspect", action="store_true", help="Print the plan without importing training libraries")
    mode.add_argument("--check-devices", action="store_true", help="Require all eight NPUs explicitly reported idle")
    args = parser.parse_args()
    if args.steps < 1 or args.seq_length < 2:
        parser.error("steps must be positive and seq-length at least two")
    config = json.loads(args.config.read_text())
    if config.get("model_type") != "qwen3" or config.get("num_hidden_layers", 0) < 1:
        parser.error("Expected a nonempty Qwen3 configuration")
    if args.inspect:
        print(json.dumps({"model": config, "world_size": 8, "steps": args.steps,
                          "seq_length": args.seq_length, "weights_required": False, "data": "synthetic"}, indent=2))
        return 0
    if args.check_devices:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=20, check=True)
        print(result.stdout)
        if idle_device_ids(result.stdout) != set(range(8)):
            parser.exit(3, "All eight NPUs must be explicitly idle before launch.\n")
        return 0
    if os.environ.get("WORLD_SIZE") != "8" or args.output_dir is None:
        parser.error("Use the guarded launcher with eight torchrun ranks and an output directory")
    train(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
